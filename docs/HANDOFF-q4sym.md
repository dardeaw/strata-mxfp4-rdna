# HANDOFF — .67 strata `--kv q4_sym`：q4 上 tensor core 的未完成工程

日期：2026-11-11
Repo：.67 (Nobara, RX9070/gfx1201) `/home/nobara-user/strata-0210`
Branch：`feat/kv-q4-s6`（自 `exp/mxfp4-int8-fused` @ `1ef65ed8` 切出）
Commit：`63b32f77 feat(kv): q4_sym storage layer ...`（已編譯進 build-hip，`ninja` 全綠；`kv_q4_parity`、`hip_prompt_attn_wmma` 皆 PASS——注意後者測的是 int8 路徑，q4 段在 AMD build 被 `#if !__HIPCC__` 排除）

---

## 1. 緣起：兩台機器的對照實驗

### .55 gufo（dense flash-next，ROCmFP4-FAST-v2q8k）——Q4 **追上並超越** Q8
blocked KV 梯測（fresh server per level，两轮漂移 <1%）：

| prompt tokens | Q8 avg | Q4 (GUFO_KV_Q4=1 GUFO_KV_V4=1) | 差距 |
|---|---|---|---|
| 31,051 | 979.9 | 1048.7 | **+7.0%** |
| 124,201 | 915.5 | 1000.2 | **+9.2%** |
| 165,601 | 899.7 | 982.9 | **+9.2%** |

機制：gufo 的 Q4 與 Q8 走**同一顆 WMMA kernel**，只是 block 從 34B→18B（KV 讀取流量減半）。級距越長贏越多。gufo 的 Q4 是「amax/7 對稱 nibble grid」（element 2e low / 2e+1 high），**無 Hadamard rotation**。

### .67 strata（jamesrogers MXFP4 125B MoE flash-next）——Q4 **比 Q8(int8) 慢**
KV probe（同 config 只換 `--kv`，冷 KV 中位數）：

| size | int8 (S6 WMMA) | k8v4 | q4_0 |
|---|---|---|---|
| ~17.5k | 1523 | 1350 (0.886×) | 1459 (0.958×) |
| ~35k | 2210 | 1460 (**0.661×**) | 1668 (**0.755×**) |

原因不是 q4 天生差：**AMD dispatch（`src/kernels/cuda/qsa_prompt_attn.cu:2499`）把 q4 pool 明確擋在 S6 tensor-core kernel 之外**（條件要求 `k_q4 == nullptr && v_q4 == nullptr`），q4_0/k8v4 一律退回 FP32 scalar prompt attention。int8 有 S6、q4 沒有 → 輸的是「有沒有上矩陣核」，不是量化本身。

### 結論性對照
.55 證明「同骨架下 Q4 流量減半 ⇒ 反超 Q8」；.67 缺的就是讓 q4 進骨架的那條路。**使用者目標：比照 .55，把 .67 的 q4 作快（要更快，不是省記憶體）。**

---

## 2. 已完成（commit 63b32f77，可編可跑）

`--kv q4_sym`：gufo 式對稱 nibble 格式的 storage 層（方案 b，使用者拍板；(a)=保留 FWHT+ggml q4_0 曾評估，(b)−(a) 預估 +2~5% prefill 且語義與 .55 一致）：

- `include/strata/kernels/kv_q4.hpp`：`block_q4_sym { uint16 s; uint8 qs[16]; }`（18B，s = amax/7，signed codes −7..7，element 2e low nibble / 2e+1 high —— **注意與 ggml q4_0 的 j/j+16 pairing 不同**）
- `src/kernels/cuda/kv_q4.cu`：`q4sym_store`（warp-per-block 確定性 amax）、`kv_append_q4sym_kernel/_batch_kernel` + wrappers（step/steps/batch，含 host copy 鏡像）
- append 全部 call site 接上：`layer.cpp`（decode）、`mtp.cpp`（drafter batch）、`verify.cpp`（batch + per-token）、`prefill.cpp`（drafter cache + 兩處 QSA prompt）
- `layer.cpp/hpp`：`g_kv_q4_sym`、`st.kv_q4_sym`、`qsa_set_kv_q4_sym()`；`kv_rot` 對 q4_sym 為 false（不旋轉）
- `generate.cpp`：`--kv q4_sym` CLI + help

現況行為：`--kv q4_sym` 能載入、能跑，但 attention 走 **FP32 scalar fallback**（跟 q4_0 一樣）——所以現在它不會比 int8 快，這正是未完成的部分。

---

## 3. 未完成（核心工作）：S6 的 q4_sym attention kernel

### 為什麼卡住（重要，前人踩過的坑）
S6 (`prompt_attn_wmma_body`) 的 V staging 是「每 cell 一行連續 codes、每 cell 一個 scale」。q4 格式每 cell 在 wave slice（64 dims）裡跨 **兩個 block、兩個 scale**：
- scores fold：kk step 0,1 屬 block 2w、2,3 屬 block 2w+1，各有 per-cell scale → 要分兩組 fold（不能像 int8 乘單一 ksc）
- p.v：per-group vmax 要改成 per-block-half 正規化，d(cell,g) 折進 A-fragment
- 我曾嘗試以 template flag 塞進既有 body，因 subtile↔scale↔fragment 索引一致性無法用 patch 保證而放棄（多次錨點失敗即此因）

### 建議做法（下一位接手者）
**另寫一顆獨立的 `prompt_attn_wmma_body_q4`**（不改 int8 body），放在 `qsa_prompt_attn.cu` S6 區塊之後：
1. 先紙上推死三者的映射：WMMA fragment（A lane l = A[l%16][(l/16)*8+i]，gfx1201 版）↔ subtile staging ↔ per-(cell,block) scale。wave w 擁有 dims [64w,64w+64) = blocks 2w,2w+1。
2. SmemQ 需要 per-subtile 結構：subtile s∈[0,4) = 16 cells × 16 dims 的 code span（pool 內 block 2w+(s>>1)、byte base 8+8*(s&1)），加 per-cell scale 陣列（每 cell 兩個 d）。
3. K 側同構（ksd per (tile, block-half)，按 fragment column=cell broadcast）。
4. 新 kernel entry + launch gate：AMD dispatch 在 int8-S6 分支之後加 q4_sym 分支（env gate 建議 `STRATA_PA_Q4SYM`，預設 off，A/B 用）。
5. 驗證順序：`qsa_prompt_attn_parity` 擴一個 q4_sym fmt（AMD 端目前 q4 段被排除，需新增）→ `kv_q4_parity` 加 sym codec 對照 → 端到端 prefill ladder（`~/attnshare67.py` 同款方法學：fresh boot、冷 KV nonce、engine log 的 "prompt N = 0 reused + N read in X ms" 行）。
6. 收益天花板參考：attention-only 微基準 int8-S6 vs FP32 fallback = **3.70×**；但 jamesrogers 端到端 attention 佔比探測只有 ~0–12%（8k 甚至 0.85×，雜訊大）→ **MoE 上 q4-S6 的端到端預期只有個位數 %**；真正吃滿 .55 那種 +9% 要在 attention-bound 的較小模型/config 上驗。

### 其他必須知道的事
- q4_sym 的 decode attention 也要路徑：`qsa_decode_attn.cu` 已有 q4_0 的 `load8_q4`（mode 2/3），sym 需要平行變體（nibble unpacking 不同：2e/2e+1 pairing、signed amax/7）。
- snapshot/streaming：q4_sym 目前沿用 k_q4/v_q4 pool 指標與 144B geometry，`kv_stream.cu` 的 kKvQ4 mover 幾何通用可直接用；但 `conversation_snapshot` 的格式判定要把 q4_sym 區分開（hash 會不同）。
- MTP drafter：q4_sym 下 drafter ring 走 `kv_append_q4sym_steps`（已接線），確認 `mtp.cpp` 的 kv_rot gate 對 sym 為 false（已處理）。
- 精度：q4_sym 無 FWHT ⇒ outlier 保護只有 per-32 scale。上游對 q4_0（有 FWHT）的實測已是文檔 ppl +8~12% vs int8；sym 版預計更差，**發版前必須做品質閘**（document ppl / needle），速度贏了也不代表能開。
- `.55` 的 Q4 env 開關：`GUFO_KV_Q4=1 GUFO_KV_V4=1`（serve flag 只認 f16|q8），原始碼 `src/core/compute/attention/blocked_kv.hip.cpp`（.55 gufo-final2）。

## 4. 環境狀態
- .67 service `strata-jamesrogers.service` 已還原原 config（`--kv k8v4`、active、health ok）。探測期間的暫存腳本已清。
- .55 gufo service 正常（bench 後已還原）。
- .67 repo 髒檔（非本次產生）：`strata-jamesrogers-mxfp4-nomtp.json`（M，探測時改過已還原原樣）、若干 `.bak-*` untracked——不要混進 commit。
- branch 上還有無關 stash@{0,1}（pr-mxfp4 的 WIP），別動。

# KICKOFF — Phase B：q4_sym 上 S6 tensor-core（`prompt_attn_wmma_body_q4`）

日期：2026-11-11
Repo：.67 (Nobara, RX9070/gfx1201) `/home/nobara-user/strata-0210`
Branch：`feat/kv-q4-s6`
前置：Phase A 已 commit `2dcfba47`（q4_sym decode KV_MODE 5，parity PASS vs FP64 3.7e-06）

---

## 0. 目標（使用者拍板「硬作沒差」）
讓 `--kv q4_sym` 的 **prompt/prefill attention** 走 WMMA 矩陣核，比照 int8-S6，
目標打平或勝過 int8-S6。目前 q4_sym prefill 退回 FP32 scalar fallback（~12.5ms vs int8-S6 ~1.5ms @ ctx8192/512q）。

## 1. 為什麼不能 patch int8 body（前人踩坑，本次另寫獨立 body）
int8 `prompt_attn_wmma_body` 假設：**每 cell 一行連續 codes、每 cell 一個 scale**。
q4_sym 每個 cell 在一個 wave slice（64 dims = HD/4）裡跨 **兩個 block、兩個 scale**：
- wave w 擁有 dims `[64w, 64w+64)` = blocks `2w`, `2w+1`（QK4_0=32 → 每 block 32 dims，64 dims = 2 blocks）
- scores fold：kk step 0,1 ∈ block 2w、kk 2,3 ∈ block 2w+1，各有 per-cell scale → 分兩組 fold，不能乘單一 ksc
- p.v：per-group vmax → per-block-half 正規化，d(cell,block) 折進 A-fragment
→ subtile↔scale↔fragment 索引一致性無法用 template flag 保證。故另寫 `prompt_attn_wmma_body_q4`。

## 2. 幾何 / 常量（沿用既有）
- THREADS=128（4 warps）、WCH=32 cells/chunk（two 16-cell tiles）、WVS=80 staged V row stride
- gfx1201: PA_KL=8, PA_KOFF(half)=(half)*8, PA_ROW(i,half)=(half)*8+i；A lane l=A[l%16][(l/16)*8+i]，B lane l=B[(l/16)*8+i][l%16]，C/D lane l=D[(l/16)*8+i][l%16]
- G=12 heads/KV-head, HD=256, NKV=2, NH=24, page_size=4
- block_q4_sym{ uint16 s; uint8 qs[16]; } = 18B；s=amax/7(fp16)，signed codes −8..7 two's-complement，element 2e low nibble / 2e+1 high nibble of byte e（**非 ggml j/j+16**）
- bytes_per_head = 8*18 = 144 = ROW4；每 cell 每 head 8 blocks

## 3. 映射推導（先紙上鎖死再寫）
### 3a. K-side（scores = q·k）
- 一個 chunk = WCH=32 cells = two 16-cell tiles（nt=0,1）
- 每 tile 每 wave 覆蓋此 wave 的 64 dims = 2 blocks（block index in-head：`2*warp`、`2*warp+1`）
- 對稱 nibble：block b 的 element e∈[0,32) → byte `qs[e/2]`，low if e even else high，signed = nib>=8?nib-16:nib，value = code * h2f(blk->s)
- kk loop 0..3：kk 對應 16-dim 子塊；kk∈{0,1}→block 2w、kk∈{2,3}→block 2w+1（因每 block 32 dims、每 kk 16 dims）
- 每個 kk 的 B-fragment = 該 (cell,block) 的 16 codes（PA_KL=8 × half 0/1）展開成 fp16，乘 **該 block 的 scale**（不是 per-cell 單 scale！）
- 兩 block 的 partial 各自 accumulate 後相加 → 得該 cell 的 q·k（跨 64 dims）
### 3b. V-side（p·v）
- 同 wave slice 64 dims = 2 blocks；V staging 需 per-(cell,block) scale（每 cell 兩個 d）
- p.v 的 A-fragment = softmax 機率 p[cell] × per-block scale（折進去），B-fragment = V codes 展開
- online softmax（mrow/lsum/alpha）與 int8 版相同結構

## 4. 新增 SmemQ（q4_sym 版 SmemWT）
- part[4][R][WCH+1]、p[R][WCH+1]、valid[WCH]、qmax/alpha/lsum/mrow 同 int8
- **改**：vs → 每 cell 兩個 scale（block 2w、2w+1）：`float vs2[4][WCH][2]`
- v codes staging：q4_sym 每 cell 每 head 144B；wave slice 只取 64 dims = 36B codes + 2 scales。staging 以「code span」存，index by (subtile s∈[0,4) = 16cells×16dims)
- 注意 bank spread：q4_sym 是 packed nibble，staging 後建議直接展開成 fp16 codes（避免每次 fragment 建時重解 nibble），或用 uint8 codes + 於 fragment 建構處 nibble-unpack

## 5. Kernel entry + dispatch gate
- `template <bool SPLIT,...> __global__ prompt_attn_wmma_q4_kernel(...)` → 呼叫 `prompt_attn_wmma_body_q4<...>`
- 新 launch helper `launch_wmma_q4(...)`（鏡像 `launch_wmma`，含 TRIM/RPF variants）
- **AMD dispatch**（`qsa_prompt_attn_batch` 內 `#if defined(__HIPCC__)` 段，line ~2498）：在 int8-S6 分支**之後**加：
  ```
  if (pools.k_q4 != nullptr && pools.v_q4 != nullptr && pools.kv_q4_sym &&
      s.head_dim == HD && s.n_head == G*s.n_head_kv && cap>0 && ids && steps && pools.page_table &&
      hip_wmma_usable(n_q,cap) && pa_q4sym_on())   // STRATA_PA_Q4SYM，預設 off
      return launch_wmma_q4(...);
  ```
- env gate `STRATA_PA_Q4SYM`（預設 off，A/B 用）

## 6. 驗證順序
1. build：`cd build-hip && ninja hip_prompt_attn_wmma`（先編得過）
2. parity harness `qsa_prompt_attn_parity.cpp`：把 fmt==3 從 decode-only 升級為也跑 new_run()（S6 q4_sym），比 new vs old(FP32 scalar) 且 vs FP64。移除 `#if !__HIPCC__` 對 q4 段的排除（AMD 端要測）
3. `kv_q4_parity` 已有 sym codec 對照（Phase storage 層建的）——確認仍綠
4. 端到端 prefill ladder（fresh boot、冷 KV nonce、engine log "prompt N read in X ms"）：q4_sym-S6 vs int8-S6 vs q4_sym-fallback
5. 品質閘（發版前必須）：document ppl / needle —— q4_sym 無 FWHT，outlier 保護只有 per-32 scale，預計比 q4_0 更差

## 7. 風險 / 已知
- MoE(jamesrogers) attention 佔比僅 0–12%（handoff §3.6）→ 端到端收益可能個位數 % 甚至淹雜訊；真正吃滿 .55 的 +9% 要在 attention-bound 小 config
- 精度：q4_sym 無 rotation，per-32 scale only → 比 q4_0(+FWHT) 更差；速度贏不代表能開
- snapshot/streaming：q4_sym 沿用 k_q4/v_q4 pool 指標 + 144B geometry，kv_stream mover 幾何通用；但 conversation_snapshot 格式判定要區分 q4_sym（hash 不同）——本 phase 不碰 streaming
- 不動 untracked junk（*.bak-*、*.pristine）；不動 stash@{0,1}

## 8. 實作進度（rolling）
- [ ] 紙上映射定稿（§3）
- [ ] `prompt_attn_wmma_body_q4` + SmemQ 寫入 qsa_prompt_attn.cu（S6 區塊後）
- [ ] `prompt_attn_wmma_q4_kernel` + `launch_wmma_q4` + `pa_q4sym_on()` gate
- [ ] AMD dispatch 接線（int8-S6 分支後）
- [ ] parity harness fmt==3 升級（AMD 端 new_run）
- [ ] build 綠 + parity PASS
- [ ] benchmark（fallback vs S6）
- [ ] 品質 gate（若 speed 有意義才做）
- [ ] commit Phase B


---

## 9. B-1 映射定稿（寫碼前鎖死）

### 幾何複習
- HD=256, QK4_0=32 → 每 head 8 blocks；block_q4_sym=18B（s:2 + qs:16）。bytes_per_head=144。
- 一個 wave 擁有 64 dims（dim0=warp*64）= **兩個 block**：block index（head 內）= `2*warp`, `2*warp+1`。
- 一個 chunk = WCH=32 cells = two 16-cell tiles（nt=0,1）。tile nt 的 cell = lane col∈[0,16)，row via __shfl(row, nt*16+col)。
- gfx1201: PA_KL=8, PA_KOFF(half)=half*8, PA_ROW(i,half)=half*8+i。A lane l=A[l%16][(l/16)*8+i]。

### K-side（scores = q·k），q4_sym vs int8 的關鍵差異
int8：一組 64 dims 一個 scale（KV_Q8_GROUP=64）→ `ksc[nt]` 單一。codes 位址 `k_q + rr*HD + dim0 + kk*16 + PA_KOFF(half)`，byte-per-elem。
q4_sym：block=32 dims → 64-dim wave slice 跨 block `2w`(kk0,kk1) 與 `2w+1`(kk2,kk3)。
  - 對稱 nibble：block b 內 element e∈[0,32) → byte `blk->qs[e/2]`，low if e even else high；signed=nib>=8?nib-16:nib；value=code*h2f(blk->s)。
  - 每個 kk（16 dims）落在某 block 的前半(e 0..15)或後半(e 16..31)：
      kk0 → block 2w,   e∈[0,16)   → bytes qs[0..7]
      kk1 → block 2w,   e∈[16,32)  → bytes qs[8..15]
      kk2 → block 2w+1, e∈[0,16)   → bytes qs[0..7]
      kk3 → block 2w+1, e∈[16,32)  → bytes qs[8..15]
  - 故 **每 tile 兩個 scale**：`ksc[nt][0]`=block 2w 的 s、`ksc[nt][1]`=block 2w+1 的 s。kk0/kk1 用 ksc[nt][0]，kk2/kk3 用 ksc[nt][1]。
  - scores fold：`s_kk = wmma_f16(qh[kk], b_kk, s_kk)`；b_kk = 該 (cell,block) 的 8 codes 展開 fp16（不含 scale）。fold 完後 `part[..] += (kk0+kk1)*ksc0 + (kk2+kk3)*ksc1`。
    → 實作：分兩條 accumulator `sa`(kk0,kk1)、`sb`(kk2,kk3)，最後 `score = sa*ksc0 + sb*ksc1`（per cell）。
  - 注意 SPLIT：qh/ql hi+lo 兩條 → sa/sb 各再拆 hi/lo = 4 accumulators/tile。可先做 non-SPLIT（SPLIT=false）版求正確，再考慮 SPLIT。

### V-side（p·v），q4_sym
int8：V staging 每 cell 一行 64 codes（`S.v[warp][lane][0..63]`）+ 一個 `vs[warp][lane]` scale。p.v 時 `b[i]=(fp16)(int8_t)S.v[warp][cb+i][d]`。
q4_sym：V 同 block 結構 → 每 cell 在 wave slice 跨 block 2w、2w+1，兩個 scale。
  - staging：每 cell 存 36B codes？不——wave slice 只取此 wave 的 64 dims = 2 blocks×(2+16B)=... 但 scale 是 per-block。
  - 簡化：**staging 直接展開成 fp16 codes**（每 cell 64 個 fp16 = 128B/wave/cell）+ per-(cell,block-half) scale 兩個。
    如此 p.v 的 B-fragment 建構 = `(fp16) staged_code`（已含？不，scale 折進 A-fragment p）。
  - scale 折法：int8 把 V scale 折進 A-fragment（`pv = S.p[col][cb+i] * (S.vs[warp][cb+i]*vup)`）。q4_sym 每 cell 有兩 scale（block 2w、2w+1）→ 折進 A-fragment 時依 kk/dim 所屬 block-half 選 scale。即 A-fragment 的 16 個 k（cells）對同一 d 用同 scale？不對——d 固定、cell 變動，scale 是 per-(cell,block-of-that-d)。
  - → V-side 正規化比 int8 複雜：需 per-(cell, block-half) vmax。建議 vup/vdown 改 per-block-half（兩組）。

### 結論：先做「展開 fp16 staging + 兩 scale」的最簡單正確版
1. SmemQ：part/p/valid/qmax/alpha/lsum/mrow 同 int8；新增 `__half vk[4][WCH][64]`（展開的 K codes，per wave/cell/64dims）其實 K 不需 staging（K 直接 global load 到 fragment）。
2. K-side：fetch 改讀 block_q4_sym，展開 nibble→fp16 codes 進 `kx`（每 kk 8 codes，共 4 kk），並抓兩個 scale。
3. V-side：staging 展開 V codes 成 fp16（`S.vh[4][WCH][64]`）+ 兩 scale `S.vs2[4][WCH][2]`；p.v 依 block-half 折 scale。
4. 先 SPLIT=false、TR=0、RPF=false 求正確；parity PASS 後再談優化變體。

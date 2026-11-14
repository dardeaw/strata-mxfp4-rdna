# Strata 7800XT (gfx1101) 優化交接清單

**日期**：2026-11-14　**接手者須知**：本機無 gh push 權限，所有 commit 走 bundle→.55 relay。

---

## 0. 一句話現況
`.67`（RX 7800 XT / gfx1101 / RDNA3）跑 Strata MXFP4-Next，已達可交付基線；能靠設定動的槓桿全用盡，剩兩個瓶頸是結構性的（見 §5、§6）。服務目前健康（smoke "Paris" ✓）。

## 1. 主機 / 連線
| 主機 | 角色 | 連線方式 |
|---|---|---|
| `192.168.68.67` `nobara-user` | **生產機**（7800XT, 77GB RAM, 20 cores） | `python "C:\Users\馮\.config\opencode\run_ssh.py" 192.168.68.67 "<cmd>"`；多步 `--runfile <local.sh>`；傳檔 `--upload/--download` |
| `192.168.68.55` `dardeaw` | push relay + embedding | 同上換 IP |
| 本機 `.102` | 操作端 | — |

- **禁止 raw ssh/scp**；inline Python/JSON 會被 PowerShell 咬，一律寫 .sh 走 --runfile。
- paramiko channel timeout ≈600s，長任務要 `setsid nohup ... &; disown` 背景化再輪詢。
- **迴圈守衛**：同一個狀態檢查命令重複貼會被 `BLOCKED_BY_LOOP_GUARD` 擋——別刷同一條。

## 2. Repo / 分支 / 上版流程
- `.67` repo：`/home/nobara-user/strata-0210`，branch **`gfx1101`**，HEAD = `af37d6e6`
- GH：`dardeaw/strata-mxfp4-rdna`（gufo-org/gufo 的 fork），branch `gfx1101`
- **上版四步**（.67 不能直接 push）：
  1. `.67`: `git bundle create ~/X.bundle <base>..gfx1101`
  2. `--download` 到本機 `$env:TEMP\opencode\`
  3. `--upload` 到 `.55:/home/dardeaw/`
  4. `.55`: `cd ~/git/strata-mxfp4-rdna && git fetch ~/X.bundle 'gfx1101:refs/heads/gfx1101' && git push origin gfx1101`
  5. 驗證：`.55` `gh api repos/dardeaw/strata-mxfp4-rdna/branches/gfx1101 --jq .commit.sha`
- token scopes 無 `workflow`——別碰 `.github/workflows/*`。
- commit 風格：`git -c user.name=opencode -c user.email=opencode@local commit -m '<type>(<scope>): ...'`

## 3. 生產啟動指令（照抄）
```
cd /home/nobara-user/strata-0210
setsid nohup bash -c 'cd /home/nobara-user/strata-0210; \
  setsid .venv/bin/python -m serve.server --engine strata \
  --config strata-jamesrogers-mxfp4-7800xt.json --port 8081 --host 0.0.0.0 \
  >> wrapper-8081.log 2>&1 < /dev/null &'
```
- API：`http://127.0.0.1:8081/v1`，token `strata67`，model id `qwen3.8-flash-next-tsaipifong-mxfp4-mtp`
- engine binary：`build-hip1101/strata`（ROCm 7.2.4 / clang 20；`build-hip1101b` 是 7.10 build，慢 7%，僅參考）
- 啟動 log：`strata-jamesrogers-nomtp.log`（engine stderr）、`wrapper-8081.log`（wrapper）

## 4. 關鍵檔案
| 路徑 | 說明 |
|---|---|
| `/home/nobara-user/strata-0210/TUNING-7800XT.md` | **完整報告**（6 節，含所有發現與數據）— 先讀這個 |
| `…/strata-jamesrogers-mxfp4-7800xt.json` | 生產 config（gitignored；user_args pin kv-resident 20480） |
| `…/ladder78.py` `react78.py` | benchmark harness（已 commit） |
| `…/src/prefill/prefill.cpp` | wait copy / Stager / ring（行號見 §5） |
| `…/src/prefill/kernels.cu` | GDN chunked kernel（見 §7） |
| `…/serve/server.py:3525` | vram_fit.fit_start 呼叫點 |
| `…/strata_control/vram_fit.py` | KV_RESIDENT_STEPS 覆寫邏輯 |
| `…/src/core/arch_defaults.cpp` | gfx11/gfx12 defaults 表 |
| `…/include/strata/kernels/gfx_arch.hpp` | arch 判定 |

## 5. 已完成 / 已驗證的發現（勿重複做）
1. **kv-resident clamp（已修）**：`vram_fit.fit_start` 有硬編碼階梯 `KV_RESIDENT_STEPS=(20480,8192,4096)`，VRAM 估算不足就踩到最小並覆寫 config。修法=config 加 `"user_args":["--kv-resident","20480"]` → `kv_fixed=True` 豁免。OVERFLOW 消失，prose decode ~30→35–39.6。**這是今天唯一實質增益。**
2. **wait copy ≠ 頻寬**：PCIe pinned H2D 實測 26.3 GB/s、NVMe direct 1.9 GB/s、ring drain 0.01s。全部正常。
3. **stager 調參無效（負面結果）**：STAGER_THREADS=32/RING=128/BATCH=8 vs baseline，14.4K prompt 1423/1425 vs 1372/1446，雜訊內。因 55GB experts 已 page-lock 在 RAM，非 SSD-fault 情境，加深 ring 沒東西可加速。已回滾。
4. **GDN chunked 不用 port**：gfx11 版 `gdn_wmma_scan_kernel` 已存在但被 SM-count gate 擋；`STRATA_GDN_CHUNKED=2` bypass 後反而慢（pp2 970 vs 1097）。維持 `=0`。
5. **PF_PAD 已在 gfx11 defaults 生效**，無需另設。

## 6. 兩個未解瓶頸（接手者的真正工作）
### A. wait copy 4.45s @19.6K（佔 GPU timeline 32.5%）
- 性質：compute stream 卡在 expert blob DMA completion event（`prefill.cpp` 的 `flush` lambda + `wait_copied`，mark 點約 4858/5776/6172/6205/7349）。依賴 + ring 深度的結構性等待。
- adaptive overlap 在 resident-RAM mode 預設關（需 `STRATA_ADAPT_OVERLAP_RESIDENT=1`，開了也無效）。
- **要壓它只能改 engine staging 邏輯**（route-aware prefetch / coalesced copy），且會跟已很緊的 VRAM（16.34/17.16 GB）搶空間。高風險 C++ 工程。
### B. gemm gate/up 20.7%（硬體閹割）
- 最快路徑 MXFP4 W4A8 需 FP8(E4M3) WMMA = **RDNA4/gfx12 only**；`fp8_wmma_device()` 對 gfx110x 直接 false → 退回 FP16 expert path。
- 這是 9070(RDNA4) 比 7800XT 快的核心原因之一。**config 層面不可恢復**，要吃到得換卡或寫 gfx11 專屬 FP16 GEMM 優化。

## 7. GDN port 細節（若堅持要做 §6-B 之外的嘗試）
- 參考 `gdn_wmma12.inc`（gfx12 版，wave32 fragment layout）+ `gdn_tc.cuh` + `gdn_chunk.cu`
- gfx12 accumulator 可直接當下一個 B operand；gfx11 不行（需 permlane/byte permute）——移植難點在此
- 驗證工具：`tests/hip/gdn_chunked.cpp`、`gdn_rec_parity.cu`、`gdn_harness.cu`
- ⚠️ 改錯 = 靜默數值錯誤，務必跑 parity（bound 1e-4 of scale）

## 8. 量測協議（必須遵守，否則數字騙人）
- 丟掉 warmup（冷 DPM ramp 損 30–40%）；重複 ≥2 次取樣
- unique NONCE + reuse=none（否則 prefix cache 回 reused 灌水）
- temp=0 求確定性；decode 看 engine log 的 `drafts accepted X of Y`（MTP acceptance 正常 85–86%）
- ladder 現況（tok/s）：604/858/1456/1377/1550/1551/1520/1403（2K→250K）
- ReAct：7/8 pass（summary 題模型選 search_docs，語意模糊非 bug）

## 9. 環境變數現值（config env，13 keys）
`STRATA_FP8_GROUPED=0, STRATA_GDN_CHUNKED=0, STRATA_GFX12_DEFAULTS=0, STRATA_HC_Q8=1, STRATA_HC_UPMIX=1, STRATA_HIP_WMMA=1, STRATA_MXFP4_FP8=0, STRATA_PA_FAST=1, STRATA_PF_FUSED_MXFP4=1, STRATA_PF_GEMM=1, STRATA_PREFILL_CPU_SHARE=1, STRATA_ROUTE_TAIL_SKIP=7, STRATA_SELECT_WMMA=1`
（注意 `STRATA_GFX12_DEFAULTS=0` 是因為這台是 gfx11；arch_default_env 另有 gfx11 表自動注入 22 switches）

## 10. 已知坑
- venv ROCm 7.10 的 rocBLAS Tensile lib 只有 gfx1200/1201 → gfx1101 binary 跑它会 SIGABRT。要用系統 `/usr/lib64`。
- hipBLASLt tuning table 對 gfx1101 生成了但零效果（大 GEMM 走 fused/MMQ 不进 contractor）。
- `--kv-grow` 在此為 no-op（單 GPU+profile+全 expert in RAM）。
- unified memory 讓 bwtest 報假的高頻寬——要 validated copy + event 計時才準。
- `--max-context 262144`，`--prefill 16384`，`--spec 4 --spec-min-p 0.5`，`--resident-budget-gib 55`。

## 11. 下一步建議優先序
1. （低報酬）試 `STRATA_RING_BYTES` / `STRATA_PREFILL_STREAM_MIN` 掃描——但 §5-3 顯示這類多半無效
2. （中報酬・高風險）engine staging 重設計攻 wait copy——需熟 prefill.cpp 且備好 parity 回歸
3. （放棄）gate/up FP8——硬體不支持，除非換 RDNA4
4. 維護：確認開機自啟（目前手動 setsid 起，未建 systemd unit——若要持久化請補）
```
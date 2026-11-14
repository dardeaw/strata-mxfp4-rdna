# Strata 7800XT (gfx1101) tuning report — 2026-11-14

## Findings

### 1. --kv-resident was silently clamped to 4096 (FIXED)
`serve/server.py:3525` calls `vram_fit.fit_start()` on every start;
`strata_control/vram_fit.py` walks `KV_RESIDENT_STEPS = (20480, 8192, 4096)`
downward when its VRAM estimate is short and **rewrites** the config's value.
Fix: `"user_args": ["--kv-resident", "20480"]` in the config sets
`kv_fixed=True`, which exempts the flag from the walk.
Result: KV streaming OVERFLOW message gone (99.6%+ block reads hit VRAM),
prose decode ~30 -> 35-39.6 tok/s.

### 2. wait copy 4.45s @19.6K is NOT bandwidth-bound
Measured: pinned H2D 26.3 GB/s (hipcc event-timed, 512MB x3), NVMe direct
1.9 GB/s, page-cache read 2.0 GB/s. Ring drain (96 slots x ~2MB MXFP4 blobs)
= 0.01s. The stall is sync/serialization of the expert staging pipeline
(ring depth + per-layer waits), not PCIe or disk.

### 3. GDN chunked WMMA exists for gfx11 but is slower here
`gdn_wmma_scan_kernel` (kernels.cu:1994) is gated by an SM-count check
(60 CU < needed blocks). STRATA_GDN_CHUNKED=2 bypasses the gate but measured
slower than the sequential recurrence (pp2 970 vs 1097 tok/s cold-cached
prompt probe). Kept OFF (STRATA_GDN_CHUNKED=0 in config env).

### 4. PF_PAD already active via gfx11 defaults table
`arch_default_env` includes STRATA_PF_PAD=1 for all gfx11; no extra action.

## New baseline (kv-resident 20480, MTP spec 4, warmed, unique nonce)
ladder (tok/s): 604 / 858 / 1456 / 1377 / 1550 / 1551 / 1520 / 1403
               (2K   4K    8K    16K   32K   65K   131K  250K)
decode: prose 35-39.6, fib 41.7-43, code 43.2, json 29.2
ReAct: 7/8 pass, median 14.0 tok/s
MTP acceptance: fib 150/174 (86%), code 90/106 (85%)

## Config state
- file: strata-jamesrogers-mxfp4-7800xt.json
- user_args: ["--kv-resident", "20480"]
- env adds: STRATA_GDN_CHUNKED=0 (explicit off)

## 5. wait copy deep-dive (2026-11-14, later) — NOT bandwidth, NOT stager
Confirmed the 4.45s @19.6K is a structural staging-sync stall, not fixable by tuning:
- PCIe pinned H2D measured 26.3 GB/s; NVMe direct 1.9 GB/s; ring drain ~0.01s. Bandwidth is fine.
- Stager profile sweep (STRATA_STAGER_THREADS=32 + RING=128 + BATCH=8, the SSD-profile
  that helps GGUF-from-SSD): on a 14.4K unique prompt gave 1423/1425 tok/s vs baseline
  1372/1446 -> within noise, NO gain. Reverted (removed from config env).
- adaptive overlap is OFF in resident-RAM mode unless STRATA_ADAPT_OVERLAP_RESIDENT=1;
  enabling it earlier also showed no change to wait copy.
Conclusion: wait copy is the compute stream waiting on expert-blob DMA completion events
(prefill.cpp flush/wait_copied), gated by dependency + ring depth, not by link speed.
Reducing it needs an engine-side staging redesign (route-aware prefetch / deeper coalesced
ring), which competes with the already-tight VRAM. Not a flag-level win.

## 6. gemm gate/up (20.7%) is hardware-gated, not tunable
The fast MXFP4 W4A8 path needs FP8 (E4M3) WMMA = RDNA4/gfx12 only (prefill.cpp fp8_wmma_device()
returns false for gfx110x). On RDNA3 the FP8 GEMMs cannot launch, so gate/up runs the FP16
expert path. All available switches (PF_FUSED_MXFP4/PF_GEMM/PF_PAD/PF_F16) are already on.
This is why the 9070 (RDNA4) beats the 7800XT here; not recoverable via config.

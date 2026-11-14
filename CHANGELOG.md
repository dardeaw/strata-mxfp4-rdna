# Changelog

## RDNA3 (GFX1101 / RX 7800 XT) Tuning, MBufo Tools Ladder & 8-Task ReAct Suite (fork) — 2026-11-14

Version 0.1.60.

### RDNA3 / RX 7800 XT Production Optimization
- **KV-Resident Clamp Fix & Memory Staging**:
  - Identified and fixed `vram_fit.fit_start()` downward walking bug that silently clamped `--kv-resident` from `20480` to `4096` when estimating tight VRAM.
  - Adding `"user_args": ["--kv-resident", "20480"]` marks `kv_fixed=True`, exempting production config from the walk.
  - Result: KV streaming VRAM overflow messages eliminated (99.6%+ block reads hit VRAM), decoding speed restored to **35–39.6 tok/s** (prose) and **45–52.5 tok/s** (fib) with 86–92% MTP acceptance.
- **7800XT Tuning Configuration (`configs/mxfp4-rdna-7800xt.json`)**:
  - GFX1101 tuned production configuration: `resident-budget-gib: 59.5`, `STRATA_GDN_CHUNKED=0`, `STRATA_PREFILL_LEND_PCT=100`, `STRATA_PREFILL_RING=288`.
  - Thorough structural analysis of expert staging pipeline: pinned PCIe H2D delivers 26.3 GB/s, NVMe direct delivers 1.9 GB/s, ring drain completes in 0.01 s.
- **GFX11 Kernel Optimizations (`src/kernels/cuda/qsa_prompt_attn.cu`)**:
  - 16-byte V fast-path for GFX11 (using dual uint64 aligned loads).
  - Explicit RDNA4 V-loop v8 fast-path guards restricting to `PA_KL==8` (GFX12 only).

### Agent Benchmarking Twin Suites ("雙壁")
- **MBufo-Style Tools Ladder (`tools/tools_ladder.py` / `tools_ladder.py`)**:
  - Faithful reimplementation of MBufo's `concbench/tool_ladder.py` architecture.
  - Tests tool-call degradation across geometric context lengths (`0 -> 500 -> 2,000 -> 8,000 -> 16,000 -> 32,000 -> 64,000 -> 128,000` tokens) with controlled filler text.
  - Validates tool dispatch accuracy, detects failure modes (`no-call`, `text-json` markdown hallucination, schema mismatch), and breaks down pass rates by context depth rung and task domain.
  - **RX 7800 XT Verified**: **24/24 PASS (100%)** across all context length rungs (0 to 128K).
- **8-Task ReAct Agent Benchmark (`tools/tools_react.py` / `tools_react.py`)**:
  - Evaluates 8 functional agent domains (`weather`, `email`, `search`, `calc`, `translate`, `read`, `shell`, `summary`) under deterministic `temperature=0` decoding.
  - **RX 7800 XT Verified**: **8/8 PASS (100%)**, median decode **14.2 tok/s**.

### Comprehensive RX 7800 XT (GFX1101, 16 GB, ROCm 7.2.4) Benchmarks
- **Prefill Ladder**: 2K **351.2** (cold) | 4K **1,095.3** | 8K **1,497.2** | 16K **1,415.1** | 32K **1,669.3** | 65K **1,794.9** | 131K **1,668.4** | 250K **1,759.8 tok/s**.
- **Needle in a Haystack**: 2K, 8K, 32K at 10%, 50%, 90% depths — **9 of 9 FOUND (100%)**.
- **Decode**: prose **34.7–37.4**, fib **45.1–52.5**, code **43.2–55.3**, json **26.8–46.4 tok/s**.
- **MTP Acceptance**: fib **92.2%** (153/166), code **81.1%** (90/111), json **88.6%** (39/44).

## RDNA3 (GFX11) Auto-Defaults, MXFP4 F16 Fused Experts & Q4_0 Prompt Attention (fork) — 2026-11-13

Version 0.1.59.

### Architecture & Auto-Detection
- **Family-Prefix AMD Detection (`include/strata/kernels/gfx_arch.hpp`, `src/core/arch_defaults.cpp`)**:
  - Replaced hardcoded `gfx1151` checks across kernels and startup defaults with prefix matching (`gfx11*` for RDNA3/RDNA3.5, `gfx12*` for RDNA4).
  - Automatically applies verified GFX11 prompt + decode tuning (`STRATA_PF_F16=1`, `STRATA_PF_FUSED=1`, `STRATA_PF_FUSED_MXFP4=1`, `STRATA_GDN_CHUNKED=1`, `STRATA_PA_TRIM=0`, `STRATA_SH_STREAM=0`, etc.) without overwriting user-specified environment variables.
  - Preserves RDNA4 (`gfx12*`) defaults and discrete-GPU `--kv-grow` / `--kv-resident` memory strategy untouched.
  - Integrated GFX11 UMA GPUs (`integrated == true`) automatically enable `STRATA_PF_W16_CACHE_MIB=16384` (with an 8 GiB free-VRAM safety reserve), while discrete GPUs keep it off (`0`) to reserve VRAM for `--kv-grow` and `--expert-cache`.

### Performance & Bug Fixes
- **GFX11 `native_f16_kernel<T_MXFP4>` (`src/prefill/moe_fused_iq.cu`)**:
  - Enabled `T_MXFP4` in `gu_covered` and `d_covered_on` on GFX11 when `STRATA_PF_F16=1` and `STRATA_PF_FUSED_MXFP4=1`.
  - Fixed `kvalues_fp4` lookup-table initialization in `native_f16_kernel` and added 64x FP16 scale compensation to prevent subnormal underflow on E8M0 block scales.
- **GFX11 Q4_0 Prompt Attention (`src/kernels/cuda/qsa_prompt_attn.cu`)**:
  - Fixed GFX11 Q4 WMMA bug and defaulted `PA_TRIM=0` on `hip_wmma_arch(dev) == 1` (`hip_prompt_attn_wmma`: `FAILURES: 0`, `2.61x` speedup on `q4_0 ctx 1500`).
- **HIP Padded FP16 Dense Weight Cache (`src/prefill/gemm.cu`)**:
  - Added `STRATA_PF_W16_CACHE_MIB` support to the HIP padded `Gemm::native` path (`ldw = K + 64`), eliminating repeated dense-weight dequantization across chunks and prompts.

### Benchmark Results (Strix Halo Radeon 8060S, `gfx1151`, DPM `auto`, `--kv q4_0`)
- **Prefill Ladder**: 2K **1,079.6** | 4K **1,166.5** | 8K **1,219.6** | 16K **1,237.9** | 32K **1,222.7** | 65K **1,218.3** | 131K **1,182.9 tok/s**.
- **Decode (Median of 3, `--spec 4 --short-read 64 --mtp-q4 head`)**: short **37.50** | prose **37.80** | code **59.70** | json **62.40** | repeat **79.20** | fib **63.30 tok/s**.

## Q4_0 Matrix-Core Acceleration & Production Default (fork) — 2026-11-12

Version 0.1.58.

### Performance & Architecture
- **Production Default**: Default KV storage switched to `--kv q4_0` across CLI (`src/program/generate.cpp`) and default configs (`strata-jamesrogers-mxfp4-nomtp.json`, `configs/mxfp4-rdna-mtp.json`). Halves host RAM & KV memory footprint while matching `int8` MTP acceptance rates.
- **RDNA4 Matrix-Core (WMMA) Prompt Attention for Q4_0**:
  - Gated into hardware WMMA path on AMD RDNA4 (gfx12) instead of falling back to scalar loop.
  - Transposed LDS structure `S.v[4][32][40]` (40-byte stride) reduces LDS footprint from 16 KB to 4.5 KB and eliminates 4-way LDS bank conflicts.
  - Aligned 8-byte `ds_read_b64` memory loads and on-the-fly fragment unpacking directly in registers (`wh8`), minimizing VGPR pressure and boosting occupancy.
  - Signed scale correction via `fabsf` achieves bit-level parity against FP64 reference (`FAILURES: 0`).
- **Benchmark Results**:
  - Long-context (≥32K) prefill restored from ~1,670 tok/s to **2,469 ~ 2,529 tok/s** (+43%~50% speedup).
  - 250K cold prefill completed in 105.1 s (2,363.5 tok/s).
  - 8/8 ReAct agent suite passed with median 63.4 tok/s.

## Q4 KV performance (fork) — 2026-11-12

Version 0.1.57.

### Performance

- `--kv q4_sym`: prompt attention runs on the gfx12 WMMA matrix cores (S6)
  instead of the FP32 scalar fallback — 4.4x on the attention kernel, ~15%
  faster prefill at 16K context (2.85 vs 12.5 ms per 8K-query chunk).
  `STRATA_PA_Q4SYM=0` restores the old path.

### Changed

- `--kv q4_sym` prompt attention is on by default.

## MXFP4 for RDNA4 (fork) — 2026-11-05

Fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), based on 0.2.10 +
PR #3575. Unrelated to gufo, MBufo, and halogen.

### Added

- MXFP4 (E2M1/E8M0) experts on the FP8 (E4M3) prompt path, following vLLM's
  Quark W4A8 model: the E8M0 block scale is folded into the E4M3 weight (exact),
  the activations are quantized per row, and the products run on gfx12's FP8
  WMMA (`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`), with FP32
  accumulators (`a62189cb`).
- A 128-thread / 4-wave 64x64 FP8 GEMM with A/B staged in LDS, used for the up
  projection from T >= 256 and the down projection from T >= 128; 2x2 and 4x4
  kernels for the smaller tiles. The micro-benchmark reaches 36 TFLOPS (up) and
  26 TFLOPS (down) on gfx1201.
- The `--kv-grow` grow configuration (`configs/mxfp4-rdna.json`) as the default:
  the dense weights stay on the GPU, the VRAM goes to the expert cache, and the
  KV grows on demand. GPU 100%, CPU near idle (vs 64.6% CPU / 79% GPU with
  `--kv-resident 32768`).
- `tools/mxfp4_rdna_fix_pack.py`: drops the in-file MTP block (`blk.48`) from
  `native_experts.txt` and `index.txt` so the trunk is 48 layers.
- English `README-MXFP4-RDNA.md` and this changelog.

### Fixed

- The FP8 per-row activation scale was stored at the gathered `o0` offset into a
  buffer sized `T` floats; for a 1716-token prompt the offset exceeded `T`,
  corrupted later buffers, and produced non-finite logits (HTTP 503). It now
  uses row 0 like the `run_impl` path.
- In-file MTP (`qwen4exp.nextn_predict_layers=1`, `block_count=49`): the
  geometry reads the trunk as `block_count - nextn_predict_layers` (48), accepts
  a 49-entry `compress_ratios` while validating the 48 trunk layers, and the
  native dense loader skips `blk.N >= 48` (the `blk.48.nextn.eh_proj` failure).

### Performance

Prefill, same model/config, `--spec 4` without MTP (tok/s):

| target | Q4_K_XL | MXFP4-FP16 | MXFP4-FP8 | FP8/FP16 | FP8/Q4 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 239.5 | 265.2 | 267.4 | 1.01x | 1.12x |
| 8192 | 629.5 | 783.0 | 789.7 | 1.01x | 1.25x |
| 32768 | 822.7 | 1296.9 | 1298.7 | 1.00x | 1.58x |
| 131072 | 996.3 | 1344.6 | 1349.1 | 1.00x | 1.35x |
| 245000 | 621.4 | 1369.8 | 1382.0 | 1.01x | 2.22x |

Decode (median tok/s): short 24.0/23.4 (FP16/FP8), prose 32.4/32.1, code
31.7/30.7, json 27.3/29.2.

### Notes

- `--spec 4` runs without the in-file MTP block; decode is non-speculative.
- The FP8 prefill win is small (1.00-1.02x); the decode is mixed and partly
  CPU/launch bound. The next step is a fused/grouped expert kernel if a larger
  margin is needed.
- Upstream base: 0.2.10 + PR #3575. This fork is about Strata/RDNA4 only.

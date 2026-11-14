# MXFP4 for RDNA4 (gfx12) and RDNA3 (gfx11) — Strata fork

This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) that runs
**MXFP4 experts** (E2M1 weights with E8M0 scales, 32-value blocks) on AMD GPUs.
On **RDNA4 / gfx12** it uses **FP8 (E4M3) compute** (vLLM's Quark W4A8 model);
on **RDNA3 / gfx11xx** — and any card without FP8 WMMA — it automatically falls
back to **FP16 compute**. It also ships a `--kv-grow` configuration that keeps
the dense weights on the GPU and gives the rest of the VRAM to the expert cache.

> This fork is about **Strata on RDNA4 and RDNA3**.
> It is not related to gufo, MBufo, or halogen, and nothing here is pushed to
> those repositories.

## Status

- GPU Architecture Support:
  - AMD RDNA4 (`gfx1200`, `gfx1201`, `gfx1250`, e.g. Radeon RX 9070 / 9070 XT)
  - AMD RDNA3 Discrete (`gfx1100`, `gfx1101`, `gfx1102`, e.g. Radeon RX 7900 XTX / 7800 XT)
  - AMD RDNA3.5 / RDNA3 Integrated APU (`gfx1150`, `gfx1151`, e.g. Strix Halo / Radeon 8060S)
- Model: `tsaipifong/Swift-1.5-Qwen3.8-Flash-Next-MXFP4-GGUF` (`ple-IQ4_NL`, 3 shards, 99.75 GB).
- Default: `configs/mxfp4-rdna.json` (grow config) + automatic architecture defaults.
- The MXFP4 experts are dequantized to E4M3 with their E8M0 block scale folded
  in (exact: a power of two times at most four significant bits), the
  activations are quantized per row to E4M3, and the products run on gfx12 FP8
  WMMA.

## GPU Support: RDNA4 vs RDNA3

The MXFP4 route is the same on both families (E2M1 weights + E8M0 scales); only
the compute precision differs. The choice is made at startup from the card's
`gcnArchName`, never from the env var alone, so a wrong `STRATA_MXFP4_FP8` can
never select a kernel the GPU cannot run.

| | RDNA4 (gfx12: RX 9070 / 9070 XT, gfx1200/1201/1250) | RDNA3 (gfx11xx: RX 7900 XTX, 7800, 7700, ...) |
|---|---|---|
| Weights | MXFP4 -> E4M3 inside GEMM (folded kernel: reads 17-byte blocks directly, no dequant pass) | MXFP4 -> **FP16** (`iq_dequant_*_f16`, llama.cpp formulas) |
| Compute | **FP8 E4M3 WMMA** (`wmma_f32_16x16x16_fp8_fp8_w32_gfx12`), FP32 accumulate | **FP16 matrix cores**, FP32 accumulate |
| Activations | dynamic per-row E4M3 quantize | FP16 (no quantize) |
| Selected by | `mx_fp8_on()` == gfx12 and `STRATA_MXFP4_FP8=1` | anything else, or `STRATA_MXFP4_FP8` unset / `0` |
| Speed | Reference setting: prefill ~2.3-2.5K tok/s at 32K-250K on RX 9070 | Optimized matrix cores: prefill ~1.8-1.9K tok/s at 32K-250K on RX 7800 XT |

**Architecture-specific execution:** `gfx11xx` has no FP8 (E4M3/E5M2) matrix-core
instructions. The kernels build `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
for `gfx1201`; on an RDNA3 card they cannot launch. The engine reads `gcnArchName`,
and if it is not `gfx12*` it automatically selects the optimized FP16 expert GEMMs:

```
prefill: STRATA_MXFP4_FP8 is set but this GPU has no FP8 WMMA
(gfx12/RDNA4 required, got gfx1101) -- falling back to the FP16 expert path
```

Overrides:

- `STRATA_MXFP4_FP8=0` (or unset on non-gfx12): use the FP16 path everywhere.
- `STRATA_FP8_WMMA=0`: force the FP16 path even on a gfx12 card.
- On RDNA4 the FP8/folded fast path is the default.

Both architectures are fully validated: RDNA4 runs native FP8 WMMA (`gfx1201`), while RDNA3 runs optimized FP16 matrix-core fused kernels (`gfx1101` RX 7800 XT, `gfx1151` Strix Halo).

## Results

### RDNA4 (AMD Radeon RX 9070, 16 GB, gfx1201)

Measured with `bench_protocol.py` on AMD Radeon RX 9070 (16 GB, gfx1201, Nobara Linux), running `--kv q4_0` (optimized WMMA matrix-core S6 kernel) with in-file MTP on (`--spec 4`, resident budget 55 GiB, 13 CPU pool workers):

#### Prefill Ladder (Cold context, unique nonce, max_tokens=1)

| Target Tokens | Actual prompt_n | Prefill Speed | Wall Time |
|---:|---:|---:|---:|
| 2,048 | 2,049 | **922.8 tok/s** | 2.2 s |
| 4,096 | 4,057 | **1,273.6 tok/s** | 3.2 s |
| 8,192 | 8,233 | **1,940.0 tok/s** | 4.3 s |
| 16,384 | 16,302 | **2,383.6 tok/s** | 6.9 s |
| 32,768 | 32,552 | **2,523.2 tok/s** | 12.9 s |
| 65,536 | 65,210 | **2,469.5 tok/s** | 26.5 s |
| 131,072 | 130,345 | **2,529.6 tok/s** | 51.6 s |
| 250,000 | 248,085 | **2,363.5 tok/s** | 105.1 s |

#### Decode Scenarios (Cold KV, MTP spec=4)

| Scenario | Temperature | max_tokens | Engine Speed | MTP Acceptance |
|---|---:|---:|---:|---:|
| prose | 0.4 | 256 | **39.7 tok/s** | 53.8% |
| fib | 0.4 | 256 | **57.8 tok/s** | 89.4% |
| json | 0.0 | 128 | **50.2 tok/s** | 83.0% |
| code | 0.0 | 128 | **63.6 tok/s** | 81.7% |
| repeat | 0.0 | 256 | **61.1 tok/s** | 87.5% |

#### 8-Task ReAct Agent Suite
- Tool selection accuracy: **8/8 (100%)**
- Median decode speed: **63.4 tok/s** (range 54.1 – 74.2 tok/s)
- Long KV tools ladder: All 6 rungs PASS with prefix caching ~0.6 s

### RDNA3 Discrete GPU (AMD Radeon RX 7800 XT, 16 GB, gfx1101)

Measured under standard server timing protocol (`bench_protocol.py`, `timings.prompt_per_second` & `predicted_per_second`) with `configs/mxfp4-rdna-7800xt.json` (`user_args: ["--kv-resident", "20480"]`, `--spec 4`, `--kv q4_0`, ROCm 7.2.4):

#### Prefill Ladder (Cold context, unique nonce, max_tokens=1)

| Prompt Size | Actual Tokens | Prefill Speed | Wall Time |
|---:|---:|---:|---:|
| 2K | ~2,054 | **622.5 ~ 625.5 tok/s** | 3.3 s |
| 4K | ~4,064 | **1,250.8 ~ 1,255.5 tok/s** | 3.3 s |
| 8K | ~8,251 | **1,763.6 ~ 1,770.7 tok/s** | 4.7 s |
| 16K | ~16,339 | **1,327.8 ~ 1,615.2 tok/s** | 10.2 ~ 12.3 s |
| 32K | ~32,635 | **1,868.3 ~ 1,871.0 tok/s** | 17.5 s |
| 65K | ~65,377 | **1,960.1 ~ 1,966.8 tok/s** | 33.3 s |
| 131K | ~130,676 | **1,836.6 ~ 1,887.2 tok/s** | 69.3 ~ 71.2 s |
| 250K | ~248,716 | **1,797.5 ~ 1,809.6 tok/s** | 137.5 ~ 138.5 s |

#### Decode & Agent Verification
- **Decode Throughput & MTP Acceptance (Engine Timings)**:
  - `code`: **55.6 ~ 56.9 tok/s** (MTP accept: 81.8% ~ 85.0%)
  - `fib`: **50.8 ~ 51.1 tok/s** (MTP accept: 87.9% ~ 88.4%)
  - `json`: **45.3 ~ 46.5 tok/s** (MTP accept: 86.7%)
  - `repeat`: **43.6 ~ 43.8 tok/s** (MTP accept: 70.0%)
  - `prose`: **36.8 ~ 38.2 tok/s** (MTP accept: 51.6% ~ 56.0%)
- **Agent Tool Calling (8/8 PASS)**:
  - Tasks 1..8 decode throughput: **46.3 ~ 62.7 tok/s** (median **54.4 tok/s**).
- **Needle in a Haystack**: 2K, 8K, 32K at 10%, 50%, 90% depths — **9 of 9 FOUND (100%)**.
- **Tools Ladder (MBufo style, 0..128K context)**: **24 of 24 PASS (100%)**.

### RDNA3.5 Integrated APU (Strix Halo Radeon 8060S, gfx1151)

Measured on Strix Halo Radeon 8060S (`gfx1151`, DPM `auto`, `--kv q4_0`, 128 GB UMA):

| Prompt Size | Tokens | Prefill Speed | Wall Time |
|---:|---:|---:|---:|
| 2K | 1,861 | **1,079.6 tok/s** | 1.76 s |
| 4K | 3,666 | **1,166.5 tok/s** | 3.18 s |
| 8K | 7,286 | **1,219.6 tok/s** | 6.02 s |
| 16K | 14,927 | **1,237.9 tok/s** | 12.10 s |
| 32K | 29,795 | **1,222.7 tok/s** | 24.42 s |
| 65K | 59,746 | **1,218.3 tok/s** | 49.10 s |
| 131K | 119,643 | **1,182.9 tok/s** | 101.23 s |

Decode medians (`--spec 4 --short-read 64 --mtp-q4 head`): `short 37.50`, `prose 37.80`, `code 59.70`, `json 62.40`, `repeat 79.20`, `fib 63.30 tok/s`.

## Architecture Auto-Detection & Tuning

The engine automatically detects AMD GPU families by `gfx` prefix (`gfx11*` for RDNA3/RDNA3.5, `gfx12*` for RDNA4) and whether the GPU is integrated (UMA) or discrete, applying architecture-specific defaults automatically (`src/core/arch_defaults.cpp`) while preserving explicit environment variable overrides.

### Automatic Strategy by GPU Type

| Setting | RDNA4 dGPU (`gfx12*`, e.g. RX 9070) | RDNA3 dGPU (`gfx11*`, e.g. RX 7800 XT) | RDNA3.5 / RDNA3 iGPU (`gfx11*` UMA, e.g. Strix Halo) |
|---|---|---|---|
| MoE Expert Path | FP8 E4M3 WMMA (`STRATA_MXFP4_FP8=1`) | **F16 Matrix-Core Fused** (`STRATA_PF_F16=1` + `STRATA_PF_FUSED_MXFP4=1`) | **F16 Matrix-Core Fused** (`STRATA_PF_F16=1` + `STRATA_PF_FUSED_MXFP4=1`) |
| Prompt Attention (`--kv q4_0`) | GFX12 WMMA (`trim=2`) | GFX11 WMMA (`STRATA_PA_TRIM=0` default) | GFX11 WMMA (`STRATA_PA_TRIM=0` default) |
| GDN Recurrence | GFX12 Chunked / Fused | `STRATA_GDN_CHUNKED=1` | `STRATA_GDN_CHUNKED=1` |
| Dense FP16 Weight Cache | Off (`0`) | Off (`0`, keeps VRAM for `--kv-grow`) | Auto **16384 MiB** (`STRATA_PF_W16_CACHE_MIB=16384`, reserves 8 GiB free) |
| KV / Expert Memory Policy | `--kv q4_0 --kv-grow --kv-resident 4096` | `--kv q4_0 --kv-grow` | `--kv q4_0 --kv-grow` |

### Tuning Knobs

1. **Discrete GPUs (RDNA4 & RDNA3 dGPU) — `--kv-grow` Strategy**:
   - Keep `--kv q4_0 --kv-grow` (and `--kv-resident 4096` or `--kv-resident 20480` on 16–32 GB cards) so KV allocations grow on demand and maximize VRAM available to `--expert-cache auto`.
   - Leave `STRATA_PF_W16_CACHE_MIB` unset (`0`) on discrete GPUs so the dense FP16 weight cache does not compete with KV or routed experts.
   - RDNA4 (`gfx12*`) keeps its `STRATA_GFX12_DEFAULTS` table active (`STRATA_GFX12_DEFAULTS=0` disables).

2. **Integrated / UMA GPUs (e.g., Strix Halo `gfx1151`, 128 GB unified memory)**:
   - Recommended flags: `--kv q4_0 --prefill auto:16384 --spec 4 --spec-min-p 0.5 --short-read 64 --mtp-q4 head --mtp-draft-vocab <path>/draft_vocab.bin --kv-grow`.
   - `STRATA_PF_W16_CACHE_MIB=16384` is auto-enabled when `integrated == true` (`gemm.cu` enforces an 8 GiB free-VRAM reserve). Override with `STRATA_PF_W16_CACHE_MIB=0` to disable or `=N` to set a custom MiB cap.
   - Keep `STRATA_WMMA_GEMM=0` and `STRATA_PF_HCDOWN=1`.
   - Disable all GFX11 auto-defaults with `STRATA_GFX11_DEFAULTS=0`.

## Quick Start

```bash
# 1. Model (3 shards)
hf download tsaipifong/Swift-1.5-Qwen3.8-Flash-Next-MXFP4-GGUF \
  Swift-1.5-Qwen3.8-Flash-Next-MXFP4-ple-IQ4_NL-00001-of-00003.gguf \
  Swift-1.5-Qwen3.8-Flash-Next-MXFP4-ple-IQ4_NL-00002-of-00003.gguf \
  Swift-1.5-Qwen3.8-Flash-Next-MXFP4-ple-IQ4_NL-00003-of-00003.gguf \
  --local-dir ~/Strata-data/models/tsaipifong-mxfp4

# 2. Pack (dense.bin, index.txt, native_experts.txt, tokenizer)
.venv/bin/python tools/iq_pack.py \
  --gguf ~/Strata-data/models/tsaipifong-mxfp4/Swift-1.5-Qwen3.8-Flash-Next-MXFP4-ple-IQ4_NL-00001-of-00003.gguf \
  --out  ~/Strata-data/packs/tsaipifong-mxfp4 --compat-bf16

# 3. Drop the in-file MTP block (blk.48) from the native expert table and index
python3 tools/mxfp4_rdna_fix_pack.py --pack ~/Strata-data/packs/tsaipifong-mxfp4 --layers 48

# 4. Build (HIP)
cmake -S . -B build-hip -G Ninja -DSTRATA_ENABLE_HIP=ON
cmake --build build-hip --target strata -j8

# 5. Run with architecture-tailored config
# For RDNA4:
./serve-mxfp4-rdna.sh
# For RDNA3 dGPU (e.g. RX 7800 XT):
./build-hip/strata --config configs/mxfp4-rdna-7800xt.json
```

## How It Works

- **Weights**: MXFP4 (E2M1 + one E8M0 scale per 32 values) -> E4M3 with the
  scale folded in. The value is a power of two times a 4-bit significand, so
  the E4M3 encoding is exact in the normal range; it is saturated only at the
  E4M3 limits. On RDNA4 the default (`STRATA_MXFP4_FOLDED=1`) converts the
  nibbles to E4M3 in the GEMM's W staging with a `v_perm` LUT and applies the
  per-row max exponent (`Wref`) once in the epilogue, so there is no separate
  dequant pass and no FP8 weight copy; because E8M0 scales are powers of two
  the result is bit-identical to the pre-dequant path. RDNA3 uses the FP16
  dequant instead (see GPU Support).
- **Activations**: dynamic per-row E4M3 quantization (`max_abs / 448`), with the
  scale applied back in the GEMM epilogue.
- **GEMM**: gfx12 `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`,
  FP32 accumulators. A 128-thread / 4-wave 64x64 block stages A and B in LDS
  once per 16-wide k step and shares them across the waves. The micro-benchmark reaches
  36 TFLOPS for the up projection and 26 TFLOPS for the down projection.
- **In-file MTP**: the tsaipifong GGUF carries `qwen4exp.nextn_predict_layers=1`
  and block 48. The engine reads `block_count - nextn_predict_layers` as the
  trunk (48), skips `blk.48` in the native dense path, and the pack fix removes
  the block from `native_experts.txt` / `index.txt`.
- **Scheduling**: `--kv-grow` (instead of a pinned `--kv-resident`) lets the
  expert cache own the VRAM; the dense weights stay on the GPU.

## Speculative Decoding & Expert Caching

`configs/mxfp4-rdna.json` and `configs/mxfp4-rdna-mtp.json` tune prefill and speculative decode:
- **Speculative MTP**: `--spec 4 --spec-min-p 0.5` achieves 80-90% acceptance on structured code, json, and reasoning.
- **Expert profiling**: The adaptive tier swaps active experts into VRAM. `--expert-profile-save` persists converged sets across restarts.

## Limitations

- The 2x2 / 4x4 / LDS kernels are tuned for these expert shapes (N=1280/K=2560 up, N=2560/K=640 down); other architectures may need new thresholds.
- The pack fix assumes the model's trunk is 48 layers (the tsaipifong Swift-1.5 Qwen3.8-Flash-Next); pass `--layers` for another block count.

## Upstream

- Fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).
- MXFP4 routed experts and MXFP4 PLE kernel integration.

## Agent & Tool Benchmarking: Twin Suites

This repository provides two complementary tool/agent benchmark suites for validating OpenAI-compatible endpoints under long context and functional tool calling:

1. **Tools Ladder (`tools/tools_ladder.py` / `tools_ladder.py`)**:
   - Directly modeled after MBufo's `tool_ladder.py`: evaluates tool-calling competence and schema compliance as context length scales along a geometric ladder (e.g. `0 -> 500 -> 2,000 -> 8,000 -> 16,000 -> 32,000 -> 64,000 -> 128,000` tokens).
   - Verifies tool dispatch accuracy, detects failure modes (`no-call` vs `text-json` markdown hallucination vs schema mismatch), and reports throughput and pass rates broken down by length rung and task.
   - Run:
     ```bash
     python tools/tools_ladder.py --url http://127.0.0.1:8081/v1 --model <id> --key strata67
     ```

2. **8-Task ReAct Agent Suite (`tools/tools_react.py` / `tools_react.py`)**:
   - Evaluates tool selection, schema validation, and decoding throughput across 8 functional domains: `weather`, `email`, `search`, `calc`, `translate`, `read_file`, `run_command`, `summarize`.
   - Runs deterministic zero-temperature calls and verifies both the tool target and structured JSON arguments.
   - Run:
     ```bash
     python tools/tools_react.py --base-url http://127.0.0.1:8081/v1 --model <id> --key strata67
     ```

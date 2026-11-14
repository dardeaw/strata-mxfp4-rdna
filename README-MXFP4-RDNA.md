# MXFP4 for RDNA4 (gfx12) — Strata fork

This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) that runs
**MXFP4 experts** (E2M1 weights with E8M0 scales, 32-value blocks) on AMD GPUs.
On **RDNA4 / gfx12** it uses **FP8 (E4M3) compute** (vLLM's Quark W4A8 model);
on **RDNA3 / gfx11xx** — and any card without FP8 WMMA — it automatically falls
back to **FP16 compute**. It also ships a `--kv-grow` configuration that keeps
the dense weights on the GPU and gives the rest of the VRAM to the expert cache.

> This fork is about **Strata on RDNA4 (and on RDNA3 through the FP16 path)**.
> It is not related to gufo, MBufo, or halogen, and nothing here is pushed to
> those repositories.

## Status

- GPU: AMD Radeon RX 9070 (gfx1201, RDNA4, 16 GB), TheRock HIP / ROCm SDK 7.x.
- Model: `tsaipifong/Swift-1.5-Qwen3.8-Flash-Next-MXFP4-GGUF` (`ple-IQ4_NL`,
  3 shards, 99.75 GB).
- Upstream base: Strata 0.2.10 + PR #3575 (MXFP4 routed experts and MXFP4 PLE).
- Default: `configs/mxfp4-rdna.json` (grow config) + `STRATA_MXFP4_FP8=1`.
- The MXFP4 experts are dequantized to E4M3 with their E8M0 block scale folded
  in (exact: a power of two times at most four significant bits), the
  activations are quantized per row to E4M3, and the products run on gfx12 FP8
  WMMA.

## GPU support: RDNA4 vs RDNA3

The MXFP4 route is the same on both families (E2M1 weights + E8M0 scales); only
the compute precision differs. The choice is made at startup from the card's
`gcnArchName`, never from the env var alone, so a wrong `STRATA_MXFP4_FP8` can
never select a kernel the GPU cannot run.

| | RDNA4 (gfx12: RX 9070 / 9070 XT, gfx1200/1201/1250) | RDNA3 (gfx11xx: RX 7900 XTX, 7800, 7700, ...) |
|---|---|---|
| Weights | MXFP4 -> E4M3 inside the GEMM (folded kernel: reads the 17-byte blocks straight, no dequant pass) | MXFP4 -> **FP16** (`iq_dequant_*_f16`, llama.cpp formulas) |
| Compute | **FP8 E4M3 WMMA** (`wmma_f32_16x16x16_fp8_fp8_w32_gfx12`), FP32 accumulate | **FP16 matrix cores**, FP32 accumulate |
| Activations | dynamic per-row E4M3 quantize | FP16 (no quantize) |
| Selected by | `mx_fp8_on()` == gfx12 and `STRATA_MXFP4_FP8=1` | anything else, or `STRATA_MXFP4_FP8` unset / `0` |
| Speed | the reference setting; prefill ~1.9-2.1K tok/s at 32K-250K on the RX 9070 | slower per token (FP16 weights are 2x the bytes of E4M3); not benchmarked on RDNA3 hardware here |

**Why RDNA3 has no FP8 path:** `gfx11xx` has no FP8 (E4M3/E5M2) matrix-core
instructions. The kernels build `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`
for `gfx1201`; on an RDNA3 card they cannot launch. Instead of failing, the
engine reads `gcnArchName`, and if it is not `gfx12*` it prints one warning and
runs the FP16 expert GEMMs it has always had:

```
prefill: STRATA_MXFP4_FP8 is set but this GPU has no FP8 WMMA
(gfx12/RDNA4 required, got gfx1100) -- falling back to the FP16 expert path
```

Overrides:

- `STRATA_MXFP4_FP8=0` (or unset): use the FP16 path everywhere.
- `STRATA_FP8_WMMA=0`: force the FP16 path even on a gfx12 card (A/B testing, or
  a driver problem with the FP8 kernels).
- On RDNA4 the FP8/folded fast path is the default; nothing needs setting.

Validation: no RDNA3 GPU was on hand to measure, so the FP16 fallback was
exercised on a gfx1201 card by forcing `STRATA_FP8_WMMA=0` — an 8K prompt
prefills at ~1.44K tok/s and generates cleanly, the same code path the automatic
gate selects on RDNA3.

## Results

Measured with `bench_protocol.py` on AMD Radeon RX 9070 (16 GB, gfx1201, Nobara Linux), running `--kv q4_0` (optimized WMMA matrix-core S6 kernel) with in-file MTP on (`--spec 4`, resident budget 55 GiB, 13 CPU pool workers):

### Prefill Ladder (Cold context, unique nonce, max_tokens=1)

| Target Tokens | Actual prompt_n | **Q4_0 (Optimized WMMA)** | Previous int8-fused | Previous folded-FP8 | Speedup vs folded-FP8 | Wall Time |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 2,049 | **922.8 tok/s** | 942.2 | 632.3 | +46% | 2.2 s |
| 4,096 | 4,057 | **1,273.6 tok/s** | 1,277.0 | 1,028.8 | +24% | 3.2 s |
| 8,192 | 8,233 | **1,940.0 tok/s** | 1,899.2 | 1,489.4 | +30% | 4.3 s |
| 16,384 | 16,302 | **2,383.6 tok/s** | 2,435.6 | 1,762.1 | +35% | 6.9 s |
| 32,768 | 32,552 | **2,523.2 tok/s** | 2,477.1 | 1,937.6 | +30% | 12.9 s |
| 65,536 | 65,210 | **2,469.5 tok/s** | 2,563.4 | 2,035.7 | +21% | 26.5 s |
| 131,072 | 130,345 | **2,529.6 tok/s** | 2,524.8 | 2,073.7 | +22% | 51.6 s |
| 250,000 | 248,085 | **2,363.5 tok/s** | 2,367.2 | 2,004.4 | +18% | 105.1 s |

### Decode Scenarios (Cold KV, MTP spec=4)

| Scenario | Temperature | max_tokens | **Q4_0 tok/s** | MTP Acceptance | int8-fused | folded-FP8 |
|---|---:|---:|---:|---:|---:|---:|
| prose | 0.4 | 256 | **39.7** | 53.8% | 38.4 | 38.6 |
| fib | 0.4 | 256 | **57.8** | 89.4% | 57.3 | 43.8 |
| json | 0.0 | 128 | **50.2** | 83.0% | 49.0 | 41.4 |
| code | 0.0 | 128 | **63.6** | 81.7% | 67.6 | 44.1 |
| repeat | 0.0 | 256 | **61.1** | 87.5% | 49.7 | 48.2 |

### 8-Task ReAct Agent Suite
- Tool selection accuracy: **8/8 (100%)**
- Median decode speed: **63.4 tok/s** (range 54.1 – 74.2 tok/s)
- Long KV tools ladder: All 6 rungs PASS with prefix caching ~0.6 s

## Quick start

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

# 3. Drop the in-file MTP block (blk.48) from the native expert table and index.
python3 tools/mxfp4_rdna_fix_pack.py --pack ~/Strata-data/packs/tsaipifong-mxfp4 --layers 48

# 4. Build (HIP)
cmake -S . -B build-hip -G Ninja -DSTRATA_ENABLE_HIP=ON
cmake --build build-hip --target strata -j8
# copy build-hip/strata to engine/strata (or point "exe" at it)

# 5. Run the default grow config (FP8 compute)
./serve-mxfp4-rdna.sh
```

Edit `configs/mxfp4-rdna.json` for your model/pack/tokenizer paths before
running. The launcher sets `STRATA_MXFP4_FP8=1` and the config uses
`--kv-grow`, `--expert-cache auto`, and `--spec 4`.

## How it works

- **Weights**: MXFP4 (E2M1 + one E8M0 scale per 32 values) -> E4M3 with the
  scale folded in. The value is a power of two times a 4-bit significand, so
  the E4M3 encoding is exact in the normal range; it is saturated only at the
  E4M3 limits. On RDNA4 the default (`STRATA_MXFP4_FOLDED=1`) converts the
  nibbles to E4M3 in the GEMM's W staging with a `v_perm` LUT and applies the
  per-row max exponent (`Wref`) once in the epilogue, so there is no separate
  dequant pass and no FP8 weight copy; because E8M0 scales are powers of two
  the result is bit-identical to the pre-dequant path. RDNA3 uses the FP16
  dequant instead (see GPU support).
- **Activations**: dynamic per-row E4M3 quantization (`max_abs / 448`), with the
  scale applied back in the GEMM epilogue.
- **GEMM**: gfx12 `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`,
  FP32 accumulators. A 128-thread / 4-wave 64x64 block stages A and B in LDS
  once per 16-wide k step and shares them across the waves (up from T >= 256,
  down from T >= 128; 2x2 for smaller tiles). The micro-benchmark reaches
  36 TFLOPS for the up projection and 26 TFLOPS for the down projection.
- **In-file MTP**: the tsaipifong GGUF carries `qwen4exp.nextn_predict_layers=1`
  and block 48. The engine reads `block_count - nextn_predict_layers` as the
  trunk (48), skips `blk.48` in the native dense path, and the pack fix removes
  the block from `native_experts.txt` / `index.txt`.
- **Scheduling**: `--kv-grow` (instead of a pinned `--kv-resident`) lets the
  expert cache own the VRAM; the dense weights stay on the GPU.

## Limitations

- FP8 wins the prefill by a small margin (1.00-1.02x); the decode is mixed
  (json +7%, short/code -3%). The FP8 path adds per-expert dequant/quantize
  launches, and the end-to-end is partly CPU/launch bound.
- `--spec 4` without MTP: the tsaipifong GGUF's in-file MTP block is not used for
  speculative decoding, so decode numbers are non-speculative.
- The 2x2 / 4x4 / LDS kernels are tuned for these expert shapes (N=1280/K=2560
  up, N=2560/K=640 down); other architectures may need new thresholds.
- The pack fix assumes the model's trunk is 48 layers (the tsaipifong Swift-1.5
  Qwen3.8-Flash-Next); pass `--layers` for another block count.

## Upstream

- Fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).
- Upstream base: 0.2.10 + PR #3575.
- Unrelated to gufo, MBufo, and halogen.

## Decode (MTP, 2026-11-06)

`configs/mxfp4-rdna-mtp.json` (launcher: `./serve-mxfp4-rdna-mtp.sh`) trades a
little prefill scheduling freedom for decode: `--spec 2 --spec-min-p 0.5` (small
window, high gate) plus `--mtp-hnorm stream` lift the MTP acceptance from 14.5%
to 72.6% on prose and 25.1% to 83.0% on code.

The second lever is the expert cache, and it is **not** the ranking in
`--expert-profile`: the adaptive tier swaps the truly hot experts into VRAM
anyway. `--expert-profile-save` persists what it learned (every 10 min and on
QUIT), so the next cold start begins at the converged set. The profile shipped
in `data/expert-profile.bin` is such a saved profile.

Cold-start decode (first request after a restart, greedy, 200 tokens), measured
with `tools/strata_bench.py`'s prompt set on the RX 9070:

| variant | code tok/s | prose tok/s |
|---|---:|---:|
| shipped / freshly built profile | 28.9 | 34.7 |
| trace-ranked "code-heavy" profile | 37.2 | 35.2 |
| **learned profile (`--expert-profile-save`)** | **39.5** | **36.6** |

Warm requests reach 41-44 tok/s on code. Two notes from the same sweep:
`--expert-cache-per-layer` changes nothing when a profile already spreads the
slots across layers, and `--mtp-q4 all` costs 6 points of acceptance without
saving VRAM. `--pool-workers 20` (20 cores: 8 P + 12 E on the Core Ultra 7 265K)
drops decode by 35% - keep 13.

The decode-side GPU hit rate is a misleading target: the learned profile runs at
56% hits and is faster than a 67%-hit trace-ranked profile, because misses drain
to the CPU pool and overlap with the serialized GPU hit path.


## RDNA3 / RDNA3.5 (gfx11xx) Auto-Detection & Special Configuration

Starting in **v0.1.59**, the engine detects AMD GPU families by `gfx` prefix (`gfx11*` for RDNA3/RDNA3.5, `gfx12*` for RDNA4) and whether the GPU is integrated (UMA) or discrete, applying architecture-specific defaults automatically (`src/core/arch_defaults.cpp`) while preserving any explicit environment variable override.

### Automatic Strategy by GPU Type

| Setting | RDNA4 dGPU (`gfx12*`, e.g. R9700 / RX 9070) | RDNA3 dGPU (`gfx11*`, e.g. RX 7900 XTX / 7800 XT) | RDNA3.5 / RDNA3 iGPU (`gfx11*` UMA, e.g. Strix Halo) |
|---|---|---|---|
| MoE Expert Path | FP8 E4M3 WMMA (`STRATA_MXFP4_FP8=1`) | **F16 Matrix-Core Fused** (`STRATA_PF_F16=1` + `STRATA_PF_FUSED_MXFP4=1`) | **F16 Matrix-Core Fused** (`STRATA_PF_F16=1` + `STRATA_PF_FUSED_MXFP4=1`) |
| Prompt Attention (`--kv q4_0`) | GFX12 WMMA (`trim=2`) | GFX11 WMMA (`STRATA_PA_TRIM=0` default) | GFX11 WMMA (`STRATA_PA_TRIM=0` default) |
| GDN Recurrence | GFX12 Chunked / Fused | `STRATA_GDN_CHUNKED=1` | `STRATA_GDN_CHUNKED=1` |
| Dense FP16 Weight Cache | Off (`0`) | Off (`0`, keeps VRAM for `--kv-grow`) | Auto **16384 MiB** (`STRATA_PF_W16_CACHE_MIB=16384`, reserves 8 GiB free) |
| KV / Expert Memory Policy | `--kv q4_0 --kv-grow --kv-resident 4096` | `--kv q4_0 --kv-grow` | `--kv q4_0 --kv-grow` |

### Special Configuration & Tuning Knobs

1. **Discrete GPUs (RDNA4 & RDNA3 dGPU) — `--kv-grow` Strategy**:
   - Keep `--kv q4_0 --kv-grow` (and `--kv-resident 4096` on 16–32 GB cards) so KV allocations grow on demand and maximize VRAM available to `--expert-cache auto`.
   - Leave `STRATA_PF_W16_CACHE_MIB` unset (`0`) on discrete GPUs so the dense FP16 weight cache does not compete with KV or routed experts.
   - RDNA4 (`gfx12*`) keeps its `STRATA_GFX12_DEFAULTS` table untouched (`STRATA_GFX12_DEFAULTS=0` disables).

2. **Integrated / UMA GPUs (e.g., Strix Halo `gfx1151`, 128 GB unified memory)**:
   - Recommended flags: `--kv q4_0 --prefill auto:16384 --spec 4 --spec-min-p 0.5 --short-read 64 --mtp-q4 head --mtp-draft-vocab <path>/draft_vocab.bin --kv-grow`.
   - `STRATA_PF_W16_CACHE_MIB=16384` is auto-enabled when `integrated == true` (`gemm.cu` enforces an 8 GiB free-VRAM reserve). Override with `STRATA_PF_W16_CACHE_MIB=0` to disable or `=N` to set a custom MiB cap.
   - Keep `STRATA_WMMA_GEMM=0` and `STRATA_PF_HCDOWN=1` (setting `STRATA_WMMA_GEMM=1` or `STRATA_PF_HCDOWN=0` drops 16K prefill from ~1,240 tok/s to ~950 tok/s).
   - Disable all GFX11 auto-defaults with `STRATA_GFX11_DEFAULTS=0`.

### RDNA3.5 Results (Strix Halo Radeon 8060S, `gfx1151`, DPM `auto`, `--kv q4_0`)

| Prompt Size | Tokens | Prefill (`tok/s`) | Wall Time |
|---:|---:|---:|---:|
| 2K | 1,861 | **1,079.6** | 1.76 s |
| 4K | 3,666 | **1,166.5** | 3.18 s |
| 8K | 7,286 | **1,219.6** | 6.02 s |
| 16K | 14,927 | **1,237.9** | 12.10 s |
| 32K | 29,795 | **1,222.7** | 24.42 s |
| 65K | 59,746 | **1,218.3** | 49.10 s |
| 131K | 119,643 | **1,182.9** | 101.23 s |

Decode medians (`--spec 4 --short-read 64 --mtp-q4 head`): `short 37.50`, `prose 37.80`, `code 59.70`, `json 62.40`, `repeat 79.20`, `fib 63.30 tok/s`.


## Agent & Tool Benchmarking: Twin Suites ("雙壁")

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

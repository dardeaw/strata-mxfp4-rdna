# MXFP4 for RDNA4 (gfx12) — Strata fork

This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) that runs
**MXFP4 experts** (E2M1 weights with E8M0 scales, 32-value blocks) on AMD
**RDNA4 / gfx12** with **FP8 (E4M3) compute**, following vLLM's Quark W4A8 model.
It also ships a `--kv-grow` configuration that keeps the dense weights on the GPU
and gives the rest of the VRAM to the expert cache.

> This fork is about **Strata and RDNA4 only**. It is not related to gufo, MBufo,
> or halogen, and nothing here is pushed to those repositories.

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

## Results

Measured with `bench_strata.py` (3 decode reps, prefill 2048..245000), the same
model and config, `--spec 4` without MTP. Prefill in tok/s:

| target | Q4_K_XL | MXFP4-FP16 | MXFP4-FP8 | FP8/FP16 | FP8/Q4 |
|---:|---:|---:|---:|---:|---:|
| 2048 | 239.5 | 265.2 | 267.4 | 1.01x | 1.12x |
| 4096 | 435.8 | 454.5 | 464.9 | 1.02x | 1.07x |
| 8192 | 629.5 | 783.0 | 789.7 | 1.01x | 1.25x |
| 16384 | 714.0 | 1192.8 | 1198.5 | 1.00x | 1.68x |
| 32768 | 822.7 | 1296.9 | 1298.7 | 1.00x | 1.58x |
| 65536 | 983.3 | 1350.3 | 1363.8 | 1.01x | 1.39x |
| 131072 | 996.3 | 1344.6 | 1349.1 | 1.00x | 1.35x |
| 245000 | 621.4 | 1369.8 | 1382.0 | 1.01x | 2.22x |

Decode (median tok/s):

| scenario | Q4_K_XL | MXFP4-FP16 | MXFP4-FP8 | FP8/FP16 |
|---|---:|---:|---:|---:|
| short | 18.6 | 24.0 | 23.4 | 0.97x |
| prose | 23.4 | 32.4 | 32.1 | 0.99x |
| code | 24.2 | 31.7 | 30.7 | 0.97x |
| json | 37.2 | 27.3 | 29.2 | 1.07x |

`--kv-grow` keeps the GPU at 100% and the CPU near idle (the old
`--kv-resident 32768` pinning left the CPU at 64.6% and the GPU at 79%). The
expert cache gets the VRAM the KV does not need.

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
  E4M3 limits.
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

# Changelog

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

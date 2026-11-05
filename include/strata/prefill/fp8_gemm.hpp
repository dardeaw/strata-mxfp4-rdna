#pragma once

#include <cstdint>

namespace strata::prefill {

/// Y[T, N] = X[T, K] . W[N, K]^T on gfx12's FP8 (E4M3) WMMA, FP32 accumulators; `row_scale[t]` (the activation
/// quantization scale) multiplies row t of the result.  Both operands are E4M3 bytes, X row-major [T, K], W row-major
/// [N, K].  false: not launched (this build/device has no FP8 WMMA, or the shape is not a multiple of the 32x32 tile).
/// vLLM's Quark W4A8 path (QuarkW4A8Fp8MoEMethod / AITER moe_gemm_a8w4) computes the same products in FP8.
bool fp8_gemm(const uint8_t* X, const uint8_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              const float* row_scale, void* stream);

/// One tile of a grouped FP8 GEMM: a 64-wide N slice (n_tile) of one expert's rows
/// [t_start, t_start + t_count), M group m_tile (256 rows, up to 4 M-tiles per block).
/// w points at that expert's pre-dequantized FP8 matrix (row-major N x K); out at its output rows.
struct Fp8GroupTile {
    const uint8_t* w;
    float* out;
    int64_t t_start;
    int64_t t_count;
    int64_t n_tile;
    int64_t m_group;
    int64_t N;
    int64_t K;
    int64_t ldy;
};

/// Grouped FP8 GEMM over pre-dequantized weights: one launch for a whole layer's tiles.
/// X is the chunk's pre-quantized FP8 activations, row_scale its per-row scales.
bool fp8_grouped_gemm(const uint8_t* X, const float* row_scale, const Fp8GroupTile* tiles, int num_tiles,
                       void* stream);

}  // namespace strata::prefill

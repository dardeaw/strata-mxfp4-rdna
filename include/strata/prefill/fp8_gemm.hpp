#pragma once

#include <cstdint>

namespace strata::prefill {

/// Y[T, N] = X[T, K] . W[N, K]^T on gfx12's FP8 (E4M3) WMMA, FP32 accumulators; `row_scale[t]` (the activation
/// quantization scale) multiplies row t of the result.  Both operands are E4M3 bytes, X row-major [T, K], W row-major
/// [N, K].  false: not launched (this build/device has no FP8 WMMA, or the shape is not a multiple of the 32x32 tile).
/// vLLM's Quark W4A8 path (QuarkW4A8Fp8MoEMethod / AITER moe_gemm_a8w4) computes the same products in FP8.
bool fp8_gemm(const uint8_t* X, const uint8_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              const float* row_scale, void* stream);

}  // namespace strata::prefill

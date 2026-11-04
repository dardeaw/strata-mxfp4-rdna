// src/prefill/fp8_gemm.cu - E4M3 x E4M3 -> FP32 WMMA for the MXFP4 expert prompt path (STRATA_MXFP4_FP8=1).
//
// The MXFP4 weights are dequantized to E4M3 with their E8M0 block scale folded in (see iq_dequant_mxfp4_fp8), the
// activations quantized per row to E4M3 (quantize_fp8_rows), and the products run on gfx12's FP8 matrix cores.  This
// is the same compute model as vLLM's Quark W4A8 (QuarkW4A8Fp8MoEMethod: MXFP4 weights -> FP8, dynamic per-token FP8
// activations, FP8 GEMM), with the block scale folded into the weight instead of applied in the kernel.
//
// One wave (32 lanes) computes a 32x32 output tile (2x2 WMMA tiles) so each A/B fragment is used twice.  The gfx12
// 16x16x16 FP8 fragment layout (checked on this card): lane l holds A row l & 15 / B column l & 15 at
// k = 8 * (l >> 4) + 0..7, and the accumulator's lane l holds column l & 15, rows 8 * (l >> 4) + 0..7.
#include "strata/prefill/fp8_gemm.hpp"

#include <cuda_runtime.h>

#include <cstring>

// The host pass of a HIP build never sees __gfx1201__ (only the device passes do), so the switch is HIP-wide and
// wmma() below picks the matrix-core builtin in the gfx12 device passes; fp8_gemm() gates on the device's arch name
// at run time, so a gfx11 build compiles the same code and never launches the kernel.
#if defined(__HIPCC__)
#define STRATA_FP8_WMMA 1
#else
#define STRATA_FP8_WMMA 0
#endif

#if STRATA_FP8_WMMA
namespace {

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((vector_size(8)));

__device__ __forceinline__ v2i ld8(const uint8_t* p) { return *reinterpret_cast<const v2i*>(p); }
__device__ __forceinline__ v8f wmma(v2i a, v2i b, v8f c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
#else
    (void) a; (void) b;
    return c;   // no FP8 matrix core in this device pass; fp8_gemm()'s arch gate never launches it
#endif
}

__global__ void fp8_gemm_kernel(const uint8_t* __restrict__ X, const uint8_t* __restrict__ W,
                                float* __restrict__ Y, int T, int N, int K, int ldy,
                                const float* __restrict__ rs) {
    const int n0 = blockIdx.x * 32, t0 = blockIdx.y * 32;
    const int l = threadIdx.x;
    const int arow = t0 + (l & 15), brow = n0 + (l & 15);
    const int k8 = 8 * (l >> 4);
    v8f c00 = {0, 0, 0, 0, 0, 0, 0, 0}, c01 = c00, c10 = c00, c11 = c00;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int kk = k0 + k8;
        v2i a0 = {0, 0}, a1 = {0, 0}, b0 = {0, 0}, b1 = {0, 0};
        if (arow < T) a0 = ld8(X + (size_t) arow * K + kk);
        if (arow + 16 < T) a1 = ld8(X + (size_t) (arow + 16) * K + kk);
        if (brow < N) b0 = ld8(W + (size_t) brow * K + kk);
        if (brow + 16 < N) b1 = ld8(W + (size_t) (brow + 16) * K + kk);
        c00 = wmma(a0, b0, c00);
        c01 = wmma(a0, b1, c01);
        c10 = wmma(a1, b0, c10);
        c11 = wmma(a1, b1, c11);
    }
    const int col = n0 + (l & 15);
    for (int i = 0; i < 8; ++i) {
        const int row = t0 + 8 * (l >> 4) + i;
        if (row >= T) break;
        const float s0 = rs[row];
        const bool r1 = row + 16 < T;
        const float s1 = r1 ? rs[row + 16] : 0.0f;   // the second 16-row tile has its own row scale
        if (col < N) {
            Y[(size_t) row * ldy + col] = c00[i] * s0;
            if (r1) Y[(size_t) (row + 16) * ldy + col] = c10[i] * s1;
        }
        if (col + 16 < N) {
            Y[(size_t) row * ldy + col + 16] = c01[i] * s0;
            if (r1) Y[(size_t) (row + 16) * ldy + col + 16] = c11[i] * s1;
        }
    }
}

}  // namespace
#endif

namespace strata::prefill {

bool fp8_gemm(const uint8_t* X, const uint8_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
              const float* row_scale, void* stream) {
#if STRATA_FP8_WMMA
    if (T <= 0) return true;
    if (N <= 0 || K <= 0 || (N % 32) != 0 || (K % 16) != 0 || ldy < N) return false;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return false;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    if (std::strncmp(prop.gcnArchName, "gfx120", 6) != 0) return false;   // FP8 WMMA is a gfx12 (RDNA4) instruction
    const dim3 grid((unsigned) (N / 32), (unsigned) ((T + 31) / 32));
    fp8_gemm_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(X, W, Y, (int) T, (int) N, (int) K, (int) ldy, row_scale);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) row_scale; (void) stream;
    return false;
#endif
}

}  // namespace strata::prefill

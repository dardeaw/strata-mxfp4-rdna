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


// 64x64 output tile per wave (4x4 16x16 tiles), for T >= 128: A and B are each read half as often as by the 32x32
// kernel (A once per 64-column strip, B once per 64-row strip), which is what the 26K..245K prompt chunks need.
__global__ void fp8_gemm_kernel_4x4(const uint8_t* __restrict__ X, const uint8_t* __restrict__ W,
                                    float* __restrict__ Y, int T, int N, int K, int ldy,
                                    const float* __restrict__ rs) {
    const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
    const int l = threadIdx.x;
    const int arow = t0 + (l & 15), brow = n0 + (l & 15);
    const int k8 = 8 * (l >> 4);
    const v2i z2 = {0, 0};
    const v8f z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8f c[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) c[i][j] = z8;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int kk = k0 + k8;
        v2i a[4], b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = arow + 16 * i;
            a[i] = (r < T) ? ld8(X + (size_t) r * K + kk) : z2;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int cidx = brow + 16 * j;
            b[j] = (cidx < N) ? ld8(W + (size_t) cidx * K + kk) : z2;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) c[i][j] = wmma(a[i], b[j], c[i][j]);
    }
    const int col0 = n0 + (l & 15);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = t0 + 16 * i + 8 * (l >> 4);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = col0 + 16 * j;
            if (col >= N) continue;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int rr = row + r;
                if (rr < T) Y[(size_t) rr * ldy + col] = c[i][j][r] * rs[rr];
            }
        }
    }
}


// 64x64 output block, 128 threads = 4 waves, A/B staged in LDS once per 16-wide k step and shared by the waves.
// Each wave computes a 32x32 tile (its own 2x2 WMMA); the block thus reads A once per 64-column strip and B once
// per 64-row strip, half the global traffic of the 32x32 kernel, with 4 accumulators per lane instead of 16.
__global__ void fp8_gemm_kernel_lds(const uint8_t* __restrict__ X, const uint8_t* __restrict__ W,
                                    float* __restrict__ Y, int T, int N, int K, int ldy,
                                    const float* __restrict__ rs) {
    __shared__ __attribute__((aligned(16))) uint8_t As[64 * 16];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[64 * 16];
    const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int msub = wave & 1, nsub = wave >> 1;
    const int arow = 32 * msub + (lane & 15), brow = 32 * nsub + (lane & 15);
    const int k8 = 8 * (lane >> 4);
    const v2i z2 = {0, 0};
    v8f c00 = {0, 0, 0, 0, 0, 0, 0, 0}, c01 = c00, c10 = c00, c11 = c00;
    const int lr = tid >> 1, lk = (tid & 1) * 8;   // each thread stages 8 bytes of A and of B
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int gr = t0 + lr, gc = n0 + lr;
        const v2i av = (gr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
        const v2i bv = (gc < N) ? ld8(W + (size_t) gc * K + k0 + lk) : z2;
        *reinterpret_cast<v2i*>(As + lr * 16 + lk) = av;
        *reinterpret_cast<v2i*>(Bs + lr * 16 + lk) = bv;
        __syncthreads();
        const v2i a0 = *reinterpret_cast<const v2i*>(As + arow * 16 + k8);
        const v2i a1 = *reinterpret_cast<const v2i*>(As + (arow + 16) * 16 + k8);
        const v2i b0 = *reinterpret_cast<const v2i*>(Bs + brow * 16 + k8);
        const v2i b1 = *reinterpret_cast<const v2i*>(Bs + (brow + 16) * 16 + k8);
        c00 = wmma(a0, b0, c00);
        c01 = wmma(a0, b1, c01);
        c10 = wmma(a1, b0, c10);
        c11 = wmma(a1, b1, c11);
        __syncthreads();
    }
    const int m0 = t0 + 32 * msub, n0w = n0 + 32 * nsub;
    const int col = n0w + (lane & 15);
    for (int i = 0; i < 8; ++i) {
        const int row = m0 + 8 * (lane >> 4) + i;
        if (row >= T) break;
        const float s0 = rs[row];
        const bool r1 = row + 16 < T;
        const float s1 = r1 ? rs[row + 16] : 0.0f;
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
    if (T >= 128 && (N % 64) == 0 && (K < 2048 || T >= 256)) {   // LDS: down from T=128, up from T=256
        const dim3 grid((unsigned) (N / 64), (unsigned) ((T + 63) / 64));
        fp8_gemm_kernel_lds<<<grid, 128, 0, (cudaStream_t) stream>>>(X, W, Y, (int) T, (int) N, (int) K, (int) ldy,
                                                                    row_scale);
    } else {
        const dim3 grid((unsigned) (N / 32), (unsigned) ((T + 31) / 32));
        fp8_gemm_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(X, W, Y, (int) T, (int) N, (int) K, (int) ldy, row_scale);
    }
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) row_scale; (void) stream;
    return false;
#endif
}

}  // namespace strata::prefill

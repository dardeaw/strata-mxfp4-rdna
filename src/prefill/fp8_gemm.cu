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

// Grouped FP8 GEMM over PRE-DEQUANTIZED weights (no in-kernel dequant).
// One launch covers a batch of experts: each block owns one (expert, N-tile, M-group).
// B comes from batch dq buffers (coalesced FP8); A is the hoisted chunk x8.
// Same LDS 4-wave 64x64 MT structure as fp8_gemm_kernel_lds (proven), plus cross-expert batching.
__global__ void fp8_gemm_kernel_grouped(const uint8_t* __restrict__ X, const float* __restrict__ rs,
                                        const strata::prefill::Fp8GroupTile* __restrict__ tiles, int num_tiles) {
    if ((int) blockIdx.x >= num_tiles) return;
    const strata::prefill::Fp8GroupTile t = tiles[blockIdx.x];
    const uint8_t* __restrict__ W = t.w;
    float* __restrict__ Y = t.out;
    const int T = (int) t.t_count;
    const int N = (int) t.N, K = (int) t.K;
    const int ldy = (int) t.ldy;
    const int64_t t_start = t.t_start;
    __shared__ __attribute__((aligned(16))) uint8_t As[4][64 * 16];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[64 * 16];
    const int n0 = (int) t.n_tile * 64, tb = (int) t.m_group * 256;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int msub = wave & 1, nsub = wave >> 1;
    const int arow = 32 * msub + (lane & 15), brow = 32 * nsub + (lane & 15);
    const int k8 = 8 * (lane >> 4);
    const v2i z2 = {0, 0};
    const int lr = tid >> 1, lk = (tid & 1) * 8;
    int nm = (T - tb + 63) >> 6;
    if (nm > 4) nm = 4;
    if (nm <= 0) return;
    v8f c[4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j) c[m][j] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = (int) (t_start + tb + 64 * m + lr);
            const v2i av = (m < nm && tb + 64 * m + lr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
            *reinterpret_cast<v2i*>(As[m] + lr * 16 + lk) = av;
        }
        const int n = n0 + lr;
        const v2i bv = (n < N) ? ld8(W + (size_t) n * K + k0 + lk) : z2;
        *reinterpret_cast<v2i*>(Bs + lr * 16 + lk) = bv;
        __syncthreads();
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= nm) continue;
            const v2i a0 = *reinterpret_cast<const v2i*>(As[m] + arow * 16 + k8);
            const v2i a1 = *reinterpret_cast<const v2i*>(As[m] + (arow + 16) * 16 + k8);
            const v2i b0 = *reinterpret_cast<const v2i*>(Bs + brow * 16 + k8);
            const v2i b1 = *reinterpret_cast<const v2i*>(Bs + (brow + 16) * 16 + k8);
            c[m][0] = wmma(a0, b0, c[m][0]);
            c[m][1] = wmma(a0, b1, c[m][1]);
            c[m][2] = wmma(a1, b0, c[m][2]);
            c[m][3] = wmma(a1, b1, c[m][3]);
        }
        __syncthreads();
    }
    const int m0 = tb + 32 * msub, n0w = n0 + 32 * nsub;
    const int col = n0w + (lane & 15);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (m >= nm) continue;
        const int t0 = tb + 64 * m;
        const int mm0 = m0 + 32 * (m >> 1) + 16 * (m & 1) - 32 * msub;
        (void) mm0;
        for (int i = 0; i < 8; ++i) {
            const int row = t0 + 32 * msub + 8 * (lane >> 4) + i;
            if (row >= T) break;
            const float s0 = rs[t_start + row];
            if (col >= N) continue;
            Y[(size_t) row * ldy + col] = c[m][0][i] * s0;
            Y[(size_t) row * ldy + col + 16] = c[m][1][i] * s0;
            if (row + 16 < T) {
                const float s1 = rs[t_start + row + 16];
                Y[(size_t) (row + 16) * ldy + col] = c[m][2][i] * s1;
                Y[(size_t) (row + 16) * ldy + col + 16] = c[m][3][i] * s1;
            }
        }
    }
}

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

// 128x64 FP8 WMMA (lds evolution): 128 threads, 4 waves, each wave 64 rows x 32 cols
// via 4x2 WMMA (8 accumulators). AI=85 vs 64. Same LDS pattern, doubled M per wave;
// A staged in two coalesced rounds (rows 0-63, then 64-127).
__global__ void fp8_gemm_kernel_lds128(const uint8_t* __restrict__ X, const uint8_t* __restrict__ W,
                                       float* __restrict__ Y, int T, int N, int K, int ldy,
                                       const float* __restrict__ rs) {
    __shared__ __attribute__((aligned(16))) uint8_t As[128 * 16];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[64 * 16];
    const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 128;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int wm = (wave >> 1) & 1, wn = wave & 1;
    const int arow = 64 * wm + (lane & 15), brow = 32 * wn + (lane & 15);
    const int k8 = 8 * (lane >> 4);
    const v2i z2 = {0, 0};
    const v8f z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8f c[4][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) c[i][j] = z8;
    const int lr = tid >> 1, lk = (tid & 1) * 8;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int gr0 = t0 + lr;
        const v2i a0 = (gr0 < T) ? ld8(X + (size_t) gr0 * K + k0 + lk) : z2;
        *reinterpret_cast<v2i*>(As + lr * 16 + lk) = a0;
        const int gr1 = t0 + 64 + lr;
        const v2i a1 = (gr1 < T) ? ld8(X + (size_t) gr1 * K + k0 + lk) : z2;
        *reinterpret_cast<v2i*>(As + (64 + lr) * 16 + lk) = a1;
        const int gc = n0 + lr;
        const v2i bv = (gc < N) ? ld8(W + (size_t) gc * K + k0 + lk) : z2;
        *reinterpret_cast<v2i*>(Bs + (lr & 63) * 16 + lk) = bv;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const v2i a = *reinterpret_cast<const v2i*>(As + (arow + 16 * i) * 16 + k8);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const v2i b = *reinterpret_cast<const v2i*>(Bs + (brow + 16 * j) * 16 + k8);
                c[i][j] = wmma(a, b, c[i][j]);
            }
        }
        __syncthreads();
    }
    const int m0 = t0 + 64 * wm, n0w = n0 + 32 * wn;
    const int col = n0w + (lane & 15);
    for (int i = 0; i < 8; ++i) {
        const int row = m0 + 8 * (lane >> 4) + i;
        if (row >= T) break;
        const float s0 = rs[row];
        if (col >= N) continue;
        Y[(size_t) row * ldy + col] = c[0][0][i] * s0;
        Y[(size_t) row * ldy + col + 16] = c[0][1][i] * s0;
        if (row + 16 < T) {
            const float s1 = rs[row + 16];
            Y[(size_t) (row + 16) * ldy + col] = c[1][0][i] * s1;
            Y[(size_t) (row + 16) * ldy + col + 16] = c[1][1][i] * s1;
        }
        if (row + 32 < T) {
            const float s2 = rs[row + 32];
            Y[(size_t) (row + 32) * ldy + col] = c[2][0][i] * s2;
            Y[(size_t) (row + 32) * ldy + col + 16] = c[2][1][i] * s2;
        }
        if (row + 48 < T) {
            const float s3 = rs[row + 48];
            Y[(size_t) (row + 48) * ldy + col] = c[3][0][i] * s3;
            Y[(size_t) (row + 48) * ldy + col + 16] = c[3][1][i] * s3;
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
    static const bool lds128 = [] { const char* e = std::getenv("STRATA_FP8_LDS128"); return e != nullptr && e[0] == '1'; }();   // opt-in only: measured slower (27 vs 34 TFLOPS at T=275)
    if (T >= 128 && (N % 64) == 0 && (K < 2048 || T >= 256) && lds128) {   // LDS128: 128x64 tiles, AI=85
        const dim3 grid((unsigned) (N / 64), (unsigned) ((T + 127) / 128));
        fp8_gemm_kernel_lds128<<<grid, 128, 0, (cudaStream_t) stream>>>(X, W, Y, (int) T, (int) N, (int) K, (int) ldy,
                                                                        row_scale);
    } else if (T >= 128 && (N % 64) == 0 && (K < 2048 || T >= 256)) {   // LDS: down from T=128, up from T=256
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


bool fp8_grouped_gemm(const uint8_t* X, const float* row_scale, const Fp8GroupTile* tiles, int num_tiles,
                      void* stream) {
#if STRATA_FP8_WMMA
    if (num_tiles <= 0) return true;
    if (!X || !row_scale || !tiles) return false;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return false;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    if (std::strncmp(prop.gcnArchName, "gfx120", 6) != 0) return false;
    fp8_gemm_kernel_grouped<<<dim3((unsigned) num_tiles), 128, 0, (cudaStream_t) stream>>>(X, row_scale, tiles, num_tiles);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) row_scale; (void) tiles; (void) num_tiles; (void) stream;
    return false;
#endif
}

}  // namespace strata::prefill

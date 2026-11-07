// src/prefill/mxfp4_folded.cu - the "folded" MXFP4 expert GEMM: a straight row-major-activation
// GEMM that dequantizes the 4-bit weights to E4M3 *inside* the W staging, so no FP8 copy of the
// weights exists and no separate dequant pass runs.  This is the radiance kernel (v_perm 4-way
// LUT + one sched_barrier per k step + wave-uniform SADDR bases + clamp-not-predicate), reading
// the GGUF 17-byte MXFP4 blocks directly.
//
// Numerics: the block scale is folded with the radiance "common exponent" trick -- the nibble is
// converted against kMag[Wref - scale] and the row's reference exponent 2^(Wref-127) is applied
// once in the epilogue.  Wref is the row's max block exponent, so nothing underflows; the price is
// that the magnitude is re-quantized into E4M3 relative to Wref instead of being looked up in the
// 4096-entry (scale, nibble) table, i.e. a sub-percent difference from the pre-dequant FP8 path.
// mxfp4_folded.* is gated behind STRATA_MXFP4_FOLDED and A/B'd against it, never a silent swap.
#include "strata/prefill/mxfp4_gemm.hpp"

#include <cuda_runtime.h>

#include <cstring>

#if defined(__HIPCC__)
#define STRATA_FOLDED_WMMA 1
#else
#define STRATA_FOLDED_WMMA 0
#endif

#if STRATA_FOLDED_WMMA
namespace strata::prefill {
namespace {

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));

// E2M1 magnitude table, as radiance builds it: entry d holds the eight non-negative E4M3 bytes for
// the MXFP4 magnitudes scaled by 2^-d.  Bit 7 is separable, so v_perm_b32 resolves four nibbles per
// instruction and the sign is re-attached with a shift.
__constant__ unsigned int kFoldMag[16][2] = {
  {0x3c383000u, 0x4c484440u}, {0x34302800u, 0x44403c38u}, {0x2c282000u, 0x3c383430u},
  {0x24201800u, 0x34302c28u}, {0x1c181000u, 0x2c282420u}, {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u}, {0x06040200u, 0x14100c08u}, {0x03020100u, 0x0c080604u},
  {0x01010000u, 0x06040302u}, {0x01000000u, 0x03020101u}, {0x00000000u, 0x01010100u},
  {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
};

constexpr int FBK = 64;                    // k slab per loop iteration
constexpr int FPAD = 8;                    // LDS row padding (bank spread)
constexpr int FASTR = FBK + FPAD;
constexpr int FWSTR = FBK + FPAD;
constexpr int FTM = 4;                     // 16-row M tiles per wave
constexpr int FWM = 4;                     // waves along M
constexpr int FWN = 2;                     // waves along N
constexpr int FNWAVE = FWM * FWN;
constexpr int FNTHREADS = FNWAVE * 32;
constexpr int FBMF = FWM * FTM * 16;       // 256 rows per block

// GU: the output rows are the interleaved pair layout (2r gate, 2r+1 up).  DOWN: plain rows.
template <int TN, bool GU>
__global__ __launch_bounds__(FNTHREADS) void folded_kernel(
    const unsigned char* __restrict__ A,      // [M, K] E4M3 row-major activations
    const unsigned char* __restrict__ w0,     // GU: gate blob, DOWN: the down blob
    const unsigned char* __restrict__ w1,     // GU: up blob, DOWN: unused
    const unsigned char* __restrict__ Wref,   // [N] per-row max block exponent
    const float* __restrict__ As,             // [M] row scales
    float* __restrict__ C,                    // [M, N]
    int M, int N, int K, int row_bytes) {
  __shared__ unsigned char sA[FBMF * FASTR];
  __shared__ unsigned char sW[FWN * TN * 16 * FWSTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / FWN, wn = wave % FWN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  constexpr int BNF_T = FWN * TN * 16;
  const int m0 = blockIdx.y * FBMF, n0 = blockIdx.x * BNF_T;

  floatx8 acc[FTM][TN];
#pragma unroll
  for (int i = 0; i < FTM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += FBK) {
    const unsigned char* __restrict__ Ab = A + (size_t) m0 * K + k0;
    const int blk0 = k0 >> 5;

    // ---- A tile: 256 x 64 bytes, 16 B per thread per iteration ----
#pragma unroll
    for (int off = 0; off < FBMF * FBK; off += FNTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / FBK, c = idx % FBK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;      // clamp, never predicate
      *(uint4_t*) (&sA[r * FASTR + c]) = *(const uint4_t*) (Ab + (rc * K + c));
    }

    // ---- W tile: BNF_T rows x 32 packed bytes; one 16-element half block (8 B) per thread ----
#pragma unroll
    for (int off = 0; off < BNF_T * (FBK / 2); off += FNTHREADS * 8) {
      const int idx = off + tid * 8;
      const int r = idx / (FBK / 2), c = idx % (FBK / 2);   // c in [0, 32), c & 15 is the byte offset
      const int rc = r < N - 1 - n0 ? r : N - 1 - n0;       // clamp, never predicate
      const int gn = n0 + rc;
      const unsigned char* rowp = GU ? (((gn & 1) ? w1 : w0) + (size_t) (gn >> 1) * row_bytes)
                                     : (w0 + (size_t) gn * row_bytes);
      const int blk_local = c >> 4;                         // 32 elements per 17-byte block
      const int p = c & 15;                                 // payload byte p = elements p and p+16
      const int blk = blk0 + blk_local;
      const uint2_t pk = *(const uint2_t*) (rowp + (size_t) blk * 17 + 1 + p);
      int dsh = (int) Wref[gn] - (int) rowp[(size_t) blk * 17];
      dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
      const unsigned int t0 = kFoldMag[dsh][0], t1 = kFoldMag[dsh][1];
      unsigned int lo[2], hi[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const unsigned int w = pk[h];
        const unsigned int ev = w & 0x0F0F0F0Fu;            // low nibbles: elements p .. p+7
        const unsigned int od = (w >> 4) & 0x0F0F0F0Fu;     // high nibbles: elements p+16 .. p+23
        lo[h] = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u) | ((ev & 0x08080808u) << 4);
        hi[h] = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u) | ((od & 0x08080808u) << 4);
      }
      const int wcol = blk_local * 32 + p;
      *(uint2_t*) (&sW[r * FWSTR + wcol]) = uint2_t{lo[0], lo[1]};
      *(uint2_t*) (&sW[r * FWSTR + wcol + 16]) = uint2_t{hi[0], hi[1]};
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < FBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[FTM], wf[TN];
#pragma unroll
      for (int i = 0; i < FTM; ++i) {
        const unsigned char* p = &sA[(wm * FTM * 16 + i * 16 + col) * FASTR + kk];
        af[i][0] = *(const int*) p; af[i][1] = *(const int*) (p + 4);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char* p = &sW[(wn * TN * 16 + j * 16 + col) * FWSTR + kk];
        wf[j][0] = *(const int*) p; wf[j][1] = *(const int*) (p + 4);
      }
      // One step's fragments stay live instead of four: without this the compiler hoists every
      // step's LDS loads above the first WMMA (hundreds of moves stalled up front).
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < FTM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }

  // Epilogue: the row's reference exponent is a plain exponent add, the row scale a float multiply.
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int) Wref[n] << 23) : 0.f;
  }
  float* __restrict__ Cb = C + (size_t) (m0 + wm * FTM * 16) * N;
  const float* __restrict__ Asb = As + m0 + wm * FTM * 16;
  const bool full = (m0 + wm * FTM * 16 + (FTM - 1) * 16 + kb8 + 7 < M) &&
                    (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
  if (full) {
#pragma unroll
    for (int i = 0; i < FTM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int r = i * 16 + kb8 + e;
          const int n = n0 + wn * TN * 16 + j * 16 + col;
          Cb[r * N + n] = acc[i][j][e] * rf[j] * Asb[r];
        }
    return;
  }
#pragma unroll
  for (int i = 0; i < FTM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * FTM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) C[(size_t) m * N + n] = acc[i][j][e] * rf[j] * As[m];
      }
}

// Per-row max block exponent.  N rows x K/32 bytes: tiny (<= 40 KB per expert), stream-ordered.
template <bool GU>
__global__ void rowmax_kernel(const unsigned char* __restrict__ w0, const unsigned char* __restrict__ w1,
                              unsigned char* __restrict__ Wref, int N, int nb, int row_bytes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const unsigned char* rowp = GU ? (((n & 1) ? w1 : w0) + (size_t) (n >> 1) * row_bytes)
                                 : (w0 + (size_t) n * row_bytes);
  int m = 0;
  for (int b = 0; b < nb; ++b) {
    const int s = rowp[(size_t) b * 17];
    if (s > m) m = s;
  }
  Wref[n] = (unsigned char) m;
}

bool gfx12_ok() {
    static const bool ok = [] {
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess) return false;
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
        return std::strncmp(prop.gcnArchName, "gfx120", 6) == 0;
    }();
    return ok;
}

}  // namespace

bool mxfp4_folded_wref(const uint8_t* gate, const uint8_t* up, const uint8_t* down, uint8_t* wref,
                       int64_t N, int64_t K, bool gu, void* stream) {
#if STRATA_FOLDED_WMMA
    if (N <= 0 || K <= 0 || (K % 32) != 0 || !gfx12_ok()) return false;
    const int nb = (int) (K / 32), rb = nb * 17;
    const unsigned blocks = (unsigned) ((N + 63) / 64);
    if (gu) rowmax_kernel<true><<<blocks, 64, 0, (cudaStream_t) stream>>>(gate, up, wref, (int) N, nb, rb);
    else    rowmax_kernel<false><<<blocks, 64, 0, (cudaStream_t) stream>>>(down, down, wref, (int) N, nb, rb);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) gate; (void) up; (void) down; (void) wref; (void) N; (void) K; (void) gu; (void) stream;
    return false;
#endif
}

bool mxfp4_folded_gu(const uint8_t* X, const uint8_t* gate, const uint8_t* up, const uint8_t* wref, float* Y,
                     int64_t T, int64_t N, int64_t K, int64_t ldy, const float* row_scale, void* stream) {
#if STRATA_FOLDED_WMMA
    if (T <= 0) return true;
    if (N <= 0 || K <= 0 || (K % 32) != 0 || (N % 64) != 0 || ldy != N || !gfx12_ok()) return false;
    const dim3 grid((unsigned) (N / 64), (unsigned) ((T + FBMF - 1) / FBMF));
    folded_kernel<2, true><<<grid, FNTHREADS, 0, (cudaStream_t) stream>>>(
        X, gate, up, wref, row_scale, Y, (int) T, (int) N, (int) K, (int) ((K / 32) * 17));
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) gate; (void) up; (void) wref; (void) Y; (void) T; (void) N; (void) K; (void) ldy;
    (void) row_scale; (void) stream;
    return false;
#endif
}

bool mxfp4_folded_d(const uint8_t* X, const uint8_t* down, const uint8_t* wref, float* Y, int64_t T, int64_t N,
                    int64_t K, int64_t ldy, const float* row_scale, void* stream) {
#if STRATA_FOLDED_WMMA
    if (T <= 0) return true;
    if (N <= 0 || K <= 0 || (K % 32) != 0 || (N % 64) != 0 || ldy != N || !gfx12_ok()) return false;
    const dim3 grid((unsigned) (N / 64), (unsigned) ((T + FBMF - 1) / FBMF));
    folded_kernel<2, false><<<grid, FNTHREADS, 0, (cudaStream_t) stream>>>(
        X, down, down, wref, row_scale, Y, (int) T, (int) N, (int) K, (int) ((K / 32) * 17));
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) down; (void) wref; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) row_scale;
    (void) stream;
    return false;
#endif
}

}  // namespace strata::prefill
#endif

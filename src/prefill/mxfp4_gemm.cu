// src/prefill/mxfp4_gemm.cu - MXFP4 expert GEMM on gfx12 with the 4-bit weights dequantized to E4M3
// inside the kernel (the vLLM Quark W4A8 / AITER moe_gemm_a8w4 model).  No FP8 copy of the weights is
// materialized: each 16-wide k step reads the 17-byte MXFP4 blocks (E8M0 scale + 16 nibbles), converts
// the 8 values a lane needs, and stages them in LDS.  The weight traffic is 0.5 byte/value, and the
// separate dequant pass the earlier FP8 path used is gone.
#include "strata/prefill/mxfp4_gemm.hpp"

#include <cuda_runtime.h>

#include <cstring>

#if defined(__HIPCC__)
#define STRATA_WMMA 1
#else
#define STRATA_WMMA 0
#endif

#if STRATA_WMMA
namespace {

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((vector_size(8)));

// MXFP4->E4M3 LUT: index (scale_byte << 4) | nibble. Byte-exact vs the old
// old ALU dequant path (host table built with parity ref_e4m3 math).
__constant__ uint8_t k_mxfp4_fp8_lut[4096] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 128, 128, 128, 128, 128, 128, 129,
    0, 0, 0, 0, 0, 1, 1, 2, 0, 128, 128, 128, 128, 129, 129, 130,
    0, 0, 0, 1, 1, 2, 2, 3, 0, 128, 128, 129, 129, 130, 130, 131,
    0, 0, 1, 2, 2, 3, 4, 6, 0, 128, 129, 130, 130, 131, 132, 134,
    0, 1, 2, 3, 4, 6, 8, 12, 0, 129, 130, 131, 132, 134, 136, 140,
    0, 2, 4, 6, 8, 12, 16, 20, 0, 130, 132, 134, 136, 140, 144, 148,
    0, 4, 8, 12, 16, 20, 24, 28, 0, 132, 136, 140, 144, 148, 152, 156,
    0, 8, 16, 20, 24, 28, 32, 36, 0, 136, 144, 148, 152, 156, 160, 164,
    0, 16, 24, 28, 32, 36, 40, 44, 0, 144, 152, 156, 160, 164, 168, 172,
    0, 24, 32, 36, 40, 44, 48, 52, 0, 152, 160, 164, 168, 172, 176, 180,
    0, 32, 40, 44, 48, 52, 56, 60, 0, 160, 168, 172, 176, 180, 184, 188,
    0, 40, 48, 52, 56, 60, 64, 68, 0, 168, 176, 180, 184, 188, 192, 196,
    0, 48, 56, 60, 64, 68, 72, 76, 0, 176, 184, 188, 192, 196, 200, 204,
    0, 56, 64, 68, 72, 76, 80, 84, 0, 184, 192, 196, 200, 204, 208, 212,
    0, 64, 72, 76, 80, 84, 88, 92, 0, 192, 200, 204, 208, 212, 216, 220,
    0, 72, 80, 84, 88, 92, 96, 100, 0, 200, 208, 212, 216, 220, 224, 228,
    0, 80, 88, 92, 96, 100, 104, 108, 0, 208, 216, 220, 224, 228, 232, 236,
    0, 88, 96, 100, 104, 108, 112, 116, 0, 216, 224, 228, 232, 236, 240, 244,
    0, 96, 104, 108, 112, 116, 120, 124, 0, 224, 232, 236, 240, 244, 248, 252,
    0, 104, 112, 116, 120, 124, 126, 126, 0, 232, 240, 244, 248, 252, 254, 254,
    0, 112, 120, 124, 126, 126, 126, 126, 0, 240, 248, 252, 254, 254, 254, 254,
    0, 120, 126, 126, 126, 126, 126, 126, 0, 248, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254,
    0, 126, 126, 126, 126, 126, 126, 126, 0, 254, 254, 254, 254, 254, 254, 254
};

// 8 consecutive k values of one MXFP4 row: q points at the 8 packed bytes, high selects the high nibbles.
__device__ __forceinline__ void dequant8(uint8_t* dst, const uint8_t* q, uint8_t scale, bool high) {
    const unsigned base = (unsigned) scale << 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int nib = high ? (q[i] >> 4) : (q[i] & 0xF);
        dst[i] = k_mxfp4_fp8_lut[base | (unsigned) nib];
    }
}

__device__ __forceinline__ v2i ld8(const uint8_t* p) { return *reinterpret_cast<const v2i*>(p); }
__device__ __forceinline__ v8f wmma(v2i a, v2i b, v8f c) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
#else
    (void) a; (void) b;
    return c;
#endif
}

__device__ __forceinline__ v2i bfrag_gu(const uint8_t* gate, const uint8_t* up, int n, int kk, int row_bytes) {
    const int src = n >> 1;
    const uint8_t* base = ((n & 1) ? up : gate) + (size_t) src * row_bytes;
    const int blk = kk >> 5, half = (kk >> 4) & 1;
    const uint8_t* q = base + (size_t) blk * 17;
    uint8_t q8[8];
    dequant8(q8, q + 1 + (kk & 15), q[0], half);
    v2i r;
    __builtin_memcpy(&r, q8, 8);
    return r;
}
__device__ __forceinline__ v2i bfrag_d(const uint8_t* down, int n, int kk, int row_bytes) {
    const uint8_t* base = down + (size_t) n * row_bytes;
    const int blk = kk >> 5, half = (kk >> 4) & 1;
    const uint8_t* q = base + (size_t) blk * 17;
    uint8_t q8[8];
    dequant8(q8, q + 1 + (kk & 15), q[0], half);
    v2i r;
    __builtin_memcpy(&r, q8, 8);
    return r;
}

__global__ void mxfp4_gemm_lds_gu(const uint8_t* __restrict__ X, const uint8_t* __restrict__ gate,
                                  const uint8_t* __restrict__ up, float* __restrict__ Y, int T, int N, int K,
                                  int ldy, int row_bytes, const float* __restrict__ rs) {
    __shared__ __attribute__((aligned(16))) uint8_t As[64 * 16];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[64 * 16];
    const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 64;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int msub = wave & 1, nsub = wave >> 1;
    const int arow = 32 * msub + (lane & 15), brow = 32 * nsub + (lane & 15);
    const int k8 = 8 * (lane >> 4);
    const v2i z2 = {0, 0};
    const v8f z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8f c00 = z8, c01 = z8, c10 = z8, c11 = z8;
    const int lr = tid >> 1, lk = (tid & 1) * 8;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int gr = t0 + lr;
        const v2i av = (gr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
        *reinterpret_cast<v2i*>(As + lr * 16 + lk) = av;
        const int n = n0 + lr;
        const int src = n >> 1;
        const uint8_t* base = ((n & 1) ? up : gate) + (size_t) src * row_bytes;
        const int kk = k0 + lk, blk = kk >> 5, half = (kk >> 4) & 1;
        const uint8_t* q = base + (size_t) blk * 17;
        dequant8(Bs + lr * 16 + lk, q + 1 + (kk & 15), q[0], half);
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

__global__ void mxfp4_gemm_lds_d(const uint8_t* __restrict__ X, const uint8_t* __restrict__ down,
                                 float* __restrict__ Y, int T, int N, int K, int ldy, int row_bytes,
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
    const v8f z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8f c00 = z8, c01 = z8, c10 = z8, c11 = z8;
    const int lr = tid >> 1, lk = (tid & 1) * 8;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int gr = t0 + lr;
        const v2i av = (gr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
        *reinterpret_cast<v2i*>(As + lr * 16 + lk) = av;
        const int n = n0 + lr;
        const uint8_t* base = down + (size_t) n * row_bytes;
        const int kk = k0 + lk, blk = kk >> 5, half = (kk >> 4) & 1;
        const uint8_t* q = base + (size_t) blk * 17;
        dequant8(Bs + lr * 16 + lk, q + 1 + (kk & 15), q[0], half);
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


// M-tile LDS kernel: one block owns 4 consecutive 64-row M tiles of one expert and dequantizes the
// 64x16 B tile once per k step, then reuses it across all 4 M tiles.  This is the amortization the
// per-64-row launch was missing (the B dequant, not the WMMA, dominates at small expert row counts).
__global__ void mxfp4_gemm_lds4_gu(const uint8_t* __restrict__ X, const uint8_t* __restrict__ gate,
                                   const uint8_t* __restrict__ up, float* __restrict__ Y, int T, int N, int K,
                                   int ldy, int row_bytes, const float* __restrict__ rs) {
    __shared__ __attribute__((aligned(16))) uint8_t As[4][64 * 16];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[64 * 16];
    const int n0 = blockIdx.x * 64, tb = blockIdx.y * 256;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int msub = wave & 1, nsub = wave >> 1;
    const int arow = 32 * msub + (lane & 15), brow = 32 * nsub + (lane & 15);
    const int k8 = 8 * (lane >> 4);
    const v2i z2 = {0, 0};
    const int lr = tid >> 1, lk = (tid & 1) * 8;
    int nm = (T - tb + 63) >> 6;
    if (nm > 4) nm = 4;
    v8f c[4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j) c[m][j] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int n = n0 + lr;
        const int src = n >> 1;
        const uint8_t* base = ((n & 1) ? up : gate) + (size_t) src * row_bytes;
        const int kk = k0 + lk, blk = kk >> 5, half = (kk >> 4) & 1;
        const uint8_t* q = base + (size_t) blk * 17;
        dequant8(Bs + lr * 16 + lk, q + 1 + (kk & 15), q[0], half);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = tb + 64 * m + lr;
            const v2i av = (m < nm && gr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
            *reinterpret_cast<v2i*>(As[m] + lr * 16 + lk) = av;
        }
        __syncthreads();
        const v2i b0 = *reinterpret_cast<const v2i*>(Bs + brow * 16 + k8);
        const v2i b1 = *reinterpret_cast<const v2i*>(Bs + (brow + 16) * 16 + k8);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= nm) continue;
            const v2i a0 = *reinterpret_cast<const v2i*>(As[m] + arow * 16 + k8);
            const v2i a1 = *reinterpret_cast<const v2i*>(As[m] + (arow + 16) * 16 + k8);
            c[m][0] = wmma(a0, b0, c[m][0]);
            c[m][1] = wmma(a0, b1, c[m][1]);
            c[m][2] = wmma(a1, b0, c[m][2]);
            c[m][3] = wmma(a1, b1, c[m][3]);
        }
        __syncthreads();
    }
    const int n0w = n0 + 32 * nsub;
    const int col = n0w + (lane & 15);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (m >= nm) continue;
        const int t0 = tb + 64 * m;
        const int m0 = t0 + 32 * msub;
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + 8 * (lane >> 4) + i;
            if (row >= T) break;
            const float s0 = rs[row];
            const bool r1 = row + 16 < T;
            const float s1 = r1 ? rs[row + 16] : 0.0f;
            if (col < N) {
                Y[(size_t) row * ldy + col] = c[m][0][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col] = c[m][2][i] * s1;
            }
            if (col + 16 < N) {
                Y[(size_t) row * ldy + col + 16] = c[m][1][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col + 16] = c[m][3][i] * s1;
            }
        }
    }
}

__global__ void mxfp4_gemm_lds4_d(const uint8_t* __restrict__ X, const uint8_t* __restrict__ down,
                                  float* __restrict__ Y, int T, int N, int K, int ldy, int row_bytes,
                                  const float* __restrict__ rs) {
    __shared__ __attribute__((aligned(16))) uint8_t As[4][64 * 16];
    __shared__ __attribute__((aligned(16))) uint8_t Bs[64 * 16];
    const int n0 = blockIdx.x * 64, tb = blockIdx.y * 256;
    const int tid = threadIdx.x;
    const int lane = tid & 31, wave = tid >> 5;
    const int msub = wave & 1, nsub = wave >> 1;
    const int arow = 32 * msub + (lane & 15), brow = 32 * nsub + (lane & 15);
    const int k8 = 8 * (lane >> 4);
    const v2i z2 = {0, 0};
    const int lr = tid >> 1, lk = (tid & 1) * 8;
    int nm = (T - tb + 63) >> 6;
    if (nm > 4) nm = 4;
    v8f c[4][4];
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int j = 0; j < 4; ++j) c[m][j] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int n = n0 + lr;
        const uint8_t* base = down + (size_t) n * row_bytes;
        const int kk = k0 + lk, blk = kk >> 5, half = (kk >> 4) & 1;
        const uint8_t* q = base + (size_t) blk * 17;
        dequant8(Bs + lr * 16 + lk, q + 1 + (kk & 15), q[0], half);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = tb + 64 * m + lr;
            const v2i av = (m < nm && gr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
            *reinterpret_cast<v2i*>(As[m] + lr * 16 + lk) = av;
        }
        __syncthreads();
        const v2i b0 = *reinterpret_cast<const v2i*>(Bs + brow * 16 + k8);
        const v2i b1 = *reinterpret_cast<const v2i*>(Bs + (brow + 16) * 16 + k8);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= nm) continue;
            const v2i a0 = *reinterpret_cast<const v2i*>(As[m] + arow * 16 + k8);
            const v2i a1 = *reinterpret_cast<const v2i*>(As[m] + (arow + 16) * 16 + k8);
            c[m][0] = wmma(a0, b0, c[m][0]);
            c[m][1] = wmma(a0, b1, c[m][1]);
            c[m][2] = wmma(a1, b0, c[m][2]);
            c[m][3] = wmma(a1, b1, c[m][3]);
        }
        __syncthreads();
    }
    const int n0w = n0 + 32 * nsub;
    const int col = n0w + (lane & 15);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (m >= nm) continue;
        const int t0 = tb + 64 * m;
        const int m0 = t0 + 32 * msub;
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + 8 * (lane >> 4) + i;
            if (row >= T) break;
            const float s0 = rs[row];
            const bool r1 = row + 16 < T;
            const float s1 = r1 ? rs[row + 16] : 0.0f;
            if (col < N) {
                Y[(size_t) row * ldy + col] = c[m][0][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col] = c[m][2][i] * s1;
            }
            if (col + 16 < N) {
                Y[(size_t) row * ldy + col + 16] = c[m][1][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col + 16] = c[m][3][i] * s1;
            }
        }
    }
}


// Grouped MXFP4 expert GEMM (vLLM W4A8 style, one launch per layer): each block owns one
// (expert, 64-wide N tile, 256-row M group) = up to 4 M-tiles, dequantizing its 64x16 B tile
// once per k step and reusing it across the M tiles.  X is the chunk's pre-quantized FP8
// (hoisted, one quantize per chunk), rs its per-row scales; the blob stays 4-bit (no 8-bit copy).
__global__ void mxfp4_grouped_gu(const uint8_t* __restrict__ X, const float* __restrict__ rs,
                                 const strata::prefill::Mxfp4ExpertTile* __restrict__ tiles,
                                 int num_tiles) {
    if ((int) blockIdx.x >= num_tiles) return;
    const strata::prefill::Mxfp4ExpertTile t = tiles[blockIdx.x];
    const uint8_t* __restrict__ gate = t.blob;
    const uint8_t* __restrict__ up = t.up;
    float* __restrict__ Y = t.out;
    const int T = (int) t.t_count;
    const int N = (int) t.N, K = (int) t.K;
    const int ldy = (int) t.ldy, row_bytes = (int) t.row_bytes;
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
        const int n = n0 + lr;
        const int src = n >> 1;
        const uint8_t* base = ((n & 1) ? up : gate) + (size_t) src * row_bytes;
        const int kk = k0 + lk, blk = kk >> 5, half = (kk >> 4) & 1;
        const uint8_t* q = base + (size_t) blk * 17;
        dequant8(Bs + lr * 16 + lk, q + 1 + (kk & 15), q[0], half);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = (int) (t_start + tb + 64 * m + lr);
            const v2i av = (m < nm && tb + 64 * m + lr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
            *reinterpret_cast<v2i*>(As[m] + lr * 16 + lk) = av;
        }
        __syncthreads();
        const v2i b0 = *reinterpret_cast<const v2i*>(Bs + brow * 16 + k8);
        const v2i b1 = *reinterpret_cast<const v2i*>(Bs + (brow + 16) * 16 + k8);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= nm) continue;
            const v2i a0 = *reinterpret_cast<const v2i*>(As[m] + arow * 16 + k8);
            const v2i a1 = *reinterpret_cast<const v2i*>(As[m] + (arow + 16) * 16 + k8);
            c[m][0] = wmma(a0, b0, c[m][0]);
            c[m][1] = wmma(a0, b1, c[m][1]);
            c[m][2] = wmma(a1, b0, c[m][2]);
            c[m][3] = wmma(a1, b1, c[m][3]);
        }
        __syncthreads();
    }
    const int n0w = n0 + 32 * nsub;
    const int col = n0w + (lane & 15);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (m >= nm) continue;
        const int t0 = tb + 64 * m;
        const int m0 = t0 + 32 * msub;
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + 8 * (lane >> 4) + i;
            if (row >= T) break;
            const float s0 = rs[t_start + row];
            const bool r1 = row + 16 < T;
            const float s1 = r1 ? rs[t_start + row + 16] : 0.0f;
            if (col < N) {
                Y[(size_t) row * ldy + col] = c[m][0][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col] = c[m][2][i] * s1;
            }
            if (col + 16 < N) {
                Y[(size_t) row * ldy + col + 16] = c[m][1][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col + 16] = c[m][3][i] * s1;
            }
        }
    }
}

__global__ void mxfp4_grouped_d(const uint8_t* __restrict__ X, const float* __restrict__ rs,
                                const strata::prefill::Mxfp4ExpertTile* __restrict__ tiles,
                                int num_tiles) {
    if ((int) blockIdx.x >= num_tiles) return;
    const strata::prefill::Mxfp4ExpertTile t = tiles[blockIdx.x];
    const uint8_t* __restrict__ down = t.blob;
    float* __restrict__ Y = t.out;
    const int T = (int) t.t_count;
    const int N = (int) t.N, K = (int) t.K;
    const int ldy = (int) t.ldy, row_bytes = (int) t.row_bytes;
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
        const int n = n0 + lr;
        const uint8_t* base = down + (size_t) n * row_bytes;
        const int kk = k0 + lk, blk = kk >> 5, half = (kk >> 4) & 1;
        const uint8_t* q = base + (size_t) blk * 17;
        dequant8(Bs + lr * 16 + lk, q + 1 + (kk & 15), q[0], half);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = (int) (t_start + tb + 64 * m + lr);
            const v2i av = (m < nm && tb + 64 * m + lr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
            *reinterpret_cast<v2i*>(As[m] + lr * 16 + lk) = av;
        }
        __syncthreads();
        const v2i b0 = *reinterpret_cast<const v2i*>(Bs + brow * 16 + k8);
        const v2i b1 = *reinterpret_cast<const v2i*>(Bs + (brow + 16) * 16 + k8);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= nm) continue;
            const v2i a0 = *reinterpret_cast<const v2i*>(As[m] + arow * 16 + k8);
            const v2i a1 = *reinterpret_cast<const v2i*>(As[m] + (arow + 16) * 16 + k8);
            c[m][0] = wmma(a0, b0, c[m][0]);
            c[m][1] = wmma(a0, b1, c[m][1]);
            c[m][2] = wmma(a1, b0, c[m][2]);
            c[m][3] = wmma(a1, b1, c[m][3]);
        }
        __syncthreads();
    }
    const int n0w = n0 + 32 * nsub;
    const int col = n0w + (lane & 15);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (m >= nm) continue;
        const int t0 = tb + 64 * m;
        const int m0 = t0 + 32 * msub;
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + 8 * (lane >> 4) + i;
            if (row >= T) break;
            const float s0 = rs[t_start + row];
            const bool r1 = row + 16 < T;
            const float s1 = r1 ? rs[t_start + row + 16] : 0.0f;
            if (col < N) {
                Y[(size_t) row * ldy + col] = c[m][0][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col] = c[m][2][i] * s1;
            }
            if (col + 16 < N) {
                Y[(size_t) row * ldy + col + 16] = c[m][1][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col + 16] = c[m][3][i] * s1;
            }
        }
    }
}


// ---- Tiled 4-bit layout (coalesced in-kernel dequant) ----
// Per expert matrix (gu: 1280 interleaved rows; down: N rows), K cols:
//   nibble plane: [n_tile][k_blk][row 0..63][16 bytes]  (1024 B per tile)
//   scale plane:  [n_tile][k_blk][row 0..63]            (64 B per tile)
// tile (nt,kb) nibbles at base + ((nt*Kblk + kb) << 10), scales at sbase + (nt*Kblk + kb)*64.
// Same bytes as row-major (N*K*17/32); fits in the dq scratch (sized N*K for FP8).
// In Mxfp4ExpertTile, blob = this expert's tiled nibble base, up = this expert's tiled scale base.
typedef int v4i __attribute__((vector_size(16)));

__global__ void mxfp4_retile_gu_kernel(const uint8_t* __restrict__ gate, const uint8_t* __restrict__ up,
                                uint8_t* __restrict__ nib, uint8_t* __restrict__ scl, int N, int K, int row_bytes) {
    const int nblk = K >> 5;
    const int nt = blockIdx.x, kb = blockIdx.y;
    const int r = threadIdx.x;   // 64 threads
    __shared__ __attribute__((aligned(16))) uint8_t tile[64][17];
    if (r < 64) {
        const int n = nt * 64 + r;
        const int src = n >> 1;
        const uint8_t* base = ((n & 1) ? up : gate) + (size_t) src * row_bytes;
        const uint8_t* q = base + (size_t) kb * 17;
#pragma unroll
        for (int i = 0; i < 16; ++i) tile[r][1 + i] = q[1 + i];
        tile[r][0] = q[0];
    }
    __syncthreads();
    if (r < 64) {
        uint8_t* nd = nib + ((size_t) (nt * nblk + kb) << 10) + (size_t) r * 16;
        *reinterpret_cast<v4i*>(nd) = *reinterpret_cast<v4i*>(&tile[r][1]);
        scl[((size_t) (nt * nblk + kb) << 6) + r] = tile[r][0];
    }
}

__global__ void mxfp4_retile_d_kernel(const uint8_t* __restrict__ down, uint8_t* __restrict__ nib, uint8_t* __restrict__ scl,
                               int N, int K, int row_bytes) {
    const int nblk = K >> 5;
    const int nt = blockIdx.x, kb = blockIdx.y;
    const int r = threadIdx.x;   // 64 threads
    __shared__ __attribute__((aligned(16))) uint8_t tile[64][17];
    if (r < 64) {
        const int n = nt * 64 + r;
        const uint8_t* q = down + (size_t) n * row_bytes + (size_t) kb * 17;
#pragma unroll
        for (int i = 0; i < 16; ++i) tile[r][1 + i] = q[1 + i];
        tile[r][0] = q[0];
    }
    __syncthreads();
    if (r < 64) {
        uint8_t* nd = nib + ((size_t) (nt * nblk + kb) << 10) + (size_t) r * 16;
        *reinterpret_cast<v4i*>(nd) = *reinterpret_cast<v4i*>(&tile[r][1]);
        scl[((size_t) (nt * nblk + kb) << 6) + r] = tile[r][0];
    }
}

// Grouped GEMM on TILED 4-bit (coalesced B reads, in-register dequant to FP8, FP8 WMMA).
// Same MT structure as mxfp4_gemm_lds4_gu but B comes from the per-expert tiled nibble/scale
// planes (t.blob = nibble base, t.up = scale base); no separate dequant pass, no 8-bit copy.
__global__ void mxfp4_grouped_tiled_gu_kernel(const uint8_t* __restrict__ X, const float* __restrict__ rs,
                                       const strata::prefill::Mxfp4ExpertTile* __restrict__ tiles, int num_tiles) {
    if ((int) blockIdx.x >= num_tiles) return;
    const strata::prefill::Mxfp4ExpertTile t = tiles[blockIdx.x];
    float* __restrict__ Y = t.out;
    const int T = (int) t.t_count;
    const int N = (int) t.N, K = (int) t.K;
    const int Kblk = K >> 5;
    const int ldy = (int) t.ldy;
    const int64_t t_start = t.t_start;
    const uint8_t* __restrict__ tnib = t.blob;
    const uint8_t* __restrict__ tscl = t.up;
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
        const int kk = k0 + lk, blk = kk >> 5;
        const v4i nv = *reinterpret_cast<const v4i*>(tnib + ((size_t) ((n0 >> 6) * Kblk + blk) << 10) + (size_t) lr * 16);
        uint8_t nb[16];
        *reinterpret_cast<v4i*>(nb) = nv;
        const uint8_t sc = tscl[((size_t) ((n0 >> 6) * Kblk + blk) << 6) + lr];
        const int o2 = ((kk & 31) >> 1);
        uint8_t out8[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t by = nb[o2 + i];
            out8[2 * i] = k_mxfp4_fp8_lut[((unsigned) sc << 4) | (by & 0xF)];
            out8[2 * i + 1] = k_mxfp4_fp8_lut[((unsigned) sc << 4) | (by >> 4)];
        }
        *reinterpret_cast<v2i*>(Bs + lr * 16 + lk) = *reinterpret_cast<v2i*>(out8);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = (int) (t_start + tb + 64 * m + lr);
            const v2i av = (m < nm && tb + 64 * m + lr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
            *reinterpret_cast<v2i*>(As[m] + lr * 16 + lk) = av;
        }
        __syncthreads();
        const v2i b0 = *reinterpret_cast<const v2i*>(Bs + brow * 16 + k8);
        const v2i b1 = *reinterpret_cast<const v2i*>(Bs + (brow + 16) * 16 + k8);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= nm) continue;
            const v2i a0 = *reinterpret_cast<const v2i*>(As[m] + arow * 16 + k8);
            const v2i a1 = *reinterpret_cast<const v2i*>(As[m] + (arow + 16) * 16 + k8);
            c[m][0] = wmma(a0, b0, c[m][0]);
            c[m][1] = wmma(a0, b1, c[m][1]);
            c[m][2] = wmma(a1, b0, c[m][2]);
            c[m][3] = wmma(a1, b1, c[m][3]);
        }
        __syncthreads();
    }
    const int n0w = n0 + 32 * nsub;
    const int col = n0w + (lane & 15);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (m >= nm) continue;
        const int t0 = tb + 64 * m;
        const int m0 = t0 + 32 * msub;
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + 8 * (lane >> 4) + i;
            if (row >= T) break;
            const float s0 = rs[t_start + row];
            const bool r1 = row + 16 < T;
            const float s1 = r1 ? rs[t_start + row + 16] : 0.0f;
            if (col < N) {
                Y[(size_t) row * ldy + col] = c[m][0][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col] = c[m][2][i] * s1;
            }
            if (col + 16 < N) {
                Y[(size_t) row * ldy + col + 16] = c[m][1][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col + 16] = c[m][3][i] * s1;
            }
        }
    }
}

__global__ void mxfp4_grouped_tiled_d_kernel(const uint8_t* __restrict__ X, const float* __restrict__ rs,
                                      const strata::prefill::Mxfp4ExpertTile* __restrict__ tiles, int num_tiles) {
    if ((int) blockIdx.x >= num_tiles) return;
    const strata::prefill::Mxfp4ExpertTile t = tiles[blockIdx.x];
    float* __restrict__ Y = t.out;
    const int T = (int) t.t_count;
    const int N = (int) t.N, K = (int) t.K;
    const int Kblk = K >> 5;
    const int ldy = (int) t.ldy;
    const int64_t t_start = t.t_start;
    const uint8_t* __restrict__ tnib = t.blob;
    const uint8_t* __restrict__ tscl = t.up;
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
        const int kk = k0 + lk, blk = kk >> 5;
        const v4i nv = *reinterpret_cast<const v4i*>(tnib + ((size_t) ((n0 >> 6) * Kblk + blk) << 10) + (size_t) lr * 16);
        uint8_t nb[16];
        *reinterpret_cast<v4i*>(nb) = nv;
        const uint8_t sc = tscl[((size_t) ((n0 >> 6) * Kblk + blk) << 6) + lr];
        const int o2 = ((kk & 31) >> 1);
        uint8_t out8[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint8_t by = nb[o2 + i];
            out8[2 * i] = k_mxfp4_fp8_lut[((unsigned) sc << 4) | (by & 0xF)];
            out8[2 * i + 1] = k_mxfp4_fp8_lut[((unsigned) sc << 4) | (by >> 4)];
        }
        *reinterpret_cast<v2i*>(Bs + lr * 16 + lk) = *reinterpret_cast<v2i*>(out8);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int gr = (int) (t_start + tb + 64 * m + lr);
            const v2i av = (m < nm && tb + 64 * m + lr < T) ? ld8(X + (size_t) gr * K + k0 + lk) : z2;
            *reinterpret_cast<v2i*>(As[m] + lr * 16 + lk) = av;
        }
        __syncthreads();
        const v2i b0 = *reinterpret_cast<const v2i*>(Bs + brow * 16 + k8);
        const v2i b1 = *reinterpret_cast<const v2i*>(Bs + (brow + 16) * 16 + k8);
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m >= nm) continue;
            const v2i a0 = *reinterpret_cast<const v2i*>(As[m] + arow * 16 + k8);
            const v2i a1 = *reinterpret_cast<const v2i*>(As[m] + (arow + 16) * 16 + k8);
            c[m][0] = wmma(a0, b0, c[m][0]);
            c[m][1] = wmma(a0, b1, c[m][1]);
            c[m][2] = wmma(a1, b0, c[m][2]);
            c[m][3] = wmma(a1, b1, c[m][3]);
        }
        __syncthreads();
    }
    const int n0w = n0 + 32 * nsub;
    const int col = n0w + (lane & 15);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        if (m >= nm) continue;
        const int t0 = tb + 64 * m;
        const int m0 = t0 + 32 * msub;
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + 8 * (lane >> 4) + i;
            if (row >= T) break;
            const float s0 = rs[t_start + row];
            const bool r1 = row + 16 < T;
            const float s1 = r1 ? rs[t_start + row + 16] : 0.0f;
            if (col < N) {
                Y[(size_t) row * ldy + col] = c[m][0][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col] = c[m][2][i] * s1;
            }
            if (col + 16 < N) {
                Y[(size_t) row * ldy + col + 16] = c[m][1][i] * s0;
                if (r1) Y[(size_t) (row + 16) * ldy + col + 16] = c[m][3][i] * s1;
            }
        }
    }
}

__global__ void mxfp4_gemm_2x2_gu(const uint8_t* __restrict__ X, const uint8_t* __restrict__ gate,
                                  const uint8_t* __restrict__ up, float* __restrict__ Y, int T, int N, int K,
                                  int ldy, int row_bytes, const float* __restrict__ rs) {
    const int n0 = blockIdx.x * 32, t0 = blockIdx.y * 32;
    const int l = threadIdx.x;
    const int arow = t0 + (l & 15), brow = n0 + (l & 15);
    const int k8 = 8 * (l >> 4);
    const v2i z2 = {0, 0};
    const v8f z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8f c00 = z8, c01 = z8, c10 = z8, c11 = z8;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int kk = k0 + k8;
        v2i a0 = z2, a1 = z2, b0 = z2, b1 = z2;
        if (arow < T) a0 = ld8(X + (size_t) arow * K + kk);
        if (arow + 16 < T) a1 = ld8(X + (size_t) (arow + 16) * K + kk);
        if (brow < N) b0 = bfrag_gu(gate, up, brow, kk, row_bytes);
        if (brow + 16 < N) b1 = bfrag_gu(gate, up, brow + 16, kk, row_bytes);
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

__global__ void mxfp4_gemm_2x2_d(const uint8_t* __restrict__ X, const uint8_t* __restrict__ down,
                                 float* __restrict__ Y, int T, int N, int K, int ldy, int row_bytes,
                                 const float* __restrict__ rs) {
    const int n0 = blockIdx.x * 32, t0 = blockIdx.y * 32;
    const int l = threadIdx.x;
    const int arow = t0 + (l & 15), brow = n0 + (l & 15);
    const int k8 = 8 * (l >> 4);
    const v2i z2 = {0, 0};
    const v8f z8 = {0, 0, 0, 0, 0, 0, 0, 0};
    v8f c00 = z8, c01 = z8, c10 = z8, c11 = z8;
    for (int k0 = 0; k0 < K; k0 += 16) {
        const int kk = k0 + k8;
        v2i a0 = z2, a1 = z2, b0 = z2, b1 = z2;
        if (arow < T) a0 = ld8(X + (size_t) arow * K + kk);
        if (arow + 16 < T) a1 = ld8(X + (size_t) (arow + 16) * K + kk);
        if (brow < N) b0 = bfrag_d(down, brow, kk, row_bytes);
        if (brow + 16 < N) b1 = bfrag_d(down, brow + 16, kk, row_bytes);
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

bool gfx12_device() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return false;
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
    return std::strncmp(prop.gcnArchName, "gfx120", 6) == 0;
}

}  // namespace
#endif

namespace strata::prefill {

bool mxfp4_gemm_gu(const uint8_t* X, const uint8_t* gate, const uint8_t* up, float* Y, int64_t T, int64_t N,
                   int64_t K, int64_t ldy, const float* row_scale, void* stream) {
#if STRATA_WMMA
    if (T <= 0) return true;
    if (N <= 0 || K <= 0 || (K % 32) != 0 || (N % 32) != 0 || ldy < N || !gfx12_device()) return false;
    const int row_bytes = (int) ((K / 32) * 17);
    const bool lds = T >= 128 && (N % 64) == 0 && (K < 2048 || T >= 256);
    if (lds) {
        const dim3 grid((unsigned) (N / 64), (unsigned) ((T + 255) / 256));
        mxfp4_gemm_lds4_gu<<<grid, 128, 0, (cudaStream_t) stream>>>(X, gate, up, Y, (int) T, (int) N, (int) K,
                                                                    (int) ldy, row_bytes, row_scale);
    } else {
        const dim3 grid((unsigned) (N / 32), (unsigned) ((T + 31) / 32));
        mxfp4_gemm_2x2_gu<<<grid, 32, 0, (cudaStream_t) stream>>>(X, gate, up, Y, (int) T, (int) N, (int) K,
                                                                  (int) ldy, row_bytes, row_scale);
    }
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) gate; (void) up; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) row_scale;
    (void) stream;
    return false;
#endif
}

bool mxfp4_gemm_d(const uint8_t* X, const uint8_t* down, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                  const float* row_scale, void* stream) {
#if STRATA_WMMA
    if (T <= 0) return true;
    if (N <= 0 || K <= 0 || (K % 32) != 0 || (N % 32) != 0 || ldy < N || !gfx12_device()) return false;
    const int row_bytes = (int) ((K / 32) * 17);
    const bool lds = T >= 128 && (N % 64) == 0 && (K < 2048 || T >= 256);
    if (lds) {
        const dim3 grid((unsigned) (N / 64), (unsigned) ((T + 255) / 256));
        mxfp4_gemm_lds4_d<<<grid, 128, 0, (cudaStream_t) stream>>>(X, down, Y, (int) T, (int) N, (int) K,
                                                                   (int) ldy, row_bytes, row_scale);
    } else {
        const dim3 grid((unsigned) (N / 32), (unsigned) ((T + 31) / 32));
        mxfp4_gemm_2x2_d<<<grid, 32, 0, (cudaStream_t) stream>>>(X, down, Y, (int) T, (int) N, (int) K,
                                                                 (int) ldy, row_bytes, row_scale);
    }
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) down; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) row_scale; (void) stream;
    return false;
#endif
}


bool mxfp4_grouped_gemm_gu(const uint8_t* X, const float* row_scale, const Mxfp4ExpertTile* tiles, int num_tiles,
                           void* stream) {
#if STRATA_WMMA
    if (num_tiles <= 0) return true;
    if (!X || !row_scale || !tiles || !gfx12_device()) return false;
    mxfp4_grouped_gu<<<dim3((unsigned) num_tiles), 128, 0, (cudaStream_t) stream>>>(X, row_scale, tiles, num_tiles);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) row_scale; (void) tiles; (void) num_tiles; (void) stream;
    return false;
#endif
}

bool mxfp4_grouped_gemm_d(const uint8_t* X, const float* row_scale, const Mxfp4ExpertTile* tiles, int num_tiles,
                          void* stream) {
#if STRATA_WMMA
    if (num_tiles <= 0) return true;
    if (!X || !row_scale || !tiles || !gfx12_device()) return false;
    mxfp4_grouped_d<<<dim3((unsigned) num_tiles), 128, 0, (cudaStream_t) stream>>>(X, row_scale, tiles, num_tiles);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) row_scale; (void) tiles; (void) num_tiles; (void) stream;
    return false;
#endif
}


bool mxfp4_retile_gu(const uint8_t* gate, const uint8_t* up, uint8_t* nib, uint8_t* scl, int64_t N, int64_t K,
                     void* stream) {
#if STRATA_WMMA
    if (N <= 0 || K <= 0 || (N % 64) != 0 || (K % 32) != 0 || !gfx12_device()) return false;
    const dim3 grid((unsigned) (N / 64), (unsigned) (K / 32));
    mxfp4_retile_gu_kernel<<<grid, 64, 0, (cudaStream_t) stream>>>(gate, up, nib, scl, (int) N, (int) K,
                                                            (int) ((K / 32) * 17));
    return cudaGetLastError() == cudaSuccess;
#else
    (void) gate; (void) up; (void) nib; (void) scl; (void) N; (void) K; (void) stream;
    return false;
#endif
}

bool mxfp4_retile_d(const uint8_t* down, uint8_t* nib, uint8_t* scl, int64_t N, int64_t K, void* stream) {
#if STRATA_WMMA
    if (N <= 0 || K <= 0 || (N % 64) != 0 || (K % 32) != 0 || !gfx12_device()) return false;
    const dim3 grid((unsigned) (N / 64), (unsigned) (K / 32));
    mxfp4_retile_d_kernel<<<grid, 64, 0, (cudaStream_t) stream>>>(down, nib, scl, (int) N, (int) K, (int) ((K / 32) * 17));
    return cudaGetLastError() == cudaSuccess;
#else
    (void) down; (void) nib; (void) scl; (void) N; (void) K; (void) stream;
    return false;
#endif
}

bool mxfp4_grouped_tiled_gu(const uint8_t* X, const float* row_scale, const Mxfp4ExpertTile* tiles, int num_tiles,
                            void* stream) {
#if STRATA_WMMA
    if (num_tiles <= 0) return true;
    if (!X || !row_scale || !tiles || !gfx12_device()) return false;
    mxfp4_grouped_tiled_gu_kernel<<<dim3((unsigned) num_tiles), 128, 0, (cudaStream_t) stream>>>(X, row_scale, tiles, num_tiles);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) row_scale; (void) tiles; (void) num_tiles; (void) stream;
    return false;
#endif
}

bool mxfp4_grouped_tiled_d(const uint8_t* X, const float* row_scale, const Mxfp4ExpertTile* tiles, int num_tiles,
                           void* stream) {
#if STRATA_WMMA
    if (num_tiles <= 0) return true;
    if (!X || !row_scale || !tiles || !gfx12_device()) return false;
    mxfp4_grouped_tiled_d_kernel<<<dim3((unsigned) num_tiles), 128, 0, (cudaStream_t) stream>>>(X, row_scale, tiles, num_tiles);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) X; (void) row_scale; (void) tiles; (void) num_tiles; (void) stream;
    return false;
#endif
}

}  // namespace strata::prefill

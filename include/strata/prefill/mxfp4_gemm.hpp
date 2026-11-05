#pragma once

#include <cstdint>

namespace strata::prefill {

/// Gate/up (interleaved) and down MXFP4 expert GEMM, vLLM W4A8 style: the weights stay in their 4-bit
/// GGUF form (17-byte blocks: E8M0 scale + 16 packed nibbles) and are dequantized to E4M3 inside the
/// kernel (the scale folded into the FP8 value).  X is E4M3 activations [T, K], Y is FP32 [T, N], and
/// row_scale[t] multiplies row t of the result.  gate/up are the expert blob pointers; gu rows are
/// interleaved as 2*r + parity (gate then up), exactly the layout the FP16 path writes.
bool mxfp4_gemm_gu(const uint8_t* X, const uint8_t* gate, const uint8_t* up, float* Y, int64_t T, int64_t N,
                   int64_t K, int64_t ldy, const float* row_scale, void* stream);
bool mxfp4_gemm_d(const uint8_t* X, const uint8_t* down, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                  const float* row_scale, void* stream);

/// One output tile of the grouped launch: a 64-wide N slice (n_tile) of one expert's rows
/// [t_start, t_start + t_count), M group m_group (256 rows, up to 4 M-tiles per block).
struct Mxfp4ExpertTile {
    const uint8_t* blob;   ///< gate (gu) or down (d) MXFP4 blob base
    const uint8_t* up;     ///< gu only: up matrix base (nullptr for down)
    float* out;            ///< this expert's output rows base (GU or Dm)
    int64_t t_start;       ///< first token row in X / row_scale
    int64_t t_count;       ///< rows for this expert
    int64_t n_tile;        ///< this block's N slice = n_tile * 64
    int64_t m_group;       ///< this block's M chunk = m_group * 256
    int64_t N;             ///< expert N (1280 gu, n_embd d)
    int64_t K;             ///< expert K (n_embd gu, 640 d)
    int64_t ldy;           ///< output row stride
    int64_t row_bytes;     ///< MXFP4 row bytes ((K / 32) * 17)
};

/// Grouped MXFP4 expert GEMM: one launch for a whole layer's tiles (see Mxfp4ExpertTile).
/// X is the chunk's pre-quantized FP8 (hoisted, one quantize per chunk), row_scale its per-row scales.
bool mxfp4_grouped_gemm_gu(const uint8_t* X, const float* row_scale, const Mxfp4ExpertTile* tiles, int num_tiles,
                           void* stream);
bool mxfp4_grouped_gemm_d(const uint8_t* X, const float* row_scale, const Mxfp4ExpertTile* tiles, int num_tiles,
                          void* stream);

// Tiled 4-bit layout (coalesced in-kernel dequant): retile row-major 17-byte blocks into
// per-expert nibble/scale planes, then grouped GEMM reads the tiles (no separate dequant,
// no 8-bit copy). In Mxfp4ExpertTile, blob = tiled nibble base, up = tiled scale base.
bool mxfp4_retile_gu(const uint8_t* gate, const uint8_t* up, uint8_t* nib, uint8_t* scl, int64_t N, int64_t K,
                     void* stream);
bool mxfp4_retile_d(const uint8_t* down, uint8_t* nib, uint8_t* scl, int64_t N, int64_t K, void* stream);
bool mxfp4_grouped_tiled_gu(const uint8_t* X, const float* row_scale, const Mxfp4ExpertTile* tiles, int num_tiles,
                            void* stream);
bool mxfp4_grouped_tiled_d(const uint8_t* X, const float* row_scale, const Mxfp4ExpertTile* tiles, int num_tiles,
                           void* stream);

}  // namespace strata::prefill

// include/strata/kernels/gfx_arch.hpp - AMD architecture prefix / family detection
#pragma once

#include <cstddef>
#include <cstring>

namespace strata::kernels {

/// `gcn` is hipDeviceProp_t::gcnArchName ("gfx1151", or "gfx1151:sramecc-:xnack-"): true when it is exactly `name`.
inline bool gfx_arch_is(const char* gcn, const char* name) {
    if (gcn == nullptr) return false;
    const std::size_t n = std::strlen(name);
    return std::strncmp(gcn, name, n) == 0 && (gcn[n] == '\0' || gcn[n] == ':');
}

/// Check prefix match: e.g. prefix "gfx11" matches "gfx1100", "gfx1151", etc.
inline bool gfx_arch_starts_with(const char* gcn, const char* prefix) {
    if (gcn == nullptr || prefix == nullptr) return false;
    const std::size_t n = std::strlen(prefix);
    return std::strncmp(gcn, prefix, n) == 0;
}

/// RDNA3 / RDNA3.5 (gfx11xx)
inline bool gfx_arch_is_gfx11(const char* gcn) {
    return gfx_arch_starts_with(gcn, "gfx11");
}

/// RDNA4 (gfx12xx)
inline bool gfx_arch_is_gfx12(const char* gcn) {
    return gfx_arch_starts_with(gcn, "gfx12");
}

/// The gfx11 targets supporting WMMA matrix-core kernels
inline bool gfx_arch_is_gfx11_wmma(const char* gcn) {
    return gfx_arch_is_gfx11(gcn);
}

/// The gfx12 targets supporting WMMA matrix-core kernels
inline bool gfx_arch_is_gfx12_wmma(const char* gcn) {
    return gfx_arch_is_gfx12(gcn);
}

/// Compatibility alias: any gfx11 (RDNA3/RDNA3.5)
inline bool gfx_arch_is_gfx1151(const char* gcn) {
    return gfx_arch_is_gfx11(gcn);
}

}  // namespace strata::kernels

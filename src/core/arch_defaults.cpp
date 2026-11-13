// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"
#include "strata/core/defaults_026.hpp"
#include "strata/core/amd_community.hpp"
#include "strata/core/gfx12_extra.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    return arch_default_env(gcn_arch, strata::qwen_defaults_026());
}

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch, bool defaults026) {
#if defined(_WIN32)
    return arch_default_env(gcn_arch, defaults026, true);
#else
    return arch_default_env(gcn_arch, defaults026, false);
#endif
}

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch, bool defaults026, bool windows) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): the PROMPT half of the gfx1151 table (reported exact on an R9700 in #1478, re-checked
        // bitwise in notes-142-amd.md) plus four decode switches measured one at a time on an R9700 in 0.2.7 (p620, IQ3_S,
        // engine 0.1.51, 8K prompt, rotated pairs): ATTN_LANECELL + ATTN_MERGE_V2 +1.9%, TSUM +1.9%, EXPDEC +0.6%, GDN_SPLIT
        // 0.0%; the four together +3.5% (3/3 pairs) with the answer md5 identical to the default.  The rest of the gfx1151
        // decode half stays out (PF_GEMM/PF_FUSED are bit-changing; Q8_PACKED, SH_STREAM and QFUSE were measured on other
        // cards only, and the whole half measured slower here when it was tried as one block).
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
        };
        if (defaults026) return t;   // STRATA_DEFAULTS=026: v0.2.6's table, the prompt half only
        // decode (bit-identical on the R9700 measurement above)
        t.push_back({"STRATA_ATTN_LANECELL", "1"});
        t.push_back({"STRATA_ATTN_MERGE_V2", "1"});
        t.push_back({"STRATA_TSUM", "1"});
        t.push_back({"STRATA_EXPDEC", "1"});
        // AMD-Q (p620, R9700, IQ3_S, serve, interleaved pairs): the constant-trip loads of Blackwell's BW-DENSE (bit 0 the router-shaped bf16 read,
        // bit 1 the GDN a/b projection) and two load-ordering variants of the Q8 hyper-connection read (2 rows a warp loaded together in the
        // up read, the down read's staging loads of all tokens together); each is the same arithmetic in the same order (kernel hashes equal).
        t.push_back({"STRATA_SMALL_UNR", "3"});
        t.push_back({"STRATA_GR_Q32_G", "2"});
        t.push_back({"STRATA_GR_Q8_PFS", "1"});
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx11(gcn_arch)) return t;
    const char* off11 = std::getenv("STRATA_GFX11_DEFAULTS");
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if ((off11 != nullptr && off11[0] == '0') || (off != nullptr && off[0] == '0')) return t;
    (void) windows;
    // Verified GFX11 (RDNA3 / RDNA3.5) defaults for MXFP4 + Q4_0 KV (applies to both dGPU and iGPU):
    t = {
        {"STRATA_HIP_WMMA", "1"},
        {"STRATA_PF_GEMM", "1"},
        {"STRATA_PA_FAST", "1"},
        {"STRATA_PA_TRIM", "0"},
        {"STRATA_PF_FUSED", "1"},
        {"STRATA_PF_FUSED_MXFP4", "1"},
        {"STRATA_PF_F16", "1"},
        {"STRATA_PF_PAD", "1"},
        {"STRATA_PF_HCDOWN", "1"},
        {"STRATA_HCD_EXACT", "1"},
        {"STRATA_SELECT_WMMA", "1"},
        {"STRATA_HC_UPMIX", "1"},
        {"STRATA_HC_Q8", "1"},
        {"STRATA_GDN_HEAD", "1"},
        {"STRATA_GDN_PP", "2"},
        {"STRATA_GDN_CONVL2", "1"},
        {"STRATA_GDN_NOY", "1"},
        {"STRATA_GDN_CHUNKED", "1"},
        {"STRATA_SH_STREAM", "0"},
        {"STRATA_WMMA_GEMM", "0"},
        {"STRATA_MXFP4_FP8", "0"},
        {"STRATA_FP8_WMMA", "0"},
    };
    if (defaults026) return t;
    t.push_back({"STRATA_ATTN_LANECELL", "1"});
    t.push_back({"STRATA_ATTN_MERGE_V2", "1"});
    t.push_back({"STRATA_TSUM", "1"});
    t.push_back({"STRATA_EXPDEC", "1"});
    t.push_back({"STRATA_GDN_SPLIT", "1"});
    t.push_back({"STRATA_Q8_PACKED", "1"});
    t.push_back({"STRATA_QFUSE", "1"});
    t.push_back({"STRATA_PIPED_MTP", "1"});
    return t;
}

bool gfx12_prefill_share_default(const char* gcn_arch, bool defaults026) {
    if (defaults026 || !strata::kernels::gfx_arch_is_gfx12(gcn_arch)) return false;
    const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
    return !(off12 != nullptr && off12[0] == '0');
}

bool gfx12_prefill_share_default(const char* gcn_arch) {
    return gfx12_prefill_share_default(gcn_arch, strata::qwen_defaults_026());
}

bool igpu_prompt_defaults_arch(const char* gcn_arch) {
    return strata::kernels::gfx_arch_is_gfx11(gcn_arch) || strata::kernels::gfx_arch_is_gfx12(gcn_arch);
}

bool igpu_prompt_set_enabled(const char* gcn_arch, bool windows) {
    const char* off = std::getenv("STRATA_IGPU_DEFAULTS");
    if (off != nullptr && off[0] == '0') return false;
    const char* e = std::getenv("STRATA_IGPU_PROMPT_SET");
    if (e != nullptr && e[0] == '1') return true;
    if (e != nullptr && e[0] == '0') return false;
    if (strata::kernels::gfx_arch_is_gfx11(gcn_arch) || strata::kernels::gfx_arch_is_gfx12(gcn_arch)) return true;
    return kIgpuPromptSetDefaultOn && !windows && strata::kernels::gfx_arch_is_gfx11(gcn_arch);
}

bool igpu_uma_arena_default(const char* gcn_arch, bool integrated, bool windows, bool on) {
    if (!integrated) return false;
    const char* reg = std::getenv("STRATA_UMA_REGISTER");
    if (reg != nullptr && reg[0] == '1') return false;
    if (windows) return strata::kernels::gfx_arch_is_gfx11(gcn_arch);
    return on && strata::kernels::gfx_arch_is_gfx11(gcn_arch);
}

std::vector<std::pair<std::string, std::string>> igpu_default_env(const char* gcn_arch, bool integrated) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!integrated || !igpu_prompt_defaults_arch(gcn_arch)) return t;
#if defined(_WIN32)
    const bool win = true;
#else
    const bool win = false;
#endif
    if (!igpu_prompt_set_enabled(gcn_arch, win)) return t;
    // Measured on a Radeon 780M (gfx1103, P2006, Q2_0, notes-153-igpu.md / notes-148-amd.md): prompts 109.5 -> 214.7 tok/s
    // with the first four, 233 with HC_UPMIX, decode unchanged.  Dense projections are 63% of a prompt's GPU time there.
    t = {
        {"STRATA_HIP_WMMA", "1"},      // the prompt attention on matrix cores
        {"STRATA_PF_GEMM", "1"},       // the prompt's FP16 projections on the 128 x 256 WMMA GEMM
        {"STRATA_PA_FAST", "1"},       // the prompt attention with single FP16 q and p
        {"STRATA_SELECT_WMMA", "1"},   // the block scorer on matrix cores
        {"STRATA_HC_UPMIX", "1"},      // the hyper-connection up projection with its epilogue
    };
    // gfx1151 (Strix Halo) on Windows: SELECT_WMMA is in since 0.2.9 (x1.008 at 8K, x1.037 at 32K, x1.077 at 64K), and PF_FUSED (the
    // fused int8 gate/up prompt experts, #136) measured x1.12 there and joins the set.  The 780M and the 890M keep the
    // table above exactly as measured on them.
    if (strata::kernels::gfx_arch_is_gfx11(gcn_arch)) {
        // Integrated UMA GFX11 GPUs have large shared VRAM pools: cache padded dense FP16 weights
        // (gemm.cu enforces an 8 GiB free-VRAM reserve). Discrete GPUs leave this off to preserve --kv-grow VRAM.
        t.push_back({"STRATA_PF_W16_CACHE_MIB", "16384"});
    }
    return t;
}

static void set_default(const std::pair<std::string, std::string>& kv) {
#if defined(_WIN32)
    _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
    setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
}

std::vector<std::pair<std::string, std::string>> parallel_default_env(const char* gcn_arch, bool windows, int batch_slots) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!windows || batch_slots < 2 || !strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    t = {
        {"STRATA_CKPT_POOL", "1"},
        {"STRATA_CKPT_POOL_READY", "4"},
        {"STRATA_SLOT_CKPT_MOVE", "1"},
        {"STRATA_SLOT_COPY_DIRECT", "1"},
        {"STRATA_SLOT_COPY_BATCHSYNC", "1"},
        {"STRATA_BATCH_GDN_GROUPS", "1"},
        {"STRATA_BATCH_ATTN_ROWS", "1"},
        {"STRATA_BATCH_SELF_COMMIT", "1"},
        {"STRATA_QSA_SLOTS", "1"},
    };
    return t;
}

std::vector<std::pair<std::string, std::string>> launch_default_env(const char* gcn_arch, bool windows) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!windows || !strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    t = {{"STRATA_VERIFY_ASYNC_LAUNCH", "1"}};
    return t;
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch, bool integrated, int batch_slots) {
    std::vector<std::string> set;
    // the community-made gfx1010 / gfx1030 changes (amd_community.hpp): one line, and only on those two cards
    if (const std::string note = strata::amd_community::startup_line(gcn_arch); !note.empty())
        std::fprintf(stderr, "%s\n", note.c_str());
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
        set_default(kv);
        set.push_back(kv.first);
    }
    {   // gfx12 extras kept apart from the table (gfx12_extra.hpp); not counted in the returned list
        std::vector<std::string> extra;
        for (const auto& kv : gfx12_exact_extra_env(gcn_arch)) {
            if (std::getenv(kv.first.c_str()) != nullptr) continue;
            set_default(kv);
            extra.push_back(kv.first);
        }
        if (!extra.empty()) {
            std::fprintf(stderr, "strata: gfx12 (RDNA4): exact extra switches on by default (STRATA_GFX12_DEFAULTS=0 turns them off; a switch you set is kept): ");
            for (size_t i = 0; i < extra.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", extra[i].c_str() + 7);
            std::fprintf(stderr, "\n");
        }
    }
    {   // gfx12 defaults that change the answers (gfx12_extra.hpp): the matrix-core QSA scorer from 28,672 tokens; its own line
        const auto vc = gfx12_value_changing_env(gcn_arch);
        bool any = false;
        for (const auto& kv : vc) {
            if (std::getenv(kv.first.c_str()) != nullptr) continue;
            set_default(kv);
            any = true;
        }
        const char* wm = std::getenv("STRATA_SELECT_WMMA");   // the line is only true while the scorer switch is not set to 0
        if (any && !(wm != nullptr && wm[0] == '0'))
            std::fprintf(stderr, "strata: gfx12 (RDNA4): the matrix-core QSA block scorer is on once a prompt is past 28,672 tokens (+5.7%% prompt speed at 98K on an R9700, shorter prompts unchanged; "
                                 "the first-token KL against the warp kernel is 0.002 on average above 30K; STRATA_SELECT_WMMA=0 or STRATA_GFX12_DEFAULTS=0 turns it off)\n");
    }
    if (!set.empty()) {
        const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : "gfx11 (RDNA3)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : "STRATA_GFX11_DEFAULTS");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    std::vector<std::string> igpu;
    for (const auto& kv : igpu_default_env(gcn_arch, integrated)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        igpu.push_back(kv.first);
    }
    if (!igpu.empty()) {
        std::fprintf(stderr, "strata: integrated GPU (%s): the matrix-core prompt set is on (1.6x to 2x faster prompts on a "
                             "Radeon 780M, the answers differ in the last bits): ", gcn_arch);
        for (size_t i = 0; i < igpu.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", igpu[i].c_str() + 7);
        std::fprintf(stderr, ".  Turn one off with NAME=0 (for example STRATA_HC_UPMIX=0), all of them with STRATA_IGPU_PROMPT_SET=0");
#if defined(_WIN32)
        std::fprintf(stderr, ".  Not yet tested on Windows");
#endif
        std::fprintf(stderr, "\n");
        set.insert(set.end(), igpu.begin(), igpu.end());
    }
    std::vector<std::string> par;
#if defined(_WIN32)
    const bool win_host = true;
#else
    const bool win_host = false;
#endif
    for (const auto& kv : parallel_default_env(gcn_arch, win_host, batch_slots)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        par.push_back(kv.first);
    }
    if (!par.empty()) {
        std::fprintf(stderr, "strata: gfx1151 (Strix Halo): the exact parallel-request switches are on for %d batch slots (STRATA_GFX1151_DEFAULTS=0 turns them off; a switch you set is kept): ", batch_slots);
        for (size_t i = 0; i < par.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", par[i].c_str() + 7);
        std::fprintf(stderr, "\n");
        set.insert(set.end(), par.begin(), par.end());
    }
    std::vector<std::string> lau;
    for (const auto& kv : launch_default_env(gcn_arch, win_host)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        lau.push_back(kv.first);
    }
    if (!lau.empty()) {
        std::fprintf(stderr, "strata: gfx1151 (Strix Halo): the verify window's graph is launched from a helper thread (the main thread serves the doorbells meanwhile; STRATA_GFX1151_DEFAULTS=0 or STRATA_VERIFY_ASYNC_LAUNCH=0 turns it off)\n");
        set.insert(set.end(), lau.begin(), lau.end());
    }
    return set;
}

}  // namespace strata::core

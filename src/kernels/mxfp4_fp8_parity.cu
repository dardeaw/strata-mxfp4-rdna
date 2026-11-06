// src/kernels/mxfp4_fp8_parity.cu - the STRATA_MXFP4_FP8 prompt path against host references, synthetic data:
//   - iq_dequant_mxfp4_fp8: every E2M1 * E8M0 value -> the E4M3 code the host conversion gives (byte exact);
//   - quantize_fp8_rows: the per-row scale and the E4M3 codes of x / scale;
//   - fp8_gemm: a 32x32-tile FP8 WMMA product (T = 40 and the partial T = 17) against an FP64 host dot.
// No model and no GPU memory beyond the buffers below.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strata/artifact/dequant.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/fp8_gemm.hpp"
#include "strata/prefill/mxfp4_gemm.hpp"

namespace {

int failures = 0;

#define CHECK(cond, ...)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                     \
            std::printf(__VA_ARGS__);                                            \
            std::printf("\n");                                                   \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

uint8_t ref_e4m3(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    const uint8_t sign = (uint8_t) ((u >> 24) & 0x80u);
    const float a = std::fabs(v);
    if (a != a) return (uint8_t) (sign | 0x7Fu);
    if (a >= 448.0f) return (uint8_t) (sign | 0x7Eu);
    if (a < (1.0f / 1024.0f)) return sign;
    int e = 0;
    const float m = std::frexp(a, &e);
    int ex = e - 1;
    int mant = (int) std::lrint((m * 2.0f - 1.0f) * 8.0f);
    if (mant == 8) { mant = 0; ++ex; }
    if (ex >= -6) {
        if (ex > 8 || (ex == 8 && mant > 6)) return (uint8_t) (sign | 0x7Eu);
        return (uint8_t) (sign | ((ex + 7) << 3) | mant);
    }
    const int sub = (int) std::lrint(a * 512.0f);
    if (sub >= 8) return (uint8_t) (sign | 0x08u);
    return (uint8_t) (sign | sub);
}

float dec_e4m3(uint8_t c) {
    const int s = (c >> 7) & 1, e = (c >> 3) & 0xF, m = c & 7;
    if (e == 15 && m == 7) return 0.0f;   // NaN never appears in the generated data
    const float v = e == 0 ? (m / 8.0f) * (1.0f / 64.0f) : (1.0f + m / 8.0f) * std::ldexp(1.0f, e - 7);
    return s ? -v : v;
}

void* dev_alloc(size_t n, const char* what) {
    void* p = nullptr;
    const hipError_t e = hipMalloc(&p, n);
    CHECK(e == hipSuccess, "%s: hipMalloc(%zu): %s", what, n, hipGetErrorString(e));
    return p;
}

void test_dequant() {
    const int nblk = 256;   // 256 MXFP4 blocks = 8192 values = 32 launches' worth of 256 values
    std::vector<uint8_t> src((size_t) nblk * 17), dst((size_t) nblk * 32), want((size_t) nblk * 32);
    std::mt19937 rng(1234);
    for (int b = 0; b < nblk; ++b) {
        src[(size_t) b * 17] = (uint8_t) (100 + (rng() % 45));   // E8M0 exponents across the normal range
        for (int j = 0; j < 16; ++j) src[(size_t) b * 17 + 1 + j] = (uint8_t) (rng() & 0xFF);
    }
    for (int b = 0; b < nblk; ++b) {
        float out[32];
        strata::dequantize_mxfp4(src.data() + (size_t) b * 17, out);
        for (int j = 0; j < 32; ++j) want[(size_t) b * 32 + j] = ref_e4m3(out[j]);
    }
    uint8_t* ds = (uint8_t*) dev_alloc(src.size(), "dequant src");
    uint8_t* dd = (uint8_t*) dev_alloc(dst.size(), "dequant dst");
    CHECK(hipMemcpy(ds, src.data(), src.size(), hipMemcpyHostToDevice) == hipSuccess, "dequant upload");
    strata::kernels::iq_dequant_mxfp4_fp8(39, ds, (int64_t) nblk * 32, dd, nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess, "dequant sync");
    CHECK(hipMemcpy(dst.data(), dd, dst.size(), hipMemcpyDeviceToHost) == hipSuccess, "dequant download");
    int bad = 0;
    for (size_t i = 0; i < dst.size(); ++i) {
        if (dst[i] != want[i]) {
            if (bad < 8) std::printf("dequant[%zu]: got 0x%02x want 0x%02x\n", i, dst[i], want[i]);
            ++bad;
        }
    }
    CHECK(bad == 0, "dequant mismatches: %d / %zu", bad, dst.size());
    std::printf("dequant: %zu values, %d mismatches\n", dst.size(), bad);
    hipFree(ds); hipFree(dd);
}

void test_quant() {
    const int rows = 4, cols = 2560;
    std::vector<_Float16> x((size_t) rows * cols);
    std::vector<uint8_t> y((size_t) rows * cols);
    std::vector<float> sc(rows), want_sc(rows);
    std::vector<uint8_t> want((size_t) rows * cols);
    std::mt19937 rng(4321);
    for (int r = 0; r < rows; ++r) {
        float m = 0;
        for (int i = 0; i < cols; ++i) {
            const float v = ((int) (rng() % 2001) - 1000) / 125.0f;   // [-8, 8]
            x[(size_t) r * cols + i] = (_Float16) v;
            m = std::fmax(m, std::fabs(v));
        }
        const float s = m > 0 ? m / 448.0f : 1.0f;
        const float inv = 1.0f / s;
        want_sc[r] = s;
        for (int i = 0; i < cols; ++i) want[(size_t) r * cols + i] = ref_e4m3((float) x[(size_t) r * cols + i] * inv);
    }
    _Float16* dx = (_Float16*) dev_alloc(x.size() * 2, "quant x");
    uint8_t* dy = (uint8_t*) dev_alloc(y.size(), "quant y");
    float* dsc = (float*) dev_alloc(rows * sizeof(float), "quant scale");
    CHECK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice) == hipSuccess, "quant upload");
    strata::kernels::quantize_fp8_rows(dx, rows, cols, dy, dsc, nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess, "quant sync");
    CHECK(hipMemcpy(y.data(), dy, y.size(), hipMemcpyDeviceToHost) == hipSuccess, "quant y download");
    CHECK(hipMemcpy(sc.data(), dsc, rows * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess, "quant scale download");
    for (int r = 0; r < rows; ++r) {
        CHECK(std::fabs(sc[r] - want_sc[r]) <= 1e-7f * std::fmax(1.0f, want_sc[r]), "quant scale[%d]: %.9g vs %.9g", r, sc[r], want_sc[r]);
        for (int i = 0; i < cols; ++i) {
            CHECK(y[(size_t) r * cols + i] == want[(size_t) r * cols + i], "quant[%d][%d]: 0x%02x vs 0x%02x", r, i,
                  y[(size_t) r * cols + i], want[(size_t) r * cols + i]);
        }
    }
    std::printf("quant: %d rows x %d cols checked\n", rows, cols);
    hipFree(dx); hipFree(dy); hipFree(dsc);
}

void test_gemm(int T, int N = 1280, int K = 2560) {
    std::vector<uint8_t> X((size_t) T * K), W((size_t) N * K);
    std::vector<float> rs(T), Y((size_t) T * N), ref((size_t) T * N);
    std::mt19937 rng(77 + T);
    for (auto& v : X) v = ref_e4m3(((int) (rng() % 2001) - 1000) / 125.0f);
    for (auto& v : W) v = ref_e4m3(((int) (rng() % 2001) - 1000) / 125.0f);
    for (int t = 0; t < T; ++t) rs[t] = 0.25f + 0.01f * t;
    uint8_t* dX = (uint8_t*) dev_alloc(X.size(), "gemm X");
    uint8_t* dW = (uint8_t*) dev_alloc(W.size(), "gemm W");
    float* dY = (float*) dev_alloc(Y.size() * 4, "gemm Y");
    float* dR = (float*) dev_alloc(T * sizeof(float), "gemm rs");
    CHECK(hipMemcpy(dX, X.data(), X.size(), hipMemcpyHostToDevice) == hipSuccess, "gemm X upload");
    CHECK(hipMemcpy(dW, W.data(), W.size(), hipMemcpyHostToDevice) == hipSuccess, "gemm W upload");
    CHECK(hipMemcpy(dR, rs.data(), T * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "gemm rs upload");
    const bool ok = strata::prefill::fp8_gemm(dX, dW, dY, T, N, K, N, dR, nullptr);
    CHECK(ok, "fp8_gemm(T=%d) returned false", T);
    CHECK(hipDeviceSynchronize() == hipSuccess, "gemm sync");
    CHECK(hipMemcpy(Y.data(), dY, Y.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "gemm download");
    if (ok) {
        double worst = 0;
        int bad = 0;
        for (int t = 0; t < T; ++t) {
            for (int n = 0; n < N; ++n) {
                double s = 0;
                for (int k = 0; k < K; ++k) s += (double) dec_e4m3(X[(size_t) t * K + k]) * dec_e4m3(W[(size_t) n * K + k]);
                const float want = (float) (s * rs[t]);
                const double err = std::fabs((double) Y[(size_t) t * N + n] - want);
                const double rel = err / std::fmax(1.0, std::fabs((double) want));
                worst = std::fmax(worst, rel);
                if (rel > 2e-4) {
                    if (bad < 8) std::printf("gemm T=%d [%d][%d]: got %.9g want %.9g rel %.3g\n", T, t, n, Y[(size_t) t * N + n], want, rel);
                    ++bad;
                }
            }
        }
        CHECK(bad == 0, "gemm T=%d: %d cells over 2e-4 (worst %.3g)", T, bad, worst);
        std::printf("gemm T=%d: %d x %d x %d, worst rel %.3g, bad %d\n", T, T, N, K, worst, bad);
    }
    hipFree(dX); hipFree(dW); hipFree(dY); hipFree(dR);
}

void bench_gemm(int T, int N, int K, int reps) {
    std::vector<uint8_t> X((size_t) T * K, 0x38), W((size_t) N * K, 0x38);   // all 1.0, no NaNs, timing only
    std::vector<float> rs(T, 1.0f);
    uint8_t* dX = (uint8_t*) dev_alloc(X.size(), "bench X");
    uint8_t* dW = (uint8_t*) dev_alloc(W.size(), "bench W");
    float* dY = (float*) dev_alloc((size_t) T * N * 4, "bench Y");
    float* dR = (float*) dev_alloc(T * sizeof(float), "bench rs");
    CHECK(hipMemcpy(dX, X.data(), X.size(), hipMemcpyHostToDevice) == hipSuccess, "bench X upload");
    CHECK(hipMemcpy(dW, W.data(), W.size(), hipMemcpyHostToDevice) == hipSuccess, "bench W upload");
    CHECK(hipMemcpy(dR, rs.data(), T * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "bench rs upload");
    for (int i = 0; i < 3; ++i) strata::prefill::fp8_gemm(dX, dW, dY, T, N, K, N, dR, nullptr);
    CHECK(hipDeviceSynchronize() == hipSuccess, "bench warmup sync");
    // STRATA_BENCH_COLD=1: evict the L2 (256MB trash memset) before every rep, so W/X read
    // cold like the end-to-end path (fresh dequant writes per expert per layer) instead of L2-hot.
    const bool cold = std::getenv("STRATA_BENCH_COLD") != nullptr;
    uint8_t* trash = nullptr;
    if (cold) CHECK(hipMalloc((void**) &trash, 96ull << 20) == hipSuccess, "bench trash alloc");
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a);
    for (int i = 0; i < reps; ++i) {
        if (cold) CHECK(hipMemset(trash, i & 0xff, 96ull << 20) == hipSuccess, "bench trash evict");
        strata::prefill::fp8_gemm(dX, dW, dY, T, N, K, N, dR, nullptr);
    }
    if (trash) CHECK(hipFree(trash) == hipSuccess, "bench trash free");
    hipEventRecord(b);
    CHECK(hipEventSynchronize(b) == hipSuccess, "bench sync");
    float ms = 0;
    hipEventElapsedTime(&ms, a, b);
    const double tflops = 2.0 * T * N * K * reps / (ms * 1e-3) / 1e12;
    std::printf("bench T=%d N=%d K=%d: %.3f ms/call, %.2f TFLOPS\n", T, N, K, ms / reps, tflops);
    hipEventDestroy(a); hipEventDestroy(b);
    hipFree(dX); hipFree(dW); hipFree(dY); hipFree(dR);
}


void test_mxfp4_gemm() {
    const int n_ff = 64, n_embd = 256;
    const int Kgu = n_embd, Ngu = 2 * n_ff;
    const int Kd = n_ff, Nd = n_embd;
    const int gu_rb = (Kgu / 32) * 17, d_rb = (Kd / 32) * 17;
    std::vector<uint8_t> gate((size_t) n_ff * gu_rb), up((size_t) n_ff * gu_rb), down((size_t) n_embd * d_rb);
    std::mt19937 rng(0x5eed);
    auto fill = [&](std::vector<uint8_t>& w, int rows, int rb) {
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < rb / 17; ++b) {
                uint8_t* blk = w.data() + (size_t) r * rb + (size_t) b * 17;
                blk[0] = (uint8_t) (100 + (rng() % 40));
                for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t) (rng() & 0xFF);
            }
    };
    fill(gate, n_ff, gu_rb); fill(up, n_ff, gu_rb); fill(down, n_embd, d_rb);

    for (int T : {128, 33}) {
        std::vector<uint8_t> X((size_t) T * Kgu), Xd((size_t) T * Kd);
        std::vector<float> rs(T); for (int tt = 0; tt < T; ++tt) rs[tt] = 0.25f + 0.01f * tt;
        for (auto& v : X) v = ref_e4m3(((int) (rng() % 2001) - 1000) / 125.0f);
        for (auto& v : Xd) v = ref_e4m3(((int) (rng() % 2001) - 1000) / 125.0f);

        uint8_t* dX = (uint8_t*) dev_alloc(X.size(), "mx X");
        uint8_t* dXd = (uint8_t*) dev_alloc(Xd.size(), "mx Xd");
        uint8_t* dG = (uint8_t*) dev_alloc(gate.size(), "mx gate");
        uint8_t* dU = (uint8_t*) dev_alloc(up.size(), "mx up");
        uint8_t* dD = (uint8_t*) dev_alloc(down.size(), "mx down");
        float* dR = (float*) dev_alloc(T * sizeof(float), "mx rs");
        float* dY = (float*) dev_alloc((size_t) T * Nd * 4, "mx Y");
        CHECK(hipMemcpy(dX, X.data(), X.size(), hipMemcpyHostToDevice) == hipSuccess, "mx X up");
        CHECK(hipMemcpy(dXd, Xd.data(), Xd.size(), hipMemcpyHostToDevice) == hipSuccess, "mx Xd up");
        CHECK(hipMemcpy(dG, gate.data(), gate.size(), hipMemcpyHostToDevice) == hipSuccess, "mx gate up");
        CHECK(hipMemcpy(dU, up.data(), up.size(), hipMemcpyHostToDevice) == hipSuccess, "mx up up");
        CHECK(hipMemcpy(dD, down.data(), down.size(), hipMemcpyHostToDevice) == hipSuccess, "mx down up");
        CHECK(hipMemcpy(dR, rs.data(), T * sizeof(float), hipMemcpyHostToDevice) == hipSuccess, "mx rs up");

        // control: pre-dequant to an FP8 buffer, then the proven fp8_gemm.
        {
            uint8_t* dF = (uint8_t*) dev_alloc((size_t) Ngu * Kgu, "mx F");
            strata::kernels::iq_dequant_gu_mxfp4_fp8(39, dG, dU, n_ff, n_embd, dF, nullptr);
            CHECK(hipDeviceSynchronize() == hipSuccess, "mx predequant sync");
            {
                std::vector<uint8_t> Frow((size_t) Ngu * Kgu);
                CHECK(hipMemcpy(Frow.data(), dF, Frow.size(), hipMemcpyDeviceToHost) == hipSuccess, "mx Frow");
                int fb = 0;
                for (int n = 0; n < Ngu; ++n) {
                    const int src = n >> 1;
                    const std::vector<uint8_t>& w = (n & 1) ? up : gate;
                    const uint8_t* row = w.data() + (size_t) src * gu_rb;
                    for (int k = 0; k < Kgu; ++k) {
                        float vals[32];
                        strata::dequantize_mxfp4(row + (size_t) (k / 32) * 17, vals);
                        const uint8_t want = ref_e4m3(vals[k % 32]);
                        if (Frow[(size_t) n * Kgu + k] != want) {
                            if (fb < 6) std::printf("Frow n=%d k=%d got 0x%02x want 0x%02x\n", n, k, Frow[(size_t) n * Kgu + k], want);
                            ++fb;
                        }
                    }
                }
                std::printf("Frow: %d mismatches / %d\n", fb, Ngu * Kgu);
            }
            CHECK(strata::prefill::fp8_gemm(dX, dF, dY, T, Ngu, Kgu, Ngu, dR, nullptr), "mx predequant gemm T=%d", T);
            CHECK(hipDeviceSynchronize() == hipSuccess, "mx predequant gemm sync");
            std::vector<float> Yp((size_t) T * Ngu);
            CHECK(hipMemcpy(Yp.data(), dY, Yp.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "mx predequant down");
            int pb = 0;
            double pw = 0;
            for (int n = 0; n < Ngu; ++n) {
                const int src = n >> 1;
                const std::vector<uint8_t>& w = (n & 1) ? up : gate;
                const uint8_t* row = w.data() + (size_t) src * gu_rb;
                for (int t = 0; t < T; ++t) {
                    double s = 0;
                    for (int k = 0; k < Kgu; ++k) {
                        float vals[32];
                        strata::dequantize_mxfp4(row + (size_t) (k / 32) * 17, vals);
                        s += (double) dec_e4m3(X[(size_t) t * Kgu + k]) * (double) ref_e4m3(vals[k % 32]);
                    }
                    const float want = (float) (s * rs[t]);
                    const float got = Yp[(size_t) t * Ngu + n];
                    const double rel = std::fabs((double) got - want) / std::fmax(1.0, std::fabs((double) want));
                    pw = std::fmax(pw, rel);
                    if (rel > 2e-4) ++pb;
                }
            }
            CHECK(pb == 0, "mx predequant T=%d: %d bad (worst %.2g)", T, pb, pw);
            std::printf("mx predequant T=%d: worst rel %.2g, bad %d\n", T, pw, pb);
            hipFree(dF);
        }

        // Control: same dX/dF/rs, the proven fp8_gemm; compare its output with the in-kernel result.
        std::vector<float> Yref((size_t) T * Ngu);
        {
            uint8_t* dF2 = (uint8_t*) dev_alloc((size_t) Ngu * Kgu, "mx F2");
            strata::kernels::iq_dequant_gu_mxfp4_fp8(39, dG, dU, n_ff, n_embd, dF2, nullptr);
            CHECK(hipDeviceSynchronize() == hipSuccess, "mx ref sync");
            CHECK(strata::prefill::fp8_gemm(dX, dF2, dY, T, Ngu, Kgu, Ngu, dR, nullptr), "mx ref gemm T=%d", T);
            CHECK(hipDeviceSynchronize() == hipSuccess, "mx ref gemm sync");
            CHECK(hipMemcpy(Yref.data(), dY, Yref.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "mx ref down");
            hipFree(dF2);
        }
        CHECK(strata::prefill::mxfp4_gemm_gu(dX, dG, dU, dY, T, Ngu, Kgu, Ngu, dR, nullptr), "mxfp4 gu T=%d", T);
        CHECK(hipDeviceSynchronize() == hipSuccess, "mx gu sync");
        std::vector<float> Ygu((size_t) T * Ngu);
        CHECK(hipMemcpy(Ygu.data(), dY, Ygu.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "mx gu down");
        int bad = 0;
        double worst = 0;
        for (int n = 0; n < Ngu; ++n) {
            const int src = n >> 1;
            const std::vector<uint8_t>& w = (n & 1) ? up : gate;
            const uint8_t* row = w.data() + (size_t) src * gu_rb;
            for (int t = 0; t < T; ++t) {
                double s = 0;
                for (int k = 0; k < Kgu; ++k) {
                    float vals[32];
                    strata::dequantize_mxfp4(row + (size_t) (k / 32) * 17, vals);
                    s += (double) dec_e4m3(X[(size_t) t * Kgu + k]) * (double) ref_e4m3(vals[k % 32]);
                }
                const float want = (float) (s * rs[t]);
                const float got = Ygu[(size_t) t * Ngu + n];
                const double rel = std::fabs((double) got - want) / std::fmax(1.0, std::fabs((double) want));
                worst = std::fmax(worst, rel);
                if (rel > 2e-4) {
                    if (bad < 5) std::printf("mx gu T=%d [%d][%d] got %.6g want %.6g rel %.2g\n", T, t, n, got, want, rel);
                    ++bad;
                }
            }
        }
        CHECK(bad == 0, "mx gu T=%d: %d bad (worst %.2g)", T, bad, worst);
        std::printf("mx gu T=%d: worst rel %.2g, bad %d\n", T, worst, bad);
        {
            double md = 0;
            for (size_t i = 0; i < Ygu.size(); ++i) md = std::fmax(md, std::fabs((double) Ygu[i] - (double) Yref[i]));
            std::printf("mx in-kernel vs pre-dequant ref T=%d: max abs diff %.3g\n", T, md);
        }

        CHECK(strata::prefill::mxfp4_gemm_d(dXd, dD, dY, T, Nd, Kd, Nd, dR, nullptr), "mxfp4 d T=%d", T);
        CHECK(hipDeviceSynchronize() == hipSuccess, "mx d sync");
        std::vector<float> Yd((size_t) T * Nd);
        CHECK(hipMemcpy(Yd.data(), dY, Yd.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "mx d down");
        bad = 0;
        worst = 0;
        for (int n = 0; n < Nd; ++n) {
            const uint8_t* row = down.data() + (size_t) n * d_rb;
            for (int t = 0; t < T; ++t) {
                double s = 0;
                for (int k = 0; k < Kd; ++k) {
                    float vals[32];
                    strata::dequantize_mxfp4(row + (size_t) (k / 32) * 17, vals);
                    s += (double) dec_e4m3(Xd[(size_t) t * Kd + k]) * (double) ref_e4m3(vals[k % 32]);
                }
                const float want = (float) (s * rs[t]);
                const float got = Yd[(size_t) t * Nd + n];
                const double rel = std::fabs((double) got - want) / std::fmax(1.0, std::fabs((double) want));
                worst = std::fmax(worst, rel);
                if (rel > 2e-4) {
                    if (bad < 5) std::printf("mx d T=%d [%d][%d] got %.6g want %.6g rel %.2g\n", T, t, n, got, want, rel);
                    ++bad;
                }
            }
        }
        CHECK(bad == 0, "mx d T=%d: %d bad (worst %.2g)", T, bad, worst);
        std::printf("mx d T=%d: worst rel %.2g, bad %d\n", T, worst, bad);
        hipFree(dX); hipFree(dXd); hipFree(dG); hipFree(dU); hipFree(dD); hipFree(dR); hipFree(dY);
    }
}


// ---- Grouped in-kernel MXFP4 (zero global staging) vs pre-dequant grouped ----
// Real prefill shapes (n_embd = 2560): GU X[T,2560] -> Y[T,1280] (gate/up blobs
// interleaved as 2*r + parity), D H[T,640] -> Dm[T,2560]. A batch holds E experts,
// expert e owning chunk rows [e*T, e*T+T) of X / H and their row scales.
struct GroupedBatch {
    int T = 0, E = 0;
    uint8_t *dX = nullptr, *dH = nullptr;
    float *dXs = nullptr, *dHs = nullptr;
    uint8_t *dGate = nullptr, *dUp = nullptr, *dDown = nullptr;
    uint8_t *dFgu = nullptr, *dFd = nullptr;
    float *dYg0 = nullptr, *dYg1 = nullptr, *dYd0 = nullptr, *dYd1 = nullptr;
    strata::prefill::Fp8GroupTile *tFgu = nullptr, *tFd = nullptr;
    strata::prefill::Mxfp4ExpertTile *tMgu = nullptr, *tMd = nullptr;
    std::vector<strata::prefill::Fp8GroupTile> fgu, fd;
    std::vector<strata::prefill::Mxfp4ExpertTile> mgu, md;
};

GroupedBatch grouped_make(int T, int E) {
    GroupedBatch b;
    b.T = T; b.E = E;
    const int n_ff = 1280 / 2, Kgu = 2560, Ngu = 1280, Kd = 640, Nd = 2560;
    const int gu_rb = (Kgu / 32) * 17, d_rb = (Kd / 32) * 17;
    std::mt19937 rng(9107 + T * 131 + E);
    std::vector<uint8_t> gate((size_t) E * n_ff * gu_rb), up((size_t) E * n_ff * gu_rb),
        down((size_t) E * Nd * d_rb);
    auto fill_w = [&](std::vector<uint8_t>& w, int rows, int rb) {
        for (int r = 0; r < rows; ++r)
            for (int bb = 0; bb < rb / 17; ++bb) {
                uint8_t* blk = w.data() + (size_t) r * rb + (size_t) bb * 17;
                blk[0] = (uint8_t) (100 + (rng() % 40));
                for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t) (rng() & 0xFF);
            }
    };
    fill_w(gate, E * n_ff, gu_rb);
    fill_w(up, E * n_ff, gu_rb);
    fill_w(down, E * Nd, d_rb);
    std::vector<_Float16> Xh((size_t) E * T * Kgu), Hh((size_t) E * T * Kd);
    for (auto& v : Xh) v = (_Float16) (((int) (rng() % 2001) - 1000) / 125.0f);
    for (auto& v : Hh) v = (_Float16) (((int) (rng() % 2001) - 1000) / 125.0f);
    b.dX = (uint8_t*) dev_alloc((size_t) E * T * Kgu, "gb X");
    b.dH = (uint8_t*) dev_alloc((size_t) E * T * Kd, "gb H");
    b.dXs = (float*) dev_alloc((size_t) E * T * 4, "gb Xs");
    b.dHs = (float*) dev_alloc((size_t) E * T * 4, "gb Hs");
    b.dGate = (uint8_t*) dev_alloc(gate.size(), "gb gate");
    b.dUp = (uint8_t*) dev_alloc(up.size(), "gb up");
    b.dDown = (uint8_t*) dev_alloc(down.size(), "gb down");
    b.dFgu = (uint8_t*) dev_alloc((size_t) E * Ngu * Kgu, "gb Fgu");
    b.dFd = (uint8_t*) dev_alloc((size_t) E * Nd * Kd, "gb Fd");
    b.dYg0 = (float*) dev_alloc((size_t) E * T * Ngu * 4, "gb Yg0");
    b.dYg1 = (float*) dev_alloc((size_t) E * T * Ngu * 4, "gb Yg1");
    b.dYd0 = (float*) dev_alloc((size_t) E * T * Nd * 4, "gb Yd0");
    b.dYd1 = (float*) dev_alloc((size_t) E * T * Nd * 4, "gb Yd1");
    _Float16* dx = (_Float16*) dev_alloc(Xh.size() * 2, "gb Xh");
    _Float16* dh = (_Float16*) dev_alloc(Hh.size() * 2, "gb Hh");
    CHECK(hipMemcpy(dx, Xh.data(), Xh.size() * 2, hipMemcpyHostToDevice) == hipSuccess, "gb Xh up");
    CHECK(hipMemcpy(dh, Hh.data(), Hh.size() * 2, hipMemcpyHostToDevice) == hipSuccess, "gb Hh up");
    CHECK(hipMemcpy(b.dGate, gate.data(), gate.size(), hipMemcpyHostToDevice) == hipSuccess, "gb gate up");
    CHECK(hipMemcpy(b.dUp, up.data(), up.size(), hipMemcpyHostToDevice) == hipSuccess, "gb up up");
    CHECK(hipMemcpy(b.dDown, down.data(), down.size(), hipMemcpyHostToDevice) == hipSuccess, "gb down up");
    strata::kernels::quantize_fp8_rows(dx, (int64_t) E * T, Kgu, b.dX, b.dXs, nullptr);
    strata::kernels::quantize_fp8_rows(dh, (int64_t) E * T, Kd, b.dH, b.dHs, nullptr);
    hipFree(dx); hipFree(dh);
    const int64_t nGuN = Ngu / 64, nDN = Nd / 64, nM = (T + 255) / 256;
    for (int e = 0; e < E; ++e) {
        const uint8_t* G = b.dGate + (size_t) e * n_ff * gu_rb;
        const uint8_t* U = b.dUp + (size_t) e * n_ff * gu_rb;
        const uint8_t* D = b.dDown + (size_t) e * Nd * d_rb;
        const uint8_t* Fg = b.dFgu + (size_t) e * Ngu * Kgu;
        const uint8_t* Fd = b.dFd + (size_t) e * Nd * Kd;
        float* Yg0 = b.dYg0 + (size_t) e * T * Ngu;
        float* Yg1 = b.dYg1 + (size_t) e * T * Ngu;
        float* Yd0 = b.dYd0 + (size_t) e * T * Nd;
        float* Yd1 = b.dYd1 + (size_t) e * T * Nd;
        const int64_t ts = (int64_t) e * T;
        for (int64_t nt = 0; nt < nGuN; ++nt)
            for (int64_t mg = 0; mg < nM; ++mg) {
                b.fgu.push_back({Fg, Yg0, ts, T, nt, mg, Ngu, Kgu, Ngu});
                strata::prefill::Mxfp4ExpertTile t;
                t.blob = G; t.up = U; t.out = Yg1;
                t.t_start = ts; t.t_count = T; t.n_tile = nt; t.m_group = mg;
                t.N = Ngu; t.K = Kgu; t.ldy = Ngu; t.row_bytes = gu_rb;
                b.mgu.push_back(t);
            }
        for (int64_t nt = 0; nt < nDN; ++nt)
            for (int64_t mg = 0; mg < nM; ++mg) {
                b.fd.push_back({Fd, Yd0, ts, T, nt, mg, Nd, Kd, Nd});
                strata::prefill::Mxfp4ExpertTile t;
                t.blob = D; t.up = nullptr; t.out = Yd1;
                t.t_start = ts; t.t_count = T; t.n_tile = nt; t.m_group = mg;
                t.N = Nd; t.K = Kd; t.ldy = Nd; t.row_bytes = d_rb;
                b.md.push_back(t);
            }
    }
    b.tFgu = (strata::prefill::Fp8GroupTile*) dev_alloc(b.fgu.size() * sizeof(strata::prefill::Fp8GroupTile), "gb tFgu");
    b.tFd = (strata::prefill::Fp8GroupTile*) dev_alloc(b.fd.size() * sizeof(strata::prefill::Fp8GroupTile), "gb tFd");
    b.tMgu = (strata::prefill::Mxfp4ExpertTile*) dev_alloc(b.mgu.size() * sizeof(strata::prefill::Mxfp4ExpertTile), "gb tMgu");
    b.tMd = (strata::prefill::Mxfp4ExpertTile*) dev_alloc(b.md.size() * sizeof(strata::prefill::Mxfp4ExpertTile), "gb tMd");
    CHECK(hipMemcpy(b.tFgu, b.fgu.data(), b.fgu.size() * sizeof(strata::prefill::Fp8GroupTile), hipMemcpyHostToDevice) == hipSuccess, "gb tFgu up");
    CHECK(hipMemcpy(b.tFd, b.fd.data(), b.fd.size() * sizeof(strata::prefill::Fp8GroupTile), hipMemcpyHostToDevice) == hipSuccess, "gb tFd up");
    CHECK(hipMemcpy(b.tMgu, b.mgu.data(), b.mgu.size() * sizeof(strata::prefill::Mxfp4ExpertTile), hipMemcpyHostToDevice) == hipSuccess, "gb tMgu up");
    CHECK(hipMemcpy(b.tMd, b.md.data(), b.md.size() * sizeof(strata::prefill::Mxfp4ExpertTile), hipMemcpyHostToDevice) == hipSuccess, "gb tMd up");
    return b;
}

void grouped_free(GroupedBatch& b) {
    hipFree(b.dX); hipFree(b.dH); hipFree(b.dXs); hipFree(b.dHs);
    hipFree(b.dGate); hipFree(b.dUp); hipFree(b.dDown);
    hipFree(b.dFgu); hipFree(b.dFd);
    hipFree(b.dYg0); hipFree(b.dYg1); hipFree(b.dYd0); hipFree(b.dYd1);
    hipFree(b.tFgu); hipFree(b.tFd); hipFree(b.tMgu); hipFree(b.tMd);
}

// Reference: per-expert dequant to a global FP8 buffer, then the proven grouped GEMMs.
void grouped_run_pre(GroupedBatch& b) {
    const int n_ff = 640, Kgu = 2560, Ngu = 1280, Kd = 640, Nd = 2560;
    const int gu_rb = (Kgu / 32) * 17, d_rb = (Kd / 32) * 17;
    for (int e = 0; e < b.E; ++e) {
        strata::kernels::iq_dequant_gu_mxfp4_fp8(39, b.dGate + (size_t) e * n_ff * gu_rb,
            b.dUp + (size_t) e * n_ff * gu_rb, n_ff, Kgu, b.dFgu + (size_t) e * Ngu * Kgu, nullptr);
        strata::kernels::iq_dequant_mxfp4_fp8(39, b.dDown + (size_t) e * Nd * d_rb,
            (int64_t) Nd * Kd, b.dFd + (size_t) e * Nd * Kd, nullptr);
    }
    CHECK(strata::prefill::fp8_grouped_gemm(b.dX, b.dXs, b.tFgu, (int) b.fgu.size(), nullptr), "gb pre gu");
    CHECK(strata::prefill::fp8_grouped_gemm(b.dH, b.dHs, b.tFd, (int) b.fd.size(), nullptr), "gb pre d");
}

// Candidate: in-kernel LUT dequant, no global staging buffer.
void grouped_run_ink(GroupedBatch& b) {
    CHECK(strata::prefill::mxfp4_grouped_gemm_gu(b.dX, b.dXs, b.tMgu, (int) b.mgu.size(), nullptr), "gb ink gu");
    CHECK(strata::prefill::mxfp4_grouped_gemm_d(b.dH, b.dHs, b.tMd, (int) b.md.size(), nullptr), "gb ink d");
}

void test_mxfp4_grouped_parity(int T, int E) {
    GroupedBatch b = grouped_make(T, E);
    grouped_run_pre(b);
    CHECK(hipDeviceSynchronize() == hipSuccess, "gb pre sync");
    grouped_run_ink(b);
    CHECK(hipDeviceSynchronize() == hipSuccess, "gb ink sync");
    std::vector<float> g0((size_t) E * T * 1280), g1((size_t) E * T * 1280);
    std::vector<float> d0((size_t) E * T * 2560), d1((size_t) E * T * 2560);
    CHECK(hipMemcpy(g0.data(), b.dYg0, g0.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "gb g0 down");
    CHECK(hipMemcpy(g1.data(), b.dYg1, g1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "gb g1 down");
    CHECK(hipMemcpy(d0.data(), b.dYd0, d0.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "gb d0 down");
    CHECK(hipMemcpy(d1.data(), b.dYd1, d1.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "gb d1 down");
    double mg = 0, md = 0;
    for (size_t i = 0; i < g0.size(); ++i) mg = std::fmax(mg, std::fabs((double) g0[i] - (double) g1[i]));
    for (size_t i = 0; i < d0.size(); ++i) md = std::fmax(md, std::fabs((double) d0[i] - (double) d1[i]));
    std::printf("grouped parity T=%d E=%d: gu maxabs %.3g, d maxabs %.3g\n", T, E, mg, md);
    CHECK(mg == 0.0, "grouped gu T=%d E=%d maxabs %.3g", T, E, mg);
    CHECK(md == 0.0, "grouped d T=%d E=%d maxabs %.3g", T, E, md);
    grouped_free(b);
}

static float grouped_event_ms(hipEvent_t a, hipEvent_t b) {
    float ms = 0;
    CHECK(hipEventElapsedTime(&ms, a, b) == hipSuccess, "gb elapsed");
    return ms;
}

void bench_mxfp4_grouped(int T, int E) {
    GroupedBatch b = grouped_make(T, E);
    grouped_run_pre(b);
    grouped_run_ink(b);
    CHECK(hipDeviceSynchronize() == hipSuccess, "gb warmup sync");
    hipEvent_t a, ev;
    hipEventCreate(&a); hipEventCreate(&ev);
    const double flops = (double) E * 2.0 * T * (1280.0 * 2560.0 + 2560.0 * 640.0);
    const double flops_gu = (double) E * 2.0 * T * 1280.0 * 2560.0;
    const double flops_d = (double) E * 2.0 * T * 2560.0 * 640.0;
    float pre[3], ink[3], preg[3], inkg[3], pred[3], inkd[3];
    for (int i = 0; i < 3; ++i) {
        hipEventRecord(a);
        grouped_run_pre(b);
        hipEventRecord(ev);
        CHECK(hipEventSynchronize(ev) == hipSuccess, "gb pre sync");
        pre[i] = grouped_event_ms(a, ev);
        hipEventRecord(a);
        grouped_run_ink(b);
        hipEventRecord(ev);
        CHECK(hipEventSynchronize(ev) == hipSuccess, "gb ink sync");
        ink[i] = grouped_event_ms(a, ev);
        hipEventRecord(a);
        CHECK(strata::prefill::fp8_grouped_gemm(b.dX, b.dXs, b.tFgu, (int) b.fgu.size(), nullptr), "gb t preg");
        hipEventRecord(ev);
        CHECK(hipEventSynchronize(ev) == hipSuccess, "gb preg sync");
        preg[i] = grouped_event_ms(a, ev);
        hipEventRecord(a);
        CHECK(strata::prefill::mxfp4_grouped_gemm_gu(b.dX, b.dXs, b.tMgu, (int) b.mgu.size(), nullptr), "gb t inkg");
        hipEventRecord(ev);
        CHECK(hipEventSynchronize(ev) == hipSuccess, "gb inkg sync");
        inkg[i] = grouped_event_ms(a, ev);
        hipEventRecord(a);
        CHECK(strata::prefill::fp8_grouped_gemm(b.dH, b.dHs, b.tFd, (int) b.fd.size(), nullptr), "gb t pred");
        hipEventRecord(ev);
        CHECK(hipEventSynchronize(ev) == hipSuccess, "gb pred sync");
        pred[i] = grouped_event_ms(a, ev);
        hipEventRecord(a);
        CHECK(strata::prefill::mxfp4_grouped_gemm_d(b.dH, b.dHs, b.tMd, (int) b.md.size(), nullptr), "gb t inkd");
        hipEventRecord(ev);
        CHECK(hipEventSynchronize(ev) == hipSuccess, "gb inkd sync");
        inkd[i] = grouped_event_ms(a, ev);
    }
    auto med = [](float v[3]) { float x = v[0], y = v[1], z = v[2]; return x < y ? (y < z ? y : (x < z ? z : x)) : (x < z ? x : (y < z ? z : y)); };
    const float m_pre = med(pre), m_ink = med(ink);
    std::printf("grouped bench T=%d E=%d (3-run median):\n", T, E);
    std::printf("  full pre-dequant path : %.3f ms/call (%.2f TFLOPS)\n", m_pre, flops / (m_pre * 1e-3) / 1e12);
    std::printf("  full in-kernel path   : %.3f ms/call (%.2f TFLOPS)\n", m_ink, flops / (m_ink * 1e-3) / 1e12);
    std::printf("  speedup: %.2fx\n", m_pre / m_ink);
    std::printf("  gu launch only: pre %.3f ms vs ink %.3f ms | d launch only: pre %.3f ms vs ink %.3f ms\n",
        med(preg), med(inkg), med(pred), med(inkd));
    (void) flops_gu; (void) flops_d;
    hipEventDestroy(a); hipEventDestroy(ev);
    grouped_free(b);
}

}  // namespace

int main() {
    test_dequant();
    test_quant();
    test_mxfp4_gemm();
    test_mxfp4_grouped_parity(275, 8);
    test_mxfp4_grouped_parity(33, 3);
    bench_mxfp4_grouped(275, 8);
    bench_mxfp4_grouped(1024, 8);
    test_gemm(128, 128, 256);   // small shape, 32x32 kernel path
    test_gemm(33, 128, 256);    // small shape, 32x32 kernel, partial tile
    test_gemm(40);              // 32x32 kernel
    test_gemm(17);              // 32x32 kernel, partial tile
    test_gemm(128);             // 32x32 kernel (below the 4x4 threshold)
    test_gemm(256);             // 64x64 kernel, up-projection shape
    test_gemm(200, 2560, 640);  // 32x32 kernel, the down-projection shape
    bench_gemm(64, 1280, 2560, 20);
    bench_gemm(128, 1280, 2560, 20);
    bench_gemm(256, 1280, 2560, 10);
    bench_gemm(128, 2560, 640, 20);
    bench_gemm(512, 2560, 1280, 10);     // R2 sweep: gate, large T
    bench_gemm(1024, 2560, 1280, 10);
    bench_gemm(2048, 2560, 1280, 10);
    bench_gemm(4096, 2560, 1280, 5);
    bench_gemm(512, 640, 2560, 10);      // R2 sweep: up, large T
    bench_gemm(1024, 640, 2560, 10);
    bench_gemm(2048, 640, 2560, 10);
    bench_gemm(512, 2560, 640, 10);      // R2 sweep: down, large T
    bench_gemm(1024, 2560, 640, 10);
    bench_gemm(2048, 2560, 640, 10);
    bench_gemm(4096, 2560, 640, 5);
    if (failures != 0) {
        std::printf("FAILED: %d checks\n", failures);
        return 1;
    }
    std::printf("mxfp4_fp8_parity: OK\n");
    return 0;
}

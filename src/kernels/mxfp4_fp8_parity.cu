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
    hipEvent_t a, b;
    hipEventCreate(&a); hipEventCreate(&b);
    hipEventRecord(a);
    for (int i = 0; i < reps; ++i) strata::prefill::fp8_gemm(dX, dW, dY, T, N, K, N, dR, nullptr);
    hipEventRecord(b);
    CHECK(hipEventSynchronize(b) == hipSuccess, "bench sync");
    float ms = 0;
    hipEventElapsedTime(&ms, a, b);
    const double tflops = 2.0 * T * N * K * reps / (ms * 1e-3) / 1e12;
    std::printf("bench T=%d N=%d K=%d: %.3f ms/call, %.2f TFLOPS\n", T, N, K, ms / reps, tflops);
    hipEventDestroy(a); hipEventDestroy(b);
    hipFree(dX); hipFree(dW); hipFree(dY); hipFree(dR);
}

}  // namespace

int main() {
    test_dequant();
    test_quant();
    test_gemm(40);              // 32x32 kernel
    test_gemm(17);              // 32x32 kernel, partial tile
    test_gemm(128);             // 32x32 kernel (below the 4x4 threshold)
    test_gemm(256);             // 64x64 kernel, up-projection shape
    test_gemm(200, 2560, 640);  // 32x32 kernel, the down-projection shape
    bench_gemm(64, 1280, 2560, 20);
    bench_gemm(128, 1280, 2560, 20);
    bench_gemm(256, 1280, 2560, 10);
    bench_gemm(128, 2560, 640, 20);
    if (failures != 0) {
        std::printf("FAILED: %d checks\n", failures);
        return 1;
    }
    std::printf("mxfp4_fp8_parity: OK\n");
    return 0;
}

// Backend-level residual/adaptive cost with immutable quantized weights.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "krk/backend.hpp"
#include "krk/quant.hpp"

using namespace krk;
static volatile f32 sink;
static void put_f16(u8 *p, f32 v) {
    u16 h; std::memcpy(&h, &v, 2);
    p[0] = static_cast<u8>(h); p[1] = static_cast<u8>(h >> 8);
}

int main(int argc, char **argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 1;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 8;
    if ((rows != 1 && rows != 8) || reps <= 0) return 2;
    constexpr i64 n = 4096, n_out = 256, rb = 84;
    constexpr i64 row_bytes = rb * (n / 256);
    std::mt19937 rng(0xC0FFEE);
    std::vector<u8> w(n_out * row_bytes);
    for (u8 &v : w) v = static_cast<u8>(rng());
    for (i64 o = 0; o < n_out; o++) {
        for (i64 b = 0; b < n / 256; b++) {
            put_f16(w.data() + o * row_bytes + b * rb + 80, 0.017f);
            put_f16(w.data() + o * row_bytes + b * rb + 82, 0.007f);
        }
    }
    std::vector<f32> x(rows * n), out(rows * n_out);
    std::normal_distribution<f32> normal(0.0f, 1.0f);
    for (f32 &v : x) v = normal(rng);
    std::unique_ptr<Backend, void(*)(Backend *)> cpu(make_cpu_backend(), [](Backend *b){ delete b; });
    const ActivationRecoveryMode modes[] = {ActivationRecoveryMode::Off,
        ActivationRecoveryMode::Residual, ActivationRecoveryMode::Adaptive,
        ActivationRecoveryMode::Both};
    const char *names[] = {"off", "residual", "adaptive", "both"};
    for (int m = 0; m < 4; m++) {
        set_activation_recovery_mode(modes[m]);
        cpu->gemm(out.data(), x.data(), w.data(), DType::Q2_K, n_out, n, rows); // warm
        const auto begin = std::chrono::steady_clock::now();
        for (int rep = 0; rep < reps; rep++) {
            cpu->gemm(out.data(), x.data(), w.data(), DType::Q2_K, n_out, n, rows);
            sink += out[(rep * 31) % out.size()];
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count() / reps;
        std::printf("rows=%d %-8s %.3f ms/GEMM\n", rows, names[m], ms);
    }
    set_activation_recovery_mode(ActivationRecoveryMode::Off);
    std::fprintf(stderr, "sink=%g\n", sink);
    return 0;
}

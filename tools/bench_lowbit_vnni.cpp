// Isolated decode/prefill low-bit kernel throughput. Run separate processes
// with KRK_CPU_Q23_VNNI=0 / KRK_CPU_TERNARY_LANES=0 for baseline arms.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "krk/quant.hpp"

using namespace krk;
static volatile f32 sink;
static void put_f16(u8 *p, f32 v) {
    u16 h;
    std::memcpy(&h, &v, 2);
    p[0] = static_cast<u8>(h);
    p[1] = static_cast<u8>(h >> 8);
}

int main(int argc, char **argv) {
    const std::string type = argc > 1 ? argv[1] : "q2_K";
    const int rows = argc > 2 ? std::atoi(argv[2]) : 1;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 12;
    if (type != "q2_K" && type != "q3_K" && type != "tq1" && type != "tq2") {
        std::fprintf(stderr, "usage: %s q2_K|q3_K|tq1|tq2 [rows=1|8] [reps]\n", argv[0]);
        return 2;
    }
    if (rows != 1 && rows != 8) return 2;
    constexpr i64 n = 4096, n_out = 512;
    const bool k3 = type == "q3_K", t1 = type == "tq1", t2 = type == "tq2";
    const i64 rb = k3 ? 110 : (type == "q2_K" ? 84 : (t1 ? 54 : 66));
    const i64 blocks = n / (k3 || type == "q2_K" || t2 ? 256 : 256);
    const i64 row_bytes = rb * blocks;
    std::mt19937 rng(0xC0FFEE);
    std::vector<u8> w(static_cast<size_t>(row_bytes * n_out));
    for (u8 &v : w) v = static_cast<u8>(rng());
    for (i64 o = 0; o < n_out; o++) {
        for (i64 nb = 0; nb < blocks; nb++) {
            u8 *b = w.data() + o * row_bytes + nb * rb;
            if (k3) put_f16(b + 108, 0.019f);
            else if (type == "q2_K") { put_f16(b + 80, 0.017f); put_f16(b + 82, 0.007f); }
            else if (t1) put_f16(b + 52, 0.019f);
            else put_f16(b + 64, 0.019f);
        }
    }
    std::vector<f32> x(rows * n);
    std::normal_distribution<f32> normal(0.0f, 1.0f);
    for (f32 &v : x) v = normal(rng);
    std::vector<i8> xq(rows * n);
    std::vector<f32> dx(rows * n / 32);
    for (int r = 0; r < rows; r++)
        quantize_q8_row(x.data() + r * n, n, xq.data() + r * n, dx.data() + r * n / 32);
    std::vector<f32> out(rows * n_out);
    const auto start = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; rep++) {
        for (i64 o = 0; o < n_out; o++) {
            const u8 *b = w.data() + o * row_bytes;
            if (rows == 8) {
                if (k3) dot_q3_K_q8_x8(b, xq.data(), dx.data(), n, n, n / 32, out.data() + o, n_out);
                else if (type == "q2_K") dot_q2_K_q8_x8(b, xq.data(), dx.data(), n, n, n / 32, out.data() + o, n_out);
                else if (t1) dot_tq1_0_q8_x8(b, xq.data(), dx.data(), n, n, n / 32, out.data() + o, n_out);
                else dot_tq2_0_q8_x8(b, xq.data(), dx.data(), n, n, n / 32, out.data() + o, n_out);
            } else {
                if (k3) out[o] = dot_q3_K_q8(b, xq.data(), dx.data(), n);
                else if (type == "q2_K") out[o] = dot_q2_K_q8(b, xq.data(), dx.data(), n);
                else if (t1) out[o] = dot_tq1_0_q8(b, xq.data(), dx.data(), n);
                else out[o] = dot_tq2_0_q8(b, xq.data(), dx.data(), n);
            }
        }
        sink += out[(rep * 97) % out.size()];
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double gmac = static_cast<double>(rows) * n_out * n * reps / sec / 1e9;
    std::printf("%s rows=%d reps=%d %.3f ms/rep %.3f GMAC/s sink=%g\n",
                type.c_str(), rows, reps, sec * 1000.0 / reps, gmac, sink);
    return 0;
}

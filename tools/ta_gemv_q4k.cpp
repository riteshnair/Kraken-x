// ============================================================================
//  ta_gemv_q4k — marginal timing of the rows=1 Q4_K dot, which is the whole of
//  a decode step on an expert-routed model, plus the arms that were tried
//  against it.
//
//  Why this tool exists: the engine-level number (tok/s on GLM) mixes the drive
//  read, the expert cache, the router readback and this kernel, so it cannot say
//  whether a kernel change helped. This harness calls the dot on a resident
//  matrix and nothing else.
//
//  It reports the two rates the decision is made on:
//      GB/s   — weight bytes consumed per second. A decode token reads every
//               weight byte once, so this is the ceiling on decode speed.
//      GMAC/s — int8 multiply-accumulates per second, against ~218/core on this
//               box (2 x vpdpbusd ymm per cycle at 3.4 GHz).
//
//  Every arm runs in ONE process, interleaved and order-balanced (AGENTS.md:
//  a sequential probe reads the warm-up as the arm). --width picks the matrix
//  size, which is how the "is it bandwidth or issue" question is answered:
//  0.3 MiB fits in L2 and 15.8 MiB streams from L3/DRAM, and if both time the
//  same the kernel is not waiting on memory.
// ============================================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "krk/quant.hpp"

using namespace krk;

using Clock = std::chrono::steady_clock;

static double now_s() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// sum of one 32-byte activation group, as i32 — what quantize_q8_row_sx fills.
static i32 group_sum(const i8 *x) {
    i32 s = 0;
    for (int i = 0; i < 32; i++) s += x[i];
    return s;
}

using DotFn = f32 (*)(const u8 *, const i8 *, const f32 *, i64);

int main(int argc, char **argv) {
    i64 n_in = 2048;    // GLM-4.7-Flash expert width
    i64 n_out = 2048;   // output rows (one expert's projection)
    int reps = 40;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--nin") && i + 1 < argc) n_in = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--nout") && i + 1 < argc) n_out = std::atoll(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    if (n_in % 256 || n_in <= 0) {
        std::printf("n_in must be a positive multiple of 256\n");
        return 2;
    }

    const i64 blk = 144;
    const i64 nblk = n_in / 256;
    const i64 wbytes = n_out * nblk * blk;
    const double gmacs = static_cast<double>(n_out) * n_in / 1e9;

    std::vector<u8> w(static_cast<size_t>(wbytes));
    std::mt19937 rng(12345);
    for (auto &b : w) b = static_cast<u8>(rng());
    for (i64 r = 0; r < n_out; r++) {
        for (i64 k = 0; k < nblk; k++) {
            u8 *b = &w[static_cast<size_t>((r * nblk + k) * blk)];
            // NOT a power of two. 0x2400 is exactly 2^-6 and 0x1c00 is 2^-8,
            // which makes every partial sum exactly representable and hides the
            // association difference between arms -- the first version of this
            // harness reported maxrel 0.0 for four different kernels because
            // of it. Mantissa bits here keep the check able to fail.
            const u16 d = 0x2642;
            const u16 dm = 0x1d84;
            b[0] = static_cast<u8>(d & 0xFF); b[1] = static_cast<u8>(d >> 8);
            b[2] = static_cast<u8>(dm & 0xFF); b[3] = static_cast<u8>(dm >> 8);
        }
    }

    std::vector<i8> xq(static_cast<size_t>(n_in));
    for (i64 i = 0; i < n_in; i++) xq[static_cast<size_t>(i)] = static_cast<i8>((rng() % 255) - 127);
    std::vector<f32> dx(static_cast<size_t>(n_in / 32), 1.0f);
    std::vector<i32> sx(static_cast<size_t>(n_in / 32));
    for (i64 g = 0; g < n_in / 32; g++)
        sx[static_cast<size_t>(g)] = group_sum(xq.data() + g * 32);

    // The arms. `serial` is the non-sx kernel; `sx_loop` is the `_sx` body
    // that shipped before this change (get_scale_min_k4 called with a RUNTIME
    // index, so its `j < 4` branch is taken eight times per block);
    // `sx_hoist` is the same arithmetic with those eight calls unrolled to
    // constant indices; `tile4` gives each 64-value chunk its own accumulator.
    const int n_arm = 4;
    const char *names[n_arm] = {"serial", "sx_loop", "sx_hoist", "tile4"};

    auto run = [&](int which) {
        f32 sink = 0.0f;
        for (i64 r = 0; r < n_out; r++) {
            const u8 *row = &w[static_cast<size_t>(r * nblk * blk)];
            f32 v;
            switch (which) {
                case 0: v = dot_q4_K_q8_serial(row, xq.data(), dx.data(), n_in); break;
                case 1: v = dot_q4_K_q8_sx_loop(row, xq.data(), dx.data(), sx.data(), n_in); break;
                case 2: v = dot_q4_K_q8_sx(row, xq.data(), dx.data(), sx.data(), n_in); break;
                default: v = dot_q4_K_q8_tile(row, xq.data(), dx.data(), n_in); break;
            }
            sink += v;
        }
        return sink;
    };

    // Agreement against the shipped serial kernel, per arm.
    std::vector<f64> maxrel(n_arm, 0.0);
    for (i64 r = 0; r < n_out; r++) {
        const u8 *row = &w[static_cast<size_t>(r * nblk * blk)];
        const f32 ref = dot_q4_K_q8_serial(row, xq.data(), dx.data(), n_in);
        for (int a = 1; a < n_arm; a++) {
            f32 v;
            switch (a) {
                case 1: v = dot_q4_K_q8_sx_loop(row, xq.data(), dx.data(), sx.data(), n_in); break;
                case 2: v = dot_q4_K_q8_sx(row, xq.data(), dx.data(), sx.data(), n_in); break;
                default: v = dot_q4_K_q8_tile(row, xq.data(), dx.data(), n_in); break;
            }
            const f64 rel = std::fabs(static_cast<f64>(ref) - static_cast<f64>(v)) /
                            (std::fabs(static_cast<f64>(ref)) + 1e-30);
            if (rel > maxrel[a]) maxrel[a] = rel;
        }
    }

    for (int a = 0; a < n_arm; a++) { volatile f32 s = run(a); (void)s; }

    std::vector<std::vector<double>> t(n_arm, std::vector<double>(reps));
    for (int i = 0; i < reps; i++) {
        // Rotate which arm takes the first slot so a drift in the machine does
        // not land on one of them.
        const int off = i % n_arm;
        for (int k = 0; k < n_arm; k++) {
            const int a = (k + off) % n_arm;
            const double s = now_s();
            volatile f32 sink = run(a);
            const double e = now_s();
            (void)sink;
            t[a][i] = e - s;
        }
    }
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    const double base = med(t[0]);

    std::printf("q4_K gemv  n_in=%lld n_out=%lld  %.2f MiB per column  reps=%d\n",
                static_cast<long long>(n_in), static_cast<long long>(n_out),
                static_cast<double>(wbytes) / (1024.0 * 1024.0), reps);
    for (int a = 0; a < n_arm; a++) {
        const double m = med(t[a]);
        const double best = *std::min_element(t[a].begin(), t[a].end());
        std::printf("  %-9s %8.3f ms  %6.2f GB/s  best %6.2f GB/s  %+7.2f%%  "
                    "peak %5.2f%%  maxrel %.1e\n",
                    names[a], m * 1e3, static_cast<double>(wbytes) / m / 1e9,
                    static_cast<double>(wbytes) / best / 1e9, (base - m) / base * 100.0,
                    (gmacs / m) / 218.0 * 100.0, maxrel[a]);
    }
    // A wrong term shows as a large relative disagreement; the association
    // difference between accumulation orders is ~1e-7.
    double worst = 0.0;
    for (int a = 1; a < n_arm; a++) worst = std::max(worst, maxrel[a]);
    std::printf("  verdict  %s (worst arm disagreement %.2e)\n",
                worst > 1e-4 ? "WRONG TERM" : "all arms agree", worst);
    return worst > 1e-4 ? 1 : 0;
}

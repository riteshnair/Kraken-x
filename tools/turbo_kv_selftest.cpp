// TurboQuant 3-bit KV: proves the codebook, the compression, and the distortion
// against the paper's own numbers before any of it is wired into attention.
//
//   clang++ -O2 -std=c++17 -Iinclude tools/turbo_kv_selftest.cpp -o turbo_selftest

#include "krk/turbo_kv.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace krk;

static int fails = 0;
static void check(bool ok, const char *what, double got, double want,
                  double tol) {
    const bool pass = ok && std::fabs(got - want) <= tol;
    if (!pass) fails++;
    std::printf("  %-52s got %10.6f want %10.6f tol %8.6f  %s\n", what, got,
                want, tol, pass ? "ok" : "FAIL");
}

// Continuous k-means on N(0,1); returns the centroids it converges to. This is
// the paper's Eq. (4) solved numerically, and its output is compared with the
// table the header ships -- so the table is checked, not trusted.
static void solve_lloyd(double *c, int n) {
    // One stratified sample of N(0,1) by inverse CDF, built once: the k-means
    // loop then runs over the array instead of re-inverting 8e8 times.
    const int S = 200000;
    std::vector<double> xs(S);
    for (int s = 0; s < S; s++) {
        const double u = (s + 0.5) / S;
        double lo = -6, hi = 6;
        for (int k = 0; k < 48; k++) {
            const double mid = 0.5 * (lo + hi);
            if (0.5 * (1.0 + std::erf(mid / std::sqrt(2.0))) < u) lo = mid;
            else hi = mid;
        }
        xs[s] = 0.5 * (lo + hi);
    }
    for (int i = 0; i < n; i++) c[i] = -3.0 + 6.0 * (i + 0.5) / n;
    for (int it = 0; it < 200; it++) {
        double sum[8] = {0}, wsum[8] = {0};
        for (int s = 0; s < S; s++) {
            const double x = xs[s];
            int best = 0;
            double bd = 1e30;
            for (int i = 0; i < n; i++) {
                const double d = (x - c[i]) * (x - c[i]);
                if (d < bd) { bd = d; best = i; }
            }
            sum[best] += x;
            wsum[best] += 1.0;
        }
        for (int i = 0; i < n; i++) c[i] = wsum[i] > 0 ? sum[i] / wsum[i] : c[i];
    }
}

int main() {
    std::printf("TurboQuant kernel selftest\n");

    // 1. The shipped table must be the Lloyd-Max optimum.
    double c[8];
    solve_lloyd(c, 8);
    double worst = 0;
    for (int i = 0; i < 8; i++) worst = std::max(worst, std::fabs(c[i] - kTurboCentroid[i]));
    check(worst < 2e-3, "shipped codebook == solved Lloyd-Max (max |d|)", worst, 0.0, 2e-3);

    std::mt19937 rng(12345);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    for (i64 d : {96, 576, 64, 128, 256}) {
        std::printf("\n d=%lld\n", (long long)d);
        std::vector<f32> x(d), y(d), rot(d);
        double mse = 0, ip_err2 = 0, mse4 = 0;
        const int trials = 64;
        for (int t = 0; t < trials; t++) {
            // a unit-norm vector, as the paper's bound assumes
            double n2 = 0;
            for (i64 j = 0; j < d; j++) { x[j] = gauss(rng); n2 += x[j] * x[j]; }
            const float inv = 1.0f / std::sqrt((float)n2);
            for (i64 j = 0; j < d; j++) x[j] *= inv;

            std::vector<u8> row(turbo_row_bytes(d));
            turbo_quant_row(x.data(), d, turbo_seed(3, 1), row.data(), rot.data());
            turbo_decode_row(row.data(), d, y.data());

            // MSE of the reconstruction, measured back in the ORIGINAL basis.
            // The inverse is S H / sqrt(d), not the forward transform again.
            turbo_unrotate(y.data(), rot.data(), d, turbo_seed(3, 1));
            double e = 0;
            for (i64 j = 0; j < d; j++) { const double dd = x[j] - rot[j]; e += dd * dd; }
            mse += e;

            // rot4: Q4_0 blocks in the SAME rotated basis. 16 levels and a scale
            // per 32 values, so it should land well under turbo3's 0.033.
            std::vector<u8> row4(rot4_row_bytes(d));
            rot4_quant_row(x.data(), d, turbo_seed(3, 1), row4.data(), rot.data());
            rot4_decode_row(row4.data(), d, y.data());
            turbo_unrotate(y.data(), rot.data(), d, turbo_seed(3, 1));
            double e4 = 0;
            for (i64 j = 0; j < d; j++) { const double dd = x[j] - rot[j]; e4 += dd * dd; }
            mse4 += e4;

            // inner-product distortion against a random query, as the paper
            // defines it: E[(<y,x> - <y,x_hat>)^2] * d / ||y||^2 should be
            // ~0.18 for b=3 (D_prod = 0.18/d). Measuring the ABSOLUTE error
            // scaled this way is the right statistic; a relative error divides
            // by a score that is itself ~1/sqrt(d) and explodes near zero.
            std::vector<f32> q(d), qr(d);
            double qn = 0;
            for (i64 j = 0; j < d; j++) { q[j] = gauss(rng); qn += q[j] * q[j]; }
            const float qi = 1.0f / std::sqrt((float)qn);
            for (i64 j = 0; j < d; j++) q[j] *= qi;
            turbo_rotate(q.data(), qr.data(), d, turbo_seed(3, 1));
            double truth = 0;
            for (i64 j = 0; j < d; j++) truth += (double)q[j] * x[j];
            const double est = turbo_dot(qr.data(), row.data(), d);
            const double err = est - truth;
            ip_err2 += err * err;
        }
        mse /= trials;
        const double dprod = ip_err2 / trials * (double)d;
        check(mse > 0.020 && mse < 0.050, "MSE at 3 bits (paper b=3: 0.03)", mse, 0.030, 0.020);
        // For an MSE-form quantizer the inner-product error against a random
        // unit y is E||x - x_hat||^2 / d, i.e. D_mse/d = 0.033/d here. That is
        // BELOW the paper's 0.18/d bound for the QJL form, which spends a bit
        // on unbiasedness and pays for it in variance. Both numbers are
        // asserted so a regression in either shows up by name.
        check(dprod > 0.015 && dprod < 0.060, "D_prod*d (== D_mse/d for this form)", dprod, 0.033, 0.020);

        mse4 /= trials;
        // 3-bit Lloyd-Max with a per-32 block scale: the same distortion as
        // global-scale turbo3 (0.033), because the codebook is optimal either
        // way -- the block scale buys robustness to a varying local variance,
        // not accuracy. The 4-bit uniform grid it replaced measured 0.0089 here
        // and still broke the model, which is why the number to hold is this one.
        check(mse4 > 0.020 && mse4 < 0.045, "rot4 MSE (block-scaled Lloyd-Max)", mse4, 0.032, 0.013);
        const double r4 = (double)(d * 2) / (double)rot4_row_bytes(d);
        check(r4 > 4.4, "rot4 compression against f16", r4, 4.57, 1e9);
        std::printf("    rot4 row %zu bytes vs %lld f16 (%0.2fx), MSE %0.5f\n",
                    rot4_row_bytes(d), (long long)(d * 2), r4, mse4);

        // rot4's DOT must agree with its own decode, and both with the truth.
        // The MSE arm above only exercises decode, so a nibble-order or
        // block-offset disagreement inside rot4_dot would pass it and still
        // score every attention call from the wrong numbers.
        double dot_err2 = 0, xor_err2 = 0;
        for (int t = 0; t < trials; t++) {
            double n2 = 0;
            for (i64 j = 0; j < d; j++) { x[j] = gauss(rng); n2 += x[j] * x[j]; }
            const float inv = 1.0f / std::sqrt((float)n2);
            for (i64 j = 0; j < d; j++) x[j] *= inv;
            std::vector<u8> row4(rot4_row_bytes(d));
            rot4_quant_row(x.data(), d, turbo_seed(3, 1), row4.data(), rot.data());

            std::vector<f32> q(d), qr(d);
            double qn = 0;
            for (i64 j = 0; j < d; j++) { q[j] = gauss(rng); qn += q[j] * q[j]; }
            const float qi = 1.0f / std::sqrt((float)qn);
            for (i64 j = 0; j < d; j++) q[j] *= qi;
            turbo_rotate(q.data(), qr.data(), d, turbo_seed(3, 1));

            double truth = 0;
            for (i64 j = 0; j < d; j++) truth += (double)q[j] * x[j];
            // (a) dot against the packed row
            const double est_dot = rot4_dot(qr.data(), row4.data(), d);
            dot_err2 += (est_dot - truth) * (est_dot - truth);
            // (b) dot against the DECODED row, which must give the same answer
            std::vector<f32> dec(d);
            rot4_decode_row(row4.data(), d, dec.data());
            double est_dec = 0;
            for (i64 j = 0; j < d; j++) est_dec += (double)qr[j] * dec[j];
            xor_err2 += (est_dec - est_dot) * (est_dec - est_dot);
        }
        const double dprod4 = dot_err2 / trials * (double)d;
        const double mismatch = xor_err2 / trials * (double)d;
        check(dprod4 < 0.060, "rot4 D_prod*d (block-scaled Lloyd-Max)", dprod4, 0.032, 0.028);
        check(mismatch < 1e-6, "rot4 dot == dot over decoded row", mismatch, 0.0, 1e-6);

        const double ratio = (double)(d * 2) / (double)turbo_row_bytes(d);
        check(ratio > 4.5, "compression against f16", ratio, 4.5, 1e9);
        std::printf("    row %zu bytes vs %lld f16 (%0.2fx)\n",
                    turbo_row_bytes(d), (long long)(d * 2), ratio);
    }

    // 2. R^-1 R = I, or a score computed in the rotated basis would not be the
    //    true score (and a decoded value vector would come back scrambled).
    {
        i64 d = 256;
        std::vector<f32> a(d), b(d), c2(d);
        for (i64 j = 0; j < d; j++) a[j] = gauss(rng);
        turbo_rotate(a.data(), b.data(), d, 7);
        turbo_unrotate(b.data(), c2.data(), d, 7);
        double worst2 = 0;
        for (i64 j = 0; j < d; j++) worst2 = std::max(worst2, (double)std::fabs(a[j] - c2[j]));
        check(worst2 < 1e-4, "unrotate(rotate(x)) == x (max |d|)", worst2, 0.0, 1e-4);
    }

    // 3. Different heads must get different rotations, or every head's error is
    //    correlated and the bias stops averaging out across heads.
    {
        i64 d = 128;
        std::vector<f32> a(d), b(d);
        for (i64 j = 0; j < d; j++) a[j] = gauss(rng);
        const uint32_t s0 = turbo_seed(0, 0), s1 = turbo_seed(0, 1);
        std::printf("    seeds %08x / %08x\n", s0, s1);
        turbo_rotate(a.data(), b.data(), d, s0);
        std::vector<f32> c2(d);
        turbo_rotate(a.data(), c2.data(), d, s1);
        double diff = 0;
        for (i64 j = 0; j < d; j++) diff += std::fabs(b[j] - c2[j]);
        check(s0 != s1 && diff > 1.0, "head 0 and head 1 rotations differ", diff, 0.0, 1e9);
    }

    std::printf("\n%s (%d failure(s))\n", fails ? "FAILED" : "ALL OK", fails);
    return fails ? 1 : 0;
}

// Bit-exactness gate for the eight-activation-row K-quant entry points.
//
// Algebraic equivalence is not sufficient here: these are the multi-row
// prefill/speculation paths and their results must be identical to eight calls
// of the rows=1 kernels used by decode and by a prefill remainder. Random block
// bytes and non-power-of-two scales deliberately exercise rounding differences
// that power-of-two-only fixtures hide.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "krk/quant.hpp"

using namespace krk;

static void put_f16(u8 *p, f32 value) {
    u16 h;
    std::memcpy(&h, &value, sizeof(h));
    p[0] = static_cast<u8>(h);
    p[1] = static_cast<u8>(h >> 8);
}

static bool same_bits(f32 a, f32 b) {
    u32 aa, bb;
    std::memcpy(&aa, &a, sizeof(aa));
    std::memcpy(&bb, &b, sizeof(bb));
    return aa == bb;
}

int main() {
    int failures = 0;
    const char *disabled = std::getenv("KRK_CPU_DOT_INT8");
    if (disabled && disabled[0] == '0' && vec_dot_int8()) {
        std::fprintf(stderr, "KRK_CPU_DOT_INT8=0 is ignored\n");
        failures++;
    }
    set_vec_dot_int8(false);
    if (vec_dot_int8()) {
        std::fprintf(stderr, "KRK_CPU_DOT_INT8/set_vec_dot_int8(false) is ignored\n");
        failures++;
    }
    set_vec_dot_int8(true);

    constexpr i64 n = 768;
    constexpr i64 nb = n / 32;
    std::mt19937 rng(0x5845524fu);
    std::vector<u8> q2(3 * 84), q3(3 * 110), q4(3 * 144), q6(3 * 210),
        q8(nb * 34);
    for (u8 &v : q2) v = static_cast<u8>(rng());
    for (u8 &v : q3) v = static_cast<u8>(rng());
    for (u8 &v : q4) v = static_cast<u8>(rng());
    for (u8 &v : q6) v = static_cast<u8>(rng());
    for (u8 &v : q8) v = static_cast<u8>(rng());
    for (i64 b = 0; b < 3; b++) {
        put_f16(q2.data() + b * 84 + 80, 0.0137f * (b + 1));
        put_f16(q2.data() + b * 84 + 82, 0.0073f * (b + 1));
        put_f16(q3.data() + b * 110 + 108, 0.0191f * (b + 1));
        put_f16(q4.data() + b * 144 + 0, 0.0137f * (b + 1));
        put_f16(q4.data() + b * 144 + 2, 0.0073f * (b + 1));
        put_f16(q6.data() + b * 210 + 208, 0.0191f * (b + 1));
    }
    for (i64 b = 0; b < nb; b++)
        put_f16(q8.data() + b * 34, 0.0113f * (b + 1));

    std::vector<f32> x(8 * n);
    std::normal_distribution<f32> normal(0.0f, 1.0f);
    for (f32 &v : x) v = normal(rng);
    std::vector<i8> xq(8 * n);
    std::vector<f32> dx(8 * nb);
    std::vector<i32> sx(8 * nb), sq4(8 * n / 4);
    for (i64 r = 0; r < 8; r++) {
        quantize_q8_row_sx(x.data() + r * n, n, xq.data() + r * n,
                           dx.data() + r * nb, sx.data() + r * nb);
        quantize_q8_row_sq4(x.data() + r * n, n, xq.data() + r * n,
                            dx.data() + r * nb, sq4.data() + r * n / 4);
    }

    f32 got[8], want[8];
    dot_q2_K_q8_x8(q2.data(), xq.data(), dx.data(), n, n, nb, got, 1);
    for (i64 r = 0; r < 8; r++) {
        want[r] = dot_q2_K_q8(q2.data(), xq.data() + r * n,
                              dx.data() + r * nb, n);
        if (!same_bits(got[r], want[r])) {
            std::fprintf(stderr, "Q2_K row %lld: x8=%a r1=%a\n",
                         static_cast<long long>(r), got[r], want[r]);
            failures++;
        }
    }

    dot_q3_K_q8_x8(q3.data(), xq.data(), dx.data(), n, n, nb, got, 1);
    for (i64 r = 0; r < 8; r++) {
        want[r] = dot_q3_K_q8(q3.data(), xq.data() + r * n,
                              dx.data() + r * nb, n);
        if (!same_bits(got[r], want[r])) {
            std::fprintf(stderr, "Q3_K row %lld: x8=%a r1=%a\n",
                         static_cast<long long>(r), got[r], want[r]);
            failures++;
        }
    }

    dot_q4_K_q8_sx_x8(q4.data(), xq.data(), dx.data(), sx.data(), n, n, nb,
                      nb, got, 1);
    for (i64 r = 0; r < 8; r++) {
        want[r] = dot_q4_K_q8_sx(q4.data(), xq.data() + r * n,
                                 dx.data() + r * nb, sx.data() + r * nb, n);
        if (!same_bits(got[r], want[r])) {
            std::fprintf(stderr, "Q4_K row %lld: x8=%a r1=%a\n",
                         static_cast<long long>(r), got[r], want[r]);
            failures++;
        }
    }

    dot_q6_K_q8_sq4_x8(q6.data(), xq.data(), dx.data(), sq4.data(), n, n, nb,
                       n / 4, got, 1);
    for (i64 r = 0; r < 8; r++) {
        want[r] = dot_q6_K_q8_sq4(q6.data(), xq.data() + r * n,
                                  dx.data() + r * nb, sq4.data() + r * n / 4, n);
        if (!same_bits(got[r], want[r])) {
            std::fprintf(stderr, "Q6_K row %lld: x8=%a r1=%a\n",
                         static_cast<long long>(r), got[r], want[r]);
            failures++;
        }
    }

    constexpr i64 q8n = 256;
    dot_q8_0_q8_x8(q8.data(), xq.data(), dx.data(), q8n, n, nb, got, 1);
    for (i64 r = 0; r < 8; r++) {
        want[r] = dot_q8_0_q8(q8.data(), xq.data() + r * n,
                              dx.data() + r * nb, q8n);
        if (!same_bits(got[r], want[r])) {
            std::fprintf(stderr, "Q8_0 row %lld: x8=%a r1=%a\n",
                         static_cast<long long>(r), got[r], want[r]);
            failures++;
        }
    }

    std::printf("K-quant x8/r1 bit equality: %s\n",
                failures ? "FAILED" : "ok (Q2_K, Q3_K, Q4_K, Q6_K, Q8_0; 8 rows each)");
    return failures ? 1 : 0;
}

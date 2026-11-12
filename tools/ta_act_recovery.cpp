// Runtime activation-residual/adaptive recovery and RV32 ternary differential.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "krk/backend.hpp"
#include "krk/rv32.hpp"

using namespace krk;
static int checks = 0, failures = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { failures++; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, m); } } while (0)

static std::vector<u8> tq1_weights(const std::vector<i8> &trits) {
    std::vector<u8> w(54, 0);
    const u32 pow3[5] = {1, 3, 9, 27, 81};
    for (i32 i = 0; i < 256; i++) {
        i32 byte = 0, plane = 0;
        if (i < 160) { byte = i % 32; plane = i / 32; }
        else if (i < 240) { const i32 t = i - 160; byte = 32 + t % 16; plane = t / 16; }
        else { const i32 t = i - 240; byte = 48 + t % 4; plane = t / 4; }
        w[byte] += static_cast<u8>((trits[i] + 1) * pow3[plane]);
    }
    const u16 one = fp32_to_fp16(1.0f);
    std::memcpy(w.data() + 52, &one, 2);
    return w;
}

static void rv32_ternary_differential() {
    std::vector<i8> trits(256), xq(256);
    std::vector<u8> codes(32 * 24); // Each VDOT is three instruction words apart.
    u32 rng = 0x12345678;
    for (i32 i = 0; i < 256; i++) {
        rng = rng * 1664525 + 1013904223;
        trits[i] = static_cast<i8>(static_cast<i32>(rng % 3) - 1);
        codes[(i / 8) * 24 + i % 8] = static_cast<u8>(trits[i] + 8);
        xq[i] = static_cast<i8>((rng >> 8) & 255);
    }
    const std::vector<u8> w = tq1_weights(trits);
    std::vector<f32> decoded(256);
    dequant_row(DType::TQ1_0, w.data(), decoded.data(), 256);
    for (i32 i = 0; i < 256; i++) {
        trits[i] = static_cast<i8>(std::lround(decoded[i]));
        codes[(i / 8) * 24 + i % 8] = static_cast<u8>(trits[i] + 8);
    }
    std::vector<f32> dx(8, 1.0f);
    const f32 native = dot_tq1_0_q8(w.data(), xq.data(), dx.data(), 256);
    std::vector<u32> p;
    p.push_back(rv32_i(static_cast<u32>(-2), 0, 0, 4, 0x13));
    p.push_back(rv32_r(0, 0, 4, 6, 0, 0x0b)); // WB delta=-4
    for (i32 i = 0; i < 32; i++) {
        p.push_back(rv32_i(1024 + 8 * i, 0, 2, 1, 0x03));
        p.push_back(rv32_i(1028 + 8 * i, 0, 2, 2, 0x03));
        p.push_back(rv32_r(0, 2, 1, 5, 5, 0x0b));
    }
    p.push_back(rv32_i(93, 0, 0, 17, 0x13));
    p.push_back(0x00000073);
    Rv32 m;
    m.set_tensor_weights(std::move(codes));
    m.store_bytes(1024, xq.data(), xq.size());
    m.load(0, p.data(), p.size() * 4);
    CHECK(m.run(), "RV32 ternary differential runs");
    CHECK(native == static_cast<f32>(static_cast<i32>(m.reg(5))),
          "RV32 VDOT exactly matches native TQ1_0 integer dot");
}

static void runtime_recovery() {
    std::vector<i8> trits(256);
    for (i32 i = 0; i < 256; i++) trits[i] = static_cast<i8>((i * 17 + 3) % 3 - 1);
    const std::vector<u8> packed = tq1_weights(trits);
    std::vector<u8> w(packed.size() * 4);
    for (i32 o = 0; o < 4; o++) {
        std::memcpy(w.data() + o * 54, packed.data(), 54);
        if (o & 1) { // A different sign pattern on alternating output rows.
            std::vector<i8> flipped(256);
            for (i32 i = 0; i < 256; i++) flipped[i] = -trits[(i + o * 13) % 256];
            const std::vector<u8> other = tq1_weights(flipped);
            std::memcpy(w.data() + o * 54, other.data(), 54);
        }
    }
    std::vector<f32> x(256), reference(4);
    for (i32 i = 0; i < 256; i++) x[i] = std::sin(i * 0.17f) * 0.7f + (i % 29 == 0 ? 0.2f : 0.0f);
    std::vector<i8> risk_q(256), plain_q(256);
    std::vector<f32> risk_d(8), plain_d(8);
    std::vector<i32> risk_sx(8), plain_sx(8), risk_sq4(64), plain_sq4(64);
    f32 relative = -1.0f;
    i64 discarded = -1;
    quantize_q8_row_sx(x.data(), 256, plain_q.data(), plain_d.data(), plain_sx.data());
    quantize_q8_row_risk(x.data(), 256, risk_q.data(), risk_d.data(), &relative,
                         &discarded, risk_sx.data(), nullptr);
    CHECK(relative > 0.0f && relative < 1.0f && discarded >= 0,
          "fused Q8 risk telemetry is finite and meaningful");
    CHECK(std::memcmp(risk_q.data(), plain_q.data(), 256) == 0 &&
          std::memcmp(risk_d.data(), plain_d.data(), 8 * sizeof(f32)) == 0 &&
          std::memcmp(risk_sx.data(), plain_sx.data(), 8 * sizeof(i32)) == 0,
          "risk pass preserves Q8 values, scales, and Q4 side sums exactly");
    quantize_q8_row_sq4(x.data(), 256, plain_q.data(), plain_d.data(), plain_sq4.data());
    quantize_q8_row_risk(x.data(), 256, risk_q.data(), risk_d.data(), &relative,
                         &discarded, nullptr, risk_sq4.data());
    CHECK(std::memcmp(risk_sq4.data(), plain_sq4.data(), 64 * sizeof(i32)) == 0,
          "risk pass preserves Q6 four-value side sums exactly");
    for (i32 o = 0; o < 4; o++) reference[o] = vec_dot(DType::TQ1_0, w.data() + o * 54, x.data(), 256);
    std::unique_ptr<Backend, void (*)(Backend *)> cpu(make_cpu_backend(), [](Backend *b) { delete b; });
    std::vector<f32> baseline(4), recovered(4), fallback(4);
    set_activation_recovery_mode(ActivationRecoveryMode::Off);
    cpu->gemm(baseline.data(), x.data(), w.data(), DType::TQ1_0, 4, 256, 1);
    std::vector<f32> grouped_a(4), grouped_b(4);
    void *grouped_out[] = {grouped_a.data(), grouped_b.data()};
    const void *grouped_w[] = {w.data(), w.data()};
    const i64 grouped_cols[] = {4, 4};
    cpu->gemm_group(grouped_out, x.data(), grouped_w, DType::TQ1_0, grouped_cols,
                    256, 2, 1);
    CHECK(std::memcmp(grouped_a.data(), baseline.data(), sizeof(baseline)) == 0 &&
          std::memcmp(grouped_b.data(), baseline.data(), sizeof(baseline)) == 0,
          "default grouped GEMM remains bit-identical to ungrouped GEMM");
    set_activation_recovery_mode(ActivationRecoveryMode::Residual);
    cpu->gemm(recovered.data(), x.data(), w.data(), DType::TQ1_0, 4, 256, 1);
    cpu->gemm_group(grouped_out, x.data(), grouped_w, DType::TQ1_0, grouped_cols,
                    256, 2, 1);
    CHECK(std::memcmp(grouped_a.data(), recovered.data(), sizeof(recovered)) == 0 &&
          std::memcmp(grouped_b.data(), recovered.data(), sizeof(recovered)) == 0,
          "residual grouped GEMM uses the same per-matrix policy");
    f32 base_error = 0, recovered_error = 0;
    for (i32 o = 0; o < 4; o++) {
        base_error = std::max(base_error, std::fabs(baseline[o] - reference[o]));
        recovered_error = std::max(recovered_error, std::fabs(recovered[o] - reference[o]));
    }
    CHECK(recovered_error < base_error * 0.05f, "residual pass removes nearly all activation rounding error");
    CHECK(recovered_error < 0.002f, "residual result agrees with native F32 activation GEMM");

    // One large value and 255 tiny values creates both clipping/discarding and
    // a large residual-to-signal ratio. Adaptive mode must choose exact F32.
    for (i32 i = 0; i < 256; i++) x[i] = i == 0 ? 1000.0f : 0.0001f * std::sin(i);
    for (i32 o = 0; o < 4; o++) reference[o] = vec_dot(DType::TQ1_0, w.data() + o * 54, x.data(), 256);
    set_activation_recovery_mode(ActivationRecoveryMode::Adaptive);
    cpu->gemm(fallback.data(), x.data(), w.data(), DType::TQ1_0, 4, 256, 1);
    CHECK(std::memcmp(fallback.data(), reference.data(), sizeof(fallback)) == 0,
          "adaptive high-risk row is bit-identical to native F32 activation GEMM");
    set_activation_recovery_mode(ActivationRecoveryMode::Off);
}

int main() {
    set_vec_dot_fast(false);
    set_vec_dot_wide(false);
    set_vec_dot_int8(true);
    rv32_ternary_differential();
    runtime_recovery();
    std::printf("Activation recovery: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}

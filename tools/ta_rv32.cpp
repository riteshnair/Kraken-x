// Conformance and lockstep gates for the engine-native RV32 coprocessor.
#include <cstdio>
#include <cstring>
#include <vector>

#include "krk/rv32.hpp"

using namespace krk;

static int checks = 0, failures = 0;
#define CHECK(cond, message) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, message); \
    } \
} while (0)

static void append(std::vector<u32> &p, u32 word) { p.push_back(word); }
static void addi(std::vector<u32> &p, u32 rd, u32 rs1, i32 imm) {
    append(p, rv32_i(static_cast<u32>(imm), rs1, 0, rd, 0x13));
}
static void op(std::vector<u32> &p, u32 rd, u32 rs1, u32 rs2, u32 f3,
               u32 f7 = 0) {
    append(p, rv32_r(f7, rs2, rs1, f3, rd, 0x33));
}
static void finish(std::vector<u32> &p, u32 code = 0) {
    addi(p, 10, 0, static_cast<i32>(code));
    addi(p, 17, 0, 93);
    append(p, 0x00000073);
}
static Rv32 execute(const std::vector<u32> &program, bool trace = false,
                    u64 limit = 10000) {
    Rv32 machine;
    CHECK(machine.load(0, program.data(), program.size() * 4), "program loads");
    machine.set_tracing(trace);
    CHECK(machine.run(limit), "program runs without an architectural fault");
    return machine;
}

static void test_arithmetic_and_memory() {
    std::vector<u32> p;
    addi(p, 1, 0, 20);
    addi(p, 2, 0, -6);
    op(p, 3, 1, 2, 0, 1);       // MUL = -120
    op(p, 4, 1, 2, 4, 1);       // DIV = -3
    op(p, 5, 1, 2, 6, 1);       // REM = 2
    op(p, 6, 1, 2, 2, 0x10);    // SH1ADD = 34
    append(p, rv32_s(128, 3, 0, 2, 0x23));
    append(p, rv32_i(128, 0, 2, 7, 0x03));
    append(p, rv32_i(128, 0, 0, 8, 0x03)); // LB = -120
    addi(p, 9, 0, -1);
    op(p, 11, 9, 9, 4, 1);      // INT_MIN? no: -1 / -1 = 1
    finish(p);
    Rv32 m = execute(p);
    CHECK(m.reg(3) == static_cast<u32>(-120), "MUL low word");
    CHECK(m.reg(4) == static_cast<u32>(-3), "signed DIV truncates toward zero");
    CHECK(m.reg(5) == 2, "signed REM has dividend's sign");
    CHECK(m.reg(6) == 34, "Zba SH1ADD");
    CHECK(m.reg(7) == static_cast<u32>(-120), "SW/LW round trip");
    CHECK(m.reg(8) == static_cast<u32>(-120), "LB sign extends");
    CHECK(m.reg(11) == 1, "division by -1");
    CHECK(m.halted() && m.exit_code() == 0, "exit ecall");

    std::vector<u32> edge;
    append(edge, rv32_u(0x80000000u, 1, 0x37));
    addi(edge, 2, 0, -1);
    op(edge, 3, 1, 2, 4, 1);
    op(edge, 4, 1, 2, 6, 1);
    op(edge, 5, 1, 0, 5, 1);
    finish(edge);
    Rv32 e = execute(edge);
    CHECK(e.reg(3) == 0x80000000u, "INT_MIN / -1 wraps");
    CHECK(e.reg(4) == 0, "INT_MIN % -1 is zero");
    CHECK(e.reg(5) == 0xffffffffu, "unsigned divide by zero");
}

static void test_control_flow_and_output() {
    std::vector<u32> p;
    addi(p, 1, 0, 0);           // counter
    addi(p, 2, 0, 0);           // sum
    addi(p, 3, 0, 10);
    const size_t loop = p.size();
    op(p, 2, 2, 1, 0);
    addi(p, 1, 1, 1);
    append(p, rv32_b(static_cast<u32>((static_cast<i64>(loop) - static_cast<i64>(p.size())) * 4),
                     3, 1, 4, 0x63)); // BLT counter,10,loop
    addi(p, 10, 0, 'O');
    addi(p, 17, 0, 1);
    append(p, 0x00000073);
    addi(p, 10, 0, 'K');
    addi(p, 17, 0, 1);
    append(p, 0x00000073);
    finish(p);
    Rv32 m = execute(p);
    CHECK(m.reg(2) == 45, "branch loop sum");
    CHECK(m.output().size() == 2 && m.output()[0] == 'O' && m.output()[1] == 'K',
          "putchar ecalls preserve byte output");
}

static void test_tensor_instructions() {
    std::vector<u8> weights(24);
    for (u32 i = 0; i < 8; i++) weights[i] = i;
    for (u32 i = 16; i < 24; i++) weights[i] = i - 8;
    std::vector<u32> p;
    append(p, rv32_u(0x04030000u, 1, 0x37));
    addi(p, 1, 1, 0x201);        // x1 = bytes 1,2,3,4
    append(p, rv32_u(0x08070000u, 2, 0x37));
    addi(p, 2, 2, 0x605);        // x2 = bytes 5,6,7,8
    addi(p, 3, 0, 100);
    addi(p, 4, 0, 0); // WB delta = -7; first VDOT at PC word 7 reads word 0
    append(p, rv32_r(0, 0, 4, 6, 0, 0x0b)); // WB x4
    append(p, rv32_r(0, 2, 1, 5, 3, 0x0b)); // VDOT x3,x1,x2
    append(p, rv32_r(0, 2, 1, 7, 5, 0x0b)); // SMAQA x5,x1,x2
    append(p, rv32_r(0, 2, 1, 5, 3, 0x0b)); // second VDOT, weights 8..15
    finish(p);
    Rv32 m;
    m.set_tensor_weights(weights);
    CHECK(m.load(0, p.data(), p.size() * 4), "tensor program loads");
    CHECK(m.run(), "tensor program runs");
    CHECK(m.reg(5) == 70, "SMAQA is four signed byte products");
    CHECK(m.reg(3) == 148, "two VDOTs accumulate and advance cursor");
    CHECK(m.weight_cursor() == 48, "weight PC follows instruction PC including non-VDOT gaps");
}

static void load_imm(std::vector<u32> &p, u32 rd, u32 value) {
    append(p, rv32_u((value + 0x800) & 0xfffff000u, rd, 0x37));
    addi(p, rd, rd, static_cast<i32>(value << 20) >> 20);
}

static void test_tensor_differential() {
    u32 rng = 0x51a7e123u;
    auto random = [&]() {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        return rng;
    };
    std::vector<u8> weights(8 * 128);
    for (u8 &v : weights) v = random() & 15;
    for (u32 trial = 0; trial < 256; trial++) {
        const u32 a = random(), b = random(), initial = random(), base = random() & 127;
        std::vector<u32> p;
        load_imm(p, 1, a); load_imm(p, 2, b); load_imm(p, 3, initial); load_imm(p, 4, base);
        append(p, rv32_r(0, 0, 4, 6, 0, 0x0b)); // WB is instruction word 8
        append(p, rv32_r(0, 2, 1, 5, 3, 0x0b)); // VDOT is word 9: WP = base
        append(p, rv32_r(0, 2, 1, 7, 5, 0x0b));
        finish(p);
        Rv32 m;
        m.set_tensor_weights(weights);
        m.load(0, p.data(), p.size() * 4);
        CHECK(m.run(), "differential tensor program runs");
        auto sb = [](u32 v, u32 lane) { return static_cast<i32>(static_cast<i8>((v >> 8 * lane) & 255)); };
        i32 dot = 0, smaqa = 0;
        for (u32 k = 0; k < 8; k++)
            dot += sb(k < 4 ? a : b, k & 3) * (static_cast<i32>(weights[base * 8 + k]) - 8);
        for (u32 k = 0; k < 4; k++) smaqa += sb(a, k) * sb(b, k);
        CHECK(m.reg(3) == initial + static_cast<u32>(dot), "VDOT matches independent model");
        CHECK(m.reg(5) == static_cast<u32>(smaqa), "SMAQA matches independent model");
        if (failures) return;
    }
}

static void test_lockstep_and_faults() {
    std::vector<u32> p;
    addi(p, 1, 0, 123);
    addi(p, 2, 0, 456);
    op(p, 3, 1, 2, 0, 1);
    append(p, rv32_s(64, 3, 0, 2, 0x23));
    append(p, rv32_i(64, 0, 2, 4, 0x03));
    finish(p);
    Rv32 a = execute(p, false), b = execute(p, true);
    CHECK(a.state_hash() == b.state_hash(), "tracing cannot change architectural state");
    CHECK(a.trace_hash() == b.trace_hash(), "trace hash is independent of trace retention");
    CHECK(b.trace().size() == b.instructions(), "one record per committed instruction");
    CHECK(b.trace()[3].memory_bytes == 4 && b.trace()[3].memory_after == a.reg(3),
          "store record carries before/after bytes");

    Rv32 bad;
    const u32 lw = rv32_i(2, 0, 2, 1, 0x03);
    bad.load(0, &lw, 4);
    CHECK(!bad.run() && bad.fault() == Rv32Fault::MisalignedLoad, "misaligned LW faults");

    Rv32 range;
    const u32 sw = rv32_s(0, 0, 0, 2, 0x23);
    range.load(0, &sw, 4);
    range.set_reg(1, 0);
    // Store at the last byte with a four-byte width must not wrap.
    const u32 sw2 = rv32_s(0, 0, 1, 2, 0x23);
    range.load(0, &sw2, 4);
    range.set_reg(1, range.memory_size());
    CHECK(!range.run() && range.fault() == Rv32Fault::StoreOutOfRange,
          "out-of-range SW faults without wrapping");

    Rv32 illegal;
    const u32 zero = 0;
    illegal.load(0, &zero, 4);
    CHECK(!illegal.run() && illegal.fault() == Rv32Fault::IllegalInstruction,
          "zero instruction is illegal");

    Rv32 limited;
    const u32 jal = rv32_j(0, 0, 0x6f);
    limited.load(0, &jal, 4);
    CHECK(!limited.run(7) && limited.fault() == Rv32Fault::InstructionLimit &&
              limited.instructions() == 7,
          "instruction budget stops runaway firmware");
}

int main() {
    test_arithmetic_and_memory();
    test_control_flow_and_output();
    test_tensor_instructions();
    test_tensor_differential();
    test_lockstep_and_faults();
    std::printf("RV32 coprocessor: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}

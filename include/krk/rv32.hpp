// Engine-native, exact RV32 compute coprocessor.
//
// This is the host-side counterpart of Qwenputer's compiled RISC-V core: the
// same integer ISA, bounded byte memory, tensor-weight memory and deterministic
// execution records, without requiring a specially compiled model checkpoint.
// It never runs implicitly inside a transformer forward pass and never changes
// routing, sampling or expert residency.
#ifndef KRK_RV32_HPP
#define KRK_RV32_HPP

#include "krk/common.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace krk {

enum class Rv32Fault : u32 {
    None = 0,
    IllegalInstruction,
    Breakpoint,
    MisalignedLoad,
    MisalignedStore,
    LoadOutOfRange,
    StoreOutOfRange,
    BadJump,
    EnvironmentCall,
    InstructionLimit,
};

const char *rv32_fault_name(Rv32Fault fault);

struct Rv32TraceEvent {
    u64 instruction = 0;
    u32 pc = 0;
    u32 word = 0;
    u32 next_pc = 0;
    u8 rd = 0;                 // 32 means no register write
    u32 rd_before = 0;
    u32 rd_after = 0;
    u32 memory_address = 0;    // valid when memory_bytes != 0
    u32 memory_bytes = 0;
    u32 memory_before = 0;     // little-endian, up to four bytes
    u32 memory_after = 0;
};

class Rv32 {
public:
    static constexpr u32 kDefaultMemoryBytes = 64 * 1024;
    static constexpr u32 kMisa = 0x40001100; // RV32I + M

    explicit Rv32(u32 memory_bytes = kDefaultMemoryBytes);

    void reset(u32 entry = 0);
    bool load(u32 address, const void *bytes, size_t size, std::string *err = nullptr);
    bool store_bytes(u32 address, const void *bytes, size_t size,
                     std::string *err = nullptr);
    bool read_bytes(u32 address, void *out, size_t size,
                    std::string *err = nullptr) const;

    // Each byte is one int4 code in [0,15], decoded as code-8. Qwenputer's
    // 15-bit weight-PC addresses eight-code words: VDOT uses WB_delta + PC/4,
    // and WB rs1 establishes that PC-relative delta. Keeping weights separate
    // from byte RAM prevents stores from modifying model/tensor weights.
    void set_tensor_weights(std::vector<u8> codes);

    bool step();
    bool run(u64 max_instructions = 1000000);

    void set_tracing(bool on) { tracing_ = on; }
    bool tracing() const { return tracing_; }
    const std::vector<Rv32TraceEvent> &trace() const { return trace_; }
    u64 trace_hash() const { return trace_hash_; }

    u32 reg(u32 index) const { return regs_[index & 31]; }
    void set_reg(u32 index, u32 value);
    u32 pc() const { return pc_; }
    u32 weight_word() const { return (weight_base_delta_ + pc_ / 4) & 32767; }
    u32 weight_cursor() const { return weight_word() * 8; }
    u64 instructions() const { return instructions_; }
    Rv32Fault fault() const { return fault_; }
    u32 fault_pc() const { return fault_pc_; }
    bool halted() const { return halted_; }
    u32 exit_code() const { return exit_code_; }
    const std::vector<u8> &output() const { return output_; }
    size_t memory_size() const { return memory_.size(); }

    // FNV-1a over all architectural byte memory and tensor codes. Unlike the
    // trace hash, this also catches state divergence with no instruction trace.
    u64 state_hash() const;

private:
    u32 load_u32(u32 address, Rv32Fault align_fault);
    void record(u32 pc, u32 word, u32 next_pc, u8 rd, u32 rd_before,
                u32 address = 0, u32 bytes = 0, u32 before = 0, u32 after = 0);
    bool fail(Rv32Fault fault, u32 pc);

    std::vector<u8> memory_;
    std::vector<u8> tensor_weights_;
    std::array<u32, 32> regs_{};
    std::vector<u8> output_;
    std::vector<Rv32TraceEvent> trace_;
    u32 pc_ = 0;
    u32 weight_base_delta_ = 0;
    u32 exit_code_ = 0;
    u32 fault_pc_ = 0;
    u64 instructions_ = 0;
    u64 trace_hash_ = 1469598103934665603ULL;
    Rv32Fault fault_ = Rv32Fault::None;
    bool halted_ = false;
    bool tracing_ = false;
};

// Encoding helpers for firmware builders and conformance tests. They use the
// architectural field layout, not a second instruction decoder.
u32 rv32_r(u32 funct7, u32 rs2, u32 rs1, u32 funct3, u32 rd, u32 opcode);
u32 rv32_i(u32 imm, u32 rs1, u32 funct3, u32 rd, u32 opcode);
u32 rv32_s(u32 imm, u32 rs2, u32 rs1, u32 funct3, u32 opcode);
u32 rv32_b(u32 imm, u32 rs2, u32 rs1, u32 funct3, u32 opcode);
u32 rv32_u(u32 imm, u32 rd, u32 opcode);
u32 rv32_j(u32 imm, u32 rd, u32 opcode);

} // namespace krk

#endif // KRK_RV32_HPP

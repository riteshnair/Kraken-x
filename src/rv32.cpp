#include "krk/rv32.hpp"

#include <climits>
#include <cstring>

namespace krk {
namespace {

inline u32 sx(u32 value, u32 bits) {
    const u32 sign = 1u << (bits - 1);
    return (value ^ sign) - sign;
}

inline i32 as_i32(u32 value) { return static_cast<i32>(value); }
inline u32 byte_at(u32 value, u32 lane) { return (value >> (8 * lane)) & 0xff; }
inline i32 signed_byte(u32 value, u32 lane) {
    return static_cast<i32>(static_cast<i8>(byte_at(value, lane)));
}

inline void hash_byte(u64 *hash, u8 value) {
    *hash ^= value;
    *hash *= 1099511628211ULL;
}

inline void hash_u32(u64 *hash, u32 value) {
    for (u32 i = 0; i < 4; i++) hash_byte(hash, (value >> (8 * i)) & 0xff);
}

inline void hash_u64(u64 *hash, u64 value) {
    hash_u32(hash, static_cast<u32>(value));
    hash_u32(hash, static_cast<u32>(value >> 32));
}

} // namespace

const char *rv32_fault_name(Rv32Fault fault) {
    switch (fault) {
        case Rv32Fault::None: return "none";
        case Rv32Fault::IllegalInstruction: return "illegal instruction";
        case Rv32Fault::Breakpoint: return "breakpoint";
        case Rv32Fault::MisalignedLoad: return "misaligned load";
        case Rv32Fault::MisalignedStore: return "misaligned store";
        case Rv32Fault::LoadOutOfRange: return "load out of range";
        case Rv32Fault::StoreOutOfRange: return "store out of range";
        case Rv32Fault::BadJump: return "bad jump";
        case Rv32Fault::EnvironmentCall: return "unsupported environment call";
        case Rv32Fault::InstructionLimit: return "instruction limit";
    }
    return "unknown";
}

Rv32::Rv32(u32 memory_bytes) : memory_(memory_bytes) { reset(); }

void Rv32::reset(u32 entry) {
    regs_.fill(0);
    output_.clear();
    trace_.clear();
    pc_ = entry;
    weight_base_delta_ = 0;
    exit_code_ = 0;
    fault_pc_ = 0;
    instructions_ = 0;
    trace_hash_ = 1469598103934665603ULL;
    fault_ = Rv32Fault::None;
    halted_ = false;
}

bool Rv32::load(u32 address, const void *bytes, size_t size, std::string *err) {
    return store_bytes(address, bytes, size, err);
}

bool Rv32::store_bytes(u32 address, const void *bytes, size_t size,
                       std::string *err) {
    if ((size != 0 && bytes == nullptr) || address > memory_.size() ||
        size > memory_.size() - address) {
        if (err) *err = "RV32 store is outside byte memory";
        return false;
    }
    if (size) std::memcpy(memory_.data() + address, bytes, size);
    return true;
}

bool Rv32::read_bytes(u32 address, void *out, size_t size,
                      std::string *err) const {
    if ((size != 0 && out == nullptr) || address > memory_.size() ||
        size > memory_.size() - address) {
        if (err) *err = "RV32 load is outside byte memory";
        return false;
    }
    if (size) std::memcpy(out, memory_.data() + address, size);
    return true;
}

void Rv32::set_tensor_weights(std::vector<u8> codes) {
    tensor_weights_ = std::move(codes);
    weight_base_delta_ = 0;
}

void Rv32::set_reg(u32 index, u32 value) {
    if ((index & 31) != 0) regs_[index & 31] = value;
}

u32 Rv32::load_u32(u32 address, Rv32Fault align_fault) {
    if ((address & 3) != 0) {
        fail(align_fault, pc_);
        return 0;
    }
    if (address > memory_.size() || memory_.size() - address < 4) {
        fail(address == pc_ ? Rv32Fault::LoadOutOfRange
                            : (align_fault == Rv32Fault::MisalignedStore
                                   ? Rv32Fault::StoreOutOfRange
                                   : Rv32Fault::LoadOutOfRange),
             pc_);
        return 0;
    }
    return static_cast<u32>(memory_[address]) |
           (static_cast<u32>(memory_[address + 1]) << 8) |
           (static_cast<u32>(memory_[address + 2]) << 16) |
           (static_cast<u32>(memory_[address + 3]) << 24);
}

bool Rv32::fail(Rv32Fault fault, u32 pc) {
    fault_ = fault;
    fault_pc_ = pc;
    halted_ = true;
    return false;
}

void Rv32::record(u32 pc, u32 word, u32 next_pc, u8 rd, u32 rd_before,
                  u32 address, u32 bytes, u32 before, u32 after) {
    const u32 rd_after = rd < 32 ? regs_[rd] : 0;
    if (tracing_) {
        trace_.push_back({instructions_, pc, word, next_pc, rd, rd_before,
                          rd_after, address, bytes, before, after});
    }
    hash_u64(&trace_hash_, instructions_);
    hash_u32(&trace_hash_, pc);
    hash_u32(&trace_hash_, word);
    hash_u32(&trace_hash_, next_pc);
    hash_byte(&trace_hash_, rd);
    hash_u32(&trace_hash_, rd_before);
    hash_u32(&trace_hash_, rd_after);
    hash_u32(&trace_hash_, address);
    hash_u32(&trace_hash_, bytes);
    hash_u32(&trace_hash_, before);
    hash_u32(&trace_hash_, after);
}

u64 Rv32::state_hash() const {
    u64 hash = 1469598103934665603ULL;
    hash_u32(&hash, pc_);
    hash_u32(&hash, weight_base_delta_);
    hash_u64(&hash, instructions_);
    for (u32 value : regs_) hash_u32(&hash, value);
    for (u8 value : memory_) hash_byte(&hash, value);
    for (u8 value : tensor_weights_) hash_byte(&hash, value);
    return hash;
}

bool Rv32::step() {
    if (halted_) return false;
    const u32 pc = pc_;
    const u32 word = load_u32(pc, Rv32Fault::MisalignedLoad);
    if (fault_ != Rv32Fault::None) return false;

    const u32 opcode = word & 0x7f;
    const u32 rd = (word >> 7) & 31;
    const u32 funct3 = (word >> 12) & 7;
    const u32 rs1 = (word >> 15) & 31;
    const u32 rs2 = (word >> 20) & 31;
    const u32 funct7 = word >> 25;
    const u32 a = regs_[rs1], b = regs_[rs2];
    const u32 old_rd = regs_[rd];
    const u32 i_imm = sx(word >> 20, 12);
    u32 result = 0;
    u32 next_pc = pc + 4;
    bool write_rd = false;

    auto write = [&](u32 value) {
        result = value;
        write_rd = true;
    };
    auto alu = [&](u32 op, u32 lhs, u32 rhs) -> u32 {
        switch (op) {
            case 0: return lhs + rhs;
            case 1: return lhs << (rhs & 31);
            case 2: return as_i32(lhs) < as_i32(rhs) ? 1u : 0u;
            case 3: return lhs < rhs ? 1u : 0u;
            case 4: return lhs ^ rhs;
            case 5: return lhs >> (rhs & 31);
            case 6: return lhs | rhs;
            case 7: return lhs & rhs;
        }
        return 0;
    };
    auto load = [&](u32 address, u32 width, bool sign, u32 *value) -> bool {
        if ((width == 2 && (address & 1)) || (width == 4 && (address & 3)))
            return fail(Rv32Fault::MisalignedLoad, pc);
        if (address > memory_.size() || memory_.size() - address < width)
            return fail(Rv32Fault::LoadOutOfRange, pc);
        u32 v = 0;
        for (u32 i = 0; i < width; i++) v |= static_cast<u32>(memory_[address + i]) << (8 * i);
        if (sign && width < 4) v = sx(v, width * 8);
        *value = v;
        return true;
    };
    auto store = [&](u32 address, u32 width, u32 value) -> bool {
        if ((width == 2 && (address & 1)) || (width == 4 && (address & 3)))
            return fail(Rv32Fault::MisalignedStore, pc);
        if (address > memory_.size() || memory_.size() - address < width)
            return fail(Rv32Fault::StoreOutOfRange, pc);
        u32 before = 0;
        for (u32 i = 0; i < width; i++) before |= static_cast<u32>(memory_[address + i]) << (8 * i);
        for (u32 i = 0; i < width; i++) memory_[address + i] = (value >> (8 * i)) & 0xff;
        record(pc, word, next_pc, 32, 0, address, width, before,
               value & (width == 4 ? 0xffffffffu : ((1u << (width * 8)) - 1)));
        return true;
    };

    switch (opcode) {
        case 0x37: // LUI
            write(word & 0xfffff000u);
            break;
        case 0x17: // AUIPC
            write(pc + (word & 0xfffff000u));
            break;
        case 0x6f: { // JAL
            const u32 imm = sx(((word >> 31) << 20) | (((word >> 12) & 0xff) << 12) |
                               (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3ff) << 1), 21);
            write(pc + 4);
            next_pc = pc + imm;
            break;
        }
        case 0x67: // JALR
            if (funct3 != 0) return fail(Rv32Fault::IllegalInstruction, pc);
            write(pc + 4);
            next_pc = (a + i_imm) & ~1u;
            break;
        case 0x63: { // branches
            const u32 imm = sx(((word >> 31) << 12) | (((word >> 7) & 1) << 11) |
                               (((word >> 25) & 0x3f) << 5) | (((word >> 8) & 15) << 1), 13);
            bool take = false;
            switch (funct3) {
                case 0: take = a == b; break;
                case 1: take = a != b; break;
                case 4: take = as_i32(a) < as_i32(b); break;
                case 5: take = as_i32(a) >= as_i32(b); break;
                case 6: take = a < b; break;
                case 7: take = a >= b; break;
                default: return fail(Rv32Fault::IllegalInstruction, pc);
            }
            if (take) next_pc = pc + imm;
            break;
        }
        case 0x03: { // loads
            const u32 width = funct3 == 0 || funct3 == 4 ? 1 :
                              funct3 == 1 || funct3 == 5 ? 2 :
                              funct3 == 2 ? 4 : 0;
            if (width == 0) return fail(Rv32Fault::IllegalInstruction, pc);
            u32 value = 0;
            if (!load(a + i_imm, width, funct3 < 4, &value)) return false;
            write(value);
            break;
        }
        case 0x23: { // stores
            const u32 width = funct3 < 3 ? 1u << funct3 : 0;
            if (width == 0) return fail(Rv32Fault::IllegalInstruction, pc);
            const u32 imm = sx((funct7 << 5) | rd, 12);
            if (!store(a + imm, width, b)) return false;
            break;
        }
        case 0x13: // immediate ALU
            if (funct3 == 1) {
                if (funct7 != 0) return fail(Rv32Fault::IllegalInstruction, pc);
                write(a << rs2);
            } else if (funct3 == 5) {
                if (funct7 == 0) write(a >> rs2);
                else if (funct7 == 0x20) write(static_cast<u32>(as_i32(a) >> rs2));
                else return fail(Rv32Fault::IllegalInstruction, pc);
            } else if (funct3 == 2 || funct3 == 3 || funct3 == 0 ||
                       funct3 == 4 || funct3 == 6 || funct3 == 7) {
                write(alu(funct3, a, i_imm));
            } else {
                return fail(Rv32Fault::IllegalInstruction, pc);
            }
            break;
        case 0x33: // register ALU, M and Zba
            if (funct7 == 0) write(alu(funct3, a, b));
            else if (funct7 == 0x20 && funct3 == 0) write(a - b);
            else if (funct7 == 0x20 && funct3 == 5)
                write(static_cast<u32>(as_i32(a) >> (b & 31)));
            else if (funct7 == 1) {
                switch (funct3) {
                    case 0: write(static_cast<u32>(static_cast<u64>(a) * b)); break;
                    case 1: write(static_cast<u32>((static_cast<i64>(as_i32(a)) * as_i32(b)) >> 32)); break;
                    case 2: write(static_cast<u32>((static_cast<i64>(as_i32(a)) * static_cast<i64>(b)) >> 32)); break;
                    case 3: write(static_cast<u32>((static_cast<u64>(a) * b) >> 32)); break;
                    case 4: { // DIV, with both architectural corner cases explicit
                        const i32 aa = as_i32(a), bb = as_i32(b);
                        write(bb == 0 ? 0xffffffffu :
                              (aa == INT32_MIN && bb == -1) ? a :
                              static_cast<u32>(aa / bb));
                        break;
                    }
                    case 5: write(b == 0 ? 0xffffffffu : a / b); break;
                    case 6: {
                        const i32 aa = as_i32(a), bb = as_i32(b);
                        write(bb == 0 ? a : (aa == INT32_MIN && bb == -1) ? 0u :
                              static_cast<u32>(aa % bb));
                        break;
                    }
                    case 7: write(b == 0 ? a : a % b); break;
                }
            } else if (funct7 == 0x10 && (funct3 == 2 || funct3 == 4 || funct3 == 6)) {
                const u32 shift = funct3 == 2 ? 1 : funct3 == 4 ? 2 : 3;
                write((a << shift) + b); // SH1ADD/SH2ADD/SH3ADD
            } else {
                return fail(Rv32Fault::IllegalInstruction, pc);
            }
            break;
        case 0x0b: // Qwenputer tensor custom-0 instructions
            if (funct7 != 0) return fail(Rv32Fault::IllegalInstruction, pc);
            if (funct3 == 5) { // VDOT: eight signed activations x int4 weights
                u32 sum = 0;
                const u32 base = 8 * weight_word();
                for (u32 k = 0; k < 8; k++) {
                    const u32 index = base + k;
                    const i32 weight = index < tensor_weights_.size()
                                           ? static_cast<i32>(tensor_weights_[index] & 15) - 8
                                           : -8;
                    const i32 activation = signed_byte(k < 4 ? a : b, k & 3);
                    sum += static_cast<u32>(activation * weight);
                }
                write(old_rd + sum);
            } else if (funct3 == 6 && rd == 0 && rs2 == 0) { // WB
                weight_base_delta_ = (a - pc / 4 - 1) & 32767;
            } else if (funct3 == 7) { // SMAQA
                u32 sum = 0;
                for (u32 k = 0; k < 4; k++)
                    sum += static_cast<u32>(signed_byte(a, k) * signed_byte(b, k));
                write(old_rd + sum);
            } else {
                return fail(Rv32Fault::IllegalInstruction, pc);
            }
            break;
        case 0x0f: // FENCE: no host-side reordering exists to constrain
            break;
        case 0x73:
            if (word == 0x00000073) { // ECALL
                if (regs_[17] == 1) output_.push_back(regs_[10] & 0xff);
                else if (regs_[17] == 93) {
                    exit_code_ = regs_[10];
                    halted_ = true;
                } else {
                    return fail(Rv32Fault::EnvironmentCall, pc);
                }
            } else if (word == 0x00100073) {
                return fail(Rv32Fault::Breakpoint, pc);
            } else if ((funct3 == 2 || funct3 == 3 || funct3 == 6 || funct3 == 7) && rs1 == 0) {
                // Qwenputer exposes pure CSR reads only; writes are illegal.
                const u32 csr = word >> 20;
                if (csr == 0x301) write(kMisa);
                else if (csr == 0xc00 || csr == 0xc01 || csr == 0xc02 ||
                         csr == 0xc80 || csr == 0xc81 || csr == 0xc82)
                    write(static_cast<u32>(instructions_));
                else if (csr == 0xf14) write(0);
                else return fail(Rv32Fault::IllegalInstruction, pc);
            } else {
                return fail(Rv32Fault::IllegalInstruction, pc);
            }
            break;
        default:
            return fail(Rv32Fault::IllegalInstruction, pc);
    }

    if (!halted_ && ((next_pc & 3) != 0 || next_pc > memory_.size() ||
                     memory_.size() - next_pc < 4))
        return fail(Rv32Fault::BadJump, pc);
    if (write_rd && rd != 0) regs_[rd] = result;
    regs_[0] = 0;
    pc_ = next_pc;
    if (opcode != 0x23)
        record(pc, word, next_pc, write_rd ? static_cast<u8>(rd) : 32, old_rd);
    instructions_++;
    return true;
}

bool Rv32::run(u64 max_instructions) {
    const u64 limit = instructions_ + max_instructions;
    while (!halted_ && instructions_ < limit) step();
    if (!halted_ && instructions_ == limit) fail(Rv32Fault::InstructionLimit, pc_);
    return fault_ == Rv32Fault::None;
}

u32 rv32_r(u32 funct7, u32 rs2, u32 rs1, u32 funct3, u32 rd, u32 opcode) {
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
           (rd << 7) | opcode;
}
u32 rv32_i(u32 imm, u32 rs1, u32 funct3, u32 rd, u32 opcode) {
    return ((imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}
u32 rv32_s(u32 imm, u32 rs2, u32 rs1, u32 funct3, u32 opcode) {
    return (((imm >> 5) & 0x7f) << 25) | (rs2 << 20) | (rs1 << 15) |
           (funct3 << 12) | ((imm & 31) << 7) | opcode;
}
u32 rv32_b(u32 imm, u32 rs2, u32 rs1, u32 funct3, u32 opcode) {
    return (((imm >> 12) & 1) << 31) | (((imm >> 5) & 0x3f) << 25) |
           (rs2 << 20) | (rs1 << 15) | (funct3 << 12) |
           (((imm >> 1) & 15) << 8) | (((imm >> 11) & 1) << 7) | opcode;
}
u32 rv32_u(u32 imm, u32 rd, u32 opcode) {
    return (imm & 0xfffff000u) | (rd << 7) | opcode;
}
u32 rv32_j(u32 imm, u32 rd, u32 opcode) {
    return (((imm >> 20) & 1) << 31) | (((imm >> 1) & 0x3ff) << 21) |
           (((imm >> 11) & 1) << 20) | (((imm >> 12) & 0xff) << 12) |
           (rd << 7) | opcode;
}

} // namespace krk

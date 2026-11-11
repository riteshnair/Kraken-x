#include "krk/rv32_calc.hpp"

#include "krk/rv32.hpp"

#include <cctype>
#include <cstdint>
#include <limits>
#include <vector>

namespace krk {
namespace {

constexpr u32 kStack = 32768;
constexpr size_t kMaxExpression = 4096;
constexpr u32 kMaxNodes = 256;

void emit(std::vector<u32> *p, u32 word) { p->push_back(word); }
void addi(std::vector<u32> *p, u32 rd, u32 rs1, i32 imm) {
    emit(p, rv32_i(static_cast<u32>(imm), rs1, 0, rd, 0x13));
}
void load_imm(std::vector<u32> *p, u32 rd, u32 value) {
    emit(p, rv32_u((value + 0x800) & 0xfffff000u, rd, 0x37));
    addi(p, rd, rd, static_cast<i32>(value << 20) >> 20);
}

class Parser {
public:
    Parser(const std::string &text, std::vector<u32> *program, std::string *err)
        : text_(text), program_(program), err_(err) {}

    bool parse(i32 *value) {
        if (text_.empty() || text_.size() > kMaxExpression)
            return fail("expression must contain 1 to 4096 bytes");
        i64 v = 0;
        if (!sum(&v)) return false;
        space();
        if (pos_ != text_.size()) return fail("unexpected character at byte " + std::to_string(pos_));
        if (!checked(v, &v)) return false;
        *value = static_cast<i32>(v);
        return true;
    }

private:
    bool fail(const std::string &message) {
        if (err_->empty()) *err_ = message;
        return false;
    }
    void space() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) pos_++;
    }
    bool take(char c) {
        space();
        if (pos_ >= text_.size() || text_[pos_] != c) return false;
        pos_++;
        return true;
    }
    bool node() {
        if (++nodes_ > kMaxNodes) return fail("expression exceeds 256 operations/constants");
        return true;
    }
    bool checked(i64 value, i64 *out) {
        if (value < std::numeric_limits<i32>::min() || value > std::numeric_limits<i32>::max())
            return fail("signed 32-bit overflow");
        *out = value;
        return true;
    }
    bool sum(i64 *value) {
        if (!product(value)) return false;
        while (true) {
            space();
            if (pos_ >= text_.size() || (text_[pos_] != '+' && text_[pos_] != '-')) return true;
            const char op = text_[pos_++];
            const u32 offset = 4 * stack_depth_++;
            emit(program_, rv32_s(offset, 1, 3, 2, 0x23));
            i64 rhs = 0;
            if (!product(&rhs)) return false;
            stack_depth_--;
            if (!checked(op == '+' ? *value + rhs : *value - rhs, value)) return false;
            emit(program_, rv32_i(offset, 3, 2, 2, 0x03));
            emit(program_, rv32_r(op == '-' ? 0x20 : 0, 1, 2, 0, 1, 0x33));
        }
    }
    bool product(i64 *value) {
        if (!unary(value)) return false;
        while (true) {
            space();
            if (pos_ >= text_.size() || (text_[pos_] != '*' && text_[pos_] != '/' && text_[pos_] != '%'))
                return true;
            const char op = text_[pos_++];
            // Preserve the left operand across the right operand's stack use.
            const u32 offset = 4 * stack_depth_++;
            emit(program_, rv32_s(offset, 1, 3, 2, 0x23));
            i64 rhs = 0;
            if (!unary(&rhs)) return false;
            stack_depth_--;
            if ((op == '/' || op == '%') && rhs == 0) return fail("division by zero");
            if ((op == '/' || op == '%') && *value == std::numeric_limits<i32>::min() && rhs == -1) {
                if (op == '%') { // INT_MIN % -1 is exactly zero and RV32 defines it.
                    emit(program_, rv32_i(offset, 3, 2, 2, 0x03));
                    emit(program_, rv32_r(1, 1, 2, 6, 1, 0x33));
                    *value = 0;
                    continue;
                }
                return fail("signed 32-bit overflow");
            }
            if (op == '*') { if (!checked(*value * rhs, value)) return false; }
            else if (op == '/') *value /= rhs;
            else *value %= rhs;
            emit(program_, rv32_i(offset, 3, 2, 2, 0x03));
            const u32 f3 = op == '*' ? 0 : op == '/' ? 4 : 6;
            emit(program_, rv32_r(1, 1, 2, f3, 1, 0x33));
        }
    }
    bool unary(i64 *value) {
        if (!node()) return false;
        space();
        if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
            const bool neg = text_[pos_++] == '-';
            if (!unary(value)) return false;
            if (neg) {
                if (!checked(-*value, value)) return false;
                emit(program_, rv32_r(0x20, 1, 0, 0, 1, 0x33));
            }
            return true;
        }
        return primary(value);
    }
    bool primary(i64 *value) {
        space();
        if (take('(')) {
            if (!sum(value)) return false;
            if (!take(')')) return fail("missing closing parenthesis");
            return true;
        }
        if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_])))
            return fail("expected an integer or parenthesis at byte " + std::to_string(pos_));
        u64 magnitude = 0;
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            const u32 digit = static_cast<u32>(text_[pos_++] - '0');
            if (magnitude > 2147483648ULL / 10 ||
                (magnitude == 2147483648ULL / 10 && digit > 8))
                return fail("integer constant magnitude exceeds 2147483648");
            magnitude = magnitude * 10 + digit;
        }
        *value = static_cast<i64>(magnitude);
        load_imm(program_, 1, static_cast<u32>(magnitude));
        return true;
    }

    const std::string &text_;
    std::vector<u32> *program_;
    std::string *err_;
    size_t pos_ = 0;
    u32 nodes_ = 0;
    u32 stack_depth_ = 0;
};

} // namespace

bool rv32_calculate(const std::string &expression, Rv32CalcResult *result,
                    std::string *err) {
    if (err) err->clear();
    std::string local_err;
    std::vector<u32> program;
    load_imm(&program, 3, kStack);
    i32 expected = 0;
    Parser parser(expression, &program, &local_err);
    if (!parser.parse(&expected)) {
        if (err) *err = local_err;
        return false;
    }
    addi(&program, 17, 0, 93);
    emit(&program, 0x00000073);
    Rv32 machine;
    if (!machine.load(0, program.data(), program.size() * sizeof(u32), &local_err) ||
        !machine.run(10000)) {
        if (err) *err = local_err.empty() ? rv32_fault_name(machine.fault()) : local_err;
        return false;
    }
    if (static_cast<i32>(machine.reg(1)) != expected) {
        if (err) *err = "RV32 result disagrees with checked expression compiler";
        return false;
    }
    if (result) {
        result->value = expected;
        result->instructions = machine.instructions();
        result->state_hash = machine.state_hash();
    }
    return true;
}

} // namespace krk

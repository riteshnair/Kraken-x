// Checked integer expressions executed by the RV32 coprocessor.
#ifndef KRK_RV32_CALC_HPP
#define KRK_RV32_CALC_HPP

#include "krk/common.hpp"

#include <string>

namespace krk {

struct Rv32CalcResult {
    i32 value = 0;
    u64 instructions = 0;
    u64 state_hash = 0;
};

// Supports decimal int32 constants, parentheses, unary +/-, and + - * / %.
// Every operation is checked for signed overflow before its RV32 instruction is
// emitted; malformed input, division by zero, and excessive complexity fail.
// This is deliberately an integer tool, not a floating-point evaluator.
bool rv32_calculate(const std::string &expression, Rv32CalcResult *result,
                    std::string *err);

} // namespace krk

#endif // KRK_RV32_CALC_HPP

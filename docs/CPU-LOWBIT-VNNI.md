# CPU low-bit VNNI and activation-pass experiments

## Implemented

- Q2_K/Q3_K have runtime-dispatched 128-bit VNNI integer dot cores. The AVX-VNNI
  rung serves CPUs such as Meteor Lake; an AVX512VL+AVX512VNNI rung executes the
  same algorithm on AVX-512 hosts. Scalar/SSSE3 remains available.
- Q2_K scalar decode unpacks each 16-value plane directly into a register and
  dots it immediately, avoiding the 256-byte temporary value array. It uses
  AVX2/FMA automatically and the VNNI rung where available. Its x8 VNNI kernel
  measured no benefit and is opt-in with `KRK_CPU_Q23_VNNI=1`; `=0` disables
  Q2_K/Q3_K VNNI. `KRK_CPU_Q23_DIRECT=0` retains the previous scalar decode.
- TQ2_0 x8 computes `q*x - sum(x)` directly from packed codes. It never
  materializes the old positive/negative arrays, handles code 3 as +2, and does
  not negate x (so -128 is exact). Dispatch is outside the row/chunk loops.
- TQ1_0/PTQ1_0 use an immutable byte-to-trit lookup table instead of repeating
  base-3 decode arithmetic per value. TQ2_0 unpacks its bit planes with AVX2.
- Ternary and 2-bit code dots use VNNI where available and retain eight F32 FMA
  lanes across 32-value chunks. This removes a horizontal reduction per chunk.
  `KRK_CPU_TERNARY_LANES=0` restores scalar chunk reductions. Lane accumulation
  changes floating-point reduction order, not integer products.
- Adaptive/residual recovery uses `quantize_q8_row_risk()` to calculate risk in
  the Q8 pass and reuses Q8 activations plus Q4_K/Q6_K sidecars in the ordinary
  GEMM. Residual mode still makes its intentional second matrix pass.

No model, GGUF layout, weight tensor, or default scalar reference is modified.
VNNI dispatch is based on the actual encoding required; AVX-VNNI is not assumed
to imply AVX-512.

## Focused validation

- Q2_K/Q3_K f64 decoder differential: pass.
- Q2_K/Q3_K x8 versus decode bit equality: pass, including forced Q2_K VNNI.
- TQ1/TQ2 extreme signed-byte and all-code tests: pass with lane accumulation on
  and off.
- Activation recovery: 10/10, including bit-exact Q8/sx/sq4 sidecar reuse.
- RV32: 855/855.
- A portable `-mavx2 -mfma -mf16c` build with AVX-VNNI target-dispatch code passes
  the same focused tests. A completely non-AVX2 build retains a pre-existing
  `vec_dot_wide_row` build failure outside this experiment.

## Isolated kernel measurements

Host: Intel Xeon 2.60 GHz, AVX512VL+AVX512VNNI, no AVX-VNNI. Release build,
4096 inputs x 512 outputs, three alternating runs of 40 repetitions, median
GMAC/s. This host exercises the AVX-512VL encoding, so these are not claimed
Meteor Lake measurements.

**Correction:** an earlier table here is withdrawn. The benchmark allocated one
weight block per row although its kernels consumed sixteen blocks, causing an
out-of-bounds read and eventually heap corruption. ASan identified the fixture
error. It now allocates and initializes all sixteen blocks. Do not compare the
old numbers with these corrected runs.

| Kernel | Rows | Previous path | Optimized | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Q2_K direct decode | 1 | 3.551 | 4.345 | 1.22x |
| TQ2_0 raw-code x8 | 8 | 7.068 | 35.594 | 5.04x |

The TQ2 control is `KRK_CPU_TERNARY_LANES=0`; the optimized control is the
default raw-code path. Q2 uses `KRK_CPU_Q23_DIRECT=0` for its control. These
ratios are kernel-only and must not be read as model-wide speedups.

Portable `-mavx2 -mfma -mf16c` with target-dispatched AVX-VNNI, same controls:
Q2_K scalar median 3.442 -> 3.862 GMAC/s (1.12x), and TQ2_0 x8 median 6.465 ->
15.296 GMAC/s (2.37x). The hardware still executes VNNI; this validates the
portable build, not native Zen 3 throughput.

Direct-register Q3_K scalar and x8 variants were also implemented and measured
but rejected: both were slower, and x8 loses the benefit of sharing one unpack
across eight activation rows. Raw-code TQ2 scalar was slower too, so its
expanded-buffer scalar kernel remains. Existing Q3_K VNNI and TQ1/TQ2 SIMD
work is retained, but the withdrawn fixture means it needs a fresh corrected
benchmark before quoting a new ratio.

## Residual recovery cost

`kraken-act-recovery-bench` uses immutable random Q2_K weights, 4096 inputs and
256 outputs. Ten repetitions after warmup produced:

| Rows | Off | Residual | Adaptive | Both |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.231 ms | 0.585 ms | 0.225 ms | 0.716 ms |
| 8 | 0.888 ms | 1.915 ms | 0.882 ms | 1.837 ms |

No fallback fired in this fixture. These are synthetic GEMM timings, not model
accuracy, perplexity, hallucination, or end-to-end tokens/s results.

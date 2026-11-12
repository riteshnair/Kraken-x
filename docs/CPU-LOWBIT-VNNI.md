# CPU low-bit VNNI and activation-pass experiments

## Implemented

- Q2_K/Q3_K have runtime-dispatched 128-bit VNNI integer dot cores. The AVX-VNNI
  rung serves CPUs such as Meteor Lake; an AVX512VL+AVX512VNNI rung executes the
  same algorithm on AVX-512 hosts. Scalar/SSSE3 remains available.
- Q2_K decode uses the VNNI rung automatically. Its x8 VNNI kernel measured no
  benefit and is opt-in with `KRK_CPU_Q23_VNNI=1`; `=0` disables Q2_K/Q3_K VNNI.
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
4096 inputs x 512 outputs, four alternating runs of 40 repetitions, median
GMAC/s. This host exercises the AVX-512VL encoding, so these are not claimed
Meteor Lake measurements.

| Kernel | Rows | Fallback | New | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Q2_K | 1 | 4.374 | 4.405 | 1.01x (neutral) |
| Q2_K | 8 | 7.419 | 7.503 | 1.01x (original x8 default) |
| Q3_K | 1 | 4.993 | 5.712 | 1.14x |
| Q3_K | 8 | 15.412 | 17.572 | 1.14x |
| TQ1_0 | 1 | 3.051 | 3.586 | 1.18x |
| TQ1_0 | 8 | 6.353 | 9.867 | 1.55x |
| TQ2_0 | 1 | 8.322 | 11.733 | 1.41x |
| TQ2_0 | 8 | 8.567 | 14.034 | 1.64x |

Run `kraken-lowbit-vnni-bench TYPE ROWS REPS`. The fallback arm sets
`KRK_CPU_Q23_VNNI=0 KRK_CPU_TERNARY_LANES=0 KRK_CPU_NO_AVXVNNI=1
KRK_CPU_NO_VLVNNI=1`; it still includes the LUT/SIMD unpackers. Earlier
pre-LUT/pre-SIMD measurements were much slower, so the table prices the VNNI
and lane-reduction stage, not the entire improvement from the original branch.

## Residual recovery cost

`kraken-act-recovery-bench` uses immutable random Q2_K weights, 4096 inputs and
256 outputs. Ten repetitions after warmup produced:

| Rows | Off | Residual | Adaptive | Both |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.231 ms | 0.585 ms | 0.225 ms | 0.716 ms |
| 8 | 0.888 ms | 1.915 ms | 0.882 ms | 1.837 ms |

No fallback fired in this fixture. These are synthetic GEMM timings, not model
accuracy, perplexity, hallucination, or end-to-end tokens/s results.

# Runtime activation recovery (CPU experiment)

This is an opt-in numerical experiment. It does **not** modify a GGUF, add repair
sidecars, reconstruct discarded weight information, or change NPU/device execution.
The default is `off`; the original GEMM and grouped-GEMM paths are unchanged when off.

## Modes

Set `KRK_ACT_RECOVER` before starting Kraken:

| Value | Behavior |
| --- | --- |
| `off`, `0`, or unset | Original execution (default). |
| `residual`, `1` | Original int8 GEMM plus `Wq * (x - Q(x))`. |
| `adaptive`, `2` | Select existing F32-activation GEMM for high-risk rows; otherwise original int8 GEMM. |
| `both`, `3` | Residual correction for low-risk rows and F32 fallback for high-risk rows. |

Weights remain in their existing quantized format. The residual correction performs
another F32-activation matrix pass; it is not an acceleration mechanism. It compensates
only activation rounding relative to the quantized weights. In particular, it does not
compute `(W - Wq) * x`.

Adaptive mode evaluates residual RMS relative to input RMS and the fraction of nonzero
values discarded by activation quantization, fused into the Q8 quantization pass. The
resulting Q8 activations and Q4_K/Q6_K sidecars are reused by the base GEMM rather than
quantizing a second time. Defaults are 0.35 relative residual and
0.05 discarded mass; either crossing selects F32. Tune with `KRK_ACT_RISK` and
`KRK_ACT_DISCARD`. Both values must be nonnegative. The counters are printed at process
exit as GEMM calls, residual corrections, and F32 fallbacks. Experimental grouped GEMMs
delegate to the per-matrix implementation for identical policy behavior.

Eligibility requires the scalar CPU backend, existing int8 activation support, and an
input dimension divisible by 256. I2_S and NPU execution retain their original paths.

## Verification

`kraken-act-recovery-tests` checks native/RV32 agreement for the signed TQ1_0 integer
dot, residual-error reduction, exact adaptive F32 fallback, grouped consistency, and
unchanged default results. `kraken-rv32-tests` retains the RV32 arithmetic oracle suite.

RV32 verifies the ternary integer core, not floating-point scale/reduction arithmetic
or end-to-end transformer accuracy. It is a reference, not an inference accelerator.

## Evaluation limits

Unchanged Tiny-LLM Q2_K, Q4_K_M, and Q6_K GGUFs were exercised in all modes. Short
greedy generations can differ between modes, but a different continuation is not proof
of improved factual quality. No perplexity, teacher-logit, hallucination-reduction,
or representative large-model speed claim is established by these smoke tests.
Measure those separately with a fixed dataset, warmed-up repeated runs, and identical
sampling settings. Fallback frequency and temporary residual/output memory must also
be included in a performance evaluation.

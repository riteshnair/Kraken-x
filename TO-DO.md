# TO-DO — Kraken-Xe (CPU-native GGUF engine)

Status as of 2026-11-12. Every number below was measured on this machine, not
estimated. Where a claim was measured **negative**, it is recorded as such — the
negatives are the expensive part and re-deriving them is the main way to waste a
session.

Target machine: **Core Ultra 5 135H** — 14 cores / 18 threads (4 P + 8 E + 2
LP-E), AVX2 + FMA3 + 256-bit AVX-VNNI, **no AVX-512**, and **one 16 GB
DDR5-5600 module = single channel**.

Goal: large MoE models under a ~4 GB RAM cap, 15–20 tok/s, at least 64K context.

---

## 1. The hardware ceilings (measure these before believing any speed claim)

`.\build-q4k\kraken-membw.exe --mb 512 --threads N` — achieved read bandwidth,
median of 9:

| threads | 1 | 2 | 4 | **8** | 14 | 18 |
|---|---|---|---|---|---|---|
| GB/s | 12.3 | 16.5 | 20.1 | **26.5** | 24.8 | 20.4 |

Consequences that are now load-bearing:

- **8 threads is the peak and 18 is 23% below it.** The pool default is already
  8 (the fast cluster's logical count); `cu_count` = 18 is only a report.
  Prefill measured **21.29 tok/s at 8** against 15.63 at 10 and 15.06 at 14.
- **P-cores-only is a loss**, not a win: `--threads 4` against the default over
  5 interleaved order-balanced pairs is **−37.04% prefill / −26.32% decode**.
- **Single-channel memory halves the ceiling.** A second SODIMM (~2× on every
  bandwidth-bound path) is worth more than any remaining kernel work. It also
  lifts the iGPU, which shares this same controller.

## 2. Where the machine stands today

Baseline, `C:\models\Qwen3.8-4B-Q4_K_M.gguf` (2.59 GB, dense), greedy, ctx 512:

| metric | value |
|---|---|
| decode | 3.0–4.1 tok/s (run-to-run spread is large; only interleaved A/B counts) |
| prefill (22 rows) | 5.6 tok/s — only 1.4–1.9× decode |
| implied decode traffic | ~10 GB/s of weights ≈ the single-channel ceiling |

- The **Q4_K dot kernel is at 70–93% of the single-core wall** (8.3–11.4 GB/s
  against 12.3). There is at most ~1.1× left inside one core. Stop tuning it.
- Dense decode is **memory-bound**: a +15–19% kernel improvement moved it by
  **0.00%**. Any decode pitch must reduce *bytes*, not instructions.

---

## 3. P0 — THE KEYSTONE: x8 ↔ r1 arithmetic equality (unblocks three things)

**The CPU backend's answer currently depends on `--chunk`, for every K-quant
model.** Verified directly on 4 short prompts — 2 of 4 differ:

```
chunk1 AB9A36C6D84F  chunk8 AB9A36C6D84F  same
chunk1 1F3F7578C1BE  chunk8 1F3F7578C1BE  same
chunk1 E49B40071691  chunk8 F5115A9CD1A7  DIFFERS   <- "Once upon a time"
chunk1 91892D63E4D3  chunk8 13FB875C9F9D  DIFFERS   <- "2 + 2 ="
```

Cause: rows inside an 8-group run `dot_*_x8`, remainder rows run `dot_*`
(single row), and they disagree at ~2e-7 relative because they structure the
scale differently:

- **x8**: `(a*d)*dx`, min term mixed into the vector accumulator
- **r1**: `dx*d*a`, scalar `macc` subtracted at the end

Measured kernel-level: Q4_K `2.078e-07`, Q6_K `6.176e-07`. **Q8_0 is invariant**,
which is why this hid — the one format that agrees is the one people test.

At `--chunk 1` every row is r1 (`rows8 = 0`); at `--chunk 8` every row is x8.

### Why it is P0

1. **Chunk invariance is a correctness property.** Chunk size is an internal
   batching choice; text must not depend on it.
2. **It makes speculation only accidentally bit-identical.** The verify is a
   multi-row forward, plain decode is single-row, so those paths compute
   different numbers. n-gram's byte-identical result (below) holds by luck.
3. **It blocks the −19% prefill padding win** (section 5), because the tail
   "correct" value is r1's and r1 ≠ x8.

### Plan

Make x8 reproduce r1's per-row chain, hoisting only the block decode:

1. Make `dot_*_x8` numerically equal to the `dot_*` rung it dispatches to, for
   Q4_K / Q6_K / Q8_0, falling back to the per-row call where it cannot.
2. Prove `chunk 1 == 8 == 16` on 3 models (dense K-quant, MoE K-quant, Q8_0).
3. Re-record the test baseline deliberately — **every pinned hash moves once**.
4. Then re-apply `C:\Users\rina0423\AppData\Local\Temp\5\opencode\padded-tail-REVERTED.patch`
   and re-measure.

Cost: one-time engine-wide numeric change. This is a decision, not a chore.
Being exact costs the per-row scale products and the scalar min chain, so the
padding win will land nearer **10–15%** than 19%.

**Note:** the KL gate would NOT have caught this. Default `logits_kl_max` is
**0.02**; the measured divergence was 0.00103 — 20× inside the threshold. A
tighter gate (or an exact-match mode) is worth adding alongside.

---

## 4. P1 — Reduced expert bytes on MoE (the stated primary goal)

Ranked by measured effect. **Everything that helps reduces bytes per token;
everything that tries to schedule bytes better has lost.**

| lever | evidence | status |
|---|---|---|
| HOT hit rate via `--router-resident` | GLM-4.7-Flash: **0.9 → 1.5 tok/s (+67%)**, HOT 51→84%, reads 19.8→6.8 s | ✅ works, keep |
| ternary multiply-free dot (2 bpw vs Q4_K 4.8) | 2.4× fewer bytes; arithmetic is exact so the gate is **byte-identical output** | ⬜ open |
| top-k 4→2 distillation | halves bytes **and** FLOPs | ⬜ open — needs a teacher-forced PPL harness |
| expert prediction / next-layer prefetch | **−9%** (114.4 vs 124.7 ms/step on REAP-320) — promotion traffic is bandwidth-bound, so moving a copy earlier buys nothing; a *better* predictor cannot fix it either | ❌ do not revive |
| speculation on MoE | net loss at every draft length (see section 6) | ❌ |

Caveat on `--router-resident`: it does **nothing** at 0% HOT — there is no
resident tier to prefer. It only pays once a tier exists.

## 5. P2 — Prefill: the padding fix, waiting on P0

`src/backend_cpu.cpp`, `gemm()`, two loops (~1718 and ~1728):

```cpp
for (i64 r = rows8; r < rows; r++)              // remainder rows
    for (i64 o = o0; o < o1; o++)
        of[r * n_out + o] = dot1(wb + o * row_bytes, r, o);
```

The weight row is read from DRAM **once** per output column — this is **not** a
bandwidth defect (an earlier note in the repo said it was; it is not). The
remainder re-*decodes* the same block from L1/L2, so the waste is instructions.

Fix (written, measured, reverted only on bit-identity): pad activation rows to a
multiple of 8, duplicate the last real row into the padding, give `dot8` an
explicit destination base, let the tail group write to `8 × n_out` scratch, copy
back only the real rows.

Measured on `Qwen3.8-4B-Q4_K_M`, same binary, A/B on `KRK_CPU_GEMM_TAIL8`,
interleaved order-balanced, 4 pairs:

| rows | rows%8 | pad-on | pad-off | delta | text |
|---|---|---|---|---|---|
| 9 | 1 | 1011.7 ms | 772.6 ms | **+31.0%** (regression) | differs |
| 20 | 4 | 1584.4 | 1703.5 | −7.0% | same |
| **22** | **6** | **1453.0** | **1799.0** | **−19.2%** | same |
| 24 | 0 | 1488 | 1536 | −3.1% / +1.4% | same (control: must not move) |

Decode unchanged (path is `rows > 1` only). The multiple-of-8 control correctly
does not move, so the change does what it claims.

- **Gate the padding to `rows % 8 >= 5`** — at `rows % 8 <= 3` it is a
  regression by construction (measured +31% at 1).
- Root cause, model-free: **22 rows costs 25.2% more than 24 rows.**

### The 8× prefill target is NOT reachable this way

A profiled 8-row group costs **4.1 single-row dots**, not 1, so sharing the
weight decode buys **~2× per row, not 8×**. And the GEMM runs at **108 GMAC/s =
7.9% of this host's 1197 GMAC/s int8 roofline**, bound by ~1 byte of int8
*activation* per MAC: an 8-row group reads 8×256 B of activations against a
144 B weight block — **activations dominate 14:1**. No amount of weight-decode
sharing reduces that. A weight-stationary layout or a wider activation type is
what would, and that is a different project.

## 6. P3 — Speculation: correct, but economically blocked

| path | mechanism | measured |
|---|---|---|
| n-gram (prompt lookup) | tokens that followed this n-gram last time it appeared | ✅ byte-identical, **63→50 rounds**; ❌ **12.5 → 9.1 tok/s** |
| MTP (nextn block) | block ships inside the target file | ✅ **100%** self-draft (52/52), 96.2% at n=96; ❌ Tiel 1.45 tok/round (33%) |
| DFlash head set | published head | ❌ 0/128 accepted on the laguna pair |
| draft model | second GGUF | ❌ self-draft was 27.8% before F8 was fixed |

n-gram is implemented and committed (`f3dd4c7`), **off by default**
(`KRK_NGRAM=1`, `KRK_NGRAM_N`=3, `KRK_NGRAM_K`=4), and refused on a delta-rule
recurrent target (no invertible rewind).

It loses for exactly one reason: a round is a *d*-row batched forward, so
**109.4 ms/round against 80.1 ms/step** — it pays a multi-row forward to buy
back ~1.2 tokens. **Fix P0 and this becomes free upside**, because it needs no
drafter at all.

## 7. P4 — MLA packed KV (64K context on GLM / DeepSeek only)

Not an optimisation — the enabling piece. GLM currently falls back to f16:

```
[warn] kv-quant turbo3 is the default but the MLA cache is a latent row with its
       own read/write path -- falling back to f16 for this model
```

**106 KiB per position** (53 MiB at ctx 512):

| ctx | f16 | int8 | turbo3 dual-copy |
|---|---|---|---|
| 8K | 846 MiB | 470 MiB | 166 MiB |
| **64K** | **6.9 GB** | 3.8 GB | **1.35 GB** |

6.9 GB of KV alone breaks a 4 GB budget outright, so MLA + 64K is a non-starter
today. Why it is not a one-liner: an MLA cache row is **one latent vector used
two ways** — the last 64 dims are roped and feed scores, all 512 feed values — so
a rotation-based packer needs a **dual copy**. 512 + 64 = 576 splits exactly on
the rope boundary, which is what makes it tractable (~5.1×).

**Scope note:** this matters for MLA files only. The 35B-A3B target is
`qwen35moe` (hybrid GDN), not MLA.

## 8. P5 — Small, contained, known

- **Q8_0 has no 256-bit AVX-VNNI rung.** `tools/dispatch_audit.py` reports
  **14 entries, 4 gaps**, all Q8_0: `dot_q8_0_q8` (the rows=1 decode entry),
  `dot_q8_0_q8_wsc8`, `dot_q8_0_q8_x8_tier`, `dot_q8_0_q8_x8`. K-quants are
  fully covered. ~150 lines exist uncommitted in `src/quant.cpp` from a dead
  worker; verify then commit, or redo.
- **`--kv-quant` help text says "default rot4"** while the actual default is
  turbo3.
- **`KRK_CPU_DOT_INT8` is a dead switch** — `vec_dot_int8()` returns true
  unconditionally under AVX2.

## 9. P6 — OpenVINO (proven reachable; value limited)

SDK **2026.1** is installed under `oneAPI\runtime` (not a standalone install —
a `Program Files\Intel\openvino*` search finds nothing and reads as "not
installed"). `tools/ov_probe.cpp` links it from **mingw g++** via the **C API**
and enumerates **CPU / GPU / NPU**, `rc=0`.

- Use the **C API**: the runtime is MSVC-built while this project is mingw, and
  passing `std::string`/exceptions across that boundary is undefined.
- The import lib links and the process then dies with **0xC0000135** — that is
  **`tbb12.dll`**, not openvino. `objdump -p openvino.dll` names it.
- **The Xe iGPU reports `uma: 1`** — unified memory, i.e. it reads through the
  same single-channel controller. **Decode cannot improve there.** Only
  prefill-shaped compute could, and that is blocked behind P2/P0.
- Do **not** use llama.cpp's `ggml-openvino` plugin as the model: it converts
  the graph to OpenVINO IR at load, materialising a second full copy of the
  weights. A **2.6 GB** model took this 16 GB box to the wall and hung.

---

## 10. Measured negatives — do not re-derive

| tried | result |
|---|---|
| Multi-accumulator "tiling" of the Q4_K dot (4 chains) | **−13 to −24%** across every run; kernel is issue-bound, not FMA-chain-bound. `dot_q4_K_q8_tile` is kept as the bench's negative arm. |
| Expert next-layer prefetch / prediction | **−9%** (114.4 vs 124.7 ms/step, REAP-320) |
| `--threads 4` (P-cores only) | **−37% prefill / −26% decode** |
| `--threads 10/14` | −27% / −29% prefill |
| Speculation on MoE (MTP) | loss at every draft length (20.6–21.8 → 7.7–8.5 tok/s) |
| Weight-repack / tile-shape changes to the Q4_K dot | no headroom: kernel is at 70–93% of the single-core wall |
| iGPU / Vulkan for decode | ceiling is the memory controller, not compute; `uma: 1` |

## 11. Harness traps that have each cost a session

1. **A harness bug that makes both arms identical reports "no difference"** and
   reads as a clean negative. It happened **three times** here: an arm table
   comparing a function against itself after it became the new default; a
   fixture whose f16 scales were powers of two (2^-6, 2^-8) making every partial
   sum exactly representable and reporting `maxrel 0.0` for four different
   kernels; and a "gate" that ran on a model the feature *refused*, so both arms
   ran plain decode. **Always prove the arms differ** — print one value from
   each, or check that the round/step count actually changed.
2. **A hang is not a slow engine.** `Pool::set_workers` held `mu_` across
   `join()`; `--threads N` hung **4 runs in 12** and a sweep read it as
   0.5 tok/s at 1 thread. Fixed in `88ce9d2`.
3. **Never benchmark beside another workload.** Two arms run concurrently
   produced 4–6 s prefills and 1.3 tok/s. This box is power-limited and drifts
   up to 2× sweep-to-sweep; only **within-sweep, interleaved, order-balanced
   pairs** are trustworthy, and the first run after idle is ~40% faster.
4. **`--profile` distorts a decode step ~10×.** Never use it in an A/B.
5. **The trace is the signal that never goes blind**; a logits-only or
   text-only comparison can pass while the arithmetic differs (see section 1's
   chunk-invariance defect, which text comparison missed on 2 of 4 prompts).

---

## 12. Build / run

```powershell
cd C:\Users\rina0423\Desktop\Kraken-Xe
$env:PATH="C:\w64devkit\bin;"+$env:PATH
ninja -C build-cpu-win kraken          # main build (has the test suite)
ninja -C build-q4k kraken kraken-gemv-q4k kraken-membw   # fast scratch (tests OFF, gitignored)
```

Useful harnesses:

| tool | answers |
|---|---|
| `kraken-membw --mb 512 --threads N` | achievable read bandwidth — the ceiling |
| `kraken-gemv-q4k --nin 2048 --nout 14336` | marginal Q4_K dot cost; 4 arms incl. the negatives |
| `tools/dispatch_audit.py` | which int8 kernels have a VNNI rung, and which tier this host takes |
| `tools/ov_probe.cpp` | is the OpenVINO SDK linkable and are its devices reachable |
| `--logits-ref` / `--logits-steps` | KL drift against a reference (default threshold 0.02 — see section 3) |

---

## 13. Recommended order

1. **P0** — x8↔r1 equality. One-time numeric change; unblocks correctness,
   speculation and the prefill win. Everything else is second-order to this.
2. **P1** — reduced expert bytes (ternary dot, then top-k distillation + PPL).
3. **P2** — re-apply the padding patch once P0 lands (gate to `rows % 8 >= 5`).
4. **P5** — Q8_0 AVX-VNNI rung (small, contained, only affects Q8_0 files).
5. **P4** — MLA dual-copy packed KV, only if 64K on GLM/DeepSeek is a goal.
6. **P6** — OpenVINO prefill, only after P0/P2 make prefill compute-bound.

Hardware, if it is on the table: **a second SODIMM is ~2× on everything
bandwidth-bound and no amount of code beats it.**

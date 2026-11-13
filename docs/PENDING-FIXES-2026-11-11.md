# Pending fixes — 2026-11-11

This register is for the low-memory MoE / quantized-KV / branch-fork work. It
reconciles the earlier pending list with the merged bounded-memory benchmark
work. It is not a claim that the papers' reported performance has been reproduced.

## Hard constraints

- Keep the low-memory objective. Do not exchange lower RSS for eager expert
  warming, pinning, an unbounded cache, or a larger default resident set.
- Keep expert weights demand-loaded from the GGUF mapping where possible. Bound
  resident expert slots/bytes and fail closed under pressure; a slower load is
  preferable to silently exceeding the memory plan or changing the computed
  model.
- Treat the one-expert-slot benchmark as a minimal-residency stress profile, not
  an automatic production recommendation. The expert-slot cap is not a total
  process-RSS cap.
- Never lift the int8 multi-slot or lossy-fork gates without the matching
  standalone-output parity evidence.
- Require performance, peak-RSS, residency, and output-quality measurements
  together. A synthetic fixture or `--ram-tier` simulation is not a real-model
  memory result.

## Done / current baseline

- **Copy-backed f16 forks:** ordinary attention copies trunk KV rows; recurrent
  and compressed-attention forks also restore prompt-boundary snapshots. CPU
  fixture `test_batch_fork` passed 140/140 checks. This is a correctness baseline,
  not shared-page storage or a speedup claim.
- **Bounded-memory benchmark driver:** `scripts/bench_bounded_moe.sh` runs Kraken's
  built-in CPU MoE benchmark at planned 16/24/32 GiB tiers. By default it caps
  the expert pool at one `(layer, expert)` slot, disables eager warmup, WARM
  duplication, and pinning, records process peak RSS with
  `scripts/measure_peak_rss.py`, writes host/cgroup and benchmark-setting
  metadata, and compares greedy output across tiers. A full model hash is
  optional (`KRK_BENCH_HASH_MODEL=1`) because it reads the entire checkpoint.
  Each `--bench` run is one process startup and one request; the separate
  generation pass is a parity check, not a repeated warm-request measurement.
  `KRK_BENCH_EXPERT_SLOTS=auto` selects the normal automatic slot budget.
- The driver passed a smoke test on the synthetic tiny-MoE fixture, including
  exact output equality at the three planned tiers. `test_moe_schema_and_laziness`
  passed 45/45 checks, including one-slot residency and equality with a larger
  cache.
- **No real 30B-A3B bounded-memory result exists yet.** The sandbox used for the
  smoke test had 3.8 GiB RAM and no 30B checkpoint. Planned `--ram-tier` values
  are policy simulations, not hard OS limits; the engine clamps them to installed
  RAM. Do not quote the synthetic RSS or rates as a real-model result.

## Ordered pending work

### 1. Fix int8 KV multi-slot slot-zero/nonzero parity — known correctness bug

**Observed:** a CPU tiny-model probe generated `5 5 5 5` in slot zero and
`5 5 3 1` in a nonzero slot with int8 KV. Serializing batch processing did not
fix it. The old divergent probe was replaced by a safety-refusal assertion, so it
must be restored as a failing regression before the implementation is changed.

**Current guard:** `Engine::batch_refusal()` refuses multi-slot int8 sessions;
`Engine::batch_fork()` refuses forks for every lossy KV format. Keep both rules.

**Work:** instrument the earliest divergent position and compare K/V data,
quantization scales, row bases, and strides from slot zero, nonzero slots, and a
single-slot oracle. Investigate prefill and decode separately. Do not assume a
scheduler race: serialization was already tried without success.

**Acceptance:** a persistent test compares logits at each step plus greedy
outputs for slots 0, 1, and a later slot; include unequal prompts, chunk
boundaries, and slot isolation. Require exact CPU equality if the kernels match;
document any backend-specific numeric tolerance. Add native device coverage before
lifting the int8 gate there.

### 2. Establish per-format quantized batch and fork parity

The KV formats are f16, int8, Turbo3, and Rot4. Only int8 multi-slot sessions
currently have the explicit `batch_refusal()` gate; this does not prove other
formats correct. Forks remain disabled for every non-f16 format. Existing Turbo3
HTTP fixture coverage is not the full slot/backend/fork matrix.

**Work / acceptance:** parameterize solo-versus-multi-slot tests across int8,
Turbo3, and Rot4 using the same model and KV format for oracle and batch runs.
Cover `batch_open`/`batch_join`, HTTP admission, slot isolation, and forks as
separate paths. Gate each unsupported format/backend combination with a tested
refusal. Passing batch decode alone does not authorize forks.

### 3. Run the actual bounded-memory MoE experiment

Use a named 30B-A3B-class GGUF and native hosts with 16, 24, and 32 GiB RAM. Run
`scripts/bench_bounded_moe.sh MODEL.gguf ./build/kraken OUTPUT_DIR`; preserve every
log. The default is a one-slot expert-pool stress run with eager warmup, WARM
copying, and expert pinning disabled. Use `KRK_BENCH_EXPERT_SLOTS=auto` only for a
separate normal-policy comparison.

**Record:** model/checkpoint and hash, quantization, prompt/token count, host
CPU/topology and installed RAM, OS, exact invocation, actual peak process RSS,
model mapping size, resident expert slots/bytes, cache hits/evictions, cold bytes
and read time, output parity, prefill rate, and decode rate. Run on each actual
RAM class; a large host with `--ram-tier 16` only checks the planner. The cap on
expert slots is not a system memory limit.

**Acceptance:** one-slot mode completes without OOM or quality drift; logs show
resident expert count never exceeds the requested slot cap, peak RSS stays within
the declared target on the matching host, and generated output is identical
across cache policies. Where the target requires a hard process/system ceiling,
run under the OS memory controller with swap disabled, and record cgroup peak,
swap, and OOM counters as well as process RSS; `--ram-tier` is never presented
as enforcement. Separate cold-start, first-request/cold-cache, and repeated
warm-request results. Report unsupported memory classes plainly rather than
claiming they fit.

### 4. Preserve and broaden f16 prompt-boundary snapshot validation

Keep the standalone-oracle coverage for dense, recurrent, and compressed
attention, nested/sibling forks, and parent continuation. Add native HIP/device
coverage for snapshot capture, reset/reuse, and suffix decode, plus prompt
boundaries at chunk edges and near context limits. If a model/backend cannot
preserve the invariant, refuse the operation; do not silently use an unverified
fallback.

### 5. Add paged KV storage and copy-on-write

The current KV storage is contiguous per slot and the fork path copies rows. Add
fixed-size pages, a per-sequence block table, page ownership/refcounts, and
copy-on-write when a shared page gets a second writer. This should reduce KV
reservation/duplication; it does not reduce expert-weight residency directly.

**Acceptance:** existing batch and prefill exactness tests remain green; add
page-boundary, allocation-failure, reset/close, slot reuse, and refcount-balance
tests. Measure single-sequence regressions and actual memory saved before claiming
a benefit.

### 6. Add trunk/branch visibility metadata for shared-tree forks

Pages alone do not say which branch may read which generated rows. Extend the
row/attention metadata to represent trunk, branch ancestry, and position. A branch
may see the shared prompt and its own ancestors, never a sibling's suffix.

**Acceptance:** compare nested and sibling branch logits/tokens with isolated
standalone runs, including page boundaries and slot/page reuse. Land only after
paged storage is correct. This is the missing shared-trunk part of Flide; the
current copy-backed f16 forks are not that implementation.

### 7. Add the per-option scoring surface before MMDM

Implement option-normalized output-head scores on both serial and session paths,
including explicit multi-token option semantics. Compare against standalone
full-prefix scoring. Then run MMDM-style decision evaluation with a named
checkpoint and reference outputs. The benchmark alone cannot close this item
while the scoring API is absent.

### 8. Keep Edge0 prediction-as-routing experimental and artifact-gated

Kraken already has lazy expert loading, bounded residency, read-ahead, and a
route predictor used for **staging only**: a wrong guess can waste an I/O read,
but it does not change the selected experts or model output. The paper's trained
predictor that substitutes its predicted experts for the router is not
implemented; no compatible trained artifact or evaluation is present.

Do not change routes or add resident experts by default. If a named,
model-specific predictor is supplied, evaluate it as an explicitly opted-in
quality-changing mode: route agreement, cold bytes, peak RSS/resident experts,
latency, held-out perplexity/task quality, and batch/solo isolation. Preserve
ordinary routing as the exact fallback. The paper's reported quality does not
transfer without its predictor and recovery adapter.

## Lower-priority / non-engine items from the paper list

- **LowRAM:** value amortisation or low-rank KV compression is an opt-in research
  experiment after a named checkpoint is available. It targets KV memory, not
  expert residency. Measure reconstruction error and output quality before
  optimizing speed.
- **e-means:** reference for a future offline codebook-learning tool only; no
  inference-time clustering task is pending.
- **SparseRead:** agent-side prompt/content reading, not the inference engine;
  out of scope unless Kraken grows an agent/retrieval surface.
- **Tree-Ising:** sidecar investigation only. Keep source GGUF read-only; any
  separately named repair overlay must be opt-in and pass checkpoint/layout and
  held-out quality gates.

For unrelated repository tasks, use `TO-DO.md`.

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
  `KRK_BENCH_REPEATS` runs multiple requests in one process: a first request
  followed by repeats. Per-request HOT/WARM hits and COLD reads reveal whether
  repeats are actually warm; exact token/text/finish parity is required. The
  separate generation pass checks output across RAM tiers.
  `KRK_BENCH_EXPERT_SLOTS=auto` selects the normal automatic slot budget.
- The driver passed a repeat-enabled smoke test on the synthetic tiny-MoE
  fixture: each tier ran one first request plus two same-process repeats, the
  repeat outputs matched exactly, all three planned tiers matched, and the pool
  stayed at one resident slot. The fixture emitted only one token before EOG
  and retained no useful expert hits: each repeat had 0 HOT/WARM hits and 4
  COLD reads. This validates honest cold-versus-repeat reporting, not a warm-cache
  speedup; its rate/RSS is not a real-model performance or fit result.
  `test_moe_schema_and_laziness` passed 45/45 checks, including one-slot
  residency and equality with a larger cache. `test_cli_arg_refusal` passed
  129/129 checks including repeat-option validation.
- **No real 30B-A3B bounded-memory result exists yet.** The sandbox used for the
  smoke test had 3.8 GiB RAM and no 30B checkpoint. Planned `--ram-tier` values
  are policy simulations, not hard OS limits; the engine clamps them to installed
  RAM. Do not quote the synthetic RSS or rates as a real-model result.

## Ordered pending work

### 1. Int8 KV multi-slot slot-zero/nonzero parity — CPU fix validated

**Root cause and fix:** CPU int8 attention included the per-slot base when
reading cached keys, but omitted it when reading cached values. Nonzero slots
therefore attended to their own keys and slot zero's values. The CPU value read
now adds the row's `kvb`, matching the key read and the append address.

**Evidence:** `test_batch_decode` compares greedy outputs from four standalone
CPU int8 runs with a four-slot batch, using distinct prompts of different
lengths. All slots, including slots 1–3, match exactly. The int8-specific global
batch refusal was removed; the backend's existing `attn_multiseq_supported()`
check still refuses backends without per-row cache addressing. The lossy-format
fork refusal remains unchanged. CPU batch decode is covered; native-device
parity and quantized forks are not.

**Remaining coverage:** add native-device parity before enabling int8 batches on
any device backend that advertises per-row attention. The separate per-format
batch/fork matrix below remains open; batch parity does not authorize forks.

### 2. Establish per-format quantized batch and fork parity

The KV formats are f16, int8, Turbo3, and Rot4. CPU `test_batch_decode` now
compares four standalone generations with a four-slot batch for int8 and Turbo3
on the tiny fixture, and Rot4 on a matching 32-wide-head fixture. Each format
uses distinct prompts of different lengths; outputs match exactly. This is
fixture-level decode evidence, not full backend/API coverage. Forks remain
disabled for every non-f16 format.

**Remaining work / acceptance:** cover `batch_open`/`batch_join` and HTTP
admission for each supported format; keep unsupported backend/shape combinations
behind tested refusals. Add native-device parity where a backend advertises
per-row attention. Test lossy forks separately; passing batch decode does not
authorize them.

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

The engine's active KV storage is still contiguous per slot and the fork path
still copies rows. A bounded, lazy-payload page arena and sequence block-table primitive exist
in `include/krk/kv_pages.hpp` / `src/kv_pages.cpp`; they cover shared references,
write-time COW, truncation/reset, exhaustion, and reuse. `AttnDesc` now carries
optional per-row physical K/V page tables, and the CPU reference backend honors
them in KV append and dense/sparse attention (including packed cache row
addressing); incomplete tables fail closed. HIP and NPU do not advertise the
capability. Backend tests compare page-mapped append/attention with contiguous
results across page boundaries, including int8 rows; a CPU sibling-fork fixture
checks that two branches see the shared trunk plus only their own suffix. The
Engine still allocates and addresses active KV contiguously, and batch forks
still copy rows, so the runtime does not yet save memory or fork-copy bandwidth.
The next work is wiring tables into Engine allocation, per-sequence lifecycle,
rollback, reset/close and fork paths; the current branch-local ancestry test is
not Engine fork coverage. This should reduce KV reservation/duplication; it does
not reduce expert-weight residency directly.

**Acceptance:** existing batch and prefill exactness tests remain green; add
page-boundary, allocation-failure, reset/close, slot reuse, and refcount-balance
tests. Measure single-sequence regressions and actual memory saved before claiming
a benefit.

### 6. Add trunk/branch visibility metadata for shared-tree forks

The CPU backend fixture now checks the core page-table invariant directly: two
prefix-cloned sibling tables share trunk pages, detach a partial page on write,
and causal row positions make each answer see only its own suffix. This is not
yet Engine/session coverage. Wire ancestry/prefix limits into the sequence and
attention row tables so an Engine fork may see the shared prompt and its own
ancestors, never a sibling's suffix.

**Acceptance:** compare nested and sibling branch logits/tokens with isolated
standalone runs through the Engine, including page boundaries and slot/page
reuse. Land only after paged storage is correct. This is the missing shared-trunk
part of Flide; the current copy-backed f16 forks are not that implementation.

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

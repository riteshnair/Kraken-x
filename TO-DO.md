# Kraken pending work

Status: 2026-11-11, after an eight-paper evaluation and the first copy-backed
branch-fork implementation: the register tracks the pages/forks/scoring path and
records both firm constraints and opt-in investigations. See
`docs/PAPER-EVALUATION-2026-11.md`.

This is the only pending-work list. Historical plans, audits, campaign logs, and
transcripts were removed because their claims and line references had become
stale. Current behavior must be established from code, tests, and reproducible
runs.

## Evidence rules

- The release CPU suite passes 8185/8185 checks; the ASan build passes 8184/8184
  across all 187 groups (`build-cpu`, and `build-asan` with
  `detect_leaks=1:halt_on_error=1`). The one-check difference is deliberate: the
  wall-clock ratio gate in `test_batch_decode` is skipped under instrumentation,
  because ASan re-times two code paths unevenly and the number would be about the
  sanitizer. ASan runs every group, so a leak in one is not hidden by another.
- ASan validation uses `detect_leaks=1:halt_on_error=1`, without suppressions.
- A synthetic fixture proves only the behavior it constructs. It does not prove
  real-model coherence, output quality, acceptance rate, or throughput.
- Performance claims require the model, hardware, workload, controls, and scope.
  Kernel throughput is not model-wide tokens per second.
- Do not silently convert models or GGUF weights during optimization.
- Lossy changes require quality evaluation and explicit opt-in.
- Revisit proposed optimizations when requested, but keep quality-changing modes
  opt-in and require the stated correctness/quality evidence before promotion.

Status labels:

- **CODE**: implementation can proceed in this repository.
- **MEASURE**: establish current behavior before changing it.
- **FILE**: blocked by a named model, head set, or reference output.
- **HARDWARE**: blocked by unavailable native hardware or operating system.

## P0: correctness against independent references

- [ ] **OLMoE shared path** (FILE). Run the exact model and bounded llama.cpp
  oracle on the same prompt. Close when texts agree or a stage diff identifies
  the first divergence. CPU/device agreement alone is insufficient.
- [ ] **DFlash and DSpark acceptance** (FILE). Use a matched DFlash2 head or
  upstream reference logits. Check the weight format before blaming checkpoint
  pairing. Close with a sharp proposer row and measured acceptance, or a named
  graph divergence.
- [ ] **Qwen4Exp real-model coherence** (FILE). Compare a real checkpoint with
  reference logits, including a previously diverging prompt. Existing synthetic
  whole-graph tests do not close this item.
- [ ] **DeepSeek V2/V3/V4 real-model verification** (FILE). Validate latent
  attention, scaling, RoPE, compressed attention, and output coherence on real
  checkpoints.
- [ ] **Gemma2 real-file graph** (FILE). Run greedy generation and a chunk sweep
  with stage comparisons.
- [ ] **Long Spark CPU coherence** (MEASURE). Complete a bounded long run or
  report the exact resource limit. Do not promote the observed matching prefix
  to a full pass.

## P1: server and scheduler

- [ ] **Admission latency on a real server** (MEASURE). The scheduler admits
  between steps (`Engine::batch_open`/`batch_join`/`batch_step`, and the service
  loop built on them), and `test_batch_admission` pins solo output equality,
  per-sequence step counts and cancellation inside a session. What is missing is
  the number: time to first token for a request arriving mid-generation over
  HTTP, with one slot against several, on a model whose steps are long enough to
  measure.
- [ ] **More than one resident prefix** (CODE, blocked by the two items below).
  The cache is one linear KV region, so the longest prefix that can be resident
  is the last run's; reuse of prefixes SHORTER than that record works and is
  pinned by `test_prefix_partial_reuse` (same prompt, a branch diverging
  mid-prompt, an extension, and nothing shared -- each compared against a cold
  engine). Two branches resident at once is not a tree structure but storage plus
  addressing: KV pages with copy-on-write, and row metadata that names a branch.
  Those are the next two items; this one closes when both are in.
- [ ] **KV pages, refcounts, and copy-on-write** (CODE). Replace the contiguous
  per-slot region (`n_slots` x `kv_cap_` rows, `src/engine.cpp:1226`) with
  fixed-size pages, a per-sequence block table, and shared pages that are copied
  only when a second writer appears. Supersedes the "KV tier with batching" item
  below, which is the same work seen from the tier's side, and is what makes a
  fork cost pages instead of a region. Gate: bit-identical output on the existing
  exactness arms (`test_batch_decode`, `test_batch_admission`,
  `test_prefill_chunk_bitexact`) plus a measured non-regression on the
  single-sequence path, where one page lookup per block is the cost.
  See `docs/PAPER-EVALUATION-2026-11.md` section 1 (PagedAttention).
- [ ] **Exact branch forking** (CODE). `Engine::batch_fork` accepts complete
  branch prompts and checks the parent's token-prefix. Ordinary f16-KV forks
  copy trunk rows into independent slots and prefill only the suffix. Compressed-
  attention and recurrent forks now also restore prompt-boundary state snapshots
  and reuse the copied trunk rows; recurrent decode remains serialized per
  sequence. `test_batch_fork` checks these paths against standalone runs,
  including nested and sibling recurrent forks, while preserving the parent's
  continuation. This is fixture correctness, not a speedup claim. Int8
  multi-slot sessions are refused at `batch_open`: a direct full-prompt
  nonzero-slot probe diverged from slot zero (`5 5 5 5` vs `5 5 3 1`). Forks
  remain gated for all lossy formats; other quantized batch-session modes need
  per-format solo-output tests before drawing a parity conclusion. Remaining for
  Flide: KV pages/refcounts/CoW and `(trunk, branch, position)`
  visibility masks. The current API still duplicates KV rows, and state snapshots
  copy data; there is no shared branch stream or measured throughput result. See
  `docs/PAPER-EVALUATION-2026-11.md` section 2 (Flide).
- [ ] **Per-option scoring surface** (CODE). A request that returns the output
  head normalised over a caller-supplied option set (Flide's Eq. 1), on both the
  serial and the session path, so a decision model can be served and measured at
  all. Small next to the two items above, and the consumer that makes them worth
  building.
- [ ] **Prefix persistence, journal, TTL, and size bounds** (CODE). Verify
  restart recovery, expiry, corruption handling, and bounded storage.
- [ ] **KV tier with batching** (CODE). Replace whole-layer, single-sequence
  paging with per-sequence page ownership -- the tier's half of the KV-pages
  item above. Require tier-forced batch versus solo equality.
- [ ] **Beam, tree, and sampling speculation** (CODE). Specify accepted-token
  and sampler contracts before implementation. A candidate tree is the branch
  mask of the forking item above, so the contract work starts from that
  primitive rather than from a new mechanism.

## P1: model and generation capabilities

- [ ] **HIP attention-logit softcap** (CODE). Implement the device operation,
  update the capability response, and add CPU/device parity on the Gemma2
  fixture.
- [ ] **Recurrent speculation and Tail-Replay** (CODE). Implement invertible
  checkpoint/rollback for recurrent and convolution state, including partial
  acceptance and cancellation.
- [ ] **Qwen4 MTP verification** (FILE). Run the real trunk and matching head
  through `scripts/mtp_spec_gate.sh`. Require the stated acceptance threshold
  and a real speed improvement before changing support status.
- [ ] **DeepSeek4 DSpark verification** (FILE). Run a real trunk and matching
  nextn block through the same gate.
- [ ] **Qwen2VL and Qwen3VL vision towers** (CODE and FILE). Text-only execution
  is not vision support. Implement towers and projector only with real artifacts
  and reference outputs.
- [ ] **LLaDA real-model verification** (FILE). Compare the implemented
  bidirectional forward and block diffusion sampler with an independent
  reference on a named checkpoint. Existing synthetic execution tests do not
  close this item.
- [ ] **Generic YaRN coverage** (MEASURE). Identify unsupported scale and
  pattern combinations. Laguna support alone does not close this item.

## P2: MoE, GPU, and memory-residency work

- [ ] **Device routing plan** (CODE). Move suitable routing work to the device
  while preserving expert identity and exact output.
- [ ] **Reduce per-expert launches** (CODE). Fuse safe upload, gather, and
  scatter operations and measure launch-count and end-to-end effects.
- [ ] **Safe graph replay** (CODE). Define stable-slot and dynamic-plan rules;
  test changing routes and residency.
- [ ] **Remaining projection and elementwise fusions** (MEASURE). Preserve each
  matrix dtype, sidecar scale, and f16 rounding.
- [ ] **Candidate sampling and logits-copy paths** (MEASURE). Keep tail mass,
  filters, and sampler equivalence intact.
- [ ] **Q5/Q6 decode ALU work** (MEASURE and HARDWARE). Optimize only against
  matched native workloads and numerical gates.
- [ ] **WMMA staging and pipelines** (HARDWARE). Evaluate dequant staging,
  producer-consumer overlap, occupancy, and split-K policy on target GPUs.
- [ ] **RDNA2 LDS and barrier work** (HARDWARE). Require gfx1031 correctness and
  matched timing.
- [ ] **Multi-shard pull readers** (CODE). Add per-shard weight-pull tables and
  test shard-boundary reads without converting the model.
- [ ] **Draft-tier resource guards** (CODE). Refuse or skip useless warmup when
  the tier cannot hold a useful routed set.
- [ ] **One-layer route hints** (MEASURE). Keep separate pollution budgets and
  default-off behavior unless an end-to-end win is measured.
- [ ] **Per-card byte and slot profiles** (HARDWARE). Record actual expert
  footprints and usable residency.
- [ ] **Expert-only quantization and native packs** (CODE and FILE). Add
  sensitivity, calibration, quality gates, and kernel-native evaluation.
- [ ] **Ternary and BITCOS evaluation** (MEASURE). Require useful accuracy and
  throughput, not merely lower storage.
- [ ] **Native validation of the prefill-freeze policy** (HARDWARE). The gate is
  unit-tested; what is missing is a matched A/B on a real MoE file: same prompt,
  same cache size, `KRK_EXPERT_PREFILL_VOTE=1` against the default, reporting
  tok/s, decode hit rate, pinned slots at the first generated token, and the
  promotion count. No native result for this has been recorded.
- [ ] **Sweep the host share with the replay tuner** (MEASURE). Pick the share
  and the cache size for one real file from one recorded trace, then check the
  prediction against a live run. The tuner exists
  (`scripts/moe_cache_tune.py`); what is unverified is its prediction error.
- [ ] **Incremental expert stack** (CODE). Keep the previous step's per-expert
  views and pin stack and rewrite only the slots whose expert changed, instead
  of rebuilding the stack every step. Output cannot change, so the gate is
  bit-identical tokens plus a measured step-time difference with the existing
  host-phase profile. See `docs/PAPER-EVALUATION-2026-11.md` section 3 (Edge0,
  the assembly tax).
- [ ] **Unmerged adapter delta** (CODE). Serve `y = W(x) + (alpha/r) * B A x`
  from a sidecar with the quantised base untouched. Merging an adapter into a
  group-quantised base and re-quantising erases most of it, because the delta is
  smaller than the group step -- the same arithmetic the paper reports, and the
  same reason it would be lost here. Independent of the paper: it is what makes
  any adapter servable at all.
- [ ] **Prediction-as-routing experiment** (FILE + MEASURE, opt-in). Preserve
  `MoeRoutePredictor` staging-only as the default exact path. Investigate serving
  a separately supplied, trained predictor as an experimental routing policy
  without modifying the base GGUF; this necessarily changes the computed function
  and is quality-dependent, unlike staging. Prerequisites: a named checkpoint,
  predictor trained against that router/model path, an explicit opt-in interface,
  route agreement and expert-load metrics, held-out perplexity/task-quality
  evaluation, plus base-file hashes and standalone-vs-batch isolation tests.
  Do not claim parity or promote to default based on staging precision. No trained
  artifact or quality result currently exists; see the Edge0 investigation in
  `docs/PAPER-EVALUATION-2026-11.md`.
- [ ] **Tree-Ising-compatible repair sidecar investigation** (DESIGN + FILE).
  Keep source GGUF weights read-only. Assess an offline tool that emits an
  independently named repair sidecar or a runtime-only, opt-in overlay loaded
  alongside the unchanged base; never rewrite/requantize the source checkpoint.
  Establish support for the model's quantization/layout first, quantify changed
  parameters/routes and task quality, hash the base before/after, and verify the
  unmodified path remains exact. A sidecar still changes effective model behavior
  and needs a named checkpoint, explicit user opt-in, and quality evidence; no
  optimizer/artifact is currently available. See Tree-Ising in
  `docs/PAPER-EVALUATION-2026-11.md`.
- [ ] **Value amortisation for the KV cache** (CODE, experimental, needs a named
  checkpoint). Per-head key-to-value map fitted during prefill with residuals for
  the worst-reconstructed tokens, keys unchanged, opt-in only. Evidence required:
  reconstruction error at a fixed ratio, then greedy-text agreement against the
  uncompressed run, then throughput. Compression ratio alone is not evidence.
  See `docs/PAPER-EVALUATION-2026-11.md` section 4 (LowRAM).
- [ ] **Multi-device expert residency** (CODE and HARDWARE). The cache is one
  pool on one device; a per-GPU split (as llama.cpp's `--moe-cache-mib` does)
  would need tensor-sharded multi-device model loading first, which this engine
  does not implement.

## P2: measurements and observability

- [ ] **Real bounded-memory MoE promise** (FILE and HARDWARE). A driver now
  exists at `scripts/bench_bounded_moe.sh` to run the built-in benchmark under
  16/24/32 GiB RAM-planning tiers with eager warmup, WARM duplication, and expert
  pinning disabled; each run defaults to a one-expert resident-slot cap (or can
  use the automatic budget), records process peak RSS, hit-rate, cold-read, and
  throughput diagnostics, then checks exact greedy-output parity across tiers.
  It passed a smoke run on the synthetic tiny MoE fixture only.
  Run it on a named 30B-A3B checkpoint and matching native 16/24/32 GiB hosts.
  `--ram-tier` alone is a policy simulation, not a hard memory limit. This item
  stays open until real-model runs and raw logs establish the memory ceiling,
  exactness, and performance trade-offs.
- [ ] **Matched llama.cpp comparisons** (MEASURE). Match model, hardware,
  prompt, generated length, sampler, and residency.
- [ ] **Per-format throughput table** (MEASURE). Rotate formats through one
  controlled benchmark and retain raw results.
- [ ] **CPU MoE dispatch and throughput** (MEASURE). Replace cross-machine
  comparisons with same-host controls.
- [ ] **Linux utilization counters and device-busy hints** (CODE). Define the
  source and accuracy of each counter before exposing it.
- [ ] **Benchmark provenance** (CODE). Record model identity, request, hardware,
  controls, truncation, and errors with every retained result.
- [ ] **Remaining HIP graph audit** (CODE and HARDWARE). Compare implemented
  operations with capability responses and model requirements.

## P3: documentation and support-state maintenance

- [ ] **Automated consistency checks** (CODE). Check documented dtype and
  architecture inventories against source, validate local links, and compare
  documented options with the CLI and server parsers.
- [ ] **Example smoke tests** (CODE and HARDWARE). Execute README and user-guide
  commands on CPU and available HIP configurations. Parser inspection alone does
  not prove that an example runs.
- [ ] **Current real-model inventory** (FILE). Record available checkpoints and
  new reference runs without reviving deleted filename-based support claims.
- [ ] **Decision-model benchmark** (FILE). MMDM-style option scoring, parked
  behind two preconditions: the per-option scoring surface in P1 and a named
  decision checkpoint with reference outputs. Until both exist the benchmark
  would only measure other engines.
- [ ] **Historical measurement provenance** (MEASURE). Replace tables that lack
  raw evidence with new controlled runs when the required model and hardware are
  available. Keep the old values explicitly historical.

## Blocked verification matrix

| Item | Required resource |
| --- | --- |
| Windows build and help consistency | Windows environment |
| Zen 3 performance | Native Zen 3 host |
| Core Ultra 135H performance | Native Core Ultra 135H host |
| gfx1031 and gfx1201 validation | Corresponding AMD GPUs and HIP runtime |
| HIP-linked host floating-point behavior | HIP host and controlled bisect |
| Real large-MoE memory promise | Named model and 16/24/32 GB RAM hosts |
| Independent model coherence | Named checkpoints and reference outputs |

## Closed or deliberately rejected

- Release and full ASan correctness gates are green.
- Continuous batching, per-row MLA addressing, and HOT/WARM/COLD KV tiers exist.
- Request cancellation is exercised at every phase a session has:
  `test_cancel_phases` covers admission before the first chunk, a prompt
  cancelled between two chunks with a neighbour still decoding, a client that
  disappears mid-stream, and teardown with a sequence attached; the MTP spec
  chain cancels inside a proposal round. Each arm also checks the slot comes
  back and that a cancelled stream does not stop the next request.
- Iteration-level admission exists: a batch session keeps its slots and the
  server admits a request between decode steps instead of per wave. A joined
  prompt is prefilled one chunk per step while anything is decoding, and drained
  in one call when nothing is.
- Do not describe them as missing.
- Prediction-as-routing (Edge0) and Tree-Ising are open investigations, not
  approved default behavior: both can change model outputs. Preserve the ordinary
  router and read-only GGUF path as defaults. Explore opt-in modes that load
  user-provided predictor/repair sidecars without changing source weights, and
  gate them on explicit quality, routing-agreement, artifact-integrity, and
  standalone-vs-batch verification. Staging-only prediction remains the exact
  baseline; no training or repair artifact is currently available here.
- Agent-side sparse reading (SparseRead) and offline EMA k-means (e-means) have
  no engine surface: recorded in `docs/PAPER-EVALUATION-2026-11.md`, not work
  items.
- Wide CPU reduction is opt-in, not the bit-exact default.
- Self-draft for speed, default CPU experts, missing-expert zero contribution,
  historical-rank seeding, and previously rejected kernel repacks remain closed
  without new evidence.

# Kraken pending work

Status: 2026-11-11, after the scheduler work: iteration-level admission,
prefill interleaved with decode, a cancellation audit, and prefix-reuse tests.

This is the only pending-work list. Historical plans, audits, campaign logs, and
transcripts were removed because their claims and line references had become
stale. Current behavior must be established from code, tests, and reproducible
runs.

## Evidence rules

- The release CPU suite passes 8045/8045 checks and the ASan build of the whole
  suite passes 8044/8044 (`build-cpu`, and `build-asan` with
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
- Do not reopen a rejected optimization without new falsifying evidence.

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
- [ ] **More than one resident prefix** (CODE, needs the KV tier with batching).
  The cache is one linear KV region, so the longest prefix that can be resident
  is the last run's; reuse of prefixes SHORTER than that record works and is
  pinned by `test_prefix_partial_reuse` (same prompt, a branch diverging
  mid-prompt, an extension, and nothing shared -- each compared against a cold
  engine). Reusing two branches at once needs per-sequence page ownership, which
  is the same work as the KV-tier-with-batching item below, so the two are one
  job rather than two.
- [ ] **Prefix persistence, journal, TTL, and size bounds** (CODE). Verify
  restart recovery, expiry, corruption handling, and bounded storage.
- [ ] **KV tier with batching** (CODE). Replace whole-layer, single-sequence
  paging with per-sequence page ownership. Require tier-forced batch versus
  solo equality.
- [ ] **Beam, tree, and sampling speculation** (CODE). Specify accepted-token
  and sampler contracts before implementation.

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
- [ ] **Multi-device expert residency** (CODE and HARDWARE). The cache is one
  pool on one device; a per-GPU split (as llama.cpp's `--moe-cache-mib` does)
  would need tensor-sharded multi-device model loading first, which this engine
  does not implement.

## P2: measurements and observability

- [ ] **Real bounded-memory MoE promise** (FILE and HARDWARE). Run a 30B-A3B
  class model at 16, 24, and 32 GB RAM. Report throughput, RSS, hot-set hit
  rate, and cold bytes per token for each budget.
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
- Wide CPU reduction is opt-in, not the bit-exact default.
- Self-draft for speed, default CPU experts, missing-expert zero contribution,
  historical-rank seeding, and previously rejected kernel repacks remain closed
  without new evidence.

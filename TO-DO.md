# Kraken pending work

Status: 2026-11-12, after the documentation support-state correction.

This is the only pending-work list. Historical plans, audits, campaign logs, and
transcripts were removed because their claims and line references had become
stale. Current behavior must be established from code, tests, and reproducible
runs.

## Evidence rules

- The release CPU and Debug ASan suites currently pass 7579/7579 checks.
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

- [ ] **Iteration-level admission** (CODE). Admit requests at iteration
  boundaries rather than holding arrivals until a wave drains. Require solo
  output equality, cancellation safety, and a latency measurement.
- [ ] **Per-slot prefill scheduling** (CODE). Interleave chunked prefill with
  active decode without starving either class.
- [ ] **Request cancellation audit** (CODE). Exercise cancellation during
  admission, prefill, decode, speculation, and teardown.
- [ ] **Radix prefix reuse** (CODE). Reuse shared prefixes shorter than the
  whole record, with exact-output tests.
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
- Do not describe them as missing.
- Wide CPU reduction is opt-in, not the bit-exact default.
- Self-draft for speed, default CPU experts, missing-expert zero contribution,
  historical-rank seeding, and previously rejected kernel repacks remain closed
  without new evidence.

# Tiered execution and residency

This document describes the current tiered design and preserves selected
historical measurements with their scope. It replaces the old design note that
described KV tiers, prediction, and related mechanisms as entirely absent.

## Current structure

Kraken manages two different residency problems:

- MoE expert weights are large and sparsely selected.
- KV state grows with sequence length and must retain per-token history.

They use separate policies and data structures. Sharing terminology does not make
their lifetimes or correctness contracts identical.

| Resource | HOT | WARM | COLD |
| --- | --- | --- | --- |
| Routed experts | Budgeted resident slots in device memory or backend memory | Host cache, optionally filled or promoted | Mapped GGUF data read on demand |
| KV state | Active device-resident pages | Host-resident pages | Spill files under `--kv-cold-dir` |

The model mapping, dense weights, activation workspaces, host warm tiers, and
allocator overhead are outside a single expert or KV budget. Total process RSS
must be measured independently.

## Adopted tiering lessons and limits

The Betelgeuse and Kanjoos references are useful as policy and measurement
examples, not as portable settings. Their hardware, model formats, and cache
sizes do not define Kraken's budgets.

### What Kraken already does

- Expert residency is distinct from the mapped GGUF source, host WARM copies,
  and device HOT slots. A disk-backed mapping is not counted as a second
  application-owned expert cache; WARM copies are separately budgeted.
- Device and host policies have explicit byte/slot limits. Host WARM admission
  rechecks live available RAM through `SharedHostRamBudget`; it is not only a
  one-time startup estimate. The engine reports the planned tiers, while the
  bounded-MoE benchmark records actual process peak RSS separately.
- Expert misses preserve the routed expert computation. Do not convert an I/O
  failure into zero weights or a different expert choice. A failed demand read
  must fail the operation rather than silently alter model output.
- Prefetch/prediction is staging only: the native router still chooses the
  experts that execute. Treat speculative staging as expendable, bounded work;
  it must never raise the resident-slot cap or evict in-use data.

These are distinct guarantees: an expert-slot or WARM-byte limit is not a hard
limit on process RSS, page cache, or total machine memory. For a deployment that
must not exceed a system ceiling, run under an OS-enforced limit (and record
swap/OOM counters) in addition to measuring RSS. On Linux, the bounded-MoE
driver can be launched inside a dedicated scope, for example
`systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- ...`; its
metadata records cgroup limits and end-of-run high-water/event counters when
visible. Use a dedicated scope for process-specific cgroup peaks. A simulated
`--ram-tier` is planner input, not enforcement.

### Safe ideas to carry forward

- Keep a budget ledger per tier: configured cap, allocated/occupied bytes or
  slots, current use, and any clamp/refusal reason. Do not merge VRAM, pinned
  host memory, pageable WARM, mapped-file page cache, KV, and scratch into one
  misleading “cache size.” Improve diagnostics only where a value is missing;
  avoid a second policy that can disagree with the engine's enforced budget.
- Preserve a hard ownership order during eviction: active leases and
  event-referenced/in-flight transfers cannot be evicted; speculative prefetch
  is lower priority than demand data. Any change to recency/frequency scoring
  must be evaluated under the same hard limits, with forced-eviction and
  concurrent-use tests.
- Consider coalescing storage reads for adjacent expert payloads only as an
  I/O-only experiment. Keep per-expert admission and execution identity
  unchanged; cap bytes and requests in flight, cancel stale speculation, and
  prove exact output parity. GGUF payload layout must be inspected rather than
  assuming adjacent experts are physically contiguous.
- Make cold-start, first-request, and repeated warm-request measurements
  separate. The bounded-MoE script records host/cgroup metadata, model size,
  settings, and process peak RSS. With `--bench-repeats`, one startup is followed
  by a first request and same-process repeats. Per-request HOT/WARM hit and
  COLD-read counters show whether those repeats are actually warm, and generated
  tokens must match the first request; a separate generation pass checks parity
  across RAM tiers. Report peak RSS, VRAM peak, swap/OOM,
  resident slots/bytes, cold bytes/read time, hit/eviction counts, prefill,
  decode, and output parity on the same named model and workload. Never
  transplant another project's token rates or cache sizes.

### Ideas not adopted as defaults

- Do not copy Betelgeuse's 8 GiB host victim cache or thousands of device slots.
  Kraken's one-slot run remains the minimum-residency stress profile; larger
  pools require an explicit comparison and measured reason.
- Do not import Kanjoos's fixed VRAM/RAM split, eviction score, prefetch horizon,
  or ReBAR requirement. Each is model/backend/device-dependent and needs a
  Kraken measurement and correctness gate. The Kanjoos 16 GiB row also appears
  inconsistent with its stated formula (`min(0.65 * RAM, RAM - 8 GiB)` gives
  8 GiB, while its table lists 10.4 GiB); none of its numeric tier values are
  treated as Kraken guidance.
- Do not adopt its lower-precision expert-pack recommendations as a memory
  shortcut. Any lossy format remains subject to Kraken's existing opt-in,
  parity, and quality gates.
- Explicit per-session resident/spilled/suspended KV admission classes could be
  considered only after per-sequence KV ownership and accounting exist. The
  current whole-layer tiering and batch refusals remain authoritative; do not
  imply a session can spill safely merely by assigning it a label.

## Expert ownership

`ExpertCache` owns resident expert slots and their eviction policy. Residency is
frequency-aware with aging and pinning. A cache miss can read from the mapped
file or promote a host copy, depending on configuration.

The important ownership rules are:

1. The backend must outlive every model, engine, and expert cache that uses it.
2. A caller must hold the required leases while using multiple acquired expert
   pointers.
3. Eviction or rebalancing must not invalidate an active pointer.
4. CPU and HIP residency paths must be validated separately.
5. A larger slot count alone does not repair pointer lifetime.

Cache-only movement should preserve expert selection. `--router-resident` and
`--cache-prior` deliberately influence routing and can change generated output.
They require separate quality evaluation.

### Routing and execution

The engine can group token rows by expert and execute grouped matrix operations.
Backend capabilities also include routing-related operations and device-side
statistics. Support and selection vary by backend and configuration.

A device routing primitive existing in `Backend` does not mean every model uses
it by default. Measure the active path with profiling and end-to-end timings.

## KV formats and tiers

The CLI accepts `f16`, `int8`, `turbo3`, and experimental `rot4` KV formats.
Quantized KV reduces storage and traffic but changes attention inputs. Compare
outputs and application quality against `f16` for the target workload.

HOT, WARM, and COLD KV tiers are implemented. The current tier pages whole
layers for one sequence. Consequently, tiered KV is excluded from the general
multi-sequence batch path. Per-sequence page ownership remains pending.

The HTTP server can retain a matching prompt prefix and reuse its KV state. This
is one held prefix record with rollback, not a radix tree. Cancellation or a
mismatched request must not leave mixed history advertised as valid prefix state.

## Batching boundary

`Engine::generate_batch` requires backend multi-sequence attention and compatible
per-row state. `Engine::batch_refusal()` explicitly rejects configurations whose
state is still sequence-wide.

The general batch path currently excludes recurrent and diffusion models,
unsupported per-row MLA, QSA or MSA, hyper-connections, PLE, route prediction,
speculation, learned positions, ALiBi, and tiered KV.

These refusals protect correctness. Increasing slot capacity does not remove
them.

## Optimization priorities

Optimization work should follow measured end-to-end costs:

1. Reduce synchronous router readback and expert-plan overhead where doing so
   preserves exact selection.
2. Overlap expert transfers with independent work while holding explicit leases.
3. Reduce gather, scatter, and launch overhead without changing matrix dtype or
   accumulation semantics.
4. Replace whole-layer KV paging with per-sequence ownership before combining
   tiered KV and batching.
5. Evaluate graph replay only after stable slots, route changes, and eviction
   safety are defined.
6. Measure attention and GEMM tiling on the intended GPU rather than transferring
   a result from another target.

No optimization should silently convert model weights. Lossy expert or KV
quantization needs explicit opt-in and quality evaluation.

## Historical measurements

The following observations are retained to explain design decisions. They are
not current-build performance claims. Their original raw trace files are no
longer retained in the repository.

### RX 6700 XT profile

Historical gfx1031 profiling of a 361-token Qwen3-8B prefill reported:

| Component | Observed share or rate |
| --- | ---: |
| `gemm(simt)` | 74.2% of 1784.7 ms |
| Attention | 24.6% |
| `gemm(simt)` arithmetic | 3.78 TFLOP/s |
| `gemm(dp4a)` arithmetic | 10.33 TFLOP/s |

The same historical profile described MoE decode as dominated by blocking warm
reads. One 3817-operation decode window attributed 95.2% to `warm_read`, with
approximately 26.6 ms per call. These figures explain why transfer overlap and
rung selection matter. They do not describe current RX 6700 XT or RX 9070 XT
throughput without a new controlled run.

### Two-vCPU expert-residency ladder

A historical synthetic 1.2 GiB Q8_0 MoE workload used 20 layers, 32 experts, and
top-4 routing on two vCPUs:

| Configuration | All resident | 64 MiB hot | 16 MiB hot |
| --- | ---: | ---: | ---: |
| Prefill, exact | 13.95 s | 13.68 s | 14.02 s |
| Prefill, AVX2 dot | 3.94 s | 3.82 s | 3.95 s |
| Decode, AVX2 dot | 41.1 ms | 73.6 ms | 74.6 ms |
| Read volume | 369 MiB | 2519 MiB | 2519 MiB |

This is one synthetic workload on one host. It shows that residency can affect
single-row decode more than batched prefill. It is not a universal cache-size
rule or a model-quality result.

A later CPU prefill correction reused each decoded weight block across activation
rows. On the recorded fixtures it produced bit-identical results and a 2.28x to
3.35x improvement for a Q8_0 MoE, and 3.83x to 4.29x for a 1 GiB F16 dense file.
These are historical kernel-path and workload results, not current model-wide
promises.

## Validation requirements

A tiering change needs:

- Solo-versus-batched output equality where batching is supported.
- Forced eviction and promotion tests.
- Cancellation during transfer and teardown.
- ASan and leak detection without suppressions.
- RSS, transferred bytes, hit rate, prefill throughput, decode throughput, and
  total latency.
- Exact model, hardware, workload, build, budget, and control identification.

Open implementation and hardware-validation items are in
[TO-DO.md](../TO-DO.md).

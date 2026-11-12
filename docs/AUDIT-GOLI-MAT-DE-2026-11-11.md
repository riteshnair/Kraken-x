# Implementation integrity audit, 2026-11-11

## Verdict and scope

The Goli Mat De skill and its optional audit matrix were applied to the current
branch at revision `0482d373cb1923092daa4e32311b6d5ac9d90eef`.

This was a targeted repository audit, not an exhaustive inspection of every
source file. The main focus was the connection between stated CPU and bounded
memory goals, public CLI behavior, buildability, tests, benchmarks, and user
documentation.

The skill itself specifies audit-only behavior. The user separately authorized
code and documentation fixes in the same request. No GitHub issues, tickets, or
external task records were created.

## Coverage ledger

| Unit | Contract and wiring inspected | Verification | Status |
| --- | --- | --- | --- |
| `README.md` | Build, backend, memory, and usage claims against current CLI | Source comparison and documentation rewrite | Fixed |
| `docs/USER-GUIDE.md` | Quick start, flags, server, numerical claims | Source comparison and documentation rewrite | Fixed |
| `src/main_cli.cpp` | Public CPU and x86 backend descriptions | Parser and backend construction inspected | Fixed |
| `src/main_server.cpp` | Help text and target compilation | Missing quote confirmed, then corrected | Fixed, build evidence recorded below |
| `src/backend_cpu.cpp` | CPU and optimized x86 construction | `make_cpu_backend`, `make_intel_backend` inspected | Inspected |
| `src/quant.cpp` | Low-bit kernel claims and optimization opportunities | Existing focused differential and benchmark documents inspected | Inspected, not re-audited exhaustively |
| `tools/bench_lowbit_vnni.cpp` | Benchmark allocation and timing boundary | Corrected fixture and measurement notes inspected | Previously fixed |
| `include/krk/expert_cache.hpp`, `src/expert_cache.cpp` | Bounded expert-residency claims | Interfaces and existing design docs inspected | Sampled |
| `src/arch.cpp`, `src/model.cpp` | Unimplemented markers and model refusal paths | Call context and explicit refusal messages sampled | Intentional limitations, not fake implementations |
| Build and test configuration | Target names and documented commands | CMake and scripts inspected, selected targets built | Partial |

Uninspected areas include every HIP kernel, every model-family graph, the full
server API, NPUXL internals, Windows execution, RV32 exhaustive behavior, and
all model-quality claims. No native Zen 3 or Core Ultra 135H hardware was
available in this environment.

## Confirmed findings and resolutions

### GMD-001, P1: server target had a compile-breaking help string

- **Location:** `src/main_server.cpp`, NPU help text in `usage()`.
- **Expected:** Adjacent C++ string literals must all be valid tokens.
- **Observed:** The line containing `-DKRK_ENABLE_NPU=ON` lacked its opening
  quote. This was direct source evidence of a malformed normal server build
  path.
- **Resolution:** Restored the string literal and clarified the CPU description
  in the same help block.
- **Acceptance:** Compile `kraken-server`, run its help command, and check that
  the option text is complete.

### GMD-002, P2: CPU help and documentation hid optimized behavior

- **Location:** `src/main_cli.cpp`, `src/main_server.cpp`, `README.md`, and
  `docs/USER-GUIDE.md`.
- **Expected:** Users should be able to distinguish reference arithmetic from
  optimized quantized CPU execution.
- **Observed:** The public text repeatedly called the CPU backend scalar even
  though the tree contains AVX2/FMA, runtime VNNI, and quantized dot kernels.
  The `--intel` text also implied an unavailable-backend fallback that its
  current constructor does not implement as an ISA availability check.
- **Resolution:** Help now describes optional optimized CPU behavior and names
  `--intel` as the optimized x86 path with no AVX-512 requirement.
- **Acceptance:** Help text and usage examples agree with backend construction
  and dispatch controls.

### GMD-003, P2: primary documentation obscured current usage and limitations

- **Location:** `README.md` and `docs/USER-GUIDE.md`.
- **Expected:** A new user should find build, run, memory, server, validation,
  and troubleshooting instructions without reading historical model campaigns.
- **Observed:** The README was 1,699 lines and mixed usage with historical
  implementation notes. The guide carried stale test counts, model support
  statements, and numerical claims without a consistent current-revision label.
- **Resolution:** Rewrote both entry documents around current commands and
  explicit limitations. Preserved the former README as
  `docs/ENGINEERING-NOTES.md` rather than discarding its history.
- **Acceptance:** Examples use current parser options, identify sample paths as
  placeholders, distinguish total RSS from tier budgets, and link to detailed
  documents.

### GMD-004, P2: performance evidence needs workload boundaries

- **Location:** Historical performance sections and low-bit benchmark docs.
- **Expected:** Kernel, prefill, decode, load, and end-to-end claims must remain
  distinct.
- **Observed:** The previous low-bit benchmark fixture had underallocated weight
  rows. That defect was fixed in earlier work and its old table withdrawn.
  Other historical tables refer to different machines and revisions.
- **Resolution:** New entry docs require revision, hardware, model, shape,
  numerical controls, warm-up, repetitions, and timing boundaries. Historical
  measurements are explicitly labeled as historical.
- **Acceptance:** No new optimization claim uses an isolated kernel ratio as a
  model-wide speedup.

## Intentional limitations, not defects

The marker search found explicit unsupported-format and graph-feature refusals
in `src/model.cpp` and `src/arch.cpp`. Their surrounding contracts show that the
loader refuses unsupported inputs by name. That is safer than returning fake
success or executing an incomplete graph.

Likewise, default no-op or fallback paths in backend interfaces are not defects
without a reachable contract violation. No such violation was inferred from a
keyword match alone.

## Optimization opportunities aligned with the stated goals

These are proposals, not measured wins or newly completed implementations.

### OPT-001: reduce recovery's second matrix pass

Activation-residual correction currently requires another matrix pass for
`Wq(x-Q(x))`. Profile formats and shapes where that pass dominates. Reuse of Q8
activations and sidecars is already present, so further work should focus on
fusion or selective application with correctness and total-latency gates.

### OPT-002: measure AVX2 and VNNI on the actual target processors

The portable AVX2/FMA build with dispatched VNNI is useful coverage, but the
available host does not establish native Zen 3 or Core Ultra 135H throughput.
Run interleaved scalar, AVX2, and VNNI controls on both target machines. Record
prefill and decode separately and retain only end-to-end wins that exceed run
variance.

### OPT-003: improve hot-expert residency before adding compression

For the approximately 1.5 GiB hot-expert goal, first measure hit rate, promoted
bytes, cold-read latency, and total RSS. Improve prefetch and residency using
actual routing traces. Compression is a separate format project because it can
reduce bytes while introducing additional quality loss and new decode costs.

### OPT-004: do not generalize MoE streaming to dense models

Bounded expert residency exploits sparse activation. Re-reading every dense
layer per token usually makes storage bandwidth the decode floor. Documentation
now states this distinction. Dense-model memory work needs a different measured
design, not the same expert-cache claim.

### OPT-005: expand benchmark correctness controls

Keep full-code and signed-extreme differential cases beside every new low-bit
kernel. Benchmark all allocated blocks, separate setup from the timed loop, and
include output strides and one-row versus eight-row contracts. Sanitizer runs
should accompany fixture or layout changes.

## Proposed follow-up tasks

No tasks have been created in GitHub or another external system.

| ID | Priority | Effort | Acceptance criteria |
| --- | --- | --- | --- |
| GMD-T1 | P1 | Small | Build server and CLI on Linux and Windows; compare both help outputs with parser options |
| GMD-T2 | P2 | Medium | Run the full test and baseline gates on the final revision; record failures rather than replacing them with focused tests |
| GMD-T3 | P2 | Medium | Benchmark native Zen 3 and Core Ultra 135H with identical model, prompt, context, and numerical controls |
| GMD-T4 | P2 | Medium | Measure a 1.5 GiB expert-tier run using RSS, read bytes, expert hit rate, prefill, decode, and output-quality controls |
| GMD-T5 | P2 | Large | Audit remaining HIP model graphs and unsupported-feature refusals family by family |
| GMD-T6 | P3 | Small | Add a documentation-link and forbidden-placeholder check for primary usage documents |

Dependencies: GMD-T3 and GMD-T4 require their respective hardware and real model
files. GMD-T5 should be divided by architecture rather than presented as one
unbounded repository-wide pass.

## Executed and unrun checks

Executed on the final edited Linux source:

- CPU release build of `kraken`, `kraken-server`, and `kraken-tests`: exit 0.
- `kraken --help` and `kraken-server --help`: exit 0, with the corrected backend
  and NPU option text present.
- `git diff --check`: clean.
- Primary rewritten documents checked for em dashes, en dashes, and three-dot
  ellipses: none present.
- Full `kraken-tests --workers 2`: exit 1, **7383/7468 checks passed, 5 groups
  crashed, 20 groups failed**. This is not a green suite. The retained failure
  set includes prefill chunk invariance, batched GEMM, KV tiers, DFlash, mixed
  expert formats, and other paths outside the documentation edits.

The broad failures were observed and recorded, not repaired or attributed to
this documentation change without a baseline bisection. They remain release
risks and are covered by follow-up task GMD-T2.

GPU coherence models, Windows builds, native Zen 3 and Core Ultra 135H
benchmarks, and model-quality evaluation remain unrun. No model files were
available for end-to-end examples, so all `model.gguf` paths in the documents
are placeholders rather than claimed executions.

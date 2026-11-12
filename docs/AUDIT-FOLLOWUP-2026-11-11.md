# Audit follow-up: cache safety and explicit memory controls

## Scope

This continues `AUDIT-GOLI-MAT-DE-2026-11-11.md` on the active branch. Work
focused on locally reproducible build, memory-safety, configuration, server,
and expert-residency failures. Native target-CPU benchmarks, Windows, GPU model
coherence, and real-model quality evaluation remain blocked by unavailable
hardware and model files.

## Fixed findings

### F-01, P1: CPU F16 cache rows were underallocated

`CpuBackend::kv_row_bytes(F16)` returned two bytes per element, but the CPU
cache append and attention paths store f32 values. ASan showed a heap-buffer
overflow in `test_kv_quant_int8` and `test_batch_decode` ended in invalid free.

The row function now reports `sizeof(f32)`. The per-layer KV test derives its
byte geometry from the same public row function instead of activation width.

Validation:

- `test_kv_quant_int8`: 17/17 under ASan.
- `test_kv_tier_per_layer_widths`: 21/21 under ASan.
- `test_batch_decode`: 86/86 under ASan before separate leak reporting.

### F-02, P1: MTP mixed byte and element strides

The MTP cache allocator special-cased F16 with activation width rather than the
backend row-size contract. After that was unified, ASan exposed a second issue:
uncompressed `kv_append` expects an element stride, while packed caches expect
a byte stride.

MTP now allocates through `kv_row_bytes()` and selects `mtp_kv_dim_` for F16
strides or `mtp_kv_row_` for packed strides, matching the trunk.

Validation with ASan, leak detection disabled only for these focused reruns:

- `test_mtp_hf_block_qwen4`: 17/17.
- `test_mtp_hf_spec_chain`: 18/18.

### F-03, P1: library users silently inherited Turbo3 KV

`EngineConfig` defaulted to Turbo3 while the CLI and server also explicitly
select Turbo3. Library callers that never requested KV quantization therefore
received changed numerics, initialization refusals, and downstream invalid
state.

The library default is restored to F16. CLI and server defaults remain Turbo3.
The HTTP test now explicitly selects Turbo3 for its in-process reference so it
compares the same numerical configuration as the real server.

Validation includes DFlash 22/22, speculative decoding 22/22, and HTTP server
127/127.

### F-04, P1: CPU policy discarded explicit expert-cache controls

The CPU memory-policy branch reset exact slot capacity, disabled positive WARM
requests, and blocked forced warmup. This made the following public options
ineffective on CPU:

- `--expert-cache-slots N`
- Positive `--expert-warm-mb N`
- `--expert-warmup 1`

The policy now preserves explicit requests. Automatic WARM and warmup remain
off on CPU to avoid unnecessary host-to-host staging.

Validation:

- `test_moe_schema_and_laziness`: 45/45.
- `test_moe_batch_fallback`: 12/12.
- `test_expert_warmup`: 15/15.

### F-05, P2: mixed expert-format test enforced an outdated restriction

Per-tensor expert slicing, strides, and GEMM dispatch now support mixed formats.
The test still expected blanket refusal. It now verifies that F16 and Q8_0 banks
load while retaining their own types. Fusions that require one common format
remain responsible for their own compatibility checks.

Validation: `test_mixed_expert_formats` 7/7.

### F-06, P2: host-capability fixture could no longer simulate refusal

`CountingBackend` forwarded host-read support to the wrapped CPU backend, which
now supports direct host-resident reads. The test's explicit override now works
in both directions, allowing the same fixture to test refusal and publication.

Validation: `test_expert_dispatch_host_backed` passes in the full suite.

## Final observed suite result

Command: release CPU build, `kraken-tests --workers 2 --report /tmp/remain-final-report.json`.

Result: **7537/7579 checks passed, 0 crashed groups, 5 failed groups**.

This improved from the audited starting point of 7383/7468 checks, 5 crashed
groups, and 20 failed groups. Check totals changed because corrected fixtures
now reach assertions that previously crashed or refused initialization.

The baseline comparator reports five newly broken groups against its recorded
green revision. They remain open:

| Group | Observed | Open contract |
| --- | ---: | --- |
| `test_gemm_batched_bit_identical` | 70/102 | Tiled GEMM does not always reproduce `vec_dot` bit for bit |
| `test_prefill_chunk_bitexact` | 24/27 | Two-row and eight-row prefill differ from single-row reference |
| `test_lfm2_shortconv` | 30/34 | Shortconv state remains chunk-dependent |
| `test_qwen4exp_whole_graph` | 20/22 | Chunk and CLI reference hashes differ |
| `test_dsv4_whole_graph` | 34/35 | Chunk-one row differs from chunk-eight row |

No assertions in these groups were weakened or removed. These are numerical
correctness failures, not documentation issues.

## Remaining sanitizer finding

LeakSanitizer reports allocations retained after some test fixtures complete,
including CountingBackend and expert-pool paths. The focused ASan runs show the
cache overflows are fixed, but teardown ownership is not fully clean. Leak
detection was disabled only for the two final MTP focused reruns after their
buffer checks passed. It was not disabled for the initial diagnosis.

A follow-up should trace engine shutdown and test-wrapper ownership, then run
ASan with leak detection enabled without suppressions.

## Audit task status

| Task | Status |
| --- | --- |
| Linux CLI and server build/help consistency | Complete |
| Windows build/help consistency | Blocked, Windows unavailable |
| Full test and baseline gate | Executed, five numerical failures remain |
| Native Zen 3 and Core Ultra benchmarks | Blocked, target hardware unavailable |
| Real 1.5 GiB expert-tier model measurement | Blocked, real model and target workload unavailable |
| Remaining HIP graph audit | Not exhaustive, five CPU numerical groups above take priority |
| Primary-document typography check | Complete |
| Leak-free ASan teardown | Open |

No external issues or task records were created.

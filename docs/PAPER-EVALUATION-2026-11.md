# Paper evaluation, November 2026

Eight papers were read for what this repository could use. This document records
the evaluation. It is not a to-do list: every item that survived became a line in
[TO-DO.md](../TO-DO.md), and this file says which.

Two kinds of statement appear below and they are kept apart on purpose.

- **What a paper claims** is the paper's, in its own framing, with its numbers.
  Nothing here was reproduced: no experiment below was run by this repository, and
  none of these numbers is evidence about Kraken.
- **What Kraken has** is read off this tree at commit `e64dfa7`, with file and
  line references, in the same terms the rest of `docs/` uses: implemented,
  fixture-tested, real-model verified, experimental, refused, or missing.

Verdicts:

| Verdict | Meaning |
| --- | --- |
| **Build now** | The change is in this repository's scope, has a test we can write today, and does not need an artifact we lack. |
| **Build later** | In scope, but ordered behind another item or behind a named prerequisite. |
| **Reference only** | Nothing to build. Kept so the next person does not re-derive it. |
| **Refuse** | Out of scope or in conflict with the rules in [TO-DO.md](../TO-DO.md); the reason is recorded so it is not re-proposed without new evidence. |

| # | Paper | Verdict | Kraken surface | Register item |
| --- | --- | --- | --- | --- |
| 3 | PagedAttention (Kwon et al., 2023) | **Build now** | KV storage, per-slot regions | KV pages with refcounts and copy-on-write |
| 2 | Flide (Li et al., 2026) | **Build now** | row tables, attention mask, recurrent state | Exact branch forking |
| 1 | Edge0 (Lin et al., 2026) | **Build now**, partly | MoE disk tier, per-step stack assembly | Incremental expert stack; unmerged LoRA |
| 1 | Edge0, prediction-as-routing | **Investigate**, opt-in only | routing, quality-dependent | named predictor + quality evaluation |
| 4 | LowRAM (Delgado et al., 2026) | **Build later** | KV compression above quantisation | Value amortisation (experimental) |
| 7 | MMDM (Shen et al., 2026) | **Build later** | no scoring surface yet | decision benchmark, parked |
| 5 | e-means (Bessa et al., 2026) | **Reference only** | offline codebook tuning only | none (reference) |
| 6 | SparseRead (Liu et al., 2026) | **Reference only** | agent-side, not engine-side | none (reference) |
| 8 | Tree-Ising (Salloum & Ali, 2026) | **Investigate an external sidecar** | keep the source GGUF read-only | compatible repair artifact + quality evidence |

The order that follows is the ranking, not the order the papers were sent in: the
number in the first column is the position in the original list.

---

## 1. PagedAttention: block-level KV with sharing (build now)

*Efficient Memory Management for Large Language Model Serving with PagedAttention.*
Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. arXiv:2309.06180v1,
SOSP '23.

**Claim.** Serving is limited by KV memory, not compute. Storing each request's KV
in one contiguous run as large as its maximum length wastes 60–80% of that memory
(internal and external fragmentation, over-reservation, and duplicate copies of
shared prefixes), and that waste is what caps the batch size. PagedAttention
stores KV in fixed-size blocks, keeps a per-sequence block table, and copies a
block only when two sequences write to a shared one. The reported result is 2–4× throughput over
FasterTransformer and Orca at equal latency, with the gap widening for long
sequences, larger models, and complex decoding (beam search, parallel sampling).

**Mechanism.** Three separable pieces: (i) a page allocator with a free list;
(ii) a block table per sequence, so attention addresses pages through an
indirection instead of computing `base + pos * stride`; (iii) reference counts
with copy-on-write, which is what turns "two sequences share a prefix" from a
copy into a pointer, and what makes beam search and parallel sampling affordable.

**Kraken today.** `n_slots` sequences each own a contiguous KV region of
`kv_cap_` rows, sized at init (`src/engine.cpp:1226`-`1231`), so KV footprint is
`n_slots × kv_cap_` up front and a slot cannot exceed the context budget.
Attention addresses that region from a row table (`fwd_row_slot_` /
`fwd_row_pos_`, `include/krk/backend.hpp:138`-`142`). Prefix reuse is one held
record, cut back with `kv_rollback` and re-prefilled from there
(`src/server.cpp`, `reusable_prefix`), and the batch path answers
`cached_tokens: 0` because a slot is written from row 0 of its own region. The
KV tier pages whole layers for one sequence and for that reason refuses batching
(`Engine::batch_refusal`, "the KV tier pages whole layers for one sequence's
positions"). Quantisation is per row and has four modes
(`include/krk/backend.hpp:130`).

**What we would build.** Pages of a fixed token count with a per-sequence block
table; refcount + copy-on-write for shared prefixes; the tier's page ownership
per sequence rather than per layer. That is the same job as the existing "KV tier
with batching" register item, so it is one item, not two.

**Cost and risk.** This touches every attention kernel and the tier. The risk is
a silent one — an indirection bug reads another sequence's history and still
produces fluent text — which is why the exactness harness built for batching
(`test_batch_decode`, `test_batch_admission`, `test_prefill_chunk_bitexact`) is
the gate, not a throughput number. The measurement that can fail is the
single-sequence one: a page lookup per block costs something, and it has to be
shown not to regress the one-stream path on a real model.

**Why before Flide.** Flide's exactness argument assumes branches share storage;
today ours cannot. Without pages, a fork either copies a whole region or shares a
linear buffer that only one branch can append to. The storage has to come first;
the metadata (next section) is what makes it useful.

---

## 2. Flide: exact state forking over a shared trunk (build now)

*Flide: Accelerating Decision Inference with Exact State Forking.* Li, Song, Fei,
Hsu, Lee. arXiv:2610.42148v1.

**Claim.** A decision request is a tree: one shared state and several questions
about it, each answered by a probability distribution over options. Existing
engines either run every question as its own sequence (recomputing the trunk, and
on hybrid models resuming recurrent state only at block boundaries) or
concatenate questions behind delimiter tokens, which changes each question's
inputs and therefore its probabilities. Flide runs trunk and questions as one flat
stream with per-token metadata, and claims each branch is *exactly* the standalone
computation: up to 9.5× faster than vLLM and SGLang while closest to an fp32
reference in total-variation distance.

**Mechanism.** Three parts, and each maps to something Kraken already has an
analogue of.

1. *Metadata instead of layout.* Every token carries `(request, branch, position)`;
   a question token attends to a key token iff same request, key in {trunk, own
   branch}, and `t_key ≤ t_query`. Positions continue after the trunk, so RoPE
   matches the standalone sequence.
2. *Recurrent layers cannot be masked*, so each Gated DeltaNet layer runs in two
   calls: the trunk segments produce the state at the last trunk token, then the
   question segments continue from *their own* trunk state through an index,
   with the short convolution reading predecessors through the same index. No
   state is copied.
3. *Flat graphs.* The round's tree lives in device buffers; the launch shape
   depends only on stream length, so a small family of captured graphs serves any
   mix of requests.

**Kraken today.** The row-table mechanism carries `(slot, position)` for
attention rows. `Engine::batch_fork` now has correctness-first fallbacks where
trunk state cannot yet be shared: compressed-attention and recurrent branches
clear their destination slot and replay the complete branch prompt. Recurrent
convolution/scan state is allocated per slot, and recurrent decode forwards are
serialized one sequence at a time; `test_batch_fork` compares two sibling branch
runs with isolated complete-prompt oracles. Ordinary f16 attention still reuses
its trunk KV rows through copying. Lossy KV remains gated because a nonzero-slot
full-prompt replay has not matched the single-slot oracle. These paths establish
fixture parity only; compressed/recurrent trunk reuse, quantized parity, CoW
pages, and a shared sibling-node visibility mask remain unimplemented.

**What we would build.** Generalise the row metadata from `(slot, position)` to
`(trunk, branch, position)` and derive the attention mask from it, then add the
two-call recurrent path so a branch continues from its trunk's state by index.
On top of KV pages this is exactly the missing piece: a fork costs pages only
where a branch writes, and sibling branches stay invisible to each other.

**Cost and risk.** The mask is the work: our kernels currently take a per-row
slot and position and no ancestry, so the first version needs either a segment
table per row or a bounded parent chain, and that is where the correctness risk
lives. The proof obligation is the paper's own: each branch must be bit-identical
to an isolated run. We already have the harness for that claim (the batch
exactness arms compare every sequence against its solo run), so the test exists
before the feature does — the arm to add is a trunk with N branches, on a dense
fixture and on a recurrent fixture.

**What it is worth beyond decision serving.** The same two primitives unblock
three register items that are currently separate: multi-branch prefix reuse,
beam/tree speculation (a candidate tree *is* a fork tree with a mask, which is
also how tree attention in speculative decoding already works), and the recurrent
half of "Recurrent speculation and Tail-Replay" (save/restore delta state,
partial acceptance). It is the highest-value paper of the eight for that reason,
and also the largest.

**What it is worth for the server.** Flide's Eq. 1 — normalise the output head
over an option set — is a scoring surface Kraken does not have at all (see
MMDM, §7). Without it, no decision model can be served or measured here, so the
forking work has no consumer. The two items are ordered together.

---

## 3. Edge0: SSD-streamed MoE with a trained prerouter (build part)

*The Other Half of the Memory Wall: Serving 35B MoEs from SSD with Trained Routing
Prediction.* Lin, Wang, Cai, Liu, Zeng (AutoArk). arXiv:2609.18063v2.

**Claim.** A 35B-class MoE at 4 bits is 19.5 GB, which does not fit beside an
operating system; sparsity reduces compute, not storage. Streaming expert weights
from SSD naively stalls once per layer per token, because layer N+1's experts are
unknown until layer N's output exists. Edge0's prerouter predicts layer N+1's
routing one token ahead and *uses the prediction as the routing*, so the staged
set and the routed set are the same by construction and nothing is dropped.
Reported: ≈20 tok/s at K=4 with 2.9 GiB peak on a single 24 GB machine, against
3.9 tok/s for the resident 4-bit path, +80–84% decode from prerouting, quality
within a few points of the fp16 teacher with an unmerged recovery LoRA.

**Mechanism.** Per-layer head `fc1 → erf-gelu → fc2` plus a linear residual
warm-started from the *next* layer's router row; input is the layer's post-attention
hidden concatenated with two top-k one-hots (this token's and the previous
token's). One flush per step predicts every staged layer at once, which keeps the
head off the per-layer critical path. Prefill keeps the whole-layer path, where
every expert is hit anyway. The remaining cost is per-step tensor assembly, which
their incremental sticky-slot stack attacks directly. The recovery LoRA is served
**unmerged**, `y = W_4bit(x) + (α/r)·BAx`: merging into the quantised base and
re-quantising retains only ~18% of the adapter's effect at the logits level
because the delta (RMS ~1e-3) sits below the 4-bit group step.

**Kraken today.**

- *Staging predictor, online and staging-only.* `MoeRoutePredictor`
  (`include/krk/moe_predictor.hpp`) predicts the next layer's experts from the
  previous token's route and from the preceding MoE layer's route, gated by
  per-layer precision EWMAs, and the cache's own comment states the contract the
  paper abandons: "a wrong guess costs a read that is dropped and never a wrong
  output" (`include/krk/expert_cache.hpp:260`-`263`). It never edits routes, and
  the batch path refuses the predictor because routing history is per sequence.
- *Disk tier.* Experts are lazily loaded from the GGUF mapping with an async
  read-ahead worker and a WARM tier (`prefetch_warm_next`,
  `prefetch_warm_join`, `prefetch_warm_taken`), pinned by
  `--expert-pin-mb` / `KRK_PIN_AUTO`, with `--hybrid-experts` sharing overflow to
  the host.
- *No adapters.* There is no LoRA or sidecar-delta support anywhere in the
  engine; the only `lora` in the tree is MLA's `q_lora_rank`, which is model
  geometry, not an adapter.

**What we would build, in order of independence.**

1. **Incremental expert stack (runtime, no artifact).** The assembly tax is a
   real cost here too: the MoE path re-derives its per-expert views and pin stacks
   every step even when the routed set did not change. Keeping the previous step's
   stack and rewriting only the slots that moved is testable for equality on the
   fixtures and measurable with the existing host-phase profile. It cannot change
   output, so the gate is runtime plus bit-identical tokens, not quality.
2. **Unmerged adapter delta (independent capability).** `y = W(x) + (α/r)·BAx`
   with the base bytes untouched. The paper's arithmetic argument is exactly our
   situation: our weights are group-quantised, so merging an adapter and
   re-quantising would erase it. Justified on its own merits — it is what makes
   any adapter servable — and the paper only supplies the reason not to merge.
3. **A trained predictor, staging only.** Only if someone trains one; the head
   needs the model's own routing as supervision, which is a training artifact this
   repository does not have. If it exists, it enters as a staging signal with the
   existing contract (wrong guess ⇒ wasted read), and the evidence is candidate
   precision plus bit-identical output. It is a FILE item for that reason.

**Prediction as routing: investigate, never silently substitute.** Directly using
predicted experts changes the function computed and can change output quality;
it is not interchangeable with the existing exact staging predictor. A possible
implementation is an explicitly enabled routing policy that consumes a separate,
user-supplied predictor artifact while leaving the base GGUF unchanged. It must
never become the default merely because staging precision is high. Prerequisites
are a named model, a predictor trained for that model/router path, a documented
artifact format, and held-out evaluations of route agreement, expert I/O, latency,
perplexity and task quality. Test the mode against itself in isolated and batched
execution, and preserve an exact baseline/fallback for requests not opting in.
The paper's recovery adapter is part of its quality story; without a compatible
trained artifact, we must not imply the paper's quality results transfer. Until
those prerequisites exist, the current staging-only signal remains the exact
baseline and prediction-as-routing remains an unimplemented investigation.

---

## 4. LowRAM: amortised values and low-rank keys (build later)

*LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at
Test-Time.* Delgado, Hazard, Benfeghoul, Wang, Fountas (Huawei Noah's Ark / UCL).
arXiv:2611.07025v1.

**Claim.** Within a head, values are largely predictable from their keys. Fitting
a lightweight per-head map at test time and storing residuals only for the
worst-reconstructed tokens gives ~4× near-lossless value compression; pairing it
with cross-layer low-rank key compression and reconstructing only a selected
subset of pairs per step gives up to 5× decode throughput at 4× compression, and
composes with quantisation and eviction to 26.9× at 94.5% of baseline on long
contexts.

**Mechanism.** A per-head least-squares fit during prefill, a residual list ranked
by reconstruction error, and a decode path that reconstructs only the pairs it
attends to. Two knobs that decide whether it is worth anything: how often the fit
is refreshed, and how many pairs are reconstructed per step.

**Kraken today.** Four KV quantisation modes (`F16`, `Int8`, `Turbo3`, `Rot4`,
`include/krk/backend.hpp:130`) with kernels and self-tests; MLA latent KV for
DeepSeek V2/V3/V4; HOT/WARM/COLD tiers that page whole layers. No low-rank
representation, no learned reconstruction, and nothing that fits parameters at
test time.

**What we would build, and why last.** The cheapest honest step is value
amortisation *alone* (keys unchanged): fit per head during prefill, store
residuals for the worst r%, and measure reconstruction error and text agreement
before touching decode scheduling. That is a bounded piece of work, but it is
lossy, it needs a real checkpoint to say anything at all, and it has to be opt-in
and reported as experimental. Our quantisation already takes the cheap, exactly
measurable wins, so this sits behind the storage work rather than beside it:
pages (item 1) change *where* KV lives, and this changes *what is stored*, and it
is much easier to argue the second once the first is in place.

**The measurement that decides it.** Reconstruction error at a fixed compression
ratio on a named model, then greedy-text agreement against the uncompressed run,
then tokens/s. A compression ratio alone is not evidence (see the register's rules
on lossy changes).

---

## 5. e-means: EMA k-means (reference only)

*e-means: outperforming k-means (in one-tenth the epochs).* Bessa, Mazaré,
Szilvasy, Touvron, Picard, Douze, Jégou (FAIR). arXiv:2611.09501v1.

**Claim.** An EMA-based mini-batch k-means with a decaying learning rate, better
initialisation, dead-cluster revival, and a finalisation pass reaches Lloyd's
25-epoch loss in 1.4–2.6 epochs on embedding datasets, i.e. 10–45× wall-clock
speedups, with a scaling rule that fixes batch size and decay without a sweep.

**Kraken today.** No clustering in the inference path. `tools/turbo_kv_selftest.cpp`
uses continuous k-means only as an assertion device inside the Turbo3 KV
quantiser's self-test.

**Verdict: reference only.** Nothing in a decode pass runs k-means. The place this
could matter is offline: if per-layer codebooks are ever learned for KV or weight
quantisation (rather than derived from group statistics), this is the algorithm to
reach for, and the scaling rule is the useful part — but that is a tuning script
with a calibration set, not an engine option. Recorded so the work is not
re-derived from scratch if codebook learning ever becomes an item.

---

## 6. SparseRead: bounded reading for agents (reference only)

*Read Less, Solve More: Token-Efficient Sparse Reading of External Content for AI
Agents.* Liu, Wu, Ma, Xu, Wang, Hu, Tao, Tan. arXiv:2608.22237v2.

**Claim.** Agents over-read: they are handed whole pages, files and tool results
when a decision needs a few passages, and the excess both costs tokens and dilutes
evidence. SparseRead interposes a reading layer — a Read Intent Protocol, pluggable
reader backends, and a budgeted gate with full-view fallback — and reports up to
92.9% fewer tokens and 89% less wall time with quality preserved or improved in
97% of settings across six models.

**Kraken today.** An inference server: `/v1/completions`, `/v1/chat/completions`,
`/v1/models`, `/health`, a built-in web UI, and one local tool endpoint
(`/v1/tools/rv32`, exact integer arithmetic). No artifact reading, no agent loop,
no retrieval.

**Verdict: reference only, and out of scope by construction.** This changes what
an *agent* puts in the prompt; it does not change what the engine computes. The
one sliver that could eventually belong here is a bounded-read contract if Kraken
grows an agent-facing tool layer, and that is a product decision, not a paper
result. Nothing to build now.

---

## 7. MMDM: a benchmark for decision models (build later, with a precondition)

*MMDM: Measuring Massive Multitask Decision Making.* Shen, Zong, Guo, Zhang, Guo,
Cheng. arXiv:2611.14309v1.

**Claim.** 5,448 questions over 16 task categories with fresh instances, option
spaces of 2–1,024, contexts to 64,817 tokens, and single/multi-select and
soft-label answer types. Across 28 decision models the pattern is that knowledge
tasks are handled and logic tasks are not: 78.75% hard single-select on world
knowledge against 53.95% on option logic and 47.95% on constraint planning, with
36.81% the best hard multi-select.

**Kraken today.** No scoring surface. The server returns text and token counts; it
exposes no logprobs and no per-option normalisation, so a decision model could not
be served even if we had one, and this benchmark could not be run against us.

**Verdict: build later, behind two named preconditions** — a per-option scoring
endpoint (which is Flide's Eq. 1 and a small server feature) and a named decision
checkpoint with reference outputs. Until both exist, adopting the benchmark would
only measure other engines. It is recorded because it is the quality harness for
the feature Flide makes possible, and because its findings about option logic are
the reason the scoring surface should report the full distribution rather than an
argmax.

---

## 8. Tree-Ising: weight repair by discrete optimisation (sidecar investigation)

*Tree-Ising: Repairing Language Models with Bit-Level Ising Edits.* Salloum, Ali.
arXiv:2611.14210v1.

**Claim.** A looping small model can be repaired without gradients by casting the
edit as a sparse discrete optimisation: one QUBO per output row over the model's
BF16 state tree, restricted to sign flips and exponent shifts that preserve each
group's shared exponent and mantissas, solved by GPU annealing. Reported on
Qwen3.5-0.8B: 54.6% MATH-500 against 51.7% for a frontier agent's edit and 7.9%
for the base model, with 13–17% fewer weights changed, 2.2–2.3× fewer bit flips,
four forward passes, and about $0.90 of GPU time per repair.

**Kraken today.** Inference only. No optimiser, gradients, calibration or
quality-evaluation pipeline exists. Preserve read-only source weights and do not
silently convert or rewrite GGUFs during optimization.

**Investigate a sidecar path, not in-place repair.** A separate offline tool could
write a distinctly named repair artifact while keeping the original GGUF
byte-identical; the inference engine could optionally load that artifact as a
runtime overlay. This leaves the source file read-only, but the effective model
still changes, so such a mode requires explicit opt-in, a compatible named
checkpoint, clear provenance, and quality validation. A lower-risk router-only
correction could be evaluated as an opt-in policy without changing stored
weights, but it also changes model behavior and must be evaluated as such. Neither
approach is implemented or approved as a default.

The implementation prerequisite is to determine whether the paper's edit
variables map to Kraken's actual quantized layouts and whether their group-scale
constraints survive as an external overlay. Preserve a base-file hash, report the
changed values/routes, compare exact unmodified output with the ordinary path, and
measure held-out perplexity/task quality before claiming a repair. The existing
structural observation remains useful: group-quantized layouts share scales and
codebooks, so constraints must be derived per supported dtype rather than assumed
to match the paper's BF16 state tree.

---

## What the eight add up to

Three of them are about where state lives and how it is addressed — PagedAttention
(the storage), Flide (the metadata and the recurrent case) — plus Edge0, which is
about where *weights* live. Those three are engine work and the first two are the
same job, split into a storage half and an execution half. The order is pages,
then forks, then the scoring surface that gives forks a consumer.

The remaining items have different prerequisites: e-means is offline codebook
research, SparseRead is agent-side, and MMDM is gated on a scoring surface we do
not yet have. Tree-Ising and prediction-as-routing are open, opt-in investigations
rather than accepted defaults: the former needs a source-preserving repair-sidecar
path compatible with supported quantization, and the latter needs a trained
predictor plus quality evidence. Both must preserve the exact unmodified baseline;
neither has a usable artifact or measured result yet.

Two things none of these papers settles for us, and which stay as they are in the
register: no number here was reproduced on our hardware, and the native-HIP side
of every item above is unverified.

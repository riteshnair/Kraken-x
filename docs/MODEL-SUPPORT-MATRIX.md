# Model support matrix

This matrix describes the current source, not a filename inventory. The decisive
sources are `src/arch.cpp`, `src/model.cpp`, `src/engine.cpp`, `src/quant.cpp`,
and `tests/test_kraken.cpp`.

Support has four independent dimensions:

1. The GGUF architecture and tensor schema must be recognized.
2. The selected backend must implement every operation used by the file.
3. Execution tests must cover the claimed graph and tensor shapes.
4. A named real checkpoint must be compared with an independent reference before
   real-model correctness is claimed.

`ArchSupport::Yes` means that the architecture has an implemented target or draft
path. It does not mean that every exported file, backend, multimodal tower, or
model-quality requirement has been verified.

## Evidence labels

| Label | Meaning |
| --- | --- |
| Implemented | Source contains a loader and execution path |
| Fixture-tested | Synthetic unit, differential, or whole-graph tests exercise the path |
| Real-model observed | A recorded run loaded or generated from a named checkpoint |
| Real-model verified | A named checkpoint was checked against an independent reference |
| Partial | A defined subset runs, with the excluded feature named |
| Draft-only | The file must attach to a target and cannot generate alone |
| Refused | The current file or configuration is rejected by name |
| Unverified here | The required checkpoint or native hardware is unavailable in this environment |

Historical real-model observations are listed in [Model status](MODEL-STATUS.md).
They are not current-build verification because the original checkpoints and raw
trace artifacts are not retained here.

## Target architectures

The following groups summarize `ArchSupport::Yes` target entries. Unless a more
specific limitation is stated, their evidence is implemented and fixture-tested,
not independently verified on a named real checkpoint.

| Architectures | Implemented path | Current boundary |
| --- | --- | --- |
| `llama`, `qwen`, `qwen2`, `qwen3`, `qwen38`, `internlm2`, `olmo`, `olmo2`, `xverse`, `seed_oss`, `smollm3` | Dense transformer families and their schema variants | `qwen38` is resolved by file tensors, not filename alone; file and backend gates still apply |
| `qwen2moe`, `qwen3moe`, `olmoe` | Routed and shared expert execution | Real-model reference comparison remains necessary for each family |
| `gpt2`, `gptneox`, `gptj`, `phi2`, `phi3`, `minicpm`, `minicpm3` | Dense pre-norm and fused-projection variants | Learned-position and ALiBi configurations have batching restrictions |
| `mpt`, `bloom`, `falcon` | ALiBi or parallel-branch variants | ALiBi configurations have batching restrictions |
| `chatglm`, `glm4`, `glm4moe` | ChatGLM and GLM dense or MoE graphs | File schema decides the runnable graph |
| `phi3moe`, `phimoe`, `mellum` | Phi and Mellum MoE graphs | Norm bias and router semantics are covered by fixtures, not a universal quality claim |
| `bitnet-b1.58` | Low-bit dense inference | Accepted weight formats and optimized kernels vary by backend |
| `qwen35`, `qwen35moe` | Gated DeltaNet recurrent and recurrent-MoE hybrids | Recurrent state excludes the general batch path |
| `lfm2`, `lfm2moe` | LFM2 recurrent and recurrent-MoE hybrids with norm biases | Recurrent state excludes the general batch path; named-checkpoint verification remains separate |
| `gemma`, `gemma3` | Dense Gemma variants | Real-file correctness is not implied by the family entry |
| `gemma4` | Dense and supported routed Gemma4 graphs | A file carrying an unimplemented routed branch layout is refused by name |
| `qwen4exp` | Gated DeltaNet, QSA, hyper-connections, PLE, and MoE | Synthetic whole-graph coverage exists; real-model coherence remains open in `TO-DO.md` |
| `deepseek2` | Lite multi-head latent attention | Split or shared layouts are selected and checked per file |
| `deepseek4`, `deepseek41` | Shared-latent MLA, compressed attention, Sinkhorn hyper-connections, hash routing, and Engram | Compressed attention depends on backend capability; real DeepSeek checkpoints remain unverified here |
| `nemotron_h`, `nemotron_h_moe` | Mamba-2, attention, and MoE layer combinations | Fixture-tested forward paths; no named-checkpoint verification in this environment |
| `laguna` | Sigmoid routing, output gates, sliding-window and full-attention layers | Whole-graph fixture coverage exists; expert residency affects throughput |
| `gpt-oss` | Attention sinks, router and expert biases, windowed layers | Synthetic whole-graph coverage exists |
| `k2-horizon` | Text stack with MoVA routed values and grouped band normalization | `test_k2_horizon_mova` covers the text graph; multimodal vision towers are not included |
| `minimax-m3` | MiniMax sparse attention and MoE schema | Synthetic graph coverage only; no 428B checkpoint has been run here |
| `muse-glimmer` | Attention gates, post-branch norms, and selective rotary | Implemented and fixture-tested |
| `spark2_5` | Fused QKV, per-head output gate, and gated FFN | Implemented and fixture-tested |
| `diffuse` | Cacheless bidirectional attention and block diffusion sampling | `test_diffuse_forward` covers execution; real LLaDA quality and reference agreement remain unverified here |

## Partial architectures and modality limits

| Architecture | State | Explicit boundary |
| --- | --- | --- |
| `qwen2vl`, `qwen3vl` | Partial | Text execution is implemented; vision towers and projector execution are not claimed |
| `gemma2` | Partial | Attention-logit softcap has a reference implementation; a backend without that operation refuses the file |
| `qwen4-mtp` | Partial, draft-only | Loading, probing, and chained drafting are implemented; real acceptance and speedup are unmeasured |
| `dspark` | Partial, draft-only | The schema and activation-free Markov chain are fixture-tested; no real checkpoint has been run here |

## Draft-head architectures

| Architectures | State | Use |
| --- | --- | --- |
| `dflash`, `qwen35-dflash-draft`, `dflash-draft` | Implemented draft-only path | Attach through the supported draft-head options; do not run as a standalone target |
| `deepseek4-dspark` | `ArchSupport::No` as a standalone model, synthetic sidecar path implemented | Attach with `--mtp`; real trunk and nextn verification remains open |

Draft recognition is not evidence of proposal quality, acceptance rate, or an
end-to-end speedup. Greedy draft verification and MTP are implemented, while
sampled speculative decoding and tree attention remain open.

## Deliberate file and configuration refusals

Kraken refuses unsupported work rather than substituting a similar graph. Common
reasons include:

- An unknown architecture without enough metadata to establish its geometry.
- A tensor that declares an unimplemented computation.
- An unsupported quantization ID.
- A backend that lacks a required attention, recurrent, normalization, or
  multi-sequence operation.
- A routed Gemma4 layout whose branches do not match the implemented graph.
- A draft-only file supplied as a standalone target.
- A batched request using recurrent state, diffusion, unsupported per-row MLA,
  QSA or MSA, hyper-connections, PLE, route prediction, speculation, learned
  positions, ALiBi, or tiered KV.

`KRK_ARCH_ALLOW_EXPERIMENTAL=1` can expose partial loader work for development.
It does not implement a missing graph or bypass backend safety checks.

Use the actual file as the final input to the verdict:

```sh
./build-cpu/kraken-inspect model.gguf --quant
./build-cpu/kraken -m model.gguf --cpu --info
```

A runnable verdict establishes loader acceptance. It does not establish output
quality or native-device correctness.

## Quantization formats

`dtype_supported()` currently accepts 40 dtype cases. Acceptance means that a
reference decode path exists. It does not mean that every backend has a dedicated
fused dot kernel or that all formats have equal throughput.

| Group | Formats |
| --- | --- |
| Floating and integer storage | `F32`, `F16`, `BF16`, `I32` |
| Legacy GGML | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` |
| Importance-matrix quants | `IQ4_NL`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ3_S`, `IQ2_S`, `IQ1_M`, `IQ4_XS` |
| Floating-point microscaling | `NVFP4`, `MXFP4`, `ROCMFP4`, `ROCMFP4_FAST` |
| Ternary and low-bit | `TQ1_0`, `TQ2_0`, `I2_S`, `Q1_0`, `Q2_0`, `Q2_0_64`, `PQ2_0`, `PTQ1_0` |
| Additional ROCmFPX decoders | `Q6_0_ROCMFPX`, `Q3_0_ROCMFPX`, `Q2_0_ROCMFPX` |

Q2_K, Q3_K, Q4_0, Q4_K, Q6_K, Q8_0, TQ1_0, PTQ1_0, TQ2_0, PQ2_0, Q2_0,
Q2_0_64, and I2_S have dedicated host dot or tiled entry points. Other accepted
formats can use generic dequantization and tiled arithmetic. Kernel selection
also depends on shape, ISA, backend, and numerical controls.

The ROCmFPX IDs 102, 104, and 107 remain refused because this tree has no decoder
for them. GGUF type ID 42 is the llama-dx 128-value Q2_0 layout here; ID 48 is the
64-value `Q2_0_64` layout.

PrismML `PQ2_0` and `PTQ1_0` files can declare a block-1024 Hadamard activation
transform in `prism.hadamard` metadata. The transform is part of execution, not a
silent weight conversion.

For performance, compare complete model runs on one host with matched controls.
Do not infer model-wide tokens/s from an isolated dot-product benchmark.

## Verification required before promoting a claim

A family moves from fixture-tested to real-model verified only with all of the
following:

1. The exact checkpoint and revision are identified.
2. An independent runtime supplies reference logits or a fixed greedy sequence.
3. Prompt, context, sampling, KV format, and backend are recorded.
4. CPU and HIP results are reported separately.
5. Execution correctness and application-specific model quality are reported as
   separate outcomes.

Current blocked validation and implementation gaps are consolidated in
[TO-DO.md](../TO-DO.md).

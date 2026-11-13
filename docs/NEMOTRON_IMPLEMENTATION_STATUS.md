# Nemotron-H and Nemotron-H-MoE

This is the current support note for `nemotron_h` and `nemotron_h_moe`. It
replaces the obsolete status report that described a broken build and placeholder
forward passes.

## Implemented graph support

Both architecture entries are `ArchSupport::Yes` in `src/arch.cpp`.

| Component | Implementation |
| --- | --- |
| Metadata and tensor schema | Loaded by `Model::load` using per-layer tensor presence |
| Mamba-2 blocks | `Engine::mamba2_forward` drives the recurrence and state |
| Causal convolution | Implemented through the backend recurrent operations |
| SSM parameters | `ssm_a`, `ssm_d`, `ssm_dt`, and related tensors are loaded |
| Attention layers | Supported for hybrid stacks that include attention |
| MoE layers | Routed and shared expert paths support the Nemotron layout |
| Activation | Gate-less squared-ReLU expert FFN is selected by the architecture flags |
| NVFP4 | Reference decoding exists; actual backend kernel selection remains shape and backend dependent |
| HIP | Mamba-2 SSD kernels and launch wrappers exist in the HIP backend |
| Tests | `test_nemotron_h_forward` and `test_nemotron_h_moe_forward` exercise forward behavior |

Nemotron Mamba-2 is not the same recurrence as Gated DeltaNet. The engine keeps
separate state geometry and forward logic rather than substituting one for the
other.

## Layer conventions

The observed Nemotron-H-MoE schema combines several layer kinds:

- Mamba-2 layers carry `ssm_in`, `ssm_out`, `ssm_conv1d`, `ssm_a`, `ssm_d`,
  `ssm_dt`, and normalization tensors.
- MoE layers carry router, routed expert, and shared expert tensors.
- Hybrid files can also contain attention layers.

Layer parity in one observed checkpoint is not a universal rule. The loader uses
the file's actual tensors and metadata.

## Inspection and execution

```sh
./build-cpu/kraken-inspect nemotron.gguf --quant
./build-cpu/kraken -m nemotron.gguf --cpu --info
./build-cpu/kraken -m nemotron.gguf --cpu \
  -p "Summarize the design." -n 32 --greedy
```

Use `--intel` for the optimized x86 backend or a HIP build for a supported AMD
GPU. Backend selection does not relax a missing-operation refusal.

## Current evidence boundary

The implementation and synthetic forward tests are present. No named Nemotron
checkpoint with independent reference logits is available in this environment.
The following claims therefore remain open:

- Real-checkpoint graph equivalence across full generation runs.
- Output quality for a specific quantization and KV configuration.
- Current native HIP throughput on gfx1031 or gfx1201.
- End-to-end speed for large expert sets under a bounded residency budget.

A CPU fixture passing does not verify every HIP kernel. An NVFP4 decoder passing
does not establish model accuracy or a dedicated device GEMM for every shape.

Relevant pending validation is consolidated in [TO-DO.md](../TO-DO.md), and the
cross-family evidence labels are defined in
[Model support](MODEL-SUPPORT-MATRIX.md).

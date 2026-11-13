# LLaDA and masked diffusion

This note describes the implemented `diffuse` path and its remaining evidence
gaps. It supersedes the earlier implementation plan that described the loader,
forward pass, and sampler as missing.

## Model behavior

LLaDA-family masked discrete diffusion starts with a prompt followed by mask
tokens. Each denoising step evaluates the sequence with bidirectional attention
and commits selected masked positions. Tokens outside the active block remain
unchanged.

LLaDA2.x adds block scheduling and grouped MoE routing. Some variants also add
editing tokens and post-edit schedules. Support for an architecture name does
not imply support for every variant or exporter schema.

## Implemented Kraken path

| Component | Current implementation |
| --- | --- |
| Loader | Reads `diffuse.*` metadata and maps the family tensor names onto canonical slots |
| Fused QKV | Splits `attn_qkv` into q, k, and v row views without dequantizing the matrix |
| Attention | Uses the full sequence with `causal = false` |
| State | Runs a cacheless full-sequence forward for each denoising step |
| Routing | Supports group-limited expert selection, selection bias, normalization, and routed scaling |
| Sampler | Implements block denoising in `src/diffusion.cpp` |
| CLI | Provides `--diffusion-steps` and `--diffusion-block` |
| Tests | `test_diffuse_forward` exercises the synthetic graph and denoising behavior |

The architecture entry is `ArchSupport::Yes` in `src/arch.cpp`. A file can still
be refused if its tensors, metadata, or selected backend do not satisfy the
implemented contract.

## Reference schema

The schema was read from `LLaDA2.2-flash-Q4_K_S.gguf`. These values describe that
file, not mandatory defaults for every `diffuse` checkpoint.

| Key | Observed value |
| --- | --- |
| `general.architecture` | `diffuse` |
| `diffuse.model_type` | `llada2_moe` |
| `diffuse.block_count` | 32 |
| `diffuse.embedding_length` | 4096 |
| `diffuse.attention.head_count` | 32 |
| `diffuse.attention.head_count_kv` | 4 |
| `diffuse.head_dim` | 128 |
| `diffuse.rotary_dim` | 64 |
| `diffuse.rope.freq_base` | 3000000 |
| `diffuse.context_length` | 131072 |
| `diffuse.vocab_size` | 157184 |
| `diffuse.mask_token_id` | 156895 |
| `diffuse.eos_token_id` | 156892 |
| `diffuse.delete_token_id` | 156930 |
| `diffuse.split_token_id` | 156931 |
| `diffuse.block_length` | 32 |
| `diffuse.moe_block_size` | 32 |
| `diffuse.expert_count` | 256 |
| `diffuse.expert_used_count` | 8 |
| `diffuse.expert_shared_count` | 1 |
| `diffuse.first_k_dense_replace` | 1 |
| `diffuse.n_group` / `topk_group` | 8 / 4 |
| `diffuse.norm_topk_prob` | 1 |
| `diffuse.routed_scaling_factor` | 2.5 |

Important tensor conventions include:

- `blk.N.attn_qkv.weight` stores fused q, k, and v rows.
- `blk.N.post_attn_norm.weight` is the pre-FFN norm.
- Routed expert banks use expert count in the final tensor dimension.
- `moe_gate.weight` and its bias implement the router.
- Layer zero can use dense FFN tensors while later layers use routed and shared
  experts.

## CLI example

```sh
./build-cpu/kraken -m llada.gguf --cpu \
  -p "Explain masked diffusion." \
  --max-tokens 128 \
  --diffusion-steps 32 \
  --diffusion-block 32 \
  --greedy
```

The prompt plus requested diffusion generation region must fit in `--ctx`.
Diffusion does not use the causal KV decode loop. Its useful measurements include
denoising steps per second, committed tokens per step, total latency, and expert
I/O. A causal tokens/s number is not a complete description of this workload.

## Memory scope

The observed LLaDA2.2 Q4_K_S file is about 59 GB. On a 16 GB GPU, routed expert
residency and storage traffic can dominate execution. Such a run measures the
complete tiered system, not only kernel throughput.

## Verification boundary

The current evidence is implemented and fixture-tested. This environment has no
named real LLaDA checkpoint run with independent reference outputs. Before
claiming real-model verification:

1. Compare mask scheduling and committed tokens with the reference implementation.
2. Check bidirectional logits on fixed real-weight inputs.
3. Exercise grouped routing and shared experts on the actual checkpoint.
4. Record editing-token behavior separately if the variant uses it.
5. Measure quality and total latency independently.

Real-model diffusion verification remains tracked in [TO-DO.md](../TO-DO.md).

# Kraken user guide

This guide covers the current command-line and server interfaces. Historical
benchmarks and model-specific engineering details remain in the other documents
in this directory. They are not promises about unmeasured hardware.

Use `kraken --help` and `kraken-server --help` for the complete option list. Run
commands from the repository root and replace `build-cpu` with your actual build
directory.

## 1. First run

Inspect the file before loading it:

```sh
./build-cpu/kraken-inspect model.gguf --quant
./build-cpu/kraken -m model.gguf --cpu --info
./build-cpu/kraken -m model.gguf --cpu \
  -p "The capital of France is" -n 64 --greedy
```

The inspector reports architecture, tensor formats, and the loader verdict.
`runnable` means the current loader recognizes the model. It does not guarantee
good output on every backend or at every numerical setting.

Use `--intel` instead of `--cpu` for the optimized x86 CPU backend:

```sh
./build-cpu/kraken -m model.gguf --intel \
  -p "The capital of France is" -n 64 --greedy
```

The name is historical. This path is also the appropriate optimized CPU entry
point for AMD Zen 3. AVX2/FMA and VNNI kernels are selected according to the
compiled and runtime-supported features.

## 2. Generation and chat

Complete a prompt:

```sh
./build-cpu/kraken -m model.gguf --intel \
  -p "Write three reasons to keep weights quantized." \
  -n 160 --temp 0.7 --top-p 0.95 --seed 42
```

Use deterministic greedy decoding for comparisons:

```sh
./build-cpu/kraken -m model.gguf --intel \
  -p "Write three reasons to keep weights quantized." \
  -n 160 --greedy
```

Chat using the model's template:

```sh
./build-cpu/kraken -m model.gguf --intel --chat \
  --system "Answer clearly and concisely." \
  -p "What is the difference between RAM and VRAM?" \
  -n 256
```

Start an interactive chat by omitting `-p`:

```sh
./build-cpu/kraken -m model.gguf --intel --chat
```

Inspect template rendering and tokenization:

```sh
./build-cpu/kraken -m model.gguf --intel --chat \
  --system "Be brief." -p "Hello" --tokenize
```

Override a template explicitly when needed:

```sh
./build-cpu/kraken -m model.gguf --intel --chat \
  --chat-template chatml -p "Hello"
```

`--chat-template` accepts a supported built-in name or a `.jinja` path. An
unsupported template feature is reported. Kraken does not silently turn every
model into ChatML.

## 3. Memory planning

### Understand the budgets

Different flags control different allocations:

| Option | What it bounds |
| --- | --- |
| `--expert-cache-mb N` | Resident MoE expert tier |
| `--expert-warm-mb N` | Host warm expert tier, with `0` disabling it |
| `--host-ram-mb N` | Host tiers covered by the process memory plan |
| `--kv-hot-mb N` | Hot KV tier |
| `--kv-warm-mb N` | Warm KV tier, subject to cold-spill policy |
| `--vram-cap-mb N` | Total HIP device memory plan |
| `--ctx N` | Maximum token context, not an immediate allocation of generated text |

The model mapping, dense weights, activations, and allocator overhead are not all
included in one expert-cache number. Check operating-system RSS and the `--info`
ledger when validating a memory target.

### Approximately 1.5 GiB of hot experts

```sh
./build-cpu/kraken -m moe.gguf --intel \
  --expert-cache-mb 1536 \
  --expert-warm-mb 4096 \
  --host-ram-mb 8192 \
  --ctx 16384 --kv-quant turbo3 \
  -p "Explain expert caching." -n 128 --greedy
```

This does not cap total RSS at 1.5 GiB. It caps the selected expert tier. The
warm tier deliberately adds host memory. Set `--expert-warm-mb 0` if every cold
expert read should come directly from the mapped file, then measure the I/O cost.

### Expert routing controls

Cache-only residency changes should not change which experts the router selects.
The following options are different because they deliberately influence routing:

- `--router-resident P`
- `--cache-prior L`

They can reduce expert transfers while changing outputs. Benchmark them with a
separate quality evaluation, not only a tokens/s comparison.

### Choosing the expert budgets from a recorded trace

A cache size, a pin budget and a host share can be compared without loading the
model once per arm. Record one run's routing, then replay it through the same
`ExpertCache` the run used:

```sh
# 1. Record the routing of a short generation (one TSV, one line per acquire).
KRK_TRACE_EXPERTS=1 KRK_TRACE_PATH=expert_trace.tsv \
  ./build-cpu/kraken -m moe.gguf --intel -p "Write a short sort." -n 128 --greedy

# 2. Compare cache sizes and host shares offline.
python3 scripts/moe_cache_tune.py --trace expert_trace.tsv \
  --expert-mib 1.95 --budget-mib 1440,2880,5760 --shares 0,0.5,1 \
  --base-ms 30
```

The script prints one row per configuration with the replayed hits, promotions,
mapping reads and host-computed experts, predicts a per-token time from the read
and promotion rates you pass (or the recorded defaults), and recommends the
lowest host share within 5% of the fastest. `--markdown` prints the table for a
report, and `--out DIR` caches replay results so trying another size is instant.

Two things to keep in mind:

- The prediction is only as good as the three rate constants. `--print-defaults`
  shows the recorded ones; substitute your own before believing a percentage.
- `--expert-mib` must match the model, because the replay works in expert slots:
  the policy divides the budget by the expert footprint, and that ratio is the
  whole decision. `kraken-inspect file.gguf --info` prints the routed set size to
  divide by the expert count.

The replay models the byte budget, the per-layer leases and the host arm's share
of the overflow. It does not model staging, prediction or lookahead, so a run
that turns those on can still beat the prediction.

### Long context

```sh
./build-cpu/kraken -m model.gguf --intel \
  --ctx 32768 --kv-quant turbo3 \
  --kv-warm-mb 8192 \
  --kv-cold-dir /fast-nvme/kraken-kv \
  -p "Paste or generate a long prompt" -n 128 --greedy
```

Choose a real writable spill directory. If a KV page cannot be retained or
spilled safely, the engine must not treat missing history as valid context.
Read startup warnings and capacity messages.

KV quantization reduces memory and traffic but changes numerical inputs to
attention. Compare quality and outputs against `f16` before adopting a new cache
format for a workload.

## 4. Backend selection

| Backend | Selection | Intended use |
| --- | --- | --- |
| HIP | Default when built and available | Supported AMD GPU execution |
| CPU | `--cpu` | Host execution and reference comparisons |
| Optimized x86 CPU | `--intel` | AVX2/FMA with runtime VNNI dispatch |
| Virtual NPU GEMM | `--npu` | Offload matrix work to an NPUXL build |
| Virtual NPU graph | `--npu-cpu` | Offload all supported graph operations |

The CPU backend contains both reference arithmetic and optimized paths. Environment
controls such as `KRK_CPU_DOT_FAST=0` and `KRK_CPU_DOT_AVX2=0` select stricter
reference behavior for differential work. Check the current help and kernel
documentation before treating a run as bit-exact.

NPU support requires a separate NPUXL checkout and `-DKRK_ENABLE_NPU=ON`. NPU
W8A8 and NPU attention are opt-in numerical or dispatch changes. They are not
silent defaults.

## 5. Prefill, decode, and determinism

`--chunk N` controls prefill batching. `--prefill exact` prioritizes arithmetic
that is independent of chunk boundaries. `--prefill fast` can select a different
GEMM path and summation order.

Compare prefill invariance:

```sh
./build-cpu/kraken -m model.gguf --intel \
  -p "A fixed prompt" --chunk 1 --prefill-hash
./build-cpu/kraken -m model.gguf --intel \
  -p "A fixed prompt" --chunk 8 --prefill-hash
```

Use the same model, backend, numerical controls, and prompt. The hash command
uses the exact rung by design. Equal hashes are evidence for this execution
property, not a measure of answer quality.

Speculative decoding:

```sh
./build-cpu/kraken -m large.gguf --intel \
  --draft small.gguf --draft-tokens 4 \
  -p "Explain speculative decoding." -n 128 --greedy
```

A greedy target verifies draft proposals. Test the specific model pair and
backend before assuming that speculation is faster. Sampled generation does not
use the greedy verify-and-accept contract.

Probe a compatible embedded or sidecar MTP head before using it for speculation:

```sh
./build-cpu/kraken -m target.gguf --intel \
  --mtp mtp-head.gguf --mtp-probe
./build-cpu/kraken -m target.gguf --intel \
  --mtp mtp-head.gguf --mtp-spec --draft-tokens 4 \
  -p "Explain multi-token prediction." -n 128 --greedy
```

The head must match the target architecture and tensor contract. Fixture-tested
loading does not establish a useful acceptance rate. Record proposals, accepted
tokens, acceptance rate, and end-to-end latency on the real matched pair.

Run a supported masked-diffusion model with an explicit denoising schedule:

```sh
./build-cpu/kraken -m llada.gguf --intel \
  -p "Explain masked diffusion." \
  --max-tokens 128 --diffusion-steps 32 --diffusion-block 32 --greedy
```

The prompt and diffusion generation region must fit in `--ctx`. This path uses
cacheless bidirectional forwards rather than the causal KV decode loop. Measure
steps per second, committed tokens, and total latency. See
[LLaDA and masked diffusion](LLaDA.md) for the evidence boundary.

## 6. HTTP server

Start the server:

```sh
./build-cpu/kraken-server -m model.gguf --cpu --ctx 8192
```

The default port is `8080`. `KRK_PORT` can select another valid port.

Chat request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "model.gguf",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is quantized inference?"}
    ],
    "temperature": 0,
    "max_tokens": 128
  }'
```

The example binds to loopback on the local machine. If you expose the server to
a network, add appropriate access control and transport security at the deployment
boundary. Do not assume that a development server provides authentication.

Stream chat deltas with SSE:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "model.gguf",
    "messages": [{"role": "user", "content": "What is quantized inference?"}],
    "temperature": 0,
    "max_tokens": 128,
    "stream": true
  }'
```

A failed streaming write raises the request cancellation flag and stops decoding
at the next token. Prefix reuse can avoid repeated prefill for matching prompts.
Cached token usage is an execution metric, not evidence of response correctness.
The held prefix is not a general radix tree.

Available endpoints are `/`, `/health`, `/v1/models`, `/v1/tools/rv32`,
`/v1/completions`, and `/v1/chat/completions`. The service loads one model.

With more than one KV slot it decodes several requests in one pass and admits a
new request between steps, so a request arriving mid-generation waits for one
step rather than for the running set to drain. The admitted prompt is prefilled
one `--chunk` slice per step: while anything is decoding, a step spends one
slice on the waiting prompt and the rest of its time on the decoders, so a long
prompt costs them one chunk of delay, not one prompt. With nothing decoding, the
prompt is drained in the same call, because then there is nobody to starve. The batch is constrained: the
engine refuses models and features whose state is per sequence rather than per
row, and prefix reuse applies only to the serial path. This is iteration-level
admission over that constrained batch, not unrestricted continuous batching.

## 7. Testing

Build the relevant targets first:

```sh
cmake --build build-cpu --target \
  kraken kraken-server kraken-tests kraken-bench kraken-inspect -j
```

Run the unit suite and numerical gate:

```sh
./build-cpu/kraken-tests
./build-cpu/kraken-bench --gate
```

Run one named test group:

```sh
./build-cpu/kraken-tests --run-group test_prefill_chunk_bitexact
```

Compare device and CPU generation where supported:

```sh
bash scripts/coherence_check.sh model.gguf
```

Keep these claims separate:

- Compile passed.
- Focused tests passed.
- Full suite passed.
- Device output matched the reference for a fixed prompt.
- Model quality met an application-specific evaluation.

One level does not prove the next.

## 8. Performance measurement

Basic model benchmark:

```sh
./build-cpu/kraken -m model.gguf --intel --bench --greedy
```

For an optimization comparison, use interleaved runs of the control and changed
binary. Record medians and spread rather than selecting the fastest run.

Minimum benchmark record:

| Field | Required detail |
| --- | --- |
| Source | Git revision and local changes |
| Build | Compiler, build type, native or portable flags |
| Hardware | CPU or GPU model, RAM, storage, operating system |
| Model | Exact GGUF, architecture, quantization |
| Workload | Prompt tokens, generated tokens, context, rows |
| Runtime | Backend, ISA, threads, KV and expert settings |
| Timing | Warm-up, repetitions, timing boundary |
| Result | Load, prefill, decode, total, RSS or transferred bytes as relevant |

Do not describe an isolated dot-product improvement as an end-to-end inference
improvement without measuring the complete run.

## 9. Troubleshooting

### Unsupported model or format

Run `kraken-inspect model.gguf --quant`. Preserve the refusal text in bug reports.
Renaming a tensor or forcing a format ID does not implement the missing graph.

### Nonsense output

1. Return to `--greedy` with a short fixed prompt.
2. Use `--kv-quant f16` and disable routing-bias experiments.
3. Compare `--cpu` with strict reference controls against the optimized backend.
4. Inspect chat tokenization with `--tokenize`.
5. Test the model with a known independent runtime if schema correctness is in
   doubt.

### Poor MoE decode speed

Measure cold and warm runs separately. Record expert hits, promotions, host reads,
and resident bytes. A smaller cache can reduce memory while increasing latency.

### Poor long-context prefill

Separate prompt ingestion from generated-token throughput. KV compression lowers
storage traffic but does not remove the attention computation over long prompts.

### Build or test confusion

Delete stale build assumptions, configure again, capture the build exit status,
and run the binaries produced by that build. A previous executable is not evidence
that the latest source compiled.

## 10. Further reading

- [README](../README.md): build and quick-start overview.
- [Architecture](ARCHITECTURE.md): major components and execution flow.
- [CPU low-bit VNNI](CPU-LOWBIT-VNNI.md): optimized quantized CPU kernels.
- [Tiered execution](ARCHITECTURE-TIERED.md): residency, ownership, and historical
  measurements with their scope.
- [Model support](MODEL-SUPPORT-MATRIX.md): architecture, quantization, and
  evidence levels.
- [Model status](MODEL-STATUS.md): historical checkpoint runs and current
  corrections.
- [Engineering notes](TECHNICAL-README.md): implementation details.
- [Pending work](../TO-DO.md): verified gaps and blocked validation.

<img width="768" height="512" alt="Octopus AI Workshop in a Vintage Study" src="https://github.com/user-attachments/assets/7cff7936-e8fe-4da3-a3c8-6be9c04c5b06" />

# Kraken

Kraken is a C++17 GGUF inference engine with native AMD HIP kernels and optimized
x86 CPU paths. It keeps model weights quantized during inference, supports dense
and Mixture of Experts models, and provides tiered expert and KV-cache residency
for machines with limited memory.

There is no PyTorch or Python in the inference hot path. The HIP backend does not
require hipBLAS, MIOpen, or rocBLAS. CPU execution does not require ROCm.

## Current capabilities

- Native HIP kernels for RDNA2, RDNA3, and RDNA4 targets.
- Portable CPU reference arithmetic, with optional AVX2/FMA and runtime AVX-VNNI
  kernels for supported quantized formats.
- Quantized weights remain quantized in memory. They are decoded for computation
  rather than expanded into a second full model.
- Dense and supported MoE GGUF models, including bounded expert residency and
  host-backed warm storage.
- Long-context KV tiers with `f16`, `int8`, `turbo3`, and experimental `rot4`
  cache formats.
- Command-line generation, interactive chat, an OpenAI-compatible HTTP server,
  model inspection, correctness tests, and benchmarks.
- Greedy speculative decoding with a draft model or a supported MTP head.

Support depends on the model architecture, tensor schema, quantization format,
and selected backend. A GGUF extension alone does not prove that a file is
runnable. Run `kraken-inspect` first. Kraken refuses unsupported features by name
instead of silently substituting a different model graph.

See the [user guide](docs/USER-GUIDE.md) for detailed operation and the current
[pending-work list](TO-DO.md) for verified gaps. This README focuses on building
and using the current tree.

## Quick start

### Inspect and run a model

```sh
./kraken-inspect model.gguf --quant
./kraken -m model.gguf --info
./kraken -m model.gguf -p "The capital of France is" -n 64 --greedy
```

Use the binary in your chosen build directory, for example
`build-hip/kraken` or `build-cpu/kraken`. On Windows, use the corresponding
`.exe` files.

For a chat model:

```sh
./kraken -m model.gguf --chat \
  --system "Answer clearly and concisely." \
  -p "Explain quantization in two paragraphs." \
  -n 256
```

`--chat` uses the model's supported Jinja chat template when available. Use
`--tokenize` to inspect the exact token IDs supplied to the model.

## Build

### Linux with AMD HIP

Install a compatible ROCm/HIP SDK, CMake, and Ninja, then run:

```sh
sh ./scripts/build_linux.sh
```

Or configure a specific GPU target directly:

```sh
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DKRK_GPU_TARGETS=gfx1201
cmake --build build-hip -j
```

Common targets include `gfx1031` for RDNA2 and `gfx1201` for RDNA4. Build for the
GPU you actually have. A HIP binary compiled for another architecture is not a
portable fallback.

### Windows with AMD HIP

Install the AMD HIP SDK, then run in PowerShell:

```powershell
.\scripts\build_windows.ps1 -Targets "gfx1031;gfx1201"
```

For one GPU, use a single target such as `-Targets gfx1201`.

### CPU only, portable build

```sh
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DKRK_ENABLE_HIP=OFF
cmake --build build-cpu -j
```

Run the portable CPU backend with `--cpu`. The optimized `--intel` backend is an
x86 CPU backend despite its historical name. It uses AVX2/FMA and dispatches VNNI
kernels at runtime when supported. It does not require AVX-512.

For a compiler-controlled AVX2 build with AVX-VNNI target dispatch:

```sh
cmake -S . -B build-avx2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DKRK_ENABLE_HIP=OFF \
  -DKRK_NATIVE_CPU=OFF \
  -DCMAKE_CXX_FLAGS="-mavx2 -mfma -mf16c -DKRK_DOT_AVXVNNI=1"
cmake --build build-avx2 -j
```

This does not make AVX-VNNI mandatory. Runtime feature checks select the VNNI
path only on supported processors.

## CPU examples

### AMD Zen 3

Zen 3 provides AVX2 and FMA3, but not AVX-VNNI or AVX-512. The same binary keeps
those unsupported instruction paths disabled at runtime.

```sh
./build-avx2/kraken -m model.gguf --intel \
  -p "Summarize the design." -n 128 --greedy
```

For deterministic reference comparisons, disable the optional block-dot
accumulation mode:

```sh
KRK_CPU_DOT_FAST=0 KRK_CPU_DOT_AVX2=0 \
  ./build-cpu/kraken -m model.gguf --cpu \
  -p "Summarize the design." -n 128 --greedy
```

The second command is a correctness oracle, not a performance configuration.

### Intel Core Ultra 135H

This processor supports AVX2, FMA3, and 256-bit AVX-VNNI. It does not require an
AVX-512 build.

```sh
./build-avx2/kraken -m model.gguf --intel \
  --ctx 16384 --kv-quant turbo3 \
  -p "Summarize the design." -n 128 --greedy
```

Use `--info` to confirm the selected backend and memory plan. Do not infer the
active ISA from the build flags alone.

## Bounded-memory MoE example

The following example limits the device expert tier to approximately 1.5 GiB,
adds a 4 GiB host warm tier, and uses a 16K token context:

```sh
./build-avx2/kraken -m moe-model.gguf --intel \
  --expert-cache-mb 1536 \
  --expert-warm-mb 4096 \
  --ctx 16384 \
  --kv-quant turbo3 \
  -p "Explain bounded expert residency." \
  -n 128 --greedy
```

`--expert-cache-mb 1536` is an expert-tier budget, not a promise that total
process RSS will be 1.5 GiB. The model mapping, dense weights, KV cache,
activations, host warm tier, and allocator overhead are separate. Measure RSS and
read bytes for the complete process.

This strategy is most useful for MoE models because only selected experts are
needed for a token. Streaming every dense layer from storage on every token is
usually bandwidth-bound and is not equivalent to bounded expert residency.

The router normally remains unchanged. Options such as `--router-resident` and
`--cache-prior` deliberately alter routing to improve locality. They can change
model outputs and quality, so evaluate them separately from cache-only changes.

## Long context and KV cache

```sh
./kraken -m model.gguf --intel \
  --ctx 32768 --kv-quant turbo3 \
  --kv-warm-mb 8192 \
  --kv-cold-dir /fast-nvme/kraken-kv \
  -p "Long prompt here" -n 128
```

| KV format | Purpose | Numerical status |
| --- | --- | --- |
| `f16` | Highest-precision cache option | Reference cache format |
| `int8` | Reduced cache traffic and memory | Quantized, outputs can differ |
| `turbo3` | Smaller cache for long contexts | Quantized, default in the CLI |
| `rot4` | Rotated 4-bit representation | Experimental, evaluate before relying on it |

A smaller KV budget does not make attention computation disappear. Long prompts
still have prefill and attention costs. Cold storage also adds I/O latency.
Use an actual long-context workload when comparing formats.

## HTTP server

Build the `kraken-server` target, then run:

```sh
./build-cpu/kraken-server --model model.gguf --cpu --ctx 8192
```

Send an OpenAI-compatible chat request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "model.gguf",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is GGUF?"}
    ],
    "temperature": 0
  }'
```

The server can reuse a matching KV prefix between requests. Its reported cached
token count describes cache reuse, not a model-quality measurement.

See [docs/USER-GUIDE.md](docs/USER-GUIDE.md) for server options and additional
examples.

## Correctness and numerical modes

Execution correctness and model accuracy are different questions.

- A differential kernel test can prove that an optimized implementation agrees
  with its reference for tested inputs.
- A greedy device-versus-CPU comparison can expose execution divergence.
- Neither test proves that the model gives better answers.
- Quantizing weights, activations, or KV state can change the generated text.
- Activation compensation cannot recover weight information discarded by an
  earlier quantization step.

For repeatable comparisons, use `--greedy`, a fixed prompt, a fixed context, and
the same model file. Keep the default numerical path unchanged when establishing
a baseline.

Useful checks:

```sh
./build-cpu/kraken-tests
./build-cpu/kraken-bench --gate
bash scripts/coherence_check.sh model.gguf
```

The coherence script compares generated text on the selected device and CPU
reference paths. It needs a supported model and the relevant backends. A unit
suite passing does not replace end-to-end model evaluation.

## Benchmarking

```sh
./kraken -m model.gguf --bench --greedy
```

Report the complete configuration:

- Git revision, compiler, build flags, operating system, and hardware.
- Model file, quantization, context, prompt length, and generated token count.
- Backend, ISA path, thread count, expert budgets, and KV format.
- Load time, prefill throughput, decode throughput, and total latency.
- Warm-up, repetitions, spread, and whether weights were already cached.

Kernel speedups are not automatically model-wide speedups. A matrix multiply can
become much faster while total decode remains limited by memory, attention,
expert I/O, synchronization, or other graph operations.

Kraken does not silently recompress GGUF weights to make a benchmark fit.
Offline compression experiments should start from original higher-precision
weights and require separate quality and latency evaluation.

## Troubleshooting

### The model is refused

```sh
./kraken-inspect model.gguf --quant
```

Read the named missing tensor, unsupported architecture, quantization, or graph
feature. Do not rename tensors to bypass a schema refusal.

### The process uses more memory than the expert budget

The expert budget covers one tier only. Inspect the dense model mapping, warm
expert tier, KV tiers, host planning limit, and total RSS. `--info` is the first
check.

### A faster mode changes generated text

Return to `--greedy`, disable experimental KV or routing options, and compare
against the reference CPU mode. A changed continuation is evidence of numerical
divergence, not automatically a correctness bug or an accuracy improvement.

### A HIP build produces a CPU-only binary

Inspect the CMake configure output for SDK detection. Confirm the selected HIP
compiler and GPU target. Rebuild before benchmarking, since an old binary can
otherwise make a failed build appear successful.

## Documentation map

- [User guide](docs/USER-GUIDE.md): commands, server use, memory controls, and
  troubleshooting.
- [Architecture](docs/ARCHITECTURE.md): engine and backend structure.
- [Tiered architecture](docs/ARCHITECTURE-TIERED.md): residency and storage.
- [CPU low-bit VNNI](docs/CPU-LOWBIT-VNNI.md): optimized kernels, controls,
  validation, and isolated measurements.
- [Model support](docs/MODEL-SUPPORT-MATRIX.md): architecture and schema support.
- [Technical reference](docs/TECHNICAL-README.md): implementation details.
- [Pending work](TO-DO.md): the single current list of gaps and blocked validation.

When documentation and `kraken --help` disagree, inspect the current parser and
report the mismatch. Performance tables are historical unless they identify the
exact current revision and workload.

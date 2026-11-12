# Contributor operations

This file contains only durable repository workflow. Current gaps live in
`TO-DO.md`; usage lives in `README.md` and `docs/USER-GUIDE.md`.

## Source of truth

- Read current code and tests before trusting a historical claim.
- Do not recreate deleted planning, audit, campaign, or transcript files.
- Record new pending work only in `TO-DO.md`.
- Keep model weights, generated fixtures, build trees, sanitizer output, traces,
  logs, and benchmark dumps out of Git.
- Never silently convert models or weights during optimization.

## Build and test

CPU-only release build:

```sh
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DKRK_ENABLE_HIP=OFF
cmake --build build-cpu --target kraken kraken-server kraken-tests -j2
./build-cpu/kraken-tests
```

ASan build:

```sh
cmake -S . -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DKRK_ENABLE_HIP=OFF \
  -DCMAKE_C_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build build-asan --target kraken kraken-server kraken-tests -j2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 ./build-asan/kraken-tests
```

Current expected result for both suites is 7579/7579 checks. Update that number
only from an actual complete run. The test runner uses `--run-group <name>`;
`--filter` is unsupported. Build `kraken` and `kraken-server` beside the test
binary because CLI and HTTP groups execute them.

## Change discipline

- Preserve existing numerical and refusal contracts unless the task explicitly
  changes them.
- Do not weaken assertions, suppress sanitizer findings, or pad production
  buffers to hide an invalid access.
- Compare optimized arithmetic against the appropriate exact oracle. Explicitly
  experimental reduction modes must remain opt-in.
- Keep backend lifetime longer than every Engine, Model, and ExpertCache using
  it. Explicit `unload`, `clear`, and `shutdown` paths must also handle partial
  initialization.
- Register new weight and workspace fields in their central teardown lists.
- Hold expert-cache leases whenever multiple returned pointers must remain valid
  across later acquisitions.

## Measurements

Every performance statement needs model, hardware, workload, controls, and
scope. Kernel throughput is not model-wide tokens per second. Use matched A/B
runs and retain enough raw evidence to reproduce the result. Do not substitute
synthetic benchmarks for unavailable native-hardware or real-model validation.

Closed optimization directions and the resources blocking pending validation
are listed in `TO-DO.md`.

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

## Native Windows / MinGW diagnosis

For CPU-only Windows validation, use a fresh build directory for each mode and
keep the compiler/toolchain identical between Debug and Release. Define the SDK
surface needed by the current Windows APIs explicitly; do not rely on a shell's
ambient defaults:

```powershell
cmake -S . -B build-win-release -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DKRK_ENABLE_HIP=OFF `
  "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00"
cmake --build build-win-release --target kraken-tests --verbose
.\build-win-release\kraken-tests.exe --run-group test_minicpm3_whole_graph

cmake -S . -B build-win-debug -G Ninja `
  -DCMAKE_BUILD_TYPE=Debug -DKRK_ENABLE_HIP=OFF `
  "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00"
cmake --build build-win-debug --target kraken-tests --verbose
.\build-win-debug\kraken-tests.exe --run-group test_minicpm3_whole_graph
```

Use separate build trees; never infer the active mode from the directory name.
Leave `KRK_NATIVE_CPU` at the same setting in both (normally its native-host
default); changing the host ISA is a separate portability experiment, not an
optimization-only comparison. Inspect verbose compile commands to confirm the
test and `krk_core` objects use the expected flags. CMake keeps `krk_core` at
`-O0` in Debug and `-O3` in optimized configurations. A Debug pass is diagnostic
evidence only; it does not
validate Release behavior. For an optimization-sensitive failure, preserve the
unmodified Release reproduction first, then try one flag per fresh tree (for
example `-fno-strict-aliasing`, then `-fno-inline`). Do not call either flag a
fix unless the fault is localized and a regression test passes in normal
Release. If available, capture the native debugger's full stack and all-thread
backtrace for the failing group. Report the MinGW/GCC version, CMake and Ninja
versions, exact configure/build commands, group output, process exit code, and
backtrace; do not report only that the process crashed. Run the full suite only
after the focused reproduction is understood, and report the count from that
actual run.

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

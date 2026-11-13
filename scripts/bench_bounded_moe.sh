#!/bin/sh
# Run Kraken's built-in MoE benchmark under several planned host-RAM classes.
# This is a measurement driver, not a hard OS memory limiter: use a native
# machine with the target RAM for RSS claims. --ram-tier changes the engine's
# tier policy but does not constrain unrelated process/system allocations.
#
# Usage:
#   scripts/bench_bounded_moe.sh MODEL.gguf [KRAKEN_EXE] [OUTPUT_DIR]
#
# Optional environment:
#   KRK_BENCH_TIERS="16 24 32"   planned RAM classes in GiB
#   KRK_BENCH_TOKENS=128          greedy decode tokens per run
#   KRK_BENCH_EXPERT_SLOTS=N      explicit resident (layer,expert) slot cap;
#                                 unset means the normal automatic budget
#
# The driver deliberately disables eager warmup, WARM duplication and pinning.
# CPU expert weights remain demand-loaded through Kraken's bounded host pool / 
# GGUF mapping. Each run retains its complete stdout/stderr, including the
# built-in peak-memory, resident-expert, hit-rate, and cold-read statistics.

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 MODEL.gguf [KRAKEN_EXE] [OUTPUT_DIR]" >&2
    exit 2
fi

model=$1
exe=${2:-${KRK_EXE:-./build/kraken}}
outdir=${3:-${KRK_BENCH_OUT:-./bounded-moe-results}}
tiers=${KRK_BENCH_TIERS:-"16 24 32"}
tokens=${KRK_BENCH_TOKENS:-128}
slots=${KRK_BENCH_EXPERT_SLOTS:-}

if [ ! -f "$model" ]; then
    echo "bounded-moe: model not found: $model" >&2
    exit 2
fi
if [ ! -x "$exe" ]; then
    echo "bounded-moe: executable not found or not executable: $exe" >&2
    exit 2
fi
case "$tokens" in
    ''|*[!0-9]*) echo "bounded-moe: KRK_BENCH_TOKENS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$tokens" -le 0 ]; then
    echo "bounded-moe: KRK_BENCH_TOKENS must be a positive integer" >&2
    exit 2
fi
if [ -n "$slots" ]; then
    case "$slots" in
        ''|*[!0-9]*) echo "bounded-moe: KRK_BENCH_EXPERT_SLOTS must be a positive integer" >&2; exit 2 ;;
    esac
    if [ "$slots" -le 0 ]; then
        echo "bounded-moe: KRK_BENCH_EXPERT_SLOTS must be positive" >&2
        exit 2
    fi
fi

mkdir -p "$outdir"
model_name=$(basename "$model")
exe_name=$(basename "$exe")
printf 'Bounded-memory MoE benchmark\nmodel: %s\nexe: %s\n' "$model" "$exe"
printf 'planned tiers (GiB): %s\ntokens: %s\n' "$tiers" "$tokens"
printf 'resident expert slots: %s\n' "${slots:-automatic}"
echo "NOTE: --ram-tier is a policy simulation, not a hard RSS cap."
echo "NOTE: output is not a low-memory validation until run on matching native RAM."

for tier in $tiers; do
    case "$tier" in
        ''|*[!0-9]*) echo "bounded-moe: invalid RAM tier: $tier" >&2; exit 2 ;;
    esac
    if [ "$tier" -le 0 ]; then
        echo "bounded-moe: RAM tiers must be positive: $tier" >&2
        exit 2
    fi
    log="$outdir/${model_name}.${tier}GiB.log"
    set -- --cpu -m "$model" --ram-tier "$tier" \
        --expert-warmup 0 --expert-warm-mb 0 --expert-pin-mb 0 \
        --greedy --max-tokens "$tokens" --bench
    if [ -n "$slots" ]; then
        set -- "$@" --expert-cache-slots "$slots"
    fi
    printf '\n== %s GiB planned RAM ==\n' "$tier"
    printf 'command:'
    printf ' %s' "$exe" "$@"
    printf '\nlog: %s\n' "$log"
    # Preserve the complete diagnostics and propagate failures. A failed run is
    # not a benchmark point and must not be disguised as a zero-throughput pass.
    "$exe" "$@" >"$log" 2>&1 || {
        rc=$?
        cat "$log"
        echo "bounded-moe: run failed (exit $rc), see $log" >&2
        exit "$rc"
    }
    grep -E '^(model|device|prefill|decode|total|experts|expert WARM|\[stats \] memory|\[stats \] expert-pool|\[stats \] experts|\[stats \] cache|\[stats \] warm|\[stats \] transfer)' "$log" || true
    echo "full run log: $log"
done

echo "Runs complete. Compare peak RSS, resident/active expert slots, cache hit rate, cold reads, and decode rate together."
echo "Do not interpret a simulated --ram-tier run as proof that the process fits in that physical RAM class."

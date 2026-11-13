#!/bin/sh
# Run Kraken's built-in MoE benchmark under several planned host-RAM classes.
# This is a measurement driver, not a hard OS memory limiter: use a native
# machine with the target RAM for RSS claims. --ram-tier changes the engine's
# tier policy but does not constrain process/system allocations. On Linux, run
# the whole script inside a dedicated cgroup to enforce a ceiling and capture
# its cgroup high-water counters, for example:
#   systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 -- scripts/bench_bounded_moe.sh MODEL.gguf ./build/kraken results-16g
#
# Usage:
#   scripts/bench_bounded_moe.sh MODEL.gguf [KRAKEN_EXE] [OUTPUT_DIR]
#
# Optional environment:
#   KRK_BENCH_TIERS="16 24 32"   planned RAM classes in GiB
#   KRK_BENCH_TOKENS=128          greedy decode tokens per run
#   KRK_BENCH_EXPERT_SLOTS=N      resident (layer,expert) slot cap (default 1);
#                                 use "auto" for Kraken's normal budget
#   KRK_BENCH_HASH_MODEL=1        include a full SHA-256 model hash (extra I/O)
#   KRK_PYTHON=python3            Python used to read child peak RSS
#
# The driver deliberately disables eager warmup, WARM duplication and pinning.
# CPU expert weights remain demand-loaded through Kraken's bounded host pool /
# GGUF mapping. The default one-slot run tests minimum expert residency; each
# run retains engine diagnostics and a peak-RSS report.

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
slots=${KRK_BENCH_EXPERT_SLOTS:-1}
python=${KRK_PYTHON:-python3}
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
rss_helper="$script_dir/measure_peak_rss.py"

if [ ! -f "$model" ]; then
    echo "bounded-moe: model not found: $model" >&2
    exit 2
fi
if [ ! -x "$exe" ]; then
    echo "bounded-moe: executable not found or not executable: $exe" >&2
    exit 2
fi
if ! command -v "$python" >/dev/null 2>&1; then
    echo "bounded-moe: Python 3 is required to measure peak RSS (set KRK_PYTHON if needed)" >&2
    exit 2
fi
if [ ! -f "$rss_helper" ]; then
    echo "bounded-moe: missing RSS helper: $rss_helper" >&2
    exit 2
fi
case "$tokens" in
    ''|*[!0-9]*) echo "bounded-moe: KRK_BENCH_TOKENS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$tokens" -le 0 ]; then
    echo "bounded-moe: KRK_BENCH_TOKENS must be a positive integer" >&2
    exit 2
fi
if [ "$slots" != auto ]; then
    case "$slots" in
        ''|*[!0-9]*) echo "bounded-moe: KRK_BENCH_EXPERT_SLOTS must be a positive integer or auto" >&2; exit 2 ;;
    esac
    if [ "$slots" -le 0 ]; then
        echo "bounded-moe: KRK_BENCH_EXPERT_SLOTS must be positive" >&2
        exit 2
    fi
fi

mkdir -p "$outdir"
model_name=$(basename "$model")
metadata="$outdir/${model_name}.metadata.txt"
model_bytes=$(wc -c <"$model" | tr -d ' ')
{
    printf 'Bounded-memory MoE benchmark metadata\n'
    printf 'timestamp_utc: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf 'model: %s\n' "$model"
    printf 'model_bytes: %s\n' "$model_bytes"
    printf 'executable: %s\n' "$exe"
    printf 'planned_ram_tiers_gib: %s\n' "$tiers"
    printf 'expert_slot_cap: %s\n' "$slots"
    printf 'tokens: %s\n' "$tokens"
    printf 'measurement_mode: one process startup plus one fixed-prompt --bench request per tier; not a repeated warm-request measurement\n'
    printf 'host_uname: %s\n' "$(uname -a)"
    if [ -r /proc/meminfo ]; then
        grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo || true
        printf 'cpu_model: '
        awk -F: '/model name[[:space:]]*:/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo
        if [ -r /proc/self/cgroup ]; then
            printf 'cgroup_membership:\n'
            cat /proc/self/cgroup
            cgroup_path=$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup)
            if [ -n "$cgroup_path" ]; then
                for field in memory.max memory.current memory.peak memory.swap.max memory.events; do
                    file="/sys/fs/cgroup$cgroup_path/$field"
                    if [ -r "$file" ]; then
                        printf 'cgroup_start_%s: ' "$field"
                        tr '\n' ' ' <"$file"
                        printf '\n'
                    fi
                done
            fi
        fi
    elif command -v sysctl >/dev/null 2>&1; then
        printf 'host_memory_bytes: %s\n' "$(sysctl -n hw.memsize 2>/dev/null || echo unavailable)"
    fi
    if [ "${KRK_BENCH_HASH_MODEL:-0}" = 1 ]; then
        if command -v sha256sum >/dev/null 2>&1; then
            printf 'model_sha256: '
            sha256sum "$model" | awk '{print $1}'
        elif command -v shasum >/dev/null 2>&1; then
            printf 'model_sha256: '
            shasum -a 256 "$model" | awk '{print $1}'
        else
            echo 'model_sha256: unavailable (need sha256sum or shasum)'
        fi
    else
        echo 'model_sha256: not computed (set KRK_BENCH_HASH_MODEL=1; reads the whole model)'
    fi
} >"$metadata"
write_cgroup_end() {
    if [ -r /proc/self/cgroup ]; then
        cgroup_path=$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup)
        if [ -n "$cgroup_path" ]; then
            {
                echo 'cgroup_end_note: high-water counters are process-specific only in a dedicated cgroup'
                for field in memory.max memory.current memory.peak memory.swap.max memory.swap.current memory.swap.peak memory.events; do
                    file="/sys/fs/cgroup$cgroup_path/$field"
                    if [ -r "$file" ]; then
                        printf 'cgroup_end_%s: ' "$field"
                        tr '\n' ' ' <"$file"
                        printf '\n'
                    fi
                done
            } >>"$metadata"
        fi
    fi
}
trap write_cgroup_end EXIT
printf 'metadata: %s\n' "$metadata"
# Keep this identical to run_bench()'s fixed prompt in src/main_cli.cpp so the
# generation-parity pass exercises the same workload as the timing pass.
bench_prompt='The history of computing is a history of abstraction: from relays to vacuum tubes, from transistors to integrated circuits, and now from monolithic kernels to distributed accelerators. In each transition the same question returns — what is the smallest primitive that still expresses the computation?'
reference_tokens=
reference_tier=
printf 'Bounded-memory MoE benchmark\nmodel: %s\nexe: %s\n' "$model" "$exe"
printf 'planned tiers (GiB): %s\ntokens: %s\n' "$tiers" "$tokens"
printf 'resident expert slots: %s\n' "${slots:-automatic}"
echo "NOTE: --ram-tier is a policy simulation, not a hard RSS cap."
echo "NOTE: cgroup limits are observed, not created; use a dedicated systemd scope for enforcement."
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
    timelog="$outdir/${model_name}.${tier}GiB.time.txt"
    tokenlog="$outdir/${model_name}.${tier}GiB.tokens.txt"
    genlog="$outdir/${model_name}.${tier}GiB.generate.log"
    set -- --cpu -m "$model" --ram-tier "$tier" \
        --expert-warmup 0 --expert-warm-mb 0 --expert-pin-mb 0 \
        --greedy --max-tokens "$tokens" --bench
    if [ "$slots" != auto ]; then
        set -- "$@" --expert-cache-slots "$slots"
    fi
    printf '\n== %s GiB planned RAM ==\n' "$tier"
    printf 'command:'
    printf ' %s' "$exe" "$@"
    printf '\nlog: %s\ntime report: %s\n' "$log" "$timelog"
    # Preserve complete engine diagnostics and measure process peak RSS outside
    # Kraken: its built-in working-set counter is Windows-only. A failed run is
    # not a benchmark point and must not be disguised as zero throughput.
    "$python" "$rss_helper" -- "$log" "$timelog" "$exe" "$@" || {
        rc=$?
        cat "$log" "$timelog"
        echo "bounded-moe: run failed (exit $rc), see $log and $timelog" >&2
        exit "$rc"
    }
    grep -E '^(model|device|prefill|decode|total|experts|expert WARM|\[stats \] expert-pool|\[stats \] experts|\[stats \] cache|\[stats \] warm|\[stats \] transfer)' "$log" || true
    grep -E '^Maximum resident set size:' "$timelog" || true
    echo "full run log: $log"
    echo "peak RSS report: $timelog"

    # The host-cache policy must not change model output. Run greedy generation
    # on the same fixed prompt, save stdout separately from stats on stderr, and
    # compare every RAM tier against the first tier's exact token/text output.
    set -- --cpu -m "$model" --ram-tier "$tier" \
        --expert-warmup 0 --expert-warm-mb 0 --expert-pin-mb 0 \
        --greedy --max-tokens "$tokens" --prompt "$bench_prompt"
    if [ "$slots" != auto ]; then
        set -- "$@" --expert-cache-slots "$slots"
    fi
    if ! "$exe" "$@" >"$tokenlog" 2>"$genlog"; then
        cat "$genlog" >&2
        echo "bounded-moe: parity generation failed at ${tier} GiB; see $genlog" >&2
        exit 1
    fi
    if [ -z "$reference_tokens" ]; then
        reference_tokens="$outdir/${model_name}.reference.tokens.txt"
        reference_tier=$tier
        cp "$tokenlog" "$reference_tokens"
        echo "output parity: baseline recorded at ${tier} GiB"
    elif ! cmp -s "$reference_tokens" "$tokenlog"; then
        echo "bounded-moe: output mismatch between ${reference_tier} GiB and ${tier} GiB" >&2
        diff -u "$reference_tokens" "$tokenlog" || true
        echo "mismatching output: $tokenlog; diagnostics: $genlog" >&2
        exit 1
    else
        echo "output parity: identical to ${reference_tier} GiB baseline"
    fi
    echo "generation output: $tokenlog"
    echo "generation diagnostics: $genlog"
done

echo "Runs complete. Require exact output parity and compare peak RSS, resident/active expert slots, cache hit rate, cold reads, and decode rate together."
echo "Do not interpret a simulated --ram-tier run as proof that the process fits in that physical RAM class."

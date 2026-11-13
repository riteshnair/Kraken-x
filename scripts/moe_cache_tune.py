#!/usr/bin/env python3
"""Pick expert-cache settings from a recorded routing trace.

Replays one `KRK_TRACE_EXPERTS=1` trace through `kraken-expert-replay`, which
drives the engine's own ExpertCache (include/krk/expert_replay.hpp), and turns
its counters into a per-token prediction. The point is to answer "cache size,
pin budget, host share" without starting the model once per arm: a trace of a
few hundred tokens is enough to compare a dozen configurations, and the answer
describes the policy that is in the tree rather than a model of it.

The time model is deliberately three constants and no curve fitting:

    ms/token = base_ms
             + read_bytes  / read_mib_s
             + promoted_bytes / promote_mib_s
             + host_takes_per_token * host_ms_per_expert

`base_ms`, `read_mib_s`, `promote_mib_s` are measured, not guessed, and the
defaults are this repository's recorded numbers on the dev box (NVMe reads at
~593 MiB/s at queue depth 1 and ~1446 MiB/s at four, a promotion at ~11 GiB/s,
~1.95 ms of host compute per expert). They are defaults for a reason: run
`--print-defaults`, then pass your own. A prediction built on another machine's
constants is another machine's answer.

The host arm overlaps the device stream, so its cost is added rather than maxed;
that makes the share look worse than it is on a machine where the copies finish
first. Treat shares within 5% of the best as tied -- which is exactly the rule
the recommendation uses.

Usage:
  python3 scripts/moe_cache_tune.py --trace expert_trace.tsv --expert-mib 1.95 \\
      --budget-mib 2048,4096,8192 --shares 0,0.5,1
  python3 scripts/moe_cache_tune.py --trace expert_trace.tsv ... --markdown
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

DEFAULTS = {
    # Recorded on the development host. See docs/OPTIONS-MATRIX.md.
    "read_mib_s": 1446.0,      # 1.82 MiB reads, queue depth 4
    "promote_mib_s": 11264.0,  # ~11 GiB/s host->VRAM promotion
    "host_ms_expert": 1.95,    # CPU expert-arm cost per expert (see --hybrid-experts)
    "base_ms": 0.0,            # device time that the cache does not change
}


def find_replay(explicit):
    if explicit:
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for candidate in (
        os.path.join(root, "build-cpu", "kraken-expert-replay"),
        os.path.join(root, "build", "kraken-expert-replay"),
        os.path.join(root, "build-hip", "kraken-expert-replay"),
    ):
        if os.path.exists(candidate):
            return candidate
    found = shutil.which("kraken-expert-replay")
    if found:
        return found
    sys.exit("moe_cache_tune: kraken-expert-replay not found; build it or pass --replay")


def parse_list(text, cast):
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def replay(replay_bin, trace, budget_slots, shares, warm_slots, pin_slots,
           layers, experts, expert_kib, cache_dir):
    """Run one replay grid, reusing a previous identical run from --out."""
    key = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha256()
        with open(trace, "rb") as stream:
            digest.update(stream.read())
        digest.update(repr((budget_slots, shares, warm_slots, pin_slots,
                            layers, experts, expert_kib)).encode())
        key = os.path.join(cache_dir, digest.hexdigest()[:16] + ".json")
        if os.path.exists(key):
            with open(key, encoding="utf-8") as stream:
                return json.load(stream)

    cmd = [replay_bin, "--trace", trace, "--json",
           "--budget-slots", ",".join(str(s) for s in budget_slots),
           "--share", ",".join("%g" % (s / 1000.0) for s in shares),
           "--expert-kib", str(expert_kib)]
    if warm_slots:
        cmd += ["--warm-slots", str(warm_slots)]
    if pin_slots:
        cmd += ["--pin-slots", str(pin_slots)]
    if layers:
        cmd += ["--layers", str(layers)]
    if experts:
        cmd += ["--experts", str(experts)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit("moe_cache_tune: %s failed:\n%s" % (cmd[0], result.stderr.strip()))
    rows = json.loads(result.stdout)

    # The tool reports slots and bytes at its own synthetic footprint. Slots are
    # the policy's own unit, so they survive the conversion to a real file; the
    # bytes have to be rescaled by the caller, which knows the real footprint.
    if cache_dir:
        with open(key, "w", encoding="utf-8") as stream:
            json.dump(rows, stream)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True,
                    help="expert_trace.tsv from KRK_TRACE_EXPERTS=1")
    ap.add_argument("--expert-mib", type=float, default=1.95,
                    help="bytes per expert in the model, in MiB (default 1.95)")
    ap.add_argument("--budget-mib", default="",
                    help="device cache sizes to compare, comma separated")
    ap.add_argument("--shares", default="0,0.5,1",
                    help="host shares of the overflow (default 0,0.5,1)")
    ap.add_argument("--warm-mib", type=float, default=0.0,
                    help="WARM (pageable host) tier in MiB (default 0)")
    ap.add_argument("--pin-mib", type=float, default=0.0,
                    help="pin ring in MiB, outside the pool budget (default 0)")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--read-mib-s", type=float,
                    help="mapping read speed, MiB/s (default %.0f)" % DEFAULTS["read_mib_s"])
    ap.add_argument("--promote-mib-s", type=float,
                    help="host->VRAM promotion speed, MiB/s (default %.0f)"
                         % DEFAULTS["promote_mib_s"])
    ap.add_argument("--host-ms-expert", type=float,
                    help="host compute per expert, ms (default %.2f)"
                         % DEFAULTS["host_ms_expert"])
    ap.add_argument("--base-ms", type=float,
                    help="device time per token the cache does not change (default 0)")
    ap.add_argument("--within", type=float, default=0.05,
                    help="treat predictions within this fraction of the best as "
                         "tied and prefer the lower share (default 0.05)")
    ap.add_argument("--replay", help="path to kraken-expert-replay")
    ap.add_argument("--out", help="directory to cache replay results in")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table")
    ap.add_argument("--print-defaults", action="store_true",
                    help="print the default cost constants and exit")
    args = ap.parse_args()

    if args.print_defaults:
        for k, v in DEFAULTS.items():
            print("%s = %s" % (k, v))
        return 0

    read_mib_s = args.read_mib_s if args.read_mib_s is not None else DEFAULTS["read_mib_s"]
    promote_mib_s = (args.promote_mib_s if args.promote_mib_s is not None
                     else DEFAULTS["promote_mib_s"])
    host_ms = (args.host_ms_expert if args.host_ms_expert is not None
               else DEFAULTS["host_ms_expert"])
    base_ms = args.base_ms if args.base_ms is not None else DEFAULTS["base_ms"]

    replay_bin = find_replay(args.replay)
    expert_bytes = max(1, int(args.expert_mib * 1048576))

    budgets = parse_list(args.budget_mib, float)
    shares = [int(round(s * 1000)) for s in parse_list(args.shares, float)]
    if not budgets:
        sys.exit("moe_cache_tune: --budget-mib is required "
                 "(the replay cannot know what fits on your card)")
    budget_slots = [max(1, int(b * 1048576 // expert_bytes)) for b in budgets]
    warm_slots = int(args.warm_mib * 1048576 // expert_bytes) if args.warm_mib else 0
    pin_slots = int(args.pin_mib * 1048576 // expert_bytes) if args.pin_mib else 0

    # The replay's synthetic expert is small on purpose: the policy sees a
    # ratio, so a 16 KiB stand-in costs the tool a memcpy instead of a 2 MiB one
    # and predicts the same hits. Byte counts are rescaled below.
    expert_kib = 16
    rows = replay(replay_bin, args.trace, budget_slots, shares, warm_slots,
                  pin_slots, args.layers, args.experts, expert_kib, args.out)
    scale = expert_bytes / (expert_kib * 1024)

    scored = []
    for row in rows:
        steps = max(1, row["steps"])
        read_mib = row["read_bytes"] * scale / 1048576.0
        promoted_mib = row["promoted_bytes"] * scale / 1048576.0
        host_per_step = row["host_takes"] / steps
        ms = (base_ms +
              read_mib / read_mib_s * 1000.0 +
              promoted_mib / promote_mib_s * 1000.0 +
              host_per_step * host_ms)
        scored.append({
            "budget_slots": row["budget_slots"],
            "budget_mib": row["budget_slots"] * expert_bytes / 1048576.0,
            "share": row["share"],
            "hits": row["hits"],
            "promotions": row["promotions"],
            "loads": row["loads"],
            "host_takes": row["host_takes"],
            "resident_slots": row["resident_slots"],
            "pinned_slots": row["pinned_slots"],
            "read_mib": read_mib,
            "promoted_mib": promoted_mib,
            "ms_per_step": ms,
            "tok_s": 1000.0 / ms if ms > 0 else 0.0,
        })

    best = min(s["ms_per_step"] for s in scored)
    # Lowest share within --within of the best: the host arm is the one that
    # changes the answer, the cache size the one that changes the memory bill,
    # so a tie goes to the arm that leaves the card holding the weights.
    tied = [s for s in scored if s["ms_per_step"] <= best * (1.0 + args.within)]
    pick = min(tied, key=lambda s: (s["share"], -s["budget_slots"]))

    if args.markdown:
        print("| budget (MiB) | share | hits | promotions | reads | host takes "
              "| ms/token | tok/s |")
        print("|---|---|---|---|---|---|---|---|")
        for s in scored:
            print("| %.0f | %.2f | %d | %d | %d | %d | %.2f | %.1f |" % (
                s["budget_mib"], s["share"], s["hits"], s["promotions"],
                s["loads"], s["host_takes"], s["ms_per_step"], s["tok_s"]))
    else:
        print("budget_mib  share  hits  promotions  reads  host  ms/token  tok/s")
        for s in scored:
            print("%10.0f  %5.2f  %6d  %10d  %5d  %4d  %8.2f  %6.1f" % (
                s["budget_mib"], s["share"], s["hits"], s["promotions"],
                s["loads"], s["host_takes"], s["ms_per_step"], s["tok_s"]))

    print()
    print("recommended: --expert-cache-mb %.0f --hybrid-frac %.2f%s%s" % (
        pick["budget_mib"], pick["share"],
        " --expert-warm-mb %.0f" % args.warm_mib if args.warm_mib else "",
        " --expert-pin-mb %.0f" % args.pin_mib if args.pin_mib else ""))
    print("predicted: %.2f ms/token (%.1f tok/s); best was %.2f ms/token"
          % (pick["ms_per_step"], pick["tok_s"], best))
    print("note: the share arm is additive here, so a machine where the host "
          "finishes before the copies will do better than this says.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

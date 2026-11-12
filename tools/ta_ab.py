#!/usr/bin/env python3
"""Interleaved, order-balanced A/B harness for the CPU backend.

A sequential probe on this box reads the warm-up as the arm: the first run
after a load pays page-cache and branch-predictor state the later ones do not,
and a single run is +-6% noise. So every comparison here runs PAIRS, and the
order inside a pair alternates, which makes the slot a run occupies a property
of the experiment rather than of the arm.

Reports, per arm, the median of each metric over all runs; and per pair, the
delta, so the win count and the MAD can be stated. A verdict needs the
order-balanced median paired gain to clear the pairs' own MAD in at least
reps-1 pairs, which is the rule the rest of this repo's harnesses use.

  python tools/ta_ab.py --reps 5 --metric decode --base "--threads 8" --arm "--threads 14"

Arms are extra argument strings appended to one common command line. Each is
split with shlex, so `--threads 8` and `KRK_X=1` both work (the latter is
placed before the binary as an environment assignment).
"""
import argparse
import os
import re
import shlex
import statistics
import subprocess
import sys

DEF_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "build-q4k", "kraken.exe")
DEF_MODEL = r"C:\models\Qwen3.8-4B-Q4_K_M.gguf"
DEF_PROMPT = ("The history of computing is a history of abstraction, from relays "
              "to vacuum tubes and")


def parse(text):
    r = {}
    m = re.search(r"\[stats \]\s*prefill\s+(\d+)\s+tok\s+in\s+([\d.]+)\s*ms\s*=?\s*([\d.]+)?",
                  text)
    if m:
        r["prefill_tok"] = int(m.group(1))
        r["prefill_ms"] = float(m.group(2))
        if m.group(3):
            r["prefill_tps"] = float(m.group(3))
    m = re.search(r"\[stats \]\s*decode\s+(\d+)\s+tok\s+in\s+([\d.]+)\s*ms.*?"
                  r"=\s*([\d.]+)\s*tok/s", text)
    if m:
        r["decode_tok"] = int(m.group(1))
        r["decode_ms"] = float(m.group(2))
        r["decode_tps"] = float(m.group(3))
    # generated text: everything after the last [stats ]/[info ]/[warn ] line,
    # with the banner dropped -- the banner names the participant width, so an
    # A/B on --threads would otherwise always report "text DIFFERS" while the
    # generated tokens were identical.
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not re.match(r"^\s*(\[stats|\[info|\[warn|\[error)", ln)
             and not ln.startswith("KRAKEN ")]
    r["text"] = "\n".join(lines).strip()
    return r


def split_arm(argstr):
    """Return (env assignments, argv) for an arm string."""
    env, argv = {}, []
    for tok in shlex.split(argstr or ""):
        if "=" in tok and not tok.startswith("-") and tok.split("=")[0].isupper():
            k, v = tok.split("=", 1)
            env[k] = v
        else:
            argv.append(tok)
    return env, argv


def run_one(binary, base_env, base_argv, arm_env, arm_argv, timeout):
    env = dict(os.environ)
    env.update(base_env)
    env.update(arm_env)
    cmd = [binary] + base_argv + arm_argv
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        raise SystemExit("rc=%d from %s" % (p.returncode, " ".join(cmd)))
    out = p.stdout + "\n" + p.stderr
    rec = parse(out)
    rec["rc"] = p.returncode
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=DEF_BIN)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--prompt", default=DEF_PROMPT)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--extra", default="")
    ap.add_argument("--base", default="", help="baseline arm (extra args)")
    ap.add_argument("--arm", default="", help="treatment arm (extra args)")
    ap.add_argument("--metric", default="decode", choices=["decode", "prefill", "both"])
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--no-text-check", action="store_true")
    a = ap.parse_args()

    base_argv = ["-m", a.model, "-p", a.prompt, "-n", str(a.n), "--greedy",
                 "--temp", "0", "--ctx", str(a.ctx)] + shlex.split(a.extra or "")
    be, bv = split_arm(a.base)
    ae, av = split_arm(a.arm)

    key = "decode_tps" if a.metric == "decode" else "prefill_tps"
    ms_key = "decode_ms" if a.metric == "decode" else "prefill_ms"

    A, B = [], []
    texts = {"base": set(), "arm": set()}
    for i in range(a.reps):
        order = [("base", be, bv), ("arm", ae, av)]
        if i % 2 == 1:
            order.reverse()
        for name, env, argv in order:
            rec = run_one(a.bin, {}, base_argv, env, argv, a.timeout)
            rec["arm"] = name
            texts[name].add(rec.get("text", ""))
            (A if name == "base" else B).append(rec)
            print("  rep%d %-4s prefill %8.1f ms  decode %6.2f tok/s"
                  % (i, name, rec.get("prefill_ms", 0), rec.get("decode_tps", 0)))

    def med(rs, k):
        vals = [r[k] for r in rs if k in r]
        return statistics.median(vals) if vals else float("nan")

    print()
    for label, rs in (("base", A), ("arm", B)):
        print("%-4s n=%d  prefill %8.1f ms (%6.2f tok/s)  decode %6.2f tok/s (%7.0f ms total)"
              % (label, len(rs), med(rs, "prefill_ms"), med(rs, "prefill_tps"),
                 med(rs, "decode_tps"), med(rs, "decode_ms")))

    # Paired deltas: index i of each list is the same rep and the opposite slot.
    keys = [("prefill_ms", "decode_tps")] if a.metric == "prefill" else \
           ([("decode_ms", "decode_tps")] if a.metric == "decode" else
            [("prefill_ms", "prefill_tps"), ("decode_ms", "decode_tps")])
    for ms_k, k in keys:
        dels = []
        for x, y in zip(A, B):
            if x.get(k) and y.get(k):
                dels.append(100.0 * (y[k] - x[k]) / x[k])
        if not dels:
            continue
        # MAD as the median absolute DEVIATION of the paired deltas around
        # their own median. median(|d|) is the wrong statistic here: a fold that
        # wins every pair by the same 8% has median(|d|) = 8% and would never
        # clear its own "noise", which reports a consistent win as a null.
        med = statistics.median(dels)
        mad = statistics.median([abs(d - med) for d in dels])
        wins = sum(1 for d in dels if d > 0)
        print("\nper-pair %s delta: %s" % (k, " ".join("%+.2f%%" % d for d in dels)))
        print("  median %+.3f%%  MAD %.3f%%  arm faster %d/%d"
              % (med, mad, wins, len(dels)))
        verdict = "WIN" if (med > mad and wins >= len(dels) - 1) else \
                  ("LOSS" if (med < -mad and wins <= 1) else "NO DIFFERENCE")
        print("  verdict: %s" % verdict)
    if not a.no_text_check:
        for k in ("base", "arm"):
            if len(texts[k]) > 1:
                print("!! %s produced %d different texts across reps" % (k, len(texts[k])))
        same = texts["base"] == texts["arm"]
        print("text: %s" % ("identical" if same else "DIFFERS"))
        if not same:
            for k, v in texts.items():
                for t in v:
                    print("  [%s] %r" % (k, t[:160]))


if __name__ == "__main__":
    main()

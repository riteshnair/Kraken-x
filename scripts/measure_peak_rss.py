#!/usr/bin/env python3
"""Run one child, capture its output, and record its peak RSS via wait4 usage.

Unix-only standard-library helper for bench_bounded_moe.sh. Linux reports
ru_maxrss in KiB; macOS reports bytes. Other platforms are rejected rather than
silently labeling a unit incorrectly.
"""

from __future__ import annotations

import resource
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 5 or sys.argv[1] != "--":
        print(
            f"usage: {sys.argv[0]} -- LOG_FILE RSS_REPORT COMMAND [ARG ...]",
            file=sys.stderr,
        )
        return 2
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        print("measure_peak_rss: supported only on Linux and macOS", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[2])
    report_path = Path(sys.argv[3])
    command = sys.argv[4:]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        try:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        except OSError as exc:
            print(f"measure_peak_rss: cannot start {command[0]}: {exc}", file=sys.stderr)
            return 127
        try:
            status = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            status = 130

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if sys.platform.startswith("linux"):
        peak = int(usage.ru_maxrss)
        unit = "KiB"
    else:
        peak = int(usage.ru_maxrss / 1024)
        unit = "KiB"
    report_path.write_text(f"Maximum resident set size: {peak} {unit}\n")
    return status if status >= 0 else min(255, 128 - status)


if __name__ == "__main__":
    raise SystemExit(main())

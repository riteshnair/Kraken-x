#!/usr/bin/env python3
"""dispatch_audit — which int8 dot kernels have a VNNI rung, and which tier
this machine actually takes.

Why it exists: a kernel can have an AVX-512 VNNI tier and no 256-bit
AVX-VNNI one, in which case a CPU that has avxvnni but not avx512vnni quietly
runs the AVX2 path -- correct, slower, and invisible unless you read every
dispatch ladder by hand. That is exactly what happened to
dot_q8_0_q8_x8_tier on a Meteor Lake box.

The ladder a dispatch entry is expected to have, best first:
    cpu_has_vnni()      -> AVX-512 VNNI   (512-bit)
    cpu_has_avxvnni()   -> AVX-VNNI       (256-bit, VEX vpdpbusd)
    fallback return     -> AVX2 / scalar
An entry whose first test is cpu_has_vnni() and which has no
cpu_has_avxvnni() branch is reported GAP when the host has avxvnni.
"""
import re
import sys
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "quant.cpp")


def host_has(feature):
    """Ask the compiler's own builtin through a one-line program."""
    import subprocess
    import tempfile
    cxx = os.environ.get("CXX", "g++")
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "p.cpp")
        e = os.path.join(d, "p.exe" if os.name == "nt" else "p")
        open(c, "w").write(
            '#include <cstdio>\nint main(){std::printf("%%d",'
            '__builtin_cpu_supports("%s")!=0);return 0;}\n' % feature)
        try:
            subprocess.run([cxx, "-O1", "-o", e, c], check=True,
                           capture_output=True)
            out = subprocess.run([e], capture_output=True, text=True)
            return out.stdout.strip() == "1"
        except Exception:
            return None


def main():
    text = open(SRC, encoding="utf-8", errors="replace").read()
    lines = text.split("\n")

    # Every dispatch entry: a public (non-static, non-attr) definition whose body
    # references a cpu_has_* tier test.
    entries = []
    for i, ln in enumerate(lines):
        m = re.match(r"^(?:f32|void)\s+(\w+)\((.*)$", ln)
        if not m:
            continue
        name = m.group(1)
        # collect the body up to a line that is exactly "}"
        body = []
        j = i
        while j < len(lines) and j < i + 80:
            body.append(lines[j])
            if lines[j] == "}":
                break
            j += 1
        b = "\n".join(body)
        if "cpu_has_vnni()" not in b and "cpu_has_avxvnni()" not in b:
            continue
        entries.append((i + 1, name, b))

    has512 = host_has("avx512vnni")
    has256 = host_has("avxvnni")
    print("host: avx512vnni=%s  avxvnni=%s" % (has512, has256))
    print()

    gaps, ok = [], []
    for line_no, name, body in entries:
        tiers = []
        for ln in body.split("\n"):
            if "cpu_has_vnni()" in ln:
                tiers.append("avx512vnni")
            elif "cpu_has_avxvnni()" in ln:
                tiers.append("avxvnni")
        taken = "avx2/scalar"
        if has512 and "avx512vnni" in tiers:
            taken = "avx512vnni"
        elif has256 and "avxvnni" in tiers:
            taken = "avxvnni"
        row = (line_no, name, tiers, taken)
        if has256 and not has512 and "avxvnni" not in tiers and "avx512vnni" in tiers:
            gaps.append(row)
        else:
            ok.append(row)

    print("%-42s %-6s %-28s %s" % ("dispatch entry", "line", "tiers present", "tier this host takes"))
    print("-" * 108)
    for line_no, name, tiers, taken in ok:
        print("%-42s %-6d %-28s %s" % (name, line_no, ",".join(tiers) or "-", taken))
    if gaps:
        print()
        print("GAP -- has an AVX-512 VNNI tier and no 256-bit AVX-VNNI rung, so this")
        print("host falls back to AVX2 even though it has vpdpbusd:")
        for line_no, name, tiers, taken in gaps:
            print("  %-40s line %-5d tiers: %-24s -> %s" % (name, line_no, ",".join(tiers), taken))
    print()
    print("%d entries, %d with a gap for this host" % (len(entries), len(gaps)))
    return 1 if gaps else 0


if __name__ == "__main__":
    sys.exit(main())

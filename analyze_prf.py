#!/usr/bin/env python3
"""Usage: ./prf > prf.csv ; python3 analyze_prf.py prf.csv"""
import csv, sys, statistics as st
from collections import defaultdict

data = defaultdict(list)
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if r["kind"] == "test":
            data[(r["class"], int(r["instrs"]))].append(float(r["ticks_per_iteration"]))

def curve(cls):
    pts = sorted((n, st.median(v)) for (c, n), v in data.items() if c == cls)
    return pts

def knee(pts):
    """First instruction count where the curve crosses the midpoint between
    the low plateau (first 3 points) and high plateau (last 3 points),
    linearly interpolated."""
    lo = st.median(y for _, y in pts[:3])
    hi = st.median(y for _, y in pts[-3:])
    if hi < lo * 1.3:
        return None, lo, hi
    mid = (lo + hi) / 2
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 < mid <= y1:
            return x0 + (mid - y0) / (y1 - y0) * (x1 - x0), lo, hi
    return None, lo, hi

res = {}
for cls in ("nop", "gpr", "simd", "mix"):
    k, lo, hi = knee(curve(cls))
    res[cls] = k
    print(f"{cls:5s} knee = {k if k is None else round(k):>5}   "
          f"(low plateau {lo:.0f}, high plateau {hi:.0f} ticks/iter)")

rob, gpr, simd, mix = (res[c] for c in ("nop", "gpr", "simd", "mix"))
print()
if rob is None:
    print("No ROB knee found: extend the sweep range in TEST_COUNTS.")
    sys.exit()
print(f"ROB entries ~ {rob:.0f}")
for name, k, arch in (("integer", gpr, 16), ("vector/FP", simd, 16)):
    if k is None:
        print(f"{name}: no knee in range -> not limiting below the ROB "
              f"(or extend the sweep)")
    elif k < rob * 0.9:
        print(f"{name}: PRF-limited. Free regs ~ {k:.0f}; "
              f"PRF ~ {k + arch:.0f} (adds ~{arch} arch regs; "
              f"use 32 for the vector file on AVX-512 parts)")
    else:
        print(f"{name}: knee ~ ROB knee -> ROB hits first; PRF >= ~{k + arch:.0f}")

if gpr and simd and mix:
    small = min(gpr, simd)
    if abs(mix - small) < 0.15 * small:
        print("mix knee ~ single-class knee -> GPR and SIMD share ONE physical file")
    elif mix > small * 1.5:
        print("mix knee ~ 2x single-class knee -> SEPARATE int and vector files")
    else:
        print("mix knee is ambiguous -> compare curves manually")

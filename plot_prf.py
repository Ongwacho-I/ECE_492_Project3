#!/usr/bin/env python3
"""Usage: python3 plot_prf.py prf_native_results.csv [out.png]"""
import csv, sys, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")          # headless: works over ssh
import matplotlib.pyplot as plt

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "prf_plot.png"

data = defaultdict(list)       # (class, kind, instrs) -> [ticks per trial]
with open(src) as f:
    for r in csv.DictReader(f):
        data[(r["class"], r["kind"], int(r["instrs"]))].append(
            float(r["ticks_per_iteration"]))

def curve(cls, kind):
    xs = sorted({n for (c, k, n) in data if c == cls and k == kind})
    med = [st.median(data[(cls, kind, x)]) for x in xs]
    lo = [min(data[(cls, kind, x)]) for x in xs]
    hi = [max(data[(cls, kind, x)]) for x in xs]
    return xs, med, lo, hi

def knee(xs, ys):
    """Midpoint crossing between low and high plateau, interpolated."""
    lo = st.median(ys[:3]); hi = st.median(ys[-3:])
    if hi < lo * 1.3:
        return None
    mid = (lo + hi) / 2
    for i in range(len(xs) - 1):
        if ys[i] < mid <= ys[i + 1]:
            return xs[i] + (mid - ys[i]) / (ys[i + 1] - ys[i]) * (xs[i + 1] - xs[i])
    return None

style = {"nop": ("tab:gray", "NOP (ROB reference)"),
         "gpr": ("tab:blue", "Integer (GPR) adds"),
         "simd": ("tab:red", "Vector (xmm) paddd"),
         "mix": ("tab:green", "Mixed GPR + SIMD")}

fig, (ax, axc) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

for cls, (color, label) in style.items():
    xs, med, lo, hi = curve(cls, "test")
    ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax.plot(xs, med, "o-", color=color, ms=3, label=label)
    k = knee(xs, med)
    if k is not None:
        ax.axvline(k, color=color, ls="--", alpha=0.7)
        ax.annotate(f"{k:.0f}", (k, ax.get_ylim()[1]), color=color,
                    xytext=(3, -14), textcoords="offset points", fontsize=9)
        print(f"{cls:5s} knee ~ {k:.0f} instructions")
    else:
        print(f"{cls:5s} no knee found in range")

    xc, medc, _, _ = curve(cls, "control")
    axc.plot(xc, medc, "-", color=color, lw=1)

ax.set_ylabel("ticks per iteration (median, min-max band)")
ax.set_title("Physical register file probe: time vs. in-flight filler instructions")
ax.legend()
ax.grid(alpha=0.3)
axc.set_ylabel("control\n(no loads)")
axc.set_xlabel("filler instructions between the two loads")
axc.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"saved {out}")

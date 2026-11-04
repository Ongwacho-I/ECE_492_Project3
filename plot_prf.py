#!/usr/bin/env python3
"""Usage: python3 plot_prf.py prf_results.csv [out.png]   (needs prf_common.py)"""
import sys
import matplotlib
matplotlib.use("Agg")          # headless: works over ssh
import matplotlib.pyplot as plt
from prf_common import load, median_curve, knee_with_ci, CLASSES

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "prf_plot.png"
data = load(src)

style = {"nop": ("tab:gray", "NOP (ROB reference)"),
         "gpr": ("tab:blue", "Integer (GPR) adds"),
         "simd": ("tab:red", "Vector (xmm) paddd"),
         "mix": ("tab:green", "Mixed GPR + SIMD")}

fig, (ax, axz, axc) = plt.subplots(
    3, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 3, 1]})

knees = {}
for i, cls in enumerate(CLASSES):
    color, label = style[cls]
    xs, med = median_curve(data, cls)
    lo = [min(data[(cls, "test", x)]) for x in xs]
    hi = [max(data[(cls, "test", x)]) for x in xs]
    k, jump, klo, khi = knee_with_ci(data, cls)
    knees[cls] = k
    print(f"{cls:5s} knee {k:4d}  CI [{klo},{khi}]  jump {jump:.1f}")

    ax.fill_between(xs, lo, hi, color=color, alpha=0.10)
    ax.plot(xs, med, "o-", color=color, ms=3, label=f"{label} (knee {k})")
    ax.axvline(k, color=color, ls="--", alpha=0.7)

    axz.plot(xs, med, "o-", color=color, ms=4, label=label)
    axz.axvline(k, color=color, ls="--", alpha=0.7)
    axz.axvspan(klo, khi, color=color, alpha=0.12)

    xc, medc = median_curve(data, cls, "control")
    axc.plot(xc, medc, "-", color=color, lw=1)

ax.set_title("Physical register file probe: time vs. in-flight filler instructions")
ax.set_ylabel("ticks / iteration (median, min-max band)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

kmin = min(knees.values()); kmax = max(knees.values())
axz.set_xlim(max(0, kmin - 48), kmax + 48)
axz.set_title("Zoom around the knees (shaded = 5-95% bootstrap interval)")
axz.set_ylabel("ticks / iteration (median)")
axz.grid(alpha=0.3)

axc.set_ylabel("control\n(no loads)")
axc.set_xlabel("filler instructions between the two loads")
axc.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"saved {out}")

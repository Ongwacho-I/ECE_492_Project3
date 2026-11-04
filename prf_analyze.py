#!/usr/bin/env python3
"""Self-contained PRF analysis + plot (no other files needed).

Usage: python3 prf_analyze.py prf_native_results_2.csv [prf_plot.png]

Prints knee positions (with 5-95% bootstrap intervals) and an
interpretation, then saves the plot. Needs matplotlib for the PNG; if it
is not installed the analysis still prints and the plot is skipped.
"""
import csv, random, sys, statistics as st
from collections import defaultdict

CLASSES = ("nop", "gpr", "simd", "mix")


# ---------------- loading / curves ----------------

def load(path):
    """-> {(cls, kind, instrs): [ticks ordered by trial]}"""
    tmp = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            tmp[(r["class"], r["kind"], int(r["instrs"]))][int(r["trial"])] = \
                float(r["ticks_per_iteration"])
    return {k: [v[t] for t in sorted(v)] for k, v in tmp.items()}


def xs_for(data, cls, kind="test"):
    return sorted({n for (c, k, n) in data if c == cls and k == kind})


def median_curve(data, cls, kind="test", idx=None):
    xs = xs_for(data, cls, kind)
    ys = []
    for x in xs:
        v = data[(cls, kind, x)]
        ys.append(st.median(v if idx is None else [v[i] for i in idx]))
    return xs, ys


# ---------------- knee detection ----------------

def _linfit(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx if sxx else 0.0
    a0 = my - b * mx
    sse = sum((c - (a0 + b * a)) ** 2 for a, c in zip(x, y))
    return a0, b, sse


def changepoint(xs, ys, min_pts=4):
    """Best single level shift: two straight-line segments, each with its
    own slope, minimising total squared error. Returns (knee_x, jump)."""
    best = None
    for s in range(min_pts, len(xs) - min_pts + 1):
        a1, b1, e1 = _linfit(xs[:s], ys[:s])
        a2, b2, e2 = _linfit(xs[s:], ys[s:])
        if best is None or e1 + e2 < best[0]:
            jump = (a2 + b2 * xs[s]) - (a1 + b1 * xs[s])
            best = (e1 + e2, xs[s], jump)
    return best[1], best[2]


def knee_with_ci(data, cls, boots=300, seed=1):
    xs, ys = median_curve(data, cls)
    knee, jump = changepoint(xs, ys)
    ntr = len(data[(cls, "test", xs[0])])
    rng = random.Random(seed)
    ks = []
    for _ in range(boots):
        idx = [rng.randrange(ntr) for _ in range(ntr)]
        bx, by = median_curve(data, cls, idx=idx)
        ks.append(changepoint(bx, by)[0])
    ks.sort()
    return knee, jump, ks[int(0.05 * boots)], ks[int(0.95 * boots) - 1]


# ---------------- main ----------------

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "prf_plot.png"
    data = load(src)

    res = {}
    print(f"{'class':5s} {'knee':>5s}  {'5-95% CI':>11s}  {'jump (ticks)':>12s}")
    for cls in CLASSES:
        k, jump, lo, hi = knee_with_ci(data, cls)
        res[cls] = (k, jump, lo, hi)
        print(f"{cls:5s} {k:5d}  [{lo:4d},{hi:4d}]  {jump:12.1f}")

    rob, gpr, simd, mix = (res[c][0] for c in CLASSES)
    print(f"\nROB entries ~ {rob}  (NOP knee; fillers take a ROB slot but no register)")
    for name, cls, arch in (("integer", "gpr", 16), ("vector/FP", "simd", 16)):
        k, jump, lo, hi = res[cls]
        if k < rob - 8 and hi < rob:
            print(f"{name}: breaks BEFORE the ROB -> PRF-limited. "
                  f"Free regs ~ {k} (CI {lo}-{hi}); PRF ~ {k + arch} "
                  f"(+{arch} arch regs; use 32 for vector on AVX-512 parts)")
        else:
            print(f"{name}: knee {k} is at/near the ROB knee -> ROB fills first; "
                  f"PRF is at least ~{k + arch}, exact size not resolved")

    small = min(gpr, simd)
    print()
    if mix <= small + 8:
        print("mix breaks with the smaller single-class knee -> GPR and SIMD "
              "appear to SHARE one physical file")
    elif mix >= min(rob, 2 * small) - 16:
        print("mix breaks near min(ROB, 2x smaller knee) -> SEPARATE integer "
              "and vector files")
    else:
        print("mix knee is between the two cases -> inspect the plot")
    print("\nNote: jumps under ~15 ticks mean a weak step; the knee is "
          "unreliable, so check the plot.")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless: works over ssh
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed: skipping the plot. "
              "Try: pip install --user matplotlib")
        return

    style = {"nop": ("tab:gray", "NOP (ROB reference)"),
             "gpr": ("tab:blue", "Integer (GPR) adds"),
             "simd": ("tab:red", "Vector (xmm) paddd"),
             "mix": ("tab:green", "Mixed GPR + SIMD")}

    fig, (ax, axz, axc) = plt.subplots(
        3, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 3, 1]})

    for cls in CLASSES:
        color, label = style[cls]
        xs, med = median_curve(data, cls)
        lo_b = [min(data[(cls, "test", x)]) for x in xs]
        hi_b = [max(data[(cls, "test", x)]) for x in xs]
        k, jump, klo, khi = res[cls]

        ax.fill_between(xs, lo_b, hi_b, color=color, alpha=0.10)
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

    kmin = min(r[0] for r in res.values())
    kmax = max(r[0] for r in res.values())
    axz.set_xlim(max(0, kmin - 48), kmax + 48)
    axz.set_title("Zoom around the knees (shaded = 5-95% bootstrap interval)")
    axz.set_ylabel("ticks / iteration (median)")
    axz.grid(alpha=0.3)

    axc.set_ylabel("control\n(no loads)")
    axc.set_xlabel("filler instructions between the two loads")
    axc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot TLB pointer-chasing benchmark CSV data.

Usage:
    python3 plot_tlb.py tlb_native_4k.csv
    python3 plot_tlb.py tlb_native_4k.csv --prefix tlb_native_4k --zoom 16 256
    python3 plot_tlb.py tlb_native_4k.csv --compare tlb_sgx_4k.csv --prefix tlb_compare

Input columns:
    page_bytes,pages,trial,ticks_per_access,ctrl_ticks_per_access

The shaded region is interquartile range (25th to 75th percentile).
Test minus control is a comparison, NOT a direct TLB miss latency estimate.
"""
import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLS = ["page_bytes", "pages", "trial", "ticks_per_access", "ctrl_ticks_per_access"]


def load_csv(path):
    # Robust against incidental progress messages in stdout. Preserve raw data.
    df = pd.read_csv(path, on_bad_lines="skip", dtype=str)
    missing = set(COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    before = len(df)
    for col in COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=COLS).copy()
    df = df[(df["page_bytes"].isin([4096, 2097152])) & (df["pages"] > 0) &
            (df["trial"] > 0) & (df["ticks_per_access"] >= 0) &
            (df["ctrl_ticks_per_access"] >= 0)]
    if df.empty:
        raise ValueError(f"{path}: no valid measurements")
    for c in ["page_bytes", "pages", "trial"]:
        df[c] = df[c].astype(int)
    if before != len(df):
        warnings.warn(f"{path}: skipped {before-len(df)} invalid row(s). Check raw CSV!")
    duplicates = df.duplicated(["page_bytes", "pages", "trial"]).sum()
    if duplicates:
        warnings.warn(f"{path}: found {duplicates} duplicate page/trial rows")
    df["difference"] = df["ticks_per_access"] - df["ctrl_ticks_per_access"]
    return df


def summary(df):
    quantities = ["ticks_per_access", "ctrl_ticks_per_access", "difference"]
    outputs = []
    for (size, pages), group in df.groupby(["page_bytes", "pages"]):
        item = {"page_bytes": size, "pages": pages, "n_trials": len(group)}
        for q in quantities:
            vals = group[q].to_numpy(dtype=float)
            item[f"{q}_median"] = float(np.median(vals))
            item[f"{q}_q25"] = float(np.percentile(vals, 25))
            item[f"{q}_q75"] = float(np.percentile(vals, 75))
        outputs.append(item)
    return pd.DataFrame(outputs).sort_values(["page_bytes", "pages"])


def line_band(ax, sub, col, label, color=None, linestyle="-"):
    x = sub["pages"].to_numpy(dtype=float)
    y = sub[f"{col}_median"].to_numpy(dtype=float)
    low = sub[f"{col}_q25"].to_numpy(dtype=float)
    high = sub[f"{col}_q75"].to_numpy(dtype=float)
    (line,) = ax.plot(x, y, linestyle=linestyle, lw=2, label=label, color=color)
    ax.fill_between(x, low, high, alpha=0.16, color=line.get_color(), linewidth=0)


def decorate(ax, title, logx=True):
    ax.set_title(title)
    ax.set_xlabel("Distinct virtual pages in pointer-chasing cycle")
    ax.set_ylabel("TSC ticks per dependent access")
    if logx:
        ax.set_xscale("log", base=2)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="native or SGX TLB result CSV")
    p.add_argument("--compare", help="optional second CSV (for example SGX)")
    p.add_argument("--prefix", help="output prefix (defaults to input file stem)")
    p.add_argument("--zoom", nargs=2, type=int, metavar=("MIN_PAGES", "MAX_PAGES"),
                   help="page-count range for detail plot (default 16 256)")
    args = p.parse_args()
    data = load_csv(args.csv)
    first = summary(data)
    second = summary(load_csv(args.compare)) if args.compare else None
    prefix = args.prefix or Path(args.csv).stem
    lo, hi = args.zoom or (16, 256)
    if not (0 < lo < hi):
        p.error("--zoom bounds must be positive and increasing")

    for size in sorted(first["page_bytes"].unique()):
        a = first[first["page_bytes"] == size]
        page_label = "4 KB" if size == 4096 else "2 MB requested (verify backing)"
        suffix = "4k" if size == 4096 else "2m"
        print(f"Page size: {page_label}, points: {len(a)}, trials/point: "
              f"{a.n_trials.min()}–{a.n_trials.max()}")
        fig, axs = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
        line_band(axs[0], a, "ticks_per_access", "Page-spaced pointer chase")
        line_band(axs[0], a, "ctrl_ticks_per_access", "Packed-lines control")
        decorate(axs[0], f"TLB-sensitive pointer chasing — {page_label}")
        line_band(axs[1], a, "difference", "Test − control (not isolated TLB cost)")
        decorate(axs[1], "Difference between workloads")
        fig.tight_layout()
        out = f"{prefix}_{suffix}_full.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out)

        zoom = a[(a["pages"] >= lo) & (a["pages"] <= hi)]
        if len(zoom) >= 2:
            fig, ax = plt.subplots(figsize=(10.5, 5.5))
            line_band(ax, zoom, "ticks_per_access", "Page-spaced pointer chase")
            line_band(ax, zoom, "ctrl_ticks_per_access", "Packed-lines control")
            decorate(ax, f"TLB-sensitive latency detail — {lo}–{hi} pages, {page_label}", logx=False)
            fig.tight_layout()
            out = f"{prefix}_{suffix}_zoom.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("Saved:", out)
        else:
            print("No zoom plot (fewer than two points in selected range).")

        if second is not None:
            b = second[second["page_bytes"] == size]
            if b.empty:
                print("No matching page size in comparison file; skipping comparison")
                continue
            fig, axs = plt.subplots(2, 1, figsize=(10.5, 8.5), sharex=True)
            line_band(axs[0], a, "ticks_per_access", "First run (native if supplied first)")
            line_band(axs[0], b, "ticks_per_access", "Second run (SGX if supplied second)")
            decorate(axs[0], f"Two-run test comparison — {page_label}")
            line_band(axs[1], a, "difference", "First: test − control")
            line_band(axs[1], b, "difference", "Second: test − control")
            decorate(axs[1], "Control-relative comparison (not isolated TLB cost)")
            fig.tight_layout()
            out = f"{prefix}_{suffix}_compare.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("Saved:", out)

    csv_out = f"{prefix}_summary.csv"
    first.to_csv(csv_out, index=False)
    print("Saved:", csv_out)
    print("Shading = interquartile range, not a confidence interval.")
    print("Timing transitions suggest possible translation resource limits but do not by themselves establish levels/capacities.")

if __name__ == "__main__":
    main()

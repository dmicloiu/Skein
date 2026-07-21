#!/usr/bin/env python3
"""[THE R KNOB: QUALITY x THROUGHPUT vs ROWS-PER-PROMPT]
Companion to plot_cross_system_frontier.py: the flock-only read of the same
cross-system artefacts. 2x2 small multiples -- columns filter (Q101) / extract
(Q103), top row F1 vs R, bottom row rows/s vs R (log2 x) -- showing WHY the
frontier bends: throughput saturates by R~8 while F1 keeps falling (filter),
and extract's quality cliff at R>=4 against a flat throughput curve. Faint
dashed line = LOTUS F1 (context, not a comparison curve).

House style per plot_sem_filter_ab.py; no run config baked into the image.
Points are medians over reps, whiskers rep min..max.

Usage:
  python analysis/plot_r_sweep.py \
      --filter-dir analysis/figures/data/cross_system_sem_filter \
      --extract-dir analysis/figures/data/cross_system_sem_extract \
      --out-dir analysis/figures
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_FLOCK = "#009E73"   # Okabe-Ito bluish green (flock, under test)
C_REF = "#0072B2"     # Okabe-Ito blue (LOTUS reference line)

SLIM_RS = [1, 2, 4, 8, 16, 32]
ROWS = 2000


def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "legend.fontsize": 11,
    })


def series(results_dir: Path, arm: str, system: str, qkey: str):
    """per-rep rows/s and F1 lists for one arm."""
    rps, f1s = [], []
    for p in sorted(results_dir.glob(f"{arm}_rep*_{system}.json")):
        try:
            q = json.loads(p.read_text()).get(qkey, {})
        except json.JSONDecodeError:
            continue
        if q.get("status") != "success":
            continue
        rps.append(ROWS / q["execution_time"])
        f1s.append(q["f1_score"])
    return rps, f1s


def sweep(results_dir: Path, qkey: str):
    out = []
    for r in SLIM_RS:
        rps, f1s = series(results_dir, f"flock_op_slim_r{r}", "flockmtl", qkey)
        if rps:
            out.append((r, rps, f1s))
    return out


def med_err(vals):
    m = statistics.median(vals)
    return m, [[m - min(vals)], [max(vals) - m]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-dir", type=Path,
                    default=Path("analysis/figures/data/cross_system_sem_filter"))
    ap.add_argument("--extract-dir", type=Path,
                    default=Path("analysis/figures/data/cross_system_sem_extract"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/figures"))
    args = ap.parse_args()
    for d in (args.filter_dir, args.extract_dir):
        if not d.exists():
            print(f"results dir not found: {d}", file=sys.stderr)
            return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()
    # sharey per row: filter and extract on identical scales, so the extract
    # collapse and its flat throughput read directly against the filter curves.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), sharex="col", sharey="row")

    panels = [("Semantic filter", args.filter_dir, "Q101", "lotus"),
              ("Semantic extract", args.extract_dir, "Q103", "lotus")]
    for col, (title, d, qkey, ref_arm) in enumerate(panels):
        data = sweep(d, qkey)
        rs = [r for r, _, _ in data]
        ax_f1, ax_tp = axes[0][col], axes[1][col]

        # --- top: F1 vs R ---
        meds = []
        for r, rps, f1s in data:
            m, err = med_err(f1s)
            meds.append(m)
            ax_f1.errorbar(r, m, yerr=err, fmt="o", color=C_FLOCK, ms=8,
                           lw=1.2, capsize=2, zorder=4)
        ax_f1.plot(rs, meds, "-", color=C_FLOCK, lw=2.0, alpha=0.7, zorder=3)
        _, ref_f1 = series(d, ref_arm, ref_arm, qkey)
        if ref_f1:
            ax_f1.axhline(statistics.median(ref_f1), color=C_REF, lw=1.4,
                          ls="--", alpha=0.8, zorder=2)
            ax_f1.annotate("LOTUS F1", (rs[-1], statistics.median(ref_f1)),
                           textcoords="offset points", xytext=(0, 5),
                           fontsize=10, color=C_REF, ha="right")
        ax_f1.set_title(title)
        if col == 0:
            ax_f1.set_ylabel("F1 vs ground truth")
        ax_f1.margins(y=0.12)

        # --- bottom: throughput vs R ---
        meds = []
        for r, rps, f1s in data:
            m, err = med_err(rps)
            meds.append(m)
            ax_tp.errorbar(r, m, yerr=err, fmt="o", color=C_FLOCK, ms=8,
                           lw=1.2, capsize=2, zorder=4)
        ax_tp.plot(rs, meds, "-", color=C_FLOCK, lw=2.0, alpha=0.7, zorder=3)
        if col == 0:
            ax_tp.set_ylabel("throughput (rows/s)")
        ax_tp.set_xlabel("rows per prompt (R)")
        ax_tp.set_xscale("log", base=2)
        ax_tp.set_xticks(SLIM_RS)
        ax_tp.set_xticklabels([str(r) for r in SLIM_RS])
        ax_tp.set_ylim(bottom=0)
        ax_tp.margins(y=0.12)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = args.out_dir / f"r_knob.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""[CROSS-SYSTEM THROUGHPUT x QUALITY FRONTIER]
The headline figure: rows/s x F1 for the flock semantic operator (batch-adaptive
slim prompt, R swept 1..32) against LOTUS and Palimpzest, one panel per task
(filter Q101 / extract Q103), from the cross-system artefacts produced by
sembench/slurm/cross_system_analysis_clariden.sh.

Points are medians over reps; whiskers span rep min..max on both axes (often
smaller than the marker -- greedy decoding is near-deterministic).

Inputs (per --results-dir, both dirs):
  <arm>_rep<r>_<system>.json    harness metrics + P/R/F1 per rep

Usage:
  python analysis/plot_cross_system_frontier.py \
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
import thesis_style as ts

C_FLOCK = "#009E73"   # Okabe-Ito bluish green: flock operator (under test)
C_LOTUS = "#0072B2"   # Okabe-Ito blue: LOTUS (baseline)
C_PZ = "#E69F00"      # Okabe-Ito orange: Palimpzest (baseline)

SLIM_RS = [1, 2, 4, 8, 16, 32]
ROWS = 2000  # sf_2000; throughput = rows / execution_time


def _setup_style():
    """House style, matching the other eval-chapter figures (thesis_style, base_font=14)."""
    ts.setup_style(base_font=14)


def arm_stats(results_dir: Path, arm: str, system: str, qkey: str):
    """median + (min, max) of rows/s and F1 over the arm's successful reps."""
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
    if not rps:
        return None
    med = (statistics.median(rps), statistics.median(f1s))
    return {"x": med[0], "y": med[1],
            "xerr": [[med[0] - min(rps)], [max(rps) - med[0]]],
            "yerr": [[med[1] - min(f1s)], [max(f1s) - med[1]]]}


def draw_panel(ax, results_dir: Path, qkey: str, title: str, winner_r: int,
               label_offsets: dict):
    # flock slim R-curve (connected in R order; annotate R at each point). The
    # recommended operating point is drawn as a star instead of a dot.
    xs, ys = [], []
    for r in SLIM_RS:
        s = arm_stats(results_dir, f"flock_op_slim_r{r}", "flockmtl", qkey)
        if not s:
            continue
        xs.append(s["x"]); ys.append(s["y"])
        marker, ms = ("*", 17) if r == winner_r else ("o", 8)
        ax.errorbar(s["x"], s["y"], xerr=s["xerr"], yerr=s["yerr"],
                    fmt=marker, color=C_FLOCK, ms=ms, lw=1.2, capsize=2,
                    markeredgecolor="white", markeredgewidth=0.6 if r == winner_r else 0,
                    zorder=5 if r == winner_r else 4)
        ax.annotate(f"R={r}", (s["x"], s["y"]), textcoords="offset points",
                    xytext=label_offsets.get(r, (0, 9)), fontsize=13,
                    color="0.25", ha="center")
    ax.plot(xs, ys, "-", color=C_FLOCK, lw=2.0, alpha=0.7, zorder=3,
            label="Skein (slim, R sweep)")

    # flock full-prompt R=1 (same system -> same hue, open diamond)
    s = arm_stats(results_dir, "flock_op_r1", "flockmtl", qkey)
    if s:
        ax.errorbar(s["x"], s["y"], xerr=s["xerr"], yerr=s["yerr"], fmt="D",
                    color=C_FLOCK, markerfacecolor="white", ms=9, lw=1.2,
                    capsize=2, zorder=4, label="Skein (full, R = 1)")

    for arm, system, color, label in [("lotus", "lotus", C_LOTUS, "LOTUS"),
                                      ("palimpzest", "palimpzest", C_PZ, "Palimpzest")]:
        s = arm_stats(results_dir, arm, system, qkey)
        if s:
            ax.errorbar(s["x"], s["y"], xerr=s["xerr"], yerr=s["yerr"], fmt="s",
                        color=color, ms=9, lw=1.2, capsize=2, zorder=4, label=label)
            ax.annotate(label, (s["x"], s["y"]), textcoords="offset points",
                        xytext=(0, -16), fontsize=13, color="0.25", ha="center")

    ax.set_title(title)
    ax.set_xlabel("throughput (rows/s)")
    ax.set_ylabel("F1 vs ground truth")
    ax.set_xlim(left=0)
    ax.margins(y=0.10)


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
    fig, (ax_f, ax_e) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    # winner (starred): R=8 on both operators -- the throughput operating point
    # fixed in subsec:eval-prompt, and the Pareto point in the data
    # (pareto_rows_f1/pareto_tok_f1 flag slim R=8 on the extract).
    # Per-R label offsets dodge collisions.
    draw_panel(ax_f, args.filter_dir, "Q101", "Semantic filter", winner_r=8,
               label_offsets={4: (-10, 10), 8: (14, 6), 16: (16, -3), 32: (-16, -4)})
    draw_panel(ax_e, args.extract_dir, "Q103", "Semantic extract", winner_r=8,
               label_offsets={1: (-14, 4), 2: (12, 8), 4: (2, 14), 8: (16, -8),
                              16: (18, 0), 32: (-18, -2)})
    ax_f.legend()
    fig.tight_layout()
    ts.no_suptitle(fig)
    for ext in ("png", "pdf"):
        path = args.out_dir / f"cross_system_frontier.{ext}"
        fig.savefig(path)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

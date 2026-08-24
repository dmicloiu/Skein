#!/usr/bin/env python3
"""[DESIGNED FOR THE MODEL FRONTIER QUALITY ANALYSIS]
Why the larger models score LOWER F1 on the movie benchmark: they are more
conservative on the thresholded predicate ("clearly positive"), and the gold
label is the critic score, not the text. Renders ONE paper figure:

  dp_model_quality  pass rate as a function of the review's normalised critic
                    score, against the score-derived gold label: the larger
                    models withhold "clearly positive" across the lukewarm
                    middle, a threshold shift rather than confusion.

Reads the operator verdict dumps (one per model; greedy decoding makes them
topology-invariant) plus the gold CSV directly -- sibling of
diagnose_rsweep.py, which established verdict-level analysis.

Usage:
  python analysis/plot_model_quality.py \
      [--summary analysis/figures/data/dp_scaling_json/dp_scaling_summary.csv] \
      [--dp-dir analysis/figures/data/dp_scaling_json] \
      [--frontier-dir analysis/figures/data/model_frontier_json] \
      [--data ../sembench/files/movie/data/sf_2000/Reviews.csv] \
      [--out-dir analysis/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

MODEL_ORDER = ["7B", "32B", "72B"]
COLOR_CONSENSUS = "#2e8b57"
COLOR_REJECTED = "#c8722a"


def _setup_style():
    """Delegate to the shared thesis_style house rcParams."""
    ts.setup_style(base_font=14)


def _model_short(m: str | None) -> str:
    mm = re.search(r"(\d+)B", m or "")
    return f"{mm.group(1)}B" if mm else "7B"


def load_verdicts(path: Path) -> dict[int, bool]:
    out: dict[int, bool] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[int(d["id"])] = (d.get("v") is True or d.get("v") is None)
    return out


def find_verdicts(dirs: list[Path]) -> dict[str, Path]:
    """First verdict dump per model; the meta sidecar names the model, legacy
    dumps without one are the pre-sidecar 7B runs."""
    out: dict[str, Path] = {}
    for d in dirs:
        for v in sorted(d.glob("verdicts_*.jsonl")):
            mm = re.search(r"_m(\d+)b_", v.name)
            model = f"{mm.group(1)}B" if mm else "7B"
            out.setdefault(model, v)
    return out


def norm_score(s: str) -> float | None:
    """Critic score as a 0..1 fraction; only x/y forms are unambiguous."""
    try:
        a, b = s.split("/")
        v = float(a) / float(b)
        return v if 0 < v <= 1 else None
    except (ValueError, ZeroDivisionError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=Path("analysis/figures/data/dp_scaling_json/dp_scaling_summary.csv"))
    ap.add_argument("--dp-dir", type=Path,
                    default=Path("analysis/figures/data/dp_scaling_json"))
    ap.add_argument("--frontier-dir", type=Path,
                    default=Path("analysis/figures/data/model_frontier_json"))
    ap.add_argument("--data", type=Path,
                    default=Path("../sembench/files/movie/data/sf_2000/Reviews.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/figures"))
    args = ap.parse_args()
    for p in (args.summary, args.data):
        if not p.exists():
            print(f"missing input: {p}", file=sys.stderr)
            return 1
    _setup_style()

    gold, scores = {}, {}
    with open(args.data, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            gold[i] = (row.get("scoreSentiment") or "").strip().upper() == "POSITIVE"
            scores[i] = norm_score(row.get("originalScore") or "")

    verdicts = {m: load_verdicts(p) for m, p in
                find_verdicts([d for d in (args.dp_dir, args.frontier_dir)
                               if d.exists()]).items()}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # where each model draws the line: per score bin, the share of
    # reviews the model calls "clearly positive" -- P(pass | score), the
    # direct form of the threshold-shift claim (a distribution of scores
    # given the verdict would be its inverse). Gold's own rate is the
    # near-step reference the models are graded against.
    MODEL_COLORS = {"7B": COLOR_CONSENSUS, "32B": COLOR_REJECTED,
                    "72B": "#8c564b"}
    bins = [x / 20 for x in range(6, 21)]
    scored = [i for i in scores if scores[i] is not None]
    if len(verdicts) >= 2 and scored:
        def rate(ids, passed):
            centers, rates, ns = [], [], []
            for lo, hi in zip(bins, bins[1:]):
                grp = [i for i in ids if lo <= scores[i] < hi or
                       (hi == 1.0 and scores[i] == 1.0)]
                if len(grp) >= 10:
                    centers.append((lo + hi) / 2)
                    rates.append(100.0 * sum(passed(i) for i in grp) / len(grp))
                    ns.append(len(grp))
            return centers, rates, ns

        cx, cy, _ = rate(scored, lambda i: gold[i])
        ax.plot(cx, cy, color="0.55", ls="--", lw=1.8, zorder=2,
                label="gold label (score-derived)")
        for m in [mm for mm in MODEL_ORDER if mm in verdicts]:
            v = verdicts[m]
            ids = [i for i in scored if i in v]
            cx, cy, _ = rate(ids, lambda i, v=v: v[i])
            ax.plot(cx, cy, marker="o", ms=6, lw=2.0, color=MODEL_COLORS[m],
                    zorder=3, label=m)
        ax.set_xlabel("critic score (normalized)")
        ax.set_ylabel('reviews passed as "clearly positive" (%)')
        ax.set_ylim(-3, 108)
        ax.set_title("Pass rate by critic score")
        ax.legend(loc="upper left", fontsize=9.5)
    else:
        ax.set_axis_off()
        print("  [warn] need verdict dumps + parseable scores -> panel skipped",
              file=sys.stderr)

    ts.no_suptitle(fig)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = args.out_dir / f"dp_model_quality.{ext}"
        fig.savefig(path)
        print(f"wrote {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

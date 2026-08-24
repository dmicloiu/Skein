#!/usr/bin/env python3
"""[GENERALIZATION: MODEL FAMILY] Qwen2.5-7B vs Llama-3.1-8B.

Figures for the cross-family section. These deliberately do NOT re-render the
Qwen figure set on Llama data: the section's claim is about transfer and
divergence, so every figure puts both families on one axis and lets the reader
see agreement or disagreement directly.

  gen_family_laws     the engine laws transfer: per-GPU efficiency of both
                      scale-out axes, and the fixed-budget topology ordering,
                      with the two families overlaid.
  gen_family_prompt   the prompt does NOT transfer: F1 vs rows-per-request for
                      full and slim heads, both families, against the
                      always-true baseline.
  gen_family_frontier the cross-system standing is family-dependent: rows/s x F1
                      per family per task.

Absolute throughput is NOT comparable across families (different tokenizer,
different parameter count), so the laws figure uses normalised quantities only
and the frontier keeps families in separate panels. F1 IS comparable: same gold
labels, same rows, same criterion.

Usage:
  python analysis/plot_generalization_family.py [--out-dir analysis/figures]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import thesis_style as ts

DATA = Path("analysis/figures/data")
QWEN, LLAMA = "Qwen2.5-7B", "Llama-3.1-8B"
COLORS = {QWEN: "#2e8b57", LLAMA: "#7048a8"}
MARKERS = {QWEN: "o", LLAMA: "s"}

# (family -> (dp summary, unified family, grid family))
DP_SRC = {
    QWEN: (DATA / "dp_scaling_json/dp_scaling_summary.csv", "unified", "grid"),
    LLAMA: (DATA / "dp_scaling_json_llama/dp_scaling_summary.csv",
            "llama_unified", "llama_grid"),
}
XS_SRC = {  # (filter summary, extract summary)
    QWEN: (DATA / "cross_system_sem_filter/cross_system_summary.csv",
           DATA / "cross_system_sem_extract/cross_system_summary.csv"),
    LLAMA: (DATA / "cross_system_sem_filter_llama/cross_system_summary.csv",
            DATA / "cross_system_sem_extract_llama/cross_system_summary.csv"),
}
ALWAYS_TRUE_F1 = 0.853  # 1487/2000 positives -> P=.744, R=1.0


def _setup_style():
    """Delegate to the shared thesis_style house rcParams."""
    ts.setup_style(base_font=14)


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts.no_suptitle(fig)  # house policy: no argumentative suptitle
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
        print(f"wrote {out_dir / f'{name}.{ext}'}", file=sys.stderr)


def dp_cells(path: Path, family: str, col: str) -> dict:
    """{(n_ep, tp, cap): median(col)} for the operator cells of one job family."""
    out: dict = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["arm"] != "op" or r.get("family") != family or not r.get(col):
                continue
            key = (int(r["n_ep"]), int(r["tp"]), int(r["cap"]))
            out.setdefault(key, []).append(float(r[col]))
    return {k: median(v) for k, v in out.items()}


def xs_rows(path: Path) -> dict:
    with open(path, newline="") as f:
        return {r["arm"]: r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# G1 -- the engine laws transfer.
# ---------------------------------------------------------------------------
def render_laws(out_dir: Path):
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.2, 4.5))

    ax_l.axhline(1.0, ls="--", lw=1.5, color="0.45", zorder=1,
                 label="ideal (no scaling loss)")
    for fam, (path, uni, _grid) in DP_SRC.items():
        eff = dp_cells(path, uni, "eff_gpu")
        dp = [(n, eff[(n, 1, c)]) for n, c in [(1, 128), (2, 256), (4, 512)]
              if (n, 1, c) in eff]
        tp = [(t, eff[(1, t, c)]) for t, c in [(1, 128), (2, 256), (4, 512)]
              if (1, t, c) in eff]
        ax_l.plot([g for g, _ in dp], [v for _, v in dp], marker=MARKERS[fam],
                  ms=8, lw=2.2, color=COLORS[fam], zorder=3, label=f"{fam} — DP")
        ax_l.plot([g for g, _ in tp], [v for _, v in tp], marker=MARKERS[fam],
                  ms=8, lw=2.2, ls="--", mfc="white", color=COLORS[fam],
                  zorder=3, label=f"{fam} — TP")
    ax_l.set_xscale("log", base=2); ax_l.set_xticks([1, 2, 4])
    ax_l.set_xticklabels(["1", "2", "4"])
    ax_l.set_xlabel("GPUs"); ax_l.set_ylabel("per-GPU efficiency")
    ax_l.set_ylim(0.5, 1.18)
    ax_l.set_title("Both scale-out axes, both families")
    ax_l.legend(loc="lower left", ncol=1)

    width = 0.35
    labels = ["4 × TP1", "2 × TP2", "1 × TP4"]
    for i, (fam, (path, _uni, grid)) in enumerate(DP_SRC.items()):
        rows = dp_cells(path, grid, "rows_s")
        base = rows.get((1, 4, 512))
        vals = [rows.get((4, 1, 512), 0) / base, rows.get((2, 2, 512), 0) / base, 1.0]
        xs = [j + (i - 0.5) * width for j in range(3)]
        ax_r.bar(xs, vals, width * 0.9, color=COLORS[fam], edgecolor="black",
                 linewidth=0.5, zorder=3, label=fam)
        for x, v in zip(xs, vals):
            ax_r.text(x, v + 0.02, f"{v:.2f}x", ha="center", va="bottom", fontsize=9.5)
    ax_r.axhline(1.0, ls=":", lw=1.4, color="0.45", zorder=1)
    ax_r.set_xticks(range(3)); ax_r.set_xticklabels(labels)
    ax_r.set_ylabel("throughput relative to 1 × TP4")
    ax_r.set_ylim(0, 1.8)
    ax_r.set_title("Fixed 4-GPU budget, total cap 512")
    ax_r.legend(loc="upper right")

    _save(fig, out_dir, "gen_family_laws")
    plt.close(fig)


# ---------------------------------------------------------------------------
# G2 -- the prompt does NOT transfer. The centrepiece.
# ---------------------------------------------------------------------------
def render_prompt(out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.axhline(ALWAYS_TRUE_F1, ls=":", lw=1.8, color="#c8722a", zorder=1)
    ax.annotate("always-true baseline", (32, ALWAYS_TRUE_F1), xytext=(0, -14),
                textcoords="offset points", ha="right", fontsize=10,
                color="#c8722a", fontweight="bold")
    for fam, (fpath, _e) in XS_SRC.items():
        rows = xs_rows(fpath)
        for head, style in (("slim", dict(ls="-", mfc=COLORS[fam])),
                            ("full", dict(ls="--", mfc="white"))):
            pts = []
            for R in (1, 2, 4, 8, 16, 32):
                arm = f"flock_op_slim_r{R}" if head == "slim" else f"flock_op_r{R}"
                if arm in rows and rows[arm].get("f1"):
                    pts.append((R, float(rows[arm]["f1"])))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[fam],
                    ms=8, lw=2.2, color=COLORS[fam], zorder=3,
                    label=f"{fam} — {head}", **style)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("rows packed per request (R)")
    ax.set_ylabel("F1")
    ax.set_ylim(0.45, 0.98)
    ax.set_title("Quality vs batch size, by prompt head and model family")
    ax.legend(loc="lower left", ncol=2)
    _save(fig, out_dir, "gen_family_prompt")
    plt.close(fig)


# ---------------------------------------------------------------------------
# G3 -- cross-system standing per family per task.
# ---------------------------------------------------------------------------
def render_frontier(out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    for row, fam in enumerate([QWEN, LLAMA]):
        for col, (task, path) in enumerate(zip(("filter", "extract"), XS_SRC[fam])):
            ax = axes[row][col]
            if not path.exists():
                ax.set_visible(False)
                continue
            rows = xs_rows(path)
            flock = [(float(r["rows_s"]), float(r["f1"]), a)
                     for a, r in rows.items()
                     if a.startswith("flock_op") and r.get("f1") and r.get("rows_s")]
            ax.scatter([p[0] for p in flock], [p[1] for p in flock], s=70,
                       color=COLORS[fam], edgecolor="black", linewidth=0.6,
                       zorder=3, label="Skein")
            best = max(flock, key=lambda p: p[1])
            ax.annotate(best[2].replace("flock_op_", "").replace("_", " "),
                        (best[0], best[1]), xytext=(6, 6),
                        textcoords="offset points", fontsize=9,
                        color=COLORS[fam], fontweight="bold")
            for arm, mk, c in (("lotus", "^", "#1f77b4"), ("palimpzest", "D", "#8c564b")):
                if arm in rows and rows[arm].get("f1"):
                    ax.scatter([float(rows[arm]["rows_s"])], [float(rows[arm]["f1"])],
                               s=110, marker=mk, color=c, edgecolor="black",
                               linewidth=0.7, zorder=4, label=arm.upper() if arm == "lotus" else "Palimpzest")
            ax.set_title(f"{fam} — {task}")
            ax.set_xlabel("rows/s"); ax.set_ylabel("F1")
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=9.5)
    _save(fig, out_dir, "gen_family_frontier")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/figures"))
    args = ap.parse_args()
    _setup_style()
    render_laws(args.out_dir)
    render_prompt(args.out_dir)
    render_frontier(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

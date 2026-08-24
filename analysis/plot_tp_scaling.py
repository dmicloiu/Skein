#!/usr/bin/env python3
"""[DESIGNED FOR THE TP SCALING EXPERIMENT ANALYSIS]
TP (tensor-parallel) scaling: PhysicalSemFilter (rewrite=on) vs the scalar
llm_filter, one vLLM endpoint sharded across 1/2/4 GH200s, full prompt, R=1;
data from analysis/slurm/tp_scaling_clariden.sh via summarize_tp.py.

Renders TWO paper figures (png + pdf each), house style of
plot_sem_filter_ab.py / plot_router_experiment.py. NO run configuration is
baked into the images (model / GPU / caps / reps belong in the LaTeX caption).

  tp_scaling_bars   grouped bars per TP: scalar + one operator bar per swept
                    cap. Carries the whole WHAT: the op/scalar gap (annotated
                    ratio) widens with TP, and the within-group green
                    staircase shows the saturating cap scaling with TP.
  tp_concurrency    the WHY (bullet-chart): measured in-flight concurrency
                    (Little's law) inside a translucent in_flight_cap budget
                    container -- the operator fills its budget at every TP,
                    the scalar is structurally pinned at 1.

Median over reps; whiskers = min-max.

Usage:
  python analysis/plot_tp_scaling.py \
      [--summary analysis/figures/data/tp_scaling_json/tp_scaling_summary.csv] \
      [--out-dir analysis/figures]
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

# Mirror plot_router_experiment.py: blue = baseline, green = behaviour under test.
COLOR_SCALAR = ts.COLOR_BASELINE      # scalar llm_filter (rewrite off) is the baseline
COLOR_OPERATOR = ts.COLOR_UNDER_TEST  # PhysicalSemFilter (rewrite on) is under test

TPS = [1, 2, 4]
CAPS = {1: [128], 2: [128, 256], 4: [128, 256, 512]}


def _setup_style():
    """Consistent styling: delegate to the shared thesis_style house rcParams."""
    ts.setup_style(base_font=14)


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts.no_suptitle(fig)  # house policy: no argumentative suptitle
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"wrote {path}", file=sys.stderr)


def collect(summary: Path, col: str = "rows_s", morsels: int = 1) -> dict:
    """{(arm, tp, cap): [<col> per rep]} for ONE morsel regime (1 = in-memory
    sf_2000; 128 = the heroic-scalar attached-DB runs)."""
    out: dict = {}
    with open(summary, newline="") as f:
        for r in csv.DictReader(f):
            if int(r.get("morsels") or 1) != morsels:
                continue
            key = (r["arm"], int(r["tp"]), int(r["cap"]) if r["cap"] else None)
            if r[col]:
                out.setdefault(key, []).append(float(r[col]))
    return out


def stat(vals: list[float]) -> tuple[float, float, float]:
    return median(vals), min(vals), max(vals)


def best_cap(data: dict, tp: int) -> int:
    return max(CAPS[tp], key=lambda c: median(data[("op", tp, c)]))


# ---------------------------------------------------------------------------
# Combined single-figure variant: grouped bars per TP -- scalar + one bar per
# swept cap. Gap = blue vs best green; TP scaling = group growth; cap story =
# the within-group green staircase.
# ---------------------------------------------------------------------------
def _bars_panel(ax, data: dict, scalar_label: str, ymax: float) -> None:
    """One grouped-bar panel: per TP a scalar bar + one operator bar per cap
    PRESENT in `data` (the sweep on the left panel, the settled cap on the
    right), value labels, min-max whiskers, gap ratio above the group."""
    cap_colors = {128: "#7fbf7f", 256: "#2e8b57", 512: "#1a5c1a"}
    width = 0.19
    seen = set()
    for gi, tp in enumerate(TPS):
        # presentation caps only: the c1024 saturation-check cell (single rep)
        # is reported in the write-up, not plotted.
        caps = sorted(c for (arm, t, c) in data
                      if arm == "op" and t == tp and c in CAPS[tp])
        bars = [("scalar", None)] + [("op", c) for c in caps]
        x0 = gi - width * (len(bars) - 1) / 2
        for bi, (arm, cap) in enumerate(bars):
            m, a, b = stat(data[(arm, tp, cap or 128)])
            x = x0 + bi * width
            if arm == "scalar":
                color, label = COLOR_SCALAR, (scalar_label if gi == 0 else None)
            else:
                color = cap_colors[cap]
                label = f"Skein, cap {cap}" if cap not in seen else None
                seen.add(cap)
            ax.bar(x, m, width * 0.92, yerr=[[m - a], [b - m]], capsize=3,
                   color=color, edgecolor="black", linewidth=0.6, label=label)
            ax.text(x, b + 8, f"{m:.0f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold")
        best = max(median(data[("op", tp, c)]) for c in caps)
        sc = median(data[("scalar", tp, 128)])
        ax.text(gi, best + ymax * 0.11, f"{best / sc:.1f}x", ha="center",
                va="bottom", fontsize=12.5, fontweight="bold", color="0.25")
    ax.set_xticks(range(len(TPS)))
    ax.set_xticklabels([f"TP={t}" for t in TPS])
    ax.set_xlabel("tensor-parallel size (GPUs per endpoint)")
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left")


def render_bars(data: dict, out_dir: Path, heroic: dict | None = None) -> None:
    """Single panel on sf_2000 data alone; two shared-y panels once the
    heroic-scalar (128-morsel) runs exist: realistic vs best-case scalar."""
    vals = [v for d in ([data, heroic] if heroic else [data]) for vs in d.values() for v in vs]
    ymax = max(vals) * 1.27
    if heroic:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
        _bars_panel(ax_l, data, "Flock", ymax)
        _bars_panel(ax_r, heroic, "Flock", ymax)
        ax_l.set_title("DuckDB threads = 8, morsels = 1")
        ax_r.set_title("DuckDB threads = 128, morsels = 128")
        ax_l.set_ylabel("throughput (rows/s)")
    else:
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        _bars_panel(ax, data, "Flock", ymax)
        ax.set_ylabel("throughput (rows/s)")
    _save(fig, out_dir, "tp_scaling_bars")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mechanism figure: measured in-flight concurrency (Little's law from the vLLM
# /metrics deltas) per arm vs TP. The operator holds ~in_flight_cap at every
# TP (dashed marker = the cap); the scalar is structurally pinned at 1.
# ---------------------------------------------------------------------------
def render_concurrency(data: dict, conc: dict, out_dir: Path) -> None:
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for gi, tp in enumerate(TPS):
        cap = best_cap(data, tp)
        m, a, b = stat(conc[("op", tp, cap)])
        sm, sa, sb = stat(conc[("scalar", tp, 128)])
        xo, xs = gi + width / 2, gi - width / 2
        # translucent budget container behind the group (bullet-chart idiom):
        # the operator fills it, the scalar leaves it unused (its concurrency
        # is knob-independent). Dashed top edge = the configured cap.
        ax.bar(gi, cap, 0.8, color="0.6", alpha=0.16, edgecolor="none", zorder=1,
               label="in_flight_cap (concurrency budget)" if gi == 0 else None)
        ax.hlines(cap, gi - 0.4, gi + 0.4, colors="0.35", linestyles="--",
                  lw=1.4, zorder=1.5)
        ax.text(gi, cap + 10, f"{cap}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="0.3")
        ax.bar(xs, sm, width * 0.9, yerr=[[sm - sa], [sb - sm]], capsize=3,
               color=COLOR_SCALAR, edgecolor="black", linewidth=0.6, zorder=3,
               label="Flock" if gi == 0 else None)
        ax.bar(xo, m, width * 0.9, yerr=[[m - a], [b - m]], capsize=3,
               color=COLOR_OPERATOR, edgecolor="black", linewidth=0.6, zorder=3,
               label="Skein (best cap)" if gi == 0 else None)
        ax.text(xs, sm + 12, f"{sm:.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", zorder=4)
    ax.set_xticks(range(len(TPS)))
    ax.set_xticklabels([f"TP={t}" for t in TPS])
    ax.set_xlabel("tensor-parallel size (GPUs per endpoint)")
    ax.set_ylabel("mean in-flight requests (Little's law)")
    ax.set_ylim(0, 590)
    ax.legend(loc="upper left")
    _save(fig, out_dir, "tp_concurrency")
    plt.close(fig)


def print_summary(data: dict) -> None:
    print("\n==================== TP scaling summary (median [min-max]) ====================")
    b_op = median(data[("op", 1, 128)])
    b_sc = median(data[("scalar", 1, 128)])
    for tp in TPS:
        cap = best_cap(data, tp)
        om, oa, ob = stat(data[("op", tp, cap)])
        sm, sa, sb = stat(data[("scalar", tp, 128)])
        print(f"TP={tp}: operator {om:6.1f} [{oa:.1f}-{ob:.1f}] @cap{cap}  eff {om/b_op:.2f} | "
              f"scalar {sm:5.1f} [{sa:.1f}-{sb:.1f}]  eff {sm/b_sc:.2f} | gap {om/sm:5.2f}x")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=Path("analysis/figures/data/tp_scaling_json/tp_scaling_summary.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/figures"))
    args = ap.parse_args()
    if not args.summary.exists():
        print(f"summary csv not found: {args.summary} (run summarize_tp.py first)", file=sys.stderr)
        return 1
    _setup_style()
    data = collect(args.summary)
    print_summary(data)
    heroic = collect(args.summary, morsels=128) or None
    render_bars(data, args.out_dir, heroic)
    render_concurrency(data, collect(args.summary, "concurrency"), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

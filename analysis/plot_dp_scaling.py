#!/usr/bin/env python3
"""[DESIGNED FOR THE DP SCALING EXPERIMENT ANALYSIS]
DP (data-parallel) scaling: N independent vLLM endpoints behind
semantic_endpoints + the round_robin EndpointRouter, full prompt, R=1, cap
128*N; data from analysis/slurm/dp_scaling_clariden.sh via summarize_dp.py.
The TP curve is read from the TP study's summary for the scale-out contrast.

Renders the paper figures (png + pdf each), house style of
plot_tp_scaling.py / plot_router_experiment.py. NO run configuration is baked
into the images (model / GPUs / caps / reps belong in the LaTeX caption).

  dp_scaling_curve   the WHAT: throughput and per-GPU efficiency vs total
                     GPUs -- DP tracks ideal-linear where TP (same GPUs, one
                     sharded endpoint) falls to ~0.7, over a flat scalar
                     reference that cannot use a fleet at all.
  dp_budget_bars     the HEADLINE: one 4-GPU budget, one total cap, three
                     topologies (4xTP1 / 2xTP2 / 1xTP4) -- spend GPUs on
                     replicas, not shards.
  dp_cap_sweep       H3 mechanism: the 128*N knee, then latency-only growth.
  dp_balance         the SANITY + bottleneck call: dispatch evenness, the cap
                     split, and driver CPU vs fleet concurrency.
  dp_model_frontier  fixed budget across model sizes: one dot-range row per
                     model ordered by F1; dot spacing = topology penalty,
                     the missing slot = the domain boundary.

Median over reps; whiskers = min-max. Every figure draws each cell from ONE
job family at 3 reps (curve / grid / unified / caps / scfleet / m*) so no
cell mixes families or rep counts.

Usage:
  python analysis/plot_dp_scaling.py \
      [--summary analysis/figures/data/dp_scaling/dp_scaling_summary.csv] \
      [--balance analysis/figures/data/dp_scaling/dp_balance.csv] \
      [--tp-summary analysis/figures/data/tp_scaling/tp_scaling_summary.csv] \
      [--out-dir analysis/figures]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# House convention (plot_tp_scaling.py): blue = the baseline being beaten,
# green = the behaviour under test. Here DP is under test; the TP curve is the
# scale-out baseline it is measured against.
COLOR_DP = "green"
COLOR_TP = "blue"
COLOR_IDEAL = "0.45"
COLOR_SCALAR = "#8c564b"   # distinct from the ideal-reference grays
COLOR_STARVE = "#c8722a"   # latency/over-cap accents

# DP scaling curve: fleet size -> (cap, total GPUs). cap = 128*N.
DP_CURVE = [(1, 128), (2, 256), (4, 512)]
# Fixed-budget grid, 4 GPUs and total cap 512: (endpoints, TP per endpoint).
GRID = [(4, 1), (2, 2), (1, 4)]
GRID_CAP = 512


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
        "legend.loc": "best",
    })


def _suptitle(fig, text: str):
    fig.suptitle(f"[PhysicalSemFilter]  {text}",
                 fontweight="bold", fontsize=12.5, x=0.02, y=1.02, ha="left")


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"wrote {path}", file=sys.stderr)


def stat(vals: list[float]) -> tuple[float, float, float]:
    return median(vals), min(vals), max(vals)


# ---- loading ---------------------------------------------------------------
def collect_dp(summary: Path, col: str, jobs: str | None = None,
               model: str | None = "7B") -> dict:
    """{(arm, n_ep, tp, cap): [<col> per rep]}. jobs filters on the job-label
    prefix ('grid' keeps the in-job fixed-budget reps); model keeps one
    model's cells (the scale-out figures are 7B -- without this the frontier
    models' cells silently pool into the same (n_ep, tp, cap) keys)."""
    out: dict = {}
    with open(summary, newline="") as f:
        for r in csv.DictReader(f):
            if jobs and not r["job"].startswith(jobs):
                continue
            if model and _model_short(r["model"]) != model:
                continue
            if not r[col]:
                continue
            key = (r["arm"], int(r["n_ep"]), int(r["tp"]), int(r["cap"]))
            out.setdefault(key, []).append(float(r[col]))
    return out


def tp_cells_from_dp(dp_data: dict) -> dict:
    """{tp: [vals]} from the unified jobs' in-job TP cells (n_ep=1, tp>1) --
    same node, same rows, same dataset as the DP cells, so the TP-vs-DP
    comparison carries no cross-study variance. TP=1 is the shared base cell."""
    out: dict = {}
    for (arm, n_ep, tp, cap), vals in dp_data.items():
        if arm == "op" and n_ep == 1:
            out.setdefault(tp, []).extend(vals)
    return out if len(out) > 1 else {}


def collect_tp(summary: Path, col: str) -> dict:
    """{tp: [<col> per rep]} for the operator at each TP's BEST cap, in-memory
    regime only -- the TP study's own presentation convention. Fallback when
    no in-job TP cells exist in the DP summary."""
    per: dict = {}
    with open(summary, newline="") as f:
        for r in csv.DictReader(f):
            if r["arm"] != "op" or int(r.get("morsels") or 1) != 1:
                continue
            if not r[col] or not r["rows_s"]:
                continue
            key = (int(r["tp"]), int(r["cap"]))
            per.setdefault(key, {"rows": [], "val": []})
            per[key]["rows"].append(float(r["rows_s"]))
            per[key]["val"].append(float(r[col]))
    out: dict = {}
    for tp in sorted({t for t, _ in per}):
        caps = [c for t, c in per if t == tp]
        best = max(caps, key=lambda c: median(per[(tp, c)]["rows"]))
        out[tp] = per[(tp, best)]["val"]
    return out


def collect_balance(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Figure 1 -- the scale-out curve, entirely from the unified jobs (both axes
# in-job, 3 reps, one family). Two panels: throughput and per-GPU efficiency;
# DP (replicas) vs TP (shards) vs ideal linear, over the flat scalar
# reference. Corrected tok/s tracks rows/s exactly (constant tokens/row), so
# it is caption material, not a panel; the cap story lives in dp_cap_sweep.
# ---------------------------------------------------------------------------
def _curve_panel(ax, dp_pts, tp_pts, ideal_base, ylabel, *, scalar=None,
                 fmt="{:.0f}", legend=False):
    gpus = [g for g, _, _, _ in dp_pts]
    ideal = [ideal_base * g for g in gpus]
    ax.plot(gpus, ideal, ls="--", lw=1.6, color=COLOR_IDEAL, zorder=1,
            label="ideal linear" if legend else None)
    if scalar is not None:
        ax.axhline(scalar, ls=":", lw=1.6, color=COLOR_SCALAR, zorder=1,
                   label="scalar llm_filter (no scale-out)" if legend else None)
        best = max(m for _, m, _, _ in dp_pts)
        ax.annotate(f"{scalar:.1f} rows/s ({best / scalar:.0f}x below)",
                    (max(gpus) * 1.2, scalar), textcoords="offset points",
                    xytext=(0, 5), ha="right", fontsize=9.5, color=COLOR_SCALAR)
    for pts, color, lab in ((tp_pts, COLOR_TP, "TP: 1 endpoint x N shards"),
                            (dp_pts, COLOR_DP, "DP: N endpoints x TP1")):
        if not pts:
            continue
        x = [g for g, _, _, _ in pts]
        m = [v for _, v, _, _ in pts]
        lo = [v - a for _, v, a, _ in pts]
        hi = [b - v for _, v, _, b in pts]
        ax.errorbar(x, m, yerr=[lo, hi], marker="o", ms=7, lw=2.0, capsize=3,
                    color=color, zorder=3, label=lab if legend else None)
    for i, (g, m, _, _) in enumerate(dp_pts):
        # leftmost label goes below-right of its marker: right of it because a
        # right-aligned one lands on the y-axis ticks, below because both
        # series rise to the right and would strike through it.
        off, ha = ((7, -14), "left") if i == 0 else ((-4, 9), "right")
        ax.annotate(fmt.format(m), (g, m), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=9.5, fontweight="bold",
                    color=COLOR_DP)
    for g, m, _, _ in tp_pts:
        if g > 1:  # the 1-GPU point is the shared base, labelled once in green
            ax.annotate(fmt.format(m), (g, m), textcoords="offset points",
                        xytext=(-2, -16), ha="right", fontsize=9.5,
                        fontweight="bold", color=COLOR_TP)
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpus)
    ax.set_xticklabels([str(g) for g in gpus])
    ax.set_xlim(min(gpus) * 0.82, max(gpus) * 1.22)
    ax.set_xlabel("GH200 GPUs")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="upper left")


def render_curve(rows, effs, tp_rows, tp_effs, scalar_rows,
                 out_dir: Path) -> None:
    fig, (ax_a, ax) = plt.subplots(1, 2, figsize=(11.4, 4.5))
    _curve_panel(ax_a, rows["dp"], tp_rows, rows["base"],
                 "throughput (rows/s)", scalar=scalar_rows, legend=True)
    ax_a.set_title("(a) throughput")

    # (b) per-GPU efficiency: ratio panel, so the ideal is a flat 1.0 line and
    # the y-axis is absolute -- no ideal_base scaling.
    ax.axhline(1.0, ls="--", lw=1.6, color=COLOR_IDEAL, zorder=1,
               label="ideal linear")
    for pts, color, lab in ((tp_effs, COLOR_TP, "TP: 1 endpoint x N shards"),
                            (effs["dp"], COLOR_DP, "DP: N endpoints x TP1")):
        if not pts:
            continue
        x = [g for g, _, _, _ in pts]
        m = [v for _, v, _, _ in pts]
        ax.errorbar(x, m, yerr=[[v - a for _, v, a, _ in pts],
                                [b - v for _, v, _, b in pts]],
                    marker="o", ms=7, lw=2.0, capsize=3, color=color,
                    zorder=3, label=lab)
    for g, m, _, _ in effs["dp"]:
        ax.annotate(f"{m:.2f}", (g, m), textcoords="offset points",
                    xytext=(-4, 9), ha="right", fontsize=9.5,
                    fontweight="bold", color=COLOR_DP)
    for g, m, _, _ in tp_effs:
        if g > 1:
            ax.annotate(f"{m:.2f}", (g, m), textcoords="offset points",
                        xytext=(-2, -15), ha="right", fontsize=9.5,
                        fontweight="bold", color=COLOR_TP)
    ax.set_xscale("log", base=2)
    ax.set_xticks([g for g, _, _, _ in effs["dp"]])
    ax.set_xticklabels([str(g) for g, _, _, _ in effs["dp"]])
    ax.set_xlim(0.82, 4 * 1.22)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("GH200 GPUs")
    ax.set_ylabel("per-GPU efficiency (x 1-GPU rows/s)")
    ax.set_title("(b) per-GPU efficiency")
    ax.legend(loc="lower left")
    _suptitle(fig, "DP scales linearly whereas TP does not")
    _save(fig, out_dir, "dp_scaling_curve")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- fixed 4-GPU budget, fixed total cap, three topologies. Same GPUs,
# same concurrency: only replicas-vs-shards differs.
# ---------------------------------------------------------------------------
def render_budget(grid_rows, grid_queue, out_dir: Path) -> None:
    shades = {4: "#1a5c1a", 2: "#2e8b57", 1: "#7fbf7f"}
    labels = {(4, 1): "4 x TP1\n(4 replicas)", (2, 2): "2 x TP2",
              (1, 4): "1 x TP4\n(1 sharded endpoint)"}
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.2, 4.7))

    # (a) the WHAT: throughput, annotated relative to the sharded baseline
    ax, ymax = ax_l, max(b for _, _, b in grid_rows.values()) * 1.30
    ref = grid_rows[(1, 4)][0]         # 1xTP4 = the vertical-only topology
    for gi, key in enumerate(GRID):
        m, a, b = grid_rows[key]
        ax.bar(gi, m, 0.56, yerr=[[m - a], [b - m]], capsize=3,
               color=shades[key[0]], edgecolor="black", linewidth=0.6,
               zorder=3)
        ax.text(gi, b + ymax * 0.02, f"{m:.0f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")
        # the explicit 1.00x names the baseline without extra text
        ax.text(gi, b + ymax * 0.10, f"{m / ref:.2f}x", ha="center",
                va="bottom", fontsize=12, fontweight="bold", color="0.25")
    ax.set_xticks(range(len(GRID)))
    ax.set_xticklabels([labels[k] for k in GRID])
    ax.set_xlabel("topology at a fixed 4-GPU budget")
    ax.set_ylabel("throughput (rows/s)")
    ax.set_ylim(0, ymax)
    ax.set_title("(a) throughput")

    # (b) the WHY: per-request scheduler queue time. One engine serialises
    # admission that N independent engines absorb.
    ax, ymax = ax_r, max(b for _, _, b in grid_queue.values()) * 1.22
    for gi, key in enumerate(GRID):
        m, a, b = grid_queue[key]
        ax.bar(gi, m, 0.56, yerr=[[m - a], [b - m]], capsize=3,
               color=shades[key[0]], edgecolor="black", linewidth=0.6,
               zorder=3)
        ax.text(gi, b + ymax * 0.02, f"{m:.0f} ms", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(len(GRID)))
    ax.set_xticklabels([labels[k] for k in GRID])
    ax.set_xlabel("topology at a fixed 4-GPU budget")
    ax.set_ylabel("mean admission latency per request (ms)")
    ax.set_ylim(0, ymax)
    ax.set_title("(b) mean admission latency")

    _suptitle(fig, "performance analysis of different topologies at 4 GPUs and the same total concurrency")
    _save(fig, out_dir, "dp_budget_bars")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- H3 mechanism: the cap sweep at N=4. Throughput saturates at the
# 128*N knee; past it the surplus in-flight only buys latency (Little's law:
# flat rows/s x growing concurrency = proportionally growing e2e).
# ---------------------------------------------------------------------------
def render_cap_sweep(rows_c: dict, e2e_c: dict, wait_c: dict, n_ep: int,
                     out_dir: Path) -> None:
    caps = sorted(rows_c)
    knee = 128 * n_ep
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.2, 4.5))

    def xstyle(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(caps)
        ax.set_xticklabels([str(c) for c in caps])
        ax.set_xlabel(f"in_flight_cap ({n_ep}-endpoint fleet)")
        ax.axvline(knee, ls="--", lw=1.4, color="0.45", zorder=1)

    m = {c: stat(rows_c[c]) for c in caps}
    ax_l.errorbar(caps, [m[c][0] for c in caps],
                  yerr=[[m[c][0] - m[c][1] for c in caps],
                        [m[c][2] - m[c][0] for c in caps]],
                  marker="o", ms=7, lw=2.0, capsize=3, color=COLOR_DP, zorder=3)
    for c in caps:
        # rightmost label shifts left so the axis edge and marker stay clear
        off, ha = ((-8, -16), "right") if c == caps[-1] else ((0, -16), "center")
        ax_l.annotate(f"{m[c][0]:.0f}", (c, m[c][0]), textcoords="offset points",
                      xytext=off, ha=ha, fontsize=9.5,
                      fontweight="bold", color=COLOR_DP)
    ax_l.text(knee * 1.05, 596, "cap = 128·N", va="top", ha="left",
              fontsize=10, color="0.35")
    xstyle(ax_l)
    ax_l.set_ylabel("throughput (rows/s)")
    ax_l.set_ylim(440, 600)
    ax_l.set_title("(a) throughput")

    e = {c: stat(e2e_c[c]) for c in caps if c in e2e_c}
    ax_r.errorbar(caps, [e[c][0] for c in caps],
                  yerr=[[e[c][0] - e[c][1] for c in caps],
                        [e[c][2] - e[c][0] for c in caps]],
                  marker="o", ms=7, lw=2.0, capsize=3, color=COLOR_STARVE,
                  zorder=3, label="mean e2e latency")
    for c in caps:
        w = wait_c.get(c)
        if w and median(w) > 5:
            ax_r.annotate(f"~{median(w):.0f} queued", (c, e[c][0]),
                          textcoords="offset points", xytext=(-8, 10),
                          ha="right", fontsize=9.5, color="0.35")
    xstyle(ax_r)
    ax_r.set_yscale("log", base=2)
    yt = [256, 512, 1024, 2048, 4096]
    ax_r.set_yticks([v for v in yt if e[caps[0]][0] * 0.8 <= v <= e[caps[-1]][0] * 1.3])
    ax_r.set_yticklabels([str(v) for v in ax_r.get_yticks()])
    ax_r.set_ylabel("mean e2e latency (ms)")
    ax_r.set_title("(b) mean e2e latency")
    _suptitle(fig, "performance analysis of the in-flight cap on a 4-endpoint fleet")
    _save(fig, out_dir, "dp_cap_sweep")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 -- the model-size frontier at a fixed 4-GPU budget. Within a model,
# F1 is topology-invariant (a horizontal band) and topology only moves the
# operating point along rows/s; the model choice picks the band. Topologies a
# model cannot legally run (weights exceed one GPU) are the domain boundary.
# ---------------------------------------------------------------------------
TOPO_SHADES = {(4, 1): "#1a5c1a", (2, 2): "#2e8b57", (1, 4): "#7fbf7f"}
TOPO_NAMES = {(4, 1): "4 x TP1", (2, 2): "2 x TP2", (1, 4): "1 x TP4"}


def _model_short(m: str | None) -> str:
    mm = re.search(r"(\d+)B", m or "")
    return f"{mm.group(1)}B" if mm else "7B"  # pre-sidecar legacy rows are 7B


def collect_frontier(summary_rows: list[dict]) -> dict:
    """{(model, n_ep, tp): {rows_s: [...], f1: [...]}} for 4-GPU cap-512 cells."""
    out: dict = {}
    for r in summary_rows:
        if (r["arm"] != "op" or r["cap"] != str(GRID_CAP)
                or int(r["n_ep"]) * int(r["tp"]) != 4):
            continue
        key = (_model_short(r["model"]), int(r["n_ep"]), int(r["tp"]))
        c = out.setdefault(key, {"rows_s": [], "f1": []})
        if r["rows_s"]:
            c["rows_s"].append(float(r["rows_s"]))
        if r["f1"]:
            c["f1"].append(float(r["f1"]))
    return out


def render_model_frontier(cells: dict, out_dir: Path) -> None:
    """Single dot-range panel: one row per model, dots = topologies at their
    absolute rows/s on a log axis (dot distance IS the relative penalty), the
    architecturally-impossible slot annotated, F1 per row. Rows are ordered by
    F1, not by model size -- the non-monotonicity is the row order."""
    def f1_of(model):
        v = [median(c["f1"]) for k, c in cells.items()
             if k[0] == model and c["f1"]]
        return median(v) if v else 0.0

    models = sorted({k[0] for k in cells}, key=f1_of)  # bottom row = lowest F1
    order = [(4, 1), (2, 2), (1, 4)]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    for yi, model in enumerate(models):
        pts = {t: median(cells[(model, *t)]["rows_s"])
               for t in order if (model, *t) in cells}
        ax.plot([min(pts.values()), max(pts.values())], [yi, yi],
                color="0.75", lw=2, zorder=2, solid_capstyle="round")
        for t, x in pts.items():
            ax.scatter([x], [yi], s=110, color=TOPO_SHADES[t],
                       edgecolor="black", linewidth=0.6, zorder=3,
                       label=TOPO_NAMES[t] if yi == len(models) - 1 else None)
        best = max(pts.values())
        ax.annotate(f"{best:.0f}", (best, yi), textcoords="offset points",
                    xytext=(9, -4), ha="left", fontsize=9.5, fontweight="bold",
                    color=TOPO_SHADES[(4, 1)])
        missing = [t for t in order if (model, *t) not in cells]
        if missing:
            ax.annotate(" + ".join(TOPO_NAMES[t] for t in missing)
                        + ": does not fit", (best, yi),
                        textcoords="offset points", xytext=(9, 10), ha="left",
                        fontsize=9, color="0.45", style="italic")
        ax.annotate(f"F1 = {f1_of(model):.3f}", (0.99, yi),
                    xycoords=("axes fraction", "data"), ha="right",
                    va="center", fontsize=10.5, color="0.25",
                    fontweight="bold")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=12)
    ax.set_ylim(-0.6, len(models) - 0.25)
    ax.set_xscale("log", base=2)
    xt = [32, 64, 128, 256, 512, 1024]
    ax.set_xticks(xt)
    ax.set_xticklabels([str(v) for v in xt])
    ax.set_xlim(30, 1600)
    ax.set_xlabel("throughput (rows/s)")
    ax.set_ylabel("model (rows ordered by F1)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0)
    _suptitle(fig, "quality and throughput across model sizes and topologies "
                   "at a fixed 4-GPU budget")
    _save(fig, out_dir, "dp_model_frontier")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 -- dispatch evenness + the client-side bottleneck call. The
# per-endpoint in-flight split is NOT plotted: its rigorous form is the
# Little's-law concurrency column, and the per-endpoint gauges carry
# sequential-poll sampling skew.
# ---------------------------------------------------------------------------
def render_balance(bal: list[dict], summary_rows: list[dict], out_dir: Path) -> None:
    # configs shown: the multi-endpoint fleets (a single endpoint has nothing
    # to balance) at one group per shape -- evenness is cap-independent --
    # then the scalar pointed at the same 4-endpoint fleet: the no-router
    # contrast. Last element = the job family the config is drawn from (one
    # family per cell, 3 reps, matching the tables).
    cfgs = [("op", 2, 1, 256, "curve"), ("op", 4, 1, 512, "curve"),
            ("op", 2, 2, 512, "grid"), ("scalar", 4, 1, 128, "scfleet")]
    names = {("op", 2, 1, 256): "2 x TP1\ncap 256",
             ("op", 4, 1, 512): "4 x TP1\ncap 512",
             ("op", 2, 2, 512): "2 x TP2\ncap 512",
             ("scalar", 4, 1, 128): "scalar\n4 x TP1"}
    ep_colors = ["#1a5c1a", "#2e8b57", "#5aab6e", "#9ccfa4"]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))

    def per_ep(cfg, col):
        arm, n_ep, tp, cap, fam = cfg
        out = []
        for ep in range(n_ep):
            v = [float(r[col]) for r in bal
                 if r["arm"] == arm and r["job"].startswith(fam)
                 and (int(r["n_ep"]), int(r["tp"]),
                      int(r["cap"])) == (n_ep, tp, cap)
                 and int(r["ep"]) == ep and r[col]]
            out.append(stat(v) if v else None)
        return out

    # (a) request share vs the ideal 1/N; the scalar group shows 100/0/0/0 --
    # no router in that path, the fleet is unreachable.
    ax = axes[0]
    width = 0.19
    for gi, cfg in enumerate(cfgs):
        vals = per_ep(cfg, "req_share_pct")
        x0 = gi - width * (len(vals) - 1) / 2
        for ep, s in enumerate(vals):
            if s is None:
                continue
            ax.bar(x0 + ep * width, s[0], width * 0.9, color=ep_colors[ep],
                   edgecolor="black", linewidth=0.5, zorder=3,
                   label=f"endpoint {ep}" if gi == 0 or (gi == 1 and ep > 1) else None)
            if cfg[0] == "scalar" and s[0] == 0:
                # only the zeroes need marking; the full bar speaks for itself
                ax.text(x0 + ep * width, s[0] + 2, "0", ha="center",
                        va="bottom", fontsize=9, fontweight="bold")
        ideal = 100.0 / cfg[1]
        ax.hlines(ideal, gi - 0.42, gi + 0.42, colors="0.3", linestyles="--",
                  lw=1.5, zorder=4,
                  label="ideal 1/N" if gi == 0 else None)
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([names[c[:4]] for c in cfgs])
    ax.set_ylabel("share of fleet requests (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(a) round_robin dispatch evenness")
    ax.legend(loc="upper left", ncol=2, fontsize=9.5)

    # (b) driver CPU vs fleet concurrency: the client-vs-GPU discriminator.
    ax = axes[1]
    pts = []
    for r in summary_rows:
        if r["arm"] != "op" or not r["client_cpu_cores"] or not r["concurrency"]:
            continue
        pts.append((float(r["concurrency"]), float(r["client_cpu_cores"])))
    sc = [(float(r["concurrency"]), float(r["client_cpu_cores"]))
          for r in summary_rows if r["arm"] == "scalar"
          and r["client_cpu_cores"] and r["concurrency"]]
    ax.axhline(1.0, ls=":", lw=1.6, color=COLOR_SCALAR, zorder=1,
               label="one saturated core")
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=64,
                   color=COLOR_DP, edgecolor="black", linewidth=0.6, zorder=3,
                   label="operator (async, 1 IO thread)")
    if sc:
        ax.scatter([p[0] for p in sc], [p[1] for p in sc], s=64, marker="s",
                   color=COLOR_TP, edgecolor="black", linewidth=0.6, zorder=3,
                   label="scalar llm_filter (blocking)")
    ax.set_xlabel("fleet concurrency (mean in-flight requests)")
    ax.set_ylabel("driver CPU (cores)")
    # scalar sits at concurrency ~1, the cap-2048 probe at ~1965: clip neither.
    ax.set_xlim(-60, max((p[0] for p in pts), default=500) * 1.08)
    ax.set_ylim(0, 1.35)
    ax.set_title("(b) driver CPU vs fleet concurrency")
    ax.legend(loc="center right")

    _suptitle(fig, "dispatch evenness and client-side cost across fleet configurations")
    _save(fig, out_dir, "dp_balance")
    plt.close(fig)


# ---- assembly --------------------------------------------------------------
def curve_series(data: dict, base_key=("op", 1, 1, 128)):
    """(gpus, median, min, max) per DP curve point, and the 1-GPU base used
    for the ideal-linear line."""
    base = median(data[base_key])
    dp = [(n * 1, *stat(data[("op", n, 1, cap)])) for n, cap in DP_CURVE]
    return {"dp": dp, "base": base}


def eff_series(data: dict):
    """Per-GPU efficiency straight from the summary's eff_gpu column."""
    return [(n, *stat(data[("op", n, 1, cap)])) for n, cap in DP_CURVE
            if ("op", n, 1, cap) in data]


def tp_series(tp_vals: dict, per_gpu_base: float | None = None):
    out = []
    for tp in sorted(tp_vals):
        m, a, b = stat(tp_vals[tp])
        if per_gpu_base:
            m, a, b = m / tp / per_gpu_base, a / tp / per_gpu_base, b / tp / per_gpu_base
        out.append((tp, m, a, b))
    return out


def print_summary(rows, effs, grid_rows, tp_rows) -> None:
    print("\n==================== DP scaling summary (median [min-max]) ====================")
    for (g, m, a, b), (_, e, ea, eb) in zip(rows["dp"], effs["dp"]):
        print(f"N={g:<2} cap={128*g:<4} {m:6.1f} rows/s [{a:.1f}-{b:.1f}]  "
              f"speedup {m/rows['base']:.2f}x  per-GPU eff {e:.2f} [{ea:.2f}-{eb:.2f}]")
    print("\n---- fixed 4-GPU budget, total cap 512 ----")
    ref = grid_rows[(1, 4)][0]
    for key in GRID:
        m, a, b = grid_rows[key]
        print(f"{key[0]} x TP{key[1]}: {m:6.1f} rows/s [{a:.1f}-{b:.1f}]  "
              f"{m/ref:.2f}x vs 1xTP4")
    print("\n---- DP vs TP at the same GPU count ----")
    for g, m, a, b in tp_rows:
        dp = next((v for gg, v, _, _ in rows["dp"] if gg == g), None)
        if dp:
            print(f"{g} GPU(s): DP {dp:6.1f} vs TP {m:6.1f} rows/s -> DP {dp/m:.2f}x, "
                  f"per-GPU eff {dp/rows['base']/g:.2f} vs {m/rows['base']/g:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=Path("analysis/figures/data/dp_scaling/dp_scaling_summary.csv"))
    ap.add_argument("--balance", type=Path,
                    default=Path("analysis/figures/data/dp_scaling/dp_balance.csv"))
    ap.add_argument("--tp-summary", type=Path,
                    default=Path("analysis/figures/data/tp_scaling/tp_scaling_summary.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/figures"))
    args = ap.parse_args()
    if not args.summary.exists():
        print(f"summary csv not found: {args.summary} (run summarize_dp.py first)",
              file=sys.stderr)
        return 1
    _setup_style()

    # One job family per artifact, 3 reps per cell: the curve figure and the
    # consolidated DP-vs-TP table draw ONLY from the unified jobs (both axes
    # measured in the same jobs on the same nodes).
    rows_u = collect_dp(args.summary, "rows_s", jobs="unified")
    effs_u = collect_dp(args.summary, "eff_gpu", jobs="unified")
    if ("op", 1, 1, 128) not in rows_u:
        print("  [warn] no unified jobs -> curve pools all 7B cells",
              file=sys.stderr)
        rows_u = collect_dp(args.summary, "rows_s")
        effs_u = collect_dp(args.summary, "eff_gpu")
    rows = curve_series(rows_u)
    effs = {"dp": eff_series(effs_u)}

    # flat no-scale-out reference: the dedicated scalar-on-fleet jobs
    # (4-endpoint fleet, operator dataset); fallback: any scalar cell.
    scal = collect_dp(args.summary, "rows_s", jobs="scfleet")
    scalar_vals = [v for (arm, n, _, _), vs in scal.items()
                   if arm == "scalar" and n == 4 for v in vs]
    if not scalar_vals:
        scalar_vals = [v for (arm, _, _, _), vs in
                       collect_dp(args.summary, "rows_s").items()
                       if arm == "scalar" for v in vs]
    scalar_rows = median(scalar_vals) if scalar_vals else None

    # TP contrast, normalised per GPU against the DP 1-GPU base so both curves
    # share one efficiency scale. Preferred source: the unified jobs' in-job
    # TP cells; fallback: the TP study's own summary (cross-study, carries
    # ~+-10% node variance).
    tp_rows_v = tp_cells_from_dp(rows_u)
    if not tp_rows_v and args.tp_summary.exists():
        print("  [warn] no in-job TP cells -> falling back to the TP study summary",
              file=sys.stderr)
        tp_rows_v = collect_tp(args.tp_summary, "rows_s")
    tp_rows = tp_series(tp_rows_v)
    tp_effs = tp_series(tp_rows_v, per_gpu_base=rows["base"])

    # fixed-budget grid: in-job reps only, so all three topologies come from
    # the same jobs (cross-job would carry the node variance).
    grid_rows_src = collect_dp(args.summary, "rows_s", jobs="grid")
    grid_queue_src = collect_dp(args.summary, "queue_ms", jobs="grid")
    missing = [k for k in GRID if ("op", k[0], k[1], GRID_CAP) not in grid_rows_src]
    if missing:
        print(f"  [warn] grid incomplete, missing {missing} -> skipping dp_budget_bars",
              file=sys.stderr)
        grid_rows = grid_queue = None
    else:
        grid_rows = {k: stat(grid_rows_src[("op", k[0], k[1], GRID_CAP)]) for k in GRID}
        grid_queue = {k: stat(grid_queue_src[("op", k[0], k[1], GRID_CAP)]) for k in GRID}

    print_summary(rows, effs, grid_rows or {}, tp_rows) if grid_rows else None
    render_curve(rows, effs, tp_rows, tp_effs, scalar_rows, args.out_dir)
    if grid_rows:
        render_budget(grid_rows, grid_queue, args.out_dir)

    # cap sweep at the largest fleet (needs >= 4 cap points to be a curve);
    # its family is the curve jobs (caps 128/512) + the caps jobs (the rest).
    N_SWEEP = 4
    sweep_jobs = ("curve", "caps")

    def sweep(col):
        return {cap: v for (arm, n, tp, cap), v in
                collect_dp(args.summary, col, jobs=sweep_jobs).items()
                if arm == "op" and n == N_SWEEP and tp == 1}

    rows_c = sweep("rows_s")
    if len(rows_c) >= 4:
        render_cap_sweep(rows_c, sweep("e2e_ms"), sweep("wait_mean"),
                         N_SWEEP, args.out_dir)
    else:
        print(f"  [warn] only {len(rows_c)} cap points at N={N_SWEEP} -> skipping dp_cap_sweep",
              file=sys.stderr)
    with open(args.summary, newline="") as f:
        summary_rows = list(csv.DictReader(f))
    if args.balance.exists():
        render_balance(collect_balance(args.balance), summary_rows, args.out_dir)
    else:
        print(f"  [warn] balance csv missing ({args.balance}) -> skipping dp_balance",
              file=sys.stderr)

    # model frontier (needs >1 model among the 4-GPU cap-512 cells)
    cells = collect_frontier(summary_rows)
    if len({k[0] for k in cells}) > 1:
        render_model_frontier(cells, args.out_dir)
    else:
        print("  [warn] single-model summary -> skipping dp_model_frontier",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Illustrate that flock::EndpointRouter behaves as designed, from the raw data
produced on Clariden by the two slurm drivers in analysis/slurm/.

Renders THREE figures (each as .png + .pdf), each
matching one claim and one data source. Each figure is written only if the data
it needs is present in --results-dir, so a run on either dir touches only its
own figures (no figure clobbers another's data):

  router_throughput     single (1 GPU) vs round_robin (pool) throughput, framed
                        as scaling efficiency vs the n-GPU linear ideal.
                        <- router_analysis_clariden.sh  (tags: single, round_robin)
  router_load_aware     two panels on a fleet with one throttled endpoint:
                        (left) per-endpoint placement, (right) p50/p95/p99 tail
                        latency -- least_loaded vs round_robin.
                        <- router_analysis_clariden.sh  (tags: least_loaded,
                           round_robin_hetero)
  router_prefix_cache   pool prefix-cache hit rate vs the number of distinct
                        recurring prefixes K -- sticky_by_prefix vs round_robin.
                        <- router_sticky_sweep_clariden.sh  (tags: sticky_k<K>,
                           round_robin_k<K>)
  router_prefix_cache_throughput
                        the performance CONSEQUENCE of that hit-rate gap: pool
                        throughput (rows/s) vs K, sticky_by_prefix vs
                        round_robin. Same sweep, same per-run blobs.
                        <- router_sticky_sweep_clariden.sh
  router_prefix_cache_combined
                        the two above as one two-panel float (hit rate |
                        throughput), sharing the K axis -- the form used in the
                        paper. <- router_sticky_sweep_clariden.sh

Inputs (in --results-dir):
  result_<tag>.json                       per-run blob from the integration binary
  metrics_{before,after}_<tag>_ep<i>.txt  raw vLLM /metrics snapshots

Usage (run once per data dir; figures land together in --out-dir):
  python analysis/plot_router_experiment.py \\
      --results-dir analysis/figures/data/router_analysis        --out-dir analysis/figures
  python analysis/plot_router_experiment.py \\
      --results-dir analysis/figures/data/sticky_routing_analysis --out-dir analysis/figures
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Per-strategy colours, used consistently across every figure. Mirrors the
# reference convention (blue = baseline, green = the behaviour under test).
COLOR_SINGLE = "#d62728"   # red    — degenerate one-GPU baseline
COLOR_RR     = "blue"      # blue   — round_robin, the comparison baseline
COLOR_STICKY = "green"     # green  — sticky_by_prefix
COLOR_LL     = "#ff7f0e"   # orange — least_loaded

TAG_SINGLE = "single"
TAG_RR = "round_robin"
TAG_LL = "least_loaded"
TAG_RR_HETERO = "round_robin_hetero"

SUPTITLE_CTX = "Qwen2.5-7B on Clariden (4x GH200 vLLM pool)"


def _setup_style():
    """Apply consistent styling to all plots (matches asyncllmclient_vs_vllm.py)."""
    import matplotlib.pyplot as plt
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
        "hatch.linewidth": 1.1,   # crisper bar-fill patterns
    })


def _label(name: str, unit: str | None = None) -> str:
    return f"{name} ({unit})" if unit else name


def _suptitle(fig, text: str):
    fig.suptitle(f"[EndpointRouter]  {text}   ·   {SUPTITLE_CTX}",
                 fontweight="bold", fontsize=12.5, x=0.02, y=1.02, ha="left")


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"wrote {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_result(results_dir: Path, tag: str):
    path = results_dir / f"result_{tag}.json"
    if not path.exists():
        print(f"  [warn] missing {path.name}", file=sys.stderr)
        return None
    with path.open() as f:
        return json.load(f)

# Account for different metric names of vLLM system
_HITS_RE = re.compile(r"^vllm:(?:gpu_)?prefix_cache_hits_total\b.*\s([0-9.eE+-]+)\s*$")
_QUERIES_RE = re.compile(r"^vllm:(?:gpu_)?prefix_cache_queries_total\b.*\s([0-9.eE+-]+)\s*$")


def _sum_metric(path: Path, regex: re.Pattern, group: int = 1) -> float | None:
    if not path.exists():
        return None
    total, matched = 0.0, False
    with path.open() as f:
        for line in f:
            m = regex.match(line.strip())
            if m:
                total += float(m.group(group))
                matched = True
    return total if matched else None


def prefix_cache_deltas(results_dir: Path, tag: str, ep: int):
    """(delta_hits, delta_queries) over the run window, or None if unparsable."""
    before = results_dir / f"metrics_before_{tag}_ep{ep}.txt"
    after = results_dir / f"metrics_after_{tag}_ep{ep}.txt"
    h0 = _sum_metric(before, _HITS_RE)
    h1 = _sum_metric(after, _HITS_RE)
    q0 = _sum_metric(before, _QUERIES_RE)
    q1 = _sum_metric(after, _QUERIES_RE)
    if None in (h0, h1, q0, q1):
        return None
    return (h1 - h0, q1 - q0)


def prefix_cache_hit_rate(results_dir: Path, tag: str, ep: int) -> float | None:
    """Per-endpoint hit rate: delta(hits) / delta(queries)."""
    d = prefix_cache_deltas(results_dir, tag, ep)
    if d is None:
        return None
    dh, dq = d
    if dq <= 0:
        return 0.0
    return max(0.0, min(1.0, dh / dq))


def prefix_cache_pool_rate(results_dir: Path, tag: str, n: int) -> float | None:
    """Pool-aggregate hit rate: sum(hits) / sum(queries) over all endpoints.
    The correct headline number -- not the mean of per-endpoint rates, which
    mis-weights endpoints with different query counts."""
    tot_h, tot_q, seen = 0.0, 0.0, False
    for ep in range(n):
        d = prefix_cache_deltas(results_dir, tag, ep)
        if d is None:
            continue
        tot_h += d[0]; tot_q += d[1]; seen = True
    if not seen or tot_q <= 0:
        return None
    return max(0.0, min(1.0, tot_h / tot_q))


def _n_endpoints(*results) -> int:
    for r in results:
        if r and r.get("endpoints"):
            return len(r["endpoints"])
    return 4


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_throughput(out_dir, single, rr, n):
    """Fig 1: single (#1) vs round_robin (#2). Both full-cache/uniform/inflight-256
    /TOTAL-2048, so the comparison is a fair scaling measurement. Reads as
    scaling EFFICIENCY against the n-GPU linear ideal, not just 'bigger'."""
    import matplotlib.pyplot as plt
    if not (single and rr):
        return
    fig, ax = plt.subplots(figsize=(6, 4.8))
    tools = ["single\n(1 GPU)", f"round_robin\n({n} GPUs)"]
    vals = [single["throughput_rows_per_s"], rr["throughput_rows_per_s"]]
    bars = ax.bar(tools, vals, width=0.5, color=[COLOR_SINGLE, COLOR_RR],
                  edgecolor="black", linewidth=0.6)
    ax.set_ylabel(_label("throughput", "rows/s"))
    ax.set_title("Throughput scaling: 1 GPU vs pool")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Linear-scaling ideal = n x single. Frame round_robin as % of that.
    if vals[0] > 0:
        ideal = n * vals[0]
        ax.set_ylim(0, ideal * 1.20)
        ax.axhline(ideal, ls="--", lw=2.6, color="0.2", zorder=1)
        ax.text(0.0, ideal, f"linear ideal ({n}x = {ideal:,.0f})", ha="left",
                va="bottom", fontsize=11, fontweight="bold", color="0.2")
        # Efficiency annotation in the gap between the pool bar and the ideal.
        eff = vals[1] / ideal
        ax.text(0.5, ideal * 0.55,
                f"{vals[1] / vals[0]:.1f}x of {n}x ideal\n({eff:.0%} efficiency)",
                ha="center", va="center", fontsize=11,
                color=COLOR_RR, fontweight="bold")
    else:
        ax.set_ylim(0, max(vals) * 1.30)
    _suptitle(fig, "round_robin spreads load across the pool")
    _save(fig, out_dir, "router_throughput")
    plt.close(fig)


def fig_prefix_cache_sweep(out_dir, results_dir, n):
    """Fig 2 (sweep form): pool prefix-cache hit rate vs the number of distinct
    recurring prefixes K, for sticky vs round_robin. Produced by
    router_sticky_sweep_clariden.sh (tags sticky_k<K> / round_robin_k<K>).

    The curve is the behaviour proof: at small K both cache everything and tie;
    as K grows past the cache's reach, round_robin collapses (it must hold all K
    per endpoint) while sticky stays high (it holds only K/NGPU). Returns True if
    sweep data was found and plotted, False otherwise (caller falls back)."""
    import matplotlib.pyplot as plt
    Ks = sorted(int(re.search(r"_k(\d+)\.json$", p.name).group(1))
                for p in results_dir.glob("result_sticky_k*.json"))
    if not Ks:
        return False
    s = _sweep_series(results_dir, "sticky", Ks, n)
    r = _sweep_series(results_dir, "round_robin", Ks, n)
    if not s:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.8))

    # Each line carries only its own per-endpoint min/max band (no fill between
    # the two curves -- that read as an ambiguous blob).
    _plot_sweep_line(ax, s, COLOR_STICKY, "-o", "sticky_by_prefix")
    _plot_sweep_line(ax, r, COLOR_RR, "-s", "round_robin")

    # Peak-divergence annotation: a vertical double-arrow spanning the gap at the
    # K where sticky's lead is widest.
    common = sorted(set(K for K, *_ in s) & set(K for K, *_ in r))
    if common:
        sm = {K: v for K, v, *_ in s}
        rm = {K: v for K, v, *_ in r}
        kbest = max(common, key=lambda K: sm[K] - rm[K])
        gap = sm[kbest] - rm[kbest]
        ax.annotate("", xy=(kbest, sm[kbest]), xytext=(kbest, rm[kbest]),
                    arrowprops=dict(arrowstyle="<->", lw=1.3, color=COLOR_STICKY))
        ax.text(kbest * 1.12, (sm[kbest] + rm[kbest]) / 2, f"+{gap*100:.0f} pts",
                ha="left", va="center", fontsize=9, fontweight="bold",
                color=COLOR_STICKY)

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("distinct recurring prefixes  K")
    ax.set_ylabel("pool prefix-cache hit rate")
    ax.set_ylim(0, 0.45)
    ax.set_title("Prefix-cache reuse vs prefix diversity")
    for K, v, *_ in s:
        ax.annotate(f"{v:.0%}", (K, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5, color=COLOR_STICKY)
    for K, v, *_ in r:
        ax.annotate(f"{v:.0%}", (K, v), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7.5, color=COLOR_RR)
    ax.legend(loc="upper right")
    _suptitle(fig, "sticky_by_prefix sustains reuse as prefix diversity grows")
    _save(fig, out_dir, "router_prefix_cache")
    plt.close(fig)
    return True


def _sweep_series(results_dir, strat, Ks, n):
    """[(K, pool_rate, per_endpoint_min, per_endpoint_max), ...] for a strategy."""
    out = []
    for K in Ks:
        pool = prefix_cache_pool_rate(results_dir, f"{strat}_k{K}", n)
        if pool is None:
            continue
        per = [prefix_cache_hit_rate(results_dir, f"{strat}_k{K}", ep)
               for ep in range(n)]
        per = [p for p in per if p is not None]
        lo = min(per) if per else pool
        hi = max(per) if per else pool
        out.append((K, pool, lo, hi))
    return out


def _plot_sweep_line(ax, series, color, style, label):
    """Pool-rate line + a per-endpoint min/max band (shows cross-endpoint
    uniformity -- a tight band means the win is structural, not one endpoint)."""
    if not series:
        return
    ks = [K for K, *_ in series]
    pool = [v for _, v, _, _ in series]
    lo = [a for _, _, a, _ in series]
    hi = [b for _, _, _, b in series]
    ax.fill_between(ks, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.plot(ks, pool, style, color=color, linewidth=2.0, markersize=6, label=label)


def _sweep_throughput_series(results_dir, strat, Ks):
    """[(K, rows_per_s), ...] read straight from the per-run blobs."""
    out = []
    for K in Ks:
        r = load_result(results_dir, f"{strat}_k{K}")
        if r and r.get("throughput_rows_per_s"):
            out.append((K, r["throughput_rows_per_s"]))
    return out


def fig_prefix_cache_throughput(out_dir, results_dir):
    """The sticky hit-rate advantage converts to higher end-to-end throughput.

    The gain mirrors the hit-rate gap as a hump -> it peaks in the mid-K regime
    where sticky's lead is widest and washes out at both ends (small K: nothing
    to win, both fit; large K: both thrashed). The y axis is truncated to make
    the gap legible, so the peak is annotated as a percentage to keep the
    magnitude honest."""
    import matplotlib.pyplot as plt
    Ks = sorted(int(re.search(r"_k(\d+)\.json$", p.name).group(1))
                for p in results_dir.glob("result_sticky_k*.json"))
    if not Ks:
        return
    s = _sweep_throughput_series(results_dir, "sticky", Ks)
    r = _sweep_throughput_series(results_dir, "round_robin", Ks)
    if not (s and r):
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot([K for K, _ in s], [v for _, v in s], "-o", color=COLOR_STICKY,
            linewidth=2.0, markersize=6, label="sticky_by_prefix")
    ax.plot([K for K, _ in r], [v for _, v in r], "-s", color=COLOR_RR,
            linewidth=2.0, markersize=6, label="round_robin")

    # Peak-uplift annotation: vertical double-arrow at the K where sticky's
    # throughput lead is widest, labelled as a percentage so the truncated y
    # axis cannot overstate the magnitude.
    sm = {K: v for K, v in s}
    rm = {K: v for K, v in r}
    common = sorted(set(sm) & set(rm))
    if common:
        kbest = max(common, key=lambda K: sm[K] - rm[K])
        upl = sm[kbest] / rm[kbest] - 1.0
        ax.annotate("", xy=(kbest, sm[kbest]), xytext=(kbest, rm[kbest]),
                    arrowprops=dict(arrowstyle="<->", lw=1.3, color=COLOR_STICKY))
        ax.text(kbest * 1.10, (sm[kbest] + rm[kbest]) / 2, f"+{upl*100:.0f}%",
                ha="left", va="center", fontsize=10, fontweight="bold",
                color=COLOR_STICKY)

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("distinct recurring prefixes  K")
    ax.set_ylabel(_label("pool throughput", "rows/s"))
    allv = [v for _, v in s] + [v for _, v in r]
    pad = (max(allv) - min(allv)) * 0.30
    ax.set_ylim(min(allv) - pad, max(allv) + pad)
    ax.set_title("Throughput vs prefix diversity")
    ax.legend(loc="upper right")
    _suptitle(fig, "sticky_by_prefix's cache reuse lifts pool throughput")
    _save(fig, out_dir, "router_prefix_cache_throughput")
    plt.close(fig)


def fig_prefix_cache_combined(out_dir, results_dir, n):
    """Single two-panel float pairing the prefix-cache MECHANISM (left: pool hit
    rate vs K) with its CONSEQUENCE (right: pool throughput vs K), sharing the K
    axis. Mirrors router_load_aware's cause|consequence layout, and avoids a
    dual-y-axis (the two metrics have incomparable units/scales). The standalone
    router_prefix_cache{,_throughput} figures stay available for slides."""
    import matplotlib.pyplot as plt
    Ks = sorted(int(re.search(r"_k(\d+)\.json$", p.name).group(1))
                for p in results_dir.glob("result_sticky_k*.json"))
    if not Ks:
        return
    s_hit = _sweep_series(results_dir, "sticky", Ks, n)
    r_hit = _sweep_series(results_dir, "round_robin", Ks, n)
    s_tp = _sweep_throughput_series(results_dir, "sticky", Ks)
    r_tp = _sweep_throughput_series(results_dir, "round_robin", Ks)
    if not (s_hit and s_tp):
        return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.0))

    # ---- (left) hit rate: the mechanism ----
    _plot_sweep_line(axA, s_hit, COLOR_STICKY, "-o", "sticky_by_prefix")
    _plot_sweep_line(axA, r_hit, COLOR_RR, "-s", "round_robin")
    sm = {K: v for K, v, *_ in s_hit}
    rm = {K: v for K, v, *_ in r_hit}
    common = sorted(set(sm) & set(rm))
    if common:
        kbest = max(common, key=lambda K: sm[K] - rm[K])
        gap = sm[kbest] - rm[kbest]
        axA.annotate("", xy=(kbest, sm[kbest]), xytext=(kbest, rm[kbest]),
                     arrowprops=dict(arrowstyle="<->", lw=1.3, color=COLOR_STICKY))
        axA.text(kbest * 1.12, (sm[kbest] + rm[kbest]) / 2, f"+{gap*100:.0f} pts",
                 ha="left", va="center", fontsize=9, fontweight="bold",
                 color=COLOR_STICKY)
    axA.set_xscale("log", base=2)
    axA.set_xticks(Ks)
    axA.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    axA.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    axA.set_xlabel("distinct recurring prefixes  K")
    axA.set_ylabel("pool prefix-cache hit rate")
    axA.set_ylim(0, 0.45)
    axA.set_title("Prefix-cache reuse")
    for K, v, *_ in s_hit:
        axA.annotate(f"{v:.0%}", (K, v), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7.5, color=COLOR_STICKY)
    for K, v, *_ in r_hit:
        axA.annotate(f"{v:.0%}", (K, v), textcoords="offset points",
                     xytext=(0, -13), ha="center", fontsize=7.5, color=COLOR_RR)
    axA.legend(loc="upper right")

    # ---- (right) throughput: the consequence ----
    axB.plot([K for K, _ in s_tp], [v for _, v in s_tp], "-o", color=COLOR_STICKY,
             linewidth=2.0, markersize=6, label="sticky_by_prefix")
    axB.plot([K for K, _ in r_tp], [v for _, v in r_tp], "-s", color=COLOR_RR,
             linewidth=2.0, markersize=6, label="round_robin")
    sm2 = {K: v for K, v in s_tp}
    rm2 = {K: v for K, v in r_tp}
    common2 = sorted(set(sm2) & set(rm2))
    if common2:
        kbest2 = max(common2, key=lambda K: sm2[K] - rm2[K])
        upl = sm2[kbest2] / rm2[kbest2] - 1.0
        axB.annotate("", xy=(kbest2, sm2[kbest2]), xytext=(kbest2, rm2[kbest2]),
                     arrowprops=dict(arrowstyle="<->", lw=1.3, color=COLOR_STICKY))
        axB.text(kbest2 * 1.10, (sm2[kbest2] + rm2[kbest2]) / 2, f"+{upl*100:.0f}%",
                 ha="left", va="center", fontsize=10, fontweight="bold",
                 color=COLOR_STICKY)
    axB.set_xscale("log", base=2)
    axB.set_xticks(Ks)
    axB.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    axB.set_xlabel("distinct recurring prefixes  K")
    axB.set_ylabel(_label("pool throughput", "rows/s"))
    allv = [v for _, v in s_tp] + [v for _, v in r_tp]
    pad = (max(allv) - min(allv)) * 0.30
    axB.set_ylim(min(allv) - pad, max(allv) + pad)
    axB.set_title("Throughput payoff")
    axB.legend(loc="upper right")

    _suptitle(fig, "sticky_by_prefix: reuse (left) lifts pool throughput (right)")
    _save(fig, out_dir, "router_prefix_cache_combined")
    plt.close(fig)


def fig_load_aware(out_dir, ll, rr_hetero, n, throttled_ep=0):
    """Fig 3: least_loaded vs round_robin_hetero on the SAME throttled
    fleet -- two views of one experiment:
      (a) placement: where each strategy sends requests when ep0 is throttled;
      (b) tail latency: the consequence of (a).
    The throttle is concurrency (--max-num-seqs), not a per-token slowdown:
    ep0 admits few concurrent requests, so a load-blind stream queues there."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import numpy as np
    if not (ll and rr_hetero):
        return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.0))
    series = [("least_loaded", ll, COLOR_LL), ("round_robin", rr_hetero, COLOR_RR)]

    # ---- (a) placement: grouped by endpoint, two named series ----
    x = np.arange(n)
    width = 0.38
    offsets = (-width / 2, width / 2)
    bar_max = 0
    for (lbl, r, c), off in zip(series, offsets):
        counts = (list(r.get("per_endpoint_count", [])) + [0] * n)[:n]
        bar_max = max(bar_max, max(counts))
        for e in range(n):
            throttled = (e == throttled_ep)
            axA.bar(x[e] + off, counts[e], width, color=c,
                    edgecolor="red" if throttled else "black",
                    linewidth=2.0 if throttled else 0.6)
            axA.annotate(f"{counts[e]}", (x[e] + off, counts[e]),
                         textcoords="offset points", xytext=(0, 4),
                         ha="center", va="bottom", fontsize=11, fontweight="bold")
    # Reference: round_robin's load-blind even split.
    rr_counts = (list(rr_hetero.get("per_endpoint_count", [])) + [0] * n)[:n]
    blind = sum(rr_counts) / n
    axA.axhline(blind, ls=(0, (6, 3)), lw=2.0, color="0.35", zorder=0)
    axA.set_xticks(x)
    axA.set_xticklabels([str(i) for i in range(n)])
    axA.set_xlabel("endpoint")
    axA.set_ylabel("#requests routed per endpoint")
    axA.set_ylim(0, bar_max * 1.9)   # headroom so the legend clears the bars/labels
    axA.set_title("Per-endpoint placement")
    axA.legend(handles=[
        Patch(facecolor=COLOR_LL, edgecolor="black", label="least_loaded"),
        Patch(facecolor=COLOR_RR, edgecolor="black", label="round_robin"),
        Patch(facecolor="white", edgecolor="red", linewidth=2.0,
              label="throttled endpoint"),
        Line2D([0], [0], ls=(0, (6, 3)), lw=2.0, color="0.35",
               label=f"even spread ({int(blind)})"),
    ], loc="upper center", ncol=2, columnspacing=1.2, fontsize=11.5)

    # ---- (b) tail latency: consequence of the placement ----
    _grouped_latency(axB, series)
    axB.set_title("Tail latency")

    _suptitle(fig, "least_loaded routes around the slow node")
    _save(fig, out_dir, "router_load_aware")
    plt.close(fig)


def _grouped_latency(ax, series):
    """Grouped p50/p95/p99 bars for (label, result, color) tuples, in seconds
    on a linear axis."""
    import numpy as np
    metrics = ["p50", "p95", "p99"]
    x = np.arange(len(metrics))
    width = 0.38
    offsets = (np.linspace(-width / 2, width / 2, len(series))
               if len(series) > 1 else [0.0])
    for (lbl, r, c), off in zip(series, offsets):
        lat = r.get("latency_ms", {})
        vals = [lat.get(m, 0.0) / 1000.0 for m in metrics]
        ax.bar(x + off, vals, width, label=lbl, color=c,
               edgecolor="black", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.1f}s", (xi + off, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", va="bottom",
                        fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel(_label("end-to-end latency", "s"))
    ax.legend()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="Directory of result_*.json + metrics_*.txt pulled from Clariden.")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/figures"),
                    help="Where to write the three figures (.png + .pdf each).")
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"results dir not found: {args.results_dir}")

    _setup_style()
    import matplotlib
    matplotlib.use("Agg")

    rd = args.results_dir
    single = load_result(rd, TAG_SINGLE)
    rr = load_result(rd, TAG_RR)
    ll = load_result(rd, TAG_LL)
    rr_hetero = load_result(rd, TAG_RR_HETERO)

    n = _n_endpoints(single, rr, ll)

    fig_throughput(args.out_dir, single, rr, n)
    fig_prefix_cache_sweep(args.out_dir, rd, n)
    fig_prefix_cache_throughput(args.out_dir, rd)
    fig_prefix_cache_combined(args.out_dir, rd, n)
    fig_load_aware(args.out_dir, ll, rr_hetero, n, throttled_ep=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

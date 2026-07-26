#!/usr/bin/env python3
"""
Figures for the thesis chapter "A Performance Study of vLLM Serving" (GH200).

Reads the GH200 sweep tree produced by sembench/scripts/vllm_perf_driver.py
(results/qwen25-7b-instruct-vllm/) plus the morsel/threads sweep results, and
emits the chapter figures in the paper's plot style.

Figures
  vllm_saturation.pdf        E1: throughput/latency vs N | queue + KV state vs N
  vllm_itl_cliff.pdf         E1: ITL distribution per N | stall-band share vs N
  vllm_batching.pdf          E5: rows/s vs R per N | prompt tokens per row vs R
  vllm_prefill_cache.pdf     E2: prefill-bound wall | CacheIso: reuse lever
  flock_morsel_bottleneck.pdf  stock-flock concurrency law + gap
      (from the GH200 morsel/threads sweep re-run when present; otherwise from
       the sem_filter A/B scalar arm as an interim rendering)

Usage
  python3 analysis/plot_vllm_perf_study.py \
      --results-dir ~/sembench/results/qwen25-7b-instruct-vllm \
      --sweep-dir   ~/sembench/results \
      --figures-dir analysis/figures
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_THROUGHPUT = "#1f77b4"   # blue
C_LATENCY = "#d62728"      # red
C_KV = "#9467bd"           # purple
C_CACHE = "#2ca02c"        # green
C_ALT = "#ff7f0e"          # orange
C_GRAY = "#7f7f7f"

STALL_MS = 100.0           # ITL band boundary: the two modes sit at ~10 and ~215 ms


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.loc": "best",
    })


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return [r for r in csv.DictReader(f) if "burn" not in r["label"]]


def fnum(row: dict, key: str) -> float:
    return float(row[key])


def save(fig, figures_dir: Path, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(figures_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"wrote {figures_dir / name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1 — E1 concurrency saturation + server state
# ---------------------------------------------------------------------------

def fig_saturation(results: Path, figures_dir: Path):
    rows = sorted(read_csv(results / "E1" / "E1.csv"),
                  key=lambda r: int(r["n_concurrent"]))
    n = [int(r["n_concurrent"]) for r in rows]
    rps = [fnum(r, "rows_s") for r in rows]
    e2e50 = [fnum(r, "e2e_ms_p50") / 1e3 for r in rows]
    e2e95 = [fnum(r, "e2e_ms_p95") / 1e3 for r in rows]
    kv = [fnum(r, "kv_util_max") for r in rows]
    running = [fnum(r, "running_mean") for r in rows]
    waiting = [fnum(r, "waiting_mean") for r in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    axL.plot(n, rps, "o-", color=C_THROUGHPUT, label="rows/s")
    axL.set_xscale("log", base=2)
    axL.set_xlabel("offered concurrency N (requests in flight) [log]")
    axL.set_ylabel("rows/s", color=C_THROUGHPUT)
    axL.tick_params(axis="y", labelcolor=C_THROUGHPUT)
    axL.axvline(128, color=C_GRAY, ls=":", lw=1)
    axL.annotate("N = 128", xy=(128, axL.get_ylim()[0]), xytext=(140, 520),
                 fontsize=9, color=C_GRAY)
    axLr = axL.twinx()
    axLr.plot(n, e2e50, "s--", color=C_LATENCY, label="e2e p50")
    axLr.plot(n, e2e95, "^--", color=C_LATENCY, alpha=0.45, label="e2e p95")
    axLr.set_yscale("log")
    axLr.set_ylabel("request latency (s) [log]", color=C_LATENCY)
    axLr.tick_params(axis="y", labelcolor=C_LATENCY)
    axLr.spines.top.set_visible(False)
    axLr.grid(False)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axLr.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="center right")
    axL.set_title("[Saturation] throughput flattens, latency keeps rising")

    axR.plot(n, running, "o-", color=C_THROUGHPUT, label="running (mean)")
    axR.plot(n, waiting, "s-", color=C_ALT, label="waiting (mean)")
    axR.set_xscale("log", base=2)
    axR.set_yscale("log")
    axR.set_ylim(0.1, 5e3)  # headroom so the legend sits above the curves
    axR.set_xlabel("offered concurrency N [log]")
    axR.set_ylabel("scheduler queue population [log]")
    axRr = axR.twinx()
    axRr.plot(n, kv, "d--", color=C_KV, label="KV-cache util (max)")
    axRr.set_ylim(0, 1.0)
    axRr.set_ylabel("KV-cache utilisation", color=C_KV)
    axRr.tick_params(axis="y", labelcolor=C_KV)
    axRr.axhline(0.9, color=C_KV, ls=":", lw=1, alpha=0.6)
    axRr.annotate("admission limit", xy=(n[-1], 0.9), xytext=(230, 0.84),
                  fontsize=8, color=C_KV)
    axRr.spines.top.set_visible(False)
    axRr.grid(False)
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = axRr.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="upper left")
    axR.set_title("[Server state] the excess queues; KV admission never binds")

    fig.tight_layout()
    save(fig, figures_dir, "vllm_saturation")


# ---------------------------------------------------------------------------
# Figure 2 — E1 inter-token-latency bimodality
# ---------------------------------------------------------------------------

def load_itls(results: Path, n_val: int) -> np.ndarray:
    out = []
    with open(results / "E1" / f"E1_N{n_val}_R32" / "requests.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("seq", -1) >= 0 and not r.get("error"):
                out.extend(r.get("itls_ms") or [])
    return np.asarray(out)


def fig_itl_cliff(results: Path, figures_dir: Path):
    ns = [16, 32, 64, 128, 256, 512, 1024]
    itls = {n: load_itls(results, n) for n in ns}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    show = [16, 64, 128, 256, 512]
    colors = {16: C_THROUGHPUT, 64: C_CACHE, 128: C_ALT, 256: C_LATENCY,
              512: C_KV}
    for n_val in show:
        xs = np.sort(itls[n_val])
        ys = np.arange(1, len(xs) + 1) / len(xs)
        axL.plot(xs, ys, color=colors[n_val], lw=1.6, label=f"N = {n_val}")
    axL.set_xscale("log")
    axL.set_xlim(4, 400)
    axL.set_ylim(0, 1.02)
    axL.set_xlabel("inter-token latency (ms) [log]")
    axL.set_ylabel("cumulative share of ITLs")
    axL.axvline(STALL_MS, color=C_GRAY, ls=":", lw=1)
    axL.text(0.13, 1.01, "decode band", transform=axL.transAxes, fontsize=9,
             color=C_GRAY)
    axL.text(0.74, 1.01, "stall band", transform=axL.transAxes, fontsize=9,
             color=C_GRAY)
    axL.legend(loc="center left")
    axL.set_title("[ITL distribution] two bands; the plateau is the empty gap")

    frac = [float((itls[n_val] > STALL_MS).mean()) for n_val in ns]
    p50 = [float(np.percentile(itls[n_val], 50)) for n_val in ns]
    axR.plot(ns, frac, "o-", color=C_LATENCY, label=f"share of ITLs > {STALL_MS:.0f} ms")
    axR.set_xscale("log", base=2)
    axR.set_ylim(0, 1)
    axR.set_xlabel("offered concurrency N [log]")
    axR.set_ylabel("stall-band share", color=C_LATENCY)
    axR.tick_params(axis="y", labelcolor=C_LATENCY)
    axR.axvline(128, color=C_GRAY, ls=":", lw=1)
    axRr = axR.twinx()
    axRr.plot(ns, p50, "s--", color=C_KV, label="ITL p50")
    axRr.set_ylabel("ITL p50 (ms)", color=C_KV)
    axRr.tick_params(axis="y", labelcolor=C_KV)
    axRr.spines.top.set_visible(False)
    axRr.grid(False)
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = axRr.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="upper left")
    axR.set_title("[The cliff] the median crosses between N=128 and N=256")

    fig.tight_layout()
    save(fig, figures_dir, "vllm_itl_cliff")


# ---------------------------------------------------------------------------
# Figure 3 — E5 rows-per-prompt amortisation
# ---------------------------------------------------------------------------

def fig_batching(results: Path, figures_dir: Path):
    rows = read_csv(results / "E5" / "E5.csv")
    ns = sorted({int(r["n_concurrent"]) for r in rows})
    rs = sorted({int(r["rows_per_request"]) for r in rows})

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    shades = {16: 0.35, 64: 0.55, 128: 0.8, 256: 1.0, 1024: 1.0}
    for n_val in ns:
        if n_val == 1024:
            continue  # duplicates N=256 at every R; keeps the panel readable
        sub = sorted((r for r in rows if int(r["n_concurrent"]) == n_val),
                     key=lambda r: int(r["rows_per_request"]))
        axL.plot([int(r["rows_per_request"]) for r in sub],
                 [fnum(r, "rows_s") for r in sub],
                 "o-", color=C_THROUGHPUT, alpha=shades.get(n_val, 1.0),
                 label=f"N = {n_val}")
    axL.set_xscale("log", base=2)
    axL.set_xlabel("rows per prompt R [log]")
    axL.set_ylabel("rows/s")
    axL.legend()
    axL.set_title("[Row packing] R amortises prefill; flat past R = 32")

    # per-row prompt-token cost (identical across N; take N=128)
    sub = sorted((r for r in rows if int(r["n_concurrent"]) == 128),
                 key=lambda r: int(r["rows_per_request"]))
    r_vals = [int(r["rows_per_request"]) for r in sub]
    tok_row = [fnum(r, "prompt_tok_p50") / int(r["rows_per_request"]) for r in sub]
    e2e = [fnum(r, "e2e_ms_p50") / 1e3 for r in sub]
    axR.plot(r_vals, tok_row, "o-", color=C_THROUGHPUT, label="prompt tokens / row")
    axR.set_xscale("log", base=2)
    axR.set_xlabel("rows per prompt R [log]")
    axR.set_ylabel("prompt tokens per row", color=C_THROUGHPUT)
    axR.tick_params(axis="y", labelcolor=C_THROUGHPUT)
    axR.axhline(tok_row[-1], color=C_GRAY, ls=":", lw=1)
    axR.annotate(f"payload floor ≈ {tok_row[-1]:.0f} tok/row",
                 xy=(r_vals[0], tok_row[-1]), xytext=(1.2, tok_row[-1] + 4),
                 fontsize=9, color=C_GRAY)
    axRr = axR.twinx()
    axRr.plot(r_vals, e2e, "s--", color=C_LATENCY, label="e2e p50 (N = 128)")
    axRr.set_yscale("log")
    axRr.set_ylabel("request latency (s) [log]", color=C_LATENCY)
    axRr.tick_params(axis="y", labelcolor=C_LATENCY)
    axRr.spines.top.set_visible(False)
    axRr.grid(False)
    h1, l1 = axR.get_legend_handles_labels()
    h2, l2 = axRr.get_legend_handles_labels()
    axR.legend(h1 + h2, l1 + l2, loc="center right")
    axR.set_title("[The trade] template amortised by R = 32; latency keeps growing")

    fig.tight_layout()
    save(fig, figures_dir, "vllm_batching")


# ---------------------------------------------------------------------------
# Figure 4 — E2 prefill wall + CacheIso prefix-cache lever
# ---------------------------------------------------------------------------

def computed_tok_s(run_dir: Path) -> tuple[float, float, float]:
    """Return (raw_tok_s, computed_tok_s, hit_share) for one driver run."""
    summ = json.load(open(run_dir / "summary.json"))
    pt = ct = 0
    with open(run_dir / "requests.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("seq", -1) >= 0 and not r.get("error"):
                pt += r["prompt_tokens"]
                ct += r["completion_tokens"]
    lines = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
    hits = 0
    if lines and lines[0].get("prefix_cache_hits") is not None:
        hits = lines[-1]["prefix_cache_hits"] - lines[0]["prefix_cache_hits"]
    w = summ["wall_clock_s"]
    return (pt + ct) / w, (pt - hits + ct) / w, (hits / pt if pt else 0.0)


def fig_prefill_cache(results: Path, figures_dir: Path):
    rows = sorted(read_csv(results / "E2" / "E2.csv"),
                  key=lambda r: fnum(r, "prompt_tok_p50"))
    ptok = [fnum(r, "prompt_tok_p50") for r in rows]
    rps = [fnum(r, "rows_s") for r in rows]
    tok = [fnum(r, "tok_s_total") / 1e3 for r in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    axL.plot(ptok, rps, "o-", color=C_THROUGHPUT, label="rows/s")
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel("prompt length (tokens, p50) [log]")
    axL.set_ylabel("rows/s [log]", color=C_THROUGHPUT)
    axL.tick_params(axis="y", labelcolor=C_THROUGHPUT)
    axLr = axL.twinx()
    axLr.plot(ptok, tok, "s--", color=C_KV, label="total tok/s")
    axLr.set_ylim(0, 42)
    axLr.set_ylabel("prefill+decode throughput (k tok/s)", color=C_KV)
    axLr.tick_params(axis="y", labelcolor=C_KV)
    axLr.spines.top.set_visible(False)
    axLr.grid(False)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axLr.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="lower left")
    axL.set_title("[Prefill wall] token rate saturates; every prompt token costs rows/s")

    # CacheIso (free-form arms, N=64, identical real prompts): cache off /
    # cache on cold / cache on warm replay.
    cells = [
        ("cache off", results / "CacheIso_NoCache" / "CacheIsoNoCache_N64_none"),
        ("cache on,\ncold", results / "CacheIso" / "CacheIso_N64_none"),
        ("cache on,\nwarm replay", results / "CacheIso_Reversed" / "CacheIsoRev_N64_none"),
    ]
    labels, rps_bars, comp_bars, hit_shares = [], [], [], []
    for label, run_dir in cells:
        summ = json.load(open(run_dir / "summary.json"))
        raw, comp, hit = computed_tok_s(run_dir)
        labels.append(label)
        rps_bars.append(summ["throughput_rows_per_s"])
        comp_bars.append(comp / 1e3)
        hit_shares.append(hit)

    x = np.arange(len(labels))
    bars = axR.bar(x, rps_bars, width=0.55, color=C_CACHE, edgecolor="black",
                   linewidth=0.6)
    for xi, b, hit, comp in zip(x, bars, hit_shares, comp_bars):
        axR.annotate(f"{b.get_height():.0f} rows/s\ncomputed {comp:.1f}k tok/s",
                     xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        axR.annotate(f"hit {hit * 100:.0f}%",
                     xy=(b.get_x() + b.get_width() / 2, b.get_height() / 2),
                     ha="center", va="center", fontsize=9, color="white",
                     fontweight="bold")
    axR.set_ylim(0, 4300)  # headroom for the two-line bar annotations
    axR.set_xticks(x, labels)
    axR.set_ylabel("rows/s")
    axR.set_title("[Prefix-cache lever] reuse converts skipped prefill into rows/s")

    fig.tight_layout()
    save(fig, figures_dir, "vllm_prefill_cache")


# ---------------------------------------------------------------------------
# Figure 5 — stock flock: concurrency = min(morsels, threads)
# ---------------------------------------------------------------------------

def peak_running(sampler_csv: Path) -> float | None:
    peak = None
    with open(sampler_csv) as f:
        for row in csv.DictReader(f):
            v = row.get("running")
            if v not in (None, "", "None"):
                v = float(v)
                peak = v if peak is None else max(peak, v)
    return peak


def load_sweep_cells(root: Path, pattern: str) -> list[dict]:
    cells = []
    for d in sorted(glob.glob(str(root / pattern))):
        d = Path(d)
        wj = d / "worker.json"
        sc = d / "sampler.csv"
        if not wj.exists():
            continue
        w = json.load(open(wj))
        w["peak_running"] = peak_running(sc) if sc.exists() else None
        w["cell"] = d.name
        cells.append(w)
    return cells


def fig_morsel_bottleneck(sweep_root: Path, ab_csv: Path, figures_dir: Path):
    """Prefer the GH200 morsel/threads sweep re-run; fall back to the
    sem_filter A/B scalar arm (interim rendering) when it is absent."""
    ms_roots = sorted(glob.glob(str(sweep_root / "morsel_sweep" / "gh200_*")))
    ms_roots = [r for r in ms_roots if not r.endswith("_burn")]
    ts1 = sorted(glob.glob(str(sweep_root / "threads_sweep" / "gh200_*_morsels1")))
    ts8 = sorted(glob.glob(str(sweep_root / "threads_sweep" / "gh200_*_morsels8")))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    if ms_roots and ts1 and ts8:
        # ---- real GH200 sweep data: aggregate every job/rep per config ----
        # threads sweeps: series per morsel count, x = SET threads
        ts_points: dict[int, dict[int, list]] = {}
        for root in ts1 + ts8:
            morsels = int(root.rsplit("_morsels", 1)[1])
            for c in load_sweep_cells(Path(root), "threads*"):
                (ts_points.setdefault(morsels, {})
                 .setdefault(c["threads_setting"], []).append(c["peak_running"]))
        colors = {1: C_THROUGHPUT, 8: C_ALT}
        max_thr = max(t for pts in ts_points.values() for t in pts)
        thr_axis = sorted({t for pts in ts_points.values() for t in pts})
        for morsels in sorted(ts_points):
            pts = ts_points[morsels]
            ts = sorted(pts)
            axL.plot(ts, [np.mean(pts[t]) for t in ts], "o-",
                     color=colors.get(morsels, C_KV),
                     label=f"morsels = {morsels}")
            axL.plot(thr_axis, [min(morsels, t) for t in thr_axis], ":",
                     color=colors.get(morsels, C_KV), alpha=0.5,
                     label=f"min({morsels}, threads)")
        axL.set_xscale("log", base=2)
        axL.set_yscale("log", base=2)
        axL.set_xlabel("SET threads [log]")
        axL.set_ylabel("peak in-flight requests at vLLM [log]")
        axL.legend()
        axL.set_title("[Threads sweep] SET threads is capped by the morsel count")

        # morsel sweeps: series per thread setting, x = morsel count
        ms_points: dict[int, dict[int, list]] = {}
        for root in ms_roots:
            for c in load_sweep_cells(Path(root), "rep*_N*"):
                (ms_points.setdefault(c["threads_setting"], {})
                 .setdefault(c["n_files"], []).append(c["peak_running"]))
        colors_ms = {8: C_ALT, 128: C_LATENCY}
        n_axis = sorted({n for pts in ms_points.values() for n in pts})
        for threads in sorted(ms_points):
            pts = ms_points[threads]
            ns = sorted(pts)
            axR.plot(ns, [np.mean(pts[n]) for n in ns], "o-",
                     color=colors_ms.get(threads, C_KV),
                     label=f"threads = {threads}")
            axR.plot(n_axis, [min(n, threads) for n in n_axis], ":",
                     color=colors_ms.get(threads, C_KV), alpha=0.5,
                     label=f"min(morsels, {threads})")
        axR.set_xscale("log", base=2)
        axR.set_yscale("log", base=2)
        axR.set_xlabel("morsels (2048-row row groups) [log]")
        axR.set_ylabel("peak in-flight requests at vLLM [log]")
        axR.legend()
        axR.set_title("[Morsel sweep] concurrency = min(morsels, threads)")
    else:
        # ---- interim: sem_filter A/B scalar arm (GH200) ----
        print("NOTE: GH200 morsel/threads sweep not found under "
              f"{sweep_root} — rendering interim A/B-based figure")
        with open(ab_csv) as f:
            ab = list(csv.DictReader(f))
        thr_cells = sorted(
            (r for r in ab if r["arm"] == "scalar"
             and r["config"] == "1morsel_1datachunk_16threads"),
            key=lambda r: int(r["threads"]))
        axL.plot([int(r["threads"]) for r in thr_cells],
                 [float(r["concurrency"]) for r in thr_cells],
                 "o-", color=C_THROUGHPUT, label="measured in-flight (1 morsel)")
        axL.set_xscale("log", base=2)
        axL.set_ylim(0, 8)
        axL.set_xlabel("SET threads [log]")
        axL.set_ylabel("mean in-flight requests at vLLM")
        axL.legend()
        axL.set_title("[Threads sweep] one morsel pins stock flock at one request")

        morsel_cells = [r for r in ab if r["arm"] == "scalar" and r["R"] == "1"]
        morsel_cells.sort(key=lambda r: int(r["morsels"]))
        conc = [float(r["concurrency"]) for r in morsel_cells]
        lim = [min(int(r["morsels"]), int(r["threads"])) for r in morsel_cells]
        axR.plot(lim, conc, "o-", color=C_THROUGHPUT, label="measured in-flight")
        axR.plot(lim, lim, ":", color=C_GRAY, label="min(morsels, threads)")
        axR.set_xscale("log", base=2)
        axR.set_yscale("log", base=2)
        axR.set_xlabel("min(morsels, threads) [log]")
        axR.set_ylabel("mean in-flight requests at vLLM [log]")
        axR.legend()
        axR.set_title("[Morsel sweep] concurrency = min(morsels, threads)")

    fig.tight_layout()
    save(fig, figures_dir, "flock_morsel_bottleneck")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="GH200 tree: .../results/qwen25-7b-instruct-vllm")
    ap.add_argument("--sweep-dir", type=Path, required=True,
                    help=".../results (holds morsel_sweep/, threads_sweep/)")
    ap.add_argument("--ab-csv", type=Path,
                    default=Path(__file__).parent / "figures" / "data"
                    / "sem_filter_ab" / "morsel_ab_summary.csv")
    ap.add_argument("--figures-dir", type=Path,
                    default=Path(__file__).parent / "figures")
    args = ap.parse_args()

    setup_style()
    os.makedirs(args.figures_dir, exist_ok=True)
    fig_saturation(args.results_dir, args.figures_dir)
    fig_itl_cliff(args.results_dir, args.figures_dir)
    fig_batching(args.results_dir, args.figures_dir)
    fig_prefill_cache(args.results_dir, args.figures_dir)
    fig_morsel_bottleneck(args.sweep_dir, args.ab_csv, args.figures_dir)


if __name__ == "__main__":
    main()

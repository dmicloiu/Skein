#!/usr/bin/env python3
"""[DESIGNED FOR SEM FILTER THREAD SWEEP A/B EXPERIMENT ANALYSIS]
sem_filter operator A/B: PhysicalSemFilter (semantic_rewrite_enabled=true) vs
the scalar llm_filter, from the data produced on Clariden by
analysis/slurm/sem_filter_ab_clariden.sh.

Renders TWO paper figures (png + pdf each), in the house style of
plot_router_experiment.py. NO run configuration is baked into the images -- the
model / GPU / cap / batch / temperature / scale belong in the LaTeX caption.

  sem_filter_threads    line: throughput (rows/s) vs threads {1..16}, operator
                        vs scalar -- operator flat (thread-independent), scalar
                        pinned at the single-chunk serial floor.
  sem_filter_tradeoff   1x3 grouped bars at one representative thread: what the
                        operator trades -- (a) per-request latency (higher), for
                        (b) throughput (higher), at (c) equal quality (survivors
                        agree). Per-request latency is the mean vLLM
                        e2e_request_latency over the run window.

Inputs (in --results-dir):
  result_<arm>_t<T>.json                      per-run blob from the driver
  metrics_{before,after}_<arm>_t<T>_ep0.txt   raw vLLM /metrics snapshots

Usage:
  python analysis/plot_sem_filter_ab.py \
      --results-dir analysis/figures/data/sem_filter_ab --out-dir analysis/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mirror plot_router_experiment.py: blue = baseline, green = behaviour under test.
COLOR_SCALAR = "blue"      # scalar llm_filter (rewrite off) is the baseline
COLOR_OPERATOR = "green"   # PhysicalSemFilter (rewrite on) is under test

THREADS = [1, 2, 4, 8, 16]
TRADEOFF_THREAD = 1        # representative thread for the bar figure (sweep is flat)


def _setup_style():
    """Consistent styling, copied from plot_router_experiment.py."""
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
        "hatch.linewidth": 1.1,
    })


def _suptitle(fig, text: str):
    """Figure-level claim, router style. Component tag only -- no run config
    (model/GPU/cap/batch/scale live in the LaTeX caption)."""
    fig.suptitle(f"[PhysicalSemFilter]  {text}",
                 fontweight="bold", fontsize=12.5, x=0.02, y=1.02, ha="left")


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"wrote {path}", file=sys.stderr)


def load_result(results_dir: Path, arm: str, t: int) -> dict:
    with open(results_dir / f"result_{arm}_t{t}.json") as f:
        return json.load(f)


def _metric(path: Path, key: str) -> float:
    for line in open(path):
        if line.startswith(key):
            return float(line.split()[-1])
    return 0.0


def metric_delta(results_dir: Path, arm: str, t: int, key: str) -> float:
    """after-minus-before delta of a vLLM /metrics counter for one run."""
    a = _metric(results_dir / f"metrics_after_{arm}_t{t}_ep0.txt", key)
    b = _metric(results_dir / f"metrics_before_{arm}_t{t}_ep0.txt", key)
    return a - b


def mean_request_latency(results_dir: Path, arm: str, t: int) -> float:
    """Mean per-request vLLM e2e latency over the run window (sum / count)."""
    s = metric_delta(results_dir, arm, t, "vllm:e2e_request_latency_seconds_sum")
    n = metric_delta(results_dir, arm, t, "vllm:e2e_request_latency_seconds_count")
    return s / n if n else 0.0


def collect(results_dir: Path) -> dict:
    out: dict = {}
    for arm in ("operator", "scalar"):
        rps, passes, elapsed, lat = [], [], [], []
        for t in THREADS:
            r = load_result(results_dir, arm, t)
            rps.append(r["rows_per_s"])
            passes.append(r["passes"])
            elapsed.append(r["elapsed_s"])
            lat.append(mean_request_latency(results_dir, arm, t))
        out[arm] = dict(rps=rps, passes=passes, elapsed=elapsed, lat=lat,
                        rows=load_result(results_dir, arm, THREADS[0])["rows"])
    return out


# ---------------------------------------------------------------------------
# Figure 1: throughput vs threads (line)
# ---------------------------------------------------------------------------
def render_threads(data: dict, out_dir: Path) -> None:
    op, sc = data["operator"], data["scalar"]
    speedups = [op["rps"][i] / sc["rps"][i] for i in range(len(THREADS))]
    mean_speedup = sum(speedups) / len(speedups)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(THREADS, op["rps"], "-o", color=COLOR_OPERATOR, lw=2.4, ms=8,
            label="operator (PhysicalSemFilter, async)")
    ax.plot(THREADS, sc["rps"], "-s", color=COLOR_SCALAR, lw=2.4, ms=8,
            label="scalar llm_filter (stock flock)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(THREADS)
    ax.set_xticklabels([str(t) for t in THREADS])
    ax.set_xlabel("DuckDB threads")
    ax.set_ylabel("throughput (rows/s)")
    ax.set_ylim(0, max(op["rps"]) * 1.25)
    ax.legend(loc="center left")
    _suptitle(fig, "operator throughput is independent of DuckDB threads")

    # Vertical double-arrow marking the (thread-independent) gap, at a mid thread.
    xmid = THREADS[2]
    ax.annotate("", xy=(xmid, op["rps"][2]), xytext=(xmid, sc["rps"][2]),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.6))
    ax.text(xmid * 1.08, (op["rps"][2] + sc["rps"][2]) / 2,
            f"{mean_speedup:.1f}x", ha="left", va="center",
            fontsize=13, fontweight="bold", color="0.2")

    _save(fig, out_dir, "sem_filter_threads")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: the operator's trade -> latency vs throughput at equal quality
# fixed experiment at a representative thread (TRADEOFF_THREAD)
# ---------------------------------------------------------------------------
def _pair_bars(ax, sc_v, op_v, ylabel, title, fmt="{:,.0f}", top_pad=1.30):
    bars = ax.bar(["scalar", "operator"], [sc_v, op_v], width=0.6,
                  color=[COLOR_SCALAR, COLOR_OPERATOR],
                  edgecolor="black", linewidth=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(sc_v, op_v) * top_pad)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    return bars


def render_tradeoff(data: dict, out_dir: Path) -> None:
    op, sc = data["operator"], data["scalar"]
    i = THREADS.index(TRADEOFF_THREAD)
    rows = op["rows"]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.5, 4.5))
    _pair_bars(a, sc["lat"][i], op["lat"][i],
               "mean per-request latency (s)", "(a) per-request latency",
               fmt="{:.2f}", top_pad=1.32)
    _pair_bars(b, sc["rps"][i], op["rps"][i],
               "throughput (rows/s)", "(b) throughput", top_pad=1.30)
    _pair_bars(c, sc["passes"][i], op["passes"][i],
               "survivors (passing rows)", "(c) survivors", top_pad=1.30)
    # Quality panel: emphasise agreement, not the raw count.
    delta = abs(op["passes"][i] - sc["passes"][i])
    c.text(0.5, max(sc["passes"][i], op["passes"][i]) * 1.16,
           f"agree to {delta} rows ({delta / rows * 100:.2f}%)",
           ha="center", va="center", fontsize=10, color="0.3")

    _suptitle(fig, "the operator trades per-request latency for throughput, at equal quality")
    _save(fig, out_dir, "sem_filter_tradeoff")
    plt.close(fig)


def print_summary(data: dict) -> None:
    op, sc = data["operator"], data["scalar"]
    print("\n==================== sem_filter A/B summary ====================")
    print(f"{'threads':>7} | {'op rows/s':>10} {'sc rows/s':>10} {'speedup':>8} | "
          f"{'op pass':>7} {'sc pass':>7} | {'op lat':>7} {'sc lat':>7}")
    for i, t in enumerate(THREADS):
        sp = op["rps"][i] / sc["rps"][i]
        print(f"{t:>7} | {op['rps'][i]:>10.1f} {sc['rps'][i]:>10.1f} {sp:>7.2f}x | "
              f"{op['passes'][i]:>7} {sc['passes'][i]:>7} | "
              f"{op['lat'][i]:>6.2f}s {sc['lat'][i]:>6.2f}s")
    rows = op["rows"]
    print(f"\nrows={rows} (single DataChunk)  pass-rate: "
          f"operator {op['passes'][0]/rows*100:.1f}%  scalar {sc['passes'][0]/rows*100:.1f}%  "
          f"(delta {abs(op['passes'][0]-sc['passes'][0])} rows = "
          f"{abs(op['passes'][0]-sc['passes'][0])/rows*100:.2f}%)")
    print(f"per-request e2e latency @t{TRADEOFF_THREAD}: "
          f"operator {op['lat'][0]:.2f}s  scalar {sc['lat'][0]:.2f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("analysis/figures/data/sem_filter_ab"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/figures"))
    args = ap.parse_args()
    if not args.results_dir.exists():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1
    _setup_style()
    data = collect(args.results_dir)
    print_summary(data)
    render_threads(data, args.out_dir)
    render_tradeoff(data, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

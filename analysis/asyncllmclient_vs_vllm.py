#!/usr/bin/env python3
"""Compare AsyncLLMClient against the sembench vLLM performance analysis on the
same workload (E1 cell). Produces a two-panel figure:

  (left)  throughput in rows/s
  (right) per-request end-to-end latency, p50 / p95 / p99

Inputs:
  --vllm-analysis-summary  path to summary.json produced by sembench's
                           scripts/vllm_perf_driver.py
  --asyncllmclient-log     path to the AsyncLLMClient integration-test stdout
                           capture; the MAIN line is parsed for stats
  --out                    output figure path (writes both .png and .pdf
                           alongside, matching sembench/scripts/vllm_perf_plot
                           convention)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# Okabe-Ito palette — colorblind-friendly with strong saturation and contrast.
# Standard choice for two-series comparison figures.
COLOR_REFERENCE = "blue"   # red — vLLM performance reference
COLOR_CLIENT    = "green"  # green — AsyncLLMClient results


def _setup_style():
    """Apply the sembench/vllm_perf_plot rcParam style: bold left-aligned
    titles, frameless legend, top/right spines off, subtle grid.
    """
    import matplotlib.pyplot as plt
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


def _label(name: str, unit: str | None = None) -> str:
    return f"{name} ({unit})" if unit else name


def parse_integration_log(path: Path) -> dict:
    """Pull the final ``MAIN ...`` key=value line out of a captured run log."""
    pat = re.compile(r"^MAIN\s+(.*)$", re.MULTILINE)
    m = pat.search(path.read_text())
    if not m:
        raise SystemExit(f"no MAIN line found in {path}")
    out: dict[str, float | str] = {}
    for tok in m.group(1).split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def load_vllm_analysis_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-analysis-summary", required=True, type=Path)
    ap.add_argument("--asyncllmclient-log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="Output path. The script writes both <out>.png and "
                         "<out>.pdf next to it.")
    ap.add_argument("--suptitle",
                    default="[Network Layer]  AsyncLLMClient vs vLLM "
                            "Performance Reference  (N=128, R=32, "
                            "Qwen2.5-7B on DGX Spark)")
    args = ap.parse_args()

    _setup_style()
    import matplotlib.pyplot as plt
    import numpy as np

    ref = load_vllm_analysis_summary(args.vllm_analysis_summary)
    cli = parse_integration_log(args.asyncllmclient_log)

    ref_rows = float(ref["throughput_rows_per_s"])
    cli_rows = float(cli["throughput_rows_per_s"])

    ref_e2e = ref["e2e_ms"]
    ref_p = [ref_e2e["p50"], ref_e2e["p95"], ref_e2e["p99"]]
    cli_p = [cli["p50_latency_ms"], cli["p95_latency_ms"], cli["p99_latency_ms"]]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- left: throughput bar ----
    tools = ["vLLM performance\nanalysis", "AsyncLLMClient"]
    rows = [ref_rows, cli_rows]
    bars = axL.bar(tools, rows, color=[COLOR_REFERENCE, COLOR_CLIENT],
                   width=0.5, edgecolor="black", linewidth=0.6)
    axL.set_ylabel(_label("throughput", "rows/s"))
    axL.set_title("Throughput on vLLM workload")
    axL.set_ylim(0, max(rows) * 1.25)
    for b, v in zip(bars, rows):
        axL.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    delta_pct = (cli_rows - ref_rows) / ref_rows * 100.0
    sign = "+" if delta_pct >= 0 else ""
    axL.text(0.5, 0.92, f"{sign}{delta_pct:.1f}% vs reference",
             transform=axL.transAxes, ha="center", fontsize=10,
             color=COLOR_CLIENT if delta_pct >= 0 else "#d62728",
             fontweight="bold")

    # ---- right: latency quantiles grouped bar ----
    labels = ["p50", "p95", "p99"]
    x = np.arange(len(labels))
    width = 0.38
    axR.bar(x - width / 2, ref_p, width=width, color=COLOR_REFERENCE,
            edgecolor="black", linewidth=0.6,
            label="vLLM performance analysis")
    axR.bar(x + width / 2, cli_p, width=width, color=COLOR_CLIENT,
            edgecolor="black", linewidth=0.6,
            label="AsyncLLMClient")
    axR.set_xticks(x)
    axR.set_xticklabels(labels)
    axR.set_ylabel(_label("end-to-end latency", "ms"))
    axR.set_title("Per-request latency distribution")
    axR.legend()

    for xi, (a, b) in enumerate(zip(ref_p, cli_p)):
        axR.text(xi - width / 2, a, f"{a / 1000:.1f}s",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
        axR.text(xi + width / 2, b, f"{b / 1000:.1f}s",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(args.suptitle, fontweight="bold", fontsize=11,
                 x=0.02, y=1.02, ha="left")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base = out.with_suffix("")
    for ext in ("png", "pdf"):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path)
        print(f"wrote {path}", file=sys.stderr)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

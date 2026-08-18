#!/usr/bin/env python3
"""Compare AsyncLLMClient against the sembench vLLM performance analysis on the
same workload (E1 cell). Produces a two-panel figure:

  (left)  throughput in rows/s
  (right) per-request end-to-end latency, p50 / p95 / p99

Both flags accept one or more paths. If N >= 2 are passed on each side, bars
show the mean across pairs and error bars show the sample standard deviation
(ddof=1). If a single path is passed on each side, the figure is rendered
without error bars.

Inputs:
  --vllm-analysis-summary  one or more summary.json files from sembench's
                           scripts/vllm_perf_driver.py
  --asyncllmclient-log     one or more AsyncLLMClient integration-test stdout
                           captures; the MAIN line is parsed for stats
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

import thesis_style as ts


COLOR_REFERENCE = ts.BLUE    # blue  — vLLM performance reference (baseline)
COLOR_CLIENT    = ts.GREEN   # green — AsyncLLMClient results (under test)


def _setup_style():
    """Apply the shared house rcParams (10 pt dense scale)."""
    ts.setup_style(base_font=14)


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


def _aggregate(values: list[float]) -> tuple[float, float]:
    """Return (mean, sample stdev). Stdev is 0 for N < 2 — we draw no error bar
    in that case so the value is just a placeholder."""
    import numpy as np
    if len(values) < 2:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-analysis-summary", required=True, type=Path,
                    nargs="+",
                    help="One or more sembench driver summary.json files.")
    ap.add_argument("--asyncllmclient-log", required=True, type=Path,
                    nargs="+",
                    help="One or more AsyncLLMClient integration-test stdout "
                         "captures (MAIN line is parsed).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output path. The script writes both <out>.png and "
                         "<out>.pdf next to it.")
    args = ap.parse_args()

    _setup_style()
    import matplotlib.pyplot as plt
    import numpy as np

    refs = [load_vllm_analysis_summary(p) for p in args.vllm_analysis_summary]
    clis = [parse_integration_log(p) for p in args.asyncllmclient_log]
    n_ref, n_cli = len(refs), len(clis)

    # Throughput across runs.
    ref_rows_all = [float(r["throughput_rows_per_s"]) for r in refs]
    cli_rows_all = [float(c["throughput_rows_per_s"]) for c in clis]
    ref_rows_mean, ref_rows_sd = _aggregate(ref_rows_all)
    cli_rows_mean, cli_rows_sd = _aggregate(cli_rows_all)

    # Latency quantiles across runs.
    ref_p_all = np.array([[r["e2e_ms"]["p50"], r["e2e_ms"]["p95"],
                           r["e2e_ms"]["p99"]] for r in refs])
    cli_p_all = np.array([[c["p50_latency_ms"], c["p95_latency_ms"],
                           c["p99_latency_ms"]] for c in clis])
    ref_p_mean = ref_p_all.mean(axis=0)
    cli_p_mean = cli_p_all.mean(axis=0)
    ref_p_sd = ref_p_all.std(axis=0, ddof=1) if n_ref > 1 else np.zeros(3)
    cli_p_sd = cli_p_all.std(axis=0, ddof=1) if n_cli > 1 else np.zeros(3)

    have_errorbars = n_ref >= 2 and n_cli >= 2

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- left: throughput bar ----
    tools = ["vLLM performance\nanalysis", "AsyncLLMClient"]
    rows_mean = [ref_rows_mean, cli_rows_mean]
    rows_sd = [ref_rows_sd, cli_rows_sd]
    bar_kwargs = dict(color=[COLOR_REFERENCE, COLOR_CLIENT],
                      width=0.5, edgecolor="black", linewidth=0.6)
    if have_errorbars:
        bar_kwargs["yerr"] = rows_sd
        bar_kwargs["capsize"] = 6
        bar_kwargs["ecolor"] = "black"
        bar_kwargs["error_kw"] = dict(elinewidth=1.2)
    bars = axL.bar(tools, rows_mean, **bar_kwargs)
    axL.set_ylabel(ts.axis_label("throughput", "rows/s"))
    axL.set_title("Throughput")
    ts.set_panel_marker(axL, 0)
    axL.set_ylim(0, (max(rows_mean) + max(rows_sd)) * 1.30)
    for b, v, sd in zip(bars, rows_mean, rows_sd):
        if have_errorbars:
            label = f"{v:.1f} ± {sd:.1f}"
            y = v + sd
        else:
            label = f"{v:.1f}"
            y = v
        axL.text(b.get_x() + b.get_width() / 2, y, label,
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
    delta_pct = (cli_rows_mean - ref_rows_mean) / ref_rows_mean * 100.0
    sign = "+" if delta_pct >= 0 else ""
    axL.text(0.5, 0.94, f"{sign}{delta_pct:.1f}% vs reference",
             transform=axL.transAxes, ha="center", fontsize=14,
             color=COLOR_CLIENT if delta_pct >= 0 else ts.GRAY_DARKER,
             fontweight="bold")

    # ---- right: latency quantiles grouped bar ----
    labels = ["p50", "p95", "p99"]
    x = np.arange(len(labels))
    width = 0.38
    ref_kw = dict(color=COLOR_REFERENCE, edgecolor="black",
                  linewidth=0.6, label="vLLM performance analysis")
    cli_kw = dict(color=COLOR_CLIENT, edgecolor="black",
                  linewidth=0.6, label="AsyncLLMClient")
    if have_errorbars:
        for kw, sd in ((ref_kw, ref_p_sd), (cli_kw, cli_p_sd)):
            kw["yerr"] = sd
            kw["capsize"] = 4
            kw["ecolor"] = "black"
            kw["error_kw"] = dict(elinewidth=1.0)
    axR.bar(x - width / 2, ref_p_mean, width=width, **ref_kw)
    axR.bar(x + width / 2, cli_p_mean, width=width, **cli_kw)
    axR.set_xticks(x)
    axR.set_xticklabels(labels)
    axR.set_ylabel(ts.axis_label("end-to-end latency", "ms"))
    axR.set_title("Per-request latency distribution")
    ts.set_panel_marker(axR, 1)
    axR.legend()

    for xi in range(len(labels)):
        rv = ref_p_mean[xi]; rs = ref_p_sd[xi]
        cv = cli_p_mean[xi]; cs = cli_p_sd[xi]
        axR.text(xi - width / 2, rv + rs, f"{rv / 1000:.1f}s",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")
        axR.text(xi + width / 2, cv + cs, f"{cv / 1000:.1f}s",
                 ha="center", va="bottom", fontsize=12, fontweight="bold")

    ts.no_suptitle(fig)
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

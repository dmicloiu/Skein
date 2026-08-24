#!/usr/bin/env python3
"""Sema vs Skein under data-parallel (DP) scale-out: an UNCONFOUNDED view.

Single-panel thesis figure (fig:eval-sema-dp). Under DP scale-out with N
independent single-GPU vLLM replicas, the peer system Sema takes a single
``llm_url`` and has no replica awareness, so it sends EVERY request to replica 0
and ZERO to the others (verified 9/9 reps at N=1/2/4). Skein's EndpointRouter
spreads the load evenly, which is the mechanism behind its near-linear DP
throughput scaling (per-GPU efficiency ~1.0; ~1.0 -> 2.09 -> 4.17x over N=1/2/4).

We deliberately plot per-replica request SHARE, not wall-clock rows/s: Sema is an
x86-only binary driving the fleet off-node, so its rows/s is confounded
(``rows_per_s_confounded: true``). Request counts are not.

The figure shows the maximum-scale-out case (N=4, four GPUs): Sema pins 100% of
requests on replica 0 and idles three GPUs; Skein hits an even 25% per replica.

Sema counts come from each replica's vLLM ``/metrics`` before/after snapshots
(``request_success_total`` deltas); Skein counts from ``dp_balance.csv``. No
argumentative suptitle / caption is baked in -- run context lives in the LaTeX.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import thesis_style as ts  # noqa: E402

SEMA_DIR = HERE / "figures" / "data" / "dp_scaling_sema_json"
SKEIN_DIR = HERE / "figures" / "data" / "dp_scaling_json"
OUT_DIR = HERE / "figures"

# Sema = peer system -> a distinct violet: NOT blue (Flock, paper-wide), NOT
# green (Skein), NOT orange (Palimpzest). Skein = system under test -> house green.
SEMA_COLOR = "#6a3d9a"          # deep, saturated violet
SKEIN_COLOR = ts.COLOR_UNDER_TEST
EVEN_LINE_COLOR = "#c1272d"     # reddish/crimson reference line


# ---------------------------------------------------------------------------
# Data loading (from the real per-replica artefacts, both systems).
# ---------------------------------------------------------------------------
def _sema_stop_total(path: Path) -> float | None:
    """Sum vllm:request_success_total over all finish reasons in a /metrics snapshot."""
    if not path.exists():
        return None
    total = 0.0
    pat = re.compile(
        r'vllm:request_success_total\{.*finished_reason="[^"]+".*\}\s+([0-9.]+)'
    )
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            total += float(m.group(1))
    return total


def sema_counts(n: int, reps=(1, 2, 3)) -> list[int]:
    """Median per-replica served-request count for Sema at replica-count n.

    Each cell's per-replica count is the after-before delta of the replica's
    request_success_total; we take the median over reps (all reps are identical
    here, so the median is exact)."""
    per_ep = defaultdict(list)
    for rep in reps:
        arm = f"sema_dp_n{n}_r1_rows32000_rep{rep}"
        for ep in range(n):
            after = _sema_stop_total(SEMA_DIR / f"metrics_after_{arm}_ep{ep}.txt")
            before = _sema_stop_total(SEMA_DIR / f"metrics_before_{arm}_ep{ep}.txt")
            if after is None:
                continue
            per_ep[ep].append(int(round(after - (before or 0.0))))
    return [int(sorted(per_ep[ep])[len(per_ep[ep]) // 2]) for ep in sorted(per_ep)]


def skein_counts(n: int, family="curve", model="Qwen/Qwen2.5-7B-Instruct") -> list[int]:
    """Median per-replica served-request count for Skein at endpoint-count n.

    Cap-matched DP curve (n=1 cap128 / n=2 cap256 / n=4 cap512), 7B."""
    rows = list(csv.DictReader((SKEIN_DIR / "dp_balance.csv").open()))
    per_ep = defaultdict(list)
    for r in rows:
        if r["family"] != family or r["model"] != model:
            continue
        if int(r["n_ep"]) != n or int(r["tp"]) != 1:
            continue
        per_ep[int(r["ep"])].append(int(r["requests"]))
    return [int(sorted(per_ep[ep])[len(per_ep[ep]) // 2]) for ep in sorted(per_ep)]


def as_share(counts: list[int]) -> list[float]:
    tot = float(sum(counts))
    return [100.0 * c / tot for c in counts]


# ---------------------------------------------------------------------------
# Figure: per-replica request share at N=4 (four-GPU DP).
# ---------------------------------------------------------------------------
def main() -> None:
    ts.setup_style(base_font=14)

    N = 4
    sema = sema_counts(N)          # [32000, 0, 0, 0]
    skein = skein_counts(N)        # [8001, 8000, 8000, 8000]
    sema_sh = as_share(sema)       # [100, 0, 0, 0]
    skein_sh = as_share(skein)     # [25, 25, 25, 25]

    # Report every N so the printed record backs the caption's scaling numbers.
    print("Per-replica served-request counts (median over 3 reps):")
    for n in (1, 2, 4):
        print(f"  N={n}: Sema={sema_counts(n)}  Skein={skein_counts(n)}")
    print(f"N=4 shares:  Sema={[round(v,1) for v in sema_sh]}  "
          f"Skein={[round(v,1) for v in skein_sh]}")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    x = list(range(N))
    w = 0.38
    xs_sema = [i - w / 2 for i in x]
    xs_skein = [i + w / 2 for i in x]

    # Even-share reference: with a load-spreading router, each of N replicas
    # should see 100/N %. Label it on the line in the empty right margin.
    even = 100.0 / N
    ax.axhline(even, color=EVEN_LINE_COLOR, ls="--", lw=1.6, zorder=1,
               label=f"{even:.0f}% even share")

    bars_sema = ax.bar(xs_sema, sema_sh, width=w, color=SEMA_COLOR,
                       label="Sema", zorder=3)
    bars_skein = ax.bar(xs_skein, skein_sh, width=w, color=SKEIN_COLOR,
                        label="Skein", zorder=3)

    # Value labels; make the three idle Sema replicas explicit with a "0".
    for b, v in zip(bars_sema, sema_sh):
        ax.annotate(f"{v:.0f}%" if v > 0 else "0",
                    (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11,
                    color=SEMA_COLOR, fontweight="bold")
    for b, v in zip(bars_skein, skein_sh):
        ax.annotate(f"{v:.0f}%",
                    (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11,
                    color=SKEIN_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"replica {i}" for i in x])
    ax.set_xlim(-0.7, N - 1 + 0.7)
    ax.set_ylim(0, 108)
    ax.set_ylabel(ts.axis_label("share of requests received", "%"))
    ax.set_title("Per-replica request share")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.96))

    ts.no_suptitle(fig)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"sema_dp_scaleout.{ext}")
    print(f"wrote {OUT_DIR / 'sema_dp_scaleout.pdf'} (+ .png)")


if __name__ == "__main__":
    main()

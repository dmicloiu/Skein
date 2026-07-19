#!/usr/bin/env python3
"""Summarise the cross-system movie experiment (flock operator vs LOTUS vs
Palimpzest on one local vLLM) into a CSV + Markdown table, matching the
sembench summary convention (cf. scripts/vllm_perf_aggregate.py).

Reads the per-(arm, rep) artefacts produced by
sembench/slurm/cross_system_analysis_clariden.sh and aggregates over reps
(median; min/max for stability). All LLM-side numbers (tokens, requests,
latency) come from the vLLM /metrics deltas so they are uniform across systems
(the harness hardcodes flock token_usage to 0). Throughput is rows_scanned /
execution_time (= --rows / time), NOT the runner's survivors-per-second.

Inputs (in --results-dir):
  <arm>_rep<r>_<system>.json                harness metrics + P/R/F1 (Q<query>)
  metrics_{before,after}_<arm>_rep<r>.txt   vLLM /metrics snapshots

Outputs:
  cross_system_summary.csv   one row per arm, full column set (the single source
                             of truth). Curated tables and interpretive findings
                             live in the write-up, not here, so nothing
                             regenerated can go stale.

Columns (derived metrics):
  rows_s          rows scanned / execution_time (NOT the runner's survivors/s).
  tok_s           total (prompt+gen) tokens / s. CAUTION: vllm:prompt_tokens_total
                  counts prefix-cache HITS (its delta == prefix_cache_queries_total's),
                  so this metric is inflated by re-sent-but-cached prompt bytes and
                  rewards fat, un-amortized templates. Kept for reference only.
  computed_tok_s  (prompt - prefix_cache_hits + gen) / s = tokens the GPU actually
                  processed per second. The honest engine-efficiency metric;
                  cache_hit_pct = hits / prompt tokens.
  prefill_tok_s   prompt-token throughput; decode_tok_s = gen-token throughput.
  req_s           requests / s; tok_per_row = total tokens / row.
  precision/recall/f1   vs ground truth (full-set for the no-limit Q101).
  pareto_rows_f1 / pareto_tok_f1   `*` = on the (throughput | engine-eff) x F1
                  frontier; the engine-eff frontier uses computed_tok_s.

Usage:
  python analysis/summarize_cross_system.py \
      --results-dir analysis/figures/data/cross_system [--rows 2000] [--query 101]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path

# arm -> (display order, sembench system). Order = table order.
ARM_SYSTEM = {
    "flock_op_r32": "flockmtl",
    "flock_op_r1": "flockmtl",
    "flock_scalar": "flockmtl",
    "flock_scalar_r1": "flockmtl",
    "lotus": "lotus",
    "palimpzest": "palimpzest",
}
ARM_ORDER = list(ARM_SYSTEM.keys())


def _metric(path: Path, name: str, label: str | None = None) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split("{")[0].split(" ")[0] != name:
            continue
        if label is not None and label not in line:
            continue
        try:
            return float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
    return None


def _delta(results_dir: Path, arm: str, rep: int, name: str, label: str | None = None):
    a = _metric(results_dir / f"metrics_after_{arm}_rep{rep}.txt", name, label)
    b = _metric(results_dir / f"metrics_before_{arm}_rep{rep}.txt", name, label)
    if a is None or b is None:
        return None
    return a - b


def parse_rep(results_dir: Path, arm: str, system: str, rep: int, query: int) -> dict | None:
    j = results_dir / f"{arm}_rep{rep}_{system}.json"
    if not j.exists():
        return None
    try:
        q = json.loads(j.read_text()).get(f"Q{query}", {})
    except json.JSONDecodeError:
        return None
    if q.get("status") != "success":
        return None
    pt = _delta(results_dir, arm, rep, "vllm:prompt_tokens_total")
    gt = _delta(results_dir, arm, rep, "vllm:generation_tokens_total")
    ch = _delta(results_dir, arm, rep, "vllm:prefix_cache_hits_total")
    rq = _delta(results_dir, arm, rep, "vllm:request_success_total", 'finished_reason="stop"')
    return {
        "time": q.get("execution_time"),
        "survivors": q.get("row_count"),
        "precision": q.get("precision"),
        "recall": q.get("recall"),
        "f1": q.get("f1_score"),
        "prompt_tok": pt,
        "gen_tok": gt,
        "cache_hit_tok": ch,
        "total_tok": (pt + gt) if (pt is not None and gt is not None) else None,
        # Tokens the GPU actually processed (cache hits are counted in
        # prompt_tokens_total but never computed).
        "computed_tok": (pt - ch + gt) if (pt is not None and ch is not None and gt is not None) else None,
        "requests": rq,
    }


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def aggregate(results_dir: Path, query: int, rows: int) -> list[dict]:
    reps = sorted({
        int(m.group(1)) for p in results_dir.glob("*_rep*")
        for m in [re.search(r"_rep(\d+)", p.name)] if m
    })
    out = []
    for arm in ARM_ORDER:
        system = ARM_SYSTEM[arm]
        rs = [parse_rep(results_dir, arm, system, r, query) for r in reps]
        rs = [r for r in rs if r]
        if not rs:
            continue
        times = [r["time"] for r in rs if r["time"] is not None]
        t = _med(times)
        tot = _med([r["total_tok"] for r in rs])
        comp = _med([r["computed_tok"] for r in rs])
        hits = _med([r["cache_hit_tok"] for r in rs])
        pt_med = _med([r["prompt_tok"] for r in rs])
        survivors = _med([r["survivors"] for r in rs])
        row = {
            "arm": arm, "system": system, "n": len(rs),
            "time_s": round(t, 2) if t else None,
            "time_min": round(min(times), 2) if times else None,
            "time_max": round(max(times), 2) if times else None,
            "rows_s": round(rows / t, 1) if t else None,
            "req_s": round(_med([r["requests"] for r in rs]) / t, 1)
            if (t and _med([r["requests"] for r in rs])) else None,
            "requests": int(_med([r["requests"] for r in rs]))
            if _med([r["requests"] for r in rs]) is not None else None,
            "prompt_tok": int(_med([r["prompt_tok"] for r in rs]))
            if _med([r["prompt_tok"] for r in rs]) is not None else None,
            "gen_tok": int(_med([r["gen_tok"] for r in rs]))
            if _med([r["gen_tok"] for r in rs]) is not None else None,
            "total_tok": int(tot) if tot is not None else None,
            "cache_hit_tok": int(hits) if hits is not None else None,
            "cache_hit_pct": round(100.0 * hits / pt_med, 1)
            if (hits is not None and pt_med) else None,
            "computed_tok": int(comp) if comp is not None else None,
            "tok_per_row": round(tot / rows, 1) if tot is not None else None,
            "tok_s": round(tot / t, 0) if (tot is not None and t) else None,
            "computed_tok_s": round(comp / t, 0) if (comp is not None and t) else None,
            "prefill_tok_s": round(_med([r["prompt_tok"] for r in rs]) / t, 0)
            if (t and _med([r["prompt_tok"] for r in rs])) else None,
            "decode_tok_s": round(_med([r["gen_tok"] for r in rs]) / t, 0)
            if (t and _med([r["gen_tok"] for r in rs])) else None,
            "survivors": int(survivors) if survivors is not None else None,
            "precision": round(_med([r["precision"] for r in rs]), 3) if _med([r["precision"] for r in rs]) is not None else None,
            "recall": round(_med([r["recall"] for r in rs]), 3) if _med([r["recall"] for r in rs]) is not None else None,
            "f1": round(_med([r["f1"] for r in rs]), 3) if _med([r["f1"] for r in rs]) is not None else None,
        }
        out.append(row)
    _mark_pareto(out, "rows_s", "f1", "pareto_rows_f1")
    # Engine-efficiency frontier on COMPUTED tokens; raw tok_s counts cached
    # prefill, so it would crown whoever re-sends the fattest template.
    _mark_pareto(out, "computed_tok_s", "f1", "pareto_tok_f1")
    return out


def _mark_pareto(rows: list[dict], xk: str, yk: str, flag: str) -> None:
    """Flag rows on the maximise-x, maximise-y Pareto frontier."""
    for r in rows:
        x, y = r.get(xk), r.get(yk)
        if x is None or y is None:
            r[flag] = ""
            continue
        dominated = any(
            o is not r and o.get(xk) is not None and o.get(yk) is not None
            and o[xk] >= x and o[yk] >= y and (o[xk] > x or o[yk] > y)
            for o in rows
        )
        r[flag] = "" if dominated else "*"


COLUMNS = [
    "arm", "system", "n", "time_s", "time_min", "time_max", "rows_s", "req_s",
    "requests", "prompt_tok", "gen_tok", "total_tok", "cache_hit_tok",
    "cache_hit_pct", "computed_tok", "tok_per_row", "tok_s", "computed_tok_s",
    "prefill_tok_s", "decode_tok_s", "survivors", "precision", "recall", "f1",
    "pareto_rows_f1", "pareto_tok_f1",
]
def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def _fmt(v) -> str:
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def print_console(rows: list[dict]) -> None:
    cols = ["arm", "n", "time_s", "rows_s", "computed_tok_s", "decode_tok_s",
            "cache_hit_pct", "total_tok", "precision", "recall", "f1", "pareto_tok_f1"]
    w = {c: max(len(c), max((len(_fmt(r.get(c))) for r in rows), default=0)) for c in cols}
    print("  ".join(c.rjust(w[c]) for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).rjust(w[c]) for c in cols))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("analysis/figures/data/cross_system"))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: same as --results-dir")
    ap.add_argument("--rows", type=int, default=2000, help="rows scanned (scale factor)")
    ap.add_argument("--query", type=int, default=101)
    args = ap.parse_args()
    if not args.results_dir.exists():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = aggregate(args.results_dir, args.query, args.rows)
    if not rows:
        print("no successful runs found", file=sys.stderr)
        return 1
    write_csv(rows, out_dir / "cross_system_summary.csv")
    print_console(rows)
    print(f"\nwrote {out_dir/'cross_system_summary.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

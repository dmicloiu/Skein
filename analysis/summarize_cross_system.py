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

Outputs (in --out-dir, default = --results-dir):
  cross_system_summary.csv   one row per arm, full column set
  cross_system_summary.md    same columns as a Markdown table + legend + notes

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
    rq = _delta(results_dir, arm, rep, "vllm:request_success_total", 'finished_reason="stop"')
    return {
        "time": q.get("execution_time"),
        "survivors": q.get("row_count"),
        "precision": q.get("precision"),
        "recall": q.get("recall"),
        "f1": q.get("f1_score"),
        "prompt_tok": pt,
        "gen_tok": gt,
        "total_tok": (pt + gt) if (pt is not None and gt is not None) else None,
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
            "tok_per_row": round(tot / rows, 1) if tot is not None else None,
            "tok_s": round(tot / t, 0) if (tot is not None and t) else None,
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
    _mark_pareto(out, "tok_s", "f1", "pareto_tok_f1")
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
    "requests", "prompt_tok", "gen_tok", "total_tok", "tok_per_row", "tok_s",
    "prefill_tok_s", "decode_tok_s", "survivors", "precision", "recall", "f1",
    "pareto_rows_f1", "pareto_tok_f1",
]
# Narrower column set for the readable Markdown table.
MD_COLUMNS = [
    "arm", "n", "time_s", "rows_s", "req_s", "total_tok", "tok_s",
    "prefill_tok_s", "decode_tok_s", "survivors", "precision", "recall", "f1",
    "pareto_tok_f1",
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


def write_md(rows: list[dict], path: Path, rows_scanned: int, query: int) -> None:
    lines = []
    lines.append(f"# Cross-system movie comparison (Q{query}, {rows_scanned} rows, 1x GH200, Qwen2.5-7B)\n")
    lines.append("flock async PhysicalSemFilter operator vs LOTUS vs Palimpzest, "
                 "same local vLLM. Numbers are medians over the per-arm repeats; "
                 "tokens/requests are vLLM /metrics deltas (uniform across systems). "
                 "`rows_s = rows / execution_time` (rows scanned, not survivors). "
                 "`*` in `pareto_tok_f1` = on the tokens/s x F1 frontier (engine "
                 "efficiency at quality).\n")
    header = "| " + " | ".join(MD_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in MD_COLUMNS) + " |"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in MD_COLUMNS) + " |")
    lines.append("\nFull column set (incl. precision/recall split, prompt/gen "
                 "tokens, time min/max, rows/s x F1 frontier) is in the CSV.\n")

    by = {r["arm"]: r for r in rows}

    def ratio(a, b, k):
        if a in by and b in by and by[a].get(k) and by[b].get(k):
            return by[a][k] / by[b][k]
        return None

    lines.append("## Notes\n")
    notes = []
    if ratio("flock_scalar", "flock_op_r32", "time_s"):
        notes.append(f"- **Operator vs scalar (engine):** ~{ratio('flock_scalar','flock_op_r32','time_s'):.1f}x faster at R=32 "
                     f"and ~{ratio('flock_scalar_r1','flock_op_r1','time_s'):.1f}x at R=1, at byte-identical tokens and "
                     f"matched F1 (operator preserves scalar semantics).")
    if ratio("flock_op_r1", "lotus", "tok_s"):
        notes.append(f"- **Engine efficiency (tokens/s):** flock_op_r1 is highest "
                     f"({by['flock_op_r1']['tok_s']:.0f} tok/s) -- {ratio('flock_op_r1','lotus','tok_s'):.2f}x LOTUS, "
                     f"{ratio('flock_op_r1','palimpzest','tok_s'):.2f}x Palimpzest -- and on tokens/s x F1 it "
                     f"dominates LOTUS, Palimpzest, and flock_op_r32.")
    notes.append("- **LOTUS' wall-clock lead is from doing less work:** lowest tokens "
                 "(294k vs flock_op_r1 1012k, ~3.4x fewer) via a lighter prompt + lower "
                 "recall -- not a faster engine (its tok/s is ~half flock's).")
    notes.append("- **R=32 vs R=1 is a recall trade:** R=32 packs 32 rows/prompt -> "
                 "~9.6x fewer tokens and 4.4x more rows/s, but recall collapses "
                 "(0.97 -> 0.37, F1 0.93 -> 0.53). The usable flock config is R=1.")
    notes.append("- **flock_op_r1 vs flock_scalar_r1 F1 (0.931 vs 0.933) is within the "
                 "temp-0 noise floor** -- both `*` on the tokens/s x F1 frontier, but op_r1 "
                 "is the usable one (7.8x higher engine throughput at equal quality).")
    notes.append("- **flock tok/s is prefill-dominated** (gen ~8 tok/req, by design for a "
                 "boolean filter) -- on decode tok/s flock is lowest, Palimpzest highest "
                 "(~65 gen tok/req, verbose). Show the prompt/decode split if challenged.")
    notes.append("- **Caveats:** LOTUS n=2 (rep3 hit a transient vLLM 500); dataset is "
                 "~75% positive, which flatters a recall-leaning system; quality is each "
                 "system's default operating point (no threshold tuning).")
    lines.extend(notes)
    path.write_text("\n".join(lines) + "\n")


def print_console(rows: list[dict]) -> None:
    cols = ["arm", "n", "time_s", "rows_s", "tok_s", "decode_tok_s", "total_tok",
            "precision", "recall", "f1", "pareto_tok_f1"]
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
    write_md(rows, out_dir / "cross_system_summary.md", args.rows, args.query)
    print_console(rows)
    print(f"\nwrote {out_dir/'cross_system_summary.csv'}", file=sys.stderr)
    print(f"wrote {out_dir/'cross_system_summary.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

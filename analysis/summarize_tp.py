#!/usr/bin/env python3
"""Summarise the TP (tensor-parallel) scaling runs into a CSV (the single
source of truth). Curated tables and interpretive findings live in the
write-up, not here. Sibling of summarize_morsel_ab.py.

Takes a list of result directories (each produced by tp_scaling_clariden.sh;
one directory = one SLURM job = one rep, possibly a partial sweep when the job
was split via TP_SWEEP/ARMS). Per directory it parses every
result_{op,scalar}_tp<K>[_c<CAP>].json plus the matching vLLM /metrics
snapshots, the scheduler-gauge samples (samples_<tag>.csv) and the operator
verdict dumps (verdicts_tp<K>.jsonl).

TP-specific derived metrics:
  eff            rows/s at TP=k / rows/s of the SAME arm at TP=1. Base comes
                 from the same directory (in-job, node-consistent) when the job
                 ran TP=1; otherwise from the pooled median of TP=1 rows/s
                 across all parsed dirs (eff_basis says which -- cross-job
                 bases carry the ~+-10% node variance).
  run/wait_mean  mean of the sampled vllm:num_requests_running/waiting gauges
                 over the run window: the saturation evidence. An under-fed
                 endpoint shows run_mean well below the in-flight cap with
                 wait_mean ~0; a saturating cap pins run_mean near the cap.
  concurrency    mean in-flight requests via Little's law (req/s x mean e2e
                 latency) -- the rigorous form of the in-script A/B gate
                 (operator ~cap, scalar ~1).
  computed_tok_s (prompt - prefix_cache_hits + gen) / s: tokens the GPU
                 actually processed (prompt_tokens_total counts cache hits).
  f1/precision/recall  operator verdicts (verdicts_tp<K>.jsonl, untimed pass)
                 joined to the gold CSV -- the F1-is-TP-invariant sanity check.

Usage:
  python analysis/summarize_tp.py DIR1 DIR2 ... [--out-dir DIR] [--data CSV]
  # default: every dir under analysis/figures/data/tp_scaling
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

TAG_RE = re.compile(r"result_(op|scalar)_tp(\d+)(?:_c(\d+))?(?:_rep(\d+))?\.json")


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


def _delta(d: Path, mid: str, name: str, label: str | None = None):
    a = _metric(d / f"metrics_after_{mid}.txt", name, label)
    b = _metric(d / f"metrics_before_{mid}.txt", name, label)
    return (a - b) if (a is not None and b is not None) else None


def _samples(d: Path, fname: str) -> dict:
    """Mean/max of the sampled scheduler gauges over the run window."""
    f = d / fname
    out = {"run_mean": None, "run_max": None, "wait_mean": None,
           "wait_max": None, "kv_mean": None, "kv_max": None}
    if not f.exists():
        return out
    cols: dict[str, list[float]] = {"running": [], "waiting": [], "kv_usage": []}
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            for k in cols:
                try:
                    cols[k].append(float(row[k]))
                except (KeyError, TypeError, ValueError):
                    pass
    for key, pre in (("running", "run"), ("waiting", "wait"), ("kv_usage", "kv")):
        v = cols[key]
        if v:
            out[f"{pre}_mean"] = round(sum(v) / len(v), 1)
            out[f"{pre}_max"] = round(max(v), 1)
    return out


# ---- operator verdicts -> P/R/F1 vs gold (diagnose_rsweep.py conventions) ---
def load_gold(csv_path: Path, gold_col: str, gold_pos: str) -> dict[int, bool]:
    out: dict[int, bool] = {}
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            out[i] = (row.get(gold_col) or "").strip().upper() == gold_pos.upper()
    return out


def verdict_prf1(path: Path, gold: dict[int, bool]) -> dict:
    tp = fp = fn = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            g = gold.get(int(d["id"]))
            if g is None:
                continue
            passed = d.get("v") is True or d.get("v") is None  # null->pass parity
            if g:
                tp += passed
                fn += not passed
            elif passed:
                fp += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


def parse_dir(d: Path, gold: dict[int, bool] | None) -> list[dict]:
    recs = []
    for rj in sorted(d.glob("result_*_tp*.json")):
        m = TAG_RE.fullmatch(rj.name)
        if not m:
            continue
        arm, tp_k, cap, rep = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        # Flat rep layout (result_..._rep<N>.json): the rep is the grouping
        # unit for the efficiency base; sibling files carry the same suffix.
        suf = f"_rep{rep}" if rep else ""
        base = f"{arm}_tp{tp_k}" + (f"_c{cap}" if cap else "")
        tag = f"{base}_ep0{suf}"   # middle of the metrics_{before,after} names
        try:
            j = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        el = j.get("elapsed_s")
        rows = j.get("rows")
        rq = _delta(d, tag, "vllm:request_success_total", 'finished_reason="stop"')
        pt = _delta(d, tag, "vllm:prompt_tokens_total")
        gt = _delta(d, tag, "vllm:generation_tokens_total")
        ch = _delta(d, tag, "vllm:prefix_cache_hits_total")
        cq = _delta(d, tag, "vllm:prefix_cache_queries_total")
        tot = (pt + gt) if (pt is not None and gt is not None) else None

        def mean_ms(base):  # per-request mean of a vLLM latency histogram, in ms
            s = _delta(d, tag, f"vllm:{base}_seconds_sum")
            c = _delta(d, tag, f"vllm:{base}_seconds_count")
            return (s / c * 1000.0) if (s and c) else None

        e2e = mean_ms("e2e_request_latency")
        conc = ((rq / el) * (e2e / 1000.0)) if (rq and e2e and el) else None
        preempt = _delta(d, tag, "vllm:num_preemptions_total")

        def rnd(x, n=1):
            return round(x, n) if x is not None else None

        rec = {
            "job": f"rep{rep}" if rep else d.name,
            "arm": arm,
            "tp": tp_k,
            "cap": int(cap) if cap else j.get("inflight"),
            "R": j.get("batch"),
            "threads": j.get("threads"),
            "rows": rows,
            "elapsed_s": rnd(el),
            # --- throughput / scaling ---
            "rows_s": rnd(j.get("rows_per_s")),
            "eff": None, "eff_basis": None,        # filled by add_efficiency
            "req_s": rnd(rq / el) if (rq and el) else None,
            "concurrency": rnd(conc),
            "tok_s": rnd(tot / el, 0) if (tot and el) else None,
            "computed_tok_s": rnd((pt - ch + gt) / el, 0)
            if (pt is not None and ch is not None and gt is not None and el) else None,
            "prefill_tok_s": rnd(pt / el, 0) if (pt and el) else None,
            "decode_tok_s": rnd(gt / el, 0) if (gt and el) else None,
            # --- token volume / shape ---
            "prompt_tok": int(pt) if pt is not None else None,
            "gen_tok": int(gt) if gt is not None else None,
            "tok_per_row": rnd(tot / rows) if (tot and rows) else None,
            "gen_per_req": rnd(gt / rq) if (gt and rq) else None,
            # --- latency breakdown (per-request means, ms) ---
            "e2e_ms": rnd(e2e, 0),
            "ttft_ms": rnd(mean_ms("time_to_first_token"), 0),
            "queue_ms": rnd(mean_ms("request_queue_time"), 0),
            "prefill_ms": rnd(mean_ms("request_prefill_time"), 0),
            "decode_ms": rnd(mean_ms("request_decode_time"), 0),
            # --- engine health / saturation ---
            "prefix_hit_pct": rnd(100 * ch / cq) if (ch is not None and cq) else None,
            "preemptions": int(preempt) if preempt is not None else None,
            "requests": int(rq) if rq is not None else None,
            **_samples(d, f"samples_{base}{suf}.csv"),
            # --- quality ---
            "passes": j.get("passes"),
            "pass_pct": rnd(100 * j["passes"] / rows) if (rows and "passes" in j) else None,
            "precision": None, "recall": None, "f1": None,
        }
        # verdict dump is per (dir, TP), taken on the operator's first-cap fleet
        vd = d / f"verdicts_tp{tp_k}{suf}.jsonl"
        if arm == "op" and gold is not None and vd.exists():
            rec.update(verdict_prf1(vd, gold))
        recs.append(rec)
    return recs


def add_efficiency(recs: list[dict]) -> None:
    """eff = rows_s / same-arm TP=1 rows_s; in-job base when available, else
    the pooled median TP=1 base across all dirs (cross-job => node variance)."""
    def median(v):
        v = sorted(v)
        n = len(v)
        return (v[n // 2] + v[(n - 1) // 2]) / 2 if v else None

    pooled = {arm: median([r["rows_s"] for r in recs
                           if r["arm"] == arm and r["tp"] == 1 and r["rows_s"]])
              for arm in ("op", "scalar")}
    in_job = {}
    for r in recs:
        if r["tp"] == 1 and r["rows_s"]:
            in_job.setdefault((r["job"], r["arm"]), r["rows_s"])
    for r in recs:
        if not r["rows_s"]:
            continue
        base = in_job.get((r["job"], r["arm"]))
        basis = "in_job"
        if base is None:
            base, basis = pooled.get(r["arm"]), "pooled_tp1"
        if base:
            r["eff"], r["eff_basis"] = round(r["rows_s"] / base, 2), basis


COLUMNS = ["job", "arm", "tp", "cap", "R", "threads", "rows", "elapsed_s",
           "rows_s", "eff", "eff_basis", "req_s", "concurrency",
           "tok_s", "computed_tok_s", "prefill_tok_s", "decode_tok_s",
           "prompt_tok", "gen_tok", "tok_per_row", "gen_per_req",
           "e2e_ms", "ttft_ms", "queue_ms", "prefill_ms", "decode_ms",
           "prefix_hit_pct", "preemptions", "requests",
           "run_mean", "run_max", "wait_mean", "wait_max", "kv_mean", "kv_max",
           "passes", "pass_pct", "precision", "recall", "f1"]
# Curated subset for the console preview (the CSV holds the full column set).
PREVIEW_COLUMNS = ["job", "arm", "tp", "cap", "rows_s", "eff", "computed_tok_s",
                   "concurrency", "run_mean", "wait_mean", "e2e_ms", "f1"]


def _fmt(v):
    return "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def write_outputs(recs, out_dir: Path):
    recs = sorted(recs, key=lambda r: (r["tp"], r["arm"], r["cap"] or 0, r["job"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tp_scaling_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    print("  ".join(c.rjust(12) for c in PREVIEW_COLUMNS))
    for r in recs:
        print("  ".join(_fmt(r.get(c)).rjust(12) for c in PREVIEW_COLUMNS))
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path,
                    help="result dirs (default: every dir under analysis/figures/data/tp_scaling)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("analysis/figures/data/tp_scaling"))
    ap.add_argument("--data", type=Path,
                    default=Path("../sembench/files/movie/data/sf_2000/Reviews.csv"),
                    help="gold CSV for the operator F1 columns (skipped if missing)")
    ap.add_argument("--gold-col", default="scoreSentiment")
    ap.add_argument("--gold-positive", default="POSITIVE")
    args = ap.parse_args()

    gold = None
    if args.data.exists():
        gold = load_gold(args.data, args.gold_col, args.gold_positive)
    else:
        print(f"  [warn] gold csv missing ({args.data}) -> F1 columns skipped", file=sys.stderr)

    data_root = Path("analysis/figures/data/tp_scaling")
    dirs = args.dirs or ([p for p in sorted(data_root.iterdir()) if p.is_dir()]
                         if data_root.exists() else [])
    recs = []
    for d in dirs:
        if not d.exists():
            print(f"  [warn] missing {d}", file=sys.stderr)
            continue
        recs += parse_dir(d, gold)
    if not recs:
        print("no result files found", file=sys.stderr)
        return 1
    add_efficiency(recs)
    write_outputs(recs, args.out_dir)
    print(f"\nwrote {args.out_dir/'tp_scaling_summary.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

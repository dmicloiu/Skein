#!/usr/bin/env python3
"""Canonical summary of a prompt sweep (sem_filter_prompt_slim_* artefact dir).

One row per (prompt head, R): median + [min-max] over reps for rows/s and the
engine-side metrics, plus quality against the source CSV gold.

Metric standard (see analysis/evaluation.md):
  - rows/s at a stated F1 is the primary number.
  - computed tok/s = (prompt_tok - prefix_cache_hits + gen_tok) / elapsed, i.e.
    tokens the GPU actually processed. Raw (prompt+gen)/elapsed counts cache hits
    and inflates fat-prompt configs, so it is reported only alongside.
  - quality is row-level P/R/F1 (one verdict per table row). set_f1 is the
    sembench convention (unique reviewId set) and is carried for cross-checking
    against the cross-system harness, which scores that way.

Usage:
  python analysis/summarize_prompt_sweep.py <artefact-dir> [--data Reviews.csv]
                                            [--out summary.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

RESULT_RE = re.compile(r"^result_r(\d+)(?:_rep(\d+))?\.json$")


def load_gold(csv_path: Path, gold_col: str, gold_pos: str) -> tuple[list[bool], list[str]]:
    """Gold labels + reviewIds keyed by scan-order row_id (threads=1 => CSV order)."""
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    gold = [(r.get(gold_col) or "").strip().upper() == gold_pos.strip().upper() for r in rows]
    ids = [r.get("reviewId", "") for r in rows]
    return gold, ids


def metric(path: Path, name: str) -> float | None:
    """Read one vLLM /metrics counter (exact metric-name prefix match)."""
    if not path.exists():
        return None
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.startswith(name):
                continue
            try:
                return float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                return None
    return None


def token_deltas(dirpath: Path, tag: str) -> dict[str, float]:
    """tag is the full artefact tag including the endpoint suffix, e.g. r1_ep0_rep2."""
    before, after = dirpath / f"metrics_before_{tag}.txt", dirpath / f"metrics_after_{tag}.txt"
    keys = {
        "prompt_tok": "vllm:prompt_tokens_total",
        "gen_tok": "vllm:generation_tokens_total",
        "cache_hit_tok": "vllm:prefix_cache_hits_total",
        "cache_query_tok": "vllm:prefix_cache_queries_total",
    }
    out: dict[str, float] = {}
    for k, m in keys.items():
        b, a = metric(before, m), metric(after, m)
        out[k] = (a - b) if (a is not None and b is not None) else float("nan")
    return out


def score(verdicts: Path, gold: list[bool], ids: list[str]) -> dict[str, float]:
    """Row-level P/R/F1 plus the unique-reviewId set F1 the sembench harness uses."""
    if not verdicts.exists():
        return {}
    passes: dict[int, bool] = {}
    with verdicts.open() as fh:
        for line in fh:
            d = json.loads(line)
            # null verdict counts as TRUE (the operator parse contract)
            passes[d["id"]] = True if d["v"] is None else bool(d["v"])
    tp = sum(1 for i, p in passes.items() if p and gold[i])
    fp = sum(1 for i, p in passes.items() if p and not gold[i])
    fn = sum(1 for i, p in passes.items() if not p and gold[i])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    gold_ids = {ids[i] for i, g in enumerate(gold) if g}
    sys_ids = {ids[i] for i, p in passes.items() if p}
    stp = len(sys_ids & gold_ids)
    sp = stp / len(sys_ids) if sys_ids else 0.0
    sr = stp / len(gold_ids) if gold_ids else 0.0
    return {
        "rows_scored": len(passes),
        "passes": sum(passes.values()),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "set_f1": 2 * sp * sr / (sp + sr) if sp + sr else 0.0,
    }


def spread(values: list[float]) -> tuple[float, float, float]:
    vals = [v for v in values if v == v]  # drop NaN
    if not vals:
        return (float("nan"),) * 3
    return statistics.median(vals), min(vals), max(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artefact_dir", type=Path)
    ap.add_argument("--data", type=Path,
                    default=Path.home() / "sembench/files/movie/data/sf_2000/Reviews.csv")
    ap.add_argument("--gold-col", default="scoreSentiment")
    ap.add_argument("--gold-positive", default="POSITIVE")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"gold CSV missing: {args.data}", file=sys.stderr)
        return 1
    gold, ids = load_gold(args.data, args.gold_col, args.gold_positive)

    heads = [d for d in sorted(args.artefact_dir.iterdir()) if d.is_dir()]
    if not heads:
        print(f"no prompt-head subdirs under {args.artefact_dir}", file=sys.stderr)
        return 1

    rows = []
    for head in heads:
        cells: dict[int, list[dict]] = {}
        for f in sorted(head.iterdir()):
            m = RESULT_RE.match(f.name)
            if not m:
                continue
            R, rep = int(m.group(1)), m.group(2) or "1"
            res = json.loads(f.read_text())
            tag = f"r{R}_ep0_rep{rep}" if m.group(2) else f"r{R}_ep0"
            tok = token_deltas(head, tag)
            vtag = f"r{R}_rep{rep}.jsonl" if m.group(2) else f"r{R}.jsonl"
            q = score(head / f"verdicts_{vtag}", gold, ids)
            elapsed = res["elapsed_s"]
            computed = tok["prompt_tok"] - tok["cache_hit_tok"] + tok["gen_tok"]
            cells.setdefault(R, []).append({
                "rep": rep,
                "rows_s": res["rows_per_s"],
                "elapsed_s": elapsed,
                "raw_tok_s": (tok["prompt_tok"] + tok["gen_tok"]) / elapsed,
                "computed_tok_s": computed / elapsed,
                "prompt_tok_row": tok["prompt_tok"] / res["rows"],
                "gen_tok_row": tok["gen_tok"] / res["rows"],
                "cache_hit_pct": 100 * tok["cache_hit_tok"] / tok["prompt_tok"],
                "tuple_format": res.get("tuple_format", "unset(XML)"),
                **q,
            })
        for R, reps in sorted(cells.items()):
            med_rows, min_rows, max_rows = spread([c["rows_s"] for c in reps])
            f1s = [c.get("f1", float("nan")) for c in reps]
            med_f1, min_f1, max_f1 = spread(f1s)
            rows.append({
                "prompt": head.name,
                "tuple_format": reps[0]["tuple_format"],
                "R": R,
                "n_reps": len(reps),
                "rows_s": round(med_rows, 1),
                "rows_s_min": round(min_rows, 1),
                "rows_s_max": round(max_rows, 1),
                "f1": round(med_f1, 4),
                "f1_min": round(min_f1, 4),
                "f1_max": round(max_f1, 4),
                "precision": round(spread([c.get("precision", float("nan")) for c in reps])[0], 4),
                "recall": round(spread([c.get("recall", float("nan")) for c in reps])[0], 4),
                "set_f1": round(spread([c.get("set_f1", float("nan")) for c in reps])[0], 4),
                "passes": int(spread([c.get("passes", float("nan")) for c in reps])[0]),
                "computed_tok_s": int(spread([c["computed_tok_s"] for c in reps])[0]),
                "raw_tok_s": int(spread([c["raw_tok_s"] for c in reps])[0]),
                "prompt_tok_row": round(spread([c["prompt_tok_row"] for c in reps])[0], 1),
                "gen_tok_row": round(spread([c["gen_tok_row"] for c in reps])[0], 2),
                "cache_hit_pct": round(spread([c["cache_hit_pct"] for c in reps])[0], 1),
            })

    hdr = list(rows[0])
    w = max(len(h) for h in hdr)
    print(f"{'prompt':>8} {'fmt':>5} {'R':>3} {'n':>2} {'rows/s':>9} {'[min-max]':>15} "
          f"{'F1':>7} {'P':>6} {'R':>6} {'setF1':>6} {'ctok/s':>8} {'tok/row':>8} {'hit%':>6}")
    for r in rows:
        rng = "[{:.0f}-{:.0f}]".format(r["rows_s_min"], r["rows_s_max"])
        print(f"{r['prompt']:>8} {r['tuple_format']:>5} {r['R']:>3} {r['n_reps']:>2} "
              f"{r['rows_s']:>9.1f} {rng:>15} "
              f"{r['f1']:>7.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['set_f1']:>6.3f} "
              f"{r['computed_tok_s']:>8d} {r['prompt_tok_row']:>8.1f} {r['cache_hit_pct']:>6.1f}")

    out = args.out or args.artefact_dir / "prompt_sweep_summary.csv"
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=hdr)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

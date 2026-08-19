#!/usr/bin/env python3
"""Miss-rate of gold positives by in-batch position, per rows-per-request R.

Why: packing R rows into one prompt costs Qwen recall (0.788 -> 0.703 from R=1 to
R=16) and *gains* Llama recall (0.783 -> 0.861), from near-identical starting
points and with precision above 0.95 throughout. For Qwen the established
mechanism is positional: rows later in the packed prompt are under-detected. This
script tests whether that ramp exists on another model, which is the difference
between "batching costs quality" being a general statement or a Qwen one.

Reads FLOCK_VERDICT_DUMP files (one JSONL line per row: query-global row_id, the
row's 0-based position within its batch, and the raw verdict) and buckets the
gold-positive rows by normalised in-batch position.

Usage:
  python analysis/diagnose_batch_position.py <dump-dir> --label Qwen [--data Reviews.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DUMP_RE = re.compile(r"verdicts_(?:.*_)?r(?P<R>\d+)(?:_rep(?P<rep>\d+))?(?:_[a-z0-9]+)?\.jsonl$")
NBUCKETS = 5


def load_gold(csv_path: Path, gold_col: str, gold_pos: str) -> list[bool]:
    with csv_path.open(newline="") as fh:
        return [(r.get(gold_col) or "").strip().upper() == gold_pos.strip().upper()
                for r in csv.DictReader(fh)]


def analyse(path: Path, R: int, gold: list[bool]) -> dict | None:
    """Miss-rate per position bucket, plus the overall recall for that R."""
    tot = [0] * NBUCKETS
    miss = [0] * NBUCKETS
    hits = seen = 0
    with path.open() as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail on a pre-flush-fix dump
            rid = d["id"]
            if rid >= len(gold) or not gold[rid]:
                continue
            v = d["v"]
            passed = True if v is None else bool(v)
            seen += 1
            hits += passed
            b = min(NBUCKETS - 1, int(NBUCKETS * d["pos"] / R)) if R > 1 else 0
            tot[b] += 1
            miss[b] += (not passed)
    if not seen:
        return None
    return {
        "R": R,
        "recall": hits / seen,
        "positives": seen,
        "miss": [(miss[i] / tot[i]) if tot[i] else None for i in range(NBUCKETS)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--label", default="model")
    ap.add_argument("--data", type=Path,
                    default=Path.home() / "sembench/files/movie/data/sf_2000/Reviews.csv")
    ap.add_argument("--gold-col", default="scoreSentiment")
    ap.add_argument("--gold-positive", default="POSITIVE")
    ap.add_argument("--rep", default="1", help="which rep to read when dumps carry a rep suffix")
    args = ap.parse_args()

    gold = load_gold(args.data, args.gold_col, args.gold_positive)

    cells: dict[int, Path] = {}
    for p in sorted(args.dump_dir.rglob("verdicts_*.jsonl")):
        m = DUMP_RE.search(p.name)
        if not m:
            continue
        if m.group("rep") and m.group("rep") != args.rep:
            continue
        cells.setdefault(int(m.group("R")), p)
    if not cells:
        print(f"no verdict dumps under {args.dump_dir}")
        return 1

    print(f"=== {args.label}: miss-rate of gold positives by in-batch position "
          f"(front -> back), {sum(gold)} positives")
    print(f"{'R':>3} {'recall':>7} | " + " ".join(f"{f'b{i}':>6}" for i in range(NBUCKETS))
          + f" | {'ramp':>7}")
    for R in sorted(cells):
        a = analyse(cells[R], R, gold)
        if not a:
            continue
        cellstr = " ".join(f"{v:>6.3f}" if v is not None else f"{'-':>6}" for v in a["miss"])
        ends = [v for v in a["miss"] if v is not None]
        ramp = (ends[-1] - ends[0]) if len(ends) > 1 else float("nan")
        print(f"{R:>3} {a['recall']:>7.3f} | {cellstr} | {ramp:>+7.3f}")
    print("\nramp = back-bucket miss-rate minus front-bucket. Positive = rows later in "
          "the prompt are under-detected (the Qwen mechanism); ~0 = position-independent; "
          "negative = later rows detected better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

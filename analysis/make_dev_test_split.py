#!/usr/bin/env python3
"""Split a sembench Reviews.csv into a dev and a test half for the prompt-variant
sweep (analysis/prompt_engineering.md Q4): tune on dev, report on test.

The split is by md5(reviewId) parity — deterministic across machines and runs
(unlike Python's salted hash()), independent of row order, and label-blind.
Row order within each half is preserved (the drivers and diagnose_rsweep join
gold by scan order).

Usage:
  python analysis/make_dev_test_split.py <Reviews.csv> --out-dir <dir>
Writes <dir>/Reviews_dev.csv and <dir>/Reviews_test.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path


def bucket(review_id: str) -> str:
    digest = hashlib.md5(review_id.encode("utf-8")).digest()
    return "dev" if digest[0] % 2 == 0 else "test"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_in", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--id-col", default="reviewId")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outs = {name: open(args.out_dir / f"Reviews_{name}.csv.tmp.{os.getpid()}", "w", newline="")
            for name in ("dev", "test")}
    counts = {"dev": 0, "test": 0}
    with open(args.csv_in, newline="") as f:
        reader = csv.DictReader(f)
        if args.id_col not in (reader.fieldnames or []):
            print(f"id column '{args.id_col}' not in {reader.fieldnames}", file=sys.stderr)
            return 1
        writers = {name: csv.DictWriter(fh, fieldnames=reader.fieldnames) for name, fh in outs.items()}
        for w in writers.values():
            w.writeheader()
        for row in reader:
            name = bucket(row[args.id_col])
            writers[name].writerow(row)
            counts[name] += 1
    for fh in outs.values():
        fh.close()
    for name in ("dev", "test"):
        os.replace(args.out_dir / f"Reviews_{name}.csv.tmp.{os.getpid()}",
                   args.out_dir / f"Reviews_{name}.csv")
    print(f"dev={counts['dev']} test={counts['test']} -> {args.out_dir}/Reviews_{{dev,test}}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

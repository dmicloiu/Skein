#!/usr/bin/env python3
"""Re-score sem_extract verdict dumps with a fixed gold label set.

This re-scores from the per-row verdict dumps (FLOCK_VERDICT_DUMP) with the macro
average restricted to the gold labels, and reports both numbers so the size of the
artefact stays visible. Off-vocabulary predictions still count as wrong: they are a
false negative for their gold class, they just do not create a class of their own.

Usage:
  python analysis/rescore_extract.py <dump-dir> [--data Reviews.csv] [--out CSV]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DUMP_RE = re.compile(r"verdicts_(?P<arm>.+?_r(?P<R>\d+))(?:_[a-z0-9]+)?\.jsonl$")


def load_gold(csv_path: Path, gold_col: str) -> list[str]:
    with csv_path.open(newline="") as fh:
        return [(r.get(gold_col) or "").strip().upper() for r in csv.DictReader(fh)]


def canon(x: str) -> str:
    """sembench's canonicalisation: substring match, else the raw value (wrong)."""
    u = str(x).strip().upper()
    if "POSITIVE" in u:
        return "POSITIVE"
    if "NEGATIVE" in u:
        return "NEGATIVE"
    return u


def macro(pairs: list[tuple[str, str]], classes: list[str]) -> tuple[float, float, float]:
    ps, rs, fs = [], [], []
    for lab in classes:
        tp = sum(1 for g, p in pairs if p == lab and g == lab)
        fp = sum(1 for g, p in pairs if p == lab and g != lab)
        fn = sum(1 for g, p in pairs if p != lab and g == lab)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        ps.append(pr)
        rs.append(rc)
        fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    n = len(classes)
    return sum(ps) / n, sum(rs) / n, sum(fs) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument("--data", type=Path,
                    default=Path.home() / "sembench/files/movie/data/sf_2000/Reviews.csv")
    ap.add_argument("--gold-col", default="scoreSentiment")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gold = load_gold(args.data, args.gold_col)
    gold_labels = sorted(set(gold))

    dumps = sorted(p for p in args.dump_dir.glob("verdicts_*.jsonl") if DUMP_RE.search(p.name))
    if not dumps:
        print(f"no verdict dumps under {args.dump_dir}")
        return 1

    rows = []
    for path in dumps:
        m = DUMP_RE.search(path.name)
        arm, R = m.group("arm"), int(m.group("R"))
        pred, malformed = {}, 0
        with path.open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1  # truncated tail: the dump is not flushed on exit
                    continue
                pred[d["id"]] = canon(d["v"])
        ids = sorted(pred)
        pairs = [(gold[i], pred[i]) for i in ids]
        off = sum(1 for _, p in pairs if p not in gold_labels)
        p_fix, r_fix, f_fix = macro(pairs, gold_labels)
        all_classes = sorted({g for g, _ in pairs} | {p for _, p in pairs})
        _, _, f_old = macro(pairs, all_classes)
        rows.append({
            "arm": arm, "R": R, "rows_scored": len(ids), "malformed_lines": malformed,
            "off_vocab_rows": off, "n_classes_seen": len(all_classes),
            "precision": round(p_fix, 4), "recall": round(r_fix, 4),
            "f1": round(f_fix, 4), "f1_unrestricted_macro": round(f_old, 4),
        })
    rows.sort(key=lambda r: (r["arm"].rsplit("_r", 1)[0], r["R"]))

    print(f"gold labels: {gold_labels}   rows: {len(gold)}")
    print(f"{'arm':>22} {'scored':>7} {'off-vocab':>10} {'classes':>8} "
          f"{'F1 (fixed)':>11} {'F1 (as reported)':>17} {'precision':>10} {'recall':>8}")
    for r in rows:
        print(f"{r['arm']:>22} {r['rows_scored']:>7} {r['off_vocab_rows']:>10} {r['n_classes_seen']:>8} "
              f"{r['f1']:>11.3f} {r['f1_unrestricted_macro']:>17.3f} "
              f"{r['precision']:>10.3f} {r['recall']:>8.3f}")

    out = args.out or args.dump_dir / "extract_rescored.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
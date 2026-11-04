#!/usr/bin/env python3
"""Calibrate Skein filter-proxy thresholds against main-model verdicts.

Run the same deterministic filter query twice with FLOCK_VERDICT_DUMP set to
separate files: once without semantic_filter_proxy_model (main verdicts), then
with the proxy model and default thresholds (-1, 2) (all proxy scores, with
every row forwarded). Use SET threads=1 and identical input order for both
runs so the diagnostic row IDs align. The labels here are the main model's
answers; evaluate the final cascade separately against ground truth.
"""

import argparse
from bisect import bisect_left, bisect_right
import json
import math
import random
from pathlib import Path


def read_dump(path: Path, want_scores: bool) -> dict[int, tuple[float | bool, int | None]]:
    rows: dict[int, tuple[float | bool, int | None]] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            verdict = record["v"]
            if want_scores:
                if type(verdict) not in (int, float):
                    continue  # Main-model decisions also appear in a proxy run.
                value = float(verdict)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid proxy score at row {record['id']}: {verdict}")
            else:
                if verdict is None:
                    value = True  # PhysicalSemFilter::ParseVerdict keeps missing items.
                elif type(verdict) is bool:
                    value = verdict
                else:
                    continue
            row_id = int(record["id"])
            if row_id in rows:
                raise ValueError(f"Duplicate row ID {row_id} in {path}")
            rows[row_id] = (value, record.get("row_hash"))
    if not rows:
        raise ValueError(f"No {'proxy scores' if want_scores else 'main labels'} found in {path}")
    return rows


def calibrate(scores: list[float], labels: list[bool],
              precision_target: float, recall_target: float) -> dict:
    """Find thresholds minimizing main-model rows under empirical P/R targets."""
    n = len(scores)
    total_positive = sum(labels)
    unique_scores = sorted(set(scores))
    ordered = sorted(zip(scores, labels))
    ordered_scores = [score for score, _ in ordered]
    positive_prefix = [0]
    for _, label in ordered:
        positive_prefix.append(positive_prefix[-1] + label)
    best = None
    for negative in [-1.0, *unique_scores]:
        low_count = bisect_right(ordered_scores, negative)
        dropped_positive = positive_prefix[low_count]
        recall = 1 - dropped_positive / total_positive if total_positive else 1.0
        if recall < recall_target:
            continue
        for positive in [*unique_scores, 2.0]:
            if negative >= positive:
                continue
            high_start = bisect_left(ordered_scores, positive)
            false_positive = (n - high_start) - (total_positive - positive_prefix[high_start])
            true_positive = total_positive - dropped_positive
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
            if precision < precision_target:
                continue
            main_rows = high_start - low_count
            candidate = (main_rows, -recall, -precision, negative, positive)
            if best is None or candidate < best[0]:
                best = (candidate, precision, recall)
    assert best is not None  # (-1, 2) sends all rows to the main model.
    (_, _, _, negative, positive), precision, recall = best
    main_rows = sum(negative < score < positive for score in scores)
    return {
        "sample_rows": n,
        "main_model_rows": main_rows,
        "main_model_fraction": main_rows / n,
        "empirical_precision_vs_main": precision,
        "empirical_recall_vs_main": recall,
        "semantic_filter_proxy_negative_threshold": negative,
        "semantic_filter_proxy_positive_threshold": positive,
    }


def evaluate(scores: list[float], labels: list[bool],
             negative: float, positive: float) -> dict:
    predicted = [False if score <= negative else True if score >= positive else label
                 for score, label in zip(scores, labels)]
    tp = sum(p and label for p, label in zip(predicted, labels))
    fp = sum(p and not label for p, label in zip(predicted, labels))
    positives = sum(labels)
    main_rows = sum(negative < score < positive for score in scores)
    return {
        "heldout_rows": len(scores),
        "heldout_main_model_fraction": main_rows / len(scores),
        "heldout_precision_vs_main": tp / (tp + fp) if tp + fp else 1.0,
        "heldout_recall_vs_main": tp / positives if positives else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-dump", type=Path, required=True)
    parser.add_argument("--main-dump", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--recall-target", type=float, default=0.95)
    parser.add_argument("--train-fraction", type=float, default=0.2,
                        help="Fraction used to choose thresholds; remaining rows are held out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.precision_target <= 1 or not 0 < args.recall_target <= 1:
        parser.error("precision and recall targets must be in (0, 1]")
    if not 0 < args.train_fraction < 1:
        parser.error("train fraction must be in (0, 1)")
    scores = read_dump(args.proxy_dump, True)
    labels = read_dump(args.main_dump, False)
    if scores.keys() != labels.keys():
        parser.error("Proxy and main dumps have different row IDs; use identical input and SET threads=1")
    mismatched = [row_id for row_id in scores if scores[row_id][1] is not None
                  and labels[row_id][1] is not None and scores[row_id][1] != labels[row_id][1]]
    if mismatched:
        parser.error(f"Input rows differ for {len(mismatched)} row IDs; use a stable input order")
    ids = sorted(scores)
    if len(ids) < 2:
        parser.error("At least two rows are needed for calibration and holdout")
    random.Random(args.seed).shuffle(ids)
    split = max(1, min(len(ids) - 1, round(len(ids) * args.train_fraction)))
    train_ids, heldout_ids = ids[:split], ids[split:]
    result = calibrate([scores[i][0] for i in train_ids], [labels[i][0] for i in train_ids],
                       args.precision_target, args.recall_target)
    result.update(evaluate([scores[i][0] for i in heldout_ids], [labels[i][0] for i in heldout_ids],
                           result["semantic_filter_proxy_negative_threshold"],
                           result["semantic_filter_proxy_positive_threshold"]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

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
from pathlib import Path


def read_dump(path: Path, want_scores: bool) -> dict[int, float | bool]:
    rows: dict[int, float | bool] = {}
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
            rows[row_id] = value
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-dump", type=Path, required=True)
    parser.add_argument("--main-dump", type=Path, required=True)
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--recall-target", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.precision_target <= 1 or not 0 < args.recall_target <= 1:
        parser.error("precision and recall targets must be in (0, 1]")
    scores = read_dump(args.proxy_dump, True)
    labels = read_dump(args.main_dump, False)
    if scores.keys() != labels.keys():
        parser.error("Proxy and main dumps have different row IDs; use identical input and SET threads=1")
    ids = sorted(scores)
    print(json.dumps(calibrate([scores[i] for i in ids], [labels[i] for i in ids],
                               args.precision_target, args.recall_target), indent=2))


if __name__ == "__main__":
    main()

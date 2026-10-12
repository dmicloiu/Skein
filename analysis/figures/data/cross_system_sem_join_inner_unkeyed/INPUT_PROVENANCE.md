# cross_system_sem_join_inner_unkeyed — input provenance

A semantic filter over an **inner unkeyed join** (Q108, `Qwen/Qwen2.5-7B-Instruct`, TP=1,
one GH200, one co-located cold vLLM per (arm, rep)), across four systems. Every arm judges the
same 4,096 candidate pairs with a sem_filter. Flock builds them through its unkeyed rewrite, and LOTUS/PZ use
`src/scenario/movie/unkeyed_join_q108.py` in sembench. This is a **candidate-matched sem_filter comparison,
not a native sem_join**. The prompt is the same as Q107's. Sibling: `../cross_system_sem_join_inner_keyed/`.

| arm | job | R | in-flight / threads | engages |
|---|---|---|---|---|
| `flock_op_r1` (Skein) | 3647759 | 1 | `in_flight_cap=128` | `TryRewriteUnkeyedJoin` → async `PhysicalSemFilter` over a CROSS_PRODUCT |
| `flock_scalar_r1` | 3647759 | 1 | `DUCKDB_THREADS=16` | stock scalar `llm_filter` |
| `lotus` | 3647759 | — | `max_batch_size=128` | `sem_filter` |
| `palimpzest` | 3647759 | — | 128 workers | `LLMFilter` |

There are 3 reps per arm, and all four arms ran in one job on the same node. Each rep gets a cold vLLM and a wiped
flock catalog, then an untimed Q107 `sf_20` warm-up (190 pairs, same prompt; Q108 has no
candidates at `sf_20`) on the same server before `metrics_before`. All 12 warm-ups succeeded.
Revisions: sembench `ad29664`, flock `14ea993b` (loadable rebuilt in job 3646735, so it
includes the unkeyed rewrite 9e0c09e8). Smoke run: job 3647675 (op_r1/lotus/PZ, 1 rep, consistent with the numbers below).

## Q108 — what is being measured

Q108 takes the first 64 distinct reviews (`ORDER BY reviewId, reviewText`) of
`ant_man_and_the_wasp_quantumania` and the first 64 of `taken_3`. The only join condition is
`JOIN ... ON llm_filter(r1, r2)`, which gives **4,096** directed cross-movie pairs. The evaluator scores
`(id1, reviewId1, id2, reviewId2)`. Gold: **2,104 of 4,096** pairs (base rate 0.514).

## Results (mean ± sd over 3 reps; pairs/s = 4,096 / mean time)

| arm | time | pairs/s | P | R | F1 | vLLM concurrency | prompt / gen tok per req |
|---|---|---|---|---|---|---|---|
| `flock_op_r1` | **29.40 s ± 0.19** (0.6%) | 139.3 | 0.664 | 0.876 | 0.7553 ± 0.0007 | ~122 | 560 / 8 |
| `flock_scalar_r1` | **220.84 s ± 5.23** (2.4%) | 18.5 | 0.664 | 0.880 | 0.7565 ± 0.0003 | ~1 | 560 / 8 |
| `lotus` | **41.35 s ± 0.70** (1.7%) | 99.1 | 0.656 | 0.928 | 0.7687 ± 0.0006 | ~62 | 195 / 81 |
| `palimpzest` | **41.63 s ± 1.15** (2.8%) | 98.4 | 0.802 | 0.863 | 0.8317 ± 0.0011 | ~91 | 488 / 100 |

Skein vs scalar: **7.51x** at equal quality (Jaccard of predicted pairs 0.983). P/R/F1 was
recomputed from the per-rep `<arm>_rep<r>_<sys>_Q108.csv` against gold (no duplicate pairs).
Concurrency = vLLM `e2e_request_latency_seconds_sum` delta / wall time. Token counts come from
the `metrics_{before,after}` diffs (the PZ harness `token_usage` counts completion tokens only).

## Read these rules before using any number here

- **Throughput denominator is 4,096 pairs, not `SCALE`.** Q108 pins two movies, so
  `SCALE_FACTOR=2000` changes only the base scan.
- **The 7.51x is mostly the stock plan running serially.** Each Q108 input is 64 rows, a single
  DataChunk, so the scalar `llm_filter` runs on one thread (`thread_count: 1`, ~1 request in
  flight). In Q107 the scalar arm reached ~7 in flight and the speedup was 2.15x. Read it as "stock
  Flock cannot parallelise a small unkeyed join; Skein keeps ~122 requests in flight". It is not a
  pure per-request overhead win.
- **Quality is above chance for every arm.** The always-positive baseline scores P=0.514 / F1=0.679.
  As in Q107, F1 tracks generated tokens per request (Flock 8, LOTUS ~81, PZ 100). That is a
  hypothesis: the prompts were not compared.
- **The arms do not do equal work per request.** LOTUS and PZ generate 10–12x more tokens than
  Flock. End-to-end time is not isolated operator overhead.
- **PZ ran cleanly here** (no Q107-style client stall). PZ is still the least stable arm in its
  predictions (rep-to-rep Jaccard ≥ 0.917, vs ≥ 0.992 for the others).
- **Operator engagement is confirmed.** All three `flock_op_r1` reps report `thread_count: 0`
  in `flock_metrics.json` (zero scalar `llm_filter` calls) and exactly 4,096 vLLM requests.

## Data fix: stale tail in three rep-1 CSVs

As they came out of the job, `flock_op_r1`, `lotus` and `palimpzest` rep-1 `_Q108.csv` were exactly as long as
the smoke run's (3647675) CSVs for those arms. The first `row_count` rows (the harness JSON) were this run's
result, all distinct, and the remaining rows were identical, row for row, to the tail of the smoke file. The
copies committed here are **truncated to `row_count`** (2784→2775, 2978→2973, 2277→2259).
`flock_scalar_r1` (not in the smoke run) and reps 2–3 were unaffected. The harness scores the file on disk,
so the **PZ rep-1 harness `f1_score` 0.832990 is contaminated**: the true value is 0.832913 (Δ −7.7e-5).
The JSON is kept as produced, and the table above uses the corrected value. The other harness F1s are exact.
Root cause unverified: sembench writes with `DataFrame.to_csv` (mode `w`). Re-runs that produce a shorter
`raw_results/<sys>/Q<n>.csv` can be affected, so check `len(csv) == row_count`.

## Not included here

- `vllm-*.log` and `run_*.log` (12 each) are left under `analysis/results/.../3647759/` on
  Clariden (repo policy, `[REPOSITORY-CLEAN-UP]`, 1d421c50).
- Per-thread `flock_metrics.json` and `vllm_sampler.csv` live in the **sembench** repo under
  `files/movie/metrics/flockmtl/<arm>_rep<n>/Q108/`.

# cross_system_sem_join_inner_unkeyed_materialized — input provenance

A controlled plan-layout counterpart to Q108 (`../cross_system_sem_join_inner_unkeyed/`). Q109 materialises
Q108's exact 4,096 candidate pairs into a temporary table first, then applies the same `llm_filter` as a plain
FILTER over that table. The question: in Q108 Skein kept ~122 requests in flight and stock Flock kept ~1. Does
the scalar arm still run serially when its predicate is a FILTER over pre-produced pairs instead of a join condition?
Model: `Qwen/Qwen2.5-7B-Instruct`, TP=1, one GH200, one co-located cold vLLM per (arm, rep).

| arm | job | R | in-flight / threads | engages |
|---|---|---|---|---|
| `flock_op_r1` (Skein) | 3691092 | 1 | `in_flight_cap=128` | EXTENSION operator (async `PhysicalSemFilter`) over the temp-table scan |
| `flock_scalar_r1` | 3691092 | 1 | `DUCKDB_THREADS=16` | stock scalar `llm_filter` in a FILTER over the temp-table scan |

There are 3 reps per arm, and both arms ran in one job on the same node. Each rep gets a cold vLLM and a wiped
flock catalog. An untimed Q107 `sf_20` warm-up (190 pairs, same prompt) runs on the same server before
`metrics_before`. All 6 warm-ups succeeded.
Command (from sembench): `QUERY=109 WARM_QUERY=107 WARM_SCALE=20 SCALE_FACTOR=2000 N_REPS=3
ARMS='flock_scalar_r1 flock_op_r1' SCALAR_BATCH=1 IN_FLIGHT=128 sbatch --export=ALL slurm/cross_system_analysis_clariden.sh`.
Revisions: sembench `22e557f` (Q109 added in `188d2d9`), flock `84dcf20f` (loadable built in job 3646735).
Smoke run: job 3691021 (both arms, 1 rep: scalar 213.0 s, op 29.0 s, the same checks passed).

## Q109 — what is being measured

`sembench/files/movie/query/flockmtl/Q109.sql`. `CREATE TEMP TABLE q109_candidates` holds the cross product of the
first 64 distinct reviews (`ORDER BY reviewId, reviewText`) of `ant_man_and_the_wasp_quantumania` and the first 64 of
`taken_3`, the same 4,096 directed pairs as Q108. Then `SELECT ... FROM q109_candidates WHERE llm_filter(...)` uses
Q108's prompt and its `review1`/`review2` context columns. A MATERIALIZED CTE is not enough: DuckDB can push the volatile
predicate into the producer, so the temp table forces a statement boundary. Gold (`_evaluate_q109`) is identical
to Q108's: **2,104 of 4,096** pairs.

## Results (mean ± sd over 3 reps; pairs/s = 4,096 / mean time)

| arm | time | pairs/s | P | R | F1 | vLLM concurrency | prompt / gen tok per req |
|---|---|---|---|---|---|---|---|
| `flock_op_r1` | **28.64 s ± 0.71** (2.5%) | 143.0 | 0.664 | 0.876 | 0.7554 ± 0.0004 | ~122 | 560 / 8 |
| `flock_scalar_r1` | **223.07 s ± 0.62** (0.3%) | 18.4 | 0.664 | 0.880 | 0.7568 ± 0.0005 | ~1 | 560 / 8 |

Skein vs scalar: **7.79x** at equal quality (Jaccard of predicted pairs 0.983). Q108: 7.51x (29.40 s / 220.84 s).
P/R/F1 was recomputed from the per-rep `_Q109.csv` against gold. Every CSV has exactly `row_count` distinct
rows, so there is no stale tail. Concurrency = vLLM `e2e_request_latency_seconds_sum` delta / wall time. Token counts come from
the `metrics_{before,after}` diffs.

## Read these rules before using any number here

- **Answer: the FILTER placement does not help stock Flock.** The scalar arm still has ~1 request in flight
  and is 1% slower than in Q108. All three scalar reps report `thread_count: 1` and 4,096 `llm_filter` calls
  in `flock_metrics.json`. The 64x64 input is too small for DuckDB to split it across threads in either plan shape.
- **Same work as Q108, request for request.** Every run issued exactly 4,096 vLLM requests (all `stop`; 0
  abort/error/length) with 2,294,848 prompt and 32,768 generated tokens, identical to Q108. The prompt and per-pair
  context are unchanged. Predicted-pair Jaccard vs the same Q108 rep is 0.993–0.997.
- **Operator engagement is confirmed.** All three `flock_op_r1` reps report `thread_count: 0` (zero scalar
  `llm_filter` calls).
- **Throughput denominator is 4,096 pairs, not `SCALE`.** `SCALE_FACTOR=2000` changes only the base scan.
- Times include building the temp table (negligible next to LLM time, but not separated out).

## Contents

- `<arm>_rep<r>_flockmtl.json` (harness metrics + P/R/F1) and `<arm>_rep<r>_flockmtl_Q109.csv`.
- `metrics_{before,after}_<arm>_rep<r>.txt`: vLLM `/metrics`, taken by the Slurm script around the measured query.
- `telemetry_<arm>_rep<r>/`: `flock_metrics.json` (per-thread Flock debug metrics), `vllm_sampler.csv`, and the
  harness's own `vllm_metrics_{before,after}.txt`. The harness snapshots are taken a moment apart from the script's,
  so the two files differ slightly. The Slurm script does not copy this directory itself. It was copied from
  sembench `files/movie/metrics/flockmtl/<arm>_rep<r>/Q109/`, which the next run with the same tag overwrites.
- `warmup_<arm>_rep<r>.log`.

## Not included here

- `vllm-*.log` and `run_*.log` (6 each) are left under `analysis/results/.../3691092/` on
  Clariden (repo policy, `[REPOSITORY-CLEAN-UP]`, 1d421c50).

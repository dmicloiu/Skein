# cross_system_sem_join_inner_keyed — input provenance

A semantic filter over an **inner keyed self-join** (Q107, `Qwen/Qwen2.5-7B-Instruct`, TP=1,
one GH200, one co-located cold vLLM per (arm, rep)), across four systems. Every arm judges the
same pre-produced candidate pairs (`src/scenario/movie/keyed_join_q107.py` in sembench) with a
sem_filter. This is a **candidate-matched sem_filter comparison, not a native sem_join**.

| arm | job | R | in-flight / threads | engages |
|---|---|---|---|---|
| `flock_op_r1` (Skein) | 3622401 | 1 | `in_flight_cap=128` | async `PhysicalSemFilter` |
| `flock_scalar_r1` | 3622401 | 1 | `DUCKDB_THREADS=16` | stock scalar `llm_filter` |
| `lotus` | 3622401 | — | `max_batch_size=64` | `sem_filter` |
| `palimpzest` | 3622401 | — | 128 workers | `LLMFilter` |

There are 3 reps per arm, and all four arms ran in one job on the same node. Each rep gets a cold vLLM and a wiped
flock catalog, then an untimed `sf_20` warm-up (190 pairs) on the same server before
`metrics_before`. All 12 warm-ups succeeded (the Lotus "Error calculating completion cost"
line is a cost-accounting warning only). Revisions: sembench `3e802e5`, flock `4349aa24`.
This supersedes job 3552248 (Flock arms only; Skein 62.77 s, scalar 128.88 s, consistent
with the numbers below).

## Q107 — what is being measured

Q107 is Q7 (all opposite-sentiment review pairs for `ant_man_and_the_wasp_quantumania`,
no LIMIT) over a **deduplicated** pair space. `reviews_2000` stores every one of that
movie's 256 review rows twice (128 distinct `reviewId`, identical `reviewText` and
`scoreSentiment`), and Q7's `r1.reviewId <> r2.reviewId` emits both orderings. So Q7
evaluates **65,280** predicates to decide **8,128** distinct unordered pairs, 8x redundant.
Q107's `SELECT DISTINCT` + `r1.reviewId < r2.reviewId` evaluates 8,128 predicates, with **identical**
precision/recall/F1 (the evaluator scores unordered pairs). Gold: 4,092 of 8,128 pairs.

## Results (mean ± sd over 3 reps; pairs/s = 8,128 / mean time)

| arm | time | pairs/s | P | R | F1 | vLLM concurrency | prompt / gen tok per req |
|---|---|---|---|---|---|---|---|
| `flock_op_r1` | **60.68 s ± 0.54** (0.9%) | 134.0 | 0.527 | 0.692 | 0.5981 ± 0.0004 | ~124 | 578 / 8 |
| `flock_scalar_r1` | **130.72 s ± 7.69** (5.9%) | 62.2 | 0.526 | 0.689 | 0.5963 ± 0.0003 | ~7 | 578 / 8 |
| `lotus` | **77.30 s ± 1.00** (1.3%) | 105.1 | 0.660 | 0.738 | 0.6966 ± 0.0001 | ~62 | 213 / 73 |
| `palimpzest` | 643.46 / 155.00 / 96.21 s | see below | 0.736 | 0.724 | 0.7305 ± 0.0012 | 14 / 59 / 94 | 507 / 124 |

Skein vs scalar: **2.15x** at equal quality. P/R/F1 was recomputed from the per-rep
`<arm>_rep<r>_<sys>_Q107.csv` against gold and matches the harness JSON exactly (no duplicate
pairs). Concurrency = vLLM `e2e_request_latency_seconds_sum` delta / wall time. Token
counts come from the `metrics_{before,after}` diffs (the PZ harness `token_usage` counts completion
tokens only, so do not use it).

## Read these rules before using any number here

- **Throughput denominator is 8,128 pairs, not `SCALE`.** Q107 pins one movie, so
  `SCALE_FACTOR` changes only the base scan. The runner's own `throughput_rows_per_s`
  divides by survivors, so ignore it.
- **Quality is near or below chance.** The always-positive baseline scores P=0.503 / F1=0.670. Both
  Flock arms (F1 ~0.597) are **below** it; Lotus (0.697) and PZ (0.730) are only just above it.
  F1 tracks generated tokens per request (Flock 8, Lotus 73, PZ 124), which suggests
  reasoning-before-answer helps. That is a hypothesis: the prompts were not compared.
- **The arms do not do equal work per request.** Lotus and PZ generate 9–15x more tokens than
  Flock, and PZ prompts are 2.4x Lotus's. End-to-end time is not isolated operator overhead.
- **PZ timings are not a clean distribution.** In rep 1, vLLM served all 8,128 requests in ~95 s
  (14:17:45–14:19:21) and then sat idle for ~9 min. The stall is client-side, and coincides with
  one LiteLLM error (cause unverified; possibly the 600 s default timeout). Rep 2 has one
  request with `finished_reason=length` (vLLM shows 8,126 stop + 1 length) and a ~60 s
  single-request tail. Only rep 3 (96.21 s, 84.5 pairs/s) is clean. PZ is also the least
  stable arm (rep-to-rep Jaccard ~0.87 vs >= 0.98 for the others).
- **The scalar arm's variance is thread allocation.** DuckDB spawned 12 / 12 / 13 threads.
  For the straggler analysis of the scalar path see the 3552248 notes (git history of this file).
- **Operator engagement is confirmed.** All three `flock_op_r1` reps report `thread_count: 0`
  in `flock_metrics.json` (zero scalar `llm_filter` calls), with 1 DuckDB thread and ~124
  concurrent requests (cap 128). Skein and scalar agree on 98.7% of predicted pairs (Jaccard).
- **Prefix-cache hits include warm-up prefill** (190 of 8,128 pairs, ~2%).

## Not included here

- `vllm-*.log` and `run_*.log` (12 each) are left under `analysis/results/.../3622401/` on
  Clariden. `run_*.log` is excluded by repo policy (`[REPOSITORY-CLEAN-UP]`, 1d421c50).
- Per-thread `flock_metrics.json` and `vllm_sampler.csv` live in the **sembench** repo under
  `files/movie/metrics/flockmtl/<arm>_rep<n>/Q107/`. The scalar rep 2 and rep 3
  `flock_metrics.json` files have trailing garbage after the first JSON object (write race), so read them with
  `json.JSONDecoder().raw_decode`.

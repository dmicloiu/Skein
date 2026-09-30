# cross_system_sem_join_inner_keyed — input provenance

Flock's async `PhysicalSemFilter` (Skein) vs the stock scalar `llm_filter`, on a semantic
filter over an **inner keyed self-join** (Q107, `Qwen/Qwen2.5-7B-Instruct`, TP=1, one
GH200, one co-located cold vLLM per (arm, rep)).

| arm | job | R | in-flight / threads | engages |
|---|---|---|---|---|
| `flock_op_r1` | 3552248 | 1 | `in_flight_cap=128` | async `PhysicalSemFilter` |
| `flock_scalar_r1` | 3552248 | 1 | `DUCKDB_THREADS=16` | stock scalar `llm_filter` |

3 reps per arm, both arms in one job (same node), cold vLLM + wiped flock catalog and an
untimed `sf_20` warm-up (190 pairs) per rep.

## Q107 — what is being measured

Q107 is Q7 (all opposite-sentiment review pairs for `ant_man_and_the_wasp_quantumania`,
no LIMIT) over a **deduplicated** pair space. `reviews_2000` stores every one of that
movie's 256 review rows twice (128 distinct `reviewId`, identical `reviewText` and
`scoreSentiment`), and Q7's `r1.reviewId <> r2.reviewId` emits both orderings — so Q7
evaluates **65,280** predicates to decide **8,128** distinct unordered pairs, 8x redundant.
Q107's `SELECT DISTINCT` + `r1.reviewId < r2.reviewId` evaluates 8,128, with **identical**
precision/recall/F1 (the evaluator scores unordered pairs). Verified: the two gold sets
are the same 4,092 pairs. All six runs here issued exactly 8,128 vLLM requests.

## Results

| arm | time (mean ± sd) | pairs/s | prompt tok/s | F1 |
|---|---|---|---|---|
| `flock_op_r1` | **62.77 s ± 0.19** (0.3%) | 129.5 | 74,800 | 0.5973 ± 0.00013 |
| `flock_scalar_r1` | **128.88 s ± 18.67** (14.5%) | 63.1 | 37,300 | 0.5976 ± 0.00007 |

**Speedup 2.05x** (per-rep range 1.75x .. 2.36x) at equal quality (dF1 = -0.0003).

## Read these rules before using any number here

- **Throughput denominator is 8,128 pairs, not `SCALE`.** SKEIN.md's `SCALE / execution_time`
  convention does not apply: Q107 pins one movie, so `SCALE_FACTOR` changes only the base
  scan. `sf_2000` and `sf_300000` both hold 256 ant_man rows / 128 distinct reviewIds — this
  query's semantic cost does not scale with SF. The runner's own `throughput_rows_per_s`
  divides by survivors; ignore it.
- **The scalar arm is not serial, and its variance is a straggler effect.** DuckDB spawned
  12 / 13 / 16 threads across the three reps with 4.8x / 8.9x / 4.9x morsel imbalance. Total
  time equals the busiest thread's busy time to within 0.2 s in every rep (148.4/148.57,
  134.1/134.28, 103.7/103.80). The 14.5% run-to-run spread is non-deterministic thread
  allocation, not measurement noise.
- **A balanced scalar path would roughly match the operator — untested inference.** Dividing
  each rep's busiest-thread time by its call count gives 86.9 / 88.8 / 125.7 ms per request;
  at perfect balance the same thread counts predict 58.8 / 55.5 / 63.9 s (mean ~59 s) against
  the operator's 62.8 s. This assumes latency holds constant under rebalancing. If it
  survives an `in_flight` sweep, the operator's win here is eliminating morsel skew rather
  than raising concurrency. Do not state it as measured.
- **Quality is near-chance for both arms.** Gold is 4,092 of 8,128 pairs (50.34% base rate);
  both arms score P~0.53 / R~0.69 / F1~0.597, i.e. accuracy ~53%. The throughput comparison
  is sound (same prompt, same model, same pair space), but "equal quality" here means
  equally weak — 7B at R=1 barely solves this pairwise task.
- **Operator engagement is confirmed, not assumed.** All three `flock_op_r1` reps report
  `thread_count: 0` in `flock_get_debug_metrics()` (zero scalar `llm_filter` calls) and the
  vLLM sampler tops out at exactly 128 concurrent — `in_flight_cap`. Independently, DuckDB
  1.5.1 `EXPLAIN` shows the `FILTER` surviving above the `HASH_JOIN` for this shape, pinned
  by `Case12_WhereAboveJoin` in `optimizer_rewrite_test.cpp`.

## Not included here

- `vllm-*.log` (6 files, 4.8 MB) and `run_*.log` (6 files) — left under
  `analysis/results/.../3552248/` on Clariden. `run_*.log` is excluded by repo policy:
  `[REPOSITORY-CLEAN-UP]` (1d421c50) deleted 843 of them from `figures/data/`, and per-run
  logs are kept only at the `thesis-submitted` tag.
- Per-thread `flock_metrics.json` and the 250 ms `vllm_sampler.csv`, which are the evidence
  for the straggler finding above: those live in the **sembench** repo under
  `files/movie/metrics/flockmtl/<arm>_rep<n>/Q107/`.
- Job `3552049` (same config, 1 rep) — its single scalar run took 152.5 s, at the slow end
  of the distribution, which is why it reported 2.5x rather than 2.05x. Superseded.

# sem_filter operator A/B — conclusion

**Claim under test.** flock's scalar `llm_filter` was re-expressed as a Tier-3 async
`PhysicalSemFilter` that decouples LLM HTTP concurrency from DuckDB's morsel/thread
parallelism. Stock flock's concurrency is `min(#threads, #morsels)`; the operator
sustains `in_flight_cap` concurrent requests regardless of threads/morsels (gated by
`SET semantic_rewrite_enabled`, default on; `llm_filter` is marked VOLATILE so a
single-column `WHERE llm_filter(...)` survives pushdown and reaches the operator).

**Setup.** `analysis/slurm/sem_filter_ab_clariden.sh` + the
`flock_sem_filter_vllm_integration` driver, one GH200 + one cold vLLM
(Qwen/Qwen2.5-7B-Instruct, `--enable-prefix-caching`, xgrammar
`disable_any_whitespace`). Same query run two ways — rewrite **on** (operator) vs
**off** (scalar) — across a threads sweep {1,2,4,8,16}, `batch_size(R)=32`,
`in_flight_cap=128`. Workload: `sembench/files/movie/data/sf_2000/Reviews.csv`,
column `reviewText`, prompt *"The following movie review is clearly positive."*,
2000 rows. Fairness alignment on both arms: chat endpoint, `temperature=0`,
`max_tokens=16·R`, same `boolean_array` guided-decoding schema, null→pass. Cold
fleet per (rewrite, threads) measurement; burn-in untimed.

Figures (regenerate with `python analysis/plot_sem_filter_ab.py`):
`analysis/figures/sem_filter_threads.{png,pdf}` (throughput vs threads) and
`analysis/figures/sem_filter_tradeoff.{png,pdf}` (per-request latency / throughput
/ survivors at one thread). Both land in the paper as Figs 8-9 of
`chapters/physicaloperators.tex` (§ Evaluation: the Semantic Filter), with the
run configuration carried in the LaTeX captions rather than baked into the images.

## Results

| threads | operator rows/s | scalar rows/s | speedup | op passes | sc passes | op s | sc s |
|--------:|----------------:|--------------:|--------:|----------:|----------:|-----:|-----:|
| 1  | 592.0 | 77.3 | 7.66× | 584 | 569 | 3.38 | 25.88 |
| 2  | 584.5 | 76.7 | 7.62× | 584 | 569 | 3.42 | 26.07 |
| 4  | 589.2 | 76.8 | 7.68× | 584 | 569 | 3.39 | 26.06 |
| 8  | 591.8 | 77.0 | 7.69× | 584 | 569 | 3.38 | 25.98 |
| 16 | 592.5 | 76.8 | 7.72× | 584 | 569 | 3.38 | 26.05 |

**Headline — morsel/thread decoupling, quantified.** Both curves are *flat* across a
16× thread sweep. The operator holds ~590 rows/s and the scalar ~77 rows/s at every
thread count, for a **thread-independent ~7.7× speedup**. The scalar's flatness is
not a defect: with 2000 rows < 2048 the table is a **single DuckDB DataChunk**, so
`min(#morsels, #threads)` collapses to **1** regardless of `SET threads` — the scalar
is strictly serial. The vLLM engine logs confirm this directly: every scalar run
reports `Running: 1 reqs, Waiting: 0 reqs` throughout. The operator ignores that
structure entirely and drives the endpoint concurrently. This is the cleanest
possible statement of the decoupling claim: **when DuckDB hands the scalar exactly
one unit of parallelism, the operator still saturates the LLM ~7.7×.**

**Correctness — the all-true bug is fixed.** Operator passes 584 (29.2%), scalar
passes 569 (28.4%); they agree to within **15 rows = 0.75%**. Before commit 485fc624
the scalar's openai schema was missing `required:["items"] + additionalProperties:false`,
so vLLM guided decoding could emit `{}` → no items → null → pass → the scalar passed
**all 2000 rows**. That survivor discrepancy is gone; both arms now select ~29% and
agree. The residual 0.75% is consistent with (a) the temp-0 noise floor — vLLM greedy
decoding is not bit-deterministic under different batch composition — and (b)
R=32-row composition sensitivity: a given row's verdict can depend on its neighbours
in the packed prompt, and operator/scalar batch the rows into prompts differently.
Per the experiment design we compare *aggregate survivor counts and quality*, not
exact survivor identity.

**Fairness corroboration (vLLM /metrics deltas).** Both arms issue a byte-for-byte
identical LLM workload — **64 requests** (63 timed + 1 burn-in), **115,021 prompt
tokens**, **4,512 generation tokens**, **14.0% prefix-cache hit rate** — verified from
the after-minus-before `/metrics` snapshots. The A/B therefore differs *only* in
dispatch/concurrency, not in what is sent to the model. The operator delivers that
identical token volume in 3.4s vs the scalar's 26s → ~7.7× the sustained token
throughput, i.e. ~7.7× the effective in-flight concurrency.

## Critical caveats (read before citing)

1. **The operator is *not* at its cap here, and ~590 rows/s is below the ~800 rows/s
   E5 ceiling.** The workload is only 2000/32 = **63 requests**, well under
   `in_flight_cap=128`, so the cap never binds. The operator fires a single 63-request
   *burst* that ramps and drains; the mean in-flight concurrency (`t_scalar/t_operator`
   ≈ 7.7, a Little's-law **lower bound**) reflects the GH200's batched-vs-single-stream
   throughput roofline for a 7B model (~8×), **not** the cap and **not** the 63-request
   peak. A larger scale factor (continuous request supply) would push the operator
   toward 800 rows/s.

2. **This run does not exhibit scalar thread-scaling.** With one DataChunk the scalar
   is pinned at the `min=1` floor at all thread counts, so the "scalar rises with
   threads toward its `min(#threads,#morsels)` plateau" half of the story is *not*
   demonstrated here — only the floor is. Showing the full picture (scalar climbing
   then plateauing, operator flat above it) requires **> 1 DataChunk**, i.e. SF ≳ 2048
   rows. The Phase-2 cross-system movie runs at higher SF are the natural place to
   exercise that regime.

3. **Variance.** Single run per cell (no repeats). The cross-arm consistency is high
   (operator 584.5–592.5 rows/s, scalar 76.7–77.3 rows/s — <1.5% spread), so the
   threads-flatness is robust, but per-cell error bars would need N repeats. The
   short 3.4s operator runs are also why no periodic vLLM engine-stat line was emitted
   for the operator arm (10s logging interval) — operator concurrency is inferred from
   token-throughput parity, not read directly.

## Bottom line

The async `PhysicalSemFilter` cleanly decouples LLM concurrency from DuckDB's
morsel/thread parallelism: a 16× thread sweep moves neither arm, and the operator
sustains a thread-independent ~7.7× throughput advantage over the scalar precisely in
the regime (one DataChunk) where DuckDB offers the scalar zero thread-parallelism. The
all-true scalar bug is fixed — both arms now agree on ~29% survivors to within 0.75%.
The result is genuine but scale-limited: at SF=2000 the operator runs a sub-cap burst
below its own ceiling, and scalar thread-scaling is not exercised; both want a larger
SF to show in full.

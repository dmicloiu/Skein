# 128 morsel / 128 thread A/B — COMPLETE (both arms)

This A/B point is now **complete**. The scalar arm was re-run (job 2604875) after the
`MetricsManager` mutex fix landed, and it ran clean — no segfault.

| arm                    | job     | rows/s | passes / 262144 | elapsed_s |
|------------------------|---------|--------|-----------------|-----------|
| operator (rewrite=on)  | 2595900 | 132.97 | 170634          | 1971.5    |
| scalar   (rewrite=off) | 2604875 | 115.95 | 170641          | 2260.8    |

**Result: the two arms do NOT converge at equal 128-way concurrency.** Even when the
scalar arm is given 128 DuckDB threads / 128 morsels — matching the operator's 128
in-flight request cap — the operator retains a **~1.15x** edge (132.97 vs 115.95
rows/s). So the operator's advantage is not purely a concurrency artefact; a residual
~15% comes from the rewrite itself. Pass counts match (170634 ≈ 170641), confirming
the two arms compute the same filter — the gap is throughput, not correctness.

The operator number also matches its own m8/m16 points (133.8 / 138.6 rows/s) → the
operator is bound by the 128 in-flight cap and indifferent to DuckDB thread count.

## History

The original scalar m128/t128 run (job 2595901) **segfaulted**: SIGSEGV in
`flock::LlmFilter::Execute` on a DuckDB worker thread, immediately after burn-in,
before any workload request reached vLLM. Root cause was an unsynchronized data race
on the shared `std::unordered_map`s in `MetricsManager` / `BaseMetricsManager` (no
locking in the metrics layer) under 128-way concurrent insert — a measurement-harness
bug, not a sem_filter computation bug. The fix added a `recursive_mutex` guarding the
metrics maps in `src/include/flock/metrics/base_manager.hpp`. Job 2604875 confirms the
fix: scalar 128/128 now completes.

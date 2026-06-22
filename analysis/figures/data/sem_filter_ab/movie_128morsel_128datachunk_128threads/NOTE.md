# 128 morsel / 128 thread A/B — OPERATOR ARM ONLY

This A/B point is **incomplete**: only the operator (rewrite=on) arm has data. The
scalar (rewrite=off) arm **segfaulted** and produced no result.

- **operator m128/t128** (job 2595900): 132.97 rows/s, 170634/262144 passes,
  elapsed 1971.5 s, sustained 128 in-flight requests. This matches the operator's
  m8/m16 numbers (133.8 / 138.6 rows/s) → operator is bound by the 128 in-flight
  request cap, indifferent to DuckDB thread count.
- **scalar m128/t128** (job 2595901): SIGSEGV in `flock::LlmFilter::Execute` on a
  DuckDB worker thread, immediately after burn-in, before any workload request
  reached vLLM. Root cause: unsynchronized data race on the shared
  `std::unordered_map`s in `MetricsManager` / `BaseMetricsManager` (no locking in
  the metrics layer) under 128-way concurrent insert. It is a measurement-harness
  bug, not a sem_filter computation bug, so the operator number above is valid.
  Core dump: `sembench/core_nid006896_232591`. Needs a mutex fix to the metrics
  maps before the scalar 128/128 number can be obtained.

The decisive "do operator and scalar converge at equal 128-way concurrency?"
question is therefore still OPEN.

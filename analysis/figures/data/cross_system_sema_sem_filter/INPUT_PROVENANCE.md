# cross_system_sema_sem_filter — input provenance

Sema (`v0.0.1 8c2d3bd2`, DuckDB CLI fork) as a third baseline on sem_filter (Q101, 2000 rows,
`Qwen/Qwen2.5-7B-Instruct`, TP=1, GH200), to be read against `cross_system_sem_filter/`
(Skein/flockmtl, LOTUS, Palimpzest). Same model, same co-located vLLM, same evaluator.

| arm | job | R | note |
|---|---|---|---|
| `sema_q101_r1` | 3174426 | 1 | Sema's default (all optimizers off) |
| `sema_q101_r32` | 3174427 | 32 | `semantic_batch_size=32` |
| `sema_q101_r1_aqe` | 3174867 | 1 | **only** delta vs `sema_q101_r1` is Sema's AQE layer |

3 reps per arm, one arm per SLURM job, cold vLLM + untimed burn-in per rep.

## Read these rules before using any number here

- **Campaign `31737xx` was DISCARDED, not superseded by choice.** Its six arm jobs shared three
  non-job-scoped harness paths (endpoint file, metrics, raw results), so arms were measured
  against each other's vLLM. Those dirs are retained under `analysis/results/` as evidence only
  and **must never be imported**. Everything here is the post-fix replacement.
- **rows/s is INPUT rows / `execution_time`** = `2000 / t`. Do **not** use the artefact field
  `throughput_rows_per_s` — that is *output* (post-predicate) rows / time and understates the
  filter by 43% (236.2 vs 413.7 for `sema_q101_r1`). `plot_cross_system_frontier.py:38`
  hardcodes `ROWS = 2000`, so the plot definition is input rows.
- **Use `computed_tok_s`, not raw `tok_s`.** `(prompt − prefix_cache_hits + gen) / t`. Raw
  prompt tokens count cache hits, and the cache-hit rate here swings 47.7% (R=1) → 5.3% (R=32).
- **rows/s is CONFOUNDED for Sema.** Sema ships x86-64 only, Clariden GPU nodes are aarch64, so
  its client runs on an x86 `xfer` node reaching vLLM over the fabric; every other arm's client
  is on the GH200 over loopback. Every artefact carries `rows_per_s_confounded: true`,
  `sema_exec_host`, `sema_exec_arch`. F1 and `computed_tok_s` (measured server-side) are clean.
- Sema is greedy (`temperature=0.0`, hardcoded) and guided by `response_format.type=json_schema`
  → the same xgrammar backend the jobs configure, so reps are near-deterministic and the
  burn-in matters (first query pays the xgrammar compile).

## Integrity — re-verified at import, 9/9 reps

Artefact `execution_time` == telemetry `wall_s` **exactly** (see `*_Q101.timing.json`), request
accounting exact (2000 at R=1 = rows; 63 at R=32 = `ceil(2000/32)`), one endpoint per job proven
on its own node. A green Slurm job proves nothing here — Sema exits 0 on error — so these two
checks are the only integrity signals.

## Medians over 3 reps (as `summarize_cross_system.py` would aggregate)

| arm | t (s) | ±rng/2 | rows/s | computed tok/s | cache% | req | F1 | P | R |
|---|---|---|---|---|---|---|---|---|---|
| `sema_q101_r1` | 4.834 | 0.210 | 413.7 | 25,197 | 47.7 | 2000 | 0.8632 | 0.989 | 0.766 |
| `sema_q101_r32` | 3.338 | 0.083 | 599.2 | 28,130 | 5.3 | 63 | 0.5705 | 0.877 | 0.423 |
| `sema_q101_r1_aqe` | 6.486 | 0.114 | 308.4 | 18,782 | 47.7 | 2000 | 0.8632 | 0.989 | 0.766 |

Two findings the arms exist to show: **batching costs Sema ~0.29 F1** (R=1 → R=32, recall
halves), and **Sema's own optimiser makes it 34% slower** at identical token counts and
unchanged F1 — its best configuration against Skein is its *unoptimised* one. The AQE arm's
2000 requests confirm the multiplexer did **not** override `semantic_batch_size`
(`semantic_filter_batch_size=4` would have shown ~500), so it is like-for-like with
`sema_q101_r1`. `enable_partial_deduction` is held OFF in both (it changes which rows are sent).

## Files

Per (arm, rep): `<arm>_rep<n>_sema.json` (artefact), `metrics_{before,after}_<arm>_rep<n>.txt`
(vLLM /metrics deltas), `<arm>_rep<n>_Q101_result.csv` (output rows),
`<arm>_rep<n>_Q101.timing.json` (integrity witness), `<arm>_rep<n>_vllm_sampler.csv` (scheduler
gauges), `run_`/`warmup_` logs. Dropped at import: Sema's self-built `.duckdb` (regenerable) and
the vLLM server stdout logs; both remain in
`analysis/results/cross_system_sema_qwen2-5-7b-instruct/<jobid>/`.

No `cross_system_summary.csv` here: `summarize_cross_system.py`'s `ARM_SYSTEM` has no sema
entries, so the summariser skips these arms until it is extended.

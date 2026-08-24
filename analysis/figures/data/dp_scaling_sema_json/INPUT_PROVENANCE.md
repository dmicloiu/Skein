# dp_scaling_sema_json — input provenance

Sema (`v0.0.1 8c2d3bd2`) data-parallel scale-out on Q101, **32000 rows (`SEMA_ROW_LIMIT`), R=1**,
`Qwen/Qwen2.5-7B-Instruct`, GH200: `n` independent vLLM replicas, each tp=1, each on its own GPU
and port, all bound 0.0.0.0. Counterpart to `dp_scaling_json/` (Skein).

| arm | job | replicas |
|---|---|---|
| `sema_dp_n1_r1_rows32000` | 3175102 | 1 |
| `sema_dp_n2_r1_rows32000` | 3175103 | 2 |
| `sema_dp_n4_r1_rows32000` | 3175104 | 4 |

3 reps per cell, one job per replica count, cold fleet + untimed burn-in per rep. Matches
Skein's DP regime (32000 rows, R=1). Shape-check job `3174718` is **not** imported (1-rep
plumbing), nor is anything from campaign `31737xx`.

## Read these rules before using any number here

- **P/R/F1 are BLANK BY CONSTRUCTION.** Every artefact carries `row_limit: 32000` and
  **`f1_invalid_row_limited: true`**: with a row cap the evaluator scores recall against rows
  Sema never saw, so the `f1_score` field (~0.147) is **meaningless and must not be reported** —
  same reason Skein's TP/DP summaries leave P/R/F1 blank. DP is a throughput axis only.
- **rows/s is INPUT rows / `execution_time`** = `32000 / t`, never `throughput_rows_per_s`.
- **Use `computed_tok_s`** = `(prompt − prefix_cache_hits + gen) / t`, summed over all replicas.
  Cache-hit rate is 44.5% in every cell.
- **rows/s is CONFOUNDED** (x86 `xfer` client over the fabric vs Skein's on-node loopback client;
  `rows_per_s_confounded: true`). It does not weaken the result below — that rests on request
  counts, not on timing.

## Integrity — re-verified at import, 9/9 reps

`execution_time` == telemetry `wall_s` exactly (`*_Q101.timing.json`), 32000 requests accounted
exactly, per-replica `/metrics` snapshots written for **every** endpoint (`_ep0`…`_ep3`).

## Medians over 3 reps

| n | t (s) | ±rng/2 | rows/s | computed tok/s | requests per replica (all 3 reps) |
|---|---|---|---|---|---|
| 1 | 68.464 | 0.224 | 467.4 | 29,427 | `[32000]` |
| 2 | 69.646 | 2.138 | 459.5 | 28,928 | **`[32000, 0]`** |
| 4 | 69.480 | 1.701 | 460.6 | 28,998 | **`[32000, 0, 0, 0]`** |

## The result: Sema cannot use replicas at all

**Every idle replica served exactly zero requests — not few, zero — at both n=2 and n=4, in
every one of the 9 reps.** Sema takes a single `llm_url` and has no replica awareness, so
data-parallel capacity is structurally unusable: four GPUs, three permanently idle. Throughput
is flat-to-slightly-negative (467.4 → 459.5 → 460.6 rows/s) across a 4× GPU increase, and the
~1.5% spread is inside the reps' own range.

This is the strongest and most robust result in the Sema set:

- **No rep count or scale can soften a zero.** The conclusion does not depend on the timing.
- The 68 s measurement window makes it the campaign's most reliable timing too — n=1 spread is
  ±0.224 s on 68.5 s (**0.3%**), against ~2% on the ~4.5 s sf_2000 arms.
- It also constrains the TP reading: n=1 sustains **467 rows/s for 68 s on one GPU**, inside the
  436–541 rows/s band of every sf_2000 TP cell, so Sema is not pinned by a client-side ceiling
  in that band. See `tp_scaling_sema_json/INPUT_PROVENANCE.md`.

## Files

Per (arm, rep): `<arm>_rep<n>_sema.json`, `metrics_{before,after}_<arm>_rep<n>_ep<k>.txt` (one
pair per replica — the zero-request evidence), `<arm>_rep<n>_Q101.timing.json`,
`<arm>_rep<n>_vllm_sampler.csv`, `run_`/`warmup_`/`fleet_` logs. Dropped at import: the
regenerable `.duckdb` and the per-replica vLLM server logs (3 MB each), both still in
`analysis/results/dp_scaling_sema_qwen2-5-7b-instruct/<jobid>/`. No result CSVs exist for these
runs (row-limited, so unscoreable).

No summary CSV: `summarize_dp.py`/`summarize_tp.py` parse Skein's `result_op_n<N>_tp<K>_c<CAP>`
filename shape, not sembench's `<arm>_rep<n>_sema.json`.

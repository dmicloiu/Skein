# tp_scaling_sema_json — input provenance

Sema (`v0.0.1 8c2d3bd2`) TP scaling on Q101, **2000 rows, R=1**, `Qwen/Qwen2.5-7B-Instruct`,
GH200, one vLLM replica sharded over `tp` GPUs. Counterpart to `tp_scaling_json/` (Skein).

| arm | job | tp |
|---|---|---|
| `sema_tp_tp1_r1_rowssf2000` | 3175099 | 1 |
| `sema_tp_tp2_r1_rowssf2000` | 3175100 | 2 |
| `sema_tp_tp4_r1_rowssf2000` | 3175101 | 4 |

3 reps per cell, **one job per tp value** (each cell is an independent cold-vLLM run, so the
split is scientifically free and lets the 1- and 2-GPU cells backfill). Matches Skein's TP main
regime: 2000 rows, R=1, 3 reps. **TP is R=1 throughout — never R=32**; the R sweep lives in
`cross_system_sema_sem_filter/`.

## Read these rules before using any number here

- **rows/s is INPUT rows / `execution_time`** = `2000 / t`, never the artefact's
  `throughput_rows_per_s` (output rows — 43% low on a filter).
- **Use `computed_tok_s`** = `(prompt − prefix_cache_hits + gen) / t`, the same formula as
  `summarize_tp.py:189`. Cache-hit rate is 47.7% in every cell, so raw `tok_s` is inflated.
- **rows/s is CONFOUNDED**: Sema is x86-64-only, so its client is an x86 `xfer` node over the
  fabric while Skein's is on the GH200 over loopback (`rows_per_s_confounded: true`).
  `computed_tok_s` and F1 are measured server-side / against shared gold and are clean.
- **Sema has no in-flight cap knob** (only `semantic_batch_size`, `llm_rate_limit`,
  `llm_max_burst_seconds`), so each Sema cell is one point where Skein has a cap curve. Plot Sema
  as a single line against Skein's best-cap-per-tp and report the missing knob as the finding.
- Campaign `31737xx` and the shape-check jobs `3173744` / `3174717` are **not** imported:
  `3173744` ran under the buggy pre-fix harness, and `3174717` was a 1-rep plumbing check whose
  TP reading (saturation at tp=2) was **disproved** by this 3-rep regime.

## Integrity — re-verified at import, 9/9 reps

`execution_time` == telemetry `wall_s` exactly (`*_Q101.timing.json`), 2000 requests per rep,
F1 flat as it must be for a throughput knob. The **TP-engaged gate** held:
`tensor_parallel_size=1/2/4` appears exactly once in each job's ep0 vLLM log (retained in
`analysis/results/tp_scaling_sema_qwen2-5-7b-instruct/<jobid>/`), so vLLM genuinely sharded.

## Medians over 3 reps

| tp | t (s) | ±rng/2 | rows/s | computed tok/s | req | F1 | step |
|---|---|---|---|---|---|---|---|
| 1 | 4.585 | 0.088 | 436.2 | 26,569 | 2000 | 0.8636 | — |
| 2 | 4.087 | 0.020 | 489.4 | 29,807 | 2000 | 0.8649 | 1.12× |
| 4 | 3.698 | 0.089 | 540.8 | 32,939 | 2000 | 0.8640 | 1.10× |

**Sema scales weakly but monotonically — it does not saturate.** vs Skein's operator at cap=128
(medians from `tp_scaling_json/tp_scaling_summary.csv`, rows=2000):

| | tp=1 | tp=2 | tp=4 | tp1→tp4 | parallel efficiency |
|---|---|---|---|---|---|
| Sema | 436.2 | 489.4 | 540.8 | **1.24×** | **31%** |
| Skein op | 140.0 | 213.4 | 302.8 | **2.16×** | **54%** |

Sema starts ~3× ahead in absolute rows/s — a pure token-budget effect (112.7 vs 497.8 prompt
tok/row; `rows/s = computed_tok/s ÷ computed_tok/row` closes for both systems with no residual)
— and the gap narrows as tp rises.

**Do not publish the 31%-vs-54% parallel-efficiency claim unqualified until heroic (262144 rows)
lands.** At sf_2000 the measured window is under 5 s and Sema's client is off-node, so the weak
slope could be a client limit rather than a GPU limit. Partial counter-evidence already exists:
`dp_scaling_sema_json` n=1 sustains 467 rows/s for 68 s on **one** GPU, inside the 436–541 band,
which favours the genuine-poor-GPU-scaling reading. State the confound if citing before heroic.

## Files

Per (arm, rep): `<arm>_rep<n>_sema.json`, `metrics_{before,after}_<arm>_rep<n>_ep0.txt`,
`<arm>_rep<n>_Q101.timing.json`, `<arm>_rep<n>_vllm_sampler.csv`, `run_`/`warmup_`/`fleet_` logs.
Dropped at import: the regenerable `.duckdb` and the vLLM server logs (kept in `analysis/results/`).

No summary CSV: `summarize_tp.py`'s `TAG_RE` only matches Skein's
`result_{op,scalar}_tp<K>[_c<CAP>].json` shape, not sembench's `<arm>_rep<n>_sema.json`.

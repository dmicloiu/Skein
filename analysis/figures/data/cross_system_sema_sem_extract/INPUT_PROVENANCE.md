# cross_system_sema_sem_extract — input provenance

Sema (`v0.0.1 8c2d3bd2`) on sem_extract (Q103, 2000 rows, `Qwen/Qwen2.5-7B-Instruct`, TP=1,
GH200), to be read against `cross_system_sem_extract/`. Same model, same co-located vLLM, same
evaluator (`_evaluate_q103`'s `canon()` substring-matches POSITIVE/NEGATIVE).

| arm | job | R | prompt |
|---|---|---|---|
| `sema_q103_verbatim_r1` | 3174428 | 1 | **verbatim flockmtl prompt** — true prompt parity |
| `sema_q103_verbatim_r32` | 3174429 | 32 | verbatim |
| `sema_q103_question_r1` | 3174430 | 1 | **question form** — disclosed prompt deviation |
| `sema_q103_question_r32` | 3174431 | 32 | question form |

3 reps per arm, one arm per SLURM job, cold vLLM + untimed burn-in per rep.

## Read these rules before using any number here

- **Campaign `31737xx` was DISCARDED** — six concurrent jobs shared three non-job-scoped harness
  paths (endpoint file, metrics, raw results) and measured against each other's vLLM. Q103 was
  the worst hit (4 jobs on one results file). Retained under `analysis/results/` as evidence
  only; **never import it**.
- **rows/s is INPUT rows / `execution_time`** = `2000 / t`, matching
  `plot_cross_system_frontier.py:38` (`ROWS = 2000`). The artefact's `throughput_rows_per_s` is
  *output* rows / time — for Q103 the map returns every row so the two coincide here, but use
  input rows anyway for consistency with the filter dir.
- **Use `computed_tok_s`, not raw `tok_s`** — `(prompt − prefix_cache_hits + gen) / t`. Cache-hit
  rate swings 34–38% (R=1) → 4.2% (R=32).
- **rows/s is CONFOUNDED for Sema**: x86-64-only binary, so its client is an x86 `xfer` node over
  the fabric while every other arm's client is on the GH200 over loopback
  (`rows_per_s_confounded: true` in every artefact). F1 and `computed_tok_s` are clean.

## Integrity — re-verified at import, 12/12 reps

Artefact `execution_time` == telemetry `wall_s` exactly (`*_Q103.timing.json`), one endpoint per
job on its own node, request accounting exact at R=32 (63 = `ceil(2000/32)`). Slurm exit codes
prove nothing — Sema exits 0 on error.

## Medians over 3 reps

| arm | t (s) | ±rng/2 | rows/s | computed tok/s | cache% | req | F1 | P | R |
|---|---|---|---|---|---|---|---|---|---|
| `sema_q103_verbatim_r1` | 47.593 | 0.113 | 42.0 | 14,158† | 34.0 | **1523** | 0.4115 | 0.859 | 0.270 |
| `sema_q103_verbatim_r32` | 4.067 | 0.105 | 491.8 | 24,278 | 4.2 | 63 | 0.6195 | 0.642 | 0.675 |
| `sema_q103_question_r1` | 7.259 | 0.276 | 275.5 | 21,160 | 38.3 | 2000 | 0.8652 | 0.869 | 0.865 |
| `sema_q103_question_r32` | 3.938 | 0.116 | 507.8 | 24,826 | 4.2 | 63 | 0.6320 | 0.649 | 0.677 |

## † `sema_q103_verbatim_r1` is a broken parse pipeline, not a quality or throughput datapoint

Sema never substitutes the placeholder — the model always sees a dangling `{reviewText}` — and
the verbatim prompt makes it fail outright:

- **Only 1523/2000 requests complete** (1534/1518/1523 across reps): ~24% prefilled and were
  aborted mid-generation after ~1,040 wasted tokens each (`request_params_max_tokens` averages
  16,270 — Sema sets no generation cap).
- **68% of output rows are unusable** — from `sema_q103_verbatim_r1_rep1_Q103_result.csv`: 32.1%
  carry POSITIVE/NEGATIVE, 31.8% are empty, 36.1% are garbage (`+`, `,-1`). The question form is
  96.3% usable on the same measure.

So its F1 0.4115 measures parsing, not model quality, and its 47.6 s / 14,158 computed tok/s
must not be cited as throughput. **Report both Q103 arms** — the pair is what separates "Sema
classifies badly" from "Sema cannot parse its own output".

**Open decision (judgement, not a data gap):** which arm is headline. Question form F1 0.8652 is
above *every* Skein extract arm (best `flock_op_slim_r8` = 0.836), so headlining it concedes
extract quality to Sema; verbatim is genuine prompt parity and is dominated comprehensively.

## Files

Per (arm, rep): `<arm>_rep<n>_sema.json`, `metrics_{before,after}_<arm>_rep<n>.txt`,
`<arm>_rep<n>_Q103_result.csv`, `<arm>_rep<n>_Q103.timing.json`,
`<arm>_rep<n>_vllm_sampler.csv`, `run_`/`warmup_` logs. Dropped at import: the regenerable
`.duckdb` and the vLLM server stdout logs, both still in
`analysis/results/cross_system_sema_qwen2-5-7b-instruct/<jobid>/`.

No `cross_system_summary.csv`: `summarize_cross_system.py`'s `ARM_SYSTEM` has no sema entries
(and it needs `--query 103` when it does).

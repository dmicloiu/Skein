# cross_system_sem_filter_llama_pz_greedy — provenance

Job `3087979`, Clariden, 2026-08-15 15:07–15:43 (COMPLETED, 35:56, exit 0:0).
Sibling of `cross_system_sem_filter_llama_pz_temp06` (job `3083254`); **one variable differs.**

| | `_pz_temp06` (3083254) | `_pz_greedy` (3087979) |
|---|---|---|
| sampling | inherited `generation_config.json`: `temperature 0.6, top_p 0.9` | `VLLM_EXTRA_ARGS='--override-generation-config {"temperature":0.0}'` |
| vLLM resolves to | `{'temperature': 0.6, 'top_p': 0.9}` | `{'temperature': 0.0, 'top_p': 0.9}` |

Everything else identical: palimpzest arm only, Q101, movie, scale 2000, 8 cold reps
(fresh vLLM server per rep), `in_flight=concurrency=128`, `max_model_len 16384`,
`meta-llama/Llama-3.1-8B-Instruct`, 4× GH200.

Filenames collide with `_pz_temp06` — that is why this is a separate directory.

## Contents

Everything in the job dir, flat: 8 result JSONs, 8 `run_*` / 8 `warmup_*` logs,
8 `vllm-*.log`, and 16 Prometheus scrapes (`metrics_before_*` = post-warmup/pre-timed-run,
`metrics_after_*` = post-timed-run).

Plus 7 files **not** from the job dir: palimpzest writes unparseable model output to
`~/projects/sembench/parse-answer-errors/error-<epoch>.txt`. Those are the runaway
transcripts and are the primary evidence for the mechanism, so they are carried here,
renamed by the rep whose time window contains them:

| file | rep | note |
|---|---|---|
| `parse_error_rep1_runaway_1786799680.txt` | 1 | 157385 B |
| `parse_error_rep2_runaway_1786799965.txt` | 2 | 151543 B |
| `parse_error_rep3_runaway_1786800248.txt` | 3 | 157451 B |
| `parse_error_rep4_benign_1786800435.txt`  | 4 | 1703 B — **not** a runaway, an ordinary verbose answer that failed to parse |
| `parse_error_rep4_runaway_1786800531.txt` | 4 | 156055 B |
| `parse_error_rep8_runaway_1786801378.txt` | 8 | 121119 B |
| `parse_error_XREF_3082227_rep1_runaway_1786725018.txt` | — | **from job `3082227` (temperature 0.6)**, carried here for cross-reference only; md5-identical to rep4's above |

Reps 5, 6, 7 produced no parse-error file at all.

## Gates

- 8/8 JSONs: `keys == ['Q101']`, `status == 'success'`, `extra == {}`,
  `model_name == vllm/meta-llama/Llama-3.1-8B-Instruct`. (LOTUS/PZ carry no `run_label`,
  as in the locked Qwen files.)
- 8/8 vLLM boots log `'override_generation_config': {'temperature': 0.0}` **and** the
  confirming `model.py:1509` warning — the override reached every server, not just the first.
- `finished_reason="length"` is 0 in all 16 `metrics_before_*` scrapes: the 20 warmup
  requests never run away, so every runaway is inside the timed 2000-row run.
- No errors, no `MIN_OP_RATE` trip, no FlashInfer JIT failure.

## Caveat

Per-request sampling params are not logged by vLLM V1 at INFO, so the override is confirmed
server-side, not client-side. The behavioural split (5/8 vs 0/8 runaways against an otherwise
identical job) is what establishes that it reached the requests.

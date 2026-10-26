# cross_system_multimodal_qwen2-5-omni-7b — input provenance

SemBench multimodal cross-system benchmark: Skein vs stock Flock vs LOTUS vs Palimpzest (PZ), one model for every system.
This file records the shared inputs and decisions. Each `<scenario>/INPUT_PROVENANCE.md` records that scenario's
jobs, per-query checks and caveats. These files draw no performance conclusions.

## Layout

`<scenario>/<system>/<slurm job id>/`. Long scenarios were split into 1-2 h jobs, so one system's results for a scenario
can come from several jobs. For every (scenario, system, query), the newest successful run is used. Pilot jobs and
superseded or failed jobs are not copied; each scenario file lists them.

| system dir | arm | what it is |
|---|---|---|
| `skein` | `flock_op_r1` | Flock with the Skein async operators, R=1, `in_flight=128` |
| `flock` | `flock_scalar_r1` | stock Flock scalar `llm_filter`/`llm_complete`, R=1 |
| `lotus` | `lotus` | `lotus-ai` 1.1.3, `concurrency=128` |
| `palimpzest` | `palimpzest` | PZ submodule, branch `clariden_compatibility_py311_fix`, with a local patch (below) |

Files per job (as written by `sembench/slurm/cross_system_analysis_clariden.sh`):
`config.txt`, `status.tsv`, `<arm>_rep1_<sys>.json` (SemBench metrics), `<arm>_rep1_<sys>_Q<n>.csv` (query output),
`metrics_{before,after}_<arm>_rep1.txt` (vLLM `/metrics` around the measured run), `warmup_<arm>_rep1.log`, and for the
Flock arms `telemetry_<arm>_rep1/Q<n>/` (per-query vLLM metrics, sampler, flock metrics). The vLLM server logs and
`run_*.log` are not copied; they stay in `analysis/results/cross_system_qwen2-5-omni-7b/<job>/` (gitignored).

## Shared setup

- Model `Qwen/Qwen2.5-Omni-7B`, snapshot `ae9e1690543ffd5c0221dc27f79834d0294cba00`, vLLM 0.22.0, `max_model_len=32768`,
  TP=1, one GH200, one co-located cold vLLM per job.
- N_REPS=1 (user: "first do 1 rep everywhere"). Each job first runs an untimed warm-up of the same queries at the
  warm-up scale on the same server; `metrics_before` is taken after the warm-up.
- Scales: movie 2000, animals 200, mmqa 200, ecomm 500, cars 19672. Medical is excluded.
- flock `7024338c` (multimodal commits `ca95a11d` + `7024338c`), extension built in job 3707457,
  sha256 `3cbde6db5d474721511f5a49d663d491a1bb13fd4b8923fbc40750614fa36bd4`. Same binary in every job.
- sembench: `1055119` for the first-round jobs, `bc7bb8e` afterwards (`max(16*batch, 512)` output-token bound for the
  Flock `llm_complete` setup; it only affects queries that use `llm_complete`). `config.txt` records the exact commit
  and the count of tracked local changes per job; those are the regenerated data files under `sembench/files` and the
  PZ submodule patch.
- Submission: `SCENARIO=<sc> QUERIES='<q...>' N_REPS=1 ARMS=<arm> MODEL=Qwen/Qwen2.5-Omni-7B sbatch --export=ALL
  --job-name=xsys-<sc>-<sys> --dependency=singleton slurm/cross_system_analysis_clariden.sh` (from `~/projects/sembench`).

## Decisions that affect the numbers

- **Stock Flock is reported as-is.** Its image label (`"ATTACHMENT COLUMN"` / `"ROW n :"` in `openai.cpp`) makes it accept
  far too many images (animals Q1: 122 rows vs gold 11). Replay job 3720152 showed the Skein label
  (`"Attachment <name>, row n:"`) fixes it. Evidence: appendix below.
- **LOTUS `sem_join` runs with its cascade** (`policy="approximate"`, recall/precision target 0.8, the
  `GenericLotusRunner` default). It makes fewer LLM calls than the other systems, which evaluate joins exactly.
  Exceptions are listed per scenario.
- **PZ local patch:** `"omni"` was added to `vision_patterns` in PZ `utils/model_info_helpers.py`. Without it the PZ
  optimizer loops forever on image queries. The patch is uncommitted in the submodule (needed only on this cluster).
- PZ prints "Completed" even when every LLM call fails, so every job was also checked for row counts and real vLLM
  request deltas, not only for `status.tsv`.

## Appendix: Stock Flock (scalar) vs Skein (async operators): request differences

Purpose: back up the quality numbers of the multimodal cross-system benchmark. Stock Flock is reported **as-is**
(user decision 2026-10-22). Its lower quality on image queries comes from its own prompt format, not from
Skein doing less work or answering differently by design.

Code state: flock `7024338c`, extension sha256 `3cbde6db…36bd4` (build job 3707457).
Model: `Qwen/Qwen2.5-Omni-7B` (snapshot `ae9e1690`), vLLM 0.22.0.

### 1. What is the same in both arms
- **Endpoint and message shape.** Both arms call `/v1/chat/completions` with one `user` message. Its content is
  the rendered prompt text part, followed by text label + `image_url` parts.
  - Scalar: `src/model_manager/providers/adapters/openai.cpp:73`.
  - Skein: `src/functions/operator/semantic_operator_base.cpp:268`.
- **Prompt text.** For an image-only column, `PromptManager::Render` puts no row data in the text, so the text part
  is the same in both arms. The Skein text was dumped with `FLOCK_PROMPT_DUMP` (replay job 3720152,
  `prompt_flock_op_r1.txt`) and is the full META_PROMPT. The slim prompt is skipped when media columns are present.
  *Not verified for mixed text + image columns (MMQA Q2a/Q2b).*
- **Image encoding.** Both arms base64 the raw file bytes into a `data:` URL, using the same extension → MIME rule
  (jpg/jpeg → `image/jpeg`, webp, gif, otherwise png) and the same default `detail: "low"`. They use separate helpers:
  - Scalar: `URLHandler::ResolveFileToBase64`, `openai.cpp:44-62`.
  - Skein: `ImageMimeType` + `EncodeImageFile`, `semantic_operator_base.cpp:43-79`.
- **Sampling and structured output.** Both use temperature 0, `max_tokens` 16 for filter, and a json_schema of
  `{"items": [bool]}` with `minItems = maxItems = rows`. Checked in the vLLM request logs of diag job 3714667.
- **Rows per request.** In the pilot, both arms sent R=1, the same number of requests, and nearly the same prompt tokens:

  | Run | Scalar requests | Skein requests | Scalar prompt tokens | Skein prompt tokens |
  |---|---|---|---|---|
  | movie Q1 | 2,000 | 1,999 | 975.5k | 975.0k |
  | animals Q1 | 200 | 200 | 324,970 | 324,970 |
  | MMQA Q2a+Q2b | 5,279 | 5,216 | 4.27M | 4.21M |

  The one-request gap on movie is unexplained and does not change the result.

### 2. What differs
| | Stock Flock (scalar) | Skein (async operator) |
|---|---|---|
| Image label parts | one `"ATTACHMENT COLUMN"` per column, then `"ROW <i> :"` before each image (`openai.cpp:20-42`) | `"Attachment <column name>, row <i>:"` before each image (`semantic_operator_base.cpp:98-99`) |
| Column name in label | no | yes |
| Requests in flight | 1 (one call per row/batch, synchronous) | up to 128 (`in_flight=128`) |
| Where base64 runs | scalar function | `SubmitBatch` on the sink thread (synchronous; only the HTTP request is async) |

The concurrency difference changes runtime only. The label difference changes answers (section 3).

### 3. Evidence that the label causes the quality gap

#### Pilot (step 3), measured runs, quality vs gold
| Query | Scalar | Skein |
|---|---|---|
| animals Q1, sf 200 (gold 11) | count 124 | count 10 |
| MMQA Q2a, sf 200 (gold 5) | 79 rows, P .063, R 1.0, F1 .119 | 15 rows, P .333, R 1.0, F1 .50 |
| MMQA Q2b, sf 200 (gold 5) | 79 rows, F1 .119 | 16 rows, F1 .476 |
| movie Q1 (text only) | F1 1.0 | F1 1.0 |

The gap appears only on image queries.

#### Diagnostic job 3714667 (animals Q1, per-row verdicts)
- Scalar passed 122 rows (correct 10, false 112, missed 1). Skein passed 10 (correct 9, false 1, missed 2).
- The model really answers `{"items": [true]}`; the rows are not passing because of a parse failure or a null default.
- The false matches span every species, so the bias is not tied to one class.

#### Replay job 3720152 (animals Q1)
- **Setup:** 200 images, one vLLM, same prompt text, same image bytes. Only the label parts change.
- **Gold:** 11.

| Layout | Label parts | Passed | Correct | False | Missed | Agrees with real arm |
|---|---|---|---|---|---|---|
| `scalar` | `"ATTACHMENT COLUMN"`, `"ROW 1 :"` | 123 | 9 | 114 | 2 | 164/199 |
| `scalar_named` | `"ATTACHMENT COLUMN wildlife image"`, `"ROW 1 :"` | 173 | 11 | 162 | 0 | |
| `skein` | `"Attachment wildlife image, row 1:"` | 10 | 9 | 1 | 2 | 199/199 |
| `none` | (no label) | 11 | 8 | 3 | 3 | |

- **Conclusion:** the uppercase `ATTACHMENT COLUMN` / `ROW n :` framing biases Omni toward "true". Adding the
  column name makes the bias worse, so missing information is not the cause.
- **Caveats:**
  - The `scalar` layout reproduces the real scalar arm in total (123 vs 122) but only 164/199 per row. The real
    scalar request probably differs in other bytes not reproduced here (unidentified). The bias itself is reproduced.
  - One of the 200 images has no row in the real arms' output, so per-row agreement is out of 199.
  - The replay covers animals only. For MMQA (text + image) the same label code applies, but no replay was run.

### 4. Artifacts
- Diag: `/iopsstor/scratch/cscs/dmicloiu/animals_q1_diag/out/3714667/`
  - `<arm>_where.csv`, `verdicts_flock_op_r1.jsonl`, vLLM logs.
  - Scripts: `diag.{sh,py}`, `inner.sh`.
- Replay: `/iopsstor/scratch/cscs/dmicloiu/animals_q1_diag/out/replay_3720152/`
  - `replay.csv` (layout, image, species, raw output, verdict), `replay.log`, `prompt_flock_op_r1.txt`, `vllm.log`.
  - Scripts: `replay.{sh,py}`, `replay_inner.sh`.
  - The first attempt, 3714868, failed at vLLM startup (6-minute wait too short).
- Pilot: `analysis/results/cross_system_qwen2-5-omni-7b/{3712847,3712849,3712850}/`.
- **Scratch is purged after ~30 days:** copy the diag/replay outputs somewhere permanent before relying on them in
  a paper.

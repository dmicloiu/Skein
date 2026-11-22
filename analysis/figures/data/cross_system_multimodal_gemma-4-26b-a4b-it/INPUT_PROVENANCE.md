# cross_system_multimodal_gemma-4-26b-a4b-it — input provenance

SemBench multimodal cross-system benchmark with `google/gemma-4-26B-A4B-it`: Skein vs stock Flock vs LOTUS vs
Palimpzest (PZ), one model for every system. These are the final 1-rep runs that back the comparison for the Skein
extension. This file records inputs, which job supplies each number, and caveats; it draws no conclusions.

## Layout

`<scenario>/<system>/<slurm job id>/`. Some systems' results for a scenario come from several jobs (reruns or
split query sets). For every (scenario, system, query) the newest successful run is used; the table below names the job.

| system dir | arm | what it is |
|---|---|---|
| `skein` | `flock_op_r1` | Skein async semantic operators, R=1, `in_flight=128` |
| `flock` | `flock_scalar_r1` | stock Flock scalar `llm_filter`/`llm_complete`, R=1 |
| `lotus` | `lotus` | `lotus-ai` 1.1.3 with local patches (below), `concurrency=128` |
| `palimpzest` | `palimpzest` | PZ submodule with a local patch (uncommitted, cluster-only) |
| `proxy` | `flock_emb_r1`, `flock_lotus_r1` | Skein embedding proxy (CLIP) and its LOTUS-style cascade ablation, mmqa Q2a Q2b Q7 and cars Q8 only |

Files per job (written by `sembench/slurm/cross_system_analysis_clariden.sh`): `config.txt` (exact model snapshot,
vLLM, flock commit, extension sha256, sembench commit + count of tracked local changes), `status.tsv`,
`<arm>_rep1_<sys>.json` (SemBench metrics), `<arm>_rep1_<sys>_Q<n>.csv` (query output),
`metrics_{before,after}_<arm>_rep1.txt` (vLLM `/metrics` around the measured run), `warmup_*`, and for
the Flock arms `telemetry_<arm>_rep1/`. The vLLM server logs (`vllm-*.log`, 6-20 MB each) and `run_*.log` are not copied; they stay
in `analysis/results/cross_system_gemma-4-26b-a4b-it/<job>/` (gitignored).

## Shared setup

- Model `google/gemma-4-26B-A4B-it`, snapshot `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, vLLM 0.22.0
  (conda env `sembench-tf5`, transformers 5.5.4), `max_model_len=32768`, TP=1, one GH200, one cold co-located vLLM
  per job.
- N_REPS=1. Each job first runs an untimed warm-up of the same
  queries at a small scale on the same server; `metrics_before` is taken after it.
- Scales: animals 200, movie 2000, mmqa 200, ecomm 500, cars 19672. Medical excluded.
- Flock extension: flock `aed19520`, sha256 `21e73d423143ea74c58944b8e071c637919afe55b5e65c04c717d0034a990353`
  (build 3958643) in every job except the two Skein LIMIT reruns 3986000 / 3986001: same code plus the LIMIT early
  stop (later committed as `5fab00e5`), sha256 `acc3f2c7d6256c9fef93c669291ca1ac8a39c5fef4a40c7e8faef96c9fff07e3`
  (build 3983680).
- sembench: `6ccad19` / `88f5adf` (main checkout, + local data changes counted in `config.txt`), proxy jobs from the
  `proxy_v2` worktree `e177814`. All of it is now in sembench `f7be355` (pushed).

## Decisions and caveats that affect the numbers

- **Skein uses the LIMIT early-stop runs** (3986000 movie, all queries; 3986001 cars Q3 Q8). Without it, movie Q5/Q6
  (LIMIT 10 self-joins) took 389/384 s with ~65000 requests (3966075); with it ~1 s. Other queries reproduce 3966075 /
  3966084 (movie Q6 F1 .89 -> .95, cars Q8 F1 .41 both).
- **Stock Flock is reported as-is** (its image label biases image filters toward true; see
  `../cross_system_multimodal_qwen2-5-omni-7b/INPUT_PROVENANCE.md`). ecomm Q12 fails deterministically (model JSON with a missing closing quote,
  `[ModelProvider] Output extraction error`) in 3966082 and in the Q12-only rerun 3970887.
- **PZ mmqa Q4 (\*)**: the query succeeded but the SemBench evaluator crashed on a NaN genre; fixed in sembench
  `88f5adf` and rescored offline: F1 .59 (P .59, R .59). The JSON in 3966080 still holds the evaluator error.
- **LOTUS** runs `sem_join` with its default cascade (`policy="approximate"`, target 0.8), so it makes fewer LLM calls
  than the exact joins of the other systems. Patches to the installed lotus-ai 1.1.3 (committed in sembench
  `patches/lotus-ai-1.1.3/`), enabled with `LOTUS_FIX_IMAGE_EMBED=1 LOTUS_FIX_JOIN_IDS=1 LOTUS_FIX_RGBA=1`:
  image embeddings of the file instead of its path, correct ids on duplicate join keys, alpha kept for transparent
  logos, and retry of failed litellm calls (on by default; 2 of 19657 cars calls hit a client-side
  `Bad file descriptor` and recovered). LOTUS accepts wrong pairs under its prompt (mmqa Q7 R 1.0 / F1 .20,
  ecomm Q8 R 1.0 / F1 .12, ecomm Q11 1296 rows vs 27 for the others); verdicts are explicit `Answer: True`, not a
  parser issue.
- **LOTUS jobs per scenario:** animals 3983681, movie 3983682; cars 3970893 for Q3 Q8 Q10 and 3983683 for Q1 Q4; ecomm split by duration:
  3970895 (Q1-6 9 12), 3970897 (Q7 13), 3970899 (Q10), 3970902 (Q11), 3970904 (Q8).
- **Proxy arms** (3966087 mmqa, `PROXY_TEMPLATE="the logo of {}"`; 3966088 cars Q8, `"a photo of a damaged car"`):
  mmqa emb Q2a/Q2b F1 .83 / Q7 1.00 in 25 s, LOTUS-style .42 / .42 / .43; cars Q8 emb F1 .42 in 155 s,
  LOTUS-style .14.
- PZ prints "Completed" even when LLM calls fail, so every job was also checked for row counts and vLLM request
  deltas, not only `status.tsv`. Every query in the table has status `success` unless marked `failed`.
- The pre-LIMIT Skein runs 3966075 / 3966084 are included for reference (3966084 still supplies cars Q1 Q4 Q10).

## Per-query results

Quality metric (F1, accuracy, or relative error RE: lower is better), measured runtime from the metrics JSON, and the
job that supplies it. Generated from the copied JSONs (newest successful run per query).

### animals

| query | skein | flock | lotus | palimpzest |
|---|---|---|---|---|
| Q1 | RE 0.27, 11 s (3966072) | RE 0.27, 42 s (3966073) | RE 0.27, 60 s (3983681) | RE 0.27, 10 s (3966074) |
| Q3 | F1 1.00, 8 s (3966072) | F1 1.00, 29 s (3966073) | F1 1.00, 48 s (3983681) | F1 0.00, 3 s (3966074) |
| Q7 | F1 0.89, 16 s (3966072) | F1 0.89, 56 s (3966073) | F1 1.00, 105 s (3983681) | F1 0.89, 15 s (3966074) |
| Q10 | F1 1.00, 9 s (3966072) | F1 1.00, 28 s (3966073) | F1 1.00, 49 s (3983681) | F1 0.00, 5 s (3966074) |

### movie

| query | skein | flock | lotus | palimpzest |
|---|---|---|---|---|
| Q1 | F1 1.00, 1 s (3986000) | F1 1.00, 126 s (3966076) | F1 1.00, 7 s (3983682) | F1 1.00, 1 s (3966077) |
| Q2 | F1 1.00, 1 s (3986000) | F1 1.00, 7 s (3966076) | F1 0.60, 0 s (3983682) | F1 1.00, 28 s (3966077) |
| Q3 | RE 0.43, 1 s (3986000) | RE 0.43, 7 s (3966076) | RE 0.14, 0 s (3983682) | RE 0.50, 4 s (3966077) |
| Q4 | RE 0.43, 1 s (3986000) | RE 0.43, 8 s (3966076) | RE 0.14, 0 s (3983682) | RE 0.50, 4 s (3966077) |
| Q5 | F1 0.80, 1 s (3986000) | F1 0.90, 131 s (3966076) | F1 0.33, 760 s (3983682) | F1 0.11, 8 s (3966077) |
| Q6 | F1 0.95, 1 s (3986000) | F1 0.95, 122 s (3966076) | F1 0.46, 182 s (3983682) | F1 0.95, 6 s (3966077) |
| Q7 | F1 0.69, 385 s (3986000) | F1 0.69, 3922 s (3966076) | F1 0.33, 195 s (3983682) | F1 0.77, 2703 s (3966077) |
| Q8 | RE 0.24, 1 s (3986000) | RE 0.24, 8 s (3966076) | RE 0.28, 1 s (3983682) | RE 0.20, 5 s (3966077) |

### mmqa

| query | skein | flock | lotus | palimpzest |
|---|---|---|---|---|
| Q1 | F1 1.00, 2 s (3966078) | F1 1.00, 15 s (3970885) | F1 1.00, 2 s (3970892) | F1 1.00, 1 s (3966080) |
| Q2a | F1 0.67, 104 s (3966078) | F1 0.48, 327 s (3970885) | F1 0.62, 88 s (3970892) | F1 0.91, 154 s (3966080) |
| Q2b | F1 0.67, 105 s (3966078) | F1 0.48, 316 s (3970885) | F1 0.62, 87 s (3970892) | F1 0.77, 141 s (3966080) |
| Q3a | F1 0.81, 2 s (3966078) | F1 0.81, 13 s (3970885) | F1 0.83, 1 s (3970892) | F1 0.81, 5 s (3966080) |
| Q3f | F1 0.75, 2 s (3966078) | F1 0.75, 13 s (3970885) | F1 0.86, 1 s (3970892) | F1 0.86, 5 s (3966080) |
| Q4 | F1 0.56, 1 s (3966078) | F1 0.55, 2 s (3970885) | F1 0.56, 1 s (3970892) | F1 0.59*, 2 s (3966080) |
| Q5 | F1 1.00, 0 s (3966078) | F1 1.00, 0 s (3970885) | F1 1.00, 0 s (3970892) | F1 1.00, 1 s (3966080) |
| Q6a | F1 1.00, 2 s (3966078) | F1 1.00, 13 s (3970885) | F1 1.00, 1 s (3970892) | F1 1.00, 3 s (3966080) |
| Q6b | F1 1.00, 2 s (3966078) | F1 1.00, 12 s (3970885) | F1 1.00, 1 s (3970892) | F1 1.00, 3 s (3966080) |
| Q6c | F1 1.00, 2 s (3966078) | F1 1.00, 13 s (3970885) | F1 0.91, 1 s (3970892) | F1 1.00, 4 s (3966080) |
| Q7 | F1 0.42, 1684 s (3966078) | F1 0.32, 5004 s (3970885) | F1 0.20, 3175 s (3970892) | F1 0.43, 2382 s (3966080) |

### ecomm

| query | skein | flock | lotus | palimpzest |
|---|---|---|---|---|
| Q1 | F1 1.00, 6 s (3966081) | F1 1.00, 35 s (3966082) | F1 1.00, 4 s (3970895) | F1 1.00, 13 s (3966083) |
| Q2 | F1 0.80, 32 s (3966081) | F1 0.77, 113 s (3966082) | F1 0.53, 140 s (3970895) | F1 0.73, 28 s (3966083) |
| Q3 | acc 0.93, 6 s (3966081) | acc 0.93, 39 s (3966082) | acc 0.87, 9 s (3970895) | acc 0.87, 14 s (3966083) |
| Q4 | acc 0.71, 14 s (3966081) | acc 0.68, 40 s (3966082) | acc 0.54, 79 s (3970895) | acc 0.78, 18 s (3966083) |
| Q5 | acc 0.98, 3 s (3966081) | acc 0.98, 17 s (3966082) | acc 0.98, 3 s (3970895) | acc 0.98, 10 s (3966083) |
| Q6 | acc 0.89, 11 s (3966081) | acc 0.92, 33 s (3966082) | acc 0.90, 77 s (3970895) | acc 0.89, 726 s (3966083) |
| Q7 | F1 0.88, 93 s (3966081) | F1 0.89, 547 s (3966082) | F1 0.86, 153 s (3970897) | F1 0.91, 284 s (3966083) |
| Q8 | F1 0.29, 23 s (3966081) | F1 0.33, 77 s (3966082) | F1 0.12, 141 s (3970904) | F1 0.40, 39 s (3966083) |
| Q9 | F1 0.75, 16 s (3966081) | F1 0.83, 75 s (3966082) | F1 0.86, 336 s (3970895) | F1 0.85, 29 s (3966083) |
| Q10 | F1 0.35, 40 s (3966081) | F1 0.00, 96 s (3966082) | F1 0.00, 117 s (3970899) | F1 0.00, 239 s (3966083) |
| Q11 | F1 0.80, 116 s (3966081) | F1 0.80, 342 s (3966082) | F1 0.07, 337 s (3970902) | F1 0.80, 100 s (3966083) |
| Q12 | F1 0.00, 10 s (3966081) | failed (3966082) | F1 0.96, 26 s (3970895) | F1 0.00, 20 s (3966083) |
| Q13 | F1 0.70, 27 s (3966081) | F1 0.70, 82 s (3966082) | F1 0.78, 145 s (3970897) | F1 0.73, 55 s (3966083) |

### cars

| query | skein | flock | lotus | palimpzest |
|---|---|---|---|---|
| Q1 | F1 0.80, 223 s (3966084) | F1 0.79, 1008 s (3966085) | F1 0.88, 126 s (3983683) | F1 0.90, 400 s (3966086) |
| Q3 | F1 1.00, 240 s (3986001) | F1 1.00, 374 s (3966085) | F1 0.80, 286 s (3970893) | F1 0.80, 3 s (3966086) |
| Q4 | RE 0.01, 228 s (3966084) | RE 0.01, 1003 s (3966085) | RE 0.01, 128 s (3983683) | RE 0.01, 435 s (3966086) |
| Q8 | F1 0.41, 148 s (3986001) | F1 0.48, 607 s (3966085) | F1 0.48, 854 s (3970893) | F1 0.30, 269 s (3966086) |
| Q10 | F1 0.54, 250 s (3966084) | F1 0.54, 1111 s (3966085) | F1 0.52, 362 s (3970893) | F1 0.50, 1330 s (3966086) |

\* rescored offline, see caveats.

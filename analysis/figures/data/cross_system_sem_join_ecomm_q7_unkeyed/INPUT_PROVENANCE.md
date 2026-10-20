# cross_system_sem_join_ecomm_q7_unkeyed — input provenance

SemBench e-commerce Q7: an unkeyed semantic self-join over products (directed pairs, self-pairs included,
`price <= 500`), evaluated with Skein (`flock_op_r1`) and stock Flock (`flock_scalar_r1`).
Model: `Qwen/Qwen2.5-7B-Instruct`, TP=1, temperature 0, one GH200, one co-located cold vLLM per (arm, rep).
This file records inputs and checks only. It draws no performance conclusions.

| arm | job | R | in-flight / threads | plan (EXPLAIN, check job 3700291) |
|---|---|---|---|---|
| `flock_op_r1` (Skein) | 3706420 | 1 | `in_flight_cap=128` | `CROSS_PRODUCT` -> `EXTENSION` operator |
| `flock_scalar_r1` | 3706420 | 1 | script default `DUCKDB_THREADS=16` | `BLOCKWISE_NL_JOIN` INNER with `llm_filter` as join condition |

There are 3 reps per arm. Both arms ran in one job on the same node, in the order op rep1-3, then scalar rep1-3.
Each rep gets a cold vLLM and a wiped flock catalog. Before `metrics_before`, each rep runs an untimed Q7 warm-up
at `sf_50` on the same server.

Command (from `~/projects/sembench`):
`SCENARIO=ecomm QUERY=7 WARM_QUERY=7 WARM_SCALE=50 SCALE_FACTOR=300 N_REPS=3 ARMS='flock_op_r1 flock_scalar_r1'
SCALAR_BATCH=1 IN_FLIGHT=128 sbatch --export=ALL slurm/cross_system_analysis_clariden.sh`.
The script's default time limit (3 h) was lowered to `00:45:00` via `scontrol` at the user's request.
Job 3706420: COMPLETED 0:0, 2026-10-20 11:36:34–11:58:19 (21:45 elapsed). Log:
`sembench/logs/slurm-cross-system-analysis-3706420.out`.

Revisions:
- sembench: HEAD `c2041cd` (the Q108 LOTUS/PZ commit rebased onto origin/main; not pushed). It sits on `7557ba5`
  "Integrate flock for ECOMM scenario", which last touched `files/ecomm/queries/dialects/flockmtl/q7.sql` and
  `slurm/cross_system_analysis_clariden.sh`.
- flock: `2f0aa340` (contains the unkeyed rewrite `9e0c09e8`). The loadable was built in job 3646735 on 2026-10-12.
  No `src` changes since, so it was not rebuilt.

## Q7 — what is being measured

`sembench/files/ecomm/queries/dialects/flockmtl/q7.sql`. A CTE projects `productDisplayName || ' - ' ||
productDescriptors.description.value` for products with `price <= 500`. Then `p1 JOIN p2 ON llm_filter(...)`
runs with SemBench's prompt ("Do both product descriptions describe products of the same category from the same
brand ...") and the `first`/`second product description` context columns. The query, prompt, model and candidate
set are unchanged from SemBench.

Data: SemBench `generate_data.py` (unchanged, seed 12345600). It was run in job 3694112 and produced
`files/ecomm/data/sf_{50,300}/styles_details.parquet`. Counts were verified in job 3700291:
- `sf_300`: 300 rows; **59** products with `price <= 500` (59 distinct ids); **3,481** directed pairs; **121** gold pairs.
- `sf_50` (warm-up): 50 rows; 23 gold pairs.

Gold comes from SemBench's ecomm scenario handler (`src/scenario/ecomm/evaluation/evaluate.py`, `get_ground_truth`).

## Checks (all 6 measured runs)

- All 6 warm-ups succeeded (`warmup_*.log`: `Q7: ✅`, 25 rows each). The warm-ups run with `|| true`, so this was checked by hand.
- All 6 measured queries report `status: success`. Row counts: op 183 / 183 / 184, scalar 186 / 186 / 186.
  Every `_Q7.csv` has exactly `row_count` rows: no duplicates, all of the form `<id>-<id>`, every id in the 59-product set.
- vLLM deltas (`metrics_{before,after}`): every run had exactly **3,481** `stop` requests (0 abort / error / length /
  repetition) and 3,064,106 prompt tokens. Generated tokens: op 31,248, scalar 31,250. Each `metrics_before` counts
  121 requests, the `sf_50` warm-up.
- Operator engagement: in `flock_metrics.json`, the three `flock_op_r1` reps report `thread_count: 0` (zero scalar
  `llm_filter` calls). The three `flock_scalar_r1` reps report `thread_count: 1` with 3,481 `llm_filter` api_calls.

## Retries, failures and deviations (whole experiment)

- Data download: 3692309 failed (container gdown too old for `--fuzzy`; Drive confirm page). 3692370 failed (Drive
  "Quota exceeded" 2 KB page, caught by the size check, deleted). 3694075 was cancelled at 0 s: `curl -C -` on a
  complete archive could have appended Drive's error page, so the script was fixed never to download when the archive
  exists. The user then copied `ecomm.tar.gz` (24,586,145,746 bytes) to scratch by hand, and data prep job 3694112 succeeded.
  `sembench/files/ecomm/source_data` is an untracked symlink to `/capstor/scratch/cscs/dmicloiu/sembench_ecomm`
  (scratch, purged after ~30 days).
- Plan/count check: 3697862 failed (`ModuleNotFoundError: tomli`). The fix was `pip install tomli` (2.5.0) in the
  sembench conda env. The rerun, 3700291, succeeded and all counts and plans matched.
- Smoke: job 3700507 (same command, `N_REPS=1`, limit lowered to `00:20:00`) completed 0:0. Both arms passed the
  same checks as above (186 rows each, 3,481 requests each).
- Harness telemetry stale tails. The smoke run used the same `rep1` tags, so overwriting files on NFS left stale tails
  in 4 files under sembench `files/ecomm/metrics/flockmtl/<tag>/Q7/`. The copies here were truncated to their last
  valid content. The sembench originals are untouched.
  - `telemetry_flock_op_r1_rep1/vllm_metrics_before.txt`: 5 bytes removed (`e+09\n`).
  - `telemetry_flock_scalar_r1_rep1/vllm_metrics_before.txt`: 2 bytes removed (`9\n`).
  - `telemetry_flock_scalar_r1_rep1/vllm_metrics_after.txt`: 17 bytes removed (`357488857079e+09\n`).
  - `telemetry_flock_scalar_r1_rep1/flock_metrics.json`: 1 byte removed (a trailing `}`; the file was invalid JSON).
  The script's own `metrics_{before,after}_*.txt` sit in a fresh per-job dir and were not affected.

## Contents

- `<arm>_rep<r>_flockmtl.json` (harness metrics + P/R/F1) and `<arm>_rep<r>_flockmtl_Q7.csv` (predicted pairs).
- `metrics_{before,after}_<arm>_rep<r>.txt`: vLLM `/metrics`, taken by the Slurm script around the measured query.
- `telemetry_<arm>_rep<r>/`: `flock_metrics.json` (per-thread Flock debug metrics), `vllm_sampler.csv`, and the
  harness's own `vllm_metrics_{before,after}.txt`. The Slurm script does not copy this directory itself. It was copied from
  sembench `files/ecomm/metrics/flockmtl/<arm>_rep<r>/Q7/`, which the next run with the same tag overwrites.
- `warmup_<arm>_rep<r>.log`.

## Not included here

- `vllm-*.log` and `run_*.log` (6 each) stay uncommitted on Clariden, at
  `/users/dmicloiu/projects/flock/analysis/results/cross_system_qwen2-5-7b-instruct/3706420/`
  (repo policy, `[REPOSITORY-CLEAN-UP]`, 1d421c50). The same applies to the smoke run, under `.../3700507/`.

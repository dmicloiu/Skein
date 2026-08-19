# cross_system_sem_extract_llama — input provenance

Cross-system sem_extract comparison (Q103, 2000 rows, **Llama-3.1-8B-Instruct**, TP=1,
GH200): FlockMTL operator arms at several rows-per-request `R`, the FlockMTL scalar, LOTUS
and Palimpzest. This is the **Llama generalisation replicate** of
`analysis/figures/data/cross_system_sem_extract/` (Qwen2.5-7B-Instruct); the two directories
are the same 11-arm sweep on two model families.

## READ FIRST — this directory is mixed-era; know which files are which

The result JSONs and metrics were **re-measured on 2026-08-18** under a changed sem_extract
response schema. The bounded-era vLLM logs were deliberately **kept**: `analysis/results/`
is gitignored, so they are the only committed copy of that era's server-side evidence.

| files | era | note |
|---|---|---|
| `*_rep*_*.json`, `metrics_*.txt`, `run_*.log`, `warmup_*.log` | **unbounded** (job 3116950) | current; fully replaced, 1:1 by filename |
| `cross_system_summary.csv` | **unbounded** | generated from the above; first ever built for this directory |
| `vllm-1787*.log` (33) | **unbounded** | from 3116950 |
| `vllm-1786*.log` (33) | **bounded** | retained from baseline job 3083182 |

No plot script or summarizer reads the vLLM logs, so the era mix cannot contaminate an
output. The bounded-era result JSONs remain recoverable in git history at **`2565fe75`**.

The two eras differ in `PhysicalSemExtract::BuildResponseFormat`: the bounded era put a
`maxLength = 4 * (max_output_tokens / batch_rows)` on the per-row string element; the
unbounded era removes it, leaving the request-level `max_tokens` as the only ceiling.
`minItems`/`maxItems` still pin the answer count in both. Full mechanism writeup:
`analysis/figures/data/sem_extract_ab_json/INPUT_PROVENANCE.md`.

**Baseline for every bounded-era number here is job `3083182`, not `3089338`** — that is a
different sweep. Only `3083182`'s vLLM logs (`vllm-17867415xx`+) match this directory.

## Generation is provably unchanged

`prompt_tok` is **bit-identical between eras in all 33 arm-reps**. `gen_tok` is bit-identical
in **26 of the 27 flock cells**; the single exception is `flock_op_r32` rep1 at
**8435 -> 8437 tokens (2 of 8435, +0.024%)**. At the summary level `total_tok` is
bit-identical in **all 9 flock arms**. Same tokens in, same tokens out, less wall time.

The two Python controls drift slightly (`lotus` +30 of 447,615; `palimpzest` +221 of
1,322,436) — they sample rather than run greedy, and they do not touch this code path.

## Effect, bounded -> unbounded

Sorted by `tok_per_row` (= prefill-dominance), which is what actually orders the arms:

| arm | R | tok/row | old rows_s | new rows_s | Δ | ms/row saved |
|---|---|---|---|---|---|---|
| `palimpzest` | – | 661.2 | 205.5 | 213.4 | **+3.8%** | 0.18 |
| `flock_op_r1` | 1 | 536.0 | 111.4 | 130.3 | +17.0% | 1.30 |
| `lotus` | – | 223.8 | 126.1 | 146.9 | **+16.5%** | 1.12 |
| `flock_op_slim_r1` | 1 | 176.0 | 206.7 | 318.2 | +53.9% | 1.70 |
| `flock_op_slim_r2` | 2 | 119.0 | 259.2 | 420.1 | +62.1% | 1.48 |
| `flock_op_slim_r4` | 4 | 83.5 | 283.0 | 515.1 | +82.0% | 1.59 |
| `flock_op_slim_r8` | 8 | 65.7 | 289.7 | 555.6 | +91.8% | 1.65 |
| `flock_op_slim_r16` | 16 | 57.3 | 289.8 | 557.3 | **+92.3%** | 1.66 |
| `flock_op_r32` | 32 | 54.7 | 293.8 | 498.8 | +69.8% | 1.40 |
| `flock_scalar` | 32 | 54.7 | 40.4 | 40.4 | **+0.0%** | 0.00 |
| `flock_op_slim_r32` | 32 | 53.1 | 286.3 | 478.7 | +67.2% | 1.40 |

**The gain is not a function of R**, reproducing Qwen: the two R=1 arms differ by 3x
(+17.0% vs +53.9%), and the slim family is an inverted U peaking at R=16, not monotonic.
What is near-constant is the **absolute** saving: **1.30–1.70 ms/row on every operator arm**,
against 0.00 ms/row on `flock_scalar`. R enters only as a proxy, because raising R amortises
the prompt and drives `tok_per_row` down.

### The between-node confound, and why it does not reach the flock arms

The three jobs ran on three different nodes (3083182 nid007117, 3115747 nid006687,
3116950 nid006240), so every cross-era comparison carries a between-node component. On
Llama that component is visible: `lotus` gained **+16.5%** and it cannot possibly be the
schema change — LOTUS never enters flock code. `palimpzest` gained +3.8%.

**`flock_scalar` bounds the effect on the flock path at zero**: it is 40.4 -> 40.4 rows/s
(bit-identical to 0.1 rows/s) across nid007117 -> nid006240, and its `f1_score` is
bit-identical across all 3 reps in both eras. So the two nodes are equivalent *for this
client*, and the operator arms' 1.30–1.70 ms/row is a schema effect rather than a node
effect. The `lotus`/`palimpzest` movement is a CPU-bound-Python-client sensitivity — the
same class of thing as discarded job `3115747`, milder and in the opposite direction.

Consequence: **use the ratios below, not the absolute `rows_s`, for any cross-era or
cross-family claim.** Both arms of each ratio ran in the same job on the same node, so the
ratio is confound-free by construction. Never compare Qwen `rows_s` to Llama `rows_s`.

## The operator/scalar ratio roughly doubles — third independent surface

| surface | old | new |
|---|---|---|
| Qwen thread sweep t16 (`sem_extract_ab_json`) | 7.12x | 12.98x |
| Qwen cross-system `flock_op_r32` / `flock_scalar` | 6.64x | 12.51x |
| **Llama cross-system `flock_op_r32` / `flock_scalar`** (498.8 / 40.4) | **7.27x** | **12.35x** |

Three surfaces, two model families — **12.98 / 12.51 / 12.35, agreeing to ~5%**. The Llama
best arm (`flock_op_slim_r16` 557.3 / `flock_scalar` 40.4) is **13.79x**, against 7.17x
bounded.

## Quality gate — use the spread test, not bit-identity

Every arm passes: each arm's old->new shift in mean `f1_score` is within that arm's
rep-to-rep spread.

| arm | old spread | new spread | \|Δ mean\| |
|---|---|---|---|
| `flock_op_r1` | 0.0022 | 0.0025 | 0.0004 |
| `flock_op_r32` | 0.0037 | 0.0000 | 0.0012 |
| `flock_op_slim_r1` | 0.0018 | 0.0009 | 0.0002 |
| `flock_op_slim_r2` | 0.0009 | 0.0013 | 0.0003 |
| `flock_op_slim_r4` | 0.0009 | 0.0000 | 0.0001 |
| `flock_op_slim_r8` | 0.0000 | 0.0004 | **0.0007** |
| `flock_op_slim_r16` | 0.0000 | 0.0005 | 0.0002 |
| `flock_op_slim_r32` | 0.0004 | 0.0000 | 0.0001 |
| `flock_scalar` | 0.0000 | 0.0000 | 0.0000 |
| `lotus` | 0.0013 | 0.0013 | 0.0003 |
| `palimpzest` | 0.0004 | 0.0009 | 0.0006 |

`flock_op_slim_r8` is the only arm whose shift (0.0007) exceeds the spread measured in
*these two* jobs (0.0004), and it is an artefact of the baseline happening to be
bit-identical across its 3 reps. A third independent measurement of the same arm —
discarded job `3115747`, whose F1 is unaffected by its client-side slowness — spans
**0.853984 / 0.854883 / 0.855333, a spread of 0.00135** that contains both era means
(0.854883 and 0.854434). The arm's real rep spread is ~0.0013, so 0.0007 is well inside it.
**Do not tighten this gate on the basis of arms that happen to come back bit-identical.**

## No absolute unbounded-era F1 exists

The `f1_score` in the result JSONs (and therefore in the figures) is the harness's
**buggy macro-average**, which averages over off-vocab classes. `3116950` was run
deliberately **without** `VERDICT_DUMP` (the dump writes during the run and would make the
timings non-citable), so no `verdicts_*.jsonl` exist here and `analysis/rescore_extract.py`
cannot be run against this directory in either era.

The quality gate above does not depend on that: it applies the same scorer to both eras,
which is valid for detecting a *shift* even though the scorer is biased.
**Do not publish an absolute F1 for the unbounded era from this directory.**

## Job 3116950

nid006240, COMPLETED, exit `0:0`, 53:24 of a 2:00:00 request, 2026-08-18. 11 arms x 3 reps,
198 files, exact 1:1 filename match with the bounded era on all 165 non-log files.
**All 33 reps succeeded — every arm is n=3**, including `lotus`, which is n=2 on Qwen and
was n=1 in the discarded Llama job. Submitted with `--exclude=nid006687`.

```
cd $HOME/projects/sembench && MODEL=meta-llama/Llama-3.1-8B-Instruct QUERY=103 \
  SCALE_FACTOR=2000 N_REPS=3 \
  ARMS="flock_op_r32 flock_op_r1 flock_op_slim_r1 flock_op_slim_r2 flock_op_slim_r4 \
flock_op_slim_r8 flock_op_slim_r16 flock_op_slim_r32 flock_scalar lotus palimpzest" \
  sbatch --time=02:00:00 --exclude=nid006687 slurm/cross_system_analysis_clariden.sh
```

### Superseded job `3115747` — do not import, do not cite its peer numbers

`3115747` (nid006687) measured the same sweep on 2026-08-18 and its **flock half was clean**
(27/27 token-identical, all F1 gates passed), but both Python peer controls suffered a
**client-side concurrency collapse**: `palimpzest` 205.5 -> 42.2 rows/s (0.20x) and `lotus`
126.1 -> 59.7 (0.47x, n=1, reps 1 and 2 died). Diagnosed as node-level degradation on
nid006687 affecting only CPU-bound Python clients — tokens bit-identical, F1 unchanged,
server-side `e2e_request_latency` *fell*, `queue_time` ~0, and achieved concurrency
collapsed from `Running` ~59 to peak 11 against `--concurrent-llm-worker 128` in both eras.

`3116950` confirms the diagnosis on all counts:

| | bounded 3083182 | degraded 3115747 | **3116950** |
|---|---|---|---|
| `palimpzest` rows_s | 205.5 | 42.2 | **213.4** |
| `lotus` rows_s (n) | 126.1 (3) | 59.7 (1) | **146.9 (3)** |
| peer peak `Running` | 55–93 | 9–30 | **51–63** |
| `flock_scalar` rows_s | 40.4 | 38.1 | **40.4** |

Peer throughput and achieved concurrency both returned to baseline, and `flock_scalar`'s
provisional −5.6% — the mild version of the same effect, it being 1-thread and
latency-bound — went away entirely. The flock operator numbers also came back **higher**
than `3115747`'s provisional table (e.g. `slim_r16` 530.7 -> 557.3), so that job was mildly
degrading the flock arms too and its numbers were conservative rather than inflated.

When reading peak `Running` from a peer's vLLM log, note the arm only runs ~9 s while vLLM
emits a state line about every 10 s, so a healthy rep can show 2–3 samples and occasionally
miss the burst entirely (`palimpzest` rep3 here shows a single active sample at 2 reqs
despite a healthy 205.3 rows/s). The `3115747` comparison is safe because its collapse made
the arms ~5x longer and therefore gave *more* sampling opportunity, all of it capped at ≤11.

## Regenerating the summary

```
/usr/bin/python3.11 analysis/summarize_cross_system.py \
  --results-dir analysis/figures/data/cross_system_sem_extract_llama --rows 2000 --query 103
```

`--query 103` is required — the script defaults to **101** (that is sem_filter). Extract is
**103**. Use `/usr/bin/python3.11`; the login node's default `python3` is 3.6 and cannot
parse the script. `--out-dir` defaults to `--results-dir`.

# cross_system_sem_extract — input provenance

Cross-system sem_extract comparison (Q103, 2000 rows, Qwen2.5-7B-Instruct, TP=1, GH200):
FlockMTL operator arms at several rows-per-request `R`, the FlockMTL scalar, LOTUS and
Palimpzest. Consumed by `analysis/plot_cross_system_frontier.py` and
`analysis/plot_r_sweep.py`, which read the per-rep `*_rep*_*.json` files **directly**
(both `rows_s` and `f1_score`) — not `cross_system_summary.csv`.

## READ FIRST — this directory is mixed-era; know which files are which

The result JSONs and metrics were **re-measured on 2026-08-18** under a changed sem_extract
response schema. Some older artefacts were deliberately **kept** because they cannot be
regenerated. Check this table before using any file here.

| files | era | note |
|---|---|---|
| `*_rep*_*.json`, `metrics_*.txt` | **unbounded** (job 3103523) | current; fully replaced, 1:1 by filename |
| `cross_system_summary.csv` | **unbounded** | regenerated from the above |
| `vllm-1787*.log` | **unbounded** | 33 logs from 3103523 |
| `vllm-1784*.log`, `vllm-1786*.log` | **bounded** | 84 logs, retained from prior jobs |
| `extract_rescored.csv` | **bounded** | **cannot be regenerated** — see below |
| `verdicts_*_diag.jsonl` (8) | **bounded** | per-row dumps the rescore was built from |

The two eras differ in `PhysicalSemExtract::BuildResponseFormat`: the bounded era put a
`maxLength = 4 * (max_output_tokens / batch_rows)` on the per-row string element; the
unbounded era removes it, leaving the request-level `max_tokens` as the only ceiling.
`minItems`/`maxItems` still pin the answer count in both. The bounded-era result JSONs
remain recoverable in git history at **`2565fe75`**. Full mechanism writeup:
`analysis/figures/data/sem_extract_ab_json/INPUT_PROVENANCE.md`.

## Generation is provably unchanged

`prompt_tok` **and** `gen_tok` are **bit-identical** between the two eras in **all 11 arms**,
flock and non-flock alike. Same tokens in, same tokens out, less wall time.

## Effect, bounded -> unbounded

Sorted by `tok_per_row` (= prefill-dominance), which is what actually orders the arms:

| arm | R | tok/row | old rows_s | new rows_s | Δ | ms/row saved |
|---|---|---|---|---|---|---|
| `palimpzest` | – | 758.9 | 83.9 | 82.9 | **−1.2%** | −0.14 |
| `flock_op_r1` | 1 | 530.8 | 115.4 | 139.1 | +20.5% | 1.48 |
| `lotus` | – | 178.9 | 247.0 | 259.8 | **+5.2%** | 0.20 |
| `flock_op_slim_r1` | 1 | 170.8 | 211.9 | 354.6 | +67.3% | 1.90 |
| `flock_op_slim_r2` | 2 | 116.7 | 261.1 | 465.4 | +78.2% | 1.68 |
| `flock_op_slim_r4` | 4 | 82.7 | 282.8 | 547.7 | +93.7% | 1.71 |
| `flock_op_slim_r8` | 8 | 65.7 | 294.4 | 591.9 | **+101.1%** | 1.71 |
| `flock_op_slim_r16` | 16 | 58.3 | 295.3 | 581.7 | +97.0% | 1.67 |
| `flock_scalar` | 32 | 55.3 | 42.9 | 43.0 | **+0.2%** | 0.05 |
| `flock_op_r32` | 32 | 55.3 | 285.0 | 537.9 | +88.7% | 1.65 |
| `flock_op_slim_r32` | 32 | 54.6 | 287.4 | 518.7 | +80.5% | 1.55 |

**The gain is not a function of R.** The two R=1 arms differ by 3x (+20.5% vs +67.3%), and
the slim family is an inverted U peaking at R=8, not monotonic. What is near-constant is the
**absolute** saving: **1.48–1.90 ms/row on every operator arm**, versus ≤0.20 ms/row on all
three controls. The relative gain is just that constant divided by the arm's baseline
per-row cost, so prefill-dominated arms gain least in percentage terms. R enters only as a
proxy, because raising R amortises the prompt and drives `tok_per_row` down.

`palimpzest` is the strongest control: it runs at 128-way concurrency, never touches this
code, and moved −1.2%. `flock_scalar` is flat but runs at concurrency ~1, so it is a weaker
control for a high-concurrency effect than it looks. `lotus` is +5.2% on **n=2** (below).

The operator/scalar ratio roughly doubles: `flock_op_r32` / `flock_scalar` goes
**6.64x -> 12.51x**, corroborating the `sem_extract_ab_json` thread sweep's
**7.12x -> 12.98x** on an independent surface (agreement to ~4%).

## Quality gate — use the spread test, not bit-identity

Every arm passes: each arm's old->new shift in mean `f1_score` is within that arm's own
rep-to-rep spread. A fixed ±0.003 gate would **fail on pre-existing noise** — baseline
`flock_op_slim_r1` was already non-identical across its own reps (spread 0.0030), and
untouched `palimpzest` drifts 0.0089, `lotus` 0.0077.

| arm | old spread | new spread | \|Δ mean\| |
|---|---|---|---|
| `flock_op_r1` | 0.0000 | 0.0004 | 0.0001 |
| `flock_op_r32` | 0.0000 | 0.0000 | 0.0000 |
| `flock_op_slim_r1` | 0.0030 | 0.0009 | 0.0020 |
| `flock_op_slim_r2` | 0.0004 | 0.0004 | 0.0000 |
| `flock_op_slim_r4` | 0.0000 | 0.0000 | 0.0000 |
| `flock_op_slim_r8` | 0.0013 | 0.0004 | 0.0006 |
| `flock_op_slim_r16` | 0.0015 | 0.0004 | 0.0001 |
| `flock_op_slim_r32` | 0.0018 | 0.0000 | 0.0011 |
| `flock_scalar` | 0.0000 | 0.0000 | 0.0000 |
| `lotus` | 0.0077 | 0.0048 | 0.0006 |
| `palimpzest` | 0.0072 | 0.0089 | 0.0006 |

`flock_op_r32`, `flock_op_slim_r4` and `flock_scalar` are bit-identical across all 3 reps in
**both** eras — the tightest quality evidence available.

## `extract_rescored.csv` is bounded-era and has no unbounded counterpart

The `f1_score` in the result JSONs (and therefore in the figures) is the harness's
**buggy macro-average**, which averages over off-vocab classes and manufactures the
apparent R>=4 quality collapse. `extract_rescored.csv` is the corrected scoring, produced by
`analysis/rescore_extract.py` from the `verdicts_*_diag.jsonl` per-row dumps.

**Job 3103523 emitted no verdict dumps** — the dump was not requested — so
`rescore_extract.py` cannot be run against the unbounded era. The knob for
`cross_system_analysis_clariden.sh` is **`VERDICT_DUMP=1`**, not `FLOCK_VERDICT_DUMP`:
line 210 of that script `unset`s any inherited `FLOCK_VERDICT_DUMP` before setting it
itself from `VERDICT_DUMP`, so passing `FLOCK_VERDICT_DUMP=1` fails **silently**. Note also
that the dump writes *during* the run, so a job that produces it cannot be cited for
timings — the dump and citable timings require two separate runs. The retained
`extract_rescored.csv` and `verdicts_*.jsonl` therefore describe the **bounded** era only.

Consequences:
- The quality gate above does not depend on them: it applies the harness scorer consistently
  to both eras, which is valid for detecting a *shift* even though the scorer is biased.
- **Do not publish an absolute F1 for the unbounded era from this directory.** That would
  require a re-run of 3103523 with `VERDICT_DUMP=1`, whose timings would not be citable.
- Do not read `extract_rescored.csv` alongside the current `cross_system_summary.csv` as if
  they were one measurement.

## Job 3103523

nid007380, COMPLETED, exit `0:0`, 48:14 of a 1:45:00 request, 2026-08-18. 11 arms x 3 reps.

**`lotus_rep2` failed** — `lotus_rep2_lotus.json` carries `"status": "failed"`, error
`'InternalServerError' object has no attribute 'choices'`, `row_count: 0`. The summarizer
correctly excludes it (`WARN lotus: 1 rep(s) excluded -> n=2`). Lotus is a control arm that
this change cannot touch, so it does not block, but its **+5.2% rests on 2 reps** and should
be reported as such rather than as a clean flat control. Note that
`summarize_cross_system.py` and the plot scripts silently drop non-`success` reps.

## Regenerating the summary

```
python3.11 analysis/summarize_cross_system.py \
  --results-dir analysis/figures/data/cross_system_sem_extract --rows 2000 --query 103
```

`--query 103` is required — the script defaults to **101** (that is sem_filter). Extract is
**103**.

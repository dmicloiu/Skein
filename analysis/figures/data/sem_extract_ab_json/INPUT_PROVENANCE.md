# sem_extract_ab_json — input provenance

This directory is the **JSON-encoding record** of the sem_extract operator A/B
(`PhysicalSemExtract`, `semantic_rewrite_enabled=true`, vs the scalar `llm_complete`).
Read this before using `morsel_ab_summary.csv`.

## READ FIRST — the response-schema era split

Every number in this directory was **re-measured on 2026-08-18** under a changed
response schema. The directory now holds the **unbounded-schema era only**. The
previous (bounded-schema) numbers are *not* mixed in — they were replaced wholesale,
all cells at once, and remain recoverable in git history at **`2565fe75`**.

| era | `BuildResponseFormat` per-row string element | jobs | in this dir |
|---|---|---|---|
| **bounded** (superseded) | `maxLength = 4 * (max_output_tokens / batch_rows)` chars | 3018609, 3093479, 3094740 | no — see `2565fe75` |
| **unbounded** (current) | no `maxLength`; ceiling is request-level `max_tokens` alone | 3103519, 3103521, 3112334 | **yes** |

`minItems`/`maxItems` still pin the answer count in both eras, so `items[i] <-> rows[i]`
is preserved. **Never mix the two eras in one table** — the operator arm moves by up to
+99% between them.

### The change is a decode-cost change, not a generation change

A `maxLength` makes the element a *length-counting* grammar, which costs xgrammar more
mask work per decoded token; at high concurrency that lands on the critical path. The
premise that generation itself is untouched is measured, not assumed:

- `prompt_tok` **bit-identical** to the bounded era in all 16 re-measured cells.
- `gen_tok` bit-identical in 14 of 16. The two exceptions are **scalar** arms — m8 `+1`
  token, m16 `+29` of 327,742 (**0.009%**) — on a code path that never executes
  `BuildResponseFormat`. Engine nondeterminism, not the change.
- The bound provably never clipped: under the bounded schema the longest per-row response
  was **8 chars** (`POSITIVE`/`NEGATIVE`/`NEUTRAL`) against a **64-char** bound, with zero
  rows within 56 chars of it.

Engine identical across the split: vLLM **v0.22.0**, xgrammar **0.2.1**,
`max_num_batched_tokens=8192`, GPU KV cache 1,278,0xx tokens, `--enable-prefix-caching`.

### Effect, old -> new (`rows_s`)

| cell | R | arm | bounded | unbounded | Δ |
|---|---|---|---|---|---|
| `m128_t128` | 1 | operator | 121.986 | **132.309** | +8.4% |
| `m128_t128` | 1 | scalar | 134.057 | 129.061 | −3.7% |
| `m8_t8` | 1 | operator | 119.0 | 130.9 | +10.0% |
| `m16_t8` | 1 | operator | 117.7 | 129.0 | +9.6% |
| `t16` | 32 | operator | 306.1 | 558.3 | +82.4% |
| `t8` | 32 | operator | 279.2 | 556.8 | +99.4% |
| any | 32 | scalar | 42.8–43.0 | 43.0–43.1 | ≤+0.5% (control) |

The operator/scalar ratio at `t16` goes **7.12x -> 12.98x**, and the m128 cell crosses
over: **0.910x -> 1.025x**. The gain is *not* a function of R — it tracks `tok_per_row`
(prefill-dominance). The absolute saving is near-constant per row; the *relative* gain is
that constant divided by the arm's baseline per-row cost, which is why the R=1 morsel cells
(`tok_per_row` 513.9) gain ~9–10% while the R=32 thread sweep (55.3) gains ~80–99%.

### `3103520` — the discarded m128 measurement

The first unbounded m128 attempt (3103520, nid006740, 2026-08-17) is **contaminated and was
never imported**. It reported operator 104.797 / scalar 116.569 rows/s, but **both** arms
fell ~13% below baseline — including the scalar control, which this change cannot affect.
The driver stalled and left the GPU idle: **62 of 249** interior samples logged
`Running: 0 reqs` (24.9%), in ~50 short stalls spread evenly across the run, mean `Running`
46.0 vs 78.1 in baseline. Errors, retries, memory pressure, engine config drift, differing
input and co-scheduling were all ruled out. `3112334` re-measures the identical cell on
nid007106 and is clean (1.5% interior zeros, both arms).

**The validity gate that caught it — use this, not the obvious ones.** `Waiting: 0`,
mean `Running > 0`, exit `0:0`, `pass_pct` 100 and `emitted == rows == passes` **all pass on
a contaminated run**. The gate is the count of interior `Running == 0` samples, taken
between the first and last active sample:

```
python3.11 - analysis/results/sem_extract_ab_json/<jobid> <<'EOF'
import re,sys,glob,os
pat=re.compile(r"Running: (\d+) reqs")
for f in sorted(glob.glob(os.path.join(sys.argv[1],"vllm-ep0-*.log"))):
    rows=[int(m.group(1)) for line in open(f,errors="ignore") if (m:=pat.search(line))]
    idx=[i for i,r in enumerate(rows) if r>0]
    if not idx: print(os.path.basename(f),"NO ACTIVE SAMPLES"); continue
    seg=rows[idx[0]:idx[-1]+1]; z=sum(1 for r in seg if r==0)
    print(f"{os.path.basename(f):26s} in_run={len(seg):4d} interior_zero={z:3d} "
          f"({100*z/len(seg):4.1f}%) avgRunning={sum(seg)/len(seg):5.1f}")
EOF
```

Reference values (operator arm first, earlier epoch suffix):

| job | era | operator | scalar |
|---|---|---|---|
| 3094740 | bounded, baseline | 0.0% zero, avg 78.1 | 1.0%, avg 73.0 |
| 3103520 | unbounded, **discarded** | **24.9% zero, avg 46.0** | 6.2%, avg 57.9 |
| 3112334 | unbounded, **current** | 1.5% zero, avg 60.6 | 1.5%, avg 70.4 |

### Why `avgRunning` is legitimately lower in 3112334 than in baseline

60.6 vs 78.1 looks like residual stalling but is the *expected* signature of the fix, and
the two are distinguishable. Under Little's law (`batch=1`, so one request per row):

| | baseline op (3094740) | current op (3112334) |
|---|---|---|
| mean `Running` (L) | 78.1 | 60.6 |
| throughput (λ) | 121.986 rows/s | 132.309 rows/s |
| **residency L/λ** | **0.640 s/req** | **0.458 s/req** (−28%) |

Occupancy falls *because* requests drain faster while throughput rises. The discriminator
against a stall is `Waiting`, which went **up** (5.0 -> 10.4): work is queued at all times,
so the GPU is never starved. The contaminated run's signature is the opposite — `Running`
at 0 with the driver not feeding.

## The two sweeps

| sweep | cells | input | R | job | node | elapsed |
|---|---|---|---|---|---|---|
| thread sweep | `*_t{1,2,4,8,16}` | `sembench/files/movie/data/sf_2000/Reviews.csv` (2000 rows, git-tracked) | 32 | 3103521 | nid007068 | 16:18 of 35:00 |
| morsel sweep | `*_m{1,8,16}_t8` | `sembench/files/movie/data/sf_300000/Reviews.csv` (300000 rows, **untracked**) | 1 | 3103519 | nid006768 | 29:29 of 45:00 |
| morsel sweep | `*_m128_t128` | same `sf_300000/Reviews.csv` | 1 | **3112334** | nid007106 | 1:11:31 of 2:00:00 |

All three COMPLETED, exit `0:0`; every cell `emitted == rows == passes`, `pass_pct` 100,
zero preemptions, GPU KV peak 2.7%.

The **input** split between the two sweeps is inherent to morsel mode, not a choice: the
thread sweep runs on an in-memory table (one row group = one morsel, so the scalar is
single-threaded regardless of `--threads`), while morsel mode materialises
`MORSELS × 2048` rows into an attached on-disk DB. `m128` needs 262144 rows, so only
`sf_300000` is large enough.

**Consequence:** `tok_per_row` is 55.3 in the thread sweep (R=32, prompt amortised) and
513.9 in the morsel sweep (R=1). Never compare `rows_s` between the two sweeps — only
operator-vs-scalar *within* a sweep.

### Node placement is not a confound

Established in the bounded era and unchanged: the m128 cell was once measured as two
single-arm jobs on two nodes (3093480/nid006051, 3093481/nid006588) and then in one
allocation (3094740/nid006103). Between-node spread was ≤1.2% (operator 121.986 vs 122.652,
scalar 134.057 vs 132.533) against a ~9–10% arm gap. Both arms of both unbounded m128 jobs
run inside a single allocation, so the question does not arise here.

## Morsel-sweep input — regenerated sample

`sf_300000` was deleted on 2026-08-14 by a `git clean -fd` in sembench and regenerated the
same day. `generate_data.py` is **not deterministic** despite `random_state=42`, so this is
a *different sample* from the one the XML-era morsel cells used.

- sha256(Reviews.csv) `5d64fdfa…`, 300000 rows, `--top-n 1400`
- first-262144 slice: mean `reviewText` 139.8 chars, p99 255, max 363

**Do not compare absolute `rows_s` between `sem_extract_ab_XML/` and the morsel cells here.**
Operator-vs-scalar ratios within this directory are sound (both arms read the identical
rows and report byte-identical `prompt_tok`).

## Morsel-sweep parameters

`MAX_OUT_MULT=256` uniformly across all morsel cells. At `rows-per-request=1` this is the
per-row output ceiling; the XML era used a mixed 64/64/256, and cap 64 binds on the 262k
scalar pass (longest review ≈ 90–120 tokens) — it is what failed job 2767405. The cap is
throughput-neutral here (clean rows stop at the JSON close token: `gen_per_req` = 10.0 in
every cell) and is applied to both arms, so the A/B stays fair.

The m128 re-run command, which reproduces 3112334 exactly:

```
MORSELS=128 MORSEL_THREADS=128 MAX_OUT_MULT=256 \
  DATA=$HOME/projects/sembench/files/movie/data/sf_300000/Reviews.csv \
  sbatch --time=02:00:00 analysis/slurm/sem_extract_ab_clariden.sh
```

Use `--time=02:00:00`, not 1:30 — 3103520 used 1:24:03 of a 1:30 request. This script has
**no** `MIN_OP_RATE` gate (that is the cross-system script).

## `Waiting` is not backpressure

High-concurrency arms log non-zero `Waiting` in both eras (bounded 3094740: 47 of 215
operator samples, 37 of 195 scalar, peaking 48–60). It is **not** engine backpressure: on
every such sample `Waiting == Deferred` exactly and `Running + Waiting` caps at exactly
**128** — the driver's own in-flight budget surfacing as briefly-deferred admissions. GPU KV
peak stays 2.7% with zero preemptions. Scalar arms below 128-way concurrency log
`Waiting: 0` always. **`Waiting: 0` is therefore not a validity gate** at 128-way
concurrency; see the interior-`Running == 0` gate above.

## Known wart in `morsel_ab_summary.csv`

`config` is synthesised as `<morsels>morsel_<datachunks>datachunk_<threads>threads`, which
does not encode R. The thread-sweep `t8` cell and the morsel `m1_t8` cell therefore collide
on `1morsel_1datachunk_8threads`, so **`(config, arm, rep)` is not a unique key in this
file — use `(config, arm, rep, R)`**. `sem_filter_ab_json` is unaffected. Rows remain
individually correct and distinguishable (`R` 32 vs 1, `rows` 2000 vs 2048).

## Regenerating the summary

```
python3.11 analysis/summarize_morsel_ab.py analysis/figures/data/sem_extract_ab_json \
  --out-dir analysis/figures/data/sem_extract_ab_json
```

Both arguments are required — the script defaults to `sem_filter_ab_json` on both sides.

## What is deliberately absent

`vllm-ep*.log` are **not** imported (~25 MB per m128 arm). This is a deliberate departure
from the router-analysis convention; sem_extract is scoped to result data. The full logs
remain on the cluster under `analysis/results/sem_extract_ab_json/<jobid>/` — the mechanism
evidence for the m128 A/B, including the `Running` trace that the validity gate reads, lives
**only** there.

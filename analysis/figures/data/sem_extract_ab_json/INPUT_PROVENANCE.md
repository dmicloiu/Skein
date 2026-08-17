# sem_extract_ab_json — input provenance

This directory is the **JSON-encoding record** of the sem_extract operator A/B
(`PhysicalSemExtract`, `semantic_rewrite_enabled=true`, vs the scalar `llm_complete`).
It holds two sweeps produced by `analysis/slurm/sem_extract_ab_clariden.sh` at different
times, on **different inputs** — and the morsel sweep itself spans two sessions. Read this
before using `morsel_ab_summary.csv`.

## The two sweeps

| sweep | cells | input | R | job(s) | date |
|---|---|---|---|---|---|
| thread sweep | `*_t{1,2,4,8,16}` | `sembench/files/movie/data/sf_2000/Reviews.csv` (2000 rows, git-tracked) | 32 | 3018609 | 2026-08-06 |
| morsel sweep | `*_m{1,8,16}_t8` | `sembench/files/movie/data/sf_300000/Reviews.csv` (300000 rows, **untracked**) | 1 | 3093479 (nid006040, both arms) | 2026-08-16 (morning) |
| morsel sweep | `*_m128_t128` | same `sf_300000/Reviews.csv` | 1 | **3094740** (nid006103, **both arms in one allocation**) | 2026-08-16 (evening re-measure) |

The morsel sweep is therefore **two sessions**, and the m128 cell is the later one. The m128
cells originally came from the morning campaign as two single-arm jobs on two nodes
(3093480 operator / nid006051, 3093481 scalar / nid006588) — split only to fit a <1h backfill
window. Because m128 is where the *equal concurrency buys equal throughput* claim lands, the
cell was re-measured in a single allocation with `ARMS="on off"`, and the cross-node pair was
replaced in place. It stays recoverable in git history at **0ff85915**.

**What the re-measure established is that the node placement never mattered here.** Comparing
the two measurements of the same configuration:

| arm | in-job, nid006103 (current) | cross-node (0ff85915) | spread |
|---|---|---|---|
| operator | 121.986 rows/s | 122.652 (nid006051) | 0.5% |
| scalar | 134.057 rows/s | 132.533 (nid006588) | 1.2% |
| **ratio op/scalar** | **0.910×** | 0.926× | — |

Between-node spread is ≤1.2% against a ~9–10% arm gap, so both measurements support the same
conclusion: the m128 cell is **a real direction, not parity**. The re-measure is therefore
*confirmatory* — it removes the caveat rather than correcting a number. Anyone auditing this
cell can stop at the current values; the point of recording the comparison here is that the
"the two arms were on different nodes" footnote is retired on evidence, not on assertion.

The m128 `result_*.json` carry a `"max_out_mult": 256` field that the `m{1,8,16}` results do
not. Expected, not a defect: the extract driver parsed `--max-out-mult` but never wrote it to
the result JSON until between the two sessions, so 3094740 is the first extract run that can
record it. The value was `256` in both sessions either way.

The **input** split between the two sweeps is inherent to morsel mode, not a choice: the
thread sweep runs on an in-memory
table (one row group = one morsel, so the scalar is single-threaded regardless of
`--threads`), while morsel mode materialises `MORSELS × 2048` rows into an attached
on-disk DB. `m128` needs 262144 rows, so only `sf_300000` is large enough. The same split
existed in the XML era.

**Consequence:** `tok_per_row` is 55.3 in the thread sweep (R=32, prompt amortised) and
513.9 in the morsel sweep (R=1). Never compare `rows_s` between the two sweeps — only
operator-vs-scalar *within* a sweep.

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

`MAX_OUT_MULT=256` uniformly across all eight morsel cells. At `rows-per-request=1` this is the
per-row output ceiling; the XML era used a mixed 64/64/256, and cap 64 binds on the 262k
scalar pass (longest review ≈ 90–120 tokens) — it is what failed job 2767405. The cap is
throughput-neutral here (clean rows stop at the JSON close token: `gen_per_req` = 10.0 in
every cell) and is applied to both arms, so the A/B stays fair.

Engine, both eras: vLLM **v0.22.0**, `StructuredOutputsConfig(backend='xgrammar',
disable_any_whitespace=True)`, `max_num_batched_tokens=8192`, GPU KV cache 1,278,032
tokens, `--enable-prefix-caching`, cold fleet per measurement. Qwen2.5-7B-Instruct, TP=1,
GH200. Verified identical from the vLLM boot logs of 2767404 and 3093480 — the engine is
**not** a confound in any cross-era comparison. Re-verified for the evening re-measure:
3094740 boots with the same `GPU KV cache size: 1,278,032 tokens`,
`max_num_batched_tokens=8192` and `backend='xgrammar'` as 3093480, so the engine is not a
confound between the superseded cross-node m128 pair and the in-job one that replaced it
either.

## Validity of the 2026-08-16 morsel runs

All four jobs COMPLETED, exit `0:0`. Every cell: `emitted == rows == passes == morsels ×
2048`, `pass_pct` 100, zero preemptions, GPU KV usage peak 2.7%, no errors in any run log.

| session | job | node | cells | elapsed |
|---|---|---|---|---|
| morning | 3093479 | nid006040 | `m{1,8,16}_t8`, both arms | 32:24 of 50:00 |
| morning | 3093480 | nid006051 | `m128_t128` operator — **superseded** | 38:50 of 45:00 |
| morning | 3093481 | nid006588 | `m128_t128` scalar — **superseded** | 36:05 of 45:00 |
| evening (m128 re-measure) | 3094740 | nid006103 | `m128_t128`, both arms — **current** | 1:12:42 of 2:00:00 |

3094740 detail: cold fleet per arm inside the one allocation; operator 262144 rows in
2148.96 s (121.986 rows/s), scalar 262144 in 1955.47 s (134.057 rows/s); both
`"max_out_mult": 256`, `tuple_format: json`, byte-identical `prompt_tok` (132093947) and
`gen_per_req` 10.0.

**Correction to the earlier `Waiting: 0 reqs throughout` claim** — that was too strong and is
withdrawn. High-concurrency arms *do* log non-zero `Waiting`: in 3094740, 47 of 215 operator
samples and 37 of 195 scalar samples, peaking near 48–60 reqs; the m1/m8/m16 operator arms
show the same, and so did the superseded cross-node m128 logs (56/214 and 32/198), so this is
not new to the re-measure. It is **not** engine backpressure: on every such sample
`Waiting == Deferred` exactly, and `Running + Waiting` caps at exactly **128** — the driver's
own in-flight budget surfacing as briefly-deferred admissions. GPU KV peak stays at 2.7% with
zero preemptions throughout. Scalar arms below 128-way concurrency log `Waiting: 0` always.

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

`vllm-ep*.log` are **not** imported (~25 MB per m128 arm, 50 MB for job 3094740 alone; and
the morning campaign's two single-arm m128 logs were named one epoch-second apart, so they
would near-collide on a flat import). This is a deliberate departure from the router-analysis
convention; sem_extract is scoped to result data. The full logs remain on the cluster under
`analysis/results/sem_extract_ab_json/<jobid>/` — the mechanism evidence for the m128 A/B
(per-step prompt throughput) lives only there.

# sem_extract_ab_json — input provenance

This directory is the **JSON-encoding record** of the sem_extract operator A/B
(`PhysicalSemExtract`, `semantic_rewrite_enabled=true`, vs the scalar `llm_complete`).
It holds two sweeps produced by `analysis/slurm/sem_extract_ab_clariden.sh` at different
times, on **different inputs**. Read this before using `morsel_ab_summary.csv`.

## The two sweeps

| sweep | cells | input | R | job(s) | date |
|---|---|---|---|---|---|
| thread sweep | `*_t{1,2,4,8,16}` | `sembench/files/movie/data/sf_2000/Reviews.csv` (2000 rows, git-tracked) | 32 | 3018609 | 2026-08-06 |
| morsel sweep | `*_m{1,8,16}_t8`, `*_m128_t128` | `sembench/files/movie/data/sf_300000/Reviews.csv` (300000 rows, **untracked**) | 1 | 3093479 / 3093480 / 3093481 | 2026-08-16 |

The split is inherent to morsel mode, not a choice: the thread sweep runs on an in-memory
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

`MAX_OUT_MULT=256` uniformly across all eight cells. At `rows-per-request=1` this is the
per-row output ceiling; the XML era used a mixed 64/64/256, and cap 64 binds on the 262k
scalar pass (longest review ≈ 90–120 tokens) — it is what failed job 2767405. The cap is
throughput-neutral here (clean rows stop at the JSON close token: `gen_per_req` = 10.0 in
every cell) and is applied to both arms, so the A/B stays fair.

Engine, both eras: vLLM **v0.22.0**, `StructuredOutputsConfig(backend='xgrammar',
disable_any_whitespace=True)`, `max_num_batched_tokens=8192`, GPU KV cache 1,278,032
tokens, `--enable-prefix-caching`, cold fleet per measurement. Qwen2.5-7B-Instruct, TP=1,
GH200. Verified identical from the vLLM boot logs of 2767404 and 3093480 — the engine is
**not** a confound in any cross-era comparison.

## Validity of the 2026-08-16 morsel run

All three jobs COMPLETED, exit `0:0`. Every cell: `emitted == rows == passes == morsels ×
2048`, `pass_pct` 100, zero preemptions, `Waiting: 0 reqs` throughout, GPU KV usage peak
2.7%. No errors in any run log. Nodes nid006040 (m1/m8/m16), nid006051 (m128 operator),
nid006588 (m128 scalar).

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

`vllm-ep*.log` are **not** imported (57 MB per m128 job, and the two m128 jobs' logs are
named one epoch-second apart, so they would near-collide on a flat import). This is a
deliberate departure from the router-analysis convention; sem_extract is scoped to result
data. The full logs remain on the cluster under
`analysis/results/sem_extract_ab_json/<jobid>/`.

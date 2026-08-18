# RESUME — unbounded sem_extract response schema

**Status 2026-08-18: CAMPAIGN COMPLETE. All 7 jobs done, every merge gate green, all
data imported, both summaries regenerated, both provenance files written. The only
remaining items are (a) the user's commit/push, and (b) the §5 ratio restatement in the
paper checkout, which does not live on this machine.**

Branch `fix/extract_unbounded_schema`, code commit **`2565fe75`** (unchanged).
Working tree carries the data import: 240 modified + 35 new files.

---

## Verdict

The change is validated. `PhysicalSemExtract::BuildResponseFormat` no longer puts a
`maxLength` on the per-row string element; the ceiling is the request-level `max_tokens`
alone, with `minItems`/`maxItems` still pinning the answer count.

**Generation is provably unchanged, throughput is up a lot.** `prompt_tok` and `gen_tok`
bit-identical in all 11 cross-system arms and 14/16 flock cells (the 2 exceptions are
scalar arms, ≤0.009%, on a path that never runs `BuildResponseFormat`).

| gate | result |
|---|---|
| `prompt_tok` / `gen_tok` unchanged | **PASS** |
| non-flock controls unchanged | **PASS** — palimpzest −1.2% at 128-way concurrency; lotus +5.2% on n=2 |
| scalar arms unchanged | **PASS** — cross-system +0.2%; morsel scalars within ±5% |
| quality unchanged | **PASS** — all 11 arms, shift < own rep spread |
| operator throughput up | **PASS** — +8.4% to +101% |
| m128 ratio >= 1.0x | **PASS** — **0.910x -> 1.025x** |

## The m128 question is settled

`3112334` (nid007106, 1:11:31, exit `0:0`) re-measured the cell that `3103520`
contaminated. **The stall did not reproduce**: interior `Running == 0` fell 24.9% -> 1.5%,
and the scalar control returned to baseline (129.1 vs 134.1, −3.7%) instead of collapsing
13%. So `3103520` was node-level degradation on nid006740, not a regression in the build.
`3103520` is discarded and was never imported.

Operator **132.309** rows/s vs scalar **129.061** = **1.025x**, against 0.910x before.
The operator stops being behind at equal concurrency, which was the substantive claim.

One subtlety worth not re-deriving: the re-measure's mean `Running` (60.6) is *below*
baseline's (78.1) even though it is faster. That is the expected signature, not residual
stalling — by Little's law per-request residency drops 0.640 -> 0.458 s (−28%), so fewer
requests are resident while throughput rises. The discriminator against a real stall is
`Waiting`, which went **up** (5.0 -> 10.4): work is queued at all times, so the GPU is
never starved. A stall looks like the opposite — `Running` at 0 with the driver not feeding.

## The mechanism: `tok_per_row`, not R

The gain is **not** a function of R. The two R=1 cross-system arms differ by 3x (+20.5% vs
+67.3%) and the slim family is an inverted U peaking at R=8. What is near-constant is the
**absolute** saving: **1.48–1.90 ms/row on every operator arm**, vs ≤0.20 ms/row on all
three controls. Relative gain = that constant / the arm's baseline per-row cost, so
prefill-dominated arms gain least in percentage terms. R is only a proxy, because raising R
amortises the prompt and drives `tok_per_row` down.

This is why the R=1 morsel cells (`tok_per_row` 513.9) gain ~8–10% while the R=32 thread
sweep (55.3) gains ~80–99%. Both surfaces agree.

## §5 — the ratio roughly doubles, on two independent surfaces

| surface | old | new |
|---|---|---|
| thread sweep t16 (558.3 / 43.0) | 7.12x | **12.98x** |
| cross-system `flock_op_r32` / `flock_scalar` (537.9 / 43.0) | 6.64x | **12.51x** |

Agreement to ~4%. **Any paper text quoting ~7.0x needs rewriting.** Nothing in *this* repo
still quotes the stale number — the only in-repo occurrences were in the two provenance
files and are updated. The `~7.7×` figures in
`analysis/figures/sem_filter_ab_conclusion.md` are **sem_filter**, a different operator
that never touches `BuildResponseFormat`; they are correctly untouched. There is no paper
checkout on this machine, so that edit must happen wherever it lives.

---

## What was done this session

1. `3112334` validity gate (interior `Running == 0`) — **passed**, stall did not reproduce.
2. Token-identity check on the m128 cell — `prompt_tok` bit-identical, `gen_tok` +9 of
   2.62M (0.0003%), zero preemptions, 262,145 successes.
3. F1 stability gate — **all 11 arms pass**.
4. Imported everything at once, so no summary ever passed through an era mix:
   - `3103519` + `3103521` + `3112334` -> `analysis/figures/data/sem_extract_ab_json/`
     (72 files, exact 1:1 filename replacement — verified by set diff; no `vllm-ep*.log`,
     per this directory's convention)
   - `3103523` -> `analysis/figures/data/cross_system_sem_extract/` (198 files incl.
     `vllm-*.log`, per *that* directory's convention)
5. Regenerated `morsel_ab_summary.csv` and `cross_system_summary.csv`.
6. Rewrote `sem_extract_ab_json/INPUT_PROVENANCE.md` and **created**
   `cross_system_sem_extract/INPUT_PROVENANCE.md`, both recording the schema-era split.

### Remaining

- **User commits and pushes** (nothing is staged or committed by this session).
- Restate 7.0x -> ~12.98x / 6.64x -> 12.51x in the paper checkout.

---

## Open caveats that survive the merge

- **No absolute F1 exists for the unbounded era.** `3103523` emitted no
  `verdicts_*.jsonl` (`FLOCK_VERDICT_DUMP` unset), so `analysis/rescore_extract.py` cannot
  run on it. The retained `extract_rescored.csv` and 8 `verdicts_*.jsonl` in
  `cross_system_sem_extract/` are **bounded-era only** and are kept because they cannot be
  regenerated. The quality gate does not depend on them — it applies the harness scorer
  consistently to both eras, which is valid for detecting a *shift* even though that scorer
  is the known-buggy macro-average. **Do not publish an absolute unbounded-era F1** without
  a re-run with the dump enabled.
- `lotus` +5.2% rests on **n=2** — `lotus_rep2` failed with
  `'InternalServerError' object has no attribute 'choices'`. It is a control arm this
  change cannot touch, so it does not block, but report it as n=2.
- `cross_system_sem_extract/` is **mixed-era by design**: 84 retained `vllm-178{4,6}*.log`
  are bounded-era, the 33 `vllm-1787*.log` are unbounded. Neither summarizer nor either
  plot script reads the logs, so no output is contaminated.

## Traps (do not re-learn these)

- **`Waiting: 0` is not a validity gate** at 128-way concurrency, and neither is mean
  `Running > 0`. `Waiting: 0`, exit `0:0`, `pass_pct` 100 and `emitted == rows == passes`
  **all pass on a contaminated run**. The gate is the count of interior `Running == 0`
  samples, between the first and last active sample. The checker script is preserved in
  `sem_extract_ab_json/INPUT_PROVENANCE.md`.
- **The "gain scales with R" mechanism was wrong** — it is `tok_per_row` / prefill
  dominance, with a near-constant absolute saving.
- **The "F1 bit-identical / ±0.003" gate was wrong** — baseline `slim_r1` was never
  bit-identical across its own reps (spread 0.0030), and untouched `palimpzest` drifts
  0.0089. Gate on "shift < own rep spread" instead.
- **`summarize_morsel_ab.py` needs both args** (defaults to `sem_filter_ab_json` on both
  sides); **`summarize_cross_system.py` defaults to query 101** — extract is **103**.
- **Never compare `rows_s` between the thread sweep and the morsel sweep** — different R,
  different input, `tok_per_row` 55.3 vs 513.9. Only operator-vs-scalar *within* a sweep.
- `morsel_ab_summary.csv`: `(config, arm, rep)` is **not** a unique key — the thread-sweep
  `t8` cell and the morsel `m1_t8` cell collide on `1morsel_1datachunk_8threads`. Use
  `(config, arm, rep, R)`.
- The m128 cell needs `MORSELS * ROW_GROUP_SIZE` = 262,144 rows, so only `sf_300000` is
  big enough; it is untracked and non-deterministically regenerated, so absolute `rows_s`
  is not comparable to the XML era. Use `--time=02:00:00` (3103520 used 1:24:03 of 1:30).

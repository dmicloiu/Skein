# Number consolidation — changelog (old → new)

**Two rounds.** Round 1 standardised the tuple encoding (XML → JSON) and re-ran the
engine, prompt and scale-out studies. Round 2 re-measured the two cross-system tables in
single sessions, after the extract table was found to mix sessions and evaluator
versions across its reps. Sections below carry both where they differ; the final column
is what the documents cite.

Every flock measurement now renders the **JSON** tuple encoding (`tuple_format: json`,
pinned in both standalone drivers and already used by the sembench cross-system
harness). Pre-standardisation runs used flock's default **XML** encoding, which sends
~6 more tokens per row and — the reason this matters — produces *different verdicts*.

All numbers below: movie sf_2000 (2000 rows) unless stated, Qwen2.5-7B-Instruct,
one GH200, greedy + xgrammar guided JSON, `in_flight_cap` 128 unless stated.
Quality is row-level P/R/F1 against `scoreSentiment`; `set F1` is the sembench
unique-`reviewId` convention, carried only to compare against §2.

---

## 0. The reconciliation this bought

evaluation.md printed **two different F1s for the same config** — §2 said 0.931
(sembench, JSON), §3 said 0.908 (standalone, XML). With the encoding standardised the
two harnesses agree:

| full prompt, R=1 | rows/s | set F1 |
|---|---|---|
| §2 cross-system (sembench, JSON, 3 reps) | 145.7 | 0.931 |
| §3 sweep (standalone, JSON, 3 reps) — **new** | 144.8 | 0.932 |
| §3 sweep (standalone, XML, 3 reps) — old | 138.7 | 0.911 |

The 0.908-vs-0.931 discrepancy was the tuple encoding, not reps, not the scorer, and
not node variance. It is now closed to 0.001 F1 / 0.6% rows/s.

## 1. Engine (§1) — operator vs scalar

R=32, one morsel, thread sweep 1–16 (2000 rows):

| | operator rows/s | scalar rows/s | ratio | tok/row |
|---|---|---|---|---|
| XML (old) | 584.5–592.5 | 76.7–77.3 | 7.7× | 59.8 |
| **JSON (new)** | **619.9–645.8** | **76.4–77.1** | **~8.1×** | **53.0** |

R=1, morsel cells (`ROW_GROUP_SIZE` 2048):

| cell | rows | XML op / scalar (ratio) | JSON op / scalar (ratio) |
|---|---|---|---|
| 8 morsels / 8 threads | 16,384 | 133.8 / 65.4 (2.05×) | **137.7 / 68.0 (2.02×)** |
| 16 morsels / 8 threads | 32,768 | 138.6 / 64.0 (2.17×) | **138.1 / 68.6 (2.01×)** |

Pass rate moves 71.7–72.1% (XML) → 81.0% (JSON), the same permissiveness shift the
full prompt shows at R=1. Operator throughput is unchanged within noise: the encoding
is a prompt-size effect, not an engine effect.

The headline 1-morsel R=1 row (old: 137.2 / 17.3 = 7.9×, 32,744 rows) has no direct
JSON re-run; the JSON **TP=1 cell** covers it at 2000 rows: 146.9 / 18.9 = **7.8×**.

## 2. Cross-system (§2) — re-measured in one session

Round 1 changed only the *metric* (raw tok/s → computed tok/s), since sembench always
configured JSON. Round 2 re-ran all twelve arms in one session, because the table had
spanned three (flock full and scalar 06-20, slim arms 07-20, LOTUS/PZ June).

| arm | rows/s R1 → R2 | computed tok/s R1 → R2 | F1 |
|---|---|---|---|
| flock op full R=1 | 145.7 → **148.4** | 33,655 → **34,284** | 0.931 |
| flock op full R=32 | 642.2 → **663.2** | 28,595 → 29,534 | 0.527 |
| flock op slim R=1 | 373.4 → 373.6 | 23,490 → 23,497 | 0.878 → 0.877 |
| flock op slim R=4 | 621.6 → **592.5** | 31,770 → 30,287 | 0.843 → 0.842 |
| flock op slim R=8 | 657.8 → **678.9** | 32,106 → 33,133 | 0.834 |
| LOTUS | 258.1 → **247.7** | 20,033 → **19,375** | 0.833 |
| Palimpzest | 137.4 → **129.8** | 17,286 → 16,347 | 0.877 → 0.878 |

- **Raw tok/s → computed tok/s** (round 1): the headline "~1.9× LOTUS" was raw. On
  computed tokens it became 1.68×, and after the single-session re-measurement
  **1.77×** (34,284 / 19,375). Palimpzest flips from second-best on raw tok/s to
  *slowest* engine — 85.5% of its prompt tokens are prefix-cache hits.
- **Individual arms moved up to 5% in both directions** between rounds, which is the
  cross-node variance a single-session table exists to remove. No systematic offset:
  the hypothesis that the 07-20 session ran fast was tested and **not** supported.
- **One narrow claim weakened; the headline is untouched.** flock's quality point is
  the full prompt at R=1, F1 **0.931** before and after — +0.098 over LOTUS and +0.053
  over Palimpzest. What flipped is only the *slim* R=1 arm against Palimpzest
  (0.878 vs 0.877 became 0.877 vs 0.878), so that arm is now 2.9× the throughput at
  parity rather than a hair ahead. The LOTUS comparison is unaffected (slim R=4: 2.4×
  the rate at +0.009 F1).
- LOTUS was n=2 for most of this work — one June rep died with an endpoint error and
  was silently dropped by the summarizer. It is n=3 from round 2 on.

## 3. Prompt (§3) — rep-hardened, JSON, 3 reps, median [min–max]

| head | R | rows/s (old, XML, 1 rep) | **rows/s (new, JSON, 3 reps)** | F1 old | **F1 new** |
|---|---|---|---|---|---|
| full | 1 | 138.7 | **144.8 [141–149]** | 0.908 | **0.926** |
| full | 2 | 229.1 | 234.9 [233–247] | 0.801 | 0.821 |
| full | 4 | 355.4 | 365.6 [353–387] | 0.799 | 0.770 |
| full | 8 | 476.9 † | 524.6 [509–528] | 0.739 † | 0.737 |
| full | 16 | 572.6 † | 618.9 [610–628] | 0.677 † | 0.669 |
| full | 32 | 589.6 † | 641.4 [638–654] | 0.478 † | 0.506 |
| slim | 1 | 331.0 | **329.8 [303–333]** | 0.887 | **0.872** |
| slim | 2 | 488.5 | 488.1 [486–521] | 0.848 | 0.846 |
| slim | 4 | 603.5 | 608.1 [571–633] | 0.825 | 0.834 |
| slim | 8 | 677.4 | 621.9 [498–628] | 0.810 | 0.825 |
| slim | 16 | 708.1 | 649.9 [591–671] | 0.775 | 0.803 |
| slim | 32 | 631.8 | 596.9 [553–617] | 0.677 | 0.732 |

† The old full-prompt sweep only ran R=1/2/4 in the `sem_filter_prompt_slim` job; the
R=8/16/32 old cells come from the earlier `prompt_analysis` R-sweep (also XML). The
two XML families agree exactly where they overlap (R=1/2/4: F1 0.908 / 0.801 / 0.799),
so the column is consistent — but only the new JSON column is one job family throughout.

Notes:

- **The slim R≥2 cells reproduce the cross-system slim arms exactly** (set F1 0.854 /
  0.842 / 0.834 / 0.813 / 0.746 vs 0.854 / 0.843 / 0.834 / 0.811 / 0.746). Expected:
  at R>1 the batch-adaptive slim head renders row-major id lines and never consults
  `tuple_format`, so those cells are encoding-independent — measured, not assumed.
- **"Throughput peaks at R=16" → "plateaus at R≈4–16."** With spread, slim R=4
  [571–633], R=8 [498–628] and R=16 [591–671] overlap. The plateau is where
  #requests ≈ `in_flight_cap`=128; R=32 (63 requests) under-fills it. Cap artifact,
  not a property of batching.
- §3(a)/(b) mechanism results (positional miss-rate ramp, schema ablation) remain
  single-rep XML and are labelled as such; they are mechanism, not operating points.

## 4. Scale-out (§4) — laws hold, absolutes requoted

Round 1 re-ran TP and DP on JSON; the fuller rep set and the added `unified` family then
firmed the numbers up. Final values:

| | XML | JSON (final) |
|---|---|---|
| TP operator, cap 128/256/512 | 140.8 / 210.4 / 370.5 | **140.0 / 232.6 / 394.3** |
| TP op/scalar gap | 8.2 / 8.7 / 12.9× | **7.45 / 9.16 / 12.93×** |
| TP per-GPU at TP=4 | 0.66 | **0.71** |
| DP 1 / 2 / 4 × TP1 | 134.5 / 282.1 / 560.7 | **136.7 / 290.6 / 582.8** |
| DP per-GPU at N=4 | 1.04 | **1.07** (base-limited) |
| fixed budget 4×TP1 / 2×TP2 / 1×TP4 | 559.0 / 436.0 / 384.6 | **574.6 / 451.6 / 368.8** |
| DP vs TP at 4 GPUs | 1.42× | **1.51×** |
| scalar on the 4-endpoint fleet | 17.7 | **18.6** |

- Every law survived: DP linear, TP sub-linear, replicas beat shards, cap knee at 128·N,
  F1 invariant. Only absolutes moved, by 2–6%.
- **The 1.42× → 1.51× shift needed its own explanation.** `scale_out.md` had argued the
  in-job re-run corrected a cross-study 1.51× down to 1.42× by removing node variance.
  The in-job JSON measurement returns 1.51× again, so the 1.42× was a measurement under
  the old prompt configuration, not a variance correction. The document now says so.
- **Per-GPU efficiency reads 1.07, above unity.** That is an under-driven N=1 base at
  cap 128, not superlinear scaling; the claim is "no measurable scaling loss".
- **C\* is now interpolated, then snapped to the nearest power-of-two cap.** The old
  rule (smallest measured cap clearing 98% of plateau) sat a knife-edge from a
  factor-of-two jump: 32B's c256 cell reads 98.5% in one run and 97.8% in another, which
  slid the whole 32B curve an octave in `dp_cap_collapse`. Interpolated C\* is ~118 per
  endpoint at 7B and ~73 at 32B.

## 4b. Extract quality — the "R≥4 collapse" was a scoring artefact

sembench's Q103 scorer called sklearn macro F1 without a fixed `labels` set, so it
averaged over every class in `y_true | y_pred`. Guided decoding emits an
off-vocabulary label on a handful of rows; each one adds a class with F1 0 and
divides the macro by 3 or 4 instead of 2. Re-scoring the per-row verdict dumps with
the average restricted to the gold labels (`analysis/rescore_extract.py`) reproduces
the reported numbers exactly under the old rule and recovers the true ones:

| R | **F1 (fixed labels)** | F1 as reported | off-vocabulary rows | classes averaged |
|---|---|---|---|---|
| 1 | 0.827 | 0.827 | 0 | 2 |
| 2 | 0.819 | 0.819 | 0 | 2 |
| 4 | **0.823** | 0.551 | 1 (`NEUTRAL`) | 3 |
| 8 | **0.834** | 0.418 | 6 `NEUTRAL` + 1 `MIXED` | 4 |
| 32 | **0.640** | 0.431 | 3 (`NEUTRAL`) | 3 |

Seven rows in 1904 — 0.37% — halved the reported score. **Batched extract does not
collapse**: quality is flat to R=8 and drops only at R=32. Ruled out along the way:
the flock operator (new e2e test `SemExtractE2E.ProductionShapePairingHolds` — 2000
rows at batch 8 / cap 128, every row keeps its own completion), model misalignment
(identity beats every within-batch permutation, 0.834 vs 0.47–0.49), row loss
(2000/2000 emitted, status success), and truncation (`finished_reason=length` = 0).

Consequences: `prompt_engineering.md` Q5's extract section must be rewritten — the
collapse, the "count/chunk does not transfer to extract" claim, and the R=1/2
operating-point recommendation all rest on the artefact. And the same scorer graded
LOTUS (0.565) and Palimpzest (0.577): they emit free text with no guided decoding,
so they produce *more* off-vocabulary variants and were penalised harder. The extract
cross-system margins are untrustworthy in both directions until all three systems are
re-scored under the fixed label set. Fix applied in
`sembench/src/scenario/movie/evaluation/evaluate.py` (uncommitted).

## 4c. Extract cross-system — the conclusion reverses, twice

Round 1 fixed the scorer. Round 2 re-ran all eleven arms in one session, because the
table still mixed sessions and evaluator versions across reps (rep1 Aug-06 with the
fixed scorer and verdict dumping on, rep2/3 July with the old scorer). Final:

| arm | rows/s | F1 as first reported | **F1 final** | |
|---|---|---|---|---|
| Palimpzest | 83.9 | 0.577 | **0.866** | max quality |
| LOTUS | 247.0 | 0.565 | **0.847** | |
| flock slim R=8 | 294.4 | 0.418 | **0.837** | best flock point |
| flock slim R=1 | 211.9 | 0.832 | 0.834 | |
| flock slim R=4 | 282.8 | 0.551 | 0.827 | |
| flock full R=1 | 115.4 | 0.807 | 0.807 | |
| flock slim R=16 | 295.3 | 0.519 | 0.779 | |
| flock slim R=32 | 287.4 | 0.431 | 0.648 | |
| flock full R=32 | 285.0 | 0.314 | 0.628 | |
| flock scalar | 42.9 | 0.411 | 0.617 | |

- **flock does not dominate extract on quality.** Palimpzest leads, LOTUS beats every
  flock point; flock's contribution is throughput (slim R=8 at 1.19× LOTUS's rate for
  −0.010 F1). The old "+0.26 F1 over LOTUS, flock dominates" is withdrawn.
- **Batched extract does not collapse.** Flat R=1 → R=8 (0.834 → 0.837), degrading only
  at R=16/32. The "count/chunk does not transfer to extract" claim is withdrawn.
- **One throughput number moved 46% in round 2**: `flock_op_r32` read 530 rows/s and now
  reads 285. The new value matches the independent standalone A/B of the same workload
  (279–306 rows/s, identical 55.3 tok/row) that the old one never did. That check —
  cross-checking the cross-system harness against a standalone measurement of the same
  configuration — is what caught it, and is worth keeping in the methods.
- **Timing hygiene**: rep1 of every arm had been a dump-enabled run, inflating one arm
  by 24% (`slim_r1` 10.47 s against 8.42/8.77). All reps are now dump-free; CV ≤3.3%.

## 5. Extract — engine A/B requoted

R=32, one morsel, thread sweep (2000 rows, JSON): operator 279–306 rows/s vs scalar
42.8–43.0 → **~7.0×**, thread-independent on both arms; 100% of rows emitted.

**Superseded by §5c below.** With the response-schema bound removed the same sweep reads
operator 536–558 vs scalar 43.0–43.1 → **~13.0×**. Anything quoting ~7.0× for the
*extract* operator is stale; the ~7.7×/8.1× figures elsewhere in this document are the
**filter**, a different operator that never carried the bound, and are unaffected.

### 5b. Extract morsel ladder — moved to JSON, then the m128 cell re-measured in-job

The chapter's morsel table was the last XML dependency in the write-up. Two rounds:

| cell | XML (2026-07-15) | JSON cross-node (3093480/81) | JSON in-job (3094740) |
|---|---|---|---|
| m128 operator | 142.4 | 122.7 | **122.0** |
| m128 scalar | 135.5 | 132.5 | **134.1** |
| ratio op/scalar | 1.05× | 0.93× | **0.91×** |

The m1/m8/m16 cells are unchanged from the morning JSON campaign (114.3/119.0/117.7
operator, 16.1/60.8/63.6 scalar). Only m128 was re-run, because it is the cell the
convergence claim rests on and it was the only one split across two nodes.

Two things changed in what may be said. First, the cross-node caveat is retired **on
evidence**: the two independent sessions agree to 0.5% (operator) and 1.2% (scalar), so
placement never mattered and the ~10% arm gap is a real direction. Second, that direction
now has a cause — the operator's bounded output schema, `CORPUS_AUDIT.md` 1.1 — which also
explains the XML→JSON drop in the operator column above, and means the cross-era rows of
this table compare two different output contracts and must not be read as a regression in
the dispatch engine.

## 5c. Extract — the length bound removed, and the numbers roughly double

`PhysicalSemExtract::BuildResponseFormat` put a `maxLength` on each per-row string,
derived from the output-token budget. A length bound is a counting grammar: xgrammar
spends **14–27× more mask work per decoded token** on it than on an open string, and the
cost is flat in the bound's value, so both job families paid it in full. It is
operator-only, which is why every scalar arm was untouched. Removed in `2565fe75`; corpus
re-measured 2026-08-18 (jobs 3103519, 3103521, 3103523, 3112334).

**Generation is provably unchanged.** `prompt_tok` and `gen_tok` are bit-identical on all
nine cross-system flock arms; F1 moves ≤0.003 (five arms exactly 0.000), inside each arm's
own rep spread. So this is a pure throughput change and every quality number carries over.

Cross-system extract (`sf_2000`, 3 reps, median):

| arm | rows/s before → after | F1 |
|---|---|---|
| flock slim R=8 | 294.4 → **591.9** (+101%) | 0.837 → 0.836 |
| flock slim R=4 | 282.8 → 547.7 (+94%) | 0.827 → 0.827 |
| flock full R=32 | 285.0 → 537.9 (+89%) | 0.628 → 0.628 |
| flock full R=1 | 115.4 → 139.1 (+21%) | 0.807 → 0.807 |
| flock scalar (control) | 42.9 → 43.0 (+0.2%) | 0.617 → 0.617 |
| Palimpzest (control) | 83.9 → 82.9 (−1.2%) | 0.866 → 0.868 |
| LOTUS (control, n=2) | 247.0 → 259.8 (+5.2%) | 0.847 → 0.845 |

Headline consequences: slim R=8 goes from **1.19× to 2.28× LOTUS's rate** at −0.009 F1;
full R=1 engine efficiency goes 27,580 → **33,240** computed tok/s, ~2.25× either peer;
and the operator/scalar ratio roughly doubles, **7.1× → 13.0×** on the thread sweep and
6.6× → 12.5× on the cross-system session, two independent surfaces agreeing to 4%.

Morsel ladder (Table 2 of the chapter):

| cell | operator | scalar | ratio before → after |
|---|---|---|---|
| m1 | 114.3 → 130.9 | 16.1 → 16.0 | 7.10× → **8.18×** |
| m8 | 119.0 → 130.9 | 60.8 → 63.4 | 1.96× → 2.06× |
| m16 | 117.7 → 129.0 | 63.6 → 61.0 | 1.85× → 2.11× |
| m128 | 122.0 → 132.3 | 134.1 → 129.1 | 0.910× → **1.025×** |

**Read m128 as parity, not as an operator lead.** The crossing owes 8.4% to the operator
and 3.7% to a scalar move the change cannot have caused; across its three measurements the
scalar reads 132.5 / 134.1 / 129.1, mean 131.9 against 132.3.

**The gain is not a function of R.** It is a near-constant **1.48–1.90 ms/row** on every
operator arm, against ≤0.20 ms/row on all three controls; relative gain is that constant
divided by the arm's baseline per-row cost. That is why prefill-dominated cells
(`tok_per_row` 513.9) gain ~10% while decode-dominated ones (55.3) double. It also matches
the CPU microbenchmark to ~10%: 4.5 generated tokens/row × ~400 µs saved ≈ 1.8 ms/row.

**Llama replicated it** (job 3116950, 11 arms, n=3): operator arms +17% to +92% at
identical F1, `flock_scalar` **+0.0%** (40.4 → 40.4, F1 bit-identical), saving
1.30–1.70 ms/row. Llama's slim R=16 becomes the highest-quality *and* fastest arm in its
table — 557.3 rows/s at F1 0.871, i.e. 3.79× LOTUS at +0.078 F1 and 2.61× Palimpzest at
+0.030 — and the extract operator/scalar ratio now reads **12.98× / 12.51× / 12.35×** on
three surfaces across two families, agreeing to ~5%. All extract data is one schema era.

## 6. The controlled encoding observation

Same build, same node family, full sf_2000, R=1 — the delta the standardisation costs
or buys, measured rather than assumed:

| head | encoding | rows/s | P | R | F1 | prompt tok/row |
|---|---|---|---|---|---|---|
| full | XML | 140.2 | 0.959 | 0.863 | 0.908 | 505.7 |
| full | **JSON** | 144.8 | 0.883 | 0.974 | **0.926** | 498.0 |
| slim | XML | 356.2 | 0.985 | 0.807 | **0.887** | 148.5 |
| slim | **JSON** | 329.8 | 0.988 | 0.780 | 0.872 | 140.8 |

**The effect is not uniformly pro-JSON.** On the full head JSON gains +0.018 F1 (a
large recall gain, 0.863 → 0.974, against a precision loss); on the slim head it costs
−0.015 F1 (recall 0.807 → 0.780). JSON is fewer tokens per row in both cases. The
case for standardising is consistency with the cross-system harness plus prompt
economy — not a uniform quality win, and the write-up should say so.

(rows/s across the two encodings is 1 job each and carries the ~±10% node variance;
the F1 column is deterministic under greedy and is the reliable half of this table.)

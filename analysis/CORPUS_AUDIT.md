# Critical audit of the measurement corpus

An adversarial pass over everything the thesis rests on: what is inconsistent, how much it
matters, and what (if anything) should be re-run. Ordered by how much damage each issue
does to a claim, not by how easy it is to fix.

Nothing here invalidates a headline result. Both Tier-1 items open at the first pass are
now **fixed rather than merely explained**: the cross-era extract shift was traced to a
length bound in the operator's guided-decoding schema and the bound removed (1.1), which
in turn converged the cross-node parity cell (1.2). Its one downstream consequence, an
extract schema-era split between the two model families, closed when the Llama extract
corpus was re-measured on the same schema (1.3).

---

## Tier 1 — affects what a claim may say

### 1.1 Operator regression between eras (extract A/B) — diagnosed and FIXED

**What.** Re-running the extract morsel ladder on JSON, the operator dropped 10–20% at
every cell while the scalar held flat:

| morsels | operator XML → JSON | scalar XML → JSON |
|---|---|---|
| 1 | 142.1 → 114.3 (−20%) | 16.2 → 16.1 (−1%) |
| 8 | 132.6 → 119.0 (−10%) | 58.8 → 60.8 (+3%) |
| 16 | 135.7 → 117.7 (−13%) | 60.1 → 63.6 (+6%) |
| 128 | 142.4 → 122.7 (−14%) | 135.5 → 132.5 (−2%) |

**Why it is not the obvious explanations.** The input sample changed (`sf_300000` was
regenerated), but that should move both arms equally and `tok_per_row` shifted only 1.5%
(521.5 → 513.9). The engine is byte-identical (vLLM 0.22.0, same flags, verified from boot
logs). Generation length is unchanged (10 tok/req in every cell, both eras). Digging into
latency: at m1 the operator's decode time rose 46% (360 → 524 ms) at unchanged concurrency
(~122), while the scalar's was identical to the millisecond (47 → 47 ms) at concurrency 1.
At m128, where the scalar *also* runs ~116 concurrent, it moved only 6%. So it is not
simply a slower node — it is specific to the operator's request stream.

**Cause: a bounded output schema in the operator's guided decoding.** The regression is
confined to the **decode** phase and to the **operator** arm, and both facts point at one
commit.

| cell | operator decode XML → JSON | scalar decode XML → JSON |
|---|---|---|
| 1 | 360 → 524 ms (+46%) | 47 → 47 ms (0%) |
| 8 | 386 → 562 ms (+46%) | 83 → 82 ms (−1%) |
| 16 | 384 → 550 ms (+43%) | 77 → 77 ms (0%) |
| 128 | 383 → 574 ms (+50%) | 449 → 471 ms (+5%) |

Prefill is untouched (202→222, 204→202, 204→200, 186→185 ms) and queueing fell. All ten
generated tokens per request in both eras, so this is ${\approx}38 \to 55$ ms per decoded
token on the operator while the scalar does not move.

Commit `29901df3` (2026-07-16 18:09) added `maxLength` to
`PhysicalSemExtract::BuildResponseFormat`, bounding each per-row string at
`max_output_tokens / R × 4` characters — 1024 at R=1 with `MAX_OUT_MULT=256`. Every XML
extract cell booted its fleet on **2026-07-15** (11:02 and 12:13 UTC), one day *before*
that commit, so the XML era decoded against an unbounded string and the JSON era against
a 1024-character counting constraint. `BuildResponseFormat` is a member of
`PhysicalSemExtract`: the scalar's otherwise identical `OutputType::STRING` schema carries
no bound, which is exactly why the scalar arm is flat. `PhysicalSemFilter`'s element is
`{"type": "boolean"}`, so the filter cannot be affected — and is not.

**The effect also reproduces inside a single job**, so it is not a cross-era artifact: at
m128 in job 3094740, matched concurrency (118.8 vs 116.0) on the same node, the operator
spends 574 ms in decode against the scalar's 471 ms for the same ten tokens — 22% more.
Decode accounts for 103 of the 109 ms end-to-end latency difference, and that latency
difference is the whole arm gap: $(118.8/116.0)/(974/865) = 0.910$, against a measured
rows/s ratio of 0.910.

**The earlier `MAX_OUT_MULT` → KV-reservation hypothesis is withdrawn.** GPU KV usage
peaks at 2.7% of a 1,278,032-token cache with **zero preemptions in both eras**, so
`max_tokens` cannot be metering admission — there is 37× headroom. `MAX_OUT_MULT` does
matter, but through the derived `maxLength`, not through KV.

**Confirmed directly, then fixed.** A CPU microbenchmark against the real
Qwen2.5-7B tokenizer priced the mechanism: an open string costs 14.6 µs of mask work per
decoded token, a `maxLength` element 210–494 µs — **14–27×**, and *flat in the bound's
value* (64 costs what 1024 costs), so both job families were paying the same tax. It also
predicted the GPU-side saving to within ~10%: ~4.5 generated tokens/row × ~400 µs ≈ 1.8
ms/row, against 1.71 ms/row measured at slim R=8.

The bound was removed in `2565fe75` and the corpus re-measured (jobs 3103519, 3103521,
3103523, 3112334). Generation is bit-identical — `prompt_tok` and `gen_tok` unchanged on
all nine cross-system flock arms — F1 moved ≤0.003, the scalar control held to ±0.5%
across five R=32 cells, and operator throughput rose 8.4% to 101%. The cross-era
confound this item describes is therefore closed at the source rather than merely
documented.

**How problematic.** *Was contained, now closed.* While the bound was in place every
reported extract throughput understated the operator, so the error ran against us and no
cited number was ever flattered by it. Three schema eras now exist and must not be mixed
on the throughput axis: XML/unbounded (to 2026-07-15), JSON/bounded (to 2026-08-17), and
JSON/unbounded (2026-08-18 on). Only the last is cited. Quality is comparable across all
three, since generation is bit-identical.

**Action taken.** `max_out_mult` is now recorded in the extract driver's
`WriteResultJson` (`cbf13ff8`), the bound is gone (`2565fe75`), and a unit test
(`SemExtract.ResponseFormat_PerRowStringIsUnbounded`) fails if it returns. The remaining
follow-up is the enum of two labels, which §5 of `evaluation.md` wants for quality; it is
also 16× cheaper than an open string on the same benchmark, so it would help both axes
again.

### 1.2 The extract parity cell was cross-node — RESOLVED

**What it was.** The 128-morsel cell needs 262,144 rows, so the morning campaign gave each
arm its own allocation (operator nid006051, scalar nid006588) to fit a backfill window.
With ±10% cross-node variance, the resulting 0.93× could not be distinguished from parity.

**What was done.** Job **3094740** re-ran the cell with `ARMS="on off"`, both arms in one
allocation on nid006103, cold fleet per arm. The two measurements agree closely:

| arm | in-job (3094740) | cross-node (0ff85915) | spread |
|---|---|---|---|
| operator | 121.986 rows/s | 122.652 | 0.5% |
| scalar | 134.057 rows/s | 132.533 | 1.2% |
| **ratio** | **0.910×** | 0.926× | — |

Node placement never mattered here: ≤1.2% between nodes against a 9–10% arm gap. So the
cell measured a **real direction** rather than noise — and 1.1 then found its cause. Both
cross-node results stay in git history at `0ff85915`.

**Then the cause was removed, and the cell converged.** With the length bound gone (1.1),
job **3112334** re-measured the same cell in one allocation:

| measurement | operator | scalar | ratio |
|---|---|---|---|
| bounded, cross-node (`0ff85915`) | 122.652 | 132.533 | 0.926× |
| bounded, in-job (3094740) | 121.986 | 134.057 | 0.910× |
| **unbounded, in-job (3112334)** | **132.309** | **129.061** | **1.025×** |

**State this as parity, not as an operator lead.** The ratio crossed 1.0 on a +8.4%
operator gain *and* a −3.7% scalar move, and the scalar cannot be affected by a change to
`PhysicalSemExtract::BuildResponseFormat`. Its three measurements of this cell read
132.5 / 134.1 / 129.1, mean 131.9 against the operator's 132.3 — so the defensible claim
is **parity at equal concurrency (≈1.00 ± 0.04)**, up from 0.91. That is exactly what the
chapter argues, now without the schema caveat, and the filter's in-job heroic cell shows
the same convergence independently (140.7 vs 138.8, 1.4% apart).

**Status.** Closed. `evaluation.tex` carries the unbounded in-job numbers; the two
superseded bounded measurements are documented in the directory's `INPUT_PROVENANCE.md`.

### 1.3 Cross-era comparisons are safe for quality, unsafe for throughput

**What.** 1.1 shows a *throughput* shift between eras that is not the tuple encoding but a
change in the extract operator's guided-decoding schema. A cross-era throughput statement
for extract therefore compares two different output contracts, and the confound is now
named and one-directional rather than unquantified.

**Where this bites.** `model_scale.md` §2 ("prompt-format sensitivity falls off with model
size") compares XML-era against JSON-era precision/recall per model. That comparison is
**safe**: generation is bit-identical across the schema eras (`prompt_tok` and `gen_tok`
unchanged on all nine flock arms), so a mask-cost difference changes when tokens are
produced, not which. The same is true of the §0 reconciliation in the changelog.

**The one live instance is now closed.** `cross_system_sem_extract_llama` was re-measured
unbounded in job **3116950** (all 11 arms, n=3), so both families share the schema era.
The Llama operator arms gained 17–92%, and `flock_scalar` gained **0.0%** (40.4 → 40.4
rows/s, F1 bit-identical across all six reps in both eras) — which is what bounds the
node component of that cross-era comparison at zero for the flock client. LOTUS and
Palimpzest drifted +16.5% and +3.8% between sessions; both are CPU-bound Python clients
that never enter flock code, and the flat scalar rules the drift out as a flock-path
effect. Ratios are in-job and confound-free; absolute rows/s still must not be compared
across model families, which was always true and is not an era issue.

**How problematic.** *Resolved.* The rule stands for the record: cross-era **quality**
comparisons are sound because generation is bit-identical, cross-era **throughput**
comparisons are not. No table in the corpus now makes one.

### 1.4 The Llama full-prompt arm cannot carry "best engine efficiency at best quality"

**What.** On Llama the full prompt passes 1994/2000 rows at F1 0.865, against an
always-true baseline of 0.853. The engine ratio computed on that arm (1.82× LOTUS) is a
valid *engine* measurement at prompt parity, but the arm has essentially no discriminating
power.

**How problematic.** *Wording only, already handled.* `model_scale.md` §7 states the
engine claim as holding "at prompt parity" and puts the quality crown on a slim arm. The
risk is a reader carrying the Qwen phrasing ("best efficiency at the best quality") across
to the Llama section, where it would be false.

---

## Tier 2 — known limits, correctly stated, worth re-checking before submission

### 2.1 Cells below three repetitions

| dataset | cells with n<3 | which |
|---|---|---|
| cross-system (all four) | 0 | — |
| `dp_scaling_json_llama` | 0 | — |
| `tp_scaling_json` | 4 of 15 | heroic scalar at TP=1 and TP=2, both arms (n=1) |
| `dp_scaling_json` | 8 of 44 | 32B/72B/7B frontier cells (n=2); multinode verdict pass (n=1) |
| `sem_filter_ab_json` | 10 of 14 | thread-sweep cells (n=1 each; five cells per arm act as replication) |
| `sem_extract_ab_json` | 17 of 18 | thread sweep n=1 per cell; morsel ladder n=1 per cell except m128, measured in three independent sessions |

**How problematic.** *Low to moderate, unevenly.* The thread sweeps are fine: five
independent cold-fleet cells per arm give a spread (operator ±2%, scalar ±0.5%) that a
three-rep single cell would not improve on. The frontier cells at n=2 are acceptable given
effect sizes of 1.3–1.5×. The one that genuinely thins is **the heroic scalar at TP=1/TP=2
(n=1)** — which is where the filter's parity claim lives. The extract morsel ladder is n=1
per cell, but its load-bearing cell (m128) now has three independent sessions, and the
three small cells corroborate each other: the operator sits at 130.9/130.9/129.0 across
layouts, a 1.5% spread that is itself the flatness claim. The R=32 thread sweep is the
strongest control in the corpus — five independent cold-fleet scalar cells within ±0.5%
(43.0–43.1) while the operator doubled beside them.

### 2.2 DP per-GPU efficiency exceeds unity (1.07)

The N=1 base at cap 128 does 136.7 rows/s while the same GPU inside a 4-endpoint fleet does
~146, so the base is marginally under-driven and the ratio overshoots. *Low* — stated in
`scale_out.md` as "no measurable scaling loss" rather than superlinearity, which is the
honest reading. A cleaner fix would be a cap sweep at N=1 to find its own knee, ~20 min.

### 2.3 1×TP4 is bimodal in latency

Queue time across reps: 90 / 242 / 74 ms, with running-set size and TTFT moving together.
*Moderate for the mechanism panel, zero for the throughput claim* — the ordering
4×TP1 > 2×TP2 > 1×TP4 is monotone in every rep. Both `scale_out.md` and the corrections
note require the range rather than a point value.

### 2.4 Sampling protocol is asymmetric by design

flock is pinned to temperature 0; LOTUS and Palimpzest run their runners' defaults, on the
principle that each system is measured as its authors ship it. *Low, given the framing.*
A Palimpzest greedy control was run on Llama during the campaign and is not reported: under
a default-configuration protocol, what a peer would do at a temperature it does not ship is
outside the comparison. The artefacts remain under
`cross_system_sem_filter_llama_pz_{greedy,temp06}/` as exploratory data.

The residual exposure is that the protocol is stated rather than evidenced — a reader takes
on trust that the defaults were not chosen to flatter the comparison. They were not (the
Palimpzest default is the *faster* of the two configurations), and one clause could say so
without reintroducing the numbers.

### 2.5 The cross-harness gap is unexplained

The sembench harness reads faster than the standalone driver on short runs — slim R=1 at
373.6 vs 329.8 rows/s (+13%) — while full R=1 agrees to 2.5%. It persists across a fresh
single session, so it is a fixed per-query difference that matters proportionally more on a
5-second run. *Low, because it is contained*: the rule "compare rows/s within a section,
never across §2 and §3" is stated. But it is an unexplained systematic, and a reviewer
comparing the two tables will notice.

### 2.6 The cross-family mechanism comparison spans two harnesses

The batch-position diagnostic uses Qwen dumps from the standalone sweep and Llama dumps
from the sembench harness. *Low* — the ramp is a within-run positional statistic and both
render byte-identical slim prompts — but it is a provenance asymmetry in a headline
mechanism result. The Llama side is also n=1.

---

## Tier 3 — provenance and instrumentation

- **`sem_filter_ab_json` has no vLLM boot log**, so its engine version is inferred from the
  run window rather than verified. Every other directory is confirmed at 0.22.0. *Very low*
  — the log is still on the cluster.
- **Extract off-vocabulary counts come from a different run than the F1 they explain.** The
  "7 rows in 1904 at R=8" figure is from the Aug-06 diag dumps; the reported F1 is from the
  Aug-08 three-rep run, which has no dumps. *Low* — the mechanism is unchanged and the
  counts are illustrative — but the two are cited as if from one run.
- **`sem_extract_ab_json/morsel_ab_summary.csv` has a non-unique key**: the thread-sweep
  `t8` cell and the morsel `m1_t8` cell collide on the synthesised `config` string. Rows are
  individually correct; any aggregation must key on `(config, arm, rep, R)`. *Low but
  sharp* — it is exactly the kind of thing that silently averages two different experiments.
- **The scalar-on-fleet `n=1` cell was never run**, so the claim "18.6 rows/s whether
  pointed at one endpoint or four" cannot be made. What survives is the dispatch evidence
  (100% of requests on ep0). *Low* — deliberately descoped.
- **No Llama positional diagnostic for extract**, and no KV-stress or multi-node data for
  Llama. *None* — deliberately out of scope; the cross-family section claims only what it
  measured.

---

## Verdict

**Nothing must be re-run for correctness.** No headline claim rests on data that is wrong.
Both Tier-1 items that were open at the first pass are now closed at the source rather
than documented around.

**Two jobs were run, and both paid off.** Job 3094740 re-measured extract m128 with both
arms in one allocation, closing 1.2 and supplying the decode decomposition that diagnosed
1.1. The 2026-08-18 campaign (3103519, 3103521, 3103523, 3112334) then re-measured the
extract corpus with the length bound removed, which lifted operator throughput 8.4–101%
at bit-identical generation and turned the m128 cell from 0.910× into parity.

**One job is in flight**, and it is the last open item: the Llama extract cross-system
re-run, which removes the schema-era split described in 1.3. Until it lands, no
cross-family extract throughput comparison may be made. It should also restore
off-vocabulary counts for at least one family, since it runs with `FLOCK_VERDICT_DUMP=1`.

(A Palimpzest greedy control on Qwen was considered and dropped: under a
default-configuration protocol the peers' behaviour at a non-default temperature is not
part of the comparison.)

**Two things are missing rather than wrong**, and each is a one-line judgement call: the
scalar-on-fleet `n=1` cell (descoped), and verdict dumps for the unbounded-era extract
run — so the per-R off-vocabulary counts in `evaluation.md` §5 are quoted from the
bounded era. They carry over because generation is bit-identical, and F1 itself needs no
dumps: the gold-label-set fix lives in the harness scorer
(`sembench/src/scenario/movie/evaluation/evaluate.py:129`), so `f1_score` in each run's
JSON is already the corrected macro-F1.

**One instrumentation fix pays for itself**: recording every knob that affects the request
shape in the result JSON. `tuple_format` was added mid-project after exactly this class of
confusion, `max_out_mult` after 1.1 (commit `cbf13ff8`). The broader lesson from 1.1 is
that the knob which moved was not a driver flag at all but a *schema derived from* one, so
what deserves logging is the rendered request shape, not the inputs to it. That is still
not logged, and it is the cheapest remaining hardening.

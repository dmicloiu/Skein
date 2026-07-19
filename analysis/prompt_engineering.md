# flock Semantic Operators — Prompt Construction, the tok/s Question, and Accuracy Tuning

Companion to `evaluation.md`. Scope: (Q1) how flock builds prompts (full vs slim,
filter vs extract), (Q2) why R=1 "tok/s" is higher than R=32 and what metric to
lead with, (Q3) ranked prompt tweaks targeting the positional under-detection,
(Q4/Q5) the sweep + cross-system plan. Q1–Q3 are complete; Q4/Q5 pending runs.

All measured numbers below are from the existing
`figures/data/cross_system_sem_filter/` /metrics snapshots (movie sf_2000,
Qwen2.5-7B-Instruct, one GH200, greedy + xgrammar; identical across reps because
prompts are deterministic).

---

## Q1 — Prompt construction

### The full path (`META_PROMPT`)

Every scalar/operator prompt is rendered by `PromptManager::Render`
(`src/include/flock/prompt_manager/prompt_manager.hpp`), which takes the static
`META_PROMPT` (`src/include/flock/prompt_manager/repository.hpp`) and replaces
four placeholders:

| section | filter (`ScalarFunctionType::FILTER`) | extract (`llm_complete`, `COMPLETE`) |
|---|---|---|
| `{{INSTRUCTIONS}}` | `INSTRUCTIONS::SCALAR_FUNCTION` — same text for both ("treat each row independently … return only the relevant answer") | identical |
| `{{RESPONSE_FORMAT}}` | `RESPONSE_FORMAT::FILTER`: "Return **'true'** if the row meets the criteria, and **'false' otherwise**. … no row is skipped." | `RESPONSE_FORMAT::COMPLETE`: "respond directly to the user's prompt … concise, relevant, based solely on the respective row." |
| `{{USER_PROMPT}}` | the query's prompt string | same mechanism |
| `{{TUPLES}}` | `ConstructInputTuples` (below) | same mechanism |

So **filter vs extract differ only in the `{{RESPONSE_FORMAT}}` paragraph and in
the guided-decoding schema**, not in structure. The guided schema (openai
adapter, `src/model_manager/providers/adapters/openai.cpp`; operator overrides
in `semantic_filter.cpp` / `semantic_extract.cpp`) is always
`{"items": [<element> × R]}` with `minItems = maxItems = R`; the element is
`boolean` for filter (`OutputType::BOOL`), `string` for extract
(`OutputType::STRING`; the operator adds a `maxLength` scaled to
`max_output_tokens / R`).

The 7 sections of `META_PROMPT`, in order: `# System Setup` (FlockMTL identity +
image/text framing), `## Table Context`, `## Processing Instructions`,
`## User's Task` ({{USER_PROMPT}}), `## Table Data` ({{TUPLES}}),
`## Instructions` ({{INSTRUCTIONS}} + 3 fixed bullets incl. "Encourage a
step-by-step reasoning process"), `## Output Format` ({{RESPONSE_FORMAT}}).
Most of the fixed text is image/audio/multi-modal boilerplate irrelevant to a
text-only benchmark, and the "step-by-step reasoning" bullet contradicts the
guided boolean schema (the model cannot emit reasoning).

### Tuple encoding (`ConstructInputTuples`, per-model `tuple_format`)

Always prefixed by
`- The Number of Tuples to Generate Responses for: <N>` then:

- **JSON** (what sembench configures: `tuple_format: "json"`): **columnar** —
  one object whose keys are column names and whose values are *arrays over
  rows*: `{"review": ["row1 text", "row2 text", …]}`, `dump(4)`-indented. No
  per-row delimiter, no row index; row i exists only as "the i-th element of
  each array".
- **XML**: row-major — `<header><column>review</column></header>` then one
  `<row><column>…</column></row>` line per row.
- **Markdown**: row-major table, one line per row.

The columnar JSON layout is directly implicated in the positional collapse: the
i-th verdict must be aligned with the i-th array element with no anchor text
(see Q3).

### Per-row token cost (measured, full prompt, filter, JSON tuples)

From the /metrics deltas: R=1 sends 995,544 prompt tokens for 2000 rows
(497.8 tok/row); R=32 sends 100,714 over 63 requests. Solving
`63·T + 2000·r = 100,714`, `T + r = 497.8` gives **template T ≈ 462 tok, review
row r ≈ 36 tok**. So the full template is ~13× the payload it wraps.

### The slim path (filter only, `FLOCK_SEM_PROMPT=slim`)

`PhysicalSemFilter::RenderPrompt` (`src/functions/operator/semantic_filter.cpp`)
replaces the whole `META_PROMPT` with:

```
For each row in the table below, decide whether it satisfies the criterion, judging every row independently on its own merits.
Criterion: <user prompt>

<ConstructInputTuples output — IDENTICAL bytes to the full path>

Return a JSON object {"items": [...]} with one boolean per row, in row order.
```

Kept: the entire tuple block (byte parity on rows), the row-independence
instruction, an output-shape statement (needed because slim drops
`RESPONSE_FORMAT::FILTER`). Dropped: all 7 META_PROMPT sections — the
system-identity, image/audio boilerplate, the duplicated per-row instructions,
and the step-by-step bullet. Head ≈ 112 tok vs 462 → **148 vs 498 tok/row at
R=1** (matches evaluation.md §3c). The `id`/`id_reason` schema modes append
their "Output (structured)" paragraph on top of either head and prepend a
`row_id` column to the tuples.

**Gap: there is no slim path for extract.** `PhysicalSemExtract::RenderPrompt`
(`src/functions/operator/semantic_extract.cpp:151`) unconditionally renders the
full `META_PROMPT`. A Q5 extract run "with the winning slim prompt" requires
mirroring the filter's slim gate there (~10 lines, same env knob).

### Rendered examples (2 real sf_2000 reviews, JSON tuples)

**full-filter** (Q101 prompt; ~560 tok):

```
# System Setup
You are **FlockMTL**, a semantic analysis tool for DBMS that can process both **text and image-derived data**.
Your task is to reason over a structured dataset where **some columns originate from text and others come from external sources** like images or separate dictionaries.

## Table Context
- The section labeled **"Table Data"** includes all rows (rows).
- Each row may contain standard fields, extra textual columns (converted from images or separated text), and image-related columns (e.g., image references or external attachments).
- **Treat all these columns as part of the same table context.**

## Processing Instructions
1. Interpret the user’s prompt precisely for each row.
2. Consider **every column**, including those derived from external content or images.
3. If the prompt involves images, **reason about them in the context of the row’s other data**.

## User’s Task
**User Prompt**:
```
The following movie review is clearly positive.
```

## Table Data
```
- The Number of Tuples to Generate Responses for: 2

{
    "review": [
        "Timed to be just long enough for most youngsters' brief attention spans -- and it's packed with plenty of interesting activity, both on land and under the water.",
        "It doesn't matter if a movie costs 300 million or only 300 dollars; good is good and bad is bad, and Bloodmask: The Possession of Nicole Lameroux is just plain bad."
    ]
}
```
*Some columns may be embedded as text; others may reference external images—treat them all equally.*

## Instructions
```
- Treat each row independently as if it were a standalone record.
- Answer the user prompt specifically for that row, without referencing other rows.
- Do not include extra formatting or explanations—return only the relevant answer.
- Ensure the output is concise, meaningful, and context-aware.
```
- Emphasize that external columns must be merged into the logical row.
- Clarify how to balance reasoning across different column types.
- Encourage a **step-by-step reasoning** process where appropriate.

## Output Format
```
For each row in the provided table, determine whether it satisfies the user's prompt. Return 'true' if the row meets the criteria, and 'false' otherwise. Ensure that each row is evaluated independently and that no row is skipped.
```
Ensure your results follow this format exactly, with **no extra commentary**.
```

**slim-filter** (~185 tok):

```
For each row in the table below, decide whether it satisfies the criterion, judging every row independently on its own merits.
Criterion: The following movie review is clearly positive.

- The Number of Tuples to Generate Responses for: 2

{
    "review": [
        "Timed to be just long enough for most youngsters' brief attention spans -- and it's packed with plenty of interesting activity, both on land and under the water.",
        "It doesn't matter if a movie costs 300 million or only 300 dollars; good is good and bad is bad, and Bloodmask: The Possession of Nicole Lameroux is just plain bad."
    ]
}


Return a JSON object {"items": [...]} with one boolean per row, in row order.
```

**full-extract** (Q103 prompt): identical to full-filter except `{{USER_PROMPT}}`
= "Classify the sentiment of this movie review as exactly one of POSITIVE or
NEGATIVE. Answer with only the single word POSITIVE or NEGATIVE." and the
`## Output Format` block carries `RESPONSE_FORMAT::COMPLETE` ("respond directly
to the user's prompt … no row is omitted …"). Guided schema element becomes a
bounded string. **slim-extract: does not exist yet** (see gap above); the
natural rendering mirrors slim-filter with "produce the requested value for each
row" framing + "one string per row, in row order."

(Render mock used for these examples:
scratchpad `render_prompts.py`, faithful to the C++ builders.)

---

## Q2 — The tok/s puzzle, resolved

**Question:** how can operator R=1 have ~10× the prompt tokens of R=32
(995,544 vs 100,714) — and is its 73.7k tok/s real work?

**Part 1 — the 10× is the un-amortized template.** Prompt tokens per arm =
`requests × template + rows × row_payload`:

- R=1: 2000 requests × (462 + 36) ≈ 995.5k — **93% of it (924k) is 2000
  repetitions of the same 462-token template.**
- R=32: 63 requests × 462 + 2000 × 36 ≈ 100.7k — template is 29% of the total.

So per-row cost falls 498 → 50 tok as the shared head amortizes; tok/s =
rows/s × tok/row, so R=1's *lower* rows/s (146 vs 642) still yields *higher*
tok/s because each row drags 10× the tokens.

**Part 2 — over half of R=1's "tokens/s" was never computed.** vLLM's
`vllm:prompt_tokens_total` counts every prompt token *including prefix-cache
hits* — in our snapshots `prompt_tokens_total` delta ==
`prefix_cache_queries_total` delta **exactly** for every arm, and
`prefix_cache_hits_total` shows what was served from cache:

| arm | prompt tok | cache hits | hit rate | gen tok | reported tok/s | **computed tok/s** | gen tok/s |
|---|---|---|---|---|---|---|---|
| flock op R=1 | 995,544 | 550,160 | **55.3%** | 16,577 | 73,716 | **33,646** | 1,207 |
| flock op R=32 | 100,714 | 16,096 | 16.0% | 4,441 | 33,812 | **28,636** | 1,428 |
| LOTUS | 257,520 | 138,608 | 53.8% | 35,997 | 37,873 | **19,988** | 4,645 |
| Palimpzest | 836,282 | 715,040 | **85.5%** | 130,704 | 66,414 | **17,304** | 8,977 |

(computed tok/s = (prompt − hits + gen) / execution time; rep1, reps 2–3
byte-identical.)

At R=1 the hits are 550,160/2000 ≈ **275 tok/row — the block-aligned (16-token
blocks) shared literal prefix up to the first review byte**. The template
*tail* (instructions + output format after the tuples, ~190 tok) sits after the
varying review text and is therefore re-**computed** for all 2000 rows — cache
can't help it. That is pure overhead a leaner prompt eliminates.

**Consequences:**

1. The R=1 vs R=32 "engine efficiency" gap largely evaporates on computed
   tokens: 33.6k vs 28.6k (1.17×), not 73.7k vs 33.8k (2.2×). The engine does
   similar real work per second at both R; R=1's headline was inflated by
   re-sent, half-cached template bytes.
2. Cross-system, raw tok/s is actively misleading: Palimpzest's 66.4k tok/s is
   85.5% cache hits (it repeats a ~480-token template per row) — computed it is
   the *slowest* engine (17.3k). **evaluation.md §2's "73.7k tok/s, ~1.9×
   LOTUS" should be restated: on computed tokens flock op R=1 is 33.6k vs LOTUS
   20.0k = 1.68×** — still the best engine at the best F1, but the honest
   number.
3. Raw tok/s *rewards* fat, un-amortized prompts: slimming lowers tok/s while
   raising rows/s at near-constant F1 — the definition of a metric pointing the
   wrong way.

**Metric recommendation:** lead the evaluation with **rows/s at matched F1**
(the Pareto plane already used in §3) — it is the metric a user experiences and
it is prompt-honest. As the engine-level diagnostic, report **computed tok/s
(non-cached prefill + generation per second)** and gen-tok/s; retire raw
(prompt+gen)/time tok/s except as a footnote. Generation tokens stay in the
computed total because they are never cached and each costs a full forward pass
— dropping them would understate decode-heavy systems (LOTUS 36k, Palimpzest
131k gen tokens vs flock's 4–17k boolean arrays); cache hits are the only
tokens counted-but-never-processed.

**Implemented:** `summarize_cross_system.py` now emits `cache_hit_tok`,
`cache_hit_pct`, `computed_tok`, `computed_tok_s`, and `pareto_tok_f1` is marked
on computed_tok_s (raw `tok_s` kept for reference); `summarize_morsel_ab.py`
and `diagnose_rsweep.py` gained `computed_tok_s` (console `ctok/s`). Both
`cross_system_sem_filter/` and `cross_system_sem_extract/` summary CSVs
regenerated.

**Extract (Q103) under the computed metric** — the correction *strengthens*
flock's engine lead there, because the competitors' raw tok/s was even more
cache-inflated (LOTUS 72.7% hits, Palimpzest 91.3%):

| arm | reported tok/s | computed tok/s | cache hit | F1 |
|---|---|---|---|---|
| flock op R=1 | 74,293 | **33,462** | 56.0% | **0.807** |
| flock op R=32 | 29,487 | 24,925 | 16.7% | 0.314 |
| LOTUS | 47,436 | **15,111** | 72.7% | 0.565 |
| Palimpzest | 63,029 | 14,787 | 91.3% | 0.577 |

Raw tok/s said flock op R=1 leads LOTUS 1.57×; computed says **2.21×** — at
+0.24 F1. flock op R=1 is the sole point on the extract computed-tok/s × F1
frontier. (Consistency check: flock's computed tok/s is ~33.5k on both filter
and extract at R=1 — same engine, same real work rate — while the raw metric
scattered with each system's prompt fat.)

---

## Q3 — Candidate prompt tweaks for the positional under-detection

Constraint: recover back-of-prompt recall **without adding decode tokens**
(id_reason works but is decode-bound and Pareto-dominated, evaluation.md §3b).
So the levers are input-side anchoring, instruction geometry, and framing.
Established mechanism: miss-rate ramps 0.42→0.80 front→back at R=32 (bool);
anchoring (`id`) flattens the ramp; the diagnosis (see
`~/flock_recall_collapse_analysis.md`) is bare boolean output + lost-in-middle
attention decay + the "'false' otherwise" default ⇒ under-detection of late
rows.

Ranked candidates (screen at R∈{8,32} where the collapse is visible; all on the
slim base):

| # | variant | change | hypothesis | token cost |
|---|---|---|---|---|
| P1 | **row-major tuples with per-row index** | replace columnar JSON with `[{"id": 1, "review": "…"}, …]` (or XML rows — existing `tuple_format` knob, zero code) | the columnar layout gives the model *no anchor text* linking array position i to output slot i; input-side ids supply the anchor that made `id` schema flatten the ramp, at zero decode cost | +4–6 in/row, 0 out |
| P2 | **instruction sandwich** | repeat the one-line criterion + "exactly N booleans" AFTER the table (slim's head only precedes it) | lost-in-middle: back rows are far from the front-loaded criterion; a trailing restatement puts the task adjacent to the rows most missed | +~30 in/prompt, 0 out |
| P3 | **symmetric verdict framing** | drop the "'false' otherwise" default; "output true if the review is clearly positive, false if it is not; do not let earlier rows influence later ones" | the asymmetric default makes "attention ran out" collapse to false ⇒ under-detection; a symmetric instruction removes the cheap fallback | 0 |
| P4 | **explicit count + correspondence** | "The table has N rows. Return exactly N booleans; the i-th boolean answers row i." (replaces the odd "Number of Tuples to Generate Responses for" line) | makes the alignment contract explicit rather than implied by schema | ~0 |
| P5 | **2-row worked example** | tiny fixed example table + `{"items": [true, false]}` before the real table | batch-prompting literature (Cheng et al. '23) shows a worked example teaches the row→slot convention; cheap and amortized | +~60 in/prompt, 0 out |
| P6 | **within-prompt chunk headers** | at R≥16, group rows under "Rows 1–8", "Rows 9–16", … separators | periodic re-anchoring resets attention decay across a long tuple block | +2/8 rows, 0 out |
| P7 | **T/F char-string output** | `{"verdicts": "TFFT…"}` — one char per row, regex-guided length N | *fewer* decode tokens than a boolean array; each char is positionally rigid under guided decoding | −out (!), exploratory: tokenizer merges may fight the alignment |
| P8 | **shuffle probe** (control) | randomize row order per batch, unshuffle verdicts | mechanism check, not a fix: converts positional bias into unbiased noise; F1 predicted ≈ unchanged, per-position miss-rate flattens | 0 |

Priority argument: P1 attacks the same mechanism that the only successful
schema fix (`id`) attacked, but from the input side where tokens are ~35×
cheaper than decode tokens at these throughputs; P2/P3 attack the two remaining
diagnosed causes directly. P1+P2+P3 compose naturally into one "slim-v2" prompt
that is generic (nothing sf_2000-specific in any of them).

Expected end-state: slim-v2 ≈ slim at R=1 (no positional problem to fix), and
recovers a chunk of the R∈{4,16} F1 drop, moving the knee of the
throughput-quality frontier right.

---

## Q4 — Sweep design (pending approval / runs)

- **Split:** hash-split sf_2000 by `reviewId` parity into dev (~1000) / test
  (~1000). Tune on dev only; report the winner on test. Candidates are generic
  prompt-engineering moves (nothing references sf_2000 labels or content), which
  is the real overfitting guard.
- **Screen (dev, 1 rep):** P1–P6 individually vs slim baseline at R∈{8,32};
  keep anything ≥ +0.02 F1 at ≤5% rows/s cost; P8 once as the mechanism
  control. Use `FLOCK_VERDICT_DUMP` on an untimed pass for per-position
  miss-rates.
- **Combine (dev, 1 rep):** top-3 singles → combined slim-v2 (expect P1+P2+P3);
  ablate one-out if the combination underperforms the best single.
- **Confirm (test, 3 reps):** winner + slim baseline + full baseline at
  R∈{1,2,4,8,16,32}: F1 + rows/s + computed tok/s, median + min/max.
- **Operating point:** re-derive the Pareto set; expected slim-v2 at R=1 for
  the balanced point, higher-R slim-v2 for max throughput.
- Infra: Clariden SLURM per `analysis/slurm/`, cold fleet, NO_PROXY=localhost,
  same vLLM config as the cross-system runs.

## Q5 — Cross-system with the winning prompt (pending)

- Filter first: flock operator (winner, R=1 and best-throughput R) vs LOTUS vs
  Palimpzest, ≥3 reps, same harness as §2; report rows/s, F1, computed tok/s.
- Extract once the slim gate exists in `PhysicalSemExtract::RenderPrompt`
  (blocked on the sem_extract track landing; `cross_system_sem_extract/` data
  dir already exists).
- **Fairness protocol:** publish each system's exact rendered prompt (appendix);
  present two flock rows — *engine win* (flock with default full prompt vs
  LOTUS default, both un-tuned) and *engine+prompt win* (slim-v2) — so the
  engine advantage is never conflated with prompt tuning. Matched prompt
  economy: slim (~148 tok/row) ≈ LOTUS (~147 tok/row) already; keep slim-v2
  within ~±15% of that budget.

## Deliverable plot (pending Q4/Q5 data)

rows/s × F1 scatter with Pareto frontier: all prompt configs (full/slim/slim-v2
× R) + LOTUS + Palimpzest, winner highlighted, min–max whiskers over reps;
`plot_*.py` conventions, data under `figures/data/`.

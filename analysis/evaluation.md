# flock Semantic Operators — Evaluation

Four directions, all on the **movie** scenario, Qwen2.5-7B-Instruct on GH200s. flock
decodes greedily (temperature 0) with xgrammar-guided JSON. All LLM-side numbers are
vLLM `/metrics` deltas, uniform across systems.

**Serving stack.** Every measurement in this document is served by **vLLM v0.22.0 (V1
engine)**, verified identical across all 970 endpoint boots in the consolidated data.
The engine configuration is held fixed everywhere and is part of what the numbers mean:

    --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching
    --gpu-memory-utilization 0.90
    --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'

Three of those are load-bearing rather than incidental. `--enable-prefix-caching` is what
makes computed tok/s meaningful at all, since the metric subtracts `prefix_cache_hits`
from the prompt total. `disable_any_whitespace` forces compact guided JSON: without it,
greedy decoding stalls emitting whitespace between array elements and truncates the
response. And `--max-model-len 16384` bounds the batched prompt, which is why the R
sweep stops at 32. A fresh endpoint is started for every measurement (cold prefix cache),
so no cell inherits another's cached prefixes.

**Sampling protocol.** Each system runs its own default sampling: flock is pinned to
temperature 0, while LOTUS and Palimpzest use their runners' defaults, since forcing a
temperature on them would mean benchmarking a configuration their authors do not ship.
Reported competitor throughput is therefore default-sampling throughput, and is labelled
as such rather than as greedy.

**Metric standard.** The primary number is **rows/s at a stated F1** — task
throughput at a stated quality, which is what a user experiences and what is honest
about prompt size. The engine-level diagnostic is **computed tok/s** =
`(prompt_tokens − prefix_cache_hits + generation_tokens) / time`, i.e. tokens the GPU
actually processed; raw `(prompt+gen)/time` counts cache hits, rewards fat
un-amortised prompts, and is not reported. Quality is precision / recall / F1 against
`scoreSentiment`. Medians over ≥3 reps with [min–max]; cross-node variance is ~±10%.

**Prompt configuration.** Every flock run renders the **JSON tuple encoding**
(`tuple_format: json`), pinned in the benchmark drivers and used by the cross-system
harness. Two prompt heads appear: the shipped 7-section `META_PROMPT` ("full") and the
lean text-only head ("slim"), which is batch-adaptive — a plain single-row form at
R=1, and a row-major indexed form with an explicit count contract at R>1. R denotes
rows packed per request (`semantic_batch_size`).

**The two levers are reported separately.** §1 and §4 isolate the *engine* lever
(async operator vs scalar `llm_filter`) at byte-identical prompts on both arms. §3
isolates the *prompt* lever (slimming and batching) on one engine. §2 reports both:
a full-prompt arm, where flock and the competitors are equally un-tuned, and slim
arms that add the prompt win on top. No figure or claim folds the two together.

---

## 1. Async operator vs scalar `llm_filter` (intra-flock)

**Question:** does the async `PhysicalSemFilter` decouple LLM concurrency from
DuckDB's thread/morsel count? Scalar concurrency is bound to `min(threads, morsels)`;
the operator's is bound to `in_flight_cap`.

**Setup:** full prompt on both arms (prompt-byte parity), one GH200, cold endpoint per
cell. Row counts differ by cell because the morsel layout is what is being varied:
one morsel = an in-memory table (2,000 rows), 8/16 morsels = an attached on-disk DB
at `ROW_GROUP_SIZE` 2048 (16,384 / 32,768 rows).

**(a) Thread-independence at one morsel** (R=32, 2,000 rows). Five independent
cold-fleet cells per arm; the operator is flat because its concurrency is the cap,
the scalar is flat at 1 because `min(threads, morsels) = 1`:

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| operator rows/s | 633.3 | 645.8 | 635.9 | 619.9 | 624.3 |
| scalar rows/s | 77.1 | 76.5 | 76.4 | 76.7 | 76.6 |

Operator ~633 rows/s (±2% across five boots) vs scalar ~77 (±0.5%) — **8.3×** — with
measured concurrency ~58 vs 1.0. Computed tok/s tracks: 28,500 vs 3,450.

**(b) Giving the scalar morsels closes part of the gap** (R=1, 3 reps):

| cell | rows | operator rows/s | scalar rows/s | op/scalar |
|---|---|---|---|---|
| 1 morsel / 8 threads | 2,000 | 140.0 [137.4–146.9] | 18.8 [18.7–18.9] | **7.4×** |
| 8 morsels / 8 threads | 16,384 | 137.7 [137.6–143.4] | 68.6 [68.0–69.7] | 2.01× |
| 16 morsels / 8 threads | 32,768 | 138.1 [137.9–143.6] | 69.4 [68.6–69.6] | 1.99× |

The operator holds ~138 rows/s regardless of layout; the scalar rises only as its
concurrency does (1 → ~8). The 1-morsel row is the TP=1 cell of §4, which measures
the same configuration at 3 reps.

**(c) The scalar's best case, and why the gap re-opens.** Handed 128 morsels and 128
threads over a 262,144-row table — a configuration that also required a
`MetricsManager` thread-safety fix to run at all — the scalar reaches concurrency
~105 and **reaches parity at TP=1** (140.7 vs the operator's 138.8 rows/s — the scalar
marginally ahead), then falls behind as the ceiling rises: 211.7 vs 227.1 at TP=2, and
275.1 vs 369.1 at TP=4, a **1.34×** operator lead. Equal concurrency buys equal
throughput: the async client has no secret sauce beyond concurrency itself. But ~105 is
a *structural* ceiling (`min(threads, morsels)`, 128 OS threads, an on-disk table),
while the operator's cap is one config knob — so the gap re-opens exactly as the
endpoint's ceiling moves past what ~105 in flight can feed.

**Finding:** the operator delivers full GPU concurrency from *any* thread count; the
scalar only approaches it when handed a thread and morsel count that a normal query
plan will not produce.

---

## 2. Cross-system: flock vs LOTUS vs Palimpzest

**Question:** against one local vLLM, does flock's tighter engine integration win?
Same benchmark criterion, same 2,000 rows, same endpoint, each system at its own
default sampling (see the protocol note above); each drives concurrency its own way. sembench harness, 3 reps, median.

| system (config) | rows/s | **computed tok/s** | tok/row | precision | recall | F1 |
|---|---|---|---|---|---|---|
| flock operator, full, R=1 | 148.4 | **34,284** | 506 | 0.892 | 0.973 | **0.931** |
| flock operator, full, R=32 | 663.2 | 29,534 | 53 | 0.889 | 0.374 | 0.527 |
| flock scalar, R=1 | 18.2 | 4,214 | 506 | 0.895 | 0.974 | 0.933 |
| flock scalar, R=32 | 77.5 | 3,451 | 53 | 0.886 | 0.366 | 0.518 |
| LOTUS | 247.7 | 19,375 | 148 | 0.991 | 0.718 | 0.833 |
| Palimpzest | 129.8 | 16,347 | 483 | 0.985 | 0.792 | 0.878 |
| flock operator, slim, R=1 | 373.6 | 23,497 | 148 | 0.990 | 0.788 | 0.877 |
| flock operator, slim, R=4 | 592.5 | 30,287 | 76 | 0.986 | 0.735 | 0.842 |
| flock operator, slim, R=8 | 678.9 | 33,133 | 61 | 0.986 | 0.722 | 0.834 |

**(a) The engine claim, un-tuned prompts on both sides.** flock's operator on its
*default* full prompt reaches **34.3k computed tok/s at F1 0.931** — the best engine
efficiency at the best quality, **1.77× LOTUS** (19.4k). LOTUS's higher wall-clock
rows/s (248) comes from doing less work: recall 0.72 vs 0.97 at 3.4× fewer tokens.
The scalar arms are 2–8× slower at the same quality, so async concurrency is the
differentiator.

Raw tok/s would have said 2.06×, and would have ranked Palimpzest second-best at 62.8k
when 85.5% of its prompt tokens are prefix-cache hits it never computed — it is in fact
the *slowest* engine at 16.3k. This is why raw tok/s is not reported.

**(b) flock holds the quality crown outright.** At F1 **0.931**, the full-prompt
operator is **+0.098 over LOTUS** (0.833) and **+0.053 over Palimpzest** (0.878) — the
largest quality margin in the comparison, on flock's *default* prompt with no tuning.
That is the quality claim, and it is independent of everything the slim head does.

**(c) The prompt lever buys throughput against that quality.** Slimming trades some of
the margin for rate, and the trade is favourable across the whole range: slim R=4
(593 rows/s, F1 0.842) beats LOTUS on **both** axes — 2.4× the rate at +0.009 F1 — and
the fastest arm that still clears LOTUS's quality, slim R=8 at 679 rows/s, holds F1
0.834 against LOTUS's 0.833 (beyond R=8, §3 shows quality drops below it). Against Palimpzest the slim arms are a throughput trade rather than a
quality one: slim R=1 runs **2.9× faster at parity** (373.6 rows/s at 0.877 vs 129.8 at
0.878), while flock's own full-prompt arm remains 0.053 ahead of Palimpzest on quality
if that is what you need. The two levers stay separable: the engine win holds with
flock's un-tuned prompt, and slimming adds throughput on top of it.

**Operating points.** *Max quality:* full + R=1, F1 0.931 — best of any system here.
*Balanced:* slim + R=1, F1 0.877 at 2.5× that throughput. *Max throughput at LOTUS-or-better
quality:* slim + R=8, 679 rows/s (4.6×) at F1 0.834. §3 maps the space between, including
the higher-R points that trade below LOTUS's quality.

*All twelve arms were measured in one session (3 reps each), so cross-arm rows/s carry
no cross-node variance. An earlier version of this table spanned three sessions; the
single-session re-measurement moved individual arms by up to 5% in both directions and
raised the engine ratio from 1.68× to 1.77×.*

---

## 3. Prompt configuration: slimming and batching

**Question:** what is the best prompting configuration, and how does quality trade
against throughput as rows are packed into one request? Standalone driver, one GH200,
cold endpoint per cell, 2,000 rows, 3 reps, median [min–max]. F1 is row-level.

| R | full rows/s | full F1 | slim rows/s | slim F1 |
|---|---|---|---|---|
| 1 | 144.8 [141–149] | **0.926** | 329.8 [303–333] | **0.872** |
| 2 | 234.9 [233–247] | 0.821 | 488.1 [486–521] | 0.846 |
| 4 | 365.6 [353–387] | 0.770 | 608.1 [571–633] | 0.834 |
| 8 | 524.6 [509–528] | 0.737 | 621.9 [498–628] | 0.825 |
| 16 | 618.9 [610–628] | 0.669 | 649.9 [591–671] | 0.803 |
| 32 | 641.4 [638–654] | 0.506 | 596.9 [553–617] | 0.732 |

**(a) Slimming is the win.** A lean text-only head cuts the prompt from 498 to 141
tokens per row and more than doubles throughput at R=1 (145 → 330 rows/s) for 0.054
F1. The gain is context length, not caching (prefix-cache hit rate barely moves,
54.8% → 59.9%): a shorter context cuts prefill, per-step decode attention, and queue
time together.

**(b) Batching costs quality, and the batch-adaptive head absorbs most of it.** On the
full prompt, F1 falls off a cliff immediately — 0.926 → 0.821 at R=2, 0.506 by R=32.
On the slim head, quality is nearly flat to R=8 (0.872 → 0.825) and still 0.732 at
R=32. The mechanism is **positional**: rows later in a packed prompt are
under-detected, with the miss-rate of gold positives ramping front-to-back
(R=8: 0.285 → 0.420; R=16: 0.403 → 0.584; R=32: 0.465 → 0.842). The explicit count
contract and row indices in the batched slim form flatten most of that ramp.

**(c) Throughput plateaus where the request count meets the cap.** Slim throughput
rises to R≈4 and is then flat within spread — R=4 [571–633], R=8 [498–628],
R=16 [591–671] all overlap — before dipping at R=32. This is an `in_flight_cap`=128
artifact, not a property of batching: at R=16 the query issues 125 requests, at R=32
only 63, which under-fills the pipe. Read the plateau as "R≈4–16", not a peak at 16.

**Recommendation:** run **slim**; default **R=1** for quality-sensitive work, scale R
toward 8 to trade ~0.05 F1 for ~1.9× throughput, and use the full prompt only when the
last 0.05 F1 matters. Because flock controls the batch-size knob, it can sidestep the
batching collapse by choosing low R rather than paying for heavier per-row output.

*Note: this sweep runs the standalone driver and §2 runs the sembench harness, so
absolute rows/s differ by harness overhead (slim R=1: 330 here vs 373 there) while F1
agrees to 0.006. Compare rows/s within a section, not across.*

---

## 4. Scale-out: TP and DP

**Question:** as GPUs are added, does the async operator track the rising ceiling
where the scalar cannot — and is it better to shard one endpoint or replicate several?
Full prompt on every arm, R=1, cold fleet per measurement, 3 reps.

**(a) Vertical (TP): one endpoint sharded across 1/2/4 GPUs.**

| TP | operator rows/s (best cap) | eff | scalar rows/s | op/scalar | F1 |
|---|---|---|---|---|---|
| 1 | 140.0 [137–147] (128) | 1.00 | 18.8 | **7.4×** | 0.927 |
| 2 | 232.6 [228–236] (256) | 1.66 | 25.4 | 9.2× | 0.926 |
| 4 | 394.3 [388–396] (512) | 2.82 | 30.5 | **12.9×** | 0.927 |

The gap widens monotonically: at concurrency 1 the scalar can only harvest TP's
per-request latency gain, which is bounded and decelerating, while the operator
harvests latency *and* the engine's growing concurrent-batch capacity. **The cap must
scale with TP (≈128·TP)**: at TP=4, 302.8 → 338.8 → 394.3 rows/s for cap 128/256/512.
**TP itself is sub-linear for a 7B model** — 2.82× on 4 GPUs, 71% per GPU — the
comm-bound cost of two all-reduces per layer per step. F1 is TP-invariant.

**(b) Horizontal (DP): N independent endpoints behind the router.**

| fleet | cap | rows/s | speedup | per-GPU | F1 |
|---|---|---|---|---|---|
| 1 × TP1 | 128 | 138.9 [138.5–142.4] | 1.00 | 1.00 | 0.926 |
| 2 × TP1 | 256 | 289.1 [288.7–293.7] | 2.08 | 1.04 | 0.926 |
| 4 × TP1 | 512 | 578.0 [561.7–583.2] | **4.16** | **1.04** | 0.926 |
| 4 × TP1 | 128 (under-fed) | 491.6 [485.6–497.2] | 3.54 | 0.89 | 0.926 |

**DP is linear to within noise** — 4.16× on 4 GPUs, and the 4% above unity sits inside
the run-to-run spread, so the honest claim is "no measurable scaling loss". The cap is
the fleet's admission knob with a **knee at ≈128·N**: 491.6 → 544.9 → 578.0 → 586.5 →
585.1 rows/s at cap 128/256/512/1024/2048. Under-feeding costs 15%; past the knee the
surplus buys only latency. F1 is fleet-size-invariant.

**(c) Fixed 4-GPU budget: replicas beat shards.** All three topologies at total cap
512, measured in the same jobs:

| topology | rows/s | vs 1×TP4 |
|---|---|---|
| **4 × TP1** | **578.0** [561.7–583.2] | **1.57×** |
| 2 × TP2 | 451.6 [446.9–453.6] | 1.22× |
| 1 × TP4 | 368.8 [366.1–405.8] | 1.00× |

Monotone, at identical GPU count and identical total concurrency, with equal F1. The
mechanism is admission latency: one scheduler serialises admission no matter how wide
its shards, while four admit in parallel. Sharding also raises variance — 1×TP4 spans
±5% across reps where 4×TP1 holds ±2%. **For a 7B-class model, spend GPUs on
replicas, not shards.**

**The scalar cannot use a fleet at all**, and this is measured rather than asserted:
pointed at the same 4-endpoint fleet it sustains 18.6 rows/s [18.3–18.6] and lands 100%
of its requests on ep0, none on the other three, in every rep. The operator on that
fleet reaches **31×** its rate.

---

## 5. Semantic extract

The extract operator (`llm_complete` → `PhysicalSemExtract`) shows the same engine
result: at R=32 over 2,000 rows, operator 536–558 rows/s vs scalar 43.0–43.1 across
five cold thread cells — **~13.0×**, thread-independent on both arms, every row emitted.
The cross-system session reproduces the ratio independently at **12.5×** (537.9 / 43.0),
agreeing to 4%.

Cross-system, the picture differs from the filter: **flock does not dominate extract
on quality.** All eleven arms in one session, 3 reps (LOTUS n=2, see below).

| arm | rows/s | F1 | computed tok/s | |
|---|---|---|---|---|
| Palimpzest | 82.9 | **0.868** | 14,753 | max quality |
| lotus | 259.8 | **0.845** | 14,754 | |
| flock slim R=8 | **591.9** | **0.836** | 30,576 | best flock point |
| flock slim R=1 | 354.6 | 0.831 | 24,554 | |
| flock slim R=4 | 547.7 | 0.827 | 29,867 | |
| flock slim R=2 | 465.4 | 0.823 | 27,848 | |
| flock full R=1 | 139.1 | 0.807 | **33,240** | max engine efficiency |
| flock slim R=16 | 581.7 | 0.780 | 29,827 | |
| flock slim R=32 | 518.7 | 0.647 | 26,497 | |
| flock full R=32 | 537.9 | 0.628 | 25,132 | |
| flock scalar | 43.0 | 0.617 | 2,009 | |

Palimpzest holds the quality crown and LOTUS beats every flock point on quality; flock's
contribution is throughput — slim R=8 runs **2.28× LOTUS's rate for −0.009 F1**, and the
full head at R=1 turns the engine 2.25× as efficiently as either peer (33,240 computed
tok/s against ~14,750). Three arms sit on the rows/s–F1 frontier: Palimpzest, LOTUS and
flock slim R=8. Batched extract holds quality flat from R=1 to R=8 (0.831 → 0.836) and
degrades only at R=16/32, mirroring the filter.

*LOTUS is n=2 here, and the reason is worth stating precisely because it is not a
measurement failure.* The endpoint served **every request in `lotus_rep2` successfully**:
2,019 chat completions, all HTTP 200, zero 5xx, zero exceptions in the vLLM log. The
2,000 LM calls completed in ~7 s at 269 calls/s, in line with rep1 (7.59 s) and rep3
(7.81 s). What failed was LOTUS's own post-processing: litellm raised an
`InternalServerError` client-side for one response, and `lotus/models/lm.py:214` does
`response.choices[0]` over every element of the batch without checking whether an element
is an error, so the whole 2,000-row query aborted. The harness wrote
`status: failed, row_count: 0, f1_score: 0.0`, and `summarize_cross_system.py` excludes
it — correctly, since averaging that 0.0 in would have dragged LOTUS's F1 to ≈0.56.

So the discarded repetition agrees with the two that survived on timing, and the missing
n is a peer-client robustness gap rather than an unmeasured cell. LOTUS is n=3 in the
other three cross-system tables. Its +5.2% against the previous session brackets session
variance rather than indicating any effect.*

**Why extract can produce an invalid label at all, and filter cannot.** The two
operators guide decoding differently: filter's response schema is
`{"type": "boolean"}`, so the output space is exactly two values and anything else is
unrepresentable; extract's is `{"type": "string"}`, a free string. The prompt asks for
"exactly one of POSITIVE or NEGATIVE", but the grammar permits any string, so on
genuinely ambiguous reviews the model occasionally answers `NEUTRAL` or `MIXED` — 0 rows
at R=1/2, 1 at R=4, 7 at R=8, 8 at R=16, 3 at R=32, out of ~1,900 scored. Those rows are
counted as errors, which is the honest measurement: part of what the task tests is
instruction-following, and a model that names a third label when told to pick one of two
has failed that row. Constraining the schema to an enum of the two labels would close the
gap, and is the natural follow-up; the ceiling on it is small (+0.001 to +0.002 F1 even
if every off-vocabulary row flipped to correct), so it does not change the standing
against LOTUS or Palimpzest.

*(Those per-R counts come from the bounded-schema era, the only extract session with
verdict dumps. They carry over because generation is provably unchanged: `prompt_tok`
and `gen_tok` are bit-identical across the two eras on all nine flock arms. F1 itself is
computed by the harness scorer from the predictions, so the table above needs no dumps.)*

**The schema used to carry a length bound, and it cost 2× throughput.** Until
`2565fe75` the per-row element was `{"type": "string", "maxLength": N}`, with N derived
from the output-token budget. A `maxLength` makes the element a length-counting grammar,
which costs xgrammar **15–25× more mask work per decoded token** than an open string and
is flat in the bound's value, so shrinking N does not help (measured directly against the
Qwen2.5-7B tokenizer). Because `BuildResponseFormat` is operator-only, the scalar never
paid it. Removing the bound left generation bit-identical and F1 within ±0.003 while
lifting every operator arm: the R=32 cells roughly doubled (279–306 → 536–558 rows/s) and
the R=1 morsel cells gained 8–15%. The saving is a near-constant **1.48–1.90 ms/row** on
every operator arm against ≤0.20 ms/row on all three controls, which is why prefill-heavy
arms gain least in percentage terms — relative gain is that constant divided by the arm's
baseline per-row cost, not a function of R. The output ceiling is now the request-level
`max_tokens` alone, with `minItems`/`maxItems` still pinning the answer count.

These F1 values required fixing the harness scorer, which macro-averaged over every
label present in prediction or gold rather than over the gold label set. Guided
decoding emits an off-vocabulary label on a handful of rows (7 in 1904 at R=8); each
added a class with F1 0 and divided the macro by 4 instead of 2, which is what
produced the earlier appearance of a collapse at R≥4. Systems that emit free text
were penalised hardest — LOTUS and Palimpzest gained +0.28 F1 each on re-scoring.

*An earlier version of this table mixed sessions and scorer versions across reps, and
its `flock_op_r32` cell read 530 rows/s for the wrong reason. Single-session
re-measurement returned 285, matching the independent standalone A/B of the same
workload (279–306 rows/s) that the old value never did; after the schema fix both move
together again, to 537.9 here and 536–558 standalone. The harness and the standalone
driver agreeing in both eras is the check that matters — the numerical coincidence
between the discredited 530 and today's 537.9 is exactly that, a coincidence. Timing CV
is ≤3.3% on every flock arm.*

---

*Caveats: all quality numbers are one criterion on one dataset (movie sf_2000, 2,000
rows), so they measure this task, not semantic filtering in general. §1's morsel cells
use 16,384 / 32,768 rows and the heroic-scalar cell 262,144 — the row count is part of
the configuration being varied, not a constant. §3's throughput plateau and §1's
128-morsel cell are both `in_flight_cap`=128 artifacts and are stated as such. §4's
per-TP jobs ran on separate nodes: each op/scalar ratio is in-job and node-consistent,
but cross-TP efficiencies carry the ~±10% cross-node variance. The 3 s scheduler-gauge
samples on 6–14 s runs are qualitative; Little's-law concurrency is the quantitative
saturation check. The generalization to larger models and multi-node fleets lives in
`model_scale.md`; those runs predate the prompt-configuration freeze used here, so
their within-chapter ratios and laws carry over directly while their absolute F1 and
rows/s are not on the same footing as the numbers above.*

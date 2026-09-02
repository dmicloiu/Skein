# flock Semantic Filter — Generalization: Model Scale, Workload Shape, Node Count, Model Family

Companion to `scale_out.md` (the 7B single-node TP/DP study): the same operator,
workload (32,000 rows of `movie/sf_300000`, full prompt, JSON tuple encoding, R=1,
greedy, vLLM v0.22.0), hardware and conventions, with the study's fixed choices swept
one at a time. Four axes, each answering a different "you overfit to your setup" objection:

- **model size** — Qwen2.5 7B / 32B / 72B at a fixed 4-GPU budget. What changes:
  absolute throughput, the quality level, the legal topology set, the saturation cap.
  What doesn't: the scale-out laws.
- **workload shape** — a decode-heavy KV stress that finds the boundary of the replica
  advantage.
- **node count** — N=8 across two nodes over a real fabric.
- **model family** — the whole study repeated on Llama-3.1-8B-Instruct (§7).

Medians over 3 reps ([min–max] where quoted); the frontier cells are 2 reps.
Figures: `dp_model_frontier`, `dp_model_quality`, `dp_model_efficiency`,
`dp_cap_collapse`, `dp_kvstress`, `dp_multinode`.

---

## 1. Fixed 4-GPU budget across model sizes

Every cell has four GPUs and total cap 512; only the topology and the model change.
Cells pool every prefill job family at that configuration (this is what `dp_model_frontier`
plots), so the 7B row differs slightly from `scale_out.md` §3, which quotes the `grid`
family alone: 574.6 / 451.6 / 368.8 for the same three topologies. Same experiment,
different selection — the pooled figures are the ones comparable across model sizes,
since 32B and 72B have no `grid` family of their own.

| model | 4 × TP1 | 2 × TP2 | 1 × TP4 | F1 |
|---|---|---|---|---|
| 7B | **580.6** rows/s | 451.6 | 377.5 | 0.926 |
| 32B | **134.2** | 110.0 | 102.6 | 0.859 |
| 72B | *does not fit* | **50.7** | 47.8 | 0.891 |

**The topology ranking survives every model size.** Replicas beat shards at 7B
(1.54× end to end) and at 32B (1.31×); at 72B the 4×TP1 slot does not exist because one
replica no longer fits on one GPU, and among the topologies that *do* fit, the less
sharded one still wins (2×TP2 > 1×TP4, 1.06×).

**The missing slot is the point.** Model size does not change which topology is best; it
changes which topologies are legal. The replica advantage shrinks as the model grows
(1.54× → 1.31× → 1.06×) because sharding becomes compulsory rather than optional.

**Quality is not monotone in model size**: 32B scores *lower* than 7B on this task
(0.859 vs 0.926) and 72B recovers only part of the gap (0.891). The task is a
binary sentiment filter with a strict criterion; the bigger models are more conservative
(recall 0.759 and 0.810 against 7B's 0.972) at near-identical precision (0.989–0.991).
Bigger is not better here, and a throughput study that assumed it would be would have
drawn the wrong frontier.

## 2. Prompt-format sensitivity falls off with model size

A side effect of standardising the tuple encoding, visible because every model was
re-measured: the encoding change moved 7B far more than the larger models.

| model | precision | recall |
|---|---|---|
| 7B | 0.960 → **0.883** | 0.862 → **0.974** |
| 32B | 0.992 → 0.989 | 0.750 → 0.760 |
| 72B | 0.992 → 0.991 | 0.786 → 0.810 |

7B moves by 0.08–0.11 on both axes; 32B and 72B by 0.003–0.024. **Sensitivity to prompt
formatting decays steeply with model size** — which is worth knowing before tuning a
prompt on a small model and assuming it transfers.

## 3. The scale-out laws replicate at 32B

The same `unified` preset at 32B: both axes' cells in one job, TP=1 the shared base.

| GPUs | DP rows/s (N×TP1) | TP rows/s (1×TP N) | DP advantage | DP per-GPU | TP per-GPU |
|---|---|---|---|---|---|
| 1 | 34.2 | (shared base) | — | 1.00 | 1.00 |
| 2 | 67.8 | 55.1 | 1.23× | 0.98 | 0.80 |
| 4 | 134.2 | 102.9 | **1.30×** | **0.98** | **0.75** |

Every 7B law holds: DP is linear (3.92× on 4 GPUs, per-GPU 0.98), TP is sub-linear
(0.75 per GPU), and the gap widens with the budget. The DP advantage is smaller than
7B's 1.51× because TP degrades more gently at 32B — a bigger model has more compute per
all-reduce, so the comm-bound penalty is proportionally smaller. That is the direction
the comm:compute ratio predicts.

**F1 is fleet-size-invariant at 32B too**: 0.857–0.860 across all cells.

## 4. The saturation cap is a per-model quantity

The cap sweep at 4×TP1, each fleet normalised by its own plateau and its own saturation
cap C\* (`dp_cap_collapse`):

| cap per endpoint | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| 7B (% of plateau) | | | **83.8** | 92.9 | 98.7 | 100.0 | 99.8 |
| 32B (% of plateau) | 66.8 | **83.9** | 94.0 | 97.5 | 100.0 | | |

Normalised by C\*, the two curves lie on top of each other — 83.8 against 83.9 at
C\*/4, identical 100.0 at 2·C\*. **The cap law is a shape, not a number**: each fleet
saturates at its own C\* and the response around it is the same curve. Interpolated C\*
is ~118 requests per endpoint at 7B and ~73 at 32B, so a bigger model saturates at lower
concurrency — it holds fewer requests in flight for the same throughput because each one
occupies the GPU longer. Quoting 128 per endpoint as universal would be quoting the 7B
instantiation of a law.

## 5. Workload shape: where the replica advantage ends

The prefill-dominated benchmark is the friendliest case for replicas. Forcing long
generations (`ignore_eos`, 512 and 1024 output tokens, 4,096 rows) pushes resident KV
past what a replica pool holds:

| workload | model | 4 × TP1 | 1 × TP4 | replica advantage |
|---|---|---|---|---|
| prefill (baseline) | 7B | 574.6 | 368.8 | **1.56×** |
| 512 out | 7B | 97.1 | 72.8 | 1.33× |
| 1024 out | 7B | 49.6 | 28.6 | **1.73×** |
| prefill | 32B | 134.2 | 102.6 | 1.31× |
| 512 out | 32B | 23.1 | 23.1 | **1.00×** |
| 1024 out | 32B | 9.9 | 12.0 | **0.83× — inverted** |

**At 7B the replica advantage survives decode stress; at 32B it erodes to parity and
then inverts.** The mechanism is KV capacity: each replica has its own KV pool sized to
one GPU, while a TP=4 endpoint pools four GPUs' worth. Deep decode makes resident KV the
binding constraint, and at 32B — where weights already consume most of each GPU — the
replica pool runs out first. The 7B control, which keeps winning at 1024 output tokens,
shows this is a memory boundary rather than a decode-length effect.

**So "replicas beat shards" is a claim about prefill-dominated semantic operators on
models that fit comfortably per GPU.** For a filter or an extraction that emits a few
tokens per row, that is the regime. For long-form generation on a large model, the
sharded topology is the right one.

## 6. Node count: N=8 across two nodes

| fleet | cap | rows/s | per-GPU | vs 4-GPU |
|---|---|---|---|---|
| 8 × TP1 (2 nodes) | 512 | 1064.0 [984.9–1072.2] | 0.96 | 1.83× |
| 8 × TP1 (2 nodes) | 1024 | **1081.1** [1034.2–1103.6] | **0.98** | 1.86× |

**Crossing the node boundary costs nothing measurable.** Per-GPU efficiency holds at
0.96–0.98 against the single-node base, and the cap knee behaves as at N=4 — 512 total
(64 per endpoint) is slightly under-fed, 1024 (128 per endpoint) reaches the plateau,
exactly the 128·N rule. The network hop is invisible at this offered load because the
operator's requests are large relative to the round-trip and the client is nowhere near
saturated (0.32–0.43 cores at N=8).

*The harness was gated before these cells were trusted: a single-node 4×TP1 control in
the same job family had to clear the gate floor (`MIN_OP_RATE` 500, set against a
556–568 rows/s reference band measured over ten earlier single-node reps), which it did.
The JSON-era single-node rate is 574.6–582.8, above that band — the gate is a
stale-harness check, not a calibration.*

## 7. Model family: Llama-3.1-8B-Instruct

Everything above is one model family. This axis repeats the load-bearing experiments on
`meta-llama/Llama-3.1-8B-Instruct` — 8B against Qwen's 7B, dense, instruction-tuned,
uniform GQA with no sliding window and no logit soft-capping, so prefix-cache accounting
(and therefore computed tok/s) behaves as it does for Qwen. Everything else is held
fixed: same dataset, prompts, JSON tuple encoding, `--max-model-len 16384`, guided
decoding backend and cold-fleet discipline. Predictions were registered before the runs.

### 7.1 The scale-out laws transfer unchanged

| claim | predicted | Qwen | **Llama** |
|---|---|---|---|
| DP per-GPU (N=1/2/4) | ~1.0 | 1.00 / 1.07 / 1.07 | **1.00 / 1.03 / 1.00** |
| TP per-GPU at TP=4 | <0.8 | 0.70 | **0.67** |
| DP vs TP at 4 GPUs | DP > TP | 1.51× | **1.47×** |
| replicas beat shards | monotone | 574.6 > 451.6 > 368.8 | **519.6 > 394.3 > 352.0** (1.48×) |
| cap knee | ≈128·N | plateau by 512 | **plateau by 512** (497 → 521 → 525 at 256/1024/2048) |
| F1 fleet-invariant | yes | 0.925–0.927 | **0.854–0.855** |
| operator vs scalar, R=1 | 7–8× | 8.2× | **7.8×** |

Every pre-registered prediction holds. Llama is marginally worse at TP (0.67 vs 0.70
per GPU), which is the direction its ~2.3× larger KV per token predicts — 32 layers × 8
KV heads against Qwen's 28 × 4.

### 7.2 The full prompt does not survive the family change

| filter, Llama | rows/s | precision | recall | F1 | rows passed |
|---|---|---|---|---|---|
| flock full R=1 | 133.3 | 0.762 | **1.000** | 0.865 | **1994 / 2000** |
| flock scalar R=1 | 17.1 | 0.761 | 1.000 | 0.865 | 1995 / 2000 |
| *always-true baseline* | — | 0.744 | 1.000 | *0.853* | *2000* |

**On Llama the 7-section `META_PROMPT` elicits essentially no discrimination**: it passes
99.7% of rows and scores 0.012 above a filter that answers yes to everything. On Qwen the
same prompt gave the best quality in the entire study (F1 0.931, 1641 rows passed). The
prompt that ships with flock is therefore not merely suboptimal on this family — it is
close to inoperative, and a study run only on Qwen would never have found that.

This changes what the prompt lever *is*. On Qwen, slimming trades quality for throughput.
On Llama it is what makes the operator work at all:

| filter, Llama | rows/s | F1 |
|---|---|---|
| flock slim R=1 | 350.7 | 0.873 |
| flock slim R=4 | 568.9 | 0.894 |
| flock slim R=8 | 615.1 | 0.909 |
| **flock slim R=16** | **629.1** | **0.915** |
| flock slim R=32 | 574.4 | 0.880 |
| LOTUS | 182.7 | 0.891 |
| Palimpzest | 109.7 | 0.901 |

flock's best point beats both competitors **on both axes** — 3.4× LOTUS's rate at +0.024
F1, 5.7× Palimpzest's at +0.014 — a cleaner win than anything in the Qwen table.

### 7.3 Batching helps on Llama and hurts on Qwen, and the mechanism separates

The most striking reversal. Slim quality *rises* with R on Llama (F1 0.873 → 0.915 from
R=1 to R=16) where it falls monotonically on Qwen (0.877 → 0.811). Precision stays above
0.95 for both throughout, so the whole divergence is recall: Qwen loses 0.085, Llama
gains 0.078, from near-identical starting points (0.788 / 0.783 at R=1).

The per-row verdict dumps decompose that into two independent quantities
(`diagnose_batch_position.py`): the **level**, i.e. the overall miss-rate of gold
positives, and the **ramp**, i.e. the back-of-prompt miss-rate minus the front-of-prompt
one. A positive ramp is the lost-in-the-middle effect — rows later in the packed prompt
under-detected.

| R | Qwen miss level | Qwen ramp | Llama miss level | Llama ramp |
|---|---|---|---|---|
| 1 | 0.220 | — | 0.225 | — |
| 4 | 0.276 | +0.129 | 0.188 | −0.027 |
| 8 | 0.291 | +0.003 | 0.159 | +0.005 |
| 16 | 0.310 | +0.041 | **0.149** | −0.011 |
| 32 | **0.363** | +0.158 | 0.180 | +0.052 |

**The two effects are separable and they behave differently.** Qwen shows a positive ramp
at most R *and* a level that degrades by two-thirds; both push recall down. Llama shows
**no positional effect at all** — its ramp scatters around zero with no trend — while its
level *improves by a third*, bottoming at R=16 before its own limit appears at R=32.

So the batching penalty is positional and model-specific, and the batching benefit is a
uniform calibration gain: packing several reviews into one prompt makes Llama better at
recognising positives everywhere in the batch, not at particular positions. An earlier
draft of this section attributed Llama's gain to the batched form's explicit
count-and-index contract correcting an over-permissive default. The dumps rule that out:
the contract is byte-identical across the two families, and it is the *level*, not the
ramp, that moves.

For the same reason "packing rows costs quality" is a Qwen statement, not a general one.
What generalises is the decomposition — level and ramp are the two things to measure —
not the sign of either.

*(Qwen's full head, for contrast, degrades both ways at once: ramp +0.377 and front-bucket
miss-rate 0.026 → 0.465 between R=1 and R=32. The batched slim form suppresses both,
which is what §3 of `evaluation.md` credits it with.)*

### 7.4 Extract inverts as well

| extract, Llama | rows/s | F1 | computed tok/s |
|---|---|---|---|
| **flock slim R=16** | **557.3** | **0.871** | 28,014 |
| flock full R=1 | 130.3 | 0.867 | **31,129** |
| flock slim R=8 | 555.6 | 0.854 | 28,686 |
| Palimpzest | 213.4 | 0.841 | 15,879 |
| LOTUS | 146.9 | 0.793 | 11,196 |
| flock scalar | 40.4 | 0.623 | 1,864 |

On Qwen, flock lost extract on quality to both competitors. On Llama it wins on both
axes, and by a wide margin: slim R=16 is simultaneously the highest-quality arm in the
table and **3.79× LOTUS's rate at +0.078 F1**, **2.61× Palimpzest's at +0.030**. It is the
sole point on both the rows/s–F1 and the engine-efficiency frontier. So the extract
standing is family-dependent and should not be stated as a property of the systems.

Job 3116950, all 11 arms n=3, measured after the response-schema fix (`2565fe75`), so
this table shares its schema era with the Qwen extract table. The operator arms gained
17–92% from that fix; `flock_scalar` gained **0.0%** (40.4 → 40.4 rows/s, F1
bit-identical across all six reps in both eras), which is what bounds the node component
of the comparison at zero for this client. LOTUS and Palimpzest drifted +16.5% and +3.8%
between the two sessions; both are CPU-bound Python clients that never enter flock code,
and the flat scalar rules the drift out as a flock-path effect. The ratios above are
in-job and therefore confound-free; the absolute rows/s still must not be compared across
model families, as ever.

### 7.5 What generalizes and what does not

**Transfers:** every engine and scale-out result — operator vs scalar, DP linearity, TP
sub-linearity, replicas over shards, the cap knee, F1 invariance to fleet size — and the
engine-efficiency lead on computed tok/s (1.82× LOTUS on Llama against 1.77× on Qwen).
These are properties of how flock drives the engine, and they do not reference the model.

The extract operator/scalar ratio is the sharpest instance. Measured on three independent
surfaces across two families, it agrees to ~5%: **12.98×** (Qwen thread sweep), **12.51×**
(Qwen cross-system) and **12.35×** (Llama cross-system). The response-schema fix roughly
doubled all three together, from 7.12 / 6.64 / 7.27, which is itself evidence that what is
being measured is a property of the dispatch path rather than of either model.

**Does not transfer:** anything about *prompt quality*. Which prompt works, whether
batching helps or hurts, and how flock's quality compares to LOTUS and Palimpzest are all
family-dependent. The thesis should present the engine results as general and the
prompt-quality results as measured-on-two-families, with the Llama collapse of the full
prompt as the concrete evidence for why that distinction matters.

*Caveat: the Llama engine ratio (1.82×) is computed against Llama's own LOTUS baseline.
Tokenizers differ between families, so absolute tok/row and computed tok/s are not
comparable across the two models — only ratios within a model are.*

## 8. Caveats

- The frontier cells are 2 reps, not 3; the 32B replication and the KV cells are 3.
- 72B has no 4×TP1 cell because one replica does not fit on one GPU. Its two-topology
  comparison is therefore weaker evidence than 7B's or 32B's three-way one.
- The 72B cap sweep at 256/1024 timed out and was not re-run: no figure consumes it, the
  frontier being fixed at cap 512.
- Quality across model sizes is one task with one criterion. The non-monotone F1 is a
  statement about this filter, not about the models.
- Two nodes is not many nodes. The claim is that the first network hop is free at this
  offered load, not that DP scales indefinitely.
- Two families is not many families. §7 establishes that the engine results are not
  Qwen-specific and that the prompt-quality results *are* family-specific; it does not
  establish which behaviour a third family would show.

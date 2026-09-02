# flock Semantic Filter — Scale-Out (TP + DP)

How far one `sem_filter` scales when GPUs are added, along the two axes a semantic
operator can use: **vertically**, by sharding one vLLM endpoint across GPUs
(`--tensor-parallel-size`), and **horizontally**, by putting one endpoint per GPU and
dispatching across the fleet (`semantic_endpoints` + the `EndpointRouter`). The second
axis is the one stock flock architecturally cannot reach: the scalar `llm_filter`
addresses a single `base_url`.

Both studies share the operating point, so they merge onto one GPU axis: full prompt on
every arm (`FLOCK_SEM_PROMPT`/`FLOCK_SEM_VARIANTS` unset — prompt-byte parity), JSON
tuple encoding, R=1, Qwen2.5-7B-Instruct bf16 greedy on GH200s,
`--enable-prefix-caching`, cold fleet per measurement, vLLM v0.22.0 (V1 engine),
medians over 3 reps with [min–max]. Data: `analysis/slurm/tp_scaling_clariden.sh` →
`summarize_tp.py`; `analysis/slurm/dp_scaling_clariden.sh` → `summarize_dp.py`.
Figures: `plot_tp_scaling.py`, `plot_dp_scaling.py`.

Throughput is reported as **rows/s** and as **computed tok/s** =
`(prompt − prefix_cache_hits + generation) / elapsed`, i.e. tokens the GPU actually
processed; raw `prompt_tokens_total` counts cache hits and inflates by ~2× at a 51%
hit rate.

---

## 1. Vertical: TP scaling of one endpoint

One endpoint sharded across 1/2/4 GPUs, operator cap swept per TP, both arms in-job.

| TP | operator rows/s (best cap) | eff | per-GPU | scalar rows/s | op/scalar |
|---|---|---|---|---|---|
| 1 | 140.0 [137.4–146.9] (128) | 1.00 | 1.00 | 18.8 [18.7–18.9] | **7.45×** |
| 2 | 232.6 [227.5–236.4] (256) | 1.66 | 0.83 | 25.4 [25.2–26.1] | 9.16× |
| 4 | 394.3 [387.8–395.8] (512) | 2.82 | 0.71 | 30.5 [30.3–30.7] | **12.93×** |

The operator tracks the rising ceiling and the scalar cannot — its concurrency is
`min(threads, morsels)` = 1 against the operator's `in_flight_cap` — so the gap widens
monotonically. At concurrency 1 the scalar can only harvest TP's per-request latency
gain, which is bounded and decelerating; the operator harvests latency *and* the
engine's growing concurrent-batch capacity.

**The cap must scale with TP (≈128·TP).** At TP=4 the operator reaches 302.8 → 338.8 →
394.3 rows/s at cap 128/256/512: at the unscaled cap the engine is client-starved.

The operator sustains an in-flight population *near* its configured cap, not exactly at
it — 121.6 of 128 at TP=1 (95%), 238.7 of 256 at TP=2 (93%), 443.2 of 512 at TP=4 (87%).
The shortfall is the dispatch and completion gap: a slot is briefly empty between a
response arriving and the next request being submitted, and that gap is a larger fraction
of a deeper window. What matters for the claim is that the population tracks the *knob*
rather than the plan, which the scalar's flat 1.0 at every layout does not.

**TP is sub-linear for a 7B model** — 71% per-GPU at TP=4, the comm-bound cost of two
all-reduces per layer per step. That shortfall is what the horizontal axis avoids.

## 2. Horizontal: DP scaling of the fleet

**Setup:** N ∈ {1, 2, 4} endpoints × TP1, `semantic_in_flight_cap` = 128·N — a constant
128 in-flight requests per GPU, the single-endpoint saturation knee. The TP cells of the
same table run the mirror-image caps (1×TP2 at 256, 1×TP4 at 512), so every cell of both
axes faces identical offered concurrency per GPU, **measured in the same jobs on the
same nodes**. 32,000 rows of `movie/sf_300000` per timed run; F1 from an untimed verdict
pass per fleet size on 2,000 gold-labelled `sf_2000` rows. Scalar reference:
`rewrite=off` pointed at the *same* 4-endpoint fleet, same dataset.

| fleet | cap | rows/s | speedup | per-GPU | computed tok/s | F1 |
|---|---|---|---|---|---|---|
| 1 × TP1 | 128 | 136.7 [136.1–138.1] | 1.00 | 1.00 | 33,778 | 0.926 |
| 2 × TP1 | 256 | 290.6 [289.9–291.7] | 2.13 | 1.07 | 71,785 | 0.926 |
| 4 × TP1 | 512 | 582.8 [582.4–585.8] | **4.26** | **1.07** | 143,975 | 0.926 |
| scalar on the 4-endpoint fleet | 128 | 18.6 [18.3–18.6] | — | — | 4,581 | — |

**Findings:**

1. **DP is linear to within noise: 4.26× on 4 GPUs.** Per-GPU efficiency reads 1.07 —
   7% above unity, with a [1.05–1.07] spread that puts it outside run-to-run noise — but
   that is the base cell, not superlinear scaling: N=1 at cap 128 does 136.7 rows/s while
   the same GPU inside the fleet does ~146, so the single endpoint is marginally
   under-driven at that cap. The honest claim is **no measurable scaling loss**.
   Computed tok/s tracks rows/s exactly (511 tokens per row at every fleet size), so the
   linearity is real work, not changed work.
2. **The operator reaches 31× the scalar at 4 GPUs** (582.8 vs 18.6, dataset-matched).
   The scalar's flat line is architectural, and measured rather than asserted: pointed at
   the identical 4-endpoint fleet it lands **100% of its requests on ep0 and none on the
   rest** (per-endpoint counter deltas, every rep, fail-loud gate). It names one
   `base_url`; no fleet is reachable at any thread count.
3. **The cap is the fleet's admission knob, with a knee at ≈128·N.** At N=4:
   491.6 → 544.9 → 578.7 → 586.5 → 585.1 rows/s for cap 128/256/512/1024/2048.
   Under-feeding costs **15%**; 128·N captures 98.6% of the plateau; past it the surplus
   buys only latency, with Little's-law concurrency tracking the cap in every cell (at
   the same 87–95% of it as in §1) — the knob, not the hardware, sets in-flight depth.
4. **F1 is fleet-size-invariant: 0.925–0.927** across every configuration and rep. DP is
   pure routing, and the numbers say so.

## 3. Fixed 4-GPU budget: replicas vs shards

All three topologies measured **in the same jobs** at total cap 512, so the comparison
carries no cross-node variance.

| topology | rows/s | vs 1×TP4 | per-GPU | computed tok/s | F1 |
|---|---|---|---|---|---|
| **4 × TP1** | **574.6** [561.7–583.2] | **1.56×** | 1.04 | 141,954 | 0.926 |
| 2 × TP2 | 451.6 [446.9–453.6] | 1.22× | 0.82 | 111,481 | 0.926 |
| 1 × TP4 | 368.8 [366.1–405.8] | 1.00× | 0.67 | 90,928 | 0.927 |

**Replicas beat shards, monotonically**, at identical GPU count, identical total cap and
identical F1. The mechanism is admission latency and it is GPU-side: one scheduler
serialises admission no matter how wide its shards, four admit in parallel. **Sharding
also raises variance** — 1×TP4 spans ±5% across reps where 4×TP1 holds ±2%. For a
7B-class model, **spend GPUs on replicas, not shards**.

*The 1×TP4 latency split is bimodal across reps — queue time 74/90/242 ms against TTFT
854/617/837 ms — so no single queueing number is quoted for that topology; the range is.
The throughput ordering, which is the claim, is monotone in every individual rep.*

## 4. Consolidated: the two axes side by side

Every cell below comes from the `unified` preset — both axes' cells in one SLURM job,
TP=1 the shared base.

| GPUs | DP rows/s (N×TP1) | TP rows/s (1×TP N) | DP advantage | DP per-GPU | TP per-GPU |
|---|---|---|---|---|---|
| 1 | 136.7 | (shared base) | — | 1.00 | 1.00 |
| 2 | 290.6 | 227.3 | 1.28× | 1.07 | 0.83 |
| 4 | 582.8 | 386.1 | **1.51×** | **1.07** | **0.70** |

DP holds per-GPU efficiency flat while TP decays to 0.70, so the gap grows with the
budget — 1.51× at 4 GPUs, and widening.

*A note on this number's history, because it changed twice. An early cross-study version
of this table said 1.51×; an in-job re-run then produced 1.42×, and we attributed the
difference to node variance flattering DP. The current in-job measurement — same preset,
same discipline, under the corrected prompt configuration — returns 1.51× again. The
1.42× was therefore not a variance correction but a measurement under a configuration we
have since fixed, and two independent sessions agreeing on 1.51× is the better evidence.
What the in-job design buys is still real — it removes cross-node variance from the
comparison — it simply did not move the number.*

**Bottleneck call: GPU-side, not client-side.** The cap-2048 probe settles it: the driver
uses 0.02 cores at 124 in flight and 0.20 at ~1,965 — CPU grows ~16× more slowly than
concurrency, and at four times the fleet's useful depth the client still has 5× headroom
to one core. The blocking scalar, by contrast, burns a full core (0.97–1.01) to sustain
concurrency 1. Both sub-linear cells are identified positively rather than by
elimination: the cap-128 fleet is client-limited by construction (empty engine queues,
the cap is a client knob), and 1×TP4 is engine-limited (deep queues, idle client).

## 5. Hypothesis verdicts

| | hypothesis | verdict |
|---|---|---|
| **H1** | DP scales near-linearly in N; DP efficiency > TP efficiency | **Supported.** 4.26× on 4 GPUs (per-GPU 1.07, base-limited — read as no measurable loss) vs TP's 0.70; DP 1.51× TP at 4 GPUs. Replicated at 32B in the generalization chapter. |
| **H2** | At a fixed 4-GPU budget, replicas beat shards | **Supported.** 574.6 > 451.6 > 368.8 in-job (1.56× end to end), monotone in every rep, equal F1. Cause: scheduler-admission queueing in the sharded engine. |
| **H3** | The cap must scale with the fleet (≈128·N) or the fleet starves | **Supported as a knee, not a cliff.** Under-feeding costs 15%; 128·N captures 98.6% of the plateau. The saturation cap is a per-model quantity — interpolated C\* is ~118 per endpoint at 7B and ~73 at 32B — so 128·N is the 7B instantiation of a law, not the law itself. |
| **H4** | F1 is invariant to fleet size *and* topology | **Supported.** 0.925–0.927 across every configuration of both axes; no routing row loss at any size. |

## 6. Caveats

- The 1×TP4 configuration is intrinsically noisy: [366–406] across reps (±5%) where
  4×TP1 holds ±2%, and its latency split is bimodal. Medians are quoted; the
  replicas-beat-shards ordering holds in every individual rep.
- Per-GPU efficiency above 1.00 reflects an under-driven N=1 base at cap 128, not
  superlinear scaling.
- The saturation cap C\* is estimated by interpolating the cap sweep on a log axis and
  then snapping to the nearest power-of-two cap for the normalised figure. The old rule
  (smallest measured cap clearing 98% of plateau) sat a knife-edge from a factor-of-two
  jump: at 32B the c256 cell reads 98.5% in one run and 97.8% in another.
- The 3 s scheduler-gauge sampler is polled per endpoint sequentially, so per-endpoint
  `run_mean` carries sampling skew and is not plotted. Request shares and Little's-law
  concurrency are the quantitative statements.
- `round_robin` on a homogeneous fleet only. Sticky/least-loaded behaviour, and any
  heterogeneous or cache-pressured fleet, is `router_analysis`'s scope.
- The scalar's heroic best case (128 threads × 128 morsels) is reported in
  `evaluation.md` §1, not here.
- Everything here is one node and loopback endpoints; the two-node extension, the larger
  models and the decode-heavy workload boundary live in `model_scale.md`.

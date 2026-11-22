# Skein vs Flock vs LOTUS vs Palimpzest: SemBench multimodal results (gemma-4-26B-A4B-it)

Data: [`figures/data/cross_system_multimodal_gemma-4-26b-a4b-it/`](figures/data/cross_system_multimodal_gemma-4-26b-a4b-it/) (commit `fc200973`, tag `paper-experiments`), one folder per scenario/system/job. Provenance, job ids and caveats: [`INPUT_PROVENANCE.md`](figures/data/cross_system_multimodal_gemma-4-26b-a4b-it/INPUT_PROVENANCE.md). Every number below is computed from the metric JSONs in those folders.

## TL;DR

- **Skein is the fastest system end to end:** 41 queries in **3826 s**, against 8088 s for LOTUS (**2.1×**), 9504 s for Palimpzest (**2.5×**) and 15813 s for stock Flock (**4.1×**).
- **Fastest total runtime in 4 of 5 scenarios.** In the fifth (animals, 4 tiny queries) only PZ is faster (34 s vs 43 s), and PZ gets 2 of those 4 queries wrong.
- **Skein is on the runtime-quality Pareto front in every scenario.** No other system is both faster and more accurate. In ecomm Skein dominates all three other systems (fastest *and* best mean quality).
- **Mean F1/accuracy over the 36 F1/acc queries:** Skein 0.779 · Flock 0.763 · PZ 0.709 · LOTUS 0.708. Skein has the highest mean quality, so its speed does not cost accuracy.
- **Against stock Flock (same SQL, same prompts):** quality ties on 27 of 41 queries, Skein is faster on 39/40 queries, and the per-query speed-up is 5.5× (geomean). It sends the same number of LLM requests (see [LLM work](#3-llm-work)), so the speed-up comes from execution, not from skipping work.

| Skein vs … | quality W / T / L (per query) | per-query speed-up (geomean) | Skein faster on | total runtime ratio |
|---|---|---|---|---|
| Flock | **8** / 27 / 6 | **5.5×** | 39 / 40 queries | **4.1×** |
| LOTUS | **18** / 13 / 10 | **2.0×** | 23 / 41 queries | **2.1×** |
| PZ | **12** / 16 / 13 | **2.1×** | 33 / 41 queries | **2.5×** |

W/T/L: Skein's quality metric is better / within ±0.005 / worse. A failed query of the other system counts as a Skein win. Speed-ups are other system ÷ Skein.

## 1. Setup

### Systems

| system | what it is | variant run here | paper | code |
|---|---|---|---|---|
| **Skein** (ours) | DuckDB extension, topology-aware semantic operators over a co-located LLM (fork of Flock) | async semantic **operators** (`llm_filter`/`llm_complete` planned as physical operators), 128 requests in flight, LIMIT early stop | MSc thesis, ETH Zurich 2026 | [dmicloiu/Skein](https://github.com/dmicloiu/Skein) |
| **Flock** (FlockMTL) | DuckDB extension, LLM calls as SQL scalar/aggregate functions | stock **scalar** `llm_filter`/`llm_complete`, batch 32 | [Dorbani et al., VLDB 2025](https://arxiv.org/abs/2504.01157) | [dais-polymtl/flock](https://github.com/dais-polymtl/flock) |
| **LOTUS** | Pandas semantic-operator engine | `lotus-ai` 1.1.3 + local bug-fix patches, concurrency 128, `sem_join` default cascade (`approximate`, target 0.8) | [Patel et al., 2024/25](https://arxiv.org/abs/2407.11418) | [lotus-data/lotus](https://github.com/lotus-data/lotus) |
| **Palimpzest** (PZ) | Declarative AI-pipeline optimizer | SemBench submodule + 1 local patch | [Liu et al., CIDR 2025](https://arxiv.org/abs/2405.14696) | [mitdbg/palimpzest](https://github.com/mitdbg/palimpzest) |

Benchmark: **SemBench** ([VLDB 2026](https://arxiv.org/abs/2511.01716), [SemBench/SemBench](https://github.com/SemBench/SemBench)), with SemBench's per-system query implementations.

### Shared conditions (identical for all systems)

| | |
|---|---|
| model | `google/gemma-4-26B-A4B-it` (snapshot `4d7ae498`), one model for every operator and modality |
| serving | vLLM 0.22.0, one cold co-located server per job, 1 × GH200, TP = 1, `max_model_len` 32768 |
| repetitions | 1 measured run per query, after an untimed warm-up at small scale on the same server |
| runtime | SemBench `execution_time` of the measured run (s) |
| quality | **F1** / **acc**uracy (↑ higher is better) or **RE** = relative error (↓ lower is better) |
| scenarios | animals, movie, mmqa, ecomm, cars (medical excluded); only queries every system implements |

### Scenarios

| scenario | scale | modalities | queries | operator mix |
|---|---|---|---|---|
| [animals](#animals) | 200 images | image | 4 | image filters + aggregation |
| [movie](#movie) | 2000 reviews | text | 8 | text filters, self-joins, LIMIT |
| [mmqa](#mmqa) | 200 rows/table | table + text + image | 11 | table ⋈ image joins, text filters/maps |
| [ecomm](#ecomm) | 500 products | text + image | 13 | filters, maps, classification, 2-4-way joins |
| [cars](#cars) | 19672 cars | table + text + image | 5 | large text/image filters, classification, LIMIT |

## 2. Results by scenario

Cells: quality metric · runtime. **Bold** marks the best quality and, separately, the fastest runtime per query. Speed-up = other system ÷ Skein.

### animals

200 images · image · 4 queries · totals: Skein 43 s · Flock 155 s · LOTUS 262 s · PZ 34 s

- Skein = Flock quality on all 4 queries, **3.6× faster** than Flock and **6.0× faster** than LOTUS.
- PZ has the lowest total time (34 s vs 43 s) but returns **wrong answers on 2 of 4 queries** (Q3, Q10: F1 0.00).
- LOTUS is the only system with F1 1.00 on Q7 (Skein/Flock/PZ 0.89), at 6.7× Skein's runtime.

| query | task | metric | Skein | Flock | LOTUS | PZ |
|---|---|---|---|---|---|---|
| Q1 | image filter + COUNT: *How many zebra pictures?* | RE ↓ | **0.27** · 11 s | **0.27** · 42 s | **0.27** · 60 s | **0.27** · **10 s** |
| Q3 | image filter + GROUP BY/arg-max: *City with most zebra pictures* | F1 ↑ | **1.00** · 8.3 s | **1.00** · 29 s | **1.00** · 48 s | 0.00 · **3.2 s** |
| Q7 | 2 image filters + set intersection: *Cities where zebras and impalas co-occur* | F1 ↑ | 0.89 · 16 s | 0.89 · 56 s | **1.00** · 105 s | 0.89 · **15 s** |
| Q10 | image filter + GROUP BY/arg-max: *City and station with most zebra pictures* | F1 ↑ | **1.00** · 8.7 s | **1.00** · 28 s | **1.00** · 49 s | 0.00 · **5.3 s** |

| speed-up of Skein | vs Flock | vs LOTUS | vs PZ |
|---|---|---|---|
| total runtime | **3.6×** | **6.0×** | **0.8×** |
| per-query geomean | 3.6× | 5.9× | 0.7× |
| queries where Skein is faster | 4/4 | 4/4 | 0/4 |
| quality W / T / L | 0 / 4 / 0 | 0 / 3 / 1 | 2 / 2 / 0 |

### movie

2000 reviews · text · 8 queries · totals: Skein 392 s · Flock 4330 s · LOTUS 1145 s · PZ 2760 s

- **Fastest total** (392 s): 2.9× faster than LOTUS, 7.0× than PZ, 11.1× than Flock.
- LIMIT early stop: the LIMIT 5/10 queries (Q1, Q2, Q5, Q6) finish in ~1 s; Flock needs 7-131 s, LOTUS 0.4-760 s.
- Q7 (all pairs, ~21500 result rows) dominates: Skein 385 s vs Flock 3922 s (10.2×), PZ 2703 s (7.0×). LOTUS is faster (195 s) only because its approximate join cascade drops pairs: 8156 rows, F1 0.33 vs 0.69.
- Where others are better: Flock Q5 (F1 0.90 vs 0.80), PZ Q7 (0.77 vs 0.69), LOTUS/PZ on the RE counts (Q3/Q4 LOTUS 0.14 vs 0.43).

| query | task | metric | Skein | Flock | LOTUS | PZ |
|---|---|---|---|---|---|---|
| Q1 | text filter, LIMIT 5: *5 clearly positive reviews* | F1 ↑ | **1.00** · **1.0 s** | **1.00** · 126 s | **1.00** · 7.5 s | **1.00** · 1.4 s |
| Q2 | text filter, LIMIT 5: *5 positive reviews of one movie* | F1 ↑ | **1.00** · 1.0 s | **1.00** · 7.4 s | 0.60 · **0.4 s** | **1.00** · 28 s |
| Q3 | text filter + COUNT: *# positive reviews of one movie* | RE ↓ | 0.43 · 0.5 s | 0.43 · 6.7 s | **0.14** · **0.4 s** | 0.50 · 3.9 s |
| Q4 | text filter + ratio: *positivity ratio of one movie* | RE ↓ | 0.43 · 1.1 s | 0.43 · 8.2 s | **0.14** · **0.3 s** | 0.50 · 4.5 s |
| Q5 | text self-join, LIMIT 10: *10 review pairs, same sentiment* | F1 ↑ | 0.80 · **1.0 s** | **0.90** · 131 s | 0.33 · 760 s | 0.11 · 8.0 s |
| Q6 | text self-join, LIMIT 10: *10 review pairs, opposite sentiment* | F1 ↑ | **0.95** · **1.0 s** | **0.95** · 122 s | 0.46 · 182 s | **0.95** · 6.0 s |
| Q7 | text self-join (all pairs): *all review pairs, opposite sentiment* | F1 ↑ | 0.69 · 385 s | 0.69 · 3922 s | 0.33 · **195 s** | **0.77** · 2703 s |
| Q8 | text classify + GROUP BY: *# positive / negative reviews* | RE ↓ | 0.24 · 1.2 s | 0.24 · 7.7 s | 0.28 · **0.5 s** | **0.20** · 4.9 s |

| speed-up of Skein | vs Flock | vs LOTUS | vs PZ |
|---|---|---|---|
| total runtime | **11.1×** | **2.9×** | **7.0×** |
| per-query geomean | 23.3× | 3.3× | 6.0× |
| queries where Skein is faster | 8/8 | 3/8 | 8/8 |
| quality W / T / L | 0 / 7 / 1 | 5 / 1 / 2 | 3 / 3 / 2 |

### mmqa

200 rows/table · table + text + image · 11 queries · totals: Skein 1906 s · Flock 5727 s · LOTUS 3357 s · PZ 2701 s

- **Fastest total** (1906 s): 1.4× faster than PZ, 1.8× than LOTUS, 3.0× than Flock.
- Q7 (table ⋈ image logos) is ~88 % of the scenario's time: Skein 1684 s, F1 0.42 ≈ PZ (2382 s, 0.43); better than LOTUS (3175 s, 0.20) and Flock (5004 s, 0.32).
- Q2a/Q2b: Skein F1 0.67 vs Flock 0.48, same plans and prompts; stock Flock's image label biases filters to true.
- LOTUS is fastest on 8 queries (6 of them under 2 s, plus Q2a/Q2b: 88 vs 104 s). PZ has the best quality (best or tied on 10/11, Q2a 0.91).

| query | task | metric | Skein | Flock | LOTUS | PZ |
|---|---|---|---|---|---|---|
| Q1 | table join + text completion: *director of a movie via actor/role* | F1 ↑ | **1.00** · 2.4 s | **1.00** · 15 s | **1.00** · 1.9 s | **1.00** · **0.9 s** |
| Q2a | table ⋈ image filter: *logo images of racetracks* | F1 ↑ | 0.67 · 104 s | 0.48 · 327 s | 0.62 · **88 s** | **0.91** · 154 s |
| Q2b | table ⋈ image filter + image map: *… and the logo color* | F1 ↑ | 0.67 · 105 s | 0.48 · 316 s | 0.62 · **87 s** | **0.77** · 141 s |
| Q3a | text filter: *which movies are comedies* | F1 ↑ | 0.81 · 2.4 s | 0.81 · 13 s | **0.83** · **1.1 s** | 0.81 · 4.9 s |
| Q3f | text filter: *which movies are romantic comedies* | F1 ↑ | 0.75 · 2.3 s | 0.75 · 13 s | **0.86** · **1.0 s** | **0.86** · 4.6 s |
| Q4 | text map (multi-label): *categorise movies by genre* | F1 ↑ | 0.56 · **0.6 s** | 0.55 · 1.8 s | 0.56 · 0.7 s | **0.59\*** · 1.6 s |
| Q5 | text completion: *actor who played in all listed movies* | F1 ↑ | **1.00** · 0.3 s | **1.00** · 0.2 s | **1.00** · **0.1 s** | **1.00** · 1.5 s |
| Q6a | text filter: *airlines flying to Frankfurt* | F1 ↑ | **1.00** · 1.7 s | **1.00** · 13 s | **1.00** · **0.8 s** | **1.00** · 3.4 s |
| Q6b | text filter: *airlines flying to Germany* | F1 ↑ | **1.00** · 1.7 s | **1.00** · 12 s | **1.00** · **0.7 s** | **1.00** · 3.0 s |
| Q6c | text filter: *airlines flying to Europe* | F1 ↑ | **1.00** · 1.7 s | **1.00** · 13 s | 0.91 · **0.9 s** | **1.00** · 3.5 s |
| Q7 | table ⋈ image filter: *logo of each airline flying to Europe* | F1 ↑ | 0.42 · **1684 s** | 0.32 · 5004 s | 0.20 · 3175 s | **0.43** · 2382 s |

| speed-up of Skein | vs Flock | vs LOTUS | vs PZ |
|---|---|---|---|
| total runtime | **3.0×** | **1.8×** | **1.4×** |
| per-query geomean | 4.0× | 0.7× | 1.7× |
| queries where Skein is faster | 10/11 | 2/11 | 10/11 |
| quality W / T / L | 4 / 7 / 0 | 4 / 5 / 2 | 0 / 6 / 5 |

\* rescored offline: the query succeeded but the SemBench evaluator crashed on a NaN genre (fixed in sembench `88f5adf`).

### ecomm

500 products · text + image · 13 queries · totals: Skein 397 s · Flock 1497 s · LOTUS 1567 s · PZ 1574 s

- **Fastest total and best mean quality**: 397 s vs 1497-1574 s for every other system (**3.8-4.0×**), mean F1/acc 0.70.
- Fastest on 9 of 13 queries. The only system with non-zero F1 on Q10 (3-way outfit join): 0.35 vs 0.00 for all others.
- Losses: Q12 F1 0.00 (Skein's `llm_complete` output holds fragments (bare ids, braces) instead of the complete JSON objects the evaluator matches; PZ also 0.00, Flock fails); Q8 0.29 vs PZ 0.40; Q9 0.75 vs 0.83-0.86.
- LOTUS Q11 returns 1296 rows vs 27 for the others (accepts wrong pairs): F1 0.07.

| query | task | metric | Skein | Flock | LOTUS | PZ |
|---|---|---|---|---|---|---|
| Q1 | text filter: *Reebok backpacks* | F1 ↑ | **1.00** · 6.1 s | **1.00** · 35 s | **1.00** · **4.2 s** | **1.00** · 13 s |
| Q2 | image filter: *yellow/silver sports shoes* | F1 ↑ | **0.80** · 32 s | 0.77 · 113 s | 0.53 · 140 s | 0.73 · **28 s** |
| Q3 | text map: *extract brand* | acc ↑ | **0.93** · **6.2 s** | **0.93** · 39 s | 0.87 · 8.5 s | 0.87 · 14 s |
| Q4 | image map: *extract primary color* | acc ↑ | 0.71 · **14 s** | 0.68 · 40 s | 0.54 · 79 s | **0.78** · 18 s |
| Q5 | text classify: *product category from text* | acc ↑ | **0.98** · 3.2 s | **0.98** · 17 s | **0.98** · **3.1 s** | **0.98** · 10 s |
| Q6 | image classify: *product category from image* | acc ↑ | 0.89 · **11 s** | **0.92** · 33 s | 0.90 · 77 s | 0.89 · 726 s |
| Q7 | text self-join: *same category + brand pairs* | F1 ↑ | 0.88 · **93 s** | 0.89 · 547 s | 0.86 · 153 s | **0.91** · 284 s |
| Q8 | text ⋈ image join: *match long descriptions to images* | F1 ↑ | 0.29 · **23 s** | 0.33 · 77 s | 0.12 · 141 s | **0.40** · 39 s |
| Q9 | image self-join: *same category + surface color pairs* | F1 ↑ | 0.75 · **16 s** | 0.83 · 75 s | **0.86** · 336 s | 0.85 · 29 s |
| Q10 | image filter + 3-way join: *matching outfits (shoes/bottom/top)* | F1 ↑ | **0.35** · **40 s** | 0.00 · 96 s | 0.00 · 117 s | 0.00 · 239 s |
| Q11 | text+image filter + 4-way join: *all-black outfits with accessory* | F1 ↑ | **0.80** · 116 s | **0.80** · 342 s | 0.07 · 337 s | **0.80** · **100 s** |
| Q12 | text+image map → JSON: *JSON {id, brand, category}* | F1 ↑ | 0.00 · **9.7 s** | ✗ failed | **0.96** · 26 s | 0.00 · 20 s |
| Q13 | text+image filter: *men's striped running shirts* | F1 ↑ | 0.70 · **27 s** | 0.70 · 82 s | **0.78** · 145 s | 0.73 · 55 s |

| speed-up of Skein | vs Flock | vs LOTUS | vs PZ |
|---|---|---|---|
| total runtime | **3.8×** | **3.9×** | **4.0×** |
| per-query geomean | 3.9× | 3.2× | 2.6× |
| queries where Skein is faster | 12/12 | 11/13 | 11/13 |
| quality W / T / L | 4 / 5 / 4 | 7 / 3 / 3 | 4 / 4 / 5 |

### cars

19672 cars · table + text + image · 5 queries · totals: Skein 1088 s · Flock 4104 s · LOTUS 1756 s · PZ 2436 s

- **Fastest total** (1088 s on 19672 cars): 1.6× faster than LOTUS, 2.2× than PZ, 3.8× than Flock.
- Q3 (LIMIT 10): only Skein and Flock reach F1 1.00 (LOTUS/PZ 0.80).
- Q1: PZ (0.90) and LOTUS (0.88) beat Skein (0.80). Q8: Flock/LOTUS 0.48 vs Skein 0.41.
- Q10 (classify 19657 complaints): Skein 250 s vs Flock 1111 s, PZ 1330 s, LOTUS 362 s, with the best F1 (0.54, tied with Flock).

| query | task | metric | Skein | Flock | LOTUS | PZ |
|---|---|---|---|---|---|---|
| Q1 | text filter: *cars that were in a crash* | F1 ↑ | 0.80 · 223 s | 0.79 · 1008 s | 0.88 · **126 s** | **0.90** · 400 s |
| Q3 | image filter ⋈ table, LIMIT 10: *10 undamaged manual cars* | F1 ↑ | **1.00** · 240 s | **1.00** · 374 s | 0.80 · 286 s | 0.80 · **3.5 s** |
| Q4 | text filter + AVG: *avg. age of cars with engine problems* | RE ↓ | **0.01** · 228 s | **0.01** · 1003 s | **0.01** · **128 s** | **0.01** · 435 s |
| Q8 | image filter, LIMIT 100: *100 cars with punctures + scratches* | F1 ↑ | 0.41 · **148 s** | **0.48** · 607 s | **0.48** · 854 s | 0.30 · 269 s |
| Q10 | text classify (19657 rows): *faulty component per complaint* | F1 ↑ | **0.54** · **250 s** | **0.54** · 1111 s | 0.52 · 362 s | 0.50 · 1330 s |

| speed-up of Skein | vs Flock | vs LOTUS | vs PZ |
|---|---|---|---|
| total runtime | **3.8×** | **1.6×** | **2.2×** |
| per-query geomean | 3.6× | 1.3× | 0.9× |
| queries where Skein is faster | 5/5 | 3/5 | 4/5 |
| quality W / T / L | 0 / 4 / 1 | 2 / 1 / 2 | 3 / 1 / 1 |

## 3. LLM work

vLLM `/metrics` deltas over the measured run (all queries of the scenario). Cars is omitted because its reported numbers come from a mix of jobs (LIMIT reruns, split LOTUS jobs).

| scenario | metric | Skein | Flock | LOTUS | PZ |
|---|---|---|---|---|---|
| animals | LLM requests | 1000 | 1000 | 1000 | 999 |
|  | prompt tokens (M) | 0.7 | 0.7 | 0.4 | 0.7 |
|  | output tokens (M) | 0.01 | 0.01 | 0.05 | 0.05 |
| movie | LLM requests | 65965 | 71568 | 114815 | 66197 |
|  | prompt tokens (M) | 38.9 | 42.0 | 28.0 | 43.7 |
|  | output tokens (M) | 0.59 | 0.65 | 0.66 | 10.17 |
| mmqa | LLM requests | 46435 | 46441 | 56022 | 46251 |
|  | prompt tokens (M) | 34.8 | 34.8 | 20.4 | 38.6 |
|  | output tokens (M) | 0.46 | 0.45 | 0.26 | 3.18 |
| ecomm | LLM requests | 15644 | 15390 | 13389 | 17400 |
|  | prompt tokens (M) | 14.3 | 14.0 | 8.0 | 21.5 |
|  | output tokens (M) | 0.14 | 0.14 | 1.17 | 3.63 |

- **Skein ≈ Flock in requests and tokens** (same plans and prompts). Skein's speed-up therefore comes from keeping the GPU saturated (asynchronous operators, 128 requests in flight), not from sending less work. In movie, Skein sends fewer requests because of the LIMIT early stop.
- **LOTUS** sends 28-51 % fewer prompt tokens than Skein. Its joins run the approximate cascade, and it has the lowest join F1 of all systems on movie Q6/Q7, mmqa Q7 and ecomm Q8/Q11.
- **PZ** generates 5-25× more output tokens than Skein (verbose structured outputs), which adds decode time.

## 4. Optional: Skein embedding proxy (not part of the comparison above)

Skein can replace an image-text LLM filter with a CLIP embedding similarity proxy. This is a different plan, so it is reported separately (jobs 3966087 / 3966088):

| query | Skein operator (above) | Skein + embedding proxy | LOTUS-style cascade on Skein |
|---|---|---|---|
| mmqa Q2a | F1 0.67 · 104 s | **F1 0.83 · 6.0 s** | F1 0.42 · 8.2 s |
| mmqa Q2b | F1 0.67 · 105 s | **F1 0.83 · 1.5 s** | F1 0.42 · 4.3 s |
| mmqa Q7 | F1 0.42 · 1684 s | **F1 1.00 · 25 s** | F1 0.43 · 1129 s |
| cars Q8 | F1 0.41 · 148 s | F1 0.42 · 155 s | F1 0.14 · 120 s |

With the proxy, mmqa Q7 goes from 1684 s to 25 s and F1 rises from 0.42 to 1.00, the best result of any system on that query.

## 5. Caveats

- **1 repetition** per query. Short queries (< 3 s) are noisy, so read them as "all fast". The long queries decide the totals.
- **Skein uses the LIMIT early-stop build** for movie (job 3986000) and cars Q3/Q8 (3986001); the other queries come from the build without it. Without early stop, movie Q5/Q6 took ~385 s each.
- **Stock Flock is reported as-is.** Its image label biases image filters toward *true* (mmqa Q2a/Q2b, Q7). ecomm Q12 fails deterministically (malformed JSON from the model → `Output extraction error`).
- **LOTUS** uses approximate joins by default (fewer LLM calls, lower join recall/precision) and runs with local bug-fix patches (image embeddings, join ids, RGBA logos, litellm retries; sembench `patches/lotus-ai-1.1.3/`).
- **PZ mmqa Q4** rescored offline (evaluator crash on a NaN genre).
- All systems use SemBench's own query implementation for that system. Prompts differ across systems, except Skein vs Flock, which run identical SQL.

_Generated from the metric JSONs in [`figures/data/cross_system_multimodal_gemma-4-26b-a4b-it/`](figures/data/cross_system_multimodal_gemma-4-26b-a4b-it/). Regenerate after any data change._

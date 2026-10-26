# movie — input provenance

SemBench movie scenario at `sf_2000` (text only, `llm_filter` only), queries Q1-Q8, warm-up at `sf_20`.
Shared setup: `../INPUT_PROVENANCE.md`. This file records inputs and checks only.

## Jobs used

| system | job | queries | sembench | started | vLLM requests (measured run) |
|---|---|---|---|---|---|
| `skein` | 3720918 | Q1-Q8 | `1055119` | 2026-10-22 13:57 | 197,552 |
| `flock` | 3720919 | Q1-Q8 | `1055119` | 2026-10-22 14:15 | 71,568 |
| `lotus` | 3720920 | Q1-Q8 | `bc7bb8e` | 2026-10-24 09:49 | 90,592 |
| `palimpzest` | 3736344 | Q1-Q6, Q8 | `bc7bb8e` | 2026-10-25 02:42 | 615 |
| `palimpzest` | 3736345 | Q7 | `bc7bb8e` | 2026-10-26 01:31 | 65,535 |

Every warm-up and measured query has status `success` in `status.tsv`. The requests are the
`vllm:request_success_total` delta between `metrics_before` and `metrics_after`. Movie Q1-Q8 use no `llm_complete`,
so the `bc7bb8e` change does not apply to the Skein and Flock runs on `1055119`.

Not copied: pilot 3712847 (Q1 only, both Flock arms). Earlier PZ job 3720921 was cancelled while pending and replaced
by the split jobs above.

## Per-query checks

Measured runtime (s, `status.tsv`), vLLM requests per query (Flock arms, from telemetry) and output rows:

| query | what | skein s | flock s | lotus s | pz s | skein reqs | flock reqs | rows skein / flock / lotus / pz |
|---|---|---|---|---|---|---|---|---|
| Q1 | filter, `LIMIT 5` | 13.9 | 104.0 | 6.0 | 0.8 | 1,964 | 2,000 | 5 / 5 / 5 / 5 |
| Q2 | filter on one movie, `LIMIT 5` | 0.4 | 5.9 | 0.3 | 11.2 | 154 | 120 | 5 / 5 / 5 / 5 |
| Q3 | filter + `COUNT(*)` | 0.5 | 5.9 | 0.3 | 2.3 | 120 | 120 | 1 / 1 / 1 / 1 |
| Q4 | positive-review ratio | 1.0 | 6.2 | 0.3 | 3.1 | 120 | 120 | 1 / 1 / 1 / 1 |
| Q5 | self-join, same sentiment, `LIMIT 10` | 296.5 | 106.0 | 83.3 | 1.2 | 65,024 | 2,032 | 10 / 10 / **0** / 10 |
| Q6 | self-join, opposite sentiment, `LIMIT 10` | 304.0 | 105.4 | 136.7 | 1.7 | 64,946 | 2,032 | 10 / 10 / 10 / 10 |
| Q7 | self-join, opposite sentiment, no limit | 300.5 | 3384.5 | 138.2 | 476.2 | 65,101 | 65,024 | 15,693 / 15,994 / 2,475 / 44,420 |
| Q8 | positive-review statistics | 1.0 | 6.3 | 0.4 | 3.7 | 120 | 120 | 2 / 2 / 2 / 2 |

## Caveats

- **Skein Q5/Q6 evaluate the full join despite `LIMIT 10`:** about 65k requests each, the same as Q7, whereas stock
  Flock stops after 2,032. That is why Skein is slower than stock Flock on Q5/Q6.
- **LOTUS joins (cascade, see shared file):**
  - Q5: thresholds (1.0, 1.0) → Search-Filter with 0 LLM calls, 404 pairs accepted from the helper alone; output 0
    rows. Probable cause of the empty result, not verified.
  - Q6: learning the cascade thresholds failed (`'InternalServerError' object has no attribute 'choices'`, litellm
    client side; all vLLM responses were 200), so it fell back to a full join (65,132 LLM calls).
  - Q7: Search-Filter, 25,964 LLM calls out of 65.5k pairs.
- PZ Q7 ran alone in 3736345 because it was expected to exceed 2 h; it took 476 s.

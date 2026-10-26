# ecomm — input provenance

SemBench e-commerce scenario at `sf_500` (product text + product images; `llm_filter`, `llm_complete`, semantic joins),
queries Q1-Q13, warm-up at `sf_50`. Shared setup: `../INPUT_PROVENANCE.md`. This file records inputs and checks only.

**Status 2026-10-26: PARTIAL.** Missing results, still pending in Slurm:

| system | missing | pending job(s) |
|---|---|---|
| `flock` | Q11 | 3736329 |
| `lotus` | Q7, Q8, Q10, Q11, Q13 | 3736332 (Q7, Q13), 3736333 (Q10), 3736334 (Q11), 3736335 (Q8) |
| `palimpzest` | Q1, Q3-Q6, Q8-Q10, Q12, Q13 | 3744826 (Q1, Q3-Q6, Q8, Q9, Q12, Q13), 3736338 (Q10) |

## Jobs used

| system | job | queries | started | vLLM requests (measured run) |
|---|---|---|---|---|
| `skein` | 3729073 | Q1-Q9, Q12, Q13 | 2026-10-23 18:47 | 13,113 |
| `skein` | 3736326 | Q10, Q11 | 2026-10-25 02:37 | 10,262 |
| `flock` | 3736327 | Q1-Q9, Q12, Q13 | 2026-10-25 02:33 | 12,469 |
| `flock` | 3736328 | Q10 | 2026-10-26 02:07 | 25,364 |
| `lotus` | 3736330 | Q1, Q3, Q5, Q6, Q12 | 2026-10-25 02:37 | 2,005 |
| `lotus` | 3736331 | Q2, Q4, Q9 | 2026-10-26 01:26 | 1,472 |
| `palimpzest` | 3736337 | Q2, Q7, Q11 | 2026-10-26 04:48 | 17,363 |

All jobs ran on sembench `bc7bb8e`. Every copied measured query has status `success`, with one exception: in 3729073,
Q10 and Q11 failed with `LLM request failed (http 0): Timeout was reached` (client default `request_timeout_ms = 60000`,
`async_llm_client.h:32`, while vLLM had up to 409 waiting requests). They were rerun in 3736326 with
`FLOCK_SEM_REQUEST_TIMEOUT_MS=600000` (env var, not recorded in `config.txt`). The 3729073 `status.tsv` keeps the
failed rows; its empty Q10/Q11 CSVs and partial Q10/Q11 telemetry were not copied.

Not copied (superseded or failed):
- 3720926 (Skein) and 3720927 (stock Flock): the warm-up of Q12 failed (`llm_complete` JSON truncated by
  `max_tokens = 16*batch`, fixed in sembench `bc7bb8e`).
- 3729074 (stock Flock): hit its 1:30 limit inside Q10, so it has no metrics.
- 3736336 (PZ): broken PZ runner (`_update_token_usage()` missing an argument since sembench `73ca30a`). Fixed in
  `palimpzest_runner.py:78`; the fix is uncommitted in sembench. 3736337 ran after the fix.

## Per-query checks

Measured runtime (s, `status.tsv`), vLLM requests per query (Flock arms, from telemetry) and output rows.
`-` = no result yet.

| query | operators | skein s | flock s | lotus s | pz s | skein reqs | flock reqs | rows skein / flock / lotus / pz |
|---|---|---|---|---|---|---|---|---|
| Q1 | text filter | 5.7 | 34.6 | 3.9 | - | 500 | 500 | 7 / 7 / 185 / - |
| Q2 | image filter + join | 95.2 | 208.9 | 206.0 | 96.5 | 500 | 500 | 13 / 17 / 7 / 29 |
| Q3 | text `llm_complete` | 5.5 | 33.7 | 4.1 | - | 500 | 500 | 500 / 500 / 500 / - |
| Q4 | image `llm_complete` + join | 58.7 | 115.4 | 124.5 | - | 294 | 294 | 294 / 294 / 294 / - |
| Q5 | text `llm_complete` | 2.8 | 14.5 | 2.9 | - | 228 | 228 | 228 / 228 / 228 / - |
| Q6 | image `llm_complete` + join | 50.8 | 100.6 | 120.7 | - | 228 | 228 | 228 / 228 / 228 / - |
| Q7 | semantic self-join (text) | 80.4 | 463.5 | - | 262.4 | 7,921 | 7,921 | 143 / 146 / - / 301 |
| Q8 | image filter + joins | 96.9 | 201.9 | - | - | 500 | 500 | 1 / 94 / - / - |
| Q9 | image filter + joins | 104.4 | 164.0 | 396.7 | - | 756 | 756 | 58 / 35 / 110 / - |
| Q10 | 5 image filters + joins | 1944.3 | 3532.5 | - | - | 6,522 | 25,364 | 1,166 / 5,648 / - / - |
| Q11 | 7 image filters + joins | 557.8 | - | - | 1267.1 | 3,740 | - | 0 / - / - / 433 |
| Q12 | image filter + `llm_complete` | 46.0 | 64.6 | 42.2 | - | 671 | 542 | 66 / 65 / 72 / - |
| Q13 | image filter + join | 101.5 | 198.0 | - | - | 500 | 500 | 15 / 14 / - / - |

## Caveats

- **Q10:** Skein sent 6,522 requests and stock Flock 25,364, for the same query. Not explained yet; check
  before comparing Q10.
- **Q11:** Skein returns 0 rows (PZ: 433). Not checked yet.
- **Q8:** Skein returns 1 row, stock Flock 94. Not checked yet.
- **LOTUS joins:** Q9 uses the cascade (Search-Filter). Q8, Q10, Q11 run exactly by design (the LOTUS query files say
  the approximate policy is not supported); Q7 is pending.

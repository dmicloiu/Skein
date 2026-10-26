# cars — input provenance

SemBench cars scenario at scale 19672 (the paper-table value; car text records + car images; `llm_filter` and
`llm_complete`), queries Q1, Q3, Q4, Q8, Q10, warm-up at scale 100. Audio queries are not run (Omni cannot transcribe).
Shared setup: `../INPUT_PROVENANCE.md`. This file records inputs and checks only.

**Status 2026-10-26: PARTIAL.** Missing: PZ Q10 (pending job 3736341).

## Jobs used

| system | job | queries | sembench | started | vLLM requests (measured run) |
|---|---|---|---|---|---|
| `skein` | 3729075 | Q1, Q3, Q4, Q8, Q10 | `bc7bb8e` | 2026-10-23 19:54 | 66,501 |
| `flock` | 3729076 | Q1, Q3, Q4, Q8, Q10 | `bc7bb8e` | 2026-10-24 09:26 | 63,067 |
| `lotus` | 3720932 | Q1, Q3, Q4, Q8, Q10 | `1055119` | 2026-10-22 14:40 | 64,006 |
| `palimpzest` | 3736339 | Q1, Q3 | `bc7bb8e` | 2026-10-25 02:41 | 19,678 |
| `palimpzest` | 3736340 | Q4, Q8 | `bc7bb8e` | 2026-10-26 01:33 | 23,419 |

Every warm-up and measured query has status `success`. LOTUS ran on `1055119`; the `bc7bb8e` change only touches the
Flock `llm_complete` setup, so it does not apply to LOTUS.

Not copied: 3720930 (Skein) and 3720931 (stock Flock). Both failed Q10 (`'car_id'`): the `llm_complete` JSON was cut
off by `max_tokens = 16*batch`, fixed in sembench `bc7bb8e`. Their other queries were rerun in 3729075/3729076 with
near-identical runtimes.

## Per-query checks

Measured runtime (s, `status.tsv`), vLLM requests per query (Flock arms, from telemetry) and output rows.

| query | operators | skein s | flock s | lotus s | pz s | skein reqs | flock reqs | rows skein / flock / lotus / pz |
|---|---|---|---|---|---|---|---|---|
| Q1 | text filter | 209.0 | 725.6 | 128.4 | 348.1 | 19,657 | 19,657 | 1,502 / 1,506 / 1,219 / 5,052 |
| Q3 | image filter + join, `LIMIT 10` | 279.0 | 430.5 | 304.0 | 2.9 | 3,765 | 2,048 | 10 / 10 / 10 / 10 |
| Q4 | text filter + join | 209.2 | 728.6 | 125.9 | 703.9 | 19,657 | 19,657 | 1 / 1 / 1 / 1 |
| Q8 | image filter, `LIMIT 100` | 273.0 | 378.8 | 895.2 | 299.9 | 3,643 | 2,048 | 100 / 100 / 100 / 100 |
| Q10 | text `llm_complete` + join | 218.8 | 874.3 | 137.6 | - | 19,779 | 19,657 | 19,657 / 19,657 / 19,657 / - |

## Caveats

- **Skein overshoots `LIMIT`:** on Q3 (`LIMIT 10`) and Q8 (`LIMIT 100`) it sent ~3.7k requests vs 2,048 for stock
  Flock. Same pattern as movie Q5/Q6, but smaller here.
- PZ Q3 takes 2.9 s vs 279-430 s for the Flock arms and LOTUS. Not checked yet.

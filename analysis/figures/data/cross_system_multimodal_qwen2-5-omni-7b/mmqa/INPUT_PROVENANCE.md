# mmqa — input provenance

SemBench MMQA scenario at `sf_200` (text + images; text and image `llm_filter`, image semantic join Q7,
`llm_complete` on text and image), queries Q1, Q2a, Q2b, Q3a, Q3f, Q4, Q5, Q6a, Q6b, Q6c, Q7, warm-up at `sf_25`.
Shared setup: `../INPUT_PROVENANCE.md`. This file records inputs and checks only.

**Status 2026-10-26: PARTIAL.** Missing: PZ for every query except Q7 (pending job 3744830).

## Jobs used

| system | job | queries | started | vLLM requests (measured run) |
|---|---|---|---|---|
| `skein` | 3729071 | all 11 | 2026-10-24 18:54 | 46,443 |
| `flock` | 3720923 | all 11 | 2026-10-24 10:15 | 46,511 |
| `lotus` | 3729072 | all 11 | 2026-10-23 16:17 | 18,820 |
| `palimpzest` | 3736343 | Q7 | 2026-10-26 01:33 | 39,996 |

All jobs ran on sembench `bc7bb8e`. Every warm-up and measured query has status `success`.

Not copied (failed or superseded):
- 3720922 (Skein): the warm-up of Q5 failed (`llm_complete` JSON cut off by `max_tokens = 16*batch`, fixed in `bc7bb8e`).
- 3720924 (LOTUS): `sentence-transformers/clip-ViT-B-32` was missing from the offline HF cache.
- 3736342 (PZ): the warm-up of Q5 failed in PZ `aggregate.py:633` (`'NoneType' object is not subscriptable`). Rerun as 3744830.

## Per-query checks

Measured runtime (s, `status.tsv`), vLLM requests per query (Flock arms, from telemetry) and output rows.

| query | operators | skein s | flock s | lotus s | pz s | skein reqs | flock reqs | rows skein / flock / lotus / pz |
|---|---|---|---|---|---|---|---|---|
| Q1 | `llm_complete` + join | 1.9 | 14.1 | 1.1 | - | 200 | 200 | 1 / 1 / 1 / - |
| Q2a | text + image filter, join | 32.2 | 241.6 | 59.1 | - | 2,600 | 2,600 | 18 / 86 / 206 / - |
| Q2b | text + image filter, `llm_complete`, join | 22.3 | 225.9 | 53.6 | - | 2,618 | 2,686 | 18 / 86 / 209 / - |
| Q3a | text filter | 1.9 | 10.9 | 0.9 | - | 200 | 200 | 15 / 15 / 17 / - |
| Q3f | text filter | 1.8 | 10.9 | 0.8 | - | 200 | 200 | 3 / 3 / 4 / - |
| Q4 | `llm_complete` | 0.3 | 1.5 | 0.2 | - | 24 | 24 | 20 / 20 / 25 / - |
| Q5 | `llm_complete` | 0.3 | 0.2 | 0.1 | - | 1 | 1 | 1 / 1 / 1 / - |
| Q6a | text filter | 1.3 | 12.0 | 0.4 | - | 200 | 200 | 4 / 4 / 1 / - |
| Q6b | text filter | 1.4 | 11.8 | 0.4 | - | 200 | 200 | 5 / 5 / 1 / - |
| Q6c | text filter | 1.4 | 11.5 | 0.4 | - | 200 | 200 | 5 / 4 / 16 / - |
| Q7 | image semantic join (40k pairs) | 723.2 | 4112.9 | 624.3 | 1093.3 | 40,000 | 40,000 | 73 / 1,377 / 23,766 / 1,072 |

## Caveats

- Q2a/Q2b/Q7: stock Flock returns many more rows than Skein on the image queries. This matches the image-label bias in
  the shared file's appendix, which was replayed for animals only (not for MMQA).
- **LOTUS joins use the cascade:** Q2a/Q2b Map-Search-Filter (~1.5k LLM calls each); Q7 Search-Filter with 5,191 LLM
  calls out of 40k pairs and 23,418 pairs accepted from CLIP alone, hence its 23,766 rows.

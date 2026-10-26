# animals — input provenance

SemBench animals scenario at `sf_200` (200 images; image `llm_filter` only), queries Q1, Q3, Q7, Q10, warm-up at `sf_20`.
Audio queries are not run (Omni cannot transcribe). Shared setup: `../INPUT_PROVENANCE.md`.
This file records inputs and checks only.

## Jobs used

| system | job | queries | sembench | started | vLLM requests (measured run) |
|---|---|---|---|---|---|
| `skein` | 3720782 | Q1, Q3, Q7, Q10 | `1055119` | 2026-10-22 13:44 | 1,000 |
| `flock` | 3720915 | Q1, Q3, Q7, Q10 | `1055119` | 2026-10-22 13:52 | 1,000 |
| `lotus` | 3720916 | Q1, Q3, Q7, Q10 | `1055119` | 2026-10-22 13:57 | 1,000 |
| `palimpzest` | 3729070 | Q1, Q3, Q7, Q10 | `bc7bb8e` | 2026-10-23 18:35 | 1,000 |

Every warm-up and measured query has status `success` in `status.tsv`. The requests are the
`vllm:request_success_total` delta between `metrics_before` and `metrics_after`. The animals queries use no
`llm_complete`, so the `bc7bb8e` change does not apply to the runs on `1055119`.

**Skein 3720782 ran 3 reps** (each with its own cold vLLM). Only rep 1 is copied, to match the 1 rep of the other
systems; `status.tsv` was trimmed to its header and the rep 1 rows. Reps 2-3 are in
`analysis/results/cross_system_qwen2-5-omni-7b/3720782/` (measured s, rep 2 / rep 3: Q1 11.36 / 11.28,
Q3 1.84 / 1.89, Q7 11.89 / 11.86, Q10 1.98 / 1.93).

Not copied: pilot 3712849 (Q1 only, both Flock arms), and PZ 3720917. 3720917 hit its time limit after sending vLLM
0 requests: the PZ optimizer looped forever because Omni was not marked as a vision model (fixed by the local PZ
patch, see shared file).

## Per-query checks

Measured runtime (s, `status.tsv`), vLLM requests per query (Flock arms, from telemetry) and output vs gold
(`sembench/files/animals/raw_results/ground_truth/`):

| query | what | skein s | flock s | lotus s | pz s | reqs (skein = flock) | gold | skein | flock | lotus | pz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | count zebra images | 11.4 | 36.6 | 59.8 | 12.4 | 200 | 11 | 10 | 124 | 8 | 20 |
| Q3 | city with most zebras | 1.8 | 15.1 | 46.8 | 2.1 | 200 | Kisumu, Eldoret (tie) | Kisumu | Nairobi | Kisumu | Mombasa, Nairobi |
| Q7 | cities with zebras and impalas | 11.8 | 39.0 | 98.1 | 14.6 | 400 | 5 cities | same 5 | same 5 | same 5 | same 5 |
| Q10 | (city, station) with most zebras | 2.0 | 15.1 | 46.9 | 2.5 | 200 | Kisumu/Station_A, Eldoret/Station_A (tie) | Kisumu/Station_A | Mombasa/Station_B | Eldoret/Station_A | Mombasa/Station_B |

## Caveats

- Stock Flock's Q1 count of 124 vs gold 11 is the image-label bias documented in the shared file's appendix.
- Q3 and Q10 run the same zebra filter over the same images as Q1, on the same vLLM server, so they may benefit from
  its cache. Every system runs the queries in the same order (Q1, Q3, Q7, Q10). Not measured separately.

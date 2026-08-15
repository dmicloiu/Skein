# Provenance — Llama-3.1-8B cross-system filter (Q101)

Unlike the single-session Qwen table in `eb016d6d`, this table **spans two jobs**. That was
an explicit cost trade, not an oversight.

| arms | job | note |
|---|---|---|
| the 10 flock arms + `lotus`, 3 reps each (33 files) | `3082227` | job hit TIMEOUT at 01:30 with 34/36 runs done |
| `palimpzest`, 3 reps (3 files) | `3083163` | re-run alone with `N_REPS=3` |

**Why the split.** `3082227` was sized at `--time=01:30:00` from a ×1.3 estimate of the Qwen
counterpart; the measured cost is ×1.73 (2.62 min/run vs 1.51), so it ran out of wall clock
during `palimpzest_rep2`. All 34 completed runs are genuine: the kill landed 8 lines into
`palimpzest_rep2` before it wrote anything, so no stale `cp -f` republish of a previous
arm's `metrics/<system>.json` occurred.

**What was deliberately excluded.** Every `*palimpzest*` artefact from `3082227` — including
its valid `palimpzest_rep1_palimpzest.json` — was left out of this directory. Keeping it
would have collided with `3083163`'s own rep1 and produced a 4-rep PZ arm assembled from two
sessions. The excluded files remain under
`analysis/results/cross_system_llama-3-1-8b-instruct/3082227/`.

Consequence: the `vllm-*.log` files here are timestamp-named and therefore **35 of them come
from `3082227`**, two of which belong to the excluded PZ rep1/rep2 servers. They are kept
because they carry the evidence for the runaway-generation diagnosis below. Do not infer the
run count from the log count; infer it from the 36 result JSONs.

## Gates applied to all 36 files

`keys == ['Q101']`, `status == 'success'`, and for the 10 flock arms
`extra.run_label == '<arm>_rep<n>'`. LOTUS and PZ carry `extra == {}` and so have no
`run_label` — the same convention as the locked Qwen files — and were confirmed against
their per-rep `✅ <s>, <n> rows` log lines instead. All 36 pass.

`row_count` is **not** 2000: it is the number of rows passing the filter (Llama 1167–1995,
Qwen 601–1642). It is a selectivity signal, not a gate.

Every server in both jobs reports
`model='.../models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659'`.

Input is `sf_2000/Reviews.csv`, git-tracked and byte-identical to what the Qwen counterpart
(`3035119`) measured, so this table **is** cross-family comparable — unlike the scale-out
data, see `../dp_scaling_json_llama/INPUT_PROVENANCE.md`.

## palimpzest runs away, occasionally

Under Llama, PZ sometimes emits one request that never produces a stop token and generates
to the limit (~14k tokens, ~170 tok/s single-stream, +86s wall). `temperature` is 0.6 —
inherited from Llama's `generation_config.json`, which PZ never pins — so it is
probabilistic. flock pins `temperature = 0.0` and caps `max_tokens`, and LOTUS is
length-capped, so neither can suffer it.

Measured 2 of 12 reps across three jobs. The runaway reps are the 3.0× mean; the clean mode
is 1.2× Qwen. **Report the clean mode as throughput and the runaway separately as a
robustness result** — a 3-rep mean is bimodal and moves with rep count.

Frequency arm (8 more reps at the inherited 0.6) is in
`../cross_system_sem_filter_llama_pz_temp06/`.

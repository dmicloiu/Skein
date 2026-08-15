# Llama-3.1-8B slim-head R-sweep — positional diagnostic

Companion to `sem_filter_prompt_slim_json/` (Qwen). Answers one question: does
the slim head's recall gain with R come from the *absence* of a lost-in-the-
middle penalty, or from a penalty that is outweighed by something else?

Only the `slim/` head was run. There is no `full/` here — see "What is missing".

## The job

```
job 3089338   COMPLETED 10:55   meta-llama/Llama-3.1-8B-Instruct (snapshot 0e9e39f2…c89659)
cd ~/projects/sembench
QUERY=101 N_REPS=1 VERDICT_DUMP=1 MODEL="meta-llama/Llama-3.1-8B-Instruct" \
  ARMS="flock_op_slim_r1 flock_op_slim_r2 flock_op_slim_r4 flock_op_slim_r8 flock_op_slim_r16 flock_op_slim_r32" \
  sbatch slurm/cross_system_analysis_clariden.sh
```

Six cold boots, one arm each, all six on the campaign snapshot. Scored against
`sf_2000/Reviews.csv`, which is git-tracked and byte-identical to what the Qwen
arms scored against — so quality comparisons across the two families are sound
(unlike the DP absolutes, see `../dp_scaling_json_llama/INPUT_PROVENANCE.md`).

**Do not cite timings from this directory.** `VERDICT_DUMP=1` writes during the
run. The harness JSONs carry elapsed times; they are contaminated by the dump.
This is also why no `result_r<R>_rep<N>.json` was imported, and why
`diagnose_rsweep.py` leaves its rows/s and tok/s columns blank here — that is
the intended state, not missing data.

## Renamed on import

The cross-system harness names artefacts by ARM; `diagnose_rsweep.py` reads the
R-based layout the `sem_filter_rsweep` harness produces. Import mapped one onto
the other so both families are analysed by one code path:

```
verdicts_flock_op_slim_r<R>_rep1.jsonl          -> verdicts_r<R>_rep1.jsonl
metrics_{before,after}_flock_op_slim_r<R>_rep1.txt
                                                -> metrics_{before,after}_r<R>_ep0_rep1.txt
```

Everything else (harness metrics JSON, run/warmup/vLLM logs) is verbatim under
its original arm name, so every file still traces to the arm that produced it.

Reproduce:

```
/usr/bin/python3.11 analysis/diagnose_rsweep.py \
  analysis/figures/data/sem_filter_prompt_slim_llama/slim \
  --data ../sembench/files/movie/data/sf_2000/Reviews.csv
```

(login-node `python3` is 3.6 and cannot parse the script.) The derived
`rsweep_{quality,position}.csv` are not committed, matching the Qwen sibling.

## Gates

- 6/6 arms present; every dump has exactly 2000 lines and 2000 distinct ids
- `max(pos) == R-1` on every arm (1→0, 2→1, 4→3, 8→7, 16→15, 32→31)
- 2000 ids rules out a republished 20-row warmup, one of the four ways a failed
  arm fakes success (see the extract-R≥4 post-mortem)
- all 6 vLLM boots report snapshot `0e9e39f2…c89659`

## What is missing

- **One rep, not three.** Qwen ran 3. Qwen's per-rep slim ramp spans 0.130–0.145,
  so rep noise is roughly ±0.015 — smaller than the cross-family gap reported in
  the write-up, but any claim tighter than that needs reps 2 and 3.
- **No `full/` head.** The Qwen full-vs-slim contrast has no Llama counterpart
  here. Note the Llama full head is near-degenerate accept-all on Q101
  (~1994/2000 rows at recall 1.000), so a Llama full arm may measure that
  degeneracy rather than a positional effect.
- **`ramp` is not computed by `diagnose_rsweep.py`.** The write-up's Qwen ramp
  figures (+0.377 full, +0.158 slim) come from a definition that is not in this
  repo: `missrate_b4 - missrate_b0` gives +0.370 / +0.130 per rep, +0.373 /
  +0.138 pooled, and an OLS slope over the five buckets gives +0.411 for full.
  Any cross-family ramp comparison must first pin that definition down and
  recompute BOTH families with it.

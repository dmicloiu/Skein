# DP input provenance — Llama arms do NOT share Qwen's sf_300000 sample

The Llama-3.1-8B DP/TP arms in this directory (`llama_unified_rep*`,
`llama_grid_rep*`, `llama_caps_rep*`, `llama_caps128_rep*`) were measured on a
**regenerated** `sf_300000`, not on the file the Qwen arms used. Read this
before comparing any absolute `rows_s` / `tok_s` figure across the two families.

## What happened

The original `files/movie/data/sf_300000/` was deleted on 2026-08-14 during a
working-tree cleanup. It was never recoverable:

- never tracked in git (`git log --all -- files/movie/data/sf_300000` is empty)
- absent from `/iopsstor` and `/capstor`
- not in `movie_database.duckdb` (that file holds only `reviews_2000` / `movies_2000`)
- not reconstructible from committed artefacts: `samples_*_client.csv` is CPU
  telemetry (`epoch,utime_ticks,stime_ticks,nthreads`) and `verdicts_*.jsonl`
  covers the 2000 `GOLD_DATA` rows as `{"id","pos","v"}` with no review text

**`generate_data.py` is not deterministic** despite `random_state=42` on all
three sample calls (lines 229, 249, 258). Verified empirically: regenerating the
git-tracked `sf_2000` changed 382 of ~2000 rows against the committed file. So
regeneration cannot reproduce the deleted sample.

## The regenerated file

```
cd ~/projects/sembench
python src/scenario/movie/preparation/generate_data.py files/movie/source_data 300000 --top-n 1400
```

`--top-n 1400` is required: the default 200 caps at 80,351 reviews, and reaching
300,000 needs the top 1,085 movies of the 67,607 in the source (1,375,738 reviews
after `dropna(subset=['reviewText'])`).

```
Reviews.csv  sha256 5d64fdfaa2071975bd5370d56b41671c50bf5c73ae100a2700819f53e7823ca4  300000 rows
Movies.csv   sha256 0d2aa6ad5c744b1ef699dbc362a64ab6d314e49430dacf19472294298a396fb8    1236 rows
```

`llama_caps128_rep{1,2,3}` (jobs `3088915/6/7`, the N=4 cap=128 below-knee cell)
ran a day after the other 33 arms, so the input was re-checked at submit time:
`Reviews.csv` still hashes to `5d64fdfa…3ca4` with an unchanged mtime, so all 36
arms in this directory share one input file and are mutually comparable.

The measured slice is the first `ROWS=32000` rows: mean `reviewText` length
139.7 chars (min 1, max 317). The corresponding statistic for the deleted file is
unknown, so the size of the divergence cannot be quantified.

## What this does and does not invalidate

Still valid — these are ratios measured **within** the Llama family on one input,
so a different sample shifts both sides equally:

- DP per-GPU throughput ~linear, TP per-GPU sub-linear, DP > TP
- replicas beat shards at a fixed 4-GPU budget
- the in-flight cap knee near 128·N

Not valid — do not compute these across families:

- absolute `rows_s`, `tok_s`, or tok/row against the Qwen numbers in
  `dp_scaling_json/`. This is a stronger version of the caveat that already
  applies from the tokenizer difference alone.

`GOLD_DATA` is unaffected: it defaults to `sf_2000/Reviews.csv`, which is
git-tracked and byte-identical to what the Qwen arms scored against, so F1 and
verdict comparisons across families remain sound.

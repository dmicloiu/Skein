# Skein proxy filter experiment

`PhysicalSemFilter` can run an optional two-model cascade. It sends every row
to a proxy model that generates a numeric score in `[0, 1]`. These are
self-reported scores, not calibrated model probabilities. Scores at or below the
negative threshold resolve to false; scores at or above the positive threshold
resolve to true; remaining rows go to the main model. The ordinary filter and
the unkeyed semantic join rewrite share this physical operator. The feature is
off when `semantic_filter_proxy_model` is empty.

The proxy must be registered with `CREATE MODEL` and served at its own
OpenAI-compatible **chat-completions** endpoint. The main model continues to
use `semantic_endpoints`. Set the proxy's endpoint to the full URL, such as
`http://127.0.0.1:8001/v1/chat/completions`. Both servers must support the
`response_format=json_schema` request format. Proxy score requests have a
numeric `items` schema; main requests retain the boolean schema.

```sql
SET semantic_filter_proxy_model = 'Qwen/Qwen2.5-0.5B-Instruct';
SET semantic_filter_proxy_endpoint = 'http://127.0.0.1:8001/v1/chat/completions';
SET semantic_filter_proxy_negative_threshold = 0.05;
SET semantic_filter_proxy_positive_threshold = 0.95;
```

Thresholds default to `-1` and `2`, which forwards every row to the main
model. This is useful for collecting scores for calibration, but it costs two
model calls per row. Thresholds set inside `[0, 1]` make the cascade
approximate: proxy errors can change the query result. `WHERE NOT llm_filter`
uses the same scores before inversion.

## Calibrate

Run the same query once with the proxy disabled, then once with the proxy
enabled and default thresholds. Set `FLOCK_VERDICT_DUMP` to a different JSONL
path in each process. Use `SET threads=1` and identical input order so the
query-local row IDs match. Verdict dumps include a row fingerprint; the tool
rejects mismatched rows rather than calibrating on unrelated examples. Then run:

```bash
python3 analysis/calibrate_proxy_filter.py \
  --main-dump /tmp/main_verdicts.jsonl \
  --proxy-dump /tmp/proxy_verdicts.jsonl \
  --precision-target 0.95 --recall-target 0.95
```

The script selects thresholds that minimize main-model rows while meeting
empirical precision and recall targets **relative to the main model** on the
training subset (default 20%). It reports precision and recall on the remaining
held-out rows. Also evaluate the resulting query against the benchmark's gold
labels. This calibration is descriptive; it provides no statistical guarantee
on unseen rows.

This follows the broad cascade pattern used by LOTUS, but uses a different
proxy signal and calibration method. LOTUS filter uses helper-LM log
probabilities (or embedding scores) and importance-sampled threshold learning;
LOTUS join uses embedding similarity and chooses a retrieval plan. Skein here
scores every candidate pair with the small LM and calibrates thresholds
offline. Compare observed quality, latency, and token use; do not describe the
methods as algorithmically identical.

## Compare filter and join

SemBench's movie queries `Q101` (filter), `Q107` (keyed pair filter), and
`Q108` (unkeyed semantic join) exercise the shared path. The movie runner
accepts `FLOCK_PROXY_MODEL`, `FLOCK_PROXY_ENDPOINT`,
`FLOCK_PROXY_NEG_THRESHOLD`, and `FLOCK_PROXY_POS_THRESHOLD` when
`FLOCK_SEMANTIC_REWRITE=on`. Run each query with the proxy disabled and enabled
under the same main model, data scale, prompt, batch size, and endpoint cap.
Use the existing SemBench result metrics for end-to-end execution time and
gold-set precision/recall/F1. The verdict dump adds `stage=proxy|main`, so it
also shows the fraction sent to the main model. Keep calibration runs out of
timed measurements; dump I/O perturbs latency.

The animals Skein runner reuses the same setup, so its existing semantic
filter queries also accept these variables. The current animals query set has
no Skein semantic join query; use the movie pair queries for the join arm.

## Clariden pilot

The existing `sembench/slurm/cross_system_analysis_clariden.sh` has an optional
`flock_proxy_r32` arm. With `PROXY_MODEL` set it starts a second vLLM server on
port 8001 for **every** arm, including the baseline. It allocates 0.65 of the
GPU to the main server and 0.20 to the proxy by default; adjust
`MAIN_GPU_UTIL` and `PROXY_GPU_UTIL` if the models cannot start together.
Both model weights must already be in the Clariden cache, and the pushed
release extension must be rebuilt there before submission.

From the SemBench checkout, first collect untimed verdicts for one query:

```bash
PROXY_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
ARMS="flock_op_r32 flock_proxy_r32" QUERY=101 N_REPS=1 VERDICT_DUMP=1 \
sbatch slurm/cross_system_analysis_clariden.sh
```

Use the job log's `OUT=` directory with the calibration command above, passing
`verdicts_flock_op_r32_rep1.jsonl` as `--main-dump` and
`verdicts_flock_proxy_r32_rep1.jsonl` as `--proxy-dump`. The default thresholds
`-1` and `2` forward all rows to the main model during this collection. Once
the held-out quality is acceptable, set `NEG_THRESHOLD` and `POS_THRESHOLD` to
the numeric values printed by the calibration tool, then run a timed comparison
with `VERDICT_DUMP=0`:

```bash
PROXY_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
PROXY_NEG_THRESHOLD="$NEG_THRESHOLD" PROXY_POS_THRESHOLD="$POS_THRESHOLD" \
ARMS="flock_op_r32 flock_proxy_r32" QUERY=101 N_REPS=3 VERDICT_DUMP=0 \
sbatch slurm/cross_system_analysis_clariden.sh
```

Repeat calibration and timing for `QUERY=108` and then `QUERY=107`; each
predicate has its own score distribution and thresholds. The script saves
main-server metrics and `proxy_metrics_{before,after}_*.txt` separately. For
join warm-up, use `WARM_SCALE=2000 WARM_QUERY=102` so the proxy and main JSON
schemas compile before the timed query. The first pilot should use one
repetition and inspect logs and result accuracy before the three-repetition
runs.

## Embedding proxy (v2): `embedding` and `lotus` kinds

`semantic_filter_proxy_kind` selects the proxy. `llm` (default) is the
LLM-scored cascade above. The two embedding kinds score each candidate with a
CLIP-style embedding model served at an OpenAI-compatible **`/v1/embeddings`**
endpoint (e.g. vLLM `--runner pooling` serving `clip-ViT-B-32/0_CLIPModel`).
`semantic_filter_proxy_model` is then the served model id, sent verbatim (no
`CREATE MODEL`), and `semantic_filter_proxy_endpoint` the full embeddings URL.

```sql
SET semantic_filter_proxy_kind = 'embedding';          -- or 'lotus'
SET semantic_filter_proxy_model = 'clip';
SET semantic_filter_proxy_endpoint = 'http://127.0.0.1:8001/v1/embeddings';
SET semantic_filter_proxy_template = 'the logo of {}';  -- '{}' = the row's text values
```

**Applicability.** The filter needs exactly one `image` context column (plus
any number of text columns). Negated filters (`NOT llm_filter`) and audio
columns run exactly, since a pruned row would be emitted. Inputs with fewer than
`min_rows` rows also run exactly, as do failed or timed-out embedding calls. Each
case prints a `[flock proxy]` line on stderr.

**How it runs.** The operator buffers its whole input. It then embeds each
distinct rendered text and each distinct image once (text in batches of 64,
images one request each as base64 data URLs; vLLM composites RGBA on white). A
pair's score is the cosine similarity. Rows whose context values are identical
form one *unit*: one main-model call, whose verdict covers all of the unit's
rows. Units are grouped by rendered text (for example, one airline). Main-model
requests then stream as decisions are made, and every unit that is never asked
is rejected.

* `embedding`: **reject-only** per-group top-k.
  * Each group sends its top `k_min` units, or `ceil(k_frac * group size)` if
    larger.
  * A TRUE at rank r extends that group's window to r + `gap`.
  * An audit samples `audit_near` of ranks (k, 2k] and `audit_far` of the ranks
    beyond that, seeded by `seed`. An audited TRUE at depth d raises k to
    d + `gap` for **every** group.
  * Only the main model can accept a row.
* `lotus` (ablation): a port of LOTUS's join cascade (`importance_sampling` +
  `learn_cascade_thresholds`; the port reproduces LOTUS's thresholds on
  reference inputs, see `proxy_policy_test.cpp`).
  * Samples are pairs drawn with replacement from the first
    `lotus_sample_range` pairs. Pair order is the order of the groups' first
    appearance, then descending score inside each group.
  * The main model answers the sampled units, and the learned thresholds then
    decide the rest: score >= tau+ is **accepted without the main model**,
    score <= tau- is rejected, and anything in between goes to the main model.
    A sampled verdict in that middle band is reused.
  * Differences from LOTUS: no Map-Search-Filter plan (LOTUS's alternative
    plan, which needs an LLM map step); scores are CLIP cosine clipped to
    [0, 1]; main-model calls are deduplicated by unit.

| option | default |
|---|---|
| `semantic_filter_proxy_kind` | `llm` |
| `semantic_filter_proxy_template` | `''` (raw text values; no text column -> the filter prompt) |
| `semantic_filter_proxy_k_min` / `_k_frac` / `_gap` | 5 / 0.0 / 2 |
| `semantic_filter_proxy_audit_near` / `_audit_far` | 0.25 / 0.01 |
| `semantic_filter_proxy_min_rows` / `_seed` | 1000 / 42 |
| `semantic_filter_proxy_lotus_{recall,precision,failure_prob}` | 0.8 / 0.8 / 0.2 (sembench LOTUS runner) |
| `semantic_filter_proxy_lotus_{sample_frac,sample_range}` | 0.1 / 200 (LOTUS defaults) |

**Run log.** Each proxied operator prints `[flock proxy] kind=... rows=...
units=... groups=... images=... embed_s=...` after planning, and
`[flock proxy] stats {...}` (sends per reason, audit hits, final k; tau+/tau-
and auto-accept counts for `lotus`) when it finishes. Report the main-model
fraction from these, not only the end-to-end time.

**No recall guarantee.** With very rare positives (mmqa Q7: 5 in 40k pairs) a
sample cannot certify recall. Treat k and the audit as engineering choices,
validate them on queries they were not tuned on, and report the audit
outcome.

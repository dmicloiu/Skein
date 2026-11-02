# Skein proxy filter experiment

`PhysicalSemFilter` can run an optional two-model cascade. It sends every row
to a proxy model that returns a probability in `[0, 1]`. Scores at or below the
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
query-local row IDs match. Then run:

```bash
python3 analysis/calibrate_proxy_filter.py \
  --main-dump /tmp/main_verdicts.jsonl \
  --proxy-dump /tmp/proxy_verdicts.jsonl \
  --precision-target 0.95 --recall-target 0.95
```

The script selects thresholds that minimize main-model rows while meeting
empirical precision and recall targets **relative to the main model** on the
calibration data. Evaluate those thresholds on a separate held-out set against
the benchmark's gold labels. This calibration is descriptive; it provides no
statistical guarantee on unseen rows.

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

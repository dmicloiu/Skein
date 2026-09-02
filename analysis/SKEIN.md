# Skein

Skein is a research fork of [Flock](https://github.com/dais-polymtl/flock) that
re-expresses Flock's *synchronous scalar* semantic UDFs (`llm_filter`,
`llm_complete`, …) as **asynchronous DuckDB physical operators** over a
**co-located** vLLM language model. Request dispatch is driven by the serving
platform's saturation point rather than by the query's morsel/thread layout, so
the engine can saturate and scale out over the local model.

This document summarises Skein's architecture and the results established in the
MSc thesis *"Skein: a DuckDB extension for topology-aware semantic query
processing over a co-located LLM"* (D. Micloiu, ETH Systems Group, 2026). It
replaces the ad-hoc analysis notes that backed the paper; those notes and the
full raw run data are preserved at the **`thesis-submitted`** git tag
(`git checkout thesis-submitted`).

## Architecture

Skein is a layered design. Each layer is a distinct component; only the
execution layer touches DuckDB's operator machinery.

- **Network layer — `AsyncLLMClient`** (`src/functions/operator/`). Owns a single
  libcurl *multi* handle driven by one dedicated I/O thread, holding `N`
  requests in flight without a proportional number of OS threads. Caller
  (DuckDB worker) threads enqueue compiled request bodies; the I/O thread drives
  the transfers and fires callbacks. Handles early/late request cancellation.
- **Routing layer — `EndpointRouter`** (`src/functions/operator/`). Makes the
  engine *topology-aware*: chooses which vLLM endpoint each request targets over
  a fleet, via a pluggable strategy (`single`, `round_robin`,
  `sticky_by_prefix`, `least_loaded`). Reconfigurable at query boundaries.
- **Execution layer — `SemanticOperatorBase`** (`src/functions/operator/`). The
  base of Skein's physical operators (`PhysicalSemFilter`, `PhysicalSemExtract`).
  A pipeline breaker that coalesces rows into batched, non-blocking requests and
  holds a bounded in-flight population, decoupling LLM concurrency from DuckDB's
  morsel/thread parallelism.
- **Configuration layer** (`src/runtime/`, `src/core/config/`). A
  database-lifetime `ExtensionState` (in DuckDB's object cache) owning the shared
  client and router, a native `SET semantic_*` surface for the tunable knobs, and
  per-model calibration persisted in the model catalogue. Resolves the effective
  parameters via `model_args → SET → built-in default` precedence.

**The switch.** The optimiser rewrite (gated on `SET semantic_rewrite_enabled`)
swaps Flock's scalar UDF for Skein's physical operator, leaving prompt building
and semantics byte-identical. This single flag enables a fair Skein-vs-Flock
comparison; with it off, the system runs exactly as upstream Flock.

## Contributions (thesis)

1. **Asynchronous execution model** decoupling LLM concurrency from DuckDB's
   morsel-driven query processing.
2. **Tunable, per-model-calibrated configuration layer** that sizes execution to
   the serving platform's saturation point.
3. **Topology-aware routing layer** for horizontal scale-out across a fleet of
   endpoints.
4. **Systematic evaluation** over a co-located language model.

## Key results

- **Skein vs Flock (dispatch only):** ~7.5–8× faster at a single morsel, ~2× on
  multi-morsel workloads; F₁ identical by construction (byte-identical prompts).
- **Configuration layer:** scales engine efficiency ~2.82× across four
  tensor-parallel shards (cap sized to `128·G` per GPU), versus ~1.24× for Sema,
  which exposes only a batch-size knob.
- **Scale-out (fixed 4-GPU budget):** replication beats sharding ~1.56× in
  prefill-heavy settings; the advantage inverts only for a large model under a
  decode-heavy, KV-cache-pressured workload.
- **Cross-system:** on semantic *filter*, Skein sits on the throughput–quality
  frontier (best processed tok/s at best F₁); on semantic *extract*, quality is
  prompt-dependent and Skein matches or trails LOTUS/Palimpzest.
- **Generalisation:** the engine and scale-out laws hold across a 10× model-size
  range (7B/32B/72B), workload shapes up to the KV boundary, a two-node
  deployment, and a second model family (Llama-3.1-8B); prompt-quality behaviour
  is family-specific.

## Repository map

- `src/` — the DuckDB extension. Skein's additions live mainly under
  `src/functions/operator/` (async operators, client, router),
  `src/runtime/` and `src/core/config/` (settings + wiring); the rest is
  upstream Flock. The repo stays mergeable with upstream.
- `analysis/` — research artifacts:
  - `plot_*.py` — figure generators (read the committed CSVs under
    `figures/data/`).
  - `summarize_*.py` — reduce raw runs to the summary CSVs.
  - `slurm/` — cluster run scripts.
  - `figures/` — the paper figures (`.pdf`) and their input data
    (`figures/data/`).

## Reproducibility

- **Redraw a figure** from committed data (no unpacking needed):
  `python analysis/plot_dp_scaling.py`, `plot_cross_system_frontier.py`, etc.
  The summary CSVs and the small per-rep JSON/`verdicts_*.jsonl` inputs are
  committed under `analysis/figures/data/`.
- **Full raw run data** (payload dumps, per-run logs) and the original,
  number-by-number analysis notes are frozen at the **`thesis-submitted`** tag:
  `git checkout thesis-submitted`. They were removed from `main` to keep the
  working tree lean for continued development.
- All measurements: vLLM 0.22.0 (V1 engine), NVIDIA GH200 (Clariden).

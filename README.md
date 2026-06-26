# Semantic Operators DuckDB Extension

DuckDB extension for **semantic operators**. Built on top of [FLOCK](https://github.com/dais-polymtl/flock).

This extension is developed as part of an MSc thesis and is benchmarked against LOTUS and Palimpzest via the `sembench` harness.

## Upstream baseline

This repository was forked from upstream FLOCK at commit [`ce3415f`](https://github.com/dais-polymtl/flock/commit/ce3415f) (DuckDB v1.5.1). Updates from upstream can be pulled via the `upstream` git remote (`git@github.com:dais-polymtl/flock.git`).

## CONTRIBUTION

An async, Tier-3 **semantic-operator runtime**. It re-expresses FLOCK's scalar `llm_filter` as a custom DuckDB `PhysicalOperator` that **decouples LLM request concurrency from DuckDB's morsel/thread count**, so a small worker pool can saturate a vLLM server. The SQL surface is unchanged: a post-optimizer pass rewrites `WHERE llm_filter(...)` into the operator (toggle with `SET semantic_rewrite_enabled`), and a `semantic_*` settings surface configures the endpoints, routing strategy, in-flight cap, and per-prompt batch size.

The additions fall in three layers — a process-lifetime async runtime, the Tier-3 operators that use it, and the test/analysis harness that exercises it.

## REPOSITORY STRUCTURE

Only the files this fork introduces are listed (public headers live under `src/include/flock/...`; implementations under `src/...`).

```
src/runtime/                          process-lifetime async LLM runtime
  llm_client.{h,cpp}                  ILLMClient interface + shared response types
  async_llm_client.{h,cpp}            non-blocking libcurl-multi HTTP client (one IO thread, generation-based cancel)
  endpoint_router.{h,cpp}             endpoint pick: single | round_robin | sticky_by_prefix | least_loaded
  extension_state.{h,cpp}             per-DatabaseInstance singleton owning the client + router
  semantic_settings.{h,cpp}           `semantic_*` SET surface + ResolveSemanticParams (model_args -> SET -> default)
  pending_request.h                   per-row request bundle (row id + captured row data)

src/functions/operator/               Tier-3 semantic operators
  semantic_operator_base.{hpp,cpp}    async pipeline-breaker engine: cap-blocking backpressure, row coalescing,
                                      completion-order drain (the morsel-decoupling core)
  semantic_filter.{hpp,cpp}           LogicalSemFilter + CreatePlan + PhysicalSemFilter (the filter operator)
  optimizer_rewrite.{hpp,cpp}         OptimizerExtension: rewrites WHERE-conjunct llm_filter -> LogicalSemFilter

test/runtime/                         runtime unit + live-vLLM tests
  async_llm_client_test.cpp           client unit tests (incl. drain-on-cancel)
  endpoint_router_test.cpp            routing-strategy unit tests
  extension_state_test.cpp            singleton lifetime
  semantic_config_test.cpp            SET surface + CREATE MODEL params + ResolveSemanticParams
  mock_vllm_server.{h,cpp}            in-process OpenAI-compatible mock server
  generate_vllm_payloads.py           E5-shape (rows-per-prompt) payload generator
  {async_llm_client,endpoint_router,sem_filter}_vllm_integration.cpp   standalone drivers against a live vLLM

test/functions/operator/              operator unit + e2e tests
  semantic_operator_base_test.cpp     scheduler-free threading test (TSan-clean) of the engine
  semantic_filter_test.cpp            operator component tests (render parity, parse, bindings, CreatePlan)
  optimizer_rewrite_test.cpp          rewrite scope: handled patterns + fallback no-ops
  sem_filter_e2e_test.cpp             full DuckDB pipeline, operator vs scalar parity via one mock server
  fake_llm_client.h                   deterministic in-process ILLMClient

analysis/                             experiment drivers, summaries, figures
  slurm/                              Clariden (GH200) SLURM drivers (bring up a cold vLLM fleet in the NGC container)
    build_setup_clariden.sh           one-shot build of an integration driver inside the container
    sem_filter_ab_clariden.sh         operator vs scalar A/B (threads sweep)
    router_analysis_clariden.sh       routing-strategy validation (cold fleet per strategy)
    router_sticky_sweep_clariden.sh   sticky-by-prefix prefix-cache sweep
    network_analysis_clariden.sh      AsyncLLMClient vs sembench's reference driver on identical wire bytes
  plot_sem_filter_ab.py               operator-vs-scalar figure
  plot_router_experiment.py           router-strategy figures
  asyncllmclient_vs_vllm.py           client-vs-driver parity figure
  summarize_morsel_ab.py              operator-vs-scalar runs -> CSV
  summarize_cross_system.py           cross-system runs -> CSV + Markdown table
  figures/data/{sem_filter_ab,router_analysis,sticky_routing_analysis,network_analysis,cross_system}/   raw results
```

## Experiments

The correctness suites run with no GPU; the throughput experiments need a local, OpenAI-compatible vLLM. The thesis runs the latter on Clariden (GH200) via the SLURM drivers in `analysis/slurm/`, which start a cold vLLM fleet in the NGC container. Build a driver first with `analysis/slurm/build_setup_clariden.sh`. Raw outputs land under `analysis/figures/data/<experiment>/`; the Python scripts turn them into CSVs and figures.

**Configuration.** The router and network drivers run as-is. The experiment regimes are baked in, no arguments. The two parameterized drivers take environment-variable overrides (defaults in parentheses):

- `build_setup_clariden.sh` — `TARGET` (which integration driver to build; `flock_sem_filter_vllm_integration`).
- `sem_filter_ab_clariden.sh` — `ROWS` (table size; `2000`), `THREADS_SWEEP` (`1 2 4 8 16`), `BATCH` (rows per prompt; `32`), `ARMS` (`on off` = operator/scalar), `DATA` / `TEXT_COL` / `PROMPT` (the workload; default the movie `sf_2000` reviews). For the morsel-size sweep instead of the threads sweep: `MORSELS` (sweep values; empty = threads mode), `ROW_GROUP_SIZE` (`2048`), `MORSEL_THREADS` (`8`).

**1. Operator vs scalar throughput (morsel decoupling)** → `analysis/slurm/sem_filter_ab_clariden.sh`.

Runs the same `WHERE llm_filter(...)` query through `PhysicalSemFilter` (rewrite on) and through the scalar `llm_filter` (rewrite off) across a `threads` sweep against one endpoint. Shows the operator sustains throughput independent of DuckDB thread/morsel count while the scalar is bounded by `min(threads, morsels)`.
→ `summarize_morsel_ab.py`, `plot_sem_filter_ab.py` · data: `sem_filter_ab/`.

**2. EndpointRouter strategy validation** → `analysis/slurm/router_analysis_clariden.sh` (+ `router_sticky_sweep_clariden.sh`).

Drives the four routing strategies against a multi-GPU vLLM fleet, each in the regime where its behaviour is visible (`sticky_by_prefix` under cache-eviction pressure; `least_loaded` under capacity heterogeneity). Each run uses a fresh cold fleet.
→ `plot_router_experiment.py` · data: `router_analysis/`, `sticky_routing_analysis/`.

**3. AsyncLLMClient vs reference driver (wire parity)** → `analysis/slurm/network_analysis_clariden.sh`.

Sends byte-identical requests through flock's `AsyncLLMClient` and through sembench's reference vLLM driver against the same server, to confirm the client reaches the same throughput ceiling.
→ `asyncllmclient_vs_vllm.py` · data: `network_analysis/`.

**4. Cross-system: flock operator vs LOTUS vs Palimpzest (movie scenario)** → orchestrated by the external `sembench` harness using this extension plus the `sem_filter_vllm_integration` driver, all on one local vLLM. Compares throughput, token cost, and quality (precision / recall / F1 vs ground truth).
→ `summarize_cross_system.py` · data: `cross_system/`.

**Correctness (CI, no GPU)** → the operator/runtime suites validate against the mock server and the in-process fake client:
`ctest --test-dir build/debug/extension/flock` (the `*_vllm_integration` drivers are excluded - - -> they require a live vLLM).

## Build

See the upstream FLOCK [README](https://github.com/dais-polymtl/flock/blob/main/README.md) and [developer guide](https://dais-polymtl.github.io/flock/docs/developer-guide) for build instructions. Build flow is unchanged from upstream.

## License

Inherits from upstream FLOCK — see [LICENSE](LICENSE).
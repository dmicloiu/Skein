# Skein

**Skein: a DuckDB extension for topology-aware semantic query processing over a co-located LLM**

Skein is a research fork of [Flock](https://github.com/dais-polymtl/flock), developed as part of an MSc thesis at ETH Zurich. It keeps Flock's SQL interface while adding asynchronous physical operators for semantic filtering and extraction over a local, OpenAI-compatible LLM server such as vLLM.

## What it does

- **Asynchronous execution:** Rewrites supported `WHERE llm_filter(...)` and top-level `SELECT llm_complete(...)` calls as DuckDB physical operators. Batched, non-blocking requests decouple LLM concurrency from DuckDB's morsel and worker counts while preserving Flock's SQL interface and prompts.
- **Per-model calibration:** Configures the in-flight request cap, rows per request, and output-token bound to match the serving model's saturation point.
- **Topology-aware routing:** Distributes requests across co-located endpoints using `single`, `round_robin`, `sticky_by_prefix`, or `least_loaded` routing, supporting replicated deployments.
- **Systematic evaluation:** Measures throughput and quality against Flock and other semantic query engines across model and deployment configurations. In the thesis experiments, Skein reached roughly 8× Flock's throughput with one morsel and roughly 2× with multiple morsels.

Skein builds on Flock's model and prompt management. The extension is still named `flock` in the build and SQL interface.

## Build and explore

```bash
git clone --recurse-submodules git@github.com:dmicloiu/Skein.git
cd Skein
./scripts/build_and_run.sh
```

The build script configures the project and starts DuckDB with the extension loaded. See the [Flock documentation](https://dais-polymtl.github.io/flock/) for model setup and the existing SQL functions. The semantic operators use a co-located OpenAI-compatible endpoint; configure its URL with `SET semantic_endpoints = 'http://localhost:8000/v1';`.

## Research and tests

The [Skein research overview](analysis/SKEIN.md) describes the architecture, thesis results, and reproducibility steps. Experiment scripts, figures, and data are under [`analysis/`](analysis/). The full raw runs and original analysis notes are preserved at the `thesis-submitted` tag.

Operator and runtime tests are under [`test/`](test/). Once built, run the registered tests with `ctest --test-dir build/debug/extension/flock`.

Skein retains Flock's [license](LICENSE).

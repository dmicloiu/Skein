#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-async-vs-vllm
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Network Analysis Experiment (Clariden / NGC container):
#   AsyncLLMClient (flock) vs sembench vllm_perf_driver.py on IDENTICAL wire
#   bytes against the SAME live vLLM. 3 seed pairs, alternating side-first.
#
# All steps run inside ONE GPU job, inside the NGC container, sharing
# 127.0.0.1 -> no cross-node networking. Mirrors the sembench slurm pattern.
#
# Components (all confirmed present):
#   async binary : $HOME/projects/flock/build/flock_async_vllm_integration
#   generator    : $HOME/projects/flock/test/runtime/generate_vllm_payloads.py
#                  (imports vllm_perf_driver -> needs sembench/scripts on PYTHONPATH)
#   driver       : $HOME/projects/sembench/scripts/vllm_perf_driver.py
#   plot         : $HOME/projects/flock/analysis/asyncllmclient_vs_vllm.py
#   EDF          : $HOME/projects/sembench/ngc-pytorch-vllm.toml (mounts /users)
# -----------------------------------------------------------------------------

set -euo pipefail

cd "$HOME/projects/flock"
mkdir -p logs

EDF="$HOME/projects/sembench/ngc-pytorch-vllm.toml"

srun -ul --environment="$EDF" bash -c '
    set -uo pipefail

    # ---- localhost only: never route the in-job traffic through a proxy ----
    export NO_PROXY="localhost,127.0.0.1"
    export no_proxy="localhost,127.0.0.1"

    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate sembench

    # ---- paths (derive from $HOME, which is mounted in the container) ------
    FLOCK="$HOME/projects/flock"
    SEMBENCH="$HOME/projects/sembench"
    BIN="$FLOCK/build/flock_async_vllm_integration"
    GEN="$FLOCK/test/runtime/generate_vllm_payloads.py"
    DRIVER="$SEMBENCH/scripts/vllm_perf_driver.py"
    PLOT="$FLOCK/analysis/asyncllmclient_vs_vllm.py"

    OUT="$FLOCK/analysis/results/network_analysis/${SLURM_JOB_ID}"
    mkdir -p "$OUT"

    # ---- experiment parameters (from the spec) -----------------------------
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    PORT=8000
    SEEDS=(1 2 3)                 # 3 pairs, one seed each
    COUNT=1000                    # payloads generated per seed
    RPR=32                        # rows per request
    OUT_TOKENS=64
    SCHEMA="boolean_array"
    INFLIGHT=128                  # == driver --n-concurrent
    WARMUP=128
    TOTAL=1000                    # requests scored per side
    TIMEOUT_MS=120000             # >= 120s (E1 p99 ~ 78s)

    echo "==================== ENV ===================="
    echo "OUT=$OUT"; python -c "import sys;print(\"python\",sys.version.split()[0])"
    # The async binary is expected to be pre-built; fail fast if it is not.
    [ -x "$BIN" ] || { echo "async binary missing: $BIN"; exit 1; }
    echo "async binary OK ($BIN)"

    # ---- 1. start vLLM, poll /health ---------------------------------------
    echo "==================== START vLLM ===================="
    vllm serve "$MODEL" \
        --dtype bfloat16 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.90 \
        --host 127.0.0.1 --port "$PORT" \
        > "$OUT/vllm-server.log" 2>&1 &
    SERVER_PID=$!
    trap "kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true" EXIT

    READY=0
    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then READY=1; break; fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then echo "vLLM died on startup (see vllm-server.log)"; exit 1; fi
        sleep 5
    done
    [ "$READY" -eq 1 ] || { echo "vLLM /health timeout"; exit 1; }
    echo "vLLM ready after ~$((i*5))s"

    # ---- 2+3. per seed: generate shared payloads, run BOTH sides -----------
    #          alternate which side runs first (async,driver,async).
    for idx in "${!SEEDS[@]}"; do
        S="${SEEDS[$idx]}"
        PF="$OUT/payloads_${S}.jsonl"

        echo "==================== SEED $S : generate payloads ===================="
        PYTHONPATH="$SEMBENCH/scripts" python "$GEN" \
            --count "$COUNT" --rows-per-request "$RPR" --output-tokens "$OUT_TOKENS" \
            --response-schema "$SCHEMA" --endpoint completions --seed "$S" \
            --out "$PF"

        run_async() {
            echo "-------------------- SEED $S : AsyncLLMClient --------------------"
            "$BIN" \
                --endpoint "http://127.0.0.1:$PORT/v1/completions" \
                --inflight "$INFLIGHT" --warmup "$WARMUP" --total "$TOTAL" \
                --timeout-ms "$TIMEOUT_MS" --rows-per-request "$RPR" \
                --payload-file "$PF" 2>&1 | tee "$OUT/client_seed${S}.log"
        }
        run_driver() {
            echo "-------------------- SEED $S : vllm_perf_driver --------------------"
            python "$DRIVER" \
                --vllm-url "http://127.0.0.1:$PORT" \
                --endpoint completions \
                --n-concurrent "$INFLIGHT" \
                --rows-per-request "$RPR" \
                --output-tokens "$OUT_TOKENS" \
                --n-requests "$TOTAL" \
                --response-schema "$SCHEMA" \
                --seed "$S" \
                --out-dir "$OUT/driver_seed${S}"
        }

        if [ $(( idx % 2 )) -eq 0 ]; then run_async; run_driver; else run_driver; run_async; fi
    done

    # ---- 4. tear down vLLM (also covered by trap) --------------------------
    echo "==================== TEARDOWN vLLM ===================="
    kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
    trap - EXIT

    # ---- 5. plot -----------------------------------------------------------
    echo "==================== PLOT ===================="
    python "$PLOT" \
        --vllm-analysis-summary "$OUT"/driver_seed*/summary.json \
        --asyncllmclient-log    "$OUT"/client_seed*.log \
        --out "$OUT/asyncllmclient_vs_vllm.png" \
        --suptitle "[Network Layer] AsyncLLMClient vs vLLM Reference (N=128, R=32, Qwen2.5-7B on Clariden GH200)"

    # ---- 6. quick validation read-out (plot does the rigorous aggregation) -
    echo "==================== VALIDATION ===================="
    echo "--- driver throughput_rows_per_s per seed ---"
    for d in "$OUT"/driver_seed*/summary.json; do
        printf "%s : " "$d"
        python -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"throughput_rows_per_s\"))" "$d"
    done
    echo "--- AsyncLLMClient MAIN lines ---"
    grep -h "MAIN" "$OUT"/client_seed*.log 2>/dev/null || true
    echo "DONE. Artefacts in: $OUT"
'

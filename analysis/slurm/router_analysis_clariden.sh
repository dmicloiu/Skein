#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-endpoint-router
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# EndpointRouter validation (Clariden / NGC container):
#   Drive a pool of FOUR live vLLM endpoints (one per GH200 GPU) through the
#   real flock::EndpointRouter and capture the system-level effect of each
#   routing strategy. Two scenarios:
#       1. one GPU  : `single`           (uses ep0 only; the other 3 idle)
#       2. many GPUs: round_robin / sticky_by_prefix / least_loaded
#
# All steps run inside ONE 4-GPU job, inside the NGC container, sharing
# 127.0.0.1 -> no cross-node networking. Mirrors network_analysis_clariden.sh.
#
# This job ONLY produces raw data; ALL plotting happens off-cluster with
# analysis/plot_router_experiment.py. Pull the OUT dir home and iterate there.
#
# Components:
#   driver  : $HOME/projects/flock/build/flock_endpoint_router_vllm_integration
#             (pre-build it like flock_async_vllm_integration; fails fast if missing)
#   generator: $HOME/projects/flock/test/runtime/generate_vllm_payloads.py
#              (imports vllm_perf_driver -> needs sembench/scripts on PYTHONPATH)
#   EDF     : $HOME/projects/sembench/ngc-pytorch-vllm.toml (mounts /users)
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

    # ---- paths -------------------------------------------------------------
    FLOCK="$HOME/projects/flock"
    SEMBENCH="$HOME/projects/sembench"
    BIN="$FLOCK/build/flock_endpoint_router_vllm_integration"
    GEN="$FLOCK/test/runtime/generate_vllm_payloads.py"

    OUT="$FLOCK/analysis/results/router_analysis/${SLURM_JOB_ID}"
    mkdir -p "$OUT"

    # ---- experiment parameters ---------------------------------------------
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=4
    BASE_PORT=8000
    COUNT=2048                    # payloads generated per workload set
    RPR=32                        # rows per request
    OUT_TOKENS=64
    SCHEMA="boolean_array"
    INFLIGHT=256                  # across the whole pool (4 GPUs)
    WARMUP=256
    TOTAL=2048                    # requests scored per strategy run
    TIMEOUT_MS=120000
    PREFIX_GROUPS=16              # distinct cacheable prefixes for sticky
    SHARED_PREFIX_WORDS=128
    SKEW_FRAC=0.2                 # 20% of requests are 8x more expensive
    SKEW_OUT_TOKENS=512
    SEED=1

    echo "==================== ENV ===================="
    echo "OUT=$OUT"; python -c "import sys;print(\"python\",sys.version.split()[0])"
    [ -x "$BIN" ] || { echo "driver binary missing: $BIN (build it first)"; exit 1; }
    echo "driver OK ($BIN)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv

    # ---- 1. build the three workload payload sets --------------------------
    echo "==================== GENERATE PAYLOADS ===================="
    UNIFORM="$OUT/payloads_uniform.jsonl"
    GROUPED="$OUT/payloads_grouped.jsonl"; GROUPED_KEYS="$OUT/keys_grouped.txt"
    SKEWED="$OUT/payloads_skewed.jsonl"

    PYTHONPATH="$SEMBENCH/scripts" python "$GEN" \
        --count "$COUNT" --rows-per-request "$RPR" --output-tokens "$OUT_TOKENS" \
        --response-schema "$SCHEMA" --endpoint completions --seed "$SEED" \
        --out "$UNIFORM"

    PYTHONPATH="$SEMBENCH/scripts" python "$GEN" \
        --count "$COUNT" --rows-per-request "$RPR" --output-tokens "$OUT_TOKENS" \
        --response-schema "$SCHEMA" --endpoint completions --seed "$SEED" \
        --prefix-groups "$PREFIX_GROUPS" --shared-prefix-words "$SHARED_PREFIX_WORDS" \
        --out "$GROUPED" --keys-out "$GROUPED_KEYS"

    PYTHONPATH="$SEMBENCH/scripts" python "$GEN" \
        --count "$COUNT" --rows-per-request "$RPR" --output-tokens "$OUT_TOKENS" \
        --response-schema "$SCHEMA" --endpoint completions --seed "$SEED" \
        --skew-frac "$SKEW_FRAC" --skew-output-tokens "$SKEW_OUT_TOKENS" \
        --out "$SKEWED"

    # ---- 2. launch one vLLM server per GPU, poll /health -------------------
    echo "==================== START vLLM x$NGPU ===================="
    declare -a PIDS=()
    ENDPOINTS=""
    for i in $(seq 0 $((NGPU-1))); do
        PORT=$((BASE_PORT+i))
        CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
            --dtype bfloat16 \
            --max-model-len 16384 \
            --gpu-memory-utilization 0.90 \
            --host 127.0.0.1 --port "$PORT" \
            > "$OUT/vllm-ep$i.log" 2>&1 &
        PIDS+=($!)
        EP="http://127.0.0.1:$PORT/v1/completions"
        ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}$EP"
    done
    cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done;
                wait 2>/dev/null || true; }
    trap cleanup EXIT
    echo "endpoints: $ENDPOINTS"

    for i in $(seq 0 $((NGPU-1))); do
        PORT=$((BASE_PORT+i))
        READY=0
        for t in $(seq 1 180); do
            if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then READY=1; break; fi
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                echo "vLLM ep$i died on startup (see vllm-ep$i.log)"; exit 1; fi
            sleep 5
        done
        [ "$READY" -eq 1 ] || { echo "vLLM ep$i /health timeout"; exit 1; }
        echo "vLLM ep$i ready"
    done

    # ---- 3. raw /metrics snapshot per endpoint -----------------------------
    snap() {  # $1 = tag
        for i in $(seq 0 $((NGPU-1))); do
            curl -s "http://127.0.0.1:$((BASE_PORT+i))/metrics" \
                 > "$OUT/metrics_${1}_ep$i.txt" || true
        done
    }

    # ---- 4. run each strategy against the SAME fleet -----------------------
    #     strategy is the only variable; the endpoint pool never changes.
    run() {  # $1=strategy $2=payloads $3=keysfile-or-"" $4=tag
        local keysarg=""
        [ -n "$3" ] && keysarg="--keys-file $3"
        echo "-------------------- RUN $4 (strategy=$1) --------------------"
        snap "before_$4"
        "$BIN" --endpoints "$ENDPOINTS" --strategy "$1" \
               --payload-file "$2" $keysarg \
               --model "$MODEL" \
               --inflight "$INFLIGHT" --warmup "$WARMUP" --total "$TOTAL" \
               --rows-per-request "$RPR" --timeout-ms "$TIMEOUT_MS" \
               --result-out "$OUT/result_$4.json" 2>&1 | tee "$OUT/run_$4.log"
        snap "after_$4"
    }

    # Scenario 1: one GPU.
    run single           "$UNIFORM" ""             single
    # Scenario 2: many GPUs.
    run round_robin      "$UNIFORM" ""             round_robin
    run sticky_by_prefix "$GROUPED" "$GROUPED_KEYS" sticky
    run round_robin      "$GROUPED" "$GROUPED_KEYS" round_robin_grouped   # sticky baseline
    run least_loaded     "$SKEWED"  ""             least_loaded
    run round_robin      "$SKEWED"  ""             round_robin_skewed     # least_loaded baseline

    # ---- 5. teardown (also covered by trap) --------------------------------
    echo "==================== TEARDOWN ===================="
    cleanup
    trap - EXIT

    # ---- 6. quick read-out (rigorous aggregation happens off-cluster) ------
    echo "==================== VALIDATION ===================="
    grep -h "per_endpoint_count" "$OUT"/run_*.log 2>/dev/null || true
    grep -h "^MAIN" "$OUT"/run_*.log 2>/dev/null || true
    echo "DONE. Artefacts in: $OUT"
    echo "Pull home with e.g.:  rsync -av <clariden>:$OUT analysis/results/router_analysis/"
'

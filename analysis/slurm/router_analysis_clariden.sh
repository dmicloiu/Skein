#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-endpoint-router
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# EndpointRouter validation.
#
# Drives live vLLM endpoints (one per GH200 GPU in the node) through the real
# flock::EndpointRouter and captures the system-level effect of every routing
# strategy. Each run gets its OWN cold fleet, configured to the regime where
# that strategy's behaviour is visible. Otherwise, a persistent fleet would let one
# run's prefix cache leak into the next and destroy the effect we measure.
#
# Six cold-fleet runs (bring up -> run -> tear down each time):
#
#   1 single               4 GPU, full cache, uniform   -> pins to ep0; 1-GPU
#                                                           throughput ceiling.
#   2 round_robin          4 GPU, full cache, uniform   -> even spread; ~Nx
#                                                           throughput vs #1.
#   3 sticky               4 GPU, CONSTRAINED cache,     -> each prefix pinned to
#                          many long prefixes (grouped)     one endpoint, cache
#                                                           stays warm.
#   4 round_robin_grouped  same constrained config       -> scatters prefixes ->
#                                                           cache thrash (#3's
#                                                           baseline).
#   5 least_loaded         ep0 THROTTLED, uniform        -> routes around the slow
#                                                           node; better tail.
#   6 round_robin_hetero   ep0 throttled, uniform        -> ships 1/4 to the slow
#                                                           node -> worse tail
#                                                           (#5's baseline).
#
# round_robin is the universal baseline, so it appears in each comparison's
# config. #3-vs-#4 and #5-vs-#6 are fair because both start from a cold fleet.
# -----------------------------------------------------------------------------

set -euo pipefail

cd "$HOME/projects/flock"
mkdir -p logs

EDF="$HOME/projects/sembench/ngc-pytorch-vllm.toml"

srun -ul --environment="$EDF" bash -c '
    set -uo pipefail

    export NO_PROXY="localhost,127.0.0.1"
    export no_proxy="localhost,127.0.0.1"

    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate sembench

    FLOCK="$HOME/projects/flock"
    SEMBENCH="$HOME/projects/sembench"
    BIN="$FLOCK/build/flock_endpoint_router_vllm_integration"
    GEN="$FLOCK/test/runtime/generate_vllm_payloads.py"
    OUT="$FLOCK/analysis/results/router_analysis/${SLURM_JOB_ID}"
    mkdir -p "$OUT"

    # ---- experiment parameters --------------------------------------------
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=4
    BASE_PORT=8000
    SCHEMA="boolean_array"
    RPR=32
    OUT_TOKENS=64
    SEED=1
    TIMEOUT_MS=120000
    COUNT=2048                  # payloads per workload set
    TOTAL=2048                  # requests scored per run
    WARMUP=256

    INFLIGHT_FULL=256           # saturate the pool (scaling, tail latency)
    INFLIGHT_CACHE=64           # low, so the cache constraint bites on resident
                                # prefixes rather than active-request KV

    # sticky pressure knobs: round_robin must keep ALL prefixes resident per
    # endpoint while sticky keeps only ~1/NGPU of them, so the KV cache has to be
    # too small for the full set but big enough for sticky'\''s slice.
    #   - at gpu-mem-util 0.40 the 97 GB GH200 still cached all 128 prefixes and
    #     sticky/round_robin TIED at 43%; the cache must be forced small directly.
    #   - num-gpu-blocks-override sets the KV block count deterministically.
    #     ~700 tok/prefix -> ~44 blocks/prefix; with inflight 64 (~960 active
    #     blocks) 3000 blocks fits sticky (~2400) but not round_robin (~6600).
    PREFIX_GROUPS=128
    SHARED_PREFIX_WORDS=512
    KV_BLOCKS_OVERRIDE=3000     # tiny KV cache -> round_robin thrashes
    FULL_UTIL=0.90             # normal gpu-memory-utilization

    SLOW_MAX_SEQS=8             # ep0 throttle for the heterogeneous fleet

    echo "==================== ENV ===================="
    echo "OUT=$OUT"; python -c "import sys;print(\"python\",sys.version.split()[0])"
    [ -x "$BIN" ] || { echo "driver binary missing: $BIN (build it first)"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv

    # ---- fleet helpers -----------------------------------------------------
    PIDS=(); ENDPOINTS=""
    # start_fleet <gpu_util> <ep0_extra_args> <other_extra_args>
    start_fleet() {
        local util="$1" ep0_extra="$2" other_extra="$3"
        PIDS=(); ENDPOINTS=""
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i))
            local extra="$other_extra"; [ "$i" -eq 0 ] && extra="$ep0_extra"
            CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
                --dtype bfloat16 --max-model-len 16384 \
                --enable-prefix-caching \
                --gpu-memory-utilization "$util" \
                --host 127.0.0.1 --port "$port" $extra \
                > "$OUT/vllm-ep$i-$(date +%s).log" 2>&1 &
            PIDS+=($!)
            ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}http://127.0.0.1:$port/v1/completions"
        done
        echo "started fleet (util=$util ep0_extra=[$ep0_extra]) -> $ENDPOINTS"
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i)) ready=0
            for t in $(seq 1 180); do
                if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
                kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "ep$i died on startup"; exit 1; }
                sleep 5
            done
            [ "$ready" -eq 1 ] || { echo "ep$i /health timeout"; exit 1; }
        done
        echo "fleet ready"
    }
    stop_fleet() {
        for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
        wait 2>/dev/null || true
        sleep 15                      # let GPU memory drain before the next start
        PIDS=()
    }
    trap stop_fleet EXIT

    snap() {  # snap <tag>
        for i in $(seq 0 $((NGPU-1))); do
            curl -s "http://127.0.0.1:$((BASE_PORT+i))/metrics" \
                 > "$OUT/metrics_${1}_ep$i.txt" || true
        done
    }
    # run <strategy> <payloads> <keys-or-""> <inflight> <tag>
    run() {
        local keysarg=""; [ -n "$3" ] && keysarg="--keys-file $3"
        echo "-------------------- RUN $5 (strategy=$1, inflight=$4) --------------------"
        snap "before_$5"
        "$BIN" --endpoints "$ENDPOINTS" --strategy "$1" \
               --payload-file "$2" $keysarg --model "$MODEL" \
               --inflight "$4" --warmup "$WARMUP" --total "$TOTAL" \
               --rows-per-request "$RPR" --timeout-ms "$TIMEOUT_MS" \
               --result-out "$OUT/result_$5.json" 2>&1 | tee "$OUT/run_$5.log"
        snap "after_$5"
    }

    # ---- payload sets ------------------------------------------------------
    echo "==================== GENERATE PAYLOADS ===================="
    UNIFORM="$OUT/payloads_uniform.jsonl"
    GROUPED="$OUT/payloads_grouped.jsonl"; GKEYS="$OUT/keys_grouped.txt"
    PYTHONPATH="$SEMBENCH/scripts" python "$GEN" \
        --count "$COUNT" --rows-per-request "$RPR" --output-tokens "$OUT_TOKENS" \
        --response-schema "$SCHEMA" --endpoint completions --seed "$SEED" \
        --out "$UNIFORM"
    PYTHONPATH="$SEMBENCH/scripts" python "$GEN" \
        --count "$COUNT" --rows-per-request "$RPR" --output-tokens "$OUT_TOKENS" \
        --response-schema "$SCHEMA" --endpoint completions --seed "$SEED" \
        --prefix-groups "$PREFIX_GROUPS" --shared-prefix-words "$SHARED_PREFIX_WORDS" \
        --out "$GROUPED" --keys-out "$GKEYS"

    # ---- 1. single: pins to ep0, 1-GPU throughput ceiling ------------------
    start_fleet "$FULL_UTIL" "" ""
    run single "$UNIFORM" "" "$INFLIGHT_FULL" single
    stop_fleet

    # ---- 2. round_robin: even spread, ~Nx throughput -----------------------
    start_fleet "$FULL_UTIL" "" ""
    run round_robin "$UNIFORM" "" "$INFLIGHT_FULL" round_robin
    stop_fleet

    # ---- 3. sticky: prefixes pinned, cache warm (tiny KV cache) ------------
    KV_OVERRIDE="--num-gpu-blocks-override $KV_BLOCKS_OVERRIDE"
    start_fleet "$FULL_UTIL" "$KV_OVERRIDE" "$KV_OVERRIDE"
    run sticky_by_prefix "$GROUPED" "$GKEYS" "$INFLIGHT_CACHE" sticky
    stop_fleet

    # ---- 4. round_robin baseline for sticky (same tiny-cache config) -------
    start_fleet "$FULL_UTIL" "$KV_OVERRIDE" "$KV_OVERRIDE"
    run round_robin "$GROUPED" "$GKEYS" "$INFLIGHT_CACHE" round_robin_grouped
    stop_fleet

    # ---- 5. least_loaded: routes around the throttled ep0 ------------------
    start_fleet "$FULL_UTIL" "--max-num-seqs $SLOW_MAX_SEQS" ""
    run least_loaded "$UNIFORM" "" "$INFLIGHT_FULL" least_loaded
    stop_fleet

    # ---- 6. round_robin baseline for least_loaded (same hetero fleet) ------
    start_fleet "$FULL_UTIL" "--max-num-seqs $SLOW_MAX_SEQS" ""
    run round_robin "$UNIFORM" "" "$INFLIGHT_FULL" round_robin_hetero
    stop_fleet
    trap - EXIT

    # ---- quick read-out (rigorous aggregation happens off-cluster) ---------
    echo "==================== VALIDATION ===================="
    grep -h "per_endpoint_count" "$OUT"/run_*.log 2>/dev/null || true
    grep -h "^MAIN" "$OUT"/run_*.log 2>/dev/null || true
    echo "DONE. Artefacts in: $OUT"
    echo "Pull home with e.g.:  rsync -av <clariden>:$OUT analysis/figures/data/router_analysis/"
'

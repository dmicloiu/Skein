#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-sticky-sweep
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# StickyByPrefix system-behaviour validation (Clariden / NGC container).
#
# Goal: show the sticky mechanism behaves as designed -- pinning each distinct
# prefix to one endpoint sustains prefix-cache reuse, where round_robin's
# scattering does not once the working set of distinct prefixes stresses the
# cache. We do NOT claim a particular Flock workload; this validates the system
# behaviour with a controlled workload.
#
# Rather than betting on one magic cache size (which failed repeatedly: too big
# -> both cache everything -> tie; too small -> active KV starves the cache ->
# tie at 0%), we SWEEP the number of distinct recurring prefixes K and plot the
# hit-rate-vs-K curve. The sweep brackets the crossover wherever it falls:
#   small K  -> both high and tied   (cache holds every prefix)
#   large K  -> round_robin collapses (must hold all K per endpoint) while
#               sticky stays high     (only holds K/NGPU)
#
# Each (K, strategy) runs on its OWN cold fleet so no run inherits another's
# warm cache. A moderate, fixed KV cap positions the crossover at modest K so
# the runs stay small; the cap is NOT a knife-edge because the sweep captures
# the crossover regardless of where the cap puts it.
#
# Produces RAW data only; plot off-cluster with
# analysis/plot_router_experiment.py (hit-rate-vs-K figure).
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
    OUT="$FLOCK/analysis/results/router_sticky_sweep/${SLURM_JOB_ID}"
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
    SHARED_PREFIX_WORDS=512
    FULL_UTIL=0.90

    # The sweep axis: number of distinct recurring prefixes.
    K_SWEEP="32 64 128 256 512"
    REUSE=8                      # requests per prefix (COUNT = REUSE*K)
    INFLIGHT=128                 # high enough that hits accumulate; active KV at
                                 # 128/NGPU x ~131 blocks ~= 4200 blocks/endpoint
    # Moderate cap: leaves ~3800 blocks for prefixes after active KV, so the
    # round_robin/sticky crossover lands inside the K sweep. Not a knife-edge --
    # the sweep brackets the crossover wherever it actually falls.
    KV_BLOCKS_OVERRIDE=8000

    echo "==================== ENV ===================="
    echo "OUT=$OUT  K_SWEEP=[$K_SWEEP]  inflight=$INFLIGHT  kv_blocks=$KV_BLOCKS_OVERRIDE"
    # Job-private HOME: flock persists its model catalog at
    # $HOME/.duckdb/flock_storage and parallel jobs race the shared file
    # (NFS lock: "Conflicting lock is held in PID -3").
    JOB_HOME="/tmp/flock_home_${SLURM_JOB_ID:-local}"
    mkdir -p "$JOB_HOME/.duckdb"
    [ -x "$BIN" ] || { echo "driver binary missing: $BIN (build it first)"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv

    # ---- fleet helpers (cold fleet per run) --------------------------------
    PIDS=(); ENDPOINTS=""
    start_fleet() {  # start_fleet <extra_args>
        local extra="$1"
        PIDS=(); ENDPOINTS=""
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i))
            CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
                --dtype bfloat16 --max-model-len 16384 \
                --enable-prefix-caching \
                --gpu-memory-utilization "$FULL_UTIL" \
                --host 127.0.0.1 --port "$port" $extra \
                > "$OUT/vllm-ep$i-$(date +%s).log" 2>&1 &
            PIDS+=($!)
            ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}http://127.0.0.1:$port/v1/completions"
        done
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i)) ready=0
            for t in $(seq 1 180); do
                if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
                kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "ep$i died on startup"; exit 1; }
                sleep 5
            done
            [ "$ready" -eq 1 ] || { echo "ep$i /health timeout"; exit 1; }
        done
    }
    stop_fleet() {
        for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
        wait 2>/dev/null || true
        sleep 15
        PIDS=()
    }
    trap stop_fleet EXIT

    snap() {  # snap <tag>
        for i in $(seq 0 $((NGPU-1))); do
            curl -s "http://127.0.0.1:$((BASE_PORT+i))/metrics" \
                 > "$OUT/metrics_${1}_ep$i.txt" || true
        done
    }
    run() {  # run <strategy> <payloads> <keys> <tag>
        echo "-------------------- RUN $4 (strategy=$1) --------------------"
        snap "before_$4"
        HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --strategy "$1" \
               --payload-file "$2" --keys-file "$3" --model "$MODEL" \
               --inflight "$INFLIGHT" --warmup 0 --total "$(wc -l < "$2")" \
               --rows-per-request "$RPR" --timeout-ms "$TIMEOUT_MS" \
               --result-out "$OUT/result_$4.json" 2>&1 | tee "$OUT/run_$4.log"
        snap "after_$4"
    }

    KV_OVERRIDE="--num-gpu-blocks-override $KV_BLOCKS_OVERRIDE"
    for K in $K_SWEEP; do
        COUNT=$((REUSE*K))
        PF="$OUT/payloads_k$K.jsonl"; KF="$OUT/keys_k$K.txt"
        echo "==================== K=$K (count=$COUNT) ===================="
        PYTHONPATH="$SEMBENCH/scripts" python "$GEN" \
            --count "$COUNT" --rows-per-request "$RPR" --output-tokens "$OUT_TOKENS" \
            --response-schema "$SCHEMA" --endpoint completions --seed "$SEED" \
            --prefix-groups "$K" --shared-prefix-words "$SHARED_PREFIX_WORDS" \
            --out "$PF" --keys-out "$KF"

        # Each strategy on its own cold fleet (no warm-cache carryover).
        start_fleet "$KV_OVERRIDE"
        run sticky_by_prefix "$PF" "$KF" "sticky_k$K"
        stop_fleet
        start_fleet "$KV_OVERRIDE"
        run round_robin "$PF" "$KF" "round_robin_k$K"
        stop_fleet
    done
    trap - EXIT

    echo "==================== VALIDATION ===================="
    grep -h "^MAIN" "$OUT"/run_*.log 2>/dev/null || true
    echo "DONE. Artefacts in: $OUT"
    echo "Pull home with e.g.:  rsync -av <clariden>:$OUT analysis/figures/data/router_analysis/"
'

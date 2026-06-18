#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-sem-filter-ab
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# sem_filter operator A/B: PhysicalSemFilter (semantic_rewrite_enabled=true) vs
# the scalar llm_filter (off). Same query, same vLLM, same batch_size; only the
# rewrite flag differs. Threads sweep analysis. Single cold endpoint (TP=1).
#
# COLD FLEET PER RUN: vllm runs with --enable-prefix-caching, so a warm instance
# would let each run hit the previous run's cached instruction/template prefixes
# and report inflated throughput (contaminating the A/B and the threads sweep).
# So every (rewrite, threads) measurement starts a fresh endpoint and tears it
# down afterwards; the before-snap is therefore a genuinely cold cache. (Same
# cold-per-run discipline as the router GH200 validation.)
#
# Goal: artifacts proving PhysicalSemFilter engages + sustains throughput.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$HOME/projects/flock"
mkdir -p logs
EDF="$HOME/projects/sembench/ngc-pytorch-vllm.toml"

srun -ul --environment="$EDF" bash -c '
    set -uo pipefail
    export NO_PROXY="localhost,127.0.0.1"; export no_proxy="localhost,127.0.0.1"
    source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sembench

    FLOCK="$HOME/projects/flock"; SEMBENCH="$HOME/projects/sembench"
    BIN="$FLOCK/build/flock_sem_filter_vllm_integration"
    OUT="$FLOCK/analysis/results/sem_filter_ab/${SLURM_JOB_ID}"; mkdir -p "$OUT"

    # flock persists its model/secret catalog at ~/.duckdb/flock_storage; its
    # CreateDirectory is non-recursive (src/core/config/config.cpp:47), so the
    # ~/.duckdb parent must already exist or flock_storage fails to attach and
    # every run dies with "Model not found". Create it up front.
    mkdir -p "$HOME/.duckdb"

    # ---- experiment parameters --------------------------------------------
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=1; BASE_PORT=8000; FULL_UTIL=0.90
    IN_FLIGHT=128; BATCH=32; TIMEOUT_MS=120000
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    ROWS="${ROWS:-2000}"
    THREADS_SWEEP="${THREADS_SWEEP:-1 2 4 8 16}"

    echo "==================== ENV ===================="
    echo "OUT=$OUT  rows=$ROWS  threads_sweep=[$THREADS_SWEEP]  (cold fleet per run)"
    [ -x "$BIN" ]  || { echo "driver missing: $BIN (build it first)"; exit 1; }
    [ -f "$DATA" ] || { echo "dataset missing: $DATA"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv

    # ---- fleet helpers (mirror router_analysis_clariden.sh) ----------------
    PIDS=(); ENDPOINTS=""
    start_fleet() {
        PIDS=(); ENDPOINTS=""
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i))
            CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
                --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
                --gpu-memory-utilization "$FULL_UTIL" --host 127.0.0.1 --port "$port" \
                > "$OUT/vllm-ep$i-$(date +%s).log" 2>&1 &
            PIDS+=($!)
            ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}http://127.0.0.1:$port/v1/completions"
        done
        echo "started fleet -> $ENDPOINTS"
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i)) ready=0
            for t in $(seq 1 180); do
                curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ready=1; break; }
                kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "ep$i died on startup"; exit 1; }
                sleep 5
            done
            [ "$ready" -eq 1 ] || { echo "ep$i /health timeout"; exit 1; }
        done
        echo "fleet ready"
    }
    stop_fleet() {
        for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
        wait 2>/dev/null || true; sleep 15; PIDS=()
    }
    trap stop_fleet EXIT

    snap() {  # snap <tag>: per-endpoint vLLM /metrics
        for i in $(seq 0 $((NGPU-1))); do
            curl -s "http://127.0.0.1:$((BASE_PORT+i))/metrics" > "$OUT/metrics_${1}_ep$i.txt" || true
        done
    }
    # run <on|off> <threads> <tag>: one measurement against the CURRENT cold fleet.
    run() {
        echo "------ RUN $3 (rewrite=$1 threads=$2 rows=$ROWS inflight=$IN_FLIGHT batch=$BATCH) ------"
        snap "before_$3"
        "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" \
               --data "$DATA" --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
               --rewrite "$1" --threads "$2" --inflight "$IN_FLIGHT" --rows-per-request "$BATCH" \
               --timeout-ms "$TIMEOUT_MS" --result-out "$OUT/result_$3.json" 2>&1 | tee "$OUT/run_$3.log"
        snap "after_$3"
    }
    # cold_run: fresh endpoint per measurement so prefix caching never carries
    # over between rewrite arms or thread counts.
    cold_run() {  # cold_run <on|off> <threads> <tag>
        start_fleet
        run "$1" "$2" "$3"
        stop_fleet
    }

    # ---- A/B across the threads sweep (cold endpoint every run) ------------
    echo "==================== A/B (threads sweep, cold per run) ===================="
    for T in $THREADS_SWEEP; do
        cold_run on  "$T" "operator_t$T"     # PhysicalSemFilter: ~flat near ceiling (cap-bound)
        cold_run off "$T" "scalar_t$T"       # scalar: rises toward min(threads,morsels)
    done
    trap - EXIT

    # ---- quick read-out (rigorous aggregation happens off-cluster) ---------
    echo "==================== VALIDATION ===================="
    grep -h "rows_per_s\|threads\|passes" "$OUT"/result_*.json 2>/dev/null || true
    echo "ceiling reference: ~800 rows/s (E5, Qwen2.5-7B, N=128 R=32)."
    echo "expect: operator ~flat near ceiling across threads; scalar plateaus at its morsel-bound max."
    echo "DONE. Artefacts in: $OUT (metrics_*_ep*, result_*.json, run_*.log, vllm-*.log)"
    echo "Pull home with e.g.:  rsync -av <clariden>:$OUT analysis/figures/data/sem_filter_ab/"
'
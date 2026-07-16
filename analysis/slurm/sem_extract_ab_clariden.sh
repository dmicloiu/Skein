#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-sem-extract-ab
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# sem_extract operator A/B: PhysicalSemExtract (semantic_rewrite_enabled=true) vs
# the scalar llm_complete (off). A PROJECTED extraction (sentiment classification
# via llm_complete), materialised so every row executes. Same query, same vLLM,
# same batch_size; only the rewrite flag differs. Threads sweep analysis. Single
# cold endpoint (TP=1). Same running config as sem_filter_ab_clariden.sh (same
# env vars / modes) -- only the driver + workload + output dir differ.
#
# COLD FLEET PER RUN: vllm runs with --enable-prefix-caching, so a warm instance
# would let each run hit the previous run's cached instruction/template prefixes
# and report inflated throughput (contaminating the A/B and the threads sweep).
# So every (rewrite, threads) measurement starts a fresh endpoint and tears it
# down afterwards; the before-snap is therefore a genuinely cold cache. (Same
# cold-per-run discipline as the router GH200 validation.)
#
# Goal: artifacts proving PhysicalSemExtract engages + sustains throughput.
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
    BIN="$FLOCK/build/flock_sem_extract_vllm_integration"
    OUT="$FLOCK/analysis/results/sem_extract_ab/${SLURM_JOB_ID}"; mkdir -p "$OUT"

    # flock persists its model/secret catalog at ~/.duckdb/flock_storage; its
    # CreateDirectory is non-recursive (src/core/config/config.cpp:47), so the
    # ~/.duckdb parent must already exist or flock_storage fails to attach and
    # every run dies with "Model not found". Create it up front.
    mkdir -p "$HOME/.duckdb"

    # ---- experiment parameters --------------------------------------------
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=1; BASE_PORT=8000; FULL_UTIL=0.90
    IN_FLIGHT=128; TIMEOUT_MS=120000
    # Per-row output-token cap = MAX_OUT_MULT * rows_per_request. Morsel mode
    # (rows_per_request=1) makes this the per-row ceiling; 16 truncates rows
    # where the model echoes the review -> invalid JSON -> query fails. Override
    # higher for morsel reruns (both arms, so the A/B stays fair).
    MAX_OUT_MULT="${MAX_OUT_MULT:-16}"
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-Classify the sentiment of this movie review as exactly POSITIVE or NEGATIVE.}"
    THREADS_SWEEP="${THREADS_SWEEP:-1 2 4 8 16}"

    # ---- morsel mode (optional) -------------------------------------------
    # Default (MORSELS unset): the threads-sweep A/B on an in-memory table =
    # ONE row group = ONE morsel, so the scalar is single-threaded regardless of
    # --threads (min(threads, morsels) = 1). Set MORSELS to a list of morsel
    # counts to instead materialise the rows into an attached on-disk DB whose
    # ROW_GROUP_SIZE makes rows/ROW_GROUP_SIZE morsels, so the scalar can
    # parallelise to min(MORSEL_THREADS, morsels). Backing file is on /tmp
    # (tmpfs/node-local) -> no Lustre. Uses R=1 by default (the config of
    # interest); each morsel = ROW_GROUP_SIZE rows so ROWS = M * ROW_GROUP_SIZE.
    # The requested extract sweep is two jobs:
    #   MORSELS="1 8 16" MORSEL_THREADS=8   sbatch <this>
    #   MORSELS="128"    MORSEL_THREADS=128 sbatch <this>
    MORSELS="${MORSELS:-}"
    ROW_GROUP_SIZE="${ROW_GROUP_SIZE:-2048}"   # must be a multiple of 2048
    MORSEL_THREADS="${MORSEL_THREADS:-8}"
    # ARMS selects which rewrite arms run (morsel mode only). Default "on off"
    # runs both (one job). Set ARMS="on" or ARMS="off" to run a single arm as
    # its own short job -> fits the <1h backfill window on a busy partition
    # (a 2-arm job at 262k rows/arm needs >1h and gets starved by short jobs).
    ARMS="${ARMS:-on off}"
    MORSEL_DB_DIR="${MORSEL_DB_DIR:-/tmp}"     # node-local; avoid Lustre $HOME
    if [ -n "$MORSELS" ]; then BATCH="${BATCH:-1}"; else BATCH="${BATCH:-32}"; fi
    ROWS="${ROWS:-2000}"

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
            # disable_any_whitespace: force COMPACT guided-JSON output. Without it,
            # xgrammar lets the model emit whitespace between array elements; under
            # greedy decoding (temperature=0) the model stalls and loops on
            # whitespace until max_tokens, yielding truncated/invalid JSON. Requires
            # backend pinned to xgrammar (validator rejects it with backend=auto).
            CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
                --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
                --gpu-memory-utilization "$FULL_UTIL" --host 127.0.0.1 --port "$port" \
                --structured-outputs-config '\''{"backend": "xgrammar", "disable_any_whitespace": true}'\'' \
                > "$OUT/vllm-ep$i-$(date +%s).log" 2>&1 &
            PIDS+=($!)
            # /v1/chat/completions: the operator posts chat requests here (parity
            # with the scalar arm). DeriveBaseUrl in the driver strips this back to
            # /v1 for the scalar secret base_url (avoid apostrophes here: this whole
            # block is inside srun bash -c '\''...'\'' and a stray quote breaks it).
            ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}http://127.0.0.1:$port/v1/chat/completions"
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
    # Reads ROWS/BATCH globals; in morsel mode also passes --row-group-size and a
    # fresh node-local --attach-db so the rows become ROWS/ROW_GROUP_SIZE morsels.
    run() {
        local rgs_args=()
        if [ "${RGS_ACTIVE:-0}" -gt 0 ]; then
            rgs_args=(--row-group-size "$ROW_GROUP_SIZE" --attach-db "$MORSEL_DB_DIR/sem_morsel_${SLURM_JOB_ID:-x}_$3.db")
        fi
        echo "------ RUN $3 (rewrite=$1 threads=$2 rows=$ROWS inflight=$IN_FLIGHT batch=$BATCH rgs=${RGS_ACTIVE:-0}) ------"
        snap "before_$3"
        "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" \
               --data "$DATA" --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
               --rewrite "$1" --threads "$2" --inflight "$IN_FLIGHT" --rows-per-request "$BATCH" \
               --max-out-mult "$MAX_OUT_MULT" --timeout-ms "$TIMEOUT_MS" "${rgs_args[@]}" \
               --result-out "$OUT/result_$3.json" 2>&1 | tee "$OUT/run_$3.log"
        snap "after_$3"
    }
    # cold_run: fresh endpoint per measurement so prefix caching never carries
    # over between rewrite arms or thread counts.
    cold_run() {  # cold_run <on|off> <threads> <tag>
        start_fleet
        run "$1" "$2" "$3"
        stop_fleet
    }

    if [ -n "$MORSELS" ]; then
        # ---- MORSEL MODE: rows/ROW_GROUP_SIZE morsels at fixed MORSEL_THREADS ---
        # Operator vs scalar at each morsel count. The scalar now reaches
        # min(MORSEL_THREADS, morsels)-way (vs the 1-morsel in-memory case);
        # the operator stays cap-bound. Answers "does the scalar catch up?".
        echo "==================== MORSEL SWEEP (cold per run, R=$BATCH, threads=$MORSEL_THREADS) ===================="
        echo "morsels=[$MORSELS]  row_group_size=$ROW_GROUP_SIZE  db_dir=$MORSEL_DB_DIR"
        # The CSV must hold enough rows for the largest morsel count, or the table
        # collapses to fewer row groups than intended (DATA=sf_2000 is too small).
        MAXM=0; for M in $MORSELS; do [ "$M" -gt "$MAXM" ] && MAXM="$M"; done
        NEED=$((MAXM * ROW_GROUP_SIZE)); AVAIL=$(($(wc -l < "$DATA") - 1))
        [ "$AVAIL" -ge "$NEED" ] || { echo "FATAL: $DATA has $AVAIL rows; morsel mode needs >= $NEED (max ${MAXM} morsels x ${ROW_GROUP_SIZE}). Point DATA at a larger Reviews.csv."; exit 1; }
        RGS_ACTIVE=1
        echo "arms=[$ARMS]"
        for M in $MORSELS; do
            ROWS=$((M * ROW_GROUP_SIZE))
            for arm in $ARMS; do
                if [ "$arm" = "on" ]; then tag="operator"; else tag="scalar"; fi
                cold_run "$arm" "$MORSEL_THREADS" "${tag}_m${M}_t${MORSEL_THREADS}"
            done
        done
    else
        # ---- A/B across the threads sweep (cold endpoint every run) ------------
        echo "==================== A/B (threads sweep, cold per run) ===================="
        RGS_ACTIVE=0
        for T in $THREADS_SWEEP; do
            cold_run on  "$T" "operator_t$T"     # PhysicalSemExtract: ~flat near ceiling (cap-bound)
            cold_run off "$T" "scalar_t$T"       # scalar: 1 morsel here -> single-threaded
        done
    fi
    trap - EXIT

    # ---- quick read-out (rigorous aggregation happens off-cluster) ---------
    echo "==================== VALIDATION ===================="
    grep -h "rows_per_s\|threads\|passes" "$OUT"/result_*.json 2>/dev/null || true
    echo "expect: operator ~flat near ceiling across threads/morsels; scalar plateaus at its morsel-bound max."
    echo "DONE. Artefacts in: $OUT (metrics_*_ep*, result_*.json, run_*.log, vllm-*.log)"
    echo "Pull home with e.g.:  rsync -av <clariden>:$OUT analysis/figures/data/sem_extract_ab/"
'

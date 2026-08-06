#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-sem-reason-len
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Cheap-reason follow-up to the Rec-1 schema ablation. id_reason recovered
# quality at high R but the per-row rationale is decode-heavy, so its throughput
# fell below LOTUS. Question: does a SHORTER reason keep the F1 gain at a lower
# token cost (higher rows/s)?
#
# Sweep FLOCK_SEM_REASON_WORDS (the rationale budget: instruction "<=N words" +
# schema string maxLength = 7N chars) at fixed FLOCK_SEM_SCHEMA=id_reason. Each
# (words, R) cell -> subdir id_reason_w<words>/, shaped like an rsweep run, so
# diagnose_rsweep.py runs per cell unchanged. Operator only, threads=1, cold
# vLLM per cell, same two-pass protocol (timing dump-off / verdict dump-on).
#
# w12 reproduces the schema-ablation id_reason point (maxLength ~84 vs the
# ablation's 80 -- immaterial); w3/w6 are the cheap variants.
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
    OUT="$FLOCK/analysis/results/sem_filter_reason_len/${SLURM_JOB_ID}"; mkdir -p "$OUT"
    mkdir -p "$HOME/.duckdb"

    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=1; BASE_PORT=8000; FULL_UTIL=0.90
    IN_FLIGHT="${IN_FLIGHT:-128}"; TIMEOUT_MS=120000
    MULT="${MULT:-48}"          # max_output_tokens = MULT*R; 48 covers R=32 * ~12-word reasons
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    ROWS="${ROWS:-2000}"
    WORDS_SWEEP="${WORDS_SWEEP:-3 6 12}"
    R_SWEEP="${R_SWEEP:-8 16 32}"

    echo "==================== ENV ===================="
    echo "OUT=$OUT rows=$ROWS inflight=$IN_FLIGHT mode=id_reason words=[$WORDS_SWEEP] R=[$R_SWEEP] mult=$MULT threads=1 (cold per cell)"
    # Job-private HOME: flock persists its model catalog at
    # $HOME/.duckdb/flock_storage and parallel jobs race the shared file
    # (NFS lock: "Conflicting lock is held in PID -3").
    JOB_HOME="/tmp/flock_home_${SLURM_JOB_ID:-local}"
    mkdir -p "$JOB_HOME/.duckdb"
    [ -x "$BIN" ]  || { echo "driver missing: $BIN (rebuild it)"; exit 1; }
    [ -f "$DATA" ] || { echo "dataset missing: $DATA"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv

    PIDS=(); ENDPOINTS=""
    start_fleet() {
        PIDS=(); ENDPOINTS=""
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i))
            CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
                --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
                --gpu-memory-utilization "$FULL_UTIL" --host 127.0.0.1 --port "$port" \
                --structured-outputs-config '\''{"backend": "xgrammar", "disable_any_whitespace": true}'\'' \
                > "$OUT/vllm-ep$i-$(date +%s).log" 2>&1 &
            PIDS+=($!)
            ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}http://127.0.0.1:$port/v1/chat/completions"
        done
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i)) ready=0
            for t in $(seq 1 180); do
                curl -sf "http://127.0.0.1:$port/health" >/dev/null && { ready=1; break; }
                kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "ep$i died on startup"; exit 1; }
                sleep 5
            done
            [ "$ready" -eq 1 ] || { echo "ep$i /health timeout"; exit 1; }
        done
        echo "fleet ready -> $ENDPOINTS"
    }
    stop_fleet() {
        for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
        wait 2>/dev/null || true; sleep 15; PIDS=()
    }
    trap stop_fleet EXIT

    snap() {  # snap <dir> <tag>
        for i in $(seq 0 $((NGPU-1))); do
            curl -s "http://127.0.0.1:$((BASE_PORT+i))/metrics" > "$1/metrics_${2}_ep$i.txt" || true
        done
    }

    for W in $WORDS_SWEEP; do
        WDIR="$OUT/id_reason_w${W}"; mkdir -p "$WDIR"
        for R in $R_SWEEP; do
            tag="r$R"
            echo "==================== words=$W R=$R (cold fleet) ===================="
            start_fleet

            # pass 1: TIMING (schema+reason on, dump off, burn-in on, cold cache)
            snap "$WDIR" "before_$tag"
            FLOCK_SEM_SCHEMA=id_reason FLOCK_SEM_REASON_WORDS="$W" \
                HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$DATA" \
                       --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                       --rewrite on --threads 1 --inflight "$IN_FLIGHT" --rows-per-request "$R" \
                       --max-out-mult "$MULT" --timeout-ms "$TIMEOUT_MS" \
                       --result-out "$WDIR/result_$tag.json" 2>&1 | tee "$WDIR/run_$tag.log"
            snap "$WDIR" "after_$tag"

            # pass 2: VERDICTS (dump on, skip burn-in, timing discarded)
            FLOCK_SEM_SCHEMA=id_reason FLOCK_SEM_REASON_WORDS="$W" \
            FLOCK_VERDICT_DUMP="$WDIR/verdicts_$tag.jsonl" \
                HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$DATA" \
                       --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                       --rewrite on --threads 1 --inflight "$IN_FLIGHT" --rows-per-request "$R" \
                       --max-out-mult "$MULT" --skip-burn-in 2>&1 | tee "$WDIR/verdict_$tag.log"
            echo "words=$W R=$R verdict lines=$(wc -l < "$WDIR/verdicts_$tag.jsonl" 2>/dev/null || echo 0) (expect $ROWS)"

            stop_fleet
        done
    done
    trap - EXIT

    echo "==================== VALIDATION ===================="
    for W in $WORDS_SWEEP; do
        echo "-- words=$W --"; grep -h "rows_per_s\|batch\|passes" "$OUT/id_reason_w${W}"/result_*.json 2>/dev/null || true
    done
    echo "DONE. Artefacts in: $OUT/id_reason_w<words>/"
    echo "Pull home:  rsync -av <clariden>:$OUT analysis/figures/data/sem_filter_reason_len/"
    echo "Diagnose per cell, e.g.:"
    echo "  python analysis/diagnose_rsweep.py analysis/figures/data/sem_filter_reason_len/${SLURM_JOB_ID}/id_reason_w3 --data <local Reviews.csv>"
'

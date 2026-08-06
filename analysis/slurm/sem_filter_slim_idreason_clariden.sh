#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-sem-slim-idreason
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Combined config: slim prompt (Rec-3) + id_reason schema (Rec-1), the one
# untested corner of the design space. slim fixes the PROMPT cost (throughput);
# id_reason fixes the BATCHED quality collapse (reasoning). Question: at high R,
# does combining them beat the current best point, slim+bool @R=1 (344 rows/s,
# F1 0.887)?
#
#   FLOCK_SEM_PROMPT=slim  + FLOCK_SEM_SCHEMA=id_reason   (both fire in RenderPrompt)
# Operator only, threads=1, cold vLLM per R, same two-pass protocol (timing
# dump-off / verdict dump-on). --max-out-mult 48 gives the per-row reason
# objects headroom (the guided schema + maxLength cap actual generation anyway).
#
# One config, swept over R -> the job dir IS an rsweep-shaped dir, so
# diagnose_rsweep.py runs on it directly. Compare its (rows/s, F1) points against
# the frontier (slim+bool, full+id_reason, LOTUS) to conclude the best config.
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
    OUT="$FLOCK/analysis/results/sem_filter_slim_idreason/${SLURM_JOB_ID}"; mkdir -p "$OUT"
    mkdir -p "$HOME/.duckdb"

    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=1; BASE_PORT=8000; FULL_UTIL=0.90
    IN_FLIGHT="${IN_FLIGHT:-128}"; TIMEOUT_MS=120000
    MULT="${MULT:-48}"
    REASON_WORDS="${REASON_WORDS:-12}"
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    ROWS="${ROWS:-2000}"
    R_SWEEP="${R_SWEEP:-1 8 16 32}"

    echo "==================== ENV ===================="
    echo "OUT=$OUT rows=$ROWS inflight=$IN_FLIGHT config=slim+id_reason(words=$REASON_WORDS) R=[$R_SWEEP] mult=$MULT threads=1 (cold per R)"
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

    snap() {  # snap <tag>
        for i in $(seq 0 $((NGPU-1))); do
            curl -s "http://127.0.0.1:$((BASE_PORT+i))/metrics" > "$OUT/metrics_${1}_ep$i.txt" || true
        done
    }

    for R in $R_SWEEP; do
        tag="r$R"
        echo "==================== slim+id_reason R=$R (cold fleet) ===================="
        start_fleet

        # pass 1: TIMING (both env on, dump off, burn-in on, cold cache)
        snap "before_$tag"
        FLOCK_SEM_PROMPT=slim FLOCK_SEM_SCHEMA=id_reason FLOCK_SEM_REASON_WORDS="$REASON_WORDS" \
            HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$DATA" \
                   --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                   --rewrite on --threads 1 --inflight "$IN_FLIGHT" --rows-per-request "$R" \
                   --max-out-mult "$MULT" --timeout-ms "$TIMEOUT_MS" \
                   --result-out "$OUT/result_$tag.json" 2>&1 | tee "$OUT/run_$tag.log"
        snap "after_$tag"

        # pass 2: VERDICTS (dump on, skip burn-in, timing discarded)
        FLOCK_SEM_PROMPT=slim FLOCK_SEM_SCHEMA=id_reason FLOCK_SEM_REASON_WORDS="$REASON_WORDS" \
        FLOCK_VERDICT_DUMP="$OUT/verdicts_$tag.jsonl" \
            HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$DATA" \
                   --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                   --rewrite on --threads 1 --inflight "$IN_FLIGHT" --rows-per-request "$R" \
                   --max-out-mult "$MULT" --skip-burn-in 2>&1 | tee "$OUT/verdict_$tag.log"
        echo "R=$R verdict lines=$(wc -l < "$OUT/verdicts_$tag.jsonl" 2>/dev/null || echo 0) (expect $ROWS)"

        stop_fleet
    done
    trap - EXIT

    echo "==================== VALIDATION ===================="
    grep -h "rows_per_s\|batch\|passes" "$OUT"/result_*.json 2>/dev/null || true
    echo "DONE. Artefacts in: $OUT"
    echo "Pull home:  rsync -av <clariden>:$OUT analysis/figures/data/sem_filter_slim_idreason/"
    echo "Diagnose:   python analysis/diagnose_rsweep.py analysis/figures/data/sem_filter_slim_idreason/${SLURM_JOB_ID} --data <local Reviews.csv>"
'
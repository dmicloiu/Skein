#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-sem-prompt-variants
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Q4 prompt-variant sweep (analysis/prompt_engineering.md): can input-side
# tweaks recover the batched-recall collapse WITHOUT reasoning's decode cost?
# All arms run FLOCK_SEM_PROMPT=slim + bool schema; FLOCK_SEM_VARIANTS composes
# the Q3 candidates (rowmajor sandwich symmetric count example chunk).
#
# Methodology (anti-overfitting): tune on the DEV half of sf_2000, report the
# winner on the TEST half. The split is md5(reviewId)-parity, generated here on
# first use by analysis/make_dev_test_split.py.
#
# Phases via env (same script, different invocation):
#   screen  (default): VARIANTS_SWEEP="base rowmajor sandwich symmetric count example chunk"
#                      R_SWEEP="8 32", SPLIT=dev, REPS=1
#   combine:           VARIANTS_SWEEP="<top singles + combos>" (e.g. "rowmajor,sandwich,symmetric")
#   confirm:           VARIANTS_SWEEP="base <winner>" R_SWEEP="1 2 4 8 16 32" SPLIT=test REPS=3
#
# Per cell: cold fleet, two-pass protocol (pass 1 timing / dump off; pass 2
# verdicts / dump on, untimed). Read-out: diagnose_rsweep.py per variant dir
# (F1, rows/s, computed tok/s, per-position miss-rates) against the SAME split csv.
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
    OUT="$FLOCK/analysis/results/sem_filter_prompt_variants/${SLURM_JOB_ID}"; mkdir -p "$OUT"
    mkdir -p "$HOME/.duckdb"

    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=1; BASE_PORT=8000; FULL_UTIL=0.90
    IN_FLIGHT="${IN_FLIGHT:-128}"; TIMEOUT_MS=120000
    SRC_DATA="${SRC_DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    SPLIT="${SPLIT:-dev}"                    # dev | test | full
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    REPS="${REPS:-1}"
    R_SWEEP="${R_SWEEP:-8 32}"
    # Space-separated FLOCK_SEM_VARIANTS values; "base" = plain slim (env empty).
    VARIANTS_SWEEP="${VARIANTS_SWEEP:-base rowmajor sandwich symmetric count example chunk}"

    # Dev/test split (md5(reviewId) parity; deterministic). "full" = the source csv.
    if [ "$SPLIT" = "full" ]; then
        DATA="$SRC_DATA"
    else
        SPLIT_DIR="$(dirname "$SRC_DATA")/variant_split"
        [ -f "$SPLIT_DIR/Reviews_$SPLIT.csv" ] || \
            python "$FLOCK/analysis/make_dev_test_split.py" "$SRC_DATA" --out-dir "$SPLIT_DIR"
        DATA="$SPLIT_DIR/Reviews_$SPLIT.csv"
    fi
    ROWS=$(( $(wc -l < "$DATA") - 1 ))

    echo "==================== ENV ===================="
    echo "OUT=$OUT split=$SPLIT rows=$ROWS reps=$REPS inflight=$IN_FLIGHT"
    echo "variants=[$VARIANTS_SWEEP] R=[$R_SWEEP] (slim + bool, threads=1, cold fleet per cell)"
    [ -x "$BIN" ]  || { echo "driver missing: $BIN (rebuild it)"; exit 1; }
    [ -f "$DATA" ] || { echo "dataset missing: $DATA"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    cp "$DATA" "$OUT/"   # freeze the exact split evaluated

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

    for V in $VARIANTS_SWEEP; do
        # base = plain slim (FLOCK_SEM_VARIANTS empty); dir name: commas -> +
        if [ "$V" = "base" ]; then VENV=""; else VENV="$V"; fi
        VDIR="$OUT/$(echo "$V" | tr , +)"; mkdir -p "$VDIR"
        for R in $R_SWEEP; do
        for REP in $(seq 1 "$REPS"); do
            # rep 1 keeps the plain r<R> tag so diagnose_rsweep.py finds its
            # result/metrics/verdicts; extra reps carry a _rep suffix.
            if [ "$REP" -gt 1 ]; then tag="r${R}_rep${REP}"; else tag="r$R"; fi
            echo "==================== variants=$V R=$R rep=$REP (cold fleet) ===================="
            start_fleet

            # pass 1: TIMING (dump off, burn-in on, cold cache)
            snap "$VDIR" "before_$tag"
            FLOCK_SEM_PROMPT=slim FLOCK_SEM_VARIANTS="$VENV" \
                "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$DATA" \
                       --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                       --rewrite on --threads 1 --inflight "$IN_FLIGHT" --rows-per-request "$R" \
                       --timeout-ms "$TIMEOUT_MS" \
                       --result-out "$VDIR/result_$tag.json" 2>&1 | tee "$VDIR/run_$tag.log"
            snap "$VDIR" "after_$tag"

            # pass 2: VERDICTS (dump on, skip burn-in, timing discarded); rep 1 only,
            # verdicts are ~deterministic under greedy decoding.
            if [ "$REP" -eq 1 ]; then
                FLOCK_SEM_PROMPT=slim FLOCK_SEM_VARIANTS="$VENV" \
                    FLOCK_VERDICT_DUMP="$VDIR/verdicts_r$R.jsonl" \
                    "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$DATA" \
                           --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                           --rewrite on --threads 1 --inflight "$IN_FLIGHT" --rows-per-request "$R" \
                           --skip-burn-in 2>&1 | tee "$VDIR/verdict_r$R.log"
                echo "variants=$V R=$R verdict lines=$(wc -l < "$VDIR/verdicts_r$R.jsonl" 2>/dev/null || echo 0) (expect $ROWS)"
            fi

            stop_fleet
        done
        done
    done
    trap - EXIT

    echo "==================== VALIDATION ===================="
    for V in $VARIANTS_SWEEP; do
        echo "-- $V --"; grep -h "rows_per_s\|batch\|passes" "$OUT/$(echo "$V" | tr , +)"/result_*.json 2>/dev/null || true
    done
    echo "DONE. Artefacts in: $OUT/<variant>/"
    echo "Pull home:  rsync -av <clariden>:$OUT analysis/figures/data/sem_filter_prompt_variants/"
    echo "Diagnose:   python analysis/diagnose_rsweep.py analysis/figures/data/sem_filter_prompt_variants/${SLURM_JOB_ID}/<variant> --data <local Reviews_${SPLIT}.csv (copied into OUT)>"
'

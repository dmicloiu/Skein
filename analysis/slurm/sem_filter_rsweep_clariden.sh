#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-sem-filter-rsweep
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# sem_filter batch-size (R) sweep for the R=1 -> R=32 recall-collapse diagnosis.
# Operator (PhysicalSemFilter, rewrite=on) only, threads=1 (2000 rows = 1 morsel
# anyway; single-thread => row_id == table scan order, the join key the
# diagnostic relies on). One cold vLLM endpoint per R.
#
# TWO PASSES PER R, deliberately separated so the verdict dump can never touch a
# measured number:
#   1. TIMING pass  -- dump OFF, burn-in ON, cold cache. This is the untouched
#      committed code path; its result_r<R>.json + /metrics deltas are the real
#      throughput / engine-efficiency at this R.
#   2. VERDICT pass -- FLOCK_VERDICT_DUMP set, --skip-burn-in, timing DISCARDED.
#      Emits verdicts_r<R>.jsonl: one line per row {id,pos,v}. Greedy decoding
#      (temperature=0) makes these deterministic, so the quality they encode is
#      the same quality the timing pass produced.
# The dump therefore only ever runs in a pass whose wall-clock we throw away.
#
# COLD FLEET PER R: vllm runs with --enable-prefix-caching; a warm instance would
# let a later R hit an earlier R's cached review prefixes and report inflated
# throughput. Fresh endpoint per R (same discipline as sem_filter_ab_clariden.sh).
#
# Output feeds analysis/diagnose_rsweep.py (run off-cluster after rsync).
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
    OUT="$FLOCK/analysis/results/sem_filter_rsweep/${SLURM_JOB_ID}"; mkdir -p "$OUT"

    # flock persists its model/secret catalog at ~/.duckdb/flock_storage; its
    # CreateDirectory is non-recursive, so the ~/.duckdb parent must exist first.
    mkdir -p "$HOME/.duckdb"

    # ---- experiment parameters --------------------------------------------
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=1; BASE_PORT=8000; FULL_UTIL=0.90
    IN_FLIGHT="${IN_FLIGHT:-128}"; TIMEOUT_MS=120000
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    ROWS="${ROWS:-2000}"
    R_SWEEP="${R_SWEEP:-1 2 4 8 16 32}"

    echo "==================== ENV ===================="
    echo "OUT=$OUT  rows=$ROWS  inflight=$IN_FLIGHT  R_sweep=[$R_SWEEP]  threads=1  (cold fleet per R)"
    [ -x "$BIN" ]  || { echo "driver missing: $BIN (build it first)"; exit 1; }
    [ -f "$DATA" ] || { echo "dataset missing: $DATA"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv

    # ---- fleet helpers (mirror sem_filter_ab_clariden.sh) ------------------
    PIDS=(); ENDPOINTS=""
    start_fleet() {
        PIDS=(); ENDPOINTS=""
        for i in $(seq 0 $((NGPU-1))); do
            local port=$((BASE_PORT+i))
            # disable_any_whitespace: force COMPACT guided-JSON output. Without it,
            # xgrammar lets the model emit whitespace between array elements and
            # greedy decoding stalls on the last verdict -> truncated JSON. Needs
            # the backend pinned to xgrammar.
            CUDA_VISIBLE_DEVICES=$i vllm serve "$MODEL" \
                --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
                --gpu-memory-utilization "$FULL_UTIL" --host 127.0.0.1 --port "$port" \
                --structured-outputs-config '\''{"backend": "xgrammar", "disable_any_whitespace": true}'\'' \
                > "$OUT/vllm-ep$i-$(date +%s).log" 2>&1 &
            PIDS+=($!)
            # Operator posts chat requests to /v1/chat/completions (DeriveBaseUrl in
            # the driver strips back to /v1 for the scalar secret; unused here).
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

    common_args=(--endpoints "" --model "$MODEL" --data "$DATA" --text-col "$TEXT_COL"
                 --prompt "$PROMPT" --rows "$ROWS" --rewrite on --threads 1
                 --inflight "$IN_FLIGHT" --timeout-ms "$TIMEOUT_MS")

    for R in $R_SWEEP; do
        tag="r$R"
        echo "==================== R=$R (cold fleet) ===================="
        start_fleet
        common_args[1]="$ENDPOINTS"   # fill in the endpoints for this fleet

        # --- pass 1: TIMING (dump off, burn-in on, cold cache) --------------
        snap "before_$tag"
        "$BIN" "${common_args[@]}" --rows-per-request "$R" \
               --result-out "$OUT/result_$tag.json" 2>&1 | tee "$OUT/run_$tag.log"
        snap "after_$tag"

        # --- pass 2: VERDICTS (dump on, skip burn-in, timing discarded) -----
        FLOCK_VERDICT_DUMP="$OUT/verdicts_$tag.jsonl" \
            "$BIN" "${common_args[@]}" --rows-per-request "$R" \
                   --skip-burn-in 2>&1 | tee "$OUT/verdict_$tag.log"
        vlines=$(wc -l < "$OUT/verdicts_$tag.jsonl" 2>/dev/null || echo 0)
        echo "R=$R verdict lines=$vlines (expect $ROWS)"

        stop_fleet
    done
    trap - EXIT

    echo "==================== VALIDATION ===================="
    grep -h "rows_per_s\|batch\|passes" "$OUT"/result_*.json 2>/dev/null || true
    echo "expect: passes (survivors) FALL as R grows (recall collapse); rows_per_s RISES with R."
    echo "DONE. Artefacts in: $OUT (result_r*.json, verdicts_r*.jsonl, metrics_*_ep*, *_r*.log)"
    echo "Pull home:  rsync -av <clariden>:$OUT analysis/figures/data/sem_filter_rsweep/"
    echo "Diagnose:   python analysis/diagnose_rsweep.py analysis/figures/data/sem_filter_rsweep/${SLURM_JOB_ID} --data <local Reviews.csv>"
'

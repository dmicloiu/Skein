#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-sem-prompt-slim
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Rec-3: does slimming the prompt lift low-R throughput at unchanged quality?
# flock's R=1 loses to LOTUS on rows/s (136 vs 258) despite winning F1 and tok/s;
# the suspected cause is the heavy 7-section META_PROMPT (~505 prompt-tok/row of
# image/audio boilerplate vs LOTUS's ~129). FLOCK_SEM_PROMPT=slim swaps it for a
# lean text-only head + the same tuples.
#
# Runs BOTH prompt heads in ONE session (same vLLM build) for a clean A/B:
#   FLOCK_SEM_PROMPT unset -> full  (the committed META_PROMPT)
#   FLOCK_SEM_PROMPT=slim  -> lean head
# bool schema (no FLOCK_SEM_SCHEMA), operator only, threads=1, cold vLLM per cell,
# same two-pass protocol (timing dump-off / verdict dump-on). Low R only -- the
# fixed head is amortised away at high R, so slimming can only matter at low R.
#
# Read-out per cell (diagnose_rsweep.py): F1 (must hold ~R=1's 0.91), rows/s,
# tok/s, and -- via the metrics -- APC hit% and prefill_ms. The question is
# whether rows/s climbs toward LOTUS's 258 without F1 regressing.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$HOME/projects/flock"
mkdir -p logs
EDF="$HOME/projects/sembench/ngc-pytorch-vllm.toml"

srun -ul --environment="$EDF" bash -c '
    set -uo pipefail
    export NO_PROXY="localhost,127.0.0.1"; export no_proxy="localhost,127.0.0.1"
    # Serve from the local HF cache only: vllm queries the Hub file-list API on
    # every boot, and a cold fleet per cell trips HF rate limiting (429 = fatal
    # boot, even with weights cached). Boot-path only; the timed region is
    # unaffected. This is what killed 3009798/3009799/3009805.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sembench

    FLOCK="$HOME/projects/flock"; SEMBENCH="$HOME/projects/sembench"
    BIN="$FLOCK/build/flock_sem_filter_vllm_integration"
    # Canonical tuple encoding for the evaluation; TUPLE_FORMAT=XML runs the
    # controlled encoding comparison against the pre-standardisation numbers. It is
    # part of the artefact path so json and XML results can never overwrite each other.
    TUPLE_FORMAT="${TUPLE_FORMAT:-json}"
    OUT="$FLOCK/analysis/results/sem_filter_prompt_slim_${TUPLE_FORMAT}/${SLURM_JOB_ID}"
    mkdir -p "$OUT"
    mkdir -p "$HOME/.duckdb"

    MODEL="Qwen/Qwen2.5-7B-Instruct"
    NGPU=1; BASE_PORT=8000; FULL_UTIL=0.90
    IN_FLIGHT="${IN_FLIGHT:-128}"; TIMEOUT_MS=120000
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    ROWS="${ROWS:-2000}"
    PROMPTS="${PROMPTS:-full slim}"
    R_SWEEP="${R_SWEEP:-1 2 4}"

    echo "==================== ENV ===================="
    echo "OUT=$OUT rows=$ROWS inflight=$IN_FLIGHT prompts=[$PROMPTS] R=[$R_SWEEP]"
    echo "tuple_format=$TUPLE_FORMAT bool schema threads=1 (cold per cell)"
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

    # Encoding gate on the prompt the run actually rendered (FLOCK_PROMPT_DUMP
    # captures the burn-in prompt, same template as the timed query). Catches a
    # stale driver or an unapplied tuple_format before the sweep spends node time.
    # json only: slim at R>1 renders row-major id lines under either setting, so
    # absence of <row> is not evidence that an XML switch took effect.
    assert_tuple_encoding() {  # assert_tuple_encoding <prompt-dump>
        [ -s "$1" ] || { echo "GATE FAIL: no rendered prompt at $1"; exit 1; }
        if [ "$TUPLE_FORMAT" = "json" ] && grep -qE "<row>|<column>" "$1"; then
            echo "GATE FAIL: XML tuple scaffolding in $1 -- tuple_format=json did not apply"
            exit 1
        fi
        echo "gate ok: rendered prompt matches tuple_format=$TUPLE_FORMAT ($1)"
    }

    for P in $PROMPTS; do
        # full = committed META_PROMPT (env unset); slim = lean head.
        if [ "$P" = "slim" ]; then PENV="slim"; else PENV=""; fi
        PDIR="$OUT/$P"; mkdir -p "$PDIR"
        for R in $R_SWEEP; do
            tag="r$R"
            echo "==================== prompt=$P R=$R (cold fleet) ===================="
            start_fleet

            # pass 1: TIMING (verdict dump off, burn-in on, cold cache). The prompt
            # dump is the burn-in one -- synthetic rows, real template -- and feeds
            # the encoding gate below at no cost.
            snap "$PDIR" "before_$tag"
            FLOCK_SEM_PROMPT="$PENV" FLOCK_PROMPT_DUMP="$PDIR/prompt_$tag.txt" \
                HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$DATA" \
                       --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                       --rewrite on --threads 1 --inflight "$IN_FLIGHT" --rows-per-request "$R" \
                       --timeout-ms "$TIMEOUT_MS" --tuple-format "$TUPLE_FORMAT" \
                       --result-out "$PDIR/result_$tag.json" 2>&1 | tee "$PDIR/run_$tag.log"
            snap "$PDIR" "after_$tag"
            assert_tuple_encoding "$PDIR/prompt_$tag.txt"

            # pass 2: VERDICTS (dump on, skip burn-in, timing discarded)
            FLOCK_SEM_PROMPT="$PENV" FLOCK_VERDICT_DUMP="$PDIR/verdicts_$tag.jsonl" \
                HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$DATA" \
                       --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                       --rewrite on --threads 1 --inflight "$IN_FLIGHT" --rows-per-request "$R" \
                       --tuple-format "$TUPLE_FORMAT" \
                       --skip-burn-in 2>&1 | tee "$PDIR/verdict_$tag.log"
            echo "prompt=$P R=$R verdict lines=$(wc -l < "$PDIR/verdicts_$tag.jsonl" 2>/dev/null || echo 0) (expect $ROWS)"

            stop_fleet
        done
    done
    trap - EXIT

    echo "==================== VALIDATION ===================="
    for P in $PROMPTS; do
        echo "-- $P --"; grep -h "rows_per_s\|batch\|passes" "$OUT/$P"/result_*.json 2>/dev/null || true
    done
    echo "DONE. Artefacts in: $OUT/<full|slim>/ (tuple_format=$TUPLE_FORMAT)"
    echo "Pull home:  rsync -av <clariden>:$OUT analysis/figures/data/sem_filter_prompt_slim_${TUPLE_FORMAT}/"
    echo "Diagnose:   python analysis/diagnose_rsweep.py analysis/figures/data/sem_filter_prompt_slim_${TUPLE_FORMAT}/${SLURM_JOB_ID}/slim --data <local Reviews.csv>"
'

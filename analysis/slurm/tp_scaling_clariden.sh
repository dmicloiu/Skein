#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-tp-scaling
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# TP (tensor-parallel) scaling: one vLLM endpoint sharded across 1/2/4 GPUs
# (--tensor-parallel-size), operator (rewrite=on) vs scalar llm_filter (off),
# both on the FULL prompt (prompt parity; FLOCK_SEM_PROMPT unset), R=1.
#
# Claim under test: the async operator tracks the rising endpoint ceiling (its
# concurrency is the in_flight_cap knob), the scalar's min(threads,morsels)=1
# concurrency cannot. Operator cap sweep per TP (128 was calibrated at TP=1):
#   TP=1: 128        TP=2: 128 256        TP=4: 128 256 512
#
# TUNABLE: TP_SWEEP and ARMS split the sweep into short jobs instead of one run.
#
# MORSEL MODE: MORSELS=128 materialises
# Example:
#   MORSELS=128 DATA=<big Reviews.csv> TP_SWEEP="1" \
#       sbatch --gpus-per-node=1 --time=02:00:00 analysis/slurm/tp_scaling_clariden.sh
#
# Cold fleet per measurement. During each timed run a background sampler
# records the vLLM scheduler gauges (num_requests_running/waiting,
# gpu_cache_usage_perc) -- the saturation evidence that before/after counter
# snapshots cannot provide.
#
# Validity gates (the arms share prompt bytes, so token deltas can NOT catch a
# stale binary here):
#   - TP engaged: after boot, exactly TP GPUs must hold >20 GB (weights + KV).
#   - behavioral A/B gate: scalar elapsed >= 3x operator elapsed per TP
#     (a stale binary makes the arms identical -> ratio ~1 -> fail loudly).
#     Only enforceable when both arms ran in this job.
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
    # Tuple encoding is part of the artefact path: the driver renders json by
    # default now, and the pre-standardisation results were XML, so the two must
    # never share a directory. TUPLE_FORMAT=XML reruns the old encoding.
    TUPLE_FORMAT="${TUPLE_FORMAT:-json}"
    OUT="$FLOCK/analysis/results/tp_scaling_${TUPLE_FORMAT}/${SLURM_JOB_ID}"; mkdir -p "$OUT"

    # Job-private HOME for the driver: flock persists its model catalog at
    # $HOME/.duckdb/flock_storage and parallel rep jobs race the shared file
    # ("Model not found"). Node-local isolation removes it; the .duckdb parent
    # must pre-exist (flock CreateDirectory is non-recursive).
    JOB_HOME="/tmp/flock_home_${SLURM_JOB_ID:-local}"
    mkdir -p "$JOB_HOME/.duckdb"

    # Full prompt on both arms: make sure no slim/variant knob leaks in.
    unset FLOCK_SEM_PROMPT FLOCK_SEM_VARIANTS

    MODEL="Qwen/Qwen2.5-7B-Instruct"
    PORT=8000; FULL_UTIL=0.90; TIMEOUT_MS=120000
    ENDPOINT="http://127.0.0.1:$PORT/v1/chat/completions"
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    ROWS="${ROWS:-2000}"
    TP_SWEEP="${TP_SWEEP:-1 2 4}"
    ARMS="${ARMS:-op scalar}"
    CAPS_TP1="${CAPS_TP1:-128}"
    CAPS_TP2="${CAPS_TP2:-128 256}"
    CAPS_TP4="${CAPS_TP4:-128 256 512}"
    OP_THREADS="${OP_THREADS:-1}"         # operator is thread-independent
    SCALAR_THREADS="${SCALAR_THREADS:-8}" # 1 in-memory morsel -> concurrency 1 anyway
    VERDICTS="${VERDICTS:-1}"             # untimed dump pass (operator, first cap of each TP)

    # ---- morsel mode (heroic scalar; see header) ---------------------------
    MORSELS="${MORSELS:-}"
    ROW_GROUP_SIZE="${ROW_GROUP_SIZE:-2048}"   # must be a multiple of 2048
    MORSEL_THREADS="${MORSEL_THREADS:-128}"
    MORSEL_DB_DIR="${MORSEL_DB_DIR:-/tmp}"     # node-local; avoid Lustre $HOME
    BESTCAP_TP1="${BESTCAP_TP1:-128}"; BESTCAP_TP2="${BESTCAP_TP2:-256}"; BESTCAP_TP4="${BESTCAP_TP4:-512}"
    MSUF=""
    if [ -n "$MORSELS" ]; then
        ROWS=$((MORSELS * ROW_GROUP_SIZE))
        VERDICTS=0
        MSUF="_m${MORSELS}"
        [ -f "$DATA" ] || { echo "dataset missing: $DATA"; exit 1; }
        AVAIL=$(($(wc -l < "$DATA") - 1))
        [ "$AVAIL" -ge "$ROWS" ] || { echo "FATAL: morsel mode needs >= $ROWS rows, $DATA has $AVAIL -- point DATA at a larger Reviews.csv"; exit 1; }
    fi

    echo "==================== ENV ===================="
    echo "OUT=$OUT rows=$ROWS tp_sweep=[$TP_SWEEP] arms=[$ARMS] morsels=[${MORSELS:-1 (in-memory)}]"
    if [ -n "$MORSELS" ]; then
        echo "morsel mode: scalar threads=$MORSEL_THREADS, op caps tp1=$BESTCAP_TP1 tp2=$BESTCAP_TP2 tp4=$BESTCAP_TP4 (no sweep)"
    else
        echo "caps: tp1=[$CAPS_TP1] tp2=[$CAPS_TP2] tp4=[$CAPS_TP4]; cold fleet per run"
    fi
    [ -x "$BIN" ]  || { echo "driver missing: $BIN (build it first)"; exit 1; }
    [ -f "$DATA" ] || { echo "dataset missing: $DATA"; exit 1; }
    MAXTP=0; for T in $TP_SWEEP; do [ "$T" -gt "$MAXTP" ] && MAXTP=$T; done
    NGPU_NODE=$(nvidia-smi --list-gpus | wc -l)
    [ "$NGPU_NODE" -ge "$MAXTP" ] || { echo "FATAL: TP_SWEEP needs $MAXTP GPUs, job has $NGPU_NODE (match --gpus-per-node at submit)"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv

    # ---- fleet: ONE endpoint, sharded across $1 GPUs -----------------------
    VPID=""; VLLM_LOG=""
    start_fleet_once() {  # $1 = TP
        local tp=$1
        local devs; devs=$(seq -s, 0 $((tp-1)))
        VLLM_LOG="$OUT/vllm-tp${tp}-$(date +%s).log"
        CUDA_VISIBLE_DEVICES=$devs vllm serve "$MODEL" \
            --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
            --tensor-parallel-size "$tp" \
            --gpu-memory-utilization "$FULL_UTIL" --host 127.0.0.1 --port "$PORT" \
            --structured-outputs-config '\''{"backend": "xgrammar", "disable_any_whitespace": true}'\'' \
            > "$VLLM_LOG" 2>&1 &
        VPID=$!
        local ready=0
        for t in $(seq 1 180); do
            if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null \
               && curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
                ready=1; break
            fi
            kill -0 "$VPID" 2>/dev/null || { echo "vllm died on startup (see $VLLM_LOG)"; return 1; }
            sleep 5
        done
        [ "$ready" -eq 1 ] || { echo "vllm readiness timeout (tp=$tp)"; return 1; }
        # TP-engaged gate: exactly tp GPUs hold weights+KV (~0.9 util each). A
        # silently degraded TP would corrupt the whole axis.
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>20000" | wc -l)
        [ "$used" -eq "$tp" ] || { echo "FATAL: tp=$tp requested but $used GPUs loaded"; return 1; }
        grep -q "tensor_parallel_size=$tp" "$VLLM_LOG" \
            || echo "WARN: tensor_parallel_size=$tp not found in vllm log (format drift?)"
        echo "fleet ready: tp=$tp on GPUs [$devs] -> $ENDPOINT"
    }
    start_fleet() {  # $1 = TP; one retry (transient boot flakes)
        start_fleet_once "$1" && return 0
        echo "fleet boot failed; retrying once"
        stop_fleet
        start_fleet_once "$1" || { echo "fleet failed twice (tp=$1)"; exit 1; }
    }
    # TERM the vllm process subtree bottom-up, rooted at $VPID (the launcher):
    # hits the API server + EngineCore + TP workers but never the driver, which
    # is the PARENT of VPID, not a descendant. The old pkill -f "vllm serve"
    # used -f (full-cmdline match) and matched the bash -c step itself, whose
    # argv literally contains the string "vllm serve", SIGKILLing the whole job
    # at the first stop_fleet -- right after the operator arm, before scalar ran.
    kill_tree() {
        local p=$1 c
        for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
        kill -TERM "$p" 2>/dev/null || true
    }
    stop_fleet() {
        [ -n "$VPID" ] && kill_tree "$VPID"
        wait 2>/dev/null || true
        # Wait for HBM to actually drain or the next boot OOMs / fails the gate.
        for t in $(seq 1 24); do
            busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>5000" | wc -l)
            [ "$busy" -eq 0 ] && break
            sleep 5
        done
        VPID=""
    }
    trap stop_fleet EXIT

    snap() {  # snap <tag>: /metrics counter snapshot (single endpoint = ep0)
        curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics_${1}_ep0.txt" || true
    }
    SAMPLER_PID=""
    start_sampler() {  # $1 = tag; scheduler gauges every 3s -> samples_<tag>.csv
        local f="$OUT/samples_$1.csv"
        echo "epoch,running,waiting,kv_usage" > "$f"
        ( while :; do
            curl -s "http://127.0.0.1:$PORT/metrics" | awk -v ts="$(date +%s)" '\''
                /^vllm:num_requests_running/ {r=$NF}
                /^vllm:num_requests_waiting/ {w=$NF}
                /^vllm:(gpu|kv)_cache_usage_perc/ {k=$NF}
                END {print ts "," r "," w "," k}'\'' >> "$f"
            sleep 3
          done ) &
        SAMPLER_PID=$!
    }
    stop_sampler() { [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""; }

    run() {  # run <on|off> <threads> <cap> <tag>: one timed measurement
        local rgs_args=()
        if [ -n "$MORSELS" ]; then
            # rows/ROW_GROUP_SIZE scan morsels via a fresh node-local attached DB
            rgs_args=(--row-group-size "$ROW_GROUP_SIZE" --attach-db "$MORSEL_DB_DIR/tp_morsel_${SLURM_JOB_ID:-x}_$4.db")
        fi
        echo "------ RUN $4 (rewrite=$1 threads=$2 cap=$3 rows=$ROWS R=1) ------"
        snap "before_$4"
        start_sampler "$4"
        HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINT" --model "$MODEL" \
            --data "$DATA" --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
            --rewrite "$1" --threads "$2" --inflight "$3" --rows-per-request 1 \
            --timeout-ms "$TIMEOUT_MS" --tuple-format "$TUPLE_FORMAT" "${rgs_args[@]}" \
            --result-out "$OUT/result_$4.json" 2>&1 | tee "$OUT/run_$4.log"
        stop_sampler
        snap "after_$4"
    }

    elapsed_of() { sed -n "s/.*\"elapsed_s\": *\([0-9.]*\).*/\1/p" "$1" | head -1; }

    for TP in $TP_SWEEP; do
        if [ -n "$MORSELS" ]; then
            bcvar="BESTCAP_TP${TP}"; CAPS="${!bcvar}"   # settled cap only
        else
            capsvar="CAPS_TP${TP}"; CAPS="${!capsvar}"
        fi
        echo "==================== TP=$TP (caps=[$CAPS], arms=[$ARMS]${MSUF:+, morsels=$MORSELS}) ===================="

        for ARM in $ARMS; do
        if [ "$ARM" = "op" ]; then
            DID_VERDICT=0
            for CAP in $CAPS; do
                start_fleet "$TP"
                run on "$OP_THREADS" "$CAP" "op_tp${TP}_c${CAP}${MSUF}"
                if [ "$DID_VERDICT" -eq 0 ] && [ "$VERDICTS" -eq 1 ]; then
                    # untimed verdict pass, same fleet (timing discarded; greedy
                    # -> deterministic, cap-invariant). F1-vs-TP computed off-cluster.
                    HOME="$JOB_HOME" FLOCK_VERDICT_DUMP="$OUT/verdicts_tp${TP}.jsonl" \
                        "$BIN" --endpoints "$ENDPOINT" --model "$MODEL" --data "$DATA" \
                               --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
                               --rewrite on --threads "$OP_THREADS" --inflight "$CAP" \
                               --rows-per-request 1 --timeout-ms "$TIMEOUT_MS" \
                               --tuple-format "$TUPLE_FORMAT" \
                               --skip-burn-in 2>&1 | tee "$OUT/verdict_tp${TP}.log"
                    echo "tp=$TP verdict lines=$(wc -l < "$OUT/verdicts_tp${TP}.jsonl" 2>/dev/null || echo 0) (expect $ROWS)"
                    DID_VERDICT=1
                fi
                stop_fleet
            done
        else
            if [ -n "$MORSELS" ]; then TH="$MORSEL_THREADS"; else TH="$SCALAR_THREADS"; fi
            start_fleet "$TP"
            run off "$TH" 128 "scalar_tp${TP}${MSUF}"
            stop_fleet
        fi
        done

        FIRSTCAP=$(set -- $CAPS; echo $1)
        OPJ="$OUT/result_op_tp${TP}_c${FIRSTCAP}${MSUF}.json"; SCJ="$OUT/result_scalar_tp${TP}${MSUF}.json"
        if [ -n "$MORSELS" ] && [ -f "$SCJ" ]; then
            # Heroic scalar legitimately approaches the operator (~1.15x at
            # TP=1), so the elapsed-ratio gate cannot separate healthy from
            # stale here. Gate instead on the scalar having PARALLELIZED:
            # rows/s far above its serial (concurrency-1) level of ~17-29.
            scr=$(sed -n "s/.*\"rows_per_s\": *\([0-9.]*\).*/\1/p" "$SCJ" | head -1)
            awk -v r="$scr" -v tp="$TP" "BEGIN{ if (r==\"\") exit 2;
                printf \"morsel gate tp=%d: scalar rows/s = %.1f (serial ~17-29)\n\", tp, r;
                exit (r>=60.0 ? 0 : 1) }" \
                || { echo "FATAL: heroic scalar did not parallelize at TP=$TP (min(threads,morsels) broken?)"; exit 1; }
        elif [ -f "$OPJ" ] && [ -f "$SCJ" ]; then
            # Behavioral A/B gate (prompt parity makes token deltas useless
            # here): concurrency ~120 vs 1 => operator must be >=3x faster or
            # the binary is stale / the rewrite did not engage.
            ope=$(elapsed_of "$OPJ"); sce=$(elapsed_of "$SCJ")
            awk -v o="$ope" -v s="$sce" -v tp="$TP" "BEGIN{ if (o==\"\" || s==\"\") exit 2;
                r=s/o; printf \"A/B gate tp=%d: scalar/operator elapsed ratio = %.1f\n\", tp, r;
                exit (r>=3.0 ? 0 : 1) }" \
                || { echo "FATAL: arms too similar at TP=$TP -- stale binary / rewrite not engaged?"; exit 1; }
        else
            echo "A/B gate tp=$TP SKIPPED (single-arm job; cross-check via summarize_tp.py concurrency column)"
        fi
    done
    trap - EXIT

    echo "==================== VALIDATION ===================="
    grep -h "rewrite\|rows_per_s\|inflight" "$OUT"/result_*.json 2>/dev/null || true
    echo "expect: operator rows/s rises with TP (pick the saturating cap per TP);"
    echo "        scalar ~flat (concurrency 1 gets only the per-request TP latency gain)."
    echo "DONE. Artefacts in: $OUT"
    echo "Pull home:  rsync -av <clariden>:$OUT analysis/figures/data/tp_scaling_${TUPLE_FORMAT}/"
    echo "Summarise:  python analysis/summarize_tp.py"
'
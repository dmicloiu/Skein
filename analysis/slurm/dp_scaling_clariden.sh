#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-dp-scaling
#SBATCH --time=01:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# DP (data-parallel) scaling: N independent vLLM endpoints (one per GPU slice,
# CUDA_VISIBLE_DEVICES), the operator dispatching across the fleet via
# semantic_endpoints + round_robin EndpointRouter. Horizontal complement of
# tp_scaling_clariden.sh (one endpoint, bigger shards). FULL prompt
# (FLOCK_SEM_PROMPT unset), R=1 on both arms -- prompt-byte parity with the TP
# study so TP+DP merge onto one scale-out axis.
#
# Hypotheses: H1 near-linear rows/s in N (no cross-GPU collectives);
# H2 replicas beat shards at a fixed 4-GPU budget (4xTP1 > 2xTP2 > 1xTP4);
# H3 in_flight_cap must scale with the fleet (cap=128*N, starvation cell below);
# H4 F1 invariant in N (drift => routing/row-loss bug).
#
# CONFIGS is a list of N:TP:CAP fleet configs, one cold fleet each. Two job
# presets (submit as SEPARATE jobs; boots dominate, jobs stay ~1h):
#   curve (default):  CONFIGS="1:1:128 2:1:256 4:1:512 4:1:128"  ARMS="op scalar"
#                     (last cell = H3 starvation probe: N=4 at the unscaled cap)
#   grid:             CONFIGS="4:1:512 2:2:512 1:4:512"          ARMS="op"
#                     (fixed 4-GPU budget, total cap 512; own 4xTP1 anchor so
#                     the replicas-vs-shards comparison is in-job)
#
# ROWS/DATA: timed runs default to 32000 rows -- at N=4 (~900 rows/s) 2000 rows
# would finish in ~2-4s, too short for timing or the 3s gauge sampler. Point
# DATA at a large Reviews.csv (same file as TP morsel mode); the row check is
# fail-loud. Verdict (F1) passes and the scalar gate arm stay on GOLD_DATA
# (sf_2000, 2000 gold rows).
#
# Validity gates (prompt parity => token deltas cannot catch a stale binary):
#   - fleet engaged: after boot, exactly N*TP GPUs hold >20 GB.
#   - rewrite engaged: operator rows/s >= 60 (serial scalar is ~17-29); works
#     in single-arm jobs.
#   - A/B gate (curve jobs): operator/scalar rows/s ratio >= 3.
#   - H4: verdict line count == GOLD_ROWS (row loss = routing bug).
#   - balance: per-endpoint request share ~1/N under round_robin (WARN only;
#     rigorous panel off-cluster).
#
# Per-endpoint artefacts for the balance panel + GPU-vs-client bottleneck call:
# /metrics snapshots and scheduler-gauge samples per endpoint, plus a driver
# /proc/<pid>/stat CPU sampler (a pegged client IO thread is directly visible).
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
    OUT="$FLOCK/analysis/results/dp_scaling/${SLURM_JOB_ID}"; mkdir -p "$OUT"

    # Job-private HOME: flock persists its model catalog at
    # $HOME/.duckdb/flock_storage and parallel rep jobs race the shared file.
    JOB_HOME="/tmp/flock_home_${SLURM_JOB_ID:-local}"
    mkdir -p "$JOB_HOME/.duckdb"

    # Full prompt on both arms: no slim/variant knob may leak in.
    unset FLOCK_SEM_PROMPT FLOCK_SEM_VARIANTS

    MODEL="Qwen/Qwen2.5-7B-Instruct"
    BASE_PORT=8000; FULL_UTIL=0.90; TIMEOUT_MS=120000
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    ROWS="${ROWS:-32000}"
    GOLD_DATA="${GOLD_DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    GOLD_ROWS="${GOLD_ROWS:-2000}"
    CONFIGS="${CONFIGS:-1:1:128 2:1:256 4:1:512 4:1:128}"
    ARMS="${ARMS:-op scalar}"
    OP_THREADS="${OP_THREADS:-1}"         # operator is thread-independent
    SCALAR_THREADS="${SCALAR_THREADS:-8}" # 1 in-memory morsel -> concurrency 1 anyway
    VERDICTS="${VERDICTS:-1}"             # untimed F1 dump, first config per (N,TP)

    echo "==================== ENV ===================="
    echo "OUT=$OUT rows=$ROWS gold_rows=$GOLD_ROWS configs=[$CONFIGS] arms=[$ARMS]"
    [ -x "$BIN" ]  || { echo "driver missing: $BIN (build it first)"; exit 1; }
    for f in "$DATA" "$GOLD_DATA"; do
        [ -f "$f" ] || { echo "dataset missing: $f"; exit 1; }
    done
    AVAIL=$(($(wc -l < "$DATA") - 1))
    [ "$AVAIL" -ge "$ROWS" ] || { echo "FATAL: need >= $ROWS rows, $DATA has $AVAIL -- point DATA at a larger Reviews.csv"; exit 1; }
    MAXG=0
    for CFG in $CONFIGS; do
        IFS=: read -r N TP CAP <<< "$CFG"
        G=$((N * TP)); [ "$G" -gt "$MAXG" ] && MAXG=$G
    done
    NGPU_NODE=$(nvidia-smi --list-gpus | wc -l)
    [ "$NGPU_NODE" -ge "$MAXG" ] || { echo "FATAL: CONFIGS needs $MAXG GPUs, job has $NGPU_NODE"; exit 1; }
    nvidia-smi --query-gpu=index,name,memory.total --format=csv

    # ---- fleet: N endpoints, each sharded across TP GPUs --------------------
    PIDS=(); ENDPOINTS=""; NEP=0; FLEET_TP=0
    start_fleet_once() {  # $1 = N endpoints, $2 = TP per endpoint
        local n=$1 tp=$2 i
        PIDS=(); ENDPOINTS=""; NEP=$n; FLEET_TP=$tp
        for i in $(seq 0 $((n-1))); do
            local port=$((BASE_PORT+i))
            local devs; devs=$(seq -s, $((i*tp)) $((i*tp+tp-1)))
            CUDA_VISIBLE_DEVICES=$devs vllm serve "$MODEL" \
                --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
                --tensor-parallel-size "$tp" \
                --gpu-memory-utilization "$FULL_UTIL" --host 127.0.0.1 --port "$port" \
                --structured-outputs-config '\''{"backend": "xgrammar", "disable_any_whitespace": true}'\'' \
                > "$OUT/vllm-n${n}tp${tp}-ep$i-$(date +%s).log" 2>&1 &
            PIDS+=($!)
            ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}http://127.0.0.1:$port/v1/chat/completions"
        done
        for i in $(seq 0 $((n-1))); do
            local port=$((BASE_PORT+i)) ready=0
            for t in $(seq 1 180); do
                if curl -sf "http://127.0.0.1:$port/health" >/dev/null \
                   && curl -sf "http://127.0.0.1:$port/v1/models" | grep -q "$MODEL"; then
                    ready=1; break
                fi
                kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "ep$i died on startup"; return 1; }
                sleep 5
            done
            [ "$ready" -eq 1 ] || { echo "ep$i readiness timeout (n=$n tp=$tp)"; return 1; }
        done
        # fleet-engaged gate: exactly N*TP GPUs hold weights+KV
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>20000" | wc -l)
        [ "$used" -eq $((n*tp)) ] || { echo "FATAL: n=$n tp=$tp requested but $used GPUs loaded"; return 1; }
        echo "fleet ready: ${n}x TP$tp -> $ENDPOINTS"
    }
    start_fleet() {  # one retry (transient boot flakes)
        start_fleet_once "$1" "$2" && return 0
        echo "fleet boot failed; retrying once"
        stop_fleet
        start_fleet_once "$1" "$2" || { echo "fleet failed twice (n=$1 tp=$2)"; exit 1; }
    }
    # TERM each vllm subtree bottom-up (pkill -f "vllm serve" matches the bash -c
    # step itself and SIGKILLs the whole job; see tp_scaling_clariden.sh).
    kill_tree() {
        local p=$1 c
        for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done
        kill -TERM "$p" 2>/dev/null || true
    }
    stop_fleet() {
        local p
        for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill_tree "$p"; done
        wait 2>/dev/null || true
        for t in $(seq 1 24); do
            busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>5000" | wc -l)
            [ "$busy" -eq 0 ] && break
            sleep 5
        done
        PIDS=()
    }
    trap stop_fleet EXIT

    snap() {  # snap <tag>: /metrics counter snapshot, one file per endpoint
        local i
        for i in $(seq 0 $((NEP-1))); do
            curl -s "http://127.0.0.1:$((BASE_PORT+i))/metrics" > "$OUT/metrics_${1}_ep$i.txt" || true
        done
    }
    SAMPLER_PID=""; CLIENT_SAMPLER_PID=""
    start_sampler() {  # $1 = tag; per-endpoint scheduler gauges every 3s
        local tag=$1 i
        for i in $(seq 0 $((NEP-1))); do
            echo "epoch,running,waiting,kv_usage" > "$OUT/samples_${tag}_ep$i.csv"
        done
        ( while :; do
            ts=$(date +%s)
            for i in $(seq 0 $((NEP-1))); do
                curl -s "http://127.0.0.1:$((BASE_PORT+i))/metrics" | awk -v ts="$ts" '\''
                    /^vllm:num_requests_running/ {r=$NF}
                    /^vllm:num_requests_waiting/ {w=$NF}
                    /^vllm:(gpu|kv)_cache_usage_perc/ {k=$NF}
                    END {print ts "," r "," w "," k}'\'' >> "$OUT/samples_${tag}_ep$i.csv"
            done
            sleep 3
          done ) &
        SAMPLER_PID=$!
    }
    start_client_sampler() {  # $1 = driver pid, $2 = tag; cumulative CPU ticks
        local f="$OUT/samples_$2_client.csv"
        echo "epoch,utime_ticks,stime_ticks,nthreads" > "$f"
        ( while kill -0 "$1" 2>/dev/null; do
            awk -v ts="$(date +%s)" "{print ts \",\" \$14 \",\" \$15 \",\" \$20}" \
                "/proc/$1/stat" >> "$f" 2>/dev/null
            sleep 3
          done ) &
        CLIENT_SAMPLER_PID=$!
    }
    stop_samplers() {
        for p in "$SAMPLER_PID" "$CLIENT_SAMPLER_PID"; do
            [ -n "$p" ] && kill "$p" 2>/dev/null || true
        done
        SAMPLER_PID=""; CLIENT_SAMPLER_PID=""
    }

    rate_of() { sed -n "s/.*\"rows_per_s\": *\([0-9.]*\).*/\1/p" "$1" | head -1; }

    run() {  # run <on|off> <threads> <cap> <rows> <data> <tag>
        local rw=$1 th=$2 cap=$3 rows=$4 data=$5 tag=$6
        echo "------ RUN $tag (rewrite=$rw threads=$th cap=$cap rows=$rows fleet=${NEP}xTP${FLEET_TP} R=1) ------"
        snap "before_$tag"
        start_sampler "$tag"
        HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" \
            --data "$data" --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$rows" \
            --rewrite "$rw" --threads "$th" --inflight "$cap" --rows-per-request 1 \
            --timeout-ms "$TIMEOUT_MS" \
            --result-out "$OUT/result_$tag.json" > "$OUT/run_$tag.log" 2>&1 &
        local dpid=$!
        start_client_sampler "$dpid" "$tag"
        wait "$dpid"; local rc=$?
        stop_samplers
        snap "after_$tag"
        tail -n 12 "$OUT/run_$tag.log"
        [ "$rc" -eq 0 ] || { echo "FATAL: driver failed for $tag (see run_$tag.log)"; exit 1; }
    }

    balance() {  # balance <tag>: per-endpoint request share (quick read-out)
        local tag=$1 i tot=0 d
        local -a del=()
        for i in $(seq 0 $((NEP-1))); do
            d=$(awk "/^vllm:request_success_total/ {b+=\$NF} END{printf \"%d\", b+0}" \
                    "$OUT/metrics_before_${tag}_ep$i.txt" 2>/dev/null)
            a=$(awk "/^vllm:request_success_total/ {b+=\$NF} END{printf \"%d\", b+0}" \
                    "$OUT/metrics_after_${tag}_ep$i.txt" 2>/dev/null)
            del[$i]=$(( ${a:-0} - ${d:-0} )); tot=$((tot + del[i]))
        done
        [ "$tot" -gt 0 ] || { echo "balance $tag: no request deltas (?)"; return 0; }
        for i in $(seq 0 $((NEP-1))); do
            awk -v x="${del[$i]}" -v t="$tot" -v n="$NEP" -v i="$i" -v tag="$tag" "BEGIN{
                s=100*x/t; ideal=100/n; dev=s-ideal;
                printf \"balance %s: ep%d %d reqs (%.1f%%, ideal %.1f%%)\n\", tag, i, x, s, ideal;
                if (dev>10 || dev<-10) printf \"WARN: ep%d share off by %.1f points under round_robin\n\", i, dev }"
        done
    }

    DONE_VERDICTS=" "
    for CFG in $CONFIGS; do
        IFS=: read -r N TP CAP <<< "$CFG"
        TAG="op_n${N}_tp${TP}_c${CAP}"
        echo "==================== CONFIG n=$N tp=$TP cap=$CAP ===================="
        start_fleet "$N" "$TP"
        run on "$OP_THREADS" "$CAP" "$ROWS" "$DATA" "$TAG"
        # rewrite-engaged gate: a stale binary degrades to the scalar path
        # (concurrency 1, ~17-29 rows/s); works in single-arm jobs.
        R=$(rate_of "$OUT/result_$TAG.json")
        awk -v r="$R" -v t="$TAG" "BEGIN{ if (r==\"\") exit 2;
            printf \"rewrite-engaged gate %s: %.1f rows/s (serial scalar ~17-29)\n\", t, r;
            exit (r>=60.0 ? 0 : 1) }" \
            || { echo "FATAL: $TAG too slow -- stale binary / rewrite not engaged?"; exit 1; }
        balance "$TAG"
        KEY="n${N}_tp${TP}"
        if [ "$VERDICTS" -eq 1 ] && [[ "$DONE_VERDICTS" != *" $KEY "* ]]; then
            # untimed F1 dump on the live fleet (greedy -> deterministic).
            # GOLD_DATA rows only: the gold labels are the sf_2000 set.
            HOME="$JOB_HOME" FLOCK_VERDICT_DUMP="$OUT/verdicts_${KEY}.jsonl" \
                "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$GOLD_DATA" \
                       --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$GOLD_ROWS" \
                       --rewrite on --threads "$OP_THREADS" --inflight "$CAP" \
                       --rows-per-request 1 --timeout-ms "$TIMEOUT_MS" \
                       --skip-burn-in 2>&1 | tee "$OUT/verdict_${KEY}.log"
            LINES=$(wc -l < "$OUT/verdicts_${KEY}.jsonl" 2>/dev/null || echo 0)
            echo "verdicts $KEY: $LINES lines (expect $GOLD_ROWS)"
            [ "$LINES" -eq "$GOLD_ROWS" ] || echo "WARN: H4 gate -- verdict row loss ($LINES != $GOLD_ROWS)"
            DONE_VERDICTS="$DONE_VERDICTS$KEY "
        fi
        stop_fleet
    done

    # scalar reference arm (curve jobs): stock path can only address ep0 --
    # the flat no-scale-out line. GOLD_ROWS only (serial, 32000 would take ~30min).
    case " $ARMS " in *" scalar "*)
        echo "==================== SCALAR (1x TP1, gold rows) ===================="
        start_fleet 1 1
        run off "$SCALAR_THREADS" 128 "$GOLD_ROWS" "$GOLD_DATA" "scalar_n1_tp1_c128"
        stop_fleet
        ;;
    esac
    trap - EXIT

    echo "==================== VALIDATION ===================="
    OPJ="$OUT/result_op_n1_tp1_c128.json"; SCJ="$OUT/result_scalar_n1_tp1_c128.json"
    if [ -f "$OPJ" ] && [ -f "$SCJ" ]; then
        # A/B gate on rows/s (arms run different row counts here)
        opr=$(rate_of "$OPJ"); scr=$(rate_of "$SCJ")
        awk -v o="$opr" -v s="$scr" "BEGIN{ if (o==\"\" || s==\"\") exit 2;
            r=o/s; printf \"A/B gate: operator/scalar rows-per-s ratio = %.1f\n\", r;
            exit (r>=3.0 ? 0 : 1) }" \
            || { echo "FATAL: arms too similar -- stale binary / rewrite not engaged?"; exit 1; }
    else
        echo "A/B gate SKIPPED (no scalar arm in this job; rewrite-engaged gate ran per config)"
    fi
    grep -h "rows_per_s\|inflight" "$OUT"/result_*.json 2>/dev/null || true
    echo "expect: operator rows/s ~linear in N at cap=128*N; starvation cell (N=4,cap=128) well below N=4,cap=512."
    echo "DONE. Artefacts in: $OUT"
    echo "Pull home:  rsync -av <clariden>:$OUT analysis/figures/data/dp_scaling/"
    echo "Summarise:  python analysis/summarize_dp.py   (built after results land)"
'

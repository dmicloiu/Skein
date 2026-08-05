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
# CONFIGS is a list of N:TP:CAP fleet configs, one cold fleet each. N:TP with
# N=1 and TP>1 is a TP cell, so this driver covers BOTH scale-out axes and they
# can be measured in one job. REWRITE picks the path the cells exercise: on =
# the async operator, off = the scalar llm_filter on the SAME workload and the
# SAME fleet -- the scalar is a first-class configuration of the sweep, not a
# side arm on its own dataset. Job presets (submit as SEPARATE jobs; boots
# dominate, jobs stay ~1h):
#   curve (default):  CONFIGS="1:1:128 2:1:256 4:1:512 4:1:128"
#                     (last cell = H3 starvation probe: N=4 at the unscaled cap)
#   grid:             CONFIGS="4:1:512 2:2:512 1:4:512"
#                     (fixed 4-GPU budget, total cap 512; own 4xTP1 anchor so
#                     the replicas-vs-shards comparison is in-job)
#   unified:          CONFIGS="1:1:128 1:2:256 1:4:512 2:1:256 4:1:512"
#                     (DP and TP cells in ONE job on ONE node at the same rows:
#                     replaces the cross-study DP-vs-TP table with an in-job one)
#   caps:             CONFIGS="4:1:256 4:1:1024 4:1:2048" VERDICTS=0
#                     (H3 cap sweep at N=4 -- 128/512 already measured -- plus
#                     the client-CPU ceiling probe at 1024/2048 in flight)
#   kvstress:         RESPONSE_FORMAT=free_form OUT_MULT=512 IGNORE_EOS=1
#                     ROWS=4096 VERDICTS=0 MIN_OP_RATE=2
#                     CONFIGS="4:1:512 1:4:512"
#                     (decode-heavy KV stress: long outputs push resident KV
#                     past the replica pool; quality out of scope. OUT_MULT
#                     512 keeps mean e2e safely under the async client 60s
#                     per-request timeout)
#   kvdeep:           kvstress knobs but OUT_MULT=1024 REQUEST_TIMEOUT_MS=300000
#                     (deeper decode: doubles resident KV per request to probe
#                     whether the erased replica advantage inverts. Mean e2e
#                     scales past 60s at this depth, hence the raised client
#                     timeout)
#   scalar:           REWRITE=off ROWS=2000 CONFIGS="4:1:128"
#                     (the no-scale-out reference: the scalar path pointed at a
#                     4-endpoint fleet, same dataset as the operator cells. Its
#                     rate is steady-state after ~100 rows, so a 2000-row prefix
#                     measures it in ~2 min where 32k rows would cost ~31 min;
#                     the per-endpoint shares must come out 100/0/0/0 (gated).)
#
# REPS: REP is a free-form label appended to every artefact name. Reps are
# separate jobs AND presets share cells (4:1:512 is in curve, grid and unified),
# so the label must carry BOTH the preset and the rep or those cells collide on a
# flat import. Job dirs stay job-id-scoped either way, so this only bites at
# import time -- set it and the artefacts are self-identifying:
#   for i in 1 2 3; do REP="unified_rep$i" CONFIGS=... \
#       sbatch --time=00:35:00 analysis/slurm/dp_scaling_clariden.sh; done
#   => result_op_n4_tp1_c512_unified_rep1.json, verdicts_n1_tp4_unified_rep1.jsonl
#
# ROWS/DATA: timed runs default to 32000 rows -- at N=4 (~900 rows/s) 2000 rows
# would finish in ~2-4s, too short for timing or the 3s gauge sampler. DATA
# therefore defaults to sf_300000/Reviews.csv (the same file the TP morsel mode
# used, so the timed rows are a prefix of that set); the row check is fail-loud.
# Verdict (F1) passes stay on GOLD_DATA (sf_2000, 2000
# gold rows) -- that is where the gold labels live.
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
    # Serve from the local HF cache only: vllm queries the Hub file-list API on
    # every boot, and a rep batch trips HF rate limiting (429 = fatal boot, even
    # with weights cached). Boot-path only; the timed region is unaffected.
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sembench

    FLOCK="$HOME/projects/flock"; SEMBENCH="$HOME/projects/sembench"
    BIN="$FLOCK/build/flock_sem_filter_vllm_integration"
    # Tuple encoding is part of the artefact path: the driver renders json by
    # default now, and the pre-standardisation results were XML, so the two must
    # never share a directory. TUPLE_FORMAT=XML reruns the old encoding.
    TUPLE_FORMAT="${TUPLE_FORMAT:-json}"
    OUT="$FLOCK/analysis/results/dp_scaling_${TUPLE_FORMAT}/${SLURM_JOB_ID}"; mkdir -p "$OUT"

    # Job-private HOME: flock persists its model catalog at
    # $HOME/.duckdb/flock_storage and parallel rep jobs race the shared file.
    JOB_HOME="/tmp/flock_home_${SLURM_JOB_ID:-local}"
    mkdir -p "$JOB_HOME/.duckdb"

    # Full prompt on both arms: no slim/variant knob may leak in.
    unset FLOCK_SEM_PROMPT FLOCK_SEM_VARIANTS

    MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    BASE_PORT=8000; FULL_UTIL=0.90; TIMEOUT_MS=120000
    TEXT_COL="${TEXT_COL:-reviewText}"
    PROMPT="${PROMPT:-The following movie review is clearly positive.}"
    DATA="${DATA:-$SEMBENCH/files/movie/data/sf_300000/Reviews.csv}"
    ROWS="${ROWS:-32000}"
    GOLD_DATA="${GOLD_DATA:-$SEMBENCH/files/movie/data/sf_2000/Reviews.csv}"
    GOLD_ROWS="${GOLD_ROWS:-2000}"
    CONFIGS="${CONFIGS:-1:1:128 2:1:256 4:1:512 4:1:128}"
    # Which path the CONFIGS cells exercise: on = operator, off = scalar
    # llm_filter on the same workload/fleet (the "scalar" preset above).
    REWRITE="${REWRITE:-on}"
    OP_THREADS="${OP_THREADS:-1}"         # operator is thread-independent
    MIN_OP_RATE="${MIN_OP_RATE:-60}"      # rewrite-engaged gate floor (rows/s)
    # Response shape: RESPONSE_FORMAT=free_form + OUT_MULT (max_output_tokens
    # per row) turn the filter into a decode-heavy workload generator for the
    # KV/decode stress cells; quality is out of scope there (VERDICTS=0).
    RESPONSE_FORMAT="${RESPONSE_FORMAT:-}"
    OUT_MULT="${OUT_MULT:-}"
    # IGNORE_EOS=1 forces generation to exactly max_output_tokens (vLLM
    # ignore_eos) so resident KV per request is deterministic.
    [ "${IGNORE_EOS:-0}" = "1" ] && export FLOCK_SEM_IGNORE_EOS=1
    # REQUEST_TIMEOUT_MS raises the async-client per-request timeout for
    # cells whose mean service time approaches the built-in 60s (long decode
    # at high concurrency).
    [ -n "${REQUEST_TIMEOUT_MS:-}" ] && export FLOCK_SEM_REQUEST_TIMEOUT_MS="$REQUEST_TIMEOUT_MS"
    SHAPE_ARGS=()
    [ -n "$RESPONSE_FORMAT" ] && SHAPE_ARGS+=(--response-format "$RESPONSE_FORMAT")
    [ -n "$OUT_MULT" ] && SHAPE_ARGS+=(--max-out-mult "$OUT_MULT")
    SCALAR_THREADS="${SCALAR_THREADS:-8}" # 1 in-memory morsel -> concurrency 1 anyway
    VERDICTS="${VERDICTS:-1}"             # untimed F1 dump, first config per (N,TP)
    # Free-form run label, appended verbatim to every artefact name. Reps are
    # separate JOBS, and presets SHARE cells (4:1:512 appears in curve, grid and
    # unified), so without a label those cells write identical filenames and a
    # flat import into analysis/figures/data/dp_scaling/ silently overwrites all
    # but the last. Pass the preset AND the rep -- REP="unified_rep2" yields
    # result_op_n4_tp1_c512_unified_rep2.json. Unset => no suffix (single shots).
    REP="${REP:-}"
    RSUF="${REP:+_$REP}"

    echo "==================== ENV ===================="
    echo "OUT=$OUT rows=$ROWS gold_rows=$GOLD_ROWS configs=[$CONFIGS] rewrite=$REWRITE rep=[${REP:-none}]"
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
                > "$OUT/vllm-n${n}tp${tp}-ep$i${RSUF}-$(date +%s).log" 2>&1 &
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
        # Sidecar metadata: the result JSON records neither the dataset nor the
        # fleet shape, and a job may now mix datasets (scalar rate cell vs gold
        # agreement cell) and later models. Without this the CSV cannot say
        # which workload a row measured.
        printf "data=%s\ndataset=%s\nrows=%s\nmodel=%s\nn_ep=%s\ntp=%s\ncap=%s\narm=%s\n" \
            "$data" "$(basename "$(dirname "$data")")" "$rows" \
            "$MODEL" "$NEP" "$FLEET_TP" "$cap" "$rw" > "$OUT/meta_$tag.txt"
        snap "before_$tag"
        start_sampler "$tag"
        HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" \
            --data "$data" --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$rows" \
            --rewrite "$rw" --threads "$th" --inflight "$cap" --rows-per-request 1 \
            --timeout-ms "$TIMEOUT_MS" --tuple-format "$TUPLE_FORMAT" ${SHAPE_ARGS[@]+"${SHAPE_ARGS[@]}"} \
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
        # Expected share depends on the path: round_robin spreads 1/N, the scalar
        # has no router and must sit entirely on ep0. Only the operator case can
        # WARN -- a scalar cell deviating is caught hard by the single-endpoint
        # gate below, so warning here too would just print 4 scary lines per run.
        for i in $(seq 0 $((NEP-1))); do
            awk -v x="${del[$i]}" -v t="$tot" -v n="$NEP" -v i="$i" -v tag="$tag" -v rw="$REWRITE" "BEGIN{
                s=100*x/t;
                ideal=(rw==\"on\") ? 100/n : ((i==0) ? 100 : 0);
                dev=s-ideal;
                printf \"balance %s: ep%d %d reqs (%.1f%%, ideal %.1f%%)\n\", tag, i, x, s, ideal;
                if (rw==\"on\" && (dev>10 || dev<-10)) printf \"WARN: ep%d share off by %.1f points under round_robin\n\", i, dev }"
        done
    }

    DONE_VERDICTS=" "
    for CFG in $CONFIGS; do
        IFS=: read -r N TP CAP <<< "$CFG"
        if [ "$REWRITE" = "on" ]; then ARM=op; TH="$OP_THREADS"; else ARM=scalar; TH="$SCALAR_THREADS"; fi
        TAG="${ARM}_n${N}_tp${TP}_c${CAP}${RSUF}"
        echo "==================== CONFIG n=$N tp=$TP cap=$CAP rewrite=$REWRITE ===================="
        start_fleet "$N" "$TP"
        run "$REWRITE" "$TH" "$CAP" "$ROWS" "$DATA" "$TAG"
        R=$(rate_of "$OUT/result_$TAG.json")
        if [ "$REWRITE" = "on" ]; then
            # rewrite-engaged gate: a stale binary degrades to the scalar path
            # (concurrency 1). MIN_OP_RATE scales with the model size.
            awk -v r="$R" -v t="$TAG" -v m="$MIN_OP_RATE" "BEGIN{ if (r==\"\") exit 2;
                printf \"rewrite-engaged gate %s: %.1f rows/s (min %s)\n\", t, r, m;
                exit (r>=m+0.0 ? 0 : 1) }" \
                || { echo "FATAL: $TAG too slow -- stale binary / rewrite not engaged?"; exit 1; }
        else
            echo "scalar cell $TAG: $R rows/s (rewrite off; no throughput gate)"
        fi
        balance "$TAG"
        if [ "$REWRITE" = "off" ] && [ "$N" -gt 1 ]; then
            # The scalar path takes its base_url from CREATE MODEL (the FIRST
            # --endpoints entry) and has no router, so on an N-endpoint fleet it
            # must reach ep0 only. Measures the "stock flock cannot address a
            # fleet" claim instead of asserting it from the code.
            OTHER=0
            for i in $(seq 1 $((N-1))); do
                a=$(awk "/^vllm:request_success_total/ {b+=\$NF} END{printf \"%d\", b+0}" \
                        "$OUT/metrics_after_${TAG}_ep$i.txt" 2>/dev/null)
                b=$(awk "/^vllm:request_success_total/ {b+=\$NF} END{printf \"%d\", b+0}" \
                        "$OUT/metrics_before_${TAG}_ep$i.txt" 2>/dev/null)
                OTHER=$((OTHER + ${a:-0} - ${b:-0}))
            done
            echo "single-endpoint gate $TAG: eps 1..$((N-1)) got $OTHER requests (expect 0)"
            [ "$OTHER" -eq 0 ] || { echo "FATAL: scalar reached $OTHER requests beyond ep0 -- NOT single-endpoint"; exit 1; }
        fi
        KEY="n${N}_tp${TP}"
        if [ "$REWRITE" = "on" ] && [ "$VERDICTS" -eq 1 ] && [[ "$DONE_VERDICTS" != *" $KEY "* ]]; then
            # untimed F1 dump on the live fleet (greedy -> deterministic).
            # GOLD_DATA rows only: the gold labels are the sf_2000 set.
            HOME="$JOB_HOME" FLOCK_VERDICT_DUMP="$OUT/verdicts_${KEY}${RSUF}.jsonl" \
                "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" --data "$GOLD_DATA" \
                       --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$GOLD_ROWS" \
                       --rewrite on --threads "$OP_THREADS" --inflight "$CAP" \
                       --rows-per-request 1 --timeout-ms "$TIMEOUT_MS" \
                       --tuple-format "$TUPLE_FORMAT" \
                       --skip-burn-in 2>&1 | tee "$OUT/verdict_${KEY}${RSUF}.log"
            LINES=$(wc -l < "$OUT/verdicts_${KEY}${RSUF}.jsonl" 2>/dev/null || echo 0)
            echo "verdicts $KEY: $LINES lines (expect $GOLD_ROWS)"
            [ "$LINES" -eq "$GOLD_ROWS" ] || echo "WARN: H4 gate -- verdict row loss ($LINES != $GOLD_ROWS)"
            DONE_VERDICTS="$DONE_VERDICTS$KEY "
        fi
        stop_fleet
    done

    trap - EXIT

    # Arm validity is per-cell: rewrite=on cells carry the rows/s >= 60
    # rewrite-engaged gate, rewrite=off cells the single-endpoint gate. No
    # in-job A/B pair needed (the arms live in separate jobs by design).
    echo "==================== VALIDATION ===================="
    grep -h "rows_per_s\|inflight" "$OUT"/result_*.json 2>/dev/null || true
    if [ "$REWRITE" = "on" ]; then
        echo "expect: operator rows/s ~linear in N at cap=128*N; starvation cell (N=4,cap=128) below N=4,cap=512."
    else
        echo "expect: scalar ~flat at 15-30 rows/s regardless of fleet size; all requests on ep0."
    fi
    echo "DONE. Artefacts in: $OUT"
    echo "Pull home:  rsync -av <clariden>:$OUT analysis/figures/data/dp_scaling_${TUPLE_FORMAT}/"
    echo "Summarise:  python analysis/summarize_dp.py   (built after results land)"
'

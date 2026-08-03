#!/bin/bash
#SBATCH --account=infra02
#SBATCH --job-name=flock-dp-multinode
#SBATCH --time=00:25:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=logs/slurm-%x-%j.out
# -----------------------------------------------------------------------------
# Multi-node DP scaling: N endpoints spread over TWO GH200 nodes (first 4 on
# node 0, the rest on node 1), driver on node 0 dispatching over the real
# network. Extends the single-node study (dp_scaling_clariden.sh) past N=4 and
# replaces its loopback caveat with measured cross-node behaviour.
#
# Operator cells only (no scalar arm, no verdict pass): quality is
# fleet-invariant per the single-node study; this script measures throughput,
# dispatch and the C* law at N=8.
#
# Presets (one cold fleet per config, remote fleet cycled via files on $OUT):
#   harness check:  CONFIGS="4:1:512" ROWS=32000 MIN_OP_RATE=500 --time=00:15:00
#                   (single-node cell on the new harness -- must reproduce the
#                   known ~560 rows/s before any N=8 number is trusted. The
#                   reference band is tight, 556-568 rows/s over 10 committed
#                   single-node reps, so the gate is set at 500: the default 60
#                   would pass a harness that had HALVED throughput and so does
#                   not implement this check at all. N<=4 => REMOTE_N=0, i.e.
#                   this cell does NOT exercise the two-node protocol.)
#   main:           CONFIGS="8:1:1024 8:1:512" ROWS=65536 --time=00:25:00
#                   (the N=8 point at 128/endpoint + the 64/endpoint cell for
#                   the collapse curve; 65536 rows keep the fastest cell ~60s.
#                   Leave MIN_OP_RATE at its loose default here -- the N=8 rate
#                   is the measurement, so gating it tightly would discard the
#                   result the job exists to produce. Submit rep1 ALONE first:
#                   it is the first real test of the two-node plumbing, and
#                   firing all three reps at once risks burning three 2-node
#                   allocations on one protocol bug.)
#
# Interpretation caveat: local endpoints are addressed as 127.0.0.1, remote ones
# as $MN_NODE1, and the driver shares node0 with 4 servers. N=8 therefore
# measures "4 local + 4 remote at unequal RTT under round_robin", not "8 remote
# endpoints" -- round_robin hands equal request counts to unequal-latency
# endpoints, so the slower half gates. The check cell is all-loopback and
# unaffected.
#
# Node-0/node-1 protocol (shared filesystem):
#   remote step:  boot -> "$OUT/gen<k>.remote_ready" (holds its loaded-GPU
#                 count, published via mv so node0 cannot read it half-written)
#                 -> wait for gen<k>.stop -> teardown+drain -> gen<k>.remote_done
#   main step:    boot local -> wait remote_ready -> cross-node health checks
#                 -> timed run + gates -> touch stop -> wait remote_done
#   orchestrator: touches gen<k>.stop unconditionally once the main step returns,
#                 so a main-step FATAL can never leave the remote step parked in
#                 its `until stop` loop until walltime.
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$HOME/projects/flock"
mkdir -p logs
EDF="$HOME/projects/sembench/ngc-pytorch-vllm.toml"

CONFIGS="${CONFIGS:-8:1:1024}"
# Reps are separate JOBS imported flat into figures/data/dp_scaling, so every
# artefact name must carry the rep label or mn_rep1/2/3 overwrite each other.
REP="${REP:-}"; RSUF="${REP:+_$REP}"

# Only cells with N>4 spill onto node1, so a single-node preset (the 4:1:512
# harness check) must be submittable with --nodes=1. Demanding a second, idle
# node would make the cheap validation cell queue behind 2-node backfill for no
# measurement benefit.
NEED_NODES=1
for CFG in $CONFIGS; do
    IFS=: read -r CN _ _ <<< "$CFG"
    if [ "$CN" -gt 4 ]; then NEED_NODES=2; fi
done
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
[ "${#NODES[@]}" -ge "$NEED_NODES" ] \
    || { echo "FATAL: CONFIGS [$CONFIGS] needs $NEED_NODES node(s), got ${NODES[*]}"; exit 1; }
export MN_NODE0="${NODES[0]}"
export MN_NODE1="${NODES[1]:-${NODES[0]}}"   # unused when NEED_NODES=1
echo "nodes: need $NEED_NODES, have ${#NODES[@]} (${NODES[*]})"
export MN_OUT="$HOME/projects/flock/analysis/results/dp_scaling/${SLURM_JOB_ID}"
mkdir -p "$MN_OUT"

GEN=0
for CFG in $CONFIGS; do
    GEN=$((GEN + 1))
    IFS=: read -r N TP CAP <<< "$CFG"
    REMOTE_N=$(( N > 4 ? N - 4 : 0 ))
    export MN_CFG="$CFG" MN_GEN="$GEN" MN_REMOTE_N="$REMOTE_N"

    if [ "$REMOTE_N" -gt 0 ]; then
        srun -ul --nodes=1 --ntasks=1 -w "$MN_NODE1" --gpus-per-node=4 \
             --environment="$EDF" bash -c '
            set -uo pipefail
            source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sembench
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            export NO_PROXY="localhost,127.0.0.1,$MN_NODE0,$MN_NODE1"; export no_proxy="$NO_PROXY"
            OUT="$MN_OUT"; GEN="$MN_GEN"; TP="'"$TP"'"
            MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
            REP="${REP:-}"; RSUF="${REP:+_$REP}"
            READY="$OUT/gen${GEN}.remote_ready"
            PIDS=()
            kill_tree() { local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done; kill -TERM "$p" 2>/dev/null || true; }
            drain() {
                local p
                for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill_tree "$p"; done
                wait 2>/dev/null || true
                for t in $(seq 1 24); do
                    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>5000" | wc -l)
                    [ "$busy" -eq 0 ] && break
                    sleep 5
                done
            }
            # Publish readiness atomically. node0 polls for the file to EXIST and
            # then reads it as an integer, so a plain redirect (which truncates
            # before the value lands) lets node0 read "" and fail its numeric
            # gate with a misleading "node1 loaded  != 4" on a healthy fleet.
            publish() { printf "%s" "$1" > "$READY.tmp" && mv -f "$READY.tmp" "$READY"; }
            fail() { publish remote_fail; drain; touch "$OUT/gen${GEN}.remote_done"; exit 1; }

            for i in $(seq 0 $((MN_REMOTE_N-1))); do
                port=$((8000+i))
                devs=$(seq -s, $((i*TP)) $((i*TP+TP-1)))
                CUDA_VISIBLE_DEVICES=$devs vllm serve "$MODEL" \
                    --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
                    --tensor-parallel-size "$TP" \
                    --gpu-memory-utilization 0.90 --host 0.0.0.0 --port "$port" \
                    --structured-outputs-config '\''{"backend": "xgrammar", "disable_any_whitespace": true}'\'' \
                    > "$OUT/vllm-remote-gen${GEN}-ep${i}${RSUF}.log" 2>&1 &
                PIDS+=($!)
            done
            for i in $(seq 0 $((MN_REMOTE_N-1))); do
                ready=0
                for t in $(seq 1 120); do
                    if curl -sf "http://127.0.0.1:$((8000+i))/health" >/dev/null \
                       && curl -sf "http://127.0.0.1:$((8000+i))/v1/models" | grep -q "$MODEL"; then
                        ready=1; break
                    fi
                    kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "remote ep$i died on startup"; break; }
                    sleep 5
                done
                [ "$ready" -eq 1 ] || { echo "remote ep$i not ready"; fail; }
            done
            publish "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>20000" | wc -l)"
            while [ ! -f "$OUT/gen${GEN}.stop" ]; do sleep 3; done
            drain
            touch "$OUT/gen${GEN}.remote_done"
        ' > "$MN_OUT/remote_gen${GEN}${RSUF}.log" 2>&1 &
        REMOTE_STEP=$!
    fi

    MAIN_RC=0
    srun -ul --nodes=1 --ntasks=1 -w "$MN_NODE0" --gpus-per-node=4 \
         --environment="$EDF" bash -c '
        set -uo pipefail
        source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sembench
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export NO_PROXY="localhost,127.0.0.1,$MN_NODE0,$MN_NODE1"; export no_proxy="$NO_PROXY"
        unset FLOCK_SEM_PROMPT FLOCK_SEM_VARIANTS

        FLOCK="$HOME/projects/flock"; SEMBENCH="$HOME/projects/sembench"
        BIN="$FLOCK/build/flock_sem_filter_vllm_integration"
        OUT="$MN_OUT"; GEN="$MN_GEN"
        IFS=: read -r N TP CAP <<< "$MN_CFG"
        LOCAL_N=$(( N > 4 ? 4 : N ))
        JOB_HOME="/tmp/flock_home_${SLURM_JOB_ID:-local}"; mkdir -p "$JOB_HOME/.duckdb"

        MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
        TEXT_COL="${TEXT_COL:-reviewText}"
        PROMPT="${PROMPT:-The following movie review is clearly positive.}"
        DATA="${DATA:-$SEMBENCH/files/movie/data/sf_300000/Reviews.csv}"
        ROWS="${ROWS:-65536}"
        MIN_OP_RATE="${MIN_OP_RATE:-60}"
        REP="${REP:-}"; RSUF="${REP:+_$REP}"
        TAG="op_n${N}_tp${TP}_c${CAP}${RSUF}"
        TIMEOUT_MS=120000

        [ -x "$BIN" ] || { echo "driver missing: $BIN"; exit 1; }
        [ -f "$DATA" ] || { echo "dataset missing: $DATA"; exit 1; }
        AVAIL=$(($(wc -l < "$DATA") - 1))
        [ "$AVAIL" -ge "$ROWS" ] || { echo "FATAL: need $ROWS rows, have $AVAIL"; exit 1; }
        # The 4-GPUs-per-node split in the orchestrator is an assumption, not a
        # measurement: a config that overruns a node (8:2 wants 16 GPUs) would
        # silently mis-slice CUDA_VISIBLE_DEVICES rather than fail. Same intent
        # as the MAXG check in dp_scaling_clariden.sh.
        NGPU=$(nvidia-smi --list-gpus | wc -l)
        for want in $((LOCAL_N*TP)) $((MN_REMOTE_N*TP)); do
            [ "$want" -le "$NGPU" ] \
                || { echo "FATAL: cfg $MN_CFG needs $want GPUs on one node, node has $NGPU"; exit 1; }
        done

        # endpoint tables: local endpoints first, remote after
        EP_HOST=(); EP_PORT=(); PIDS=()
        kill_tree() { local p=$1 c; for c in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$c"; done; kill -TERM "$p" 2>/dev/null || true; }
        stop_local() {  # TERM the local vllm subtrees, then wait for GPU memory to drain
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
        # Without this every FATAL below leaves 4 loaded GPUs behind; the remote
        # step is released separately (the orchestrator touches gen<k>.stop).
        trap stop_local EXIT
        for i in $(seq 0 $((LOCAL_N-1))); do
            port=$((8000+i))
            devs=$(seq -s, $((i*TP)) $((i*TP+TP-1)))
            CUDA_VISIBLE_DEVICES=$devs vllm serve "$MODEL" \
                --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching \
                --tensor-parallel-size "$TP" \
                --gpu-memory-utilization 0.90 --host 0.0.0.0 --port "$port" \
                --structured-outputs-config '\''{"backend": "xgrammar", "disable_any_whitespace": true}'\'' \
                > "$OUT/vllm-local-gen${GEN}-ep${i}${RSUF}.log" 2>&1 &
            PIDS+=($!)
            EP_HOST+=("127.0.0.1"); EP_PORT+=($port)
        done
        for i in $(seq 0 $((MN_REMOTE_N-1))); do
            EP_HOST+=("$MN_NODE1"); EP_PORT+=($((8000+i)))
        done

        for i in $(seq 0 $((LOCAL_N-1))); do
            ready=0
            for t in $(seq 1 120); do
                if curl -sf "http://127.0.0.1:${EP_PORT[$i]}/health" >/dev/null \
                   && curl -sf "http://127.0.0.1:${EP_PORT[$i]}/v1/models" | grep -q "$MODEL"; then
                    ready=1; break
                fi
                kill -0 "${PIDS[$i]}" 2>/dev/null || { echo "local ep$i died"; exit 1; }
                sleep 5
            done
            [ "$ready" -eq 1 ] || { echo "local ep$i timeout"; exit 1; }
        done
        LOCAL_LOADED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk "\$1>20000" | wc -l)
        [ "$LOCAL_LOADED" -eq $((LOCAL_N*TP)) ] || { echo "FATAL: node0 loaded $LOCAL_LOADED != $((LOCAL_N*TP))"; exit 1; }

        REMOTE_LOADED=0
        if [ "$MN_REMOTE_N" -gt 0 ]; then
            for t in $(seq 1 140); do
                [ -f "$OUT/gen${GEN}.remote_ready" ] && break
                sleep 5
            done
            [ -f "$OUT/gen${GEN}.remote_ready" ] || { echo "FATAL: remote fleet never ready"; exit 1; }
            REMOTE_LOADED=$(cat "$OUT/gen${GEN}.remote_ready")
            [ "$REMOTE_LOADED" = "remote_fail" ] \
                && { echo "FATAL: remote boot failed (see remote_gen${GEN}${RSUF}.log)"; exit 1; }
            case "$REMOTE_LOADED" in
                ""|*[!0-9]*) echo "FATAL: remote_ready holds non-numeric [$REMOTE_LOADED]"; exit 1;;
            esac
            [ "$REMOTE_LOADED" -eq $((MN_REMOTE_N*TP)) ] || { echo "FATAL: node1 loaded $REMOTE_LOADED != $((MN_REMOTE_N*TP))"; exit 1; }
            # cross-node reachability is part of the experiment: gate on it
            for i in $(seq $LOCAL_N $((N-1))); do
                curl -sf --max-time 10 "http://${EP_HOST[$i]}:${EP_PORT[$i]}/health" >/dev/null \
                    || { echo "FATAL: remote ep$i unreachable from node0"; exit 1; }
            done
        fi
        ENDPOINTS=""
        for i in $(seq 0 $((N-1))); do
            ENDPOINTS="${ENDPOINTS:+$ENDPOINTS,}http://${EP_HOST[$i]}:${EP_PORT[$i]}/v1/chat/completions"
        done
        echo "fleet ready: ${N}x TP${TP} across $MN_NODE0+$MN_NODE1 -> $ENDPOINTS"

        snap() {
            for i in $(seq 0 $((N-1))); do
                curl -s --max-time 10 "http://${EP_HOST[$i]}:${EP_PORT[$i]}/metrics" \
                    > "$OUT/metrics_${1}_ep$i.txt" || true
            done
        }
        SAMPLER_PID=""; CLIENT_SAMPLER_PID=""
        start_sampler() {
            local tag=$1 i
            for i in $(seq 0 $((N-1))); do
                echo "epoch,running,waiting,kv_usage" > "$OUT/samples_${tag}_ep$i.csv"
            done
            ( while :; do
                ts=$(date +%s)
                for i in $(seq 0 $((N-1))); do
                    curl -s --max-time 5 "http://${EP_HOST[$i]}:${EP_PORT[$i]}/metrics" | awk -v ts="$ts" '\''
                        /^vllm:num_requests_running/ {r=$NF}
                        /^vllm:num_requests_waiting/ {w=$NF}
                        /^vllm:(gpu|kv)_cache_usage_perc/ {k=$NF}
                        END {print ts "," r "," w "," k}'\'' >> "$OUT/samples_${tag}_ep$i.csv"
                done
                sleep 3
              done ) &
            SAMPLER_PID=$!
        }
        start_client_sampler() {
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

        # nodes= must be measured, not assumed: the harness-check cell (N<=4) has
        # REMOTE_N=0 and runs entirely on node0, so hardcoding 2 would label a
        # single-node measurement as cross-node.
        NODES_USED=1; [ "$MN_REMOTE_N" -gt 0 ] && NODES_USED=2
        printf "data=%s\nrows=%s\nmodel=%s\nn_ep=%s\ntp=%s\ncap=%s\narm=on\nnodes=%s\n" \
            "$DATA" "$ROWS" "$MODEL" "$N" "$TP" "$CAP" "$NODES_USED" > "$OUT/meta_$TAG.txt"
        echo "------ RUN $TAG (rows=$ROWS fleet=${N}xTP${TP} over $NODES_USED node(s)) ------"
        snap "before_$TAG"
        start_sampler "$TAG"
        HOME="$JOB_HOME" "$BIN" --endpoints "$ENDPOINTS" --model "$MODEL" \
            --data "$DATA" --text-col "$TEXT_COL" --prompt "$PROMPT" --rows "$ROWS" \
            --rewrite on --threads 1 --inflight "$CAP" --rows-per-request 1 \
            --timeout-ms "$TIMEOUT_MS" \
            --result-out "$OUT/result_$TAG.json" > "$OUT/run_$TAG.log" 2>&1 &
        dpid=$!
        start_client_sampler "$dpid" "$TAG"
        wait "$dpid"; rc=$?
        stop_samplers
        snap "after_$TAG"
        tail -n 12 "$OUT/run_$TAG.log"
        [ "$rc" -eq 0 ] || { echo "FATAL: driver failed for $TAG"; touch "$OUT/gen${GEN}.stop"; exit 1; }

        R=$(sed -n "s/.*\"rows_per_s\": *\([0-9.]*\).*/\1/p" "$OUT/result_$TAG.json" | head -1)
        awk -v r="$R" -v t="$TAG" -v m="$MIN_OP_RATE" "BEGIN{ if (r==\"\") exit 2;
            printf \"rewrite-engaged gate %s: %.1f rows/s (min %s)\n\", t, r, m;
            exit (r>=m+0.0 ? 0 : 1) }" \
            || { echo "FATAL: $TAG too slow"; touch "$OUT/gen${GEN}.stop"; exit 1; }

        # dispatch evenness across BOTH nodes (quick read-out)
        tot=0; declare -a del
        for i in $(seq 0 $((N-1))); do
            b=$(awk "/^vllm:request_success_total/ {s+=\$NF} END{printf \"%d\", s+0}" "$OUT/metrics_before_${TAG}_ep$i.txt" 2>/dev/null)
            a=$(awk "/^vllm:request_success_total/ {s+=\$NF} END{printf \"%d\", s+0}" "$OUT/metrics_after_${TAG}_ep$i.txt" 2>/dev/null)
            del[$i]=$(( ${a:-0} - ${b:-0} )); tot=$((tot + del[i]))
        done
        if [ "$tot" -gt 0 ]; then
            for i in $(seq 0 $((N-1))); do
                awk -v x="${del[$i]}" -v t="$tot" -v n="$N" -v i="$i" "BEGIN{
                    printf \"balance ep%d: %d reqs (%.2f%%, ideal %.2f%%)\n\", i, x, 100*x/t, 100/n }"
            done
        else
            echo "balance $TAG: no request deltas (?)"
        fi

        touch "$OUT/gen${GEN}.stop"
        trap - EXIT
        stop_local
    ' || MAIN_RC=$?
    if [ "$REMOTE_N" -gt 0 ]; then
        # The main step reaches its own "touch stop" only on the happy path and on
        # its two post-run gate failures. Every earlier FATAL (driver/dataset
        # missing, local boot, GPU-count gate, remote-ready gate, cross-node
        # reachability) exits without it -- and those are exactly the failures a
        # first multinode run hits. Releasing the remote step here instead means a
        # failed main step costs seconds, not the rest of the allocation with 8
        # GPUs held and the real error an hour up the log.
        touch "$MN_OUT/gen${GEN}.stop"
        wait "$REMOTE_STEP" || true
        if [ ! -f "$MN_OUT/gen${GEN}.remote_done" ]; then
            echo "WARN: remote teardown unconfirmed (gen $GEN) -- tail of remote log:"
            tail -n 20 "$MN_OUT/remote_gen${GEN}${RSUF}.log" 2>/dev/null || true
        fi
    fi
    [ "$MAIN_RC" -eq 0 ] || exit "$MAIN_RC"
done

echo "DONE. Artefacts in: $MN_OUT"
echo "Pull home:  rsync -av <clariden>:$MN_OUT analysis/figures/data/dp_scaling/"

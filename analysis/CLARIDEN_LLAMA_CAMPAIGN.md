# Clariden campaign — cross-family generalization on Llama-3.1-8B-Instruct

## Objective

Every result in the thesis so far is measured on `Qwen/Qwen2.5-7B-Instruct`. This
campaign re-runs the load-bearing experiments on a second model family to show the
findings are not Qwen-specific. **Model: `meta-llama/Llama-3.1-8B-Instruct`** — closest
match on size (8B vs 7B), dense, instruction-tuned, uniform GQA with no sliding window
and no logit soft-capping, so prefix-cache accounting behaves as it does for Qwen. That
last property matters because the primary engine metric is
`computed tok/s = (prompt − prefix_cache_hits + generation) / elapsed`.

Nothing else changes: same dataset, same prompts, same JSON tuple encoding, same
`--max-model-len 16384`, same guided-decoding backend, same cold-fleet discipline.

## Preflight — all blocking, do these first

1. **The model is HF-gated and every script runs with `HF_HUB_OFFLINE=1`.** Accept the
   licence and pre-download the weights on the **login node** (compute nodes have no
   internet). A cache miss shows up as a fatal vLLM boot, not as a clear error.
   ```bash
   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
   ls "$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct" 2>/dev/null || \
     ls ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct
   ```
2. **Pull both repos** so the `MODEL` override is present.
   `cross_system_analysis_clariden.sh` and `tp_scaling_clariden.sh` previously hardcoded
   the model; they now read `${MODEL:-Qwen/...}`. Verify:
   ```bash
   grep -n 'MODEL="${MODEL' ~/projects/sembench/slurm/cross_system_analysis_clariden.sh \
                            ~/projects/flock/analysis/slurm/tp_scaling_clariden.sh
   ```
   If either still shows a hardcoded Qwen id, **stop** — the job would silently produce
   Qwen numbers under Llama filenames, which is unrecoverable without re-running.
3. **Binaries exist** (both are model-independent; rebuild only if missing):
   `~/projects/flock/build/flock_sem_filter_vllm_integration` and
   `~/projects/flock/build/release/extension/flock/flock.duckdb_extension`.

## Experiments

Run **A first** and check it before queueing the rest — it is the cheapest end-to-end
validation that the model works through every arm. A and E need 1 GPU; B, C and D need
4 and can queue in parallel with A/E if the partition allows.

| # | what it generalizes | GPUs | ~cost |
|---|---|---|---|
| A | engine lever, all of cross-system, prompt lever | 1 | 70 min |
| B | DP linear / TP sub-linear / DP > TP | 4 | 3 × 35 min |
| C | replicas beat shards at a fixed budget | 4 | 3 × 35 min |
| D | the cap knee under a different KV footprint | 4 | 3 × 35 min |
| E | extract standing | 1 | 60 min |

```bash
export MODEL="meta-llama/Llama-3.1-8B-Instruct"

# ---- A. cross-system filter, 12 arms, 3 reps -------------------------------
cd ~/projects/sembench
QUERY=101 N_REPS=3 MODEL="$MODEL" \
  ARMS="flock_op_r1 flock_op_r32 flock_op_slim_r1 flock_op_slim_r2 flock_op_slim_r4 flock_op_slim_r8 flock_op_slim_r16 flock_op_slim_r32 flock_scalar flock_scalar_r1 lotus palimpzest" \
  sbatch slurm/cross_system_analysis_clariden.sh

# ---- B. DP + TP in one job (both axes, same node) --------------------------
cd ~/projects/flock
for i in 1 2 3; do REP="llama_unified_rep$i" MODEL="$MODEL" \
  CONFIGS="1:1:128 1:2:256 1:4:512 2:1:256 4:1:512" \
  sbatch --time=01:00:00 analysis/slurm/dp_scaling_clariden.sh; done

# ---- C. fixed 4-GPU budget: replicas vs shards -----------------------------
for i in 1 2 3; do REP="llama_grid_rep$i" MODEL="$MODEL" \
  CONFIGS="4:1:512 2:2:512 1:4:512" \
  sbatch --time=00:35:00 analysis/slurm/dp_scaling_clariden.sh; done

# ---- D. cap sweep at N=4 (128 and 512 come from B/C) -----------------------
for i in 1 2 3; do REP="llama_caps_rep$i" MODEL="$MODEL" VERDICTS=0 \
  CONFIGS="4:1:256 4:1:1024 4:1:2048" \
  sbatch --time=00:35:00 analysis/slurm/dp_scaling_clariden.sh; done

# ---- E. cross-system extract, 11 arms, 3 reps ------------------------------
cd ~/projects/sembench
QUERY=103 N_REPS=3 MODEL="$MODEL" \
  ARMS="flock_op_r1 flock_op_r32 flock_op_slim_r1 flock_op_slim_r2 flock_op_slim_r4 flock_op_slim_r8 flock_op_slim_r16 flock_op_slim_r32 flock_scalar lotus palimpzest" \
  sbatch slurm/cross_system_analysis_clariden.sh
```

## After each job — check before trusting

- **The right model actually served.** `grep -m1 'model=' <vllm log>` in the job dir must
  name Llama. The cross-system jobs land in
  `analysis/results/cross_system_llama-3-1-8b-instruct/<jobid>/` (the path is derived
  from the model id, so a Qwen run cannot overwrite a Llama one).
- **KV cache size will differ from Qwen's** and that is expected, not a fault: Llama-3.1
  has 32 layers × 8 KV heads against Qwen's 28 × 4, so KV per token is ~2.3× larger and
  the reported `GPU KV cache size: N tokens` should be roughly half of Qwen's 1,278,032.
- **The scripts' own gates must pass**: fleet engaged (exactly N×TP GPUs >20 GB), rewrite
  engaged (operator ≥ `MIN_OP_RATE`), and for the DP curve jobs the operator/scalar ratio
  ≥ 3. If the rewrite gate trips because Llama is slower, do **not** lower it silently —
  report the measured rate.
- **Arm count**: A should produce 36 `*_flockmtl.json` (12 arms × 3 reps, counting
  LOTUS/PZ separately) — verify none of the twelve arms is missing before importing.

## Import convention

Keep Llama artefacts in their own directories; do not merge them into the Qwen ones.

```bash
cd ~/projects/flock
mkdir -p analysis/figures/data/cross_system_sem_filter_llama \
         analysis/figures/data/cross_system_sem_extract_llama \
         analysis/figures/data/dp_scaling_json_llama
cp analysis/results/cross_system_llama-3-1-8b-instruct/<jobid-A>/* \
   analysis/figures/data/cross_system_sem_filter_llama/
cp analysis/results/cross_system_llama-3-1-8b-instruct/<jobid-E>/* \
   analysis/figures/data/cross_system_sem_extract_llama/
cp analysis/results/dp_scaling_json/<jobid-B,C,D>/* \
   analysis/figures/data/dp_scaling_json_llama/
git add analysis/figures/data/*_llama
```
Then commit per family (one commit for the cross-system imports, one for the scale-out
imports) and push.

## Do not

- Do not lower `MIN_OP_RATE`, change `--max-model-len`, or switch the attention backend
  to make a job pass. Any of those changes the configuration held fixed across the whole
  thesis; report the failure instead.
- Do not re-run the Qwen arms. They are final and single-session; a re-run would replace
  a citable table with a mixed one.
- Do not set `FLOCK_VERDICT_DUMP` on a timed run — the dump writes during execution and
  inflates the timing (measured at up to +24% on short arms).

## Predictions, registered before the runs

Recorded so a hit counts as evidence rather than a post-hoc fit. Operator/scalar ~7–8×
at one morsel (a concurrency argument that does not reference the model). DP per-GPU
~1.0, TP per-GPU <0.8 at TP=4, so DP > TP. Replicas beat shards, monotone. Cap knee near
128·N — the larger KV should not bind, since occupancy was 1–2% at 7B and stays under 5%
even at 2.3×. Slim ~2.3× full on rows/s at R=1, and batching still degrades F1 with R.

**Quality is deliberately unpredicted.** Llama's calibration on "clearly positive" is
unknown, and where its precision/recall balance lands is the interesting result.

**One measurement caveat**: Llama's tokenizer differs, so tok/row and computed tok/s are
not comparable across families. Recompute the engine ratio against Llama's *own* LOTUS
baseline; never compare it to Qwen's absolute value.

# RESUME — unbounded sem_extract response schema

**Status 2026-08-19. The measurement campaign is COMPLETE on both model families.**
Job **`3116950`** landed clean and passed every gate, including the peer gate that blocked
`3115747`. The Llama data is **imported** into `analysis/figures/data/cross_system_sem_extract_llama/`
with an `INPUT_PROVENANCE.md`, and that directory now has a `cross_system_summary.csv` for
the first time. Nothing about the change is provisional any more.

Branch `fix/extract_unbounded_schema`, code commit **`2565fe75`** (unchanged — no code was
touched this session). Qwen data is already committed and pushed as `7e1329f5`.

**Uncommitted, awaiting the user's commit** (see *Staged for commit* below): the Llama
import, its provenance file, its summary CSV, a factual fix to the Qwen provenance file,
and this file.

**The only work left is outside this repo**: the §5 ratio restatement and the corrections
note, both in the paper checkout, which does not live on this machine.

---

## Verdict

`PhysicalSemExtract::BuildResponseFormat` no longer puts a `maxLength` on the per-row
string element; the ceiling is the request-level `max_tokens` alone, with
`minItems`/`maxItems` still pinning the answer count.

**Generation is provably unchanged, throughput is up a lot, on both families.**

| gate | Qwen (merged `7e1329f5`) | Llama (merged, `3116950`) |
|---|---|---|
| `prompt_tok` / `gen_tok` unchanged | **PASS** — 11/11 cross-system arms, 14/16 flock cells (2 scalar cells ≤0.009%) | **PASS** — `prompt_tok` bit-identical 33/33; `gen_tok` 26/27 flock cells, one at 2 tok of 8435 |
| quality unchanged | **PASS** — all 11 arms, shift < own rep spread | **PASS** — all 11 arms |
| operator throughput up | **PASS** — +8.4% to +101% | **PASS** — +17% to +92% |
| scalar arms unchanged | **PASS** — cross-system +0.2%; morsel scalars ±5% | **PASS** — **40.4 -> 40.4**, bit-identical |
| non-flock controls unchanged | **PASS** — palimpzest −1.2%; lotus +5.2% on n=2 | **PASS** — palimpzest +3.8%; lotus +16.5%, both n=3 (see note) |
| m128 ratio >= 1.0x | **PASS** — **0.910x -> 1.025x** | n/a (Qwen-only surface) |

**Every gate in the study now passes.** The one failing gate as of Session 2 — the Llama
peer controls — was node-level degradation on nid006687 and is gone.

## §5 — the ratio roughly doubles, on three independent surfaces

| surface | old | new |
|---|---|---|
| Qwen thread sweep t16 (558.3 / 43.0) | 7.12x | **12.98x** |
| Qwen cross-system `flock_op_r32` / `flock_scalar` (537.9 / 43.0) | 6.64x | **12.51x** |
| **Llama** cross-system `flock_op_r32` / `flock_scalar` (498.8 / 40.4) | 7.27x | **12.35x** |

Three surfaces, two model families — **12.98 / 12.51 / 12.35, agreeing to ~5%**. The Llama
best arm (`flock_op_slim_r16` 557.3 / `flock_scalar` 40.4) is **13.79x**, against 7.17x
bounded. That the doubling reproduces across model families is a materially stronger claim
than Qwen alone and is the headline change to §5.

**Any paper text quoting ~7.0x needs rewriting.** Nothing in *this* repo still quotes the
stale number. The `~7.7×` figures in `analysis/figures/sem_filter_ab_conclusion.md` are
**sem_filter**, a different operator that never touches `BuildResponseFormat`; they are
correctly untouched. There is no paper checkout on this machine, so that edit must happen
wherever it lives.

Note the Llama numbers moved from the provisional table: the §5 surface is **12.35x**, not
the 12.18x computed from `3115747`. Use 12.35x.

## The mechanism: `tok_per_row`, not R

The gain is **not** a function of R. Qwen's two R=1 cross-system arms differ by 3x (+20.5%
vs +67.3%) and the slim family is an inverted U. What is near-constant is the **absolute**
saving: **1.48–1.90 ms/row** on every Qwen operator arm, **1.30–1.70 ms/row** on every Llama
operator arm, vs ≤0.20 ms/row on the Qwen controls and 0.00 on `flock_scalar` in both.
Relative gain = that constant / the arm's baseline per-row cost, so prefill-dominated arms
gain least in percentage terms. R is only a proxy, because raising R amortises the prompt and
drives `tok_per_row` down.

Llama reproduces it independently: smallest gain `flock_op_r1` at +17.0% (536 tok/row, the
most prefill-dominated arm), largest `flock_op_slim_r16` at +92.3% (57 tok/row). Three
surfaces now agree on the mechanism.

## Final Llama table (job `3116950`, all arms n=3)

| arm | tok/row | bounded | unbounded | x |
|---|---|---|---|---|
| `flock_op_slim_r16` | 57.3 | 289.8 | **557.3** | 1.92x |
| `flock_op_slim_r8` | 65.7 | 289.7 | 555.6 | 1.92x |
| `flock_op_slim_r4` | 83.5 | 283.0 | 515.1 | 1.82x |
| `flock_op_r32` | 54.7 | 293.8 | 498.8 | 1.70x |
| `flock_op_slim_r32` | 53.1 | 286.3 | 478.7 | 1.67x |
| `flock_op_slim_r2` | 119.0 | 259.2 | 420.1 | 1.62x |
| `flock_op_slim_r1` | 176.0 | 206.7 | 318.2 | 1.54x |
| `flock_op_r1` | 536.0 | 111.4 | 130.3 | 1.17x |
| `flock_scalar` | 54.7 | 40.4 | 40.4 | 1.00x |
| `lotus` | 223.8 | 126.1 | 146.9 | 1.17x |
| `palimpzest` | 661.2 | 205.5 | 213.4 | 1.04x |

These are the summarizer's medians (`--query 103 --rows 2000`); means differ in the first
decimal and the ratios are unaffected. Flock numbers came back **higher** than `3115747`'s
provisional table (`slim_r16` 530.7 -> 557.3), so that job was mildly degrading the flock
arms too — its numbers were conservative, not inflated.

### The between-node confound, and why it does not reach the flock arms

The three jobs ran on three different nodes: `3083182` nid007117, `3115747` nid006687,
`3116950` nid006240. So every cross-era comparison carries a between-node component, and on
Llama it is visible — `lotus` gained **+16.5%**, which cannot be the schema change because
LOTUS never enters flock code.

**`flock_scalar` bounds the effect on the flock path at zero**: 40.4 -> 40.4 rows/s across
nid007117 -> nid006240, with `f1_score` bit-identical across all 3 reps in both eras. The
two nodes are equivalent *for this client*, so the operator arms' 1.30–1.70 ms/row is a
schema effect, not a node effect. The `lotus`/`palimpzest` movement is a
CPU-bound-Python-client sensitivity — same class as `3115747`, milder and in the opposite
direction.

**Consequence for the paper: quote ratios, never absolute Llama `rows_s` against Qwen
`rows_s`.** Both arms of each ratio ran in the same job on the same node, so the ratio is
confound-free by construction.

## The cross-family era split is closed

Both families are now unbounded, so the corrections note should be rewritten around the
same-era pair **Qwen `flock_op_r1` 139.1 vs Llama `flock_op_r1` 130.3 rows/s**. (Not 114.0 —
that was the `3115747` provisional value and is superseded.) The old text explained
**115.4 vs 111.4** as "different models, not a transcription slip"; both were bounded-era
when written, so it was legitimate then, but **115.4 exists nowhere in the data now**.
Quality claims were never affected — `model_scale.md` §7.4 and chapter line 360 survive
untouched, because F1 did not move in either family.

## The m128 question is settled

`3112334` (nid007106, 1:11:31, exit `0:0`) re-measured the cell that `3103520`
contaminated. **The stall did not reproduce**: interior `Running == 0` fell 24.9% -> 1.5%,
and the scalar control returned to baseline (129.1 vs 134.1, −3.7%) instead of collapsing
13%. So `3103520` was node-level degradation on nid006740, not a regression in the build.
`3103520` is discarded and was never imported.

Operator **132.309** rows/s vs scalar **129.061** = **1.025x**, against 0.910x before.
The operator stops being behind at equal concurrency, which was the substantive claim.

One subtlety worth not re-deriving: the re-measure's mean `Running` (60.6) is *below*
baseline's (78.1) even though it is faster. That is the expected signature, not residual
stalling — by Little's law per-request residency drops 0.640 -> 0.458 s (−28%), so fewer
requests are resident while throughput rises. The discriminator against a real stall is
`Waiting`, which went **up** (5.0 -> 10.4): work is queued at all times, so the GPU is
never starved. A stall looks like the opposite — `Running` at 0 with the driver not feeding.

---

# `3116950` — the confirming re-run, and why `3115747` is discarded

`3116950`: nid006240, COMPLETED, exit `0:0`, **53:24** of a 2:00:00 request, submitted with
`--exclude=nid006687`. 198 files, exact 1:1 filename match with the bounded era on all 165
non-log files. **All 33 reps succeeded — every arm is n=3**, including `lotus`.

The peer gate, which the doc gated everything else on, passes decisively:

| | bounded `3083182` | degraded `3115747` | **`3116950`** |
|---|---|---|---|
| `palimpzest` rows_s | 205.5 | 42.2 (0.20x) | **213.4** |
| `lotus` rows_s (n) | 126.1 (3) | 59.7 (1) | **146.9 (3)** |
| peer peak `Running` | 55–93 | 9–30 | **51–63** |
| `flock_scalar` rows_s | 40.4 | 38.1 | **40.4** |

Peer throughput and achieved concurrency both returned to baseline, and `flock_scalar`'s
provisional −5.6% went away entirely. The `3115747` diagnosis — node-level degradation on
nid006687 hitting only CPU-bound Python clients — is confirmed on every count. `3115747`
is **discarded and was never imported**; its flock half was clean and is now superseded.

## Import — DONE

```
figures/data/cross_system_sem_extract_llama/   198 files from 3116950 (165 replaced 1:1,
                                               33 unbounded vllm logs added alongside the
                                               33 retained bounded-era logs) = 231 files
                                               + cross_system_summary.csv  (NEW)
                                               + INPUT_PROVENANCE.md       (NEW)
```

Verified byte-identical against the job dir for all 198 files. The bounded-era vLLM logs
were **retained** because `analysis/results/` is gitignored, making them the only committed
copy of that era's server-side evidence — same mixed-era convention as the Qwen dir
(117 logs = 84 bounded + 33 unbounded). No summarizer or plot script reads the logs.

One harmless oddity, already chased down: `warmup_flock_op_slim_r8_rep1.log` is the single
file that came back byte-identical to its predecessor rather than modified. The only
variable field in a warmup log is the 20-row sf_20 warmup wall-clock, and that rep hit
`1.03s` in both jobs. At 10 ms resolution over ~30 samples that is a likely coincidence, not
a stale file.

## Staged for commit (user commits and pushes)

- `analysis/figures/data/cross_system_sem_extract_llama/` — 164 modified, 34 new
  (33 vllm logs + `INPUT_PROVENANCE.md`), plus `cross_system_summary.csv`
- `analysis/figures/data/cross_system_sem_extract/INPUT_PROVENANCE.md` — corrected the
  verdict-dump variable (it said `FLOCK_VERDICT_DUMP`; the knob for
  `cross_system_analysis_clariden.sh` is `VERDICT_DUMP=1`, and the script `unset`s the
  former, so the old text pointed at the trap rather than away from it)
- `analysis/RESUME_extract_unbounded_schema.md` — this file

## Remaining

- **In the paper checkout, not on this machine:** restate 7.0x -> ~12.98x and
  6.64x -> 12.51x, and add the Llama **12.35x** surface.
- **In the paper checkout:** rewrite the corrections note around Qwen 139.1 vs Llama 130.3.

Nothing else. No further jobs are needed.

---

## Open caveats

- **No absolute F1 exists for the unbounded era, on either family.** `3103523`, `3115747`
  and `3116950` were all run without `VERDICT_DUMP` (deliberately — the dump writes during
  the run and makes timings non-citable), so `analysis/rescore_extract.py` cannot run on any
  of them. The retained `extract_rescored.csv` and 8 `verdicts_*.jsonl` in
  `cross_system_sem_extract/` are **bounded-era only** and are kept because they cannot be
  regenerated. The quality gate does not depend on them — it applies the harness scorer
  consistently to both eras, which is valid for detecting a *shift* even though that scorer
  is the known-buggy macro-average. **Do not publish an absolute unbounded-era F1** without
  a separate re-run with the dump enabled, whose timings would then not be citable.
- **`lotus` n is short on Qwen only.** Qwen's +5.2% rests on **n=2**; report the n. Llama is
  now n=3 on every arm, so the Llama caveat from Session 2 is retired.
- **`lotus` +16.5% on Llama is a between-node artefact, not a control failure.** See the
  confound section above; `flock_scalar` at exactly 40.4 -> 40.4 is what rules it out as
  affecting the flock path.
- Both `cross_system_sem_extract/` and `cross_system_sem_extract_llama/` are **mixed-era by
  design** in their vLLM logs only. Neither summarizer nor either plot script reads the
  logs, so no output is contaminated. Each directory's `INPUT_PROVENANCE.md` carries the
  file-by-file era table.

## Traps (do not re-learn these)

### Validity gates

- **`Waiting: 0` is not a validity gate** at 128-way concurrency, and neither is mean
  `Running > 0`. `Waiting: 0`, exit `0:0`, `pass_pct` 100 and `emitted == rows == passes`
  **all pass on a contaminated run**. The gate is the count of interior `Running == 0`
  samples, between the first and last active sample. The checker script is preserved in
  `sem_extract_ab_json/INPUT_PROVENANCE.md`.
- **A slow arm with bit-identical tokens is a *client-side* problem, and the vLLM
  Prometheus metrics prove which side.** The discriminating triple is: `prompt_tok`
  identical, `queue_time` ~0, and server-side `e2e_request_latency` flat or *lower* while
  wall-clock rises. That combination can only mean the client stopped issuing requests —
  confirm with `Running` from the `loggers.py` lines, which is the achieved concurrency, not
  the requested one. `--concurrent-llm-worker 128` is what was *asked for*; `Running` is
  what happened. This is how the `3115747` peer collapse was pinned in minutes.
- **Peak `Running` is undersampled on short arms.** vLLM emits a state line about every 10 s
  and the peer arms run ~9 s, so a *healthy* rep can show 2–3 samples and miss the burst
  entirely — `palimpzest` rep3 of `3116950` shows one active sample at **2 reqs** despite a
  perfectly healthy 205.3 rows/s. Never fail an arm on peak `Running` alone; cross-check
  wall-clock. The `3115747` comparison was safe only because its collapse made the arms ~5x
  longer, giving *more* sampling opportunity, all of it capped at ≤11.
- **The "F1 bit-identical / ±0.003" gate was wrong** — baseline `slim_r1` was never
  bit-identical across its own reps (spread 0.0030), and untouched `palimpzest` drifts
  0.0089. Gate on "shift < own rep spread" instead, and **estimate the spread from every
  job you have, not just the two being compared**. Llama `flock_op_slim_r8` looks like a
  failure on `3083182`-vs-`3116950` alone (shift 0.0007 > spread 0.0004) purely because the
  baseline happened to come back bit-identical 3x; `3115747` measures that same arm across
  0.853984–0.855333, a spread of 0.00135 that contains both era means.
- **The "gain scales with R" mechanism was wrong** — it is `tok_per_row` / prefill
  dominance, with a near-constant absolute saving.
- **Cross-era jobs land on different nodes, so bound the node effect before trusting an
  absolute delta.** The bound that works is a *same-client* control that the change cannot
  touch: `flock_scalar` 40.4 -> 40.4 proved the flock path was node-insensitive even though
  `lotus` moved +16.5% on the same pair of nodes. A Python peer is the wrong control for
  this, because the Python clients are exactly what node degradation hits.

### Misleading log lines

- **`401 Unauthorized` in the vLLM logs is a red herring.** There are 6 across all 33 logs
  of `3115747`, all on unauthenticated `GET /v1/models` readiness probes at boot, 2 per
  fleet in **every** rep including the successful ones. Every `POST /v1/chat/completions`
  in the whole job returned 200 (31,406 of them, zero non-200). Do not chain a peer failure
  to these.
- **`grep -c '500 '` on a vLLM log matches port numbers, not status codes** (`127.0.0.1:38500`).
  Match `HTTP/1.1" <code>` instead.
- **`lotus: ✅ Completed` is printed even when the query failed** — the arm-level line is not
  a success gate; the per-query line (`Q103: ❌`) is. Cross-check the JSON `status` field.

### Environment and tooling

- **`lotus` has no retry.** A single bad response raises out of
  `lotus/models/lm.py:214 _get_top_choice` (called from a list comprehension at
  `lm.py:119`) and kills the entire arm, losing all 2000 rows. This is why one transient
  server error costs a whole rep, and why Qwen's `lotus` n is short.
- **`VERDICT_DUMP=1` and citable timings are mutually exclusive** in
  `sembench/slurm/cross_system_analysis_clariden.sh`. The script's own comment (line 207):
  the dump writes *during* the run, so "timings from such a job must not be cited". A job
  needing both requires two runs.
- **The knob is `VERDICT_DUMP`, NOT `FLOCK_VERDICT_DUMP`.** Passing `FLOCK_VERDICT_DUMP=1`
  to `cross_system_analysis_clariden.sh` is not merely ignored — line 210 explicitly
  `unset`s it before the script sets it itself from `VERDICT_DUMP`. This is exactly why
  `3103523` produced no dumps, and it fails **silently**. Note `FLOCK_VERDICT_DUMP` *is* the
  variable the flock binary reads, and the many `analysis/slurm/*.sh` scripts set it
  directly and correctly — the trap is specific to the cross-system script's wrapper.
- **`cross_system_analysis_clariden.sh` has no `MIN_OP_RATE` gate** (that is the sem_filter
  script), so slow arms like `flock_scalar` at ~40 rows/s do not abort it. It also exports
  `HF_HUB_OFFLINE=1` itself at line 47 — do not add it.

### Data hygiene

- **`summarize_morsel_ab.py` needs both args** (defaults to `sem_filter_ab_json` on both
  sides); **`summarize_cross_system.py` defaults to query 101** — extract is **103**. Its
  `--out-dir` defaults to `--results-dir`, so pointing `--results-dir` at a figures dir
  writes the CSV in place. Run it under `/usr/bin/python3.11`; the login node's default
  `python3` is 3.6 and cannot parse it.
- **Never compare `rows_s` between the thread sweep and the morsel sweep** — different R,
  different input, `tok_per_row` 55.3 vs 513.9. Only operator-vs-scalar *within* a sweep.
  Likewise never compare Qwen `rows_s` to Llama `rows_s`; compare ratios.
- `morsel_ab_summary.csv`: `(config, arm, rep)` is **not** a unique key — the thread-sweep
  `t8` cell and the morsel `m1_t8` cell collide on `1morsel_1datachunk_8threads`. Use
  `(config, arm, rep, R)`.
- The m128 cell needs `MORSELS * ROW_GROUP_SIZE` = 262,144 rows, so only `sf_300000` is
  big enough; it is untracked and non-deterministically regenerated, so absolute `rows_s`
  is not comparable to the XML era. Use `--time=02:00:00` (3103520 used 1:24:03 of 1:30).
- The Llama bounded-era baseline is **`3083182`**, not `3089338`. Both live under
  `analysis/results/cross_system_llama-3-1-8b-instruct/`; only `3083182`'s vllm logs
  (`vllm-17867415xx`+) match `figures/data/cross_system_sem_extract_llama/`.
- **`analysis/results/` is gitignored.** Raw job dirs are not committed, so any bounded-era
  artefact that only exists there is lost on scratch cleanup. That is why the figures dirs
  retain old vLLM logs instead of replacing them.

---

## Session log

### Session 1 — Qwen campaign closed

1. `3112334` validity gate (interior `Running == 0`) — **passed**, stall did not reproduce.
2. Token-identity check on the m128 cell — `prompt_tok` bit-identical, `gen_tok` +9 of
   2.62M (0.0003%), zero preemptions, 262,145 successes.
3. F1 stability gate — **all 11 arms pass**.
4. Imported everything at once, so no summary ever passed through an era mix:
   - `3103519` + `3103521` + `3112334` -> `analysis/figures/data/sem_extract_ab_json/`
     (72 files, exact 1:1 filename replacement — verified by set diff; no `vllm-ep*.log`,
     per this directory's convention)
   - `3103523` -> `analysis/figures/data/cross_system_sem_extract/` (198 files incl.
     `vllm-*.log`, per *that* directory's convention)
5. Regenerated `morsel_ab_summary.csv` and `cross_system_summary.csv`.
6. Rewrote `sem_extract_ab_json/INPUT_PROVENANCE.md` and **created**
   `cross_system_sem_extract/INPUT_PROVENANCE.md`, both recording the schema-era split.
7. Submitted `3115747` (Llama) to close the cross-family era split.

### Session 2 — Llama measured, peer degradation diagnosed, re-run queued

1. `3115747` landed clean (exit `0:0`). Token-identity gate across eras for all 33 arm-reps
   from the vLLM Prometheus `metrics_before/after` diffs — **27/27 flock cells
   bit-identical** on both `prompt_tok` and `gen_tok`.
2. F1 gate — all 9 flock arms pass.
3. Found the peer regression and diagnosed it to client-side concurrency collapse, ruling
   out fleet config, script, Python env, arm ordering, node sharing and jitter.
4. Established the 401s are a red herring and that `lotus` is n=1, not n=2.
5. Computed a provisional third §5 surface (12.18x).
6. Submitted `3116950` with `--exclude=nid006687`. Imported nothing, by design.

### Session 3 — `3116950` confirms everything; Llama imported; campaign closed

1. `3116950` COMPLETED, exit `0:0`, nid006240, 53:24. 198 files, all 33 reps successful.
2. **Peer gate first, as required — PASS.** `palimpzest` 42.2 -> 213.4 (vs 205.5 bounded),
   `lotus` 59.7 (n=1) -> 146.9 (n=3, vs 126.1), peer peak `Running` 9–30 -> 51–63 against a
   bounded-era 55–93. Node-level diagnosis confirmed; `3115747` discarded.
3. Token-identity gate — `prompt_tok` bit-identical **33/33 arm-reps**; `gen_tok`
   bit-identical in 26/27 flock cells, lone exception `flock_op_r32` rep1 at 8435 -> 8437
   (2 tokens); `total_tok` bit-identical in all 9 flock arms at the summary level.
4. F1 gate — all 11 arms pass. Resolved the apparent `slim_r8` miss using `3115747` as a
   third spread sample (see Traps).
5. Found and bounded the **between-node confound** (three jobs, three nodes; `lotus` +16.5%
   cannot be the change). `flock_scalar` 40.4 -> 40.4 proves the flock path is
   node-insensitive, so the operator ms/row is real.
6. Imported 198 files 1:1 into `cross_system_sem_extract_llama/`, verified byte-identical,
   retaining the 33 bounded-era vLLM logs (231 files total).
7. Built that directory's **first** `cross_system_summary.csv` and wrote its
   `INPUT_PROVENANCE.md`.
8. Corrected the `FLOCK_VERDICT_DUMP` -> `VERDICT_DUMP` error in the Qwen provenance file.
9. Final §5 surface: Llama **12.35x** (supersedes the provisional 12.18x).

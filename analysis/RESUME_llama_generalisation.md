# RESUME — Llama-3.1-8B cross-family generalisation campaign

**Status 2026-08-15: MEASUREMENT COMPLETE — including `3087979` and the below-knee cap cell
`3088915/6/7`. Nothing is in flight and nothing is outstanding except committing.** All five
experiments (A–E), both palimpzest sensitivity arms and the cap=128 starvation cell are run,
gated and imported. The runaway open question is **RESOLVED** (see "palimpzest runaway
generation" below).

Both repos: flock is at `cee240d6` with the Llama data staged but **not committed**;
sembench has an uncommitted one-comment fix to `slurm/cross_system_analysis_clariden.sh`
(see "The 3083303 failure" below) that should be committed.

---

## Final job map

| job | what | outcome |
|---|---|---|
| `3082227` | A, 12 arms × 3 reps | TIMEOUT at 01:30, 34/36 runs — 33 kept |
| `3083163` | A: `palimpzest` ×3, the missing arm | COMPLETED — 3 kept |
| `3083167` | B rep1 (canary) | COMPLETED |
| `3083171/2` | B reps 2–3 | COMPLETED |
| `3083173/4/5` | C reps 1–3 | COMPLETED |
| `3083176/7/8` | D reps 1–3 | COMPLETED |
| `3083182` | E, extract Q103 | COMPLETED |
| `3083254` | PZ ×8, runaway frequency at inherited temp 0.6 | COMPLETED |
| ~~`3083303`~~ | PZ ×8 greedy sensitivity | **FAILED in 21s — script bug, see below** |
| `3087979` | PZ ×8 greedy sensitivity, resubmitted after the fix | COMPLETED 35:56 — 8 kept, **5/8 runaways** |
| `3088915/6/7` | scale-out: N=4 TP1 **cap=128**, the below-knee cell, 1 rep each | COMPLETED ~4:45 each — 3 kept |

The canary strategy worked: no FlashInfer JIT lock failure, no `DependencyNeverSatisfied`.

## Gates — ALL PASS

- **78/78 cross-system JSONs** pass `keys == [Q]`, `status == 'success'`, and
  `extra.run_label == '<arm>_rep<n>'` for the flock arms. LOTUS/PZ carry `extra == {}` and
  have no `run_label`, as in the locked Qwen files.
- **36/36 dp_scaling result JSONs**, `model == meta-llama/Llama-3.1-8B-Instruct` and
  `tuple_format == json` on every one (no mixed encoding, so the `summarize_dp` guard
  passes). Families are `llama_unified`, `llama_grid`, `llama_caps`, `llama_caps128`,
  3 reps each.
- Every vLLM server in all 16 jobs reports snapshot `0e9e39f2…c89659`. No Qwen leakage.
- All 18 N=4 TP1 cells share one workload: 509.4 tok/row, 8.8 gen tok/req, 51.2% prefix
  hit, 32001 requests, 99.8% pass. Only the in-flight cap differs.
- All script gates passed on their own: **`MIN_OP_RATE` was never tripped** and did not need
  lowering — Llama's slowest DP cell is 128.8 rows/s, far above the 60 default.

## Results — the headline

**The performance story generalises tightly. The quality-vs-batch-size story does not.**

### Scale-out (B/C/D) — 5 of 5 registered predictions HIT

| prediction | measured | |
|---|---|---|
| DP per-GPU ≈ 1.0 | 1.03 (N=2), 1.01 (N=4) | HIT |
| TP per-GPU < 0.8 at TP=4 | 0.68 | HIT |
| DP > TP at 4 GPU | 518.6 vs 349.6 rows/s = 1.48× | HIT |
| replicas beat shards, monotone | 4×TP1 518.7 > 2×TP2 392.6 > 1×TP4 351.3 | HIT |
| cap knee near 128·N | flat from cap 512 (=128·4) onward — full sweep below | HIT |

Normalised, Llama sits on top of Qwen's own curve — efficiency 1×TP2 0.76 vs 0.82,
1×TP4 **0.68 vs 0.69**, 2×TP1 1.03 vs 1.06, 4×TP1 1.01 vs 1.07. Compare **ratios only**;
absolute rows/s are not cross-family comparable here (regenerated `sf_300000`, see
`figures/data/dp_scaling_json_llama/INPUT_PROVENANCE.md`).

### The cap sweep at N=4 TP1 — completed by `3088915/6/7`

`3088915/6/7` add the missing **below-knee** point (cap=128, one client slot per 32 rows of
in-flight budget vs the 4 endpoints). Median of 3 reps, `llama_*` families pooled per cap:

| cap | rows/s (median) | reps | spread | eff_gpu | fleet in-flight | wait_mean | e2e ms |
|---|---|---|---|---|---|---|---|
| **128** | **447.2** | 3 | **0.5%** | 0.87 | 98 / 128 | **0.0** | **279** |
| 256 | 497.0 | 3 | 17.8% | 0.97 | 185–196 | 0.0 | 496–502 |
| 512 | 518.2 | 6 | 1.8% | 1.01 | 379–402 | 0–15 | 953–972 |
| 1024 | 520.8 | 3 | 5.6% | 1.01 | 724–762 | 38–58 | 1867–1948 |
| 2048 | 524.9 | 3 | 5.7% | 1.02 | 991–1077 | 651–670 | 3712–3873 |

Three things the new point establishes that the sweep could not show before:

1. **The knee is client-side starvation, and the mechanism is now visible.** At cap=128 the
   fleet runs ~98 concurrent requests against an offered 128, `wait_mean` is **exactly 0.0**
   and `queue_ms` is 0 on all three reps: nothing ever queues at the server, the client
   simply never offers enough work. At cap=2048 `wait_mean` is 651–670 and `queue_ms`
   1234–1344. The knee separates a client-limited regime from a server-queued one.
2. **Throughput saturates ~16× before latency does.** Over the 128→2048 range rows/s moves
   447→525 (**+17%**) while e2e moves 279→3749 ms (**+1244%**). cap=128 is the
   latency-optimal operating point at 86% of peak throughput; everything above cap 512 buys
   nothing but queue.
3. **The cap=256 "dip" is a node artefact, not a cap effect.** Its mean (473.2) is dragged
   below its median (497.0) by `llama_caps_rep2` at 419.1 — which is *slower than cap=128*.
   That rep reached only 110.9 in-flight (siblings: 185–196) at 0.33 client CPU cores
   (siblings: 0.11–0.12) and 330 ms TTFT (siblings: 139–147). It is job `3083177` on
   `nid007562`, and **all three of that job's cells** (256/1024/2048) are the slowest in
   their cap group with 2.5–3.5× the client CPU of their siblings. One noisy node, three
   cells. **Plot the cap sweep on medians**; the mean misstates the knee shape.

The new cell is the tightest in the study (447.0/447.2/449.3, 0.5% across three *different*
nodes) — partly a real result and partly because a client-limited cell is insensitive to
node speed by construction, which is itself the confirmation that it sits below the knee.

### Cross-system Q101 filter (A)

Operator/scalar **7.72×** at one morsel (predicted 7–8 — HIT), 8.19× at R=32.
Slim vs full at R=1 **2.65×** (predicted ~2.3). Wall-clock per arm is within ~10% of Qwen
(`flock_op_r32` 3.29s vs 3.02s), so the ×1.73 job-level slowdown is vLLM **boot** cost, not
query cost.

**The prediction that MISSED: "batching still degrades F1 with R."** True for Qwen, false
for Llama.

| arm | Qwen F1 | Llama F1 |
|---|---|---|
| `flock_op_r32` (full prompt) | 0.527 (recall 0.374 — collapses) | **0.805** (recall 0.808 — no collapse) |
| `flock_op_slim` R=1 → R=16 | 0.878 → 0.811 (**degrades**) | 0.873 → **0.914** (**improves**) |
| `flock_op_slim_r32` | 0.747 | 0.880 |

Llama's slim arm *peaks* at R=16 and only dips at R=32. Separately, Llama's **full-prompt**
arms are near-degenerate accept-all on this query: `flock_op_r1` and `flock_scalar_r1` both
return ~1994/2000 rows at recall 1.000, precision 0.762. The slim prompt is the one that
discriminates (precision 0.97–0.99). Qwen showed neither behaviour.

### Cross-system Q103 extract (E)

Operator/scalar 7.26× (Qwen 6.75×). Timings track Qwen closely for flock — but **the
baselines move a lot**: LOTUS is ~1.9× *slower* under Llama (15.97s vs 8.36s) and PZ is
~2.4× *faster* (9.72s vs 23.78s). So the engine ratio vs PZ falls from 3.44× (Qwen) to
**1.43×** (Llama). This is exactly why the ratio must be recomputed against Llama's own
baselines and never carried over.

### palimpzest runaway generation — RESOLVED by `3087979`

The runaway is a **degenerate repetition loop, and its probability is set by the decoding
temperature.** It is not a rep-position effect, not a cold-cache effect, and not a
palimpzest bug.

`3083254` (8 reps, inherited `temperature 0.6, top_p 0.9`) was completely clean:
18.05–18.96s, 154.2–154.7k tokens, 0/8. `3087979` is the identical job with the single
change `--override-generation-config {"temperature":0.0}` (greedy), verified applied on all
8 boots:

| rep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| time (s) | **101.10** | **105.83** | **106.85** | **105.59** | 18.14 | 18.23 | 18.04 | **100.87** |
| tokens (k) | **170.5** | **170.2** | **170.3** | **169.9** | 154.7 | 154.5 | 155.2 | **169.7** |
| runaway | ✗ | ✗ | ✗ | ✗ | — | — | — | ✗ |

**5/8 greedy vs 0/8 at temperature 0.6 — Fisher exact p = 0.026.** Against the pooled
temp-0.6 sample (2/12) it is p = 0.062. Lowering the temperature makes the pathology *more*
frequent, which is the textbook greedy-decoding failure mode.

Evidence, each independent:

1. **Prometheus.** Every affected rep has exactly `finished_reason="length"` = 1 and
   `stop` = 2019; clean reps have `length` = 0, `stop` = 2020. The one long request lands in
   the `+Inf` bucket of `request_generation_tokens` (all others ≤ 200). The runaway does not
   stop on its own — it is truncated by `max_model_len 16384`, which is why the token delta
   is near-constant at ~15.3k across every affected rep.
2. **`metrics_before_*` (post-warmup, pre-timed-run) has `length` = 0 in all 16 scrapes.**
   The 20 warmup requests never run away; every runaway is inside the timed 2000-row run.
3. **The transcript.** Palimpzest dumps unparseable output to `parse-answer-errors/`; those
   files are imported alongside the run. The runaway is a verbatim cycle of
   `However, the word "X" is used which is a Y word.` repeated to the context ceiling.
4. **Only three distinct reviews trigger it**, and they recur: the "spectacle" review
   (rep1, rep4), the "bad joke / sad joke" review (rep2, rep8), the "going through the
   motions" review (rep3). It is a property of a handful of ambiguous-sentiment rows, not a
   random request.
5. **The killer.** The temperature-0.6 runaway from `3082227` rep1 is **md5-identical**
   (`186f4932…`, 156055 B) to greedy rep4's. A 15k-token sampled trajectory reproducing a
   greedy one exactly means the loop is an absorbing state where the argmax token has
   probability ≈ 1 — sampling at 0.6 cannot escape it either. That is why lowering the
   temperature raises the rate: it only affects the chance of *entering* the loop.

Why the earlier "both runaways were rep1" pattern was a red herring: under greedy the
runaways are reps 1, 2, 3, 4 and 8 while 5, 6, 7 are clean, so rep position is plainly not
the trigger. The 2/12 rep1 coincidence at p ≈ 4.5% was found post hoc and is not evidence.

Greedy is **not** a fix and buys nothing: quality is statistically indistinguishable
(greedy F1 0.8978–0.9016, precision 0.967–0.972, recall 0.834–0.841; temp-0.6 F1
0.8963–0.9017) and clean-mode time is identical (18.04–18.23s vs 18.05–18.96s). Greedy is
also not deterministic here — row counts vary 1275–1289 across the 8 greedy reps, the same
spread as at temperature 0.6, because continuous batching makes the argmax
batch-composition-dependent.

Reporting guidance, now on firm ground: publish **clean-mode** palimpzest throughput
(18.0–19.0s, **1.2×** Qwen) from the temperature-0.6 arm, which is the inherited default and
the configuration every other system in the comparison ran under. The 3.0× three-rep mean
from `3082227`/`3083163` is one runaway amortised and must not be published as a throughput
number. Report the runaway separately as a decoding-mode pathology with the frequencies
above — 2/12 at temperature 0.6, 5/8 greedy — never as a single pooled rate.

Quality is unaffected in both arms: F1 0.896–0.902, precision ~0.970, recall ~0.837.

## The 3083303 failure — fixed, and the lesson

`3083303` died in 21s with `-c: line 60: syntax error: unexpected end of file`.

**The comment documenting the new `VLLM_EXTRA_ARGS` hook is what broke it, not the hook.**
Line 99 sat inside the single-quoted `srun -ul --environment="$EDF" bash -c '…'` body and
contained two literal `'` characters:

```
# arms, e.g. VLLM_EXTRA_ARGS='--override-generation-config {"temperature":0.0}'
```

The first quote closes the body, the space then word-splits it, and `bash -c` receives only a
truncated fragment. `${VLLM_EXTRA_ARGS}` at line 111 was always correct.

**`bash -n` on the file is a FALSE-PASS gate for this** — the two quotes rebalance, so the
outer script parses fine while the inner body is silently cut in half. The real check is the
argument count:

```bash
# expect ARGC=5 (-ul, --environment=x, bash, -c, body); 6 means the body split
sed -n '41,$p' slurm/cross_system_analysis_clariden.sh > /tmp/b.sh
{ echo 'argc(){ echo "ARGC=$#"; [ $# -eq 5 ] && bash -n <<<"$5" && echo "BODY OK"; }'; echo 'EDF=x'; \
  sed 's|^srun -ul --environment="\$EDF" bash -c|argc -ul --environment=x bash -c|' /tmp/b.sh; } > /tmp/c.sh
bash /tmp/c.sh
```

Verified: broken → `ARGC=6`, fixed → `ARGC=5` + `BODY OK`. The comment is now quote-free and
carries a warning. **No literal single quote may ever appear in that srun body.**

## Import — DONE

```
figures/data/cross_system_sem_filter_llama/        36 json  (33 from 3082227 + 3 PZ from 3083163)
figures/data/cross_system_sem_extract_llama/       33 json  (3083182)
figures/data/cross_system_sem_filter_llama_pz_temp06/  8 json  (3083254)
figures/data/cross_system_sem_filter_llama_pz_greedy/  8 json  (3087979)
figures/data/dp_scaling_json_llama/                36 json  (3083167/71/72/73/74/75/76/77/78
                                                            + 3088915/6/7)
```

All five are staged. `PROVENANCE.md` (filter dir, and a second one in the greedy dir) and
`INPUT_PROVENANCE.md` (scale-out dir) are staged with them. Every `*palimpzest*` artefact
from `3082227` was excluded from the filter dir — see that `PROVENANCE.md` for why and for
the log-count caveat.

The greedy arm went to its **own** directory because its filenames collide with `_pz_temp06`.
It also carries seven `parse_error_*.txt` transcripts that are not from the job dir (they
come from sembench's `parse-answer-errors/`) — see that dir's `PROVENANCE.md`.

## Remaining work

1. Commit: one commit per family (cross-system, scale-out), style `[TAG] Sentence`, no
   trailers, message via `-F -`. Commit the sembench script fix too.
2. `plot_dp_scaling.py` cannot render a Llama figure yet — **two** independent blocks:
   - `collect_dp` selects on `family` **exactly** (`dp-family-selection-refactor`), so
     `llama_unified` / `llama_grid` / `llama_caps` / `llama_caps128` must be added
     explicitly. For the cap sweep specifically, the Qwen analogue is
     `sweep_jobs = ("curve", "caps")`; the Llama analogue must be
     `("llama_caps128", "llama_caps", "llama_grid")` — **not** `llama_unified` as well, or
     cap 512 pools 6 reps against 3 at every other cap. Alternatively add
     `"llama_caps128": "llama_caps"` to `summarize_dp.FAMILY_ALIASES`, which is the
     sanctioned route and collapses the sweep to two families.
   - `collect_dp` also defaults to `model="7B"`, and `_model_short` maps
     `Llama-3.1-8B-Instruct` to **`8B`**, so every Llama row is filtered out before family
     selection even matters. Both must be fixed together.
3. No `dp_scaling_summary.csv` / `dp_balance.csv` has ever been built for
   `dp_scaling_json_llama/` — `summarize_dp.py` writes to `dp_scaling_json/` by default and
   needs `--out-dir analysis/figures/data/dp_scaling_json_llama`. Run it under
   `/usr/bin/python3.11`; the default `python3` on the login node is 3.6 and cannot parse
   the script.

## Do not

- Do not lower `MIN_OP_RATE` (it never tripped), change `--max-model-len 16384`, switch the
  attention backend, or upgrade anything in the sembench env — including
  `huggingface_hub`, pinned at 0.36.2.
- Do not re-run the Qwen arms. They are final and single-session.
- Do not compare absolute DP rows/s across families (regenerated `sf_300000`). Ratios only.
- Do not set `FLOCK_VERDICT_DUMP` on a timed run.
- Do not "fix" the palimpzest runaway by forcing greedy — it makes it 5/8, not 0/8.
- Do not pool the runaway rate across the two decoding modes; report 2/12 and 5/8 separately.

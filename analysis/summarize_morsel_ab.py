#!/usr/bin/env python3
"""Summarise the sem_filter operator-vs-scalar morsel/threads A/B runs into a
CSV (the single source of truth). Curated tables and interpretive findings live
in the write-up, not here, so nothing regenerated can go stale. Sibling of
summarize_cross_system.py.

Takes a list of result directories (each produced by sem_filter_ab_clariden.sh,
default or MORSELS mode). Per directory it parses every
result_{operator,scalar}_*.json plus the matching vLLM /metrics snapshots —
including the flat JSON-era layout, where one dir holds every cell as
`result_<arm>[_m<M>]_t<T>[_rep<N>].json` with metrics at
`metrics_{before,after}_<arm>…_ep0[_rep<N>].txt` (rep AFTER `_ep0`).
Config per cell comes from the dir name `…_<M>morsel_<D>datachunk_<T>threads`
when it has one, else from the JSON's own fields (morsels, threads, rows). The
JSON fields (morsels, row_group_size, batch) always win. The morsel count is the
number of DuckDB row groups (scan morsels); the datachunk count is rows/2048
(vectors). The scalar's concurrency is min(threads, morsels); the operator's is
the in_flight_cap.

Columns (derived metrics):
  rows_s         rows scanned / elapsed (end-to-end filter throughput).
  computed_tok_s (prompt - prefix_cache_hits + gen) tokens / s = tokens the GPU
                 actually processed (prompt_tokens_total counts cache hits).
  tok_s          total (prompt+gen) tokens / s, cache hits included (how hard the
                 system drives vLLM).
  prefill_tok_s  prompt-token throughput; decode_tok_s = gen-token throughput.
                 flock is prefill-dominated (tiny boolean output).
  concurrency    mean in-flight requests (Little's law = req/s x mean e2e
                 latency); scalar ~ min(threads,morsels), operator ~ in_flight_cap.
  tok_per_row    total tokens / row; prompt template cost amortised over R
                 (R=32 ~ 53, R=1 ~ 510).
  gen_per_req    output tokens / request (verbosity; flock ~ 8, Palimpzest ~ 65).
  e2e/ttft/queue/prefill/decode_ms   per-request latency breakdown (means).
  prefix_hit_pct vLLM prefix-cache hit rate (instruction/schema prefix reuse).
  preemptions    vLLM KV-cache preemptions (memory pressure at high concurrency).
  pass_pct       survivors / rows (the A/B driver counts passes, not P/R/F1).

Usage:
  python analysis/summarize_morsel_ab.py DIR1 DIR2 ... [--out-dir DIR]
  # default: every movie_* dir under analysis/figures/data/sem_filter_ab
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

VEC = 2048  # STANDARD_VECTOR_SIZE


def _metric(path: Path, name: str, label: str | None = None) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split("{")[0].split(" ")[0] != name:
            continue
        if label is not None and label not in line:
            continue
        try:
            return float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
    return None


def _delta(d: Path, mtag: str, name: str, label: str | None = None):
    """mtag is the full metrics-file stem, e.g. `operator_t8_ep0` or
    `operator_m8_t8_ep0_rep2` — the rep suffix goes AFTER `_ep0`."""
    a = _metric(d / f"metrics_after_{mtag}.txt", name, label)
    b = _metric(d / f"metrics_before_{mtag}.txt", name, label)
    return (a - b) if (a is not None and b is not None) else None


def _dir_config(name: str) -> dict:
    m = re.search(r"(\d+)morsel", name)
    d = re.search(r"(\d+)datachunk", name)
    t = re.search(r"(\d+)threads", name)
    return {
        "morsels_name": int(m.group(1)) if m else None,
        "datachunks_name": int(d.group(1)) if d else None,
        "threads_name": int(t.group(1)) if t else None,
    }


def parse_dir(d: Path) -> list[dict]:
    cfg = _dir_config(d.name)
    recs = []
    for arm in ("operator", "scalar"):
        # `_*` not `_t*`: the flat JSON-era import names morsel cells
        # `result_operator_m8_t8[_rep2].json`, which a `_t*` glob skips entirely.
        for rj in sorted(d.glob(f"result_{arm}_*.json")):
            tag = rj.name[len("result_"):-len(".json")]   # e.g. operator_m8_t8_rep2
            rm = re.search(r"_rep(\d+)$", tag)
            rep = int(rm.group(1)) if rm else None
            base = tag[:rm.start()] if rm else tag
            mtag = f"{base}_ep0" + (f"_rep{rep}" if rep else "")
            try:
                j = json.loads(rj.read_text())
            except json.JSONDecodeError:
                continue
            el = j.get("elapsed_s")
            rows = j.get("rows")
            rq = _delta(d, mtag, "vllm:request_success_total", 'finished_reason="stop"')
            pt = _delta(d, mtag, "vllm:prompt_tokens_total")
            gt = _delta(d, mtag, "vllm:generation_tokens_total")
            tot = (pt + gt) if (pt is not None and gt is not None) else None

            def mean_ms(base, mtag=mtag):  # per-request mean of a vLLM latency histogram, in ms
                s = _delta(d, mtag, f"vllm:{base}_seconds_sum")
                c = _delta(d, mtag, f"vllm:{base}_seconds_count")
                return (s / c * 1000.0) if (s and c) else None

            e2e = mean_ms("e2e_request_latency")
            conc = ((rq / el) * (e2e / 1000.0)) if (rq and e2e and el) else None
            ch = _delta(d, mtag, "vllm:prefix_cache_hits_total")
            cq = _delta(d, mtag, "vllm:prefix_cache_queries_total")
            preempt = _delta(d, mtag, "vllm:num_preemptions_total")

            def rnd(x, n=1):
                return round(x, n) if x is not None else None

            morsels = j.get("morsels", cfg["morsels_name"])
            # ceil: 2000 rows is one (partial) vector, not zero.
            datachunks = cfg["datachunks_name"] or (-(-rows // VEC) if rows else None)
            threads = j.get("threads")
            # The XML era imported one dir per config; the flat JSON-era import
            # has a single dir, so fall back to the cell's own JSON fields.
            config = (d.name.replace("movie_", "") if cfg["threads_name"] is not None
                      else f"{morsels}morsel_{datachunks}datachunk_{threads}threads")

            recs.append({
                "config": config,
                "arm": arm,
                "rep": rep or 1,
                "morsels": morsels,
                "datachunks": datachunks,
                "threads": threads,
                "R": j.get("batch"),
                "row_group_size": j.get("row_group_size"),
                "rows": rows,
                "elapsed_s": rnd(el),
                # --- throughput / engine efficiency ---
                "rows_s": rnd(j.get("rows_per_s")),                       # rows scanned / s
                "req_s": rnd(rq / el) if (rq and el) else None,
                "concurrency": rnd(conc),                                 # mean in-flight (Little's law)
                "tok_s": rnd(tot / el, 0) if (tot and el) else None,      # TOTAL token throughput (counts cached prefill — inflated)
                # tokens the GPU actually processed: prompt_tokens_total counts
                # prefix-cache hits, so subtract them.
                "computed_tok_s": rnd((pt - ch + gt) / el, 0)
                if (pt is not None and ch is not None and gt is not None and el) else None,
                "prefill_tok_s": rnd(pt / el, 0) if (pt and el) else None,
                "decode_tok_s": rnd(gt / el, 0) if (gt and el) else None,
                # --- token volume / shape ---
                "prompt_tok": int(pt) if pt is not None else None,
                "gen_tok": int(gt) if gt is not None else None,
                "total_tok": int(tot) if tot is not None else None,
                "tok_per_row": rnd(tot / rows) if (tot and rows) else None,   # prompt amortization
                "gen_per_req": rnd(gt / rq) if (gt and rq) else None,         # output verbosity
                # --- latency breakdown (per-request means, ms) ---
                "e2e_ms": rnd(e2e, 0),
                "ttft_ms": rnd(mean_ms("time_to_first_token"), 0),
                "queue_ms": rnd(mean_ms("request_queue_time"), 0),
                "prefill_ms": rnd(mean_ms("request_prefill_time"), 0),
                "decode_ms": rnd(mean_ms("request_decode_time"), 0),
                # --- engine health ---
                "prefix_hit_pct": rnd(100 * ch / cq) if (ch is not None and cq) else None,
                "preemptions": int(preempt) if preempt is not None else None,
                "requests": int(rq) if rq is not None else None,
                # --- quality ---
                "passes": j.get("passes"),
                "pass_pct": rnd(100 * j["passes"] / rows) if (rows and "passes" in j) else None,
            })
    return recs


# Full set (CSV). Derived + raw, ordered: config -> throughput/efficiency ->
# token shape -> latency breakdown -> health -> quality.
COLUMNS = ["config", "arm", "rep", "morsels", "datachunks", "threads", "R",
           "row_group_size", "rows", "elapsed_s",
           "rows_s", "req_s", "concurrency", "tok_s", "computed_tok_s", "prefill_tok_s", "decode_tok_s",
           "prompt_tok", "gen_tok", "total_tok", "tok_per_row", "gen_per_req",
           "e2e_ms", "ttft_ms", "queue_ms", "prefill_ms", "decode_ms",
           "prefix_hit_pct", "preemptions", "requests", "passes", "pass_pct"]
# Curated subset for the console preview (the CSV holds the full column set).
PREVIEW_COLUMNS = ["config", "arm", "rep", "morsels", "threads", "R", "rows_s", "computed_tok_s",
                   "decode_tok_s", "concurrency", "e2e_ms", "tok_per_row", "pass_pct"]


def _fmt(v):
    return "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def write_outputs(recs, out_dir: Path):
    recs = sorted(recs, key=lambda r: ((r["morsels"] or 0), (r["threads"] or 0), r["arm"],
                                       (r.get("rep") or 0)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "morsel_ab_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    # console preview of the curated subset; the CSV is the source of truth.
    print("  ".join(c.rjust(12) for c in PREVIEW_COLUMNS))
    for r in recs:
        print("  ".join(_fmt(r.get(c)).rjust(12) for c in PREVIEW_COLUMNS))
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path,
                    help="result dirs (default: movie_* under analysis/figures/data/sem_filter_ab)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("analysis/figures/data/sem_filter_ab"))
    args = ap.parse_args()
    dirs = args.dirs or [Path(p) for p in sorted(
        glob.glob("analysis/figures/data/sem_filter_ab/movie_*")) if Path(p).is_dir()]
    recs = []
    for d in dirs:
        if not d.exists():
            print(f"  [warn] missing {d}", file=sys.stderr)
            continue
        recs += parse_dir(d)
    if not recs:
        print("no result files found", file=sys.stderr)
        return 1
    write_outputs(recs, args.out_dir)
    print(f"\nwrote {args.out_dir/'morsel_ab_summary.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

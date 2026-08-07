#!/usr/bin/env python3
"""Summarise the DP (data-parallel) scaling runs into CSVs (the single source
of truth). Curated tables and interpretive findings live in the write-up, not
here. Sibling of summarize_tp.py; the column set is a superset of its so the
TP+DP consolidation can merge on the shared columns.

Takes a list of result directories (each produced by dp_scaling_clariden.sh;
one directory = one SLURM job = one rep unless artefacts carry a _rep<N>
suffix, in which case a flat import directory holds several reps). Per
directory it parses every result_{op,scalar}_n<N>_tp<K>_c<CAP>[_rep<R>].json
plus the matching per-endpoint vLLM /metrics snapshots, the per-endpoint
scheduler-gauge samples, the driver /proc CPU samples and the operator verdict
dumps (verdicts_n<N>_tp<K>[_rep<R>].jsonl).

DP-specific derived metrics (vs summarize_tp.py):
  n_ep/gpus       fleet size and total GPUs (n_ep * tp).
  eff             rows/s / same-arm 1x TP1 rows/s (in-job base when the job ran
                  n=1/tp=1, else pooled median across dirs; eff_basis says
                  which). eff_gpu = eff / gpus, the per-GPU efficiency that
                  compares directly against the TP study.
  fleet counters  /metrics deltas are summed across the N endpoints before any
                  rate is derived (computed_tok_s, concurrency via Little's
                  law on fleet aggregates, ...). run_mean/run_max are the SUM
                  of per-endpoint gauge means/maxes (fleet in-flight);
                  kv_mean/kv_max stay per-endpoint (mean of means, max of
                  maxes).
  client_cpu_cores  mean driver CPU cores over the run window from the
                  /proc/<pid>/stat sampler ((d_utime+d_stime)/d_t/HZ): the
                  client-vs-GPU bottleneck discriminator.
  dp_balance.csv  one row per (job, config, endpoint): request/token shares
                  and gauge means -- the load-balance panel input. The main
                  CSV carries req_share_spread (max-min request share, pct
                  points) as the one-glance evenness scalar.

Usage:
  python analysis/summarize_dp.py DIR1 DIR2 ... [--out-dir DIR] [--data CSV]
  # default: analysis/figures/data/dp_scaling_json itself (flat import) plus every
  # subdirectory of it
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

TAG_RE = re.compile(r"result_(op|scalar)_n(\d+)_tp(\d+)_c(\d+)((?:_.+)?)\.json")
HZ = 100  # kernel tick rate for /proc/<pid>/stat utime/stime

# `REP` is free-form at submit time, so the same experiment can carry different
# labels in different eras: the XML-era DP curve ran as `curve_rep<N>`, the JSON
# re-run as `confirm_rep<N>`. `family` is the canonical name the plots select
# on; `job` keeps the label the run actually recorded, so any number stays
# traceable back to the job that produced it. Add an alias here rather than
# renaming committed artefacts or teaching a plot a second spelling.
FAMILY_ALIASES = {"confirm": "curve"}


def canon_family(job: str) -> str:
    """Job label -> canonical family: drop the `_rep<N>` suffix, then alias.
    Labels without a rep suffix (`mn_verd`, `mn_harness`) pass through."""
    fam = re.sub(r"_rep\d+$", "", job)
    return FAMILY_ALIASES.get(fam, fam)


def dir_encodings(d: Path) -> set[str]:
    """Encodings present among a dir's result files. Post-fix drivers write
    `tuple_format`; XML-era results predate the field entirely, so a missing
    field means XML (verified: 0 of 89 committed dp_scaling results carry it)."""
    out = set()
    for rj in d.glob("result_*.json"):
        try:
            out.add(json.loads(rj.read_text()).get("tuple_format") or "XML")
        except json.JSONDecodeError:
            continue
    return out


def _metric(path: Path, name: str, label: str | None = None) -> float | None:
    """Sum over the metric's label series (a counter may split by label, e.g.
    request_success_total by finished_reason); label narrows to one series."""
    if not path.exists():
        return None
    total, seen = 0.0, False
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split("{")[0].split(" ")[0] != name:
            continue
        if label is not None and label not in line:
            continue
        try:
            total += float(line.rsplit(" ", 1)[1])
            seen = True
        except (ValueError, IndexError):
            continue
    return total if seen else None


def _delta(d: Path, mid: str, ep: int, name: str, label: str | None = None):
    a = _metric(d / f"metrics_after_{mid}_ep{ep}.txt", name, label)
    b = _metric(d / f"metrics_before_{mid}_ep{ep}.txt", name, label)
    return (a - b) if (a is not None and b is not None) else None


def _fleet_delta(d: Path, mid: str, eps: list[int], name: str,
                 label: str | None = None) -> float | None:
    """Sum a counter delta across the fleet; None only if EVERY endpoint lacks it."""
    vals = [_delta(d, mid, ep, name, label) for ep in eps]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def _gauge_stats(f: Path) -> dict:
    out = {"run_mean": None, "run_max": None, "wait_mean": None,
           "wait_max": None, "kv_mean": None, "kv_max": None}
    if not f.exists():
        return out
    cols: dict[str, list[float]] = {"running": [], "waiting": [], "kv_usage": []}
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            for k in cols:
                try:
                    cols[k].append(float(row[k]))
                except (KeyError, TypeError, ValueError):
                    pass
    for key, pre, nd in (("running", "run", 1), ("waiting", "wait", 1),
                         ("kv_usage", "kv", 4)):
        v = cols[key]
        if v:
            # kv_usage is a 0..1 fraction; 1 decimal would floor it to 0.0.
            out[f"{pre}_mean"] = round(sum(v) / len(v), nd)
            out[f"{pre}_max"] = round(max(v), nd)
    return out


def _meta(f: Path) -> dict:
    """Sidecar written by run() in the driver: which dataset/model this cell
    measured. The result JSON records neither, and jobs mix them (scale-out
    cells on sf_300000 at 32k rows, the scalar reference on sf_2000 gold rows,
    the scalar-on-fleet job on sf_300000 at 2k rows)."""
    out = {"dataset": None, "model": None}
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        k, _, v = line.partition("=")
        # dataset comes from the data path's parent dir (the sidecar's own
        # dataset= field was mis-derived in early wave-1 runs).
        if k == "data" and v:
            out["dataset"] = Path(v).parent.name
        elif k == "model" and v:
            out["model"] = v
    return out


def _client_cpu(f: Path) -> dict:
    """Mean driver CPU cores + peak thread count from the /proc sampler."""
    out = {"client_cpu_cores": None, "client_threads_max": None}
    if not f.exists():
        return out
    rows = []
    with open(f, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append((int(row["epoch"]),
                             int(row["utime_ticks"]) + int(row["stime_ticks"]),
                             int(row["nthreads"])))
            except (KeyError, TypeError, ValueError):
                pass
    if len(rows) >= 2:
        dt = rows[-1][0] - rows[0][0]
        dticks = rows[-1][1] - rows[0][1]
        if dt > 0:
            out["client_cpu_cores"] = round(dticks / dt / HZ, 2)
    if rows:
        out["client_threads_max"] = max(r[2] for r in rows)
    return out


# ---- operator verdicts -> P/R/F1 vs gold (summarize_tp.py conventions) ------
def load_gold(csv_path: Path, gold_col: str, gold_pos: str) -> dict[int, bool]:
    out: dict[int, bool] = {}
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            out[i] = (row.get(gold_col) or "").strip().upper() == gold_pos.upper()
    return out


def verdict_prf1(path: Path, gold: dict[int, bool]) -> dict:
    tp = fp = fn = 0
    n = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            g = gold.get(int(d["id"]))
            if g is None:
                continue
            n += 1
            passed = d.get("v") is True or d.get("v") is None  # null->pass parity
            if g:
                tp += passed
                fn += not passed
            elif passed:
                fp += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "verdict_rows": n}


def parse_dir(d: Path, gold: dict[int, bool] | None):
    recs, balance = [], []
    for rj in sorted(d.glob("result_*_n*_tp*_c*.json")):
        m = TAG_RE.fullmatch(rj.name)
        if not m:
            continue
        arm, n_ep, tp_k, cap, suf = (m.group(1), int(m.group(2)), int(m.group(3)),
                                     int(m.group(4)), m.group(5))
        # suf is the free rep label ("_rep1", "_grid_rep2", "" ...): the job
        # grouping unit for the efficiency base; sibling files share it.
        job = suf.lstrip("_") if suf else d.name
        family = canon_family(job)
        stem = rj.name[len("result_"):-len(".json")]  # middle of sibling names
        try:
            j = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        eps = sorted(int(re.search(r"_ep(\d+)\.txt$", p.name).group(1))
                     for p in d.glob(f"metrics_after_{stem}_ep*.txt"))
        if not eps:
            eps = list(range(n_ep))
        el = j.get("elapsed_s")
        rows = j.get("rows")

        rq = _fleet_delta(d, stem, eps, "vllm:request_success_total")
        pt = _fleet_delta(d, stem, eps, "vllm:prompt_tokens_total")
        gt = _fleet_delta(d, stem, eps, "vllm:generation_tokens_total")
        ch = _fleet_delta(d, stem, eps, "vllm:prefix_cache_hits_total")
        cq = _fleet_delta(d, stem, eps, "vllm:prefix_cache_queries_total")
        tot = (pt + gt) if (pt is not None and gt is not None) else None

        def mean_ms(base):  # fleet per-request mean of a latency histogram, ms
            s = _fleet_delta(d, stem, eps, f"vllm:{base}_seconds_sum")
            c = _fleet_delta(d, stem, eps, f"vllm:{base}_seconds_count")
            return (s / c * 1000.0) if (s and c) else None

        e2e = mean_ms("e2e_request_latency")
        conc = ((rq / el) * (e2e / 1000.0)) if (rq and e2e and el) else None
        preempt = _fleet_delta(d, stem, eps, "vllm:num_preemptions_total")

        # per-endpoint gauge stats + fleet aggregates
        per_ep_gauges = {ep: _gauge_stats(d / f"samples_{stem}_ep{ep}.csv")
                         for ep in eps}
        def gsum(key):
            v = [per_ep_gauges[ep][key] for ep in eps
                 if per_ep_gauges[ep][key] is not None]
            return round(sum(v), 1) if v else None
        kv_means = [per_ep_gauges[ep]["kv_mean"] for ep in eps
                    if per_ep_gauges[ep]["kv_mean"] is not None]
        kv_maxes = [per_ep_gauges[ep]["kv_max"] for ep in eps
                    if per_ep_gauges[ep]["kv_max"] is not None]

        # per-endpoint balance rows + request-share spread
        meta = _meta(d / f"meta_{stem}.txt")
        ep_reqs = {}
        for ep in eps:
            erq = _delta(d, stem, ep, "vllm:request_success_total")
            ept = _delta(d, stem, ep, "vllm:prompt_tokens_total")
            egt = _delta(d, stem, ep, "vllm:generation_tokens_total")
            es = _delta(d, stem, ep, "vllm:e2e_request_latency_seconds_sum")
            ec = _delta(d, stem, ep, "vllm:e2e_request_latency_seconds_count")
            ep_reqs[ep] = erq
            g = per_ep_gauges[ep]
            balance.append({
                "job": job, "family": family, "model": meta["model"],
                "arm": arm, "n_ep": n_ep, "tp": tp_k, "cap": cap, "ep": ep,
                "requests": int(erq) if erq is not None else None,
                "req_share_pct": None,  # filled below once the total is known
                "prompt_tok": int(ept) if ept is not None else None,
                "gen_tok": int(egt) if egt is not None else None,
                "run_mean": g["run_mean"], "wait_mean": g["wait_mean"],
                "kv_mean": g["kv_mean"],
                "e2e_ms": round(es / ec * 1000.0) if (es and ec) else None,
            })
        shares = None
        tot_req = sum(v for v in ep_reqs.values() if v is not None)
        if tot_req:
            shares = {ep: 100.0 * v / tot_req for ep, v in ep_reqs.items()
                      if v is not None}
            for b in balance[-len(eps):]:
                if b["requests"] is not None:
                    b["req_share_pct"] = round(shares[b["ep"]], 2)

        def rnd(x, n=1):
            return round(x, n) if x is not None else None

        rec = {
            "job": job,
            "family": family,
            "arm": arm,
            "n_ep": n_ep,
            "tp": tp_k,
            "gpus": n_ep * tp_k,
            "cap": cap,
            "R": j.get("batch"),
            "threads": j.get("threads"),
            "rows": rows,
            **meta,
            "elapsed_s": rnd(el),
            # --- throughput / scaling ---
            "rows_s": rnd(j.get("rows_per_s")),
            "eff": None, "eff_basis": None, "eff_gpu": None,  # add_efficiency
            "req_s": rnd(rq / el) if (rq and el) else None,
            "concurrency": rnd(conc),
            "tok_s": rnd(tot / el, 0) if (tot and el) else None,
            "computed_tok_s": rnd((pt - ch + gt) / el, 0)
            if (pt is not None and ch is not None and gt is not None and el) else None,
            "prefill_tok_s": rnd(pt / el, 0) if (pt and el) else None,
            "decode_tok_s": rnd(gt / el, 0) if (gt and el) else None,
            # --- token volume / shape ---
            "prompt_tok": int(pt) if pt is not None else None,
            "gen_tok": int(gt) if gt is not None else None,
            "tok_per_row": rnd(tot / rows) if (tot and rows) else None,
            "gen_per_req": rnd(gt / rq) if (gt and rq) else None,
            # --- latency breakdown (fleet per-request means, ms) ---
            "e2e_ms": rnd(e2e, 0),
            "ttft_ms": rnd(mean_ms("time_to_first_token"), 0),
            "queue_ms": rnd(mean_ms("request_queue_time"), 0),
            "prefill_ms": rnd(mean_ms("request_prefill_time"), 0),
            "decode_ms": rnd(mean_ms("request_decode_time"), 0),
            # --- engine health / saturation (fleet) ---
            "prefix_hit_pct": rnd(100 * ch / cq) if (ch is not None and cq) else None,
            "preemptions": int(preempt) if preempt is not None else None,
            "requests": int(rq) if rq is not None else None,
            "run_mean": gsum("run_mean"), "run_max": gsum("run_max"),
            "wait_mean": gsum("wait_mean"), "wait_max": gsum("wait_max"),
            "kv_mean": rnd(sum(kv_means) / len(kv_means), 3) if kv_means else None,
            "kv_max": rnd(max(kv_maxes), 3) if kv_maxes else None,
            # --- dispatch evenness / client side ---
            "req_share_spread": rnd(max(shares.values()) - min(shares.values()), 2)
            if shares and len(shares) > 1 else None,
            **_client_cpu(d / f"samples_{stem}_client.csv"),
            # --- quality ---
            "passes": j.get("passes"),
            "pass_pct": rnd(100 * j["passes"] / rows) if (rows and "passes" in j) else None,
            "precision": None, "recall": None, "f1": None, "verdict_rows": None,
        }
        vd = d / f"verdicts_n{n_ep}_tp{tp_k}{suf}.jsonl"
        if arm == "op" and gold is not None and vd.exists():
            rec.update(verdict_prf1(vd, gold))
        recs.append(rec)
    return recs, balance


def add_efficiency(recs: list[dict]) -> None:
    """eff = rows_s / same-arm SAME-MODEL 1xTP1 rows_s (in-job base, else
    pooled median); eff_gpu = eff / gpus. Grid jobs carry no 1xTP1 point, so
    they normalise against the pooled curve base (eff_basis=pooled_1x1).
    Models without any 1xTP1 cell (32B/72B fixed-budget grids -- and 72B has
    no legal 1xTP1 at all) keep eff empty rather than borrowing another
    model's base. Legacy runs without a meta sidecar have model=None, which
    only ever matches other pre-sidecar (7B) runs."""
    def median(v):
        v = sorted(v)
        n = len(v)
        return (v[n // 2] + v[(n - 1) // 2]) / 2 if v else None

    pooled: dict = {}
    in_job: dict = {}
    for r in recs:
        if r["n_ep"] == 1 and r["tp"] == 1 and r["rows_s"]:
            pooled.setdefault((r["arm"], r["model"]), []).append(r["rows_s"])
            in_job.setdefault((r["job"], r["arm"], r["model"]), r["rows_s"])
    pooled = {k: median(v) for k, v in pooled.items()}
    for r in recs:
        if not r["rows_s"]:
            continue
        base = in_job.get((r["job"], r["arm"], r["model"]))
        basis = "in_job"
        if base is None:
            base, basis = pooled.get((r["arm"], r["model"])), "pooled_1x1"
        if base:
            r["eff"] = round(r["rows_s"] / base, 2)
            r["eff_basis"] = basis
            r["eff_gpu"] = round(r["rows_s"] / base / r["gpus"], 2)


COLUMNS = ["job", "family", "arm", "n_ep", "tp", "gpus", "cap", "R", "threads", "rows",
           "dataset", "model", "elapsed_s", "rows_s", "eff", "eff_basis", "eff_gpu", "req_s",
           "concurrency", "tok_s", "computed_tok_s", "prefill_tok_s",
           "decode_tok_s", "prompt_tok", "gen_tok", "tok_per_row", "gen_per_req",
           "e2e_ms", "ttft_ms", "queue_ms", "prefill_ms", "decode_ms",
           "prefix_hit_pct", "preemptions", "requests",
           "run_mean", "run_max", "wait_mean", "wait_max", "kv_mean", "kv_max",
           "req_share_spread", "client_cpu_cores", "client_threads_max",
           "passes", "pass_pct", "precision", "recall", "f1", "verdict_rows"]
BALANCE_COLUMNS = ["job", "family", "model", "arm", "n_ep", "tp", "cap", "ep", "requests",
                   "req_share_pct", "prompt_tok", "gen_tok",
                   "run_mean", "wait_mean", "kv_mean", "e2e_ms"]
PREVIEW_COLUMNS = ["job", "arm", "n_ep", "tp", "cap", "rows_s", "eff", "eff_gpu",
                   "computed_tok_s", "run_mean", "wait_mean", "e2e_ms",
                   "req_share_spread", "client_cpu_cores", "f1"]


def _fmt(v):
    return "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def write_outputs(recs, balance, out_dir: Path):
    key = lambda r: (r["arm"], r["gpus"], r["n_ep"], r["cap"], r["job"])
    recs = sorted(recs, key=key)
    balance = sorted(balance, key=lambda b: (b["arm"], b["gpus"] if "gpus" in b
                     else b["n_ep"] * b["tp"], b["n_ep"], b["cap"], b["job"], b["ep"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dp_scaling_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    with open(out_dir / "dp_balance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BALANCE_COLUMNS)
        w.writeheader()
        for b in balance:
            w.writerow({k: b.get(k, "") for k in BALANCE_COLUMNS})
    print("  ".join(c.rjust(12) for c in PREVIEW_COLUMNS))
    for r in recs:
        print("  ".join(_fmt(r.get(c)).rjust(12) for c in PREVIEW_COLUMNS))
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", type=Path,
                    help="result dirs (default: analysis/figures/data/dp_scaling_json"
                         " itself plus its subdirectories)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("analysis/figures/data/dp_scaling_json"))
    ap.add_argument("--data", type=Path,
                    default=Path("../sembench/files/movie/data/sf_2000/Reviews.csv"),
                    help="gold CSV for the operator F1 columns (skipped if missing)")
    ap.add_argument("--gold-col", default="scoreSentiment")
    ap.add_argument("--gold-positive", default="POSITIVE")
    args = ap.parse_args()

    gold = None
    if args.data.exists():
        gold = load_gold(args.data, args.gold_col, args.gold_positive)
    else:
        print(f"  [warn] gold csv missing ({args.data}) -> F1 columns skipped",
              file=sys.stderr)

    data_root = Path("analysis/figures/data/dp_scaling_json")
    dirs = args.dirs
    if not dirs and data_root.exists():
        dirs = [data_root] + [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    # One summary = one encoding. A CSV spanning both eras would let a median
    # pool XML and JSON cells for the same config, which is silent and
    # unrecoverable downstream -- so refuse to write one at all.
    seen = {}
    for d in dirs:
        if d.exists():
            for enc in dir_encodings(d):
                seen.setdefault(enc, []).append(d.name)
    if len(seen) > 1:
        print("error: inputs span more than one encoding -- refusing to mix.",
              file=sys.stderr)
        for enc, ds in sorted(seen.items()):
            print(f"    {enc:5s} <- {', '.join(sorted(set(ds)))}", file=sys.stderr)
        if "XML" in seen:
            print("    note: 'XML' also means the result JSON has no"
                  " tuple_format field at all. If these are new runs, the"
                  " driver may simply not be writing it -- check before"
                  " assuming an encoding split.", file=sys.stderr)
        print("  Summarise each encoding separately (one --out-dir each).",
              file=sys.stderr)
        return 2

    recs, balance = [], []
    for d in dirs:
        if not d.exists():
            print(f"  [warn] missing {d}", file=sys.stderr)
            continue
        r, b = parse_dir(d, gold)
        recs += r
        balance += b
    if not recs:
        print("no result files found", file=sys.stderr)
        return 1
    add_efficiency(recs)
    for b in balance:
        b["gpus"] = b["n_ep"] * b["tp"]
    write_outputs(recs, balance, args.out_dir)
    print(f"\nwrote {args.out_dir/'dp_scaling_summary.csv'} and dp_balance.csv",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Diagnose the sem_filter R=1 -> R=32 recall collapse from an R-sweep produced
by analysis/slurm/sem_filter_rsweep_clariden.sh.

For each R it joins the per-row verdict dump (verdicts_r<R>[_rep<N>].jsonl:
{id,pos,v}; one rep per R, selected with --rep -- see main() for why reps are
never pooled) (id == query-global row_id == source-CSV row index at threads=1) to the source
Reviews CSV (gold scoreSentiment + review length) and reports:

  1. QUALITY curve      precision / recall / F1 / survivors per R (+ throughput
                        from result_r<R>.json and tok/s from the /metrics deltas).
                        -> the F1(R) knee: the largest R that still holds quality.
  2. POSITION test (C)  miss-rate of gold-positive rows by normalised in-batch
                        position. Misses concentrated mid-batch => lost-in-the-
                        middle.
  3. FLIP/SIGNAL (C/D)  among positives that R=1 gets right, the ones dropped at
                        higher R, characterised by position AND review length.
                        Position-concentrated => C; length-correlated and
                        position-uniform => conservative-default (D).

Verdict "v": true or null => pass (null->pass parity with the operator),
false => drop. Read-only.

Usage:
  python analysis/diagnose_rsweep.py <results-dir> --data <Reviews.csv> \
      [--rows 2000] [--gold-col scoreSentiment] [--gold-positive POSITIVE] \
      [--text-col reviewText] [--out-dir <results-dir>]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

N_POS_BUCKETS = 5


# --- source-CSV gold + review length, keyed by scan-order row_id -------------
def load_gold(csv_path: Path, rows: int, gold_col: str, gold_pos: str, text_col: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= rows:
                break
            text = row.get(text_col) or ""
            out[i] = {
                "gold": (row.get(gold_col) or "").strip().upper() == gold_pos.strip().upper(),
                "len": len(text),
            }
    return out


# --- one R's verdict dump: id -> (pos, pass) ---------------------------------
def load_verdicts(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = d.get("v")
            out[int(d["id"])] = {"pos": int(d["pos"]), "pass": (v is True or v is None)}
    return out


# --- vLLM /metrics helpers (optional throughput/efficiency context) ----------
def _metric(path: Path, name: str) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.split("{")[0].split(" ")[0] != name:
            continue
        try:
            return float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
    return None


def _delta(d: Path, tag: str, name: str, sfx: str = "") -> float | None:
    a = _metric(d / f"metrics_after_{tag}_ep0{sfx}.txt", name)
    b = _metric(d / f"metrics_before_{tag}_ep0{sfx}.txt", name)
    return (a - b) if (a is not None and b is not None) else None


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def analyse_R(rdir: Path, R: int, gold: dict[int, dict], sfx: str = "") -> dict:
    verdicts = load_verdicts(rdir / f"verdicts_r{R}{sfx}.jsonl")
    tp = fp = fn = 0
    # miss-rate of gold-positives by normalised in-batch position
    pos_tot = [0] * N_POS_BUCKETS
    pos_miss = [0] * N_POS_BUCKETS
    for rid, v in verdicts.items():
        g = gold.get(rid)
        if g is None:
            continue
        passed = v["pass"]
        if g["gold"]:
            if passed:
                tp += 1
            else:
                fn += 1
            npos = (v["pos"] / (R - 1)) if R > 1 else 0.0
            b = min(int(npos * N_POS_BUCKETS), N_POS_BUCKETS - 1)
            pos_tot[b] += 1
            if not passed:
                pos_miss[b] += 1
        elif passed:
            fp += 1
    p, r, f = prf1(tp, fp, fn)

    # throughput / engine efficiency from the (separate, untimed-free) timing pass
    rj = rdir / f"result_r{R}{sfx}.json"
    rows_s = elapsed = None
    if rj.exists():
        try:
            j = json.loads(rj.read_text())
            rows_s = j.get("rows_per_s")
            elapsed = j.get("elapsed_s")
        except json.JSONDecodeError:
            pass
    pt = _delta(rdir, f"r{R}", "vllm:prompt_tokens_total", sfx)
    gt = _delta(rdir, f"r{R}", "vllm:generation_tokens_total", sfx)
    ch = _delta(rdir, f"r{R}", "vllm:prefix_cache_hits_total", sfx)
    tok_s = round((pt + gt) / elapsed) if (pt is not None and gt is not None and elapsed) else None
    # prompt_tokens_total counts prefix-cache hits; subtract them for the
    # GPU-processed token throughput.
    computed_tok_s = round((pt - ch + gt) / elapsed) \
        if (pt is not None and ch is not None and gt is not None and elapsed) else None

    return {
        "R": R, "survivors": tp + fp, "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
        "rows_s": rows_s, "tok_s": tok_s, "computed_tok_s": computed_tok_s,
        "pos_tot": pos_tot, "pos_miss": pos_miss,
        "verdicts": verdicts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--data", type=Path, required=True, help="source Reviews.csv (gold + length)")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--gold-col", default="scoreSentiment")
    ap.add_argument("--gold-positive", default="POSITIVE")
    ap.add_argument("--text-col", default="reviewText")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--rep", type=int, default=None,
                    help="which rep to analyse when dumps carry _rep<N> "
                         "(default: lowest present). Reps are never pooled.")
    args = ap.parse_args()
    if not args.results_dir.exists():
        print(f"results dir not found: {args.results_dir}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"data csv not found: {args.data}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold(args.data, args.rows, args.gold_col, args.gold_positive, args.text_col)
    n_pos = sum(1 for g in gold.values() if g["gold"])
    # Dumps come in two layouts: `verdicts_r<R>.jsonl` (single-rep jobs) and
    # `verdicts_r<R>_rep<N>.jsonl` (the 3-rep prompt sweeps). Discover both and
    # analyse ONE rep per R -- reps are not pooled, because the same row_id
    # recurs across reps at a different in-batch position, so pooling would
    # average the very positional signal this script exists to measure.
    found: dict[int, dict[int, str]] = {}   # R -> {rep: suffix}; rep 0 = no suffix
    for p in args.results_dir.glob("verdicts_r*.jsonl"):
        m = re.fullmatch(r"verdicts_r(\d+)(?:_rep(\d+))?\.jsonl", p.name)
        if m:
            rep = int(m.group(2)) if m.group(2) else 0
            found.setdefault(int(m.group(1)), {})[rep] = f"_rep{rep}" if m.group(2) else ""
    if not found:
        print(f"no verdicts_r<R>[_rep<N>].jsonl in {args.results_dir}", file=sys.stderr)
        return 1
    chosen: dict[int, str] = {}
    for R, reps in sorted(found.items()):
        if args.rep is not None:
            if args.rep not in reps:
                print(f"R={R}: no rep {args.rep} (have {sorted(reps)}) -- skipped",
                      file=sys.stderr)
                continue
            chosen[R] = reps[args.rep]
        else:
            chosen[R] = reps[min(reps)]
    if not chosen:
        print(f"no R survived rep selection in {args.results_dir}", file=sys.stderr)
        return 1
    Rs = sorted(chosen)
    for R in Rs:
        print(f"  R={R:<3} <- verdicts_r{R}{chosen[R]}.jsonl", file=sys.stderr)
    res = {R: analyse_R(args.results_dir, R, gold, chosen[R]) for R in Rs}

    # ---- 1. QUALITY curve --------------------------------------------------
    qcols = ["R", "survivors", "tp", "fp", "fn", "precision", "recall", "f1", "rows_s", "tok_s", "computed_tok_s"]
    with open(out_dir / "rsweep_quality.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=qcols)
        w.writeheader()
        for R in Rs:
            w.writerow({k: res[R].get(k, "") for k in qcols})
    print(f"\ngold positives in first {args.rows} rows: {n_pos}\n")
    print("== 1. QUALITY curve (F1(R) knee) ==")
    hdr = (f"{'R':>4} {'surv':>6} {'prec':>6} {'recall':>7} {'F1':>6} {'rows/s':>8} "
           f"{'tok/s':>8} {'ctok/s':>8}")
    print(hdr)
    for R in Rs:
        x = res[R]
        print(f"{R:>4} {x['survivors']:>6} {x['precision']:>6} {x['recall']:>7} {x['f1']:>6} "
              f"{(x['rows_s'] or ''):>8} {(x['tok_s'] or ''):>8} {(x['computed_tok_s'] or ''):>8}")

    # ---- 2. POSITION test (mechanism C) ------------------------------------
    pcols = ["R"] + [f"missrate_b{b}" for b in range(N_POS_BUCKETS)] + ["mid_over_outer"]
    prows = []
    print("\n== 2. POSITION test: miss-rate of gold-positives by normalised in-batch position ==")
    print(f"(buckets 0..{N_POS_BUCKETS-1} span batch start..end; mid_over_outer>1 => lost-in-the-middle)")
    print(f"{'R':>4}  " + "  ".join(f"b{b}" for b in range(N_POS_BUCKETS)) + "   mid/outer")
    for R in Rs:
        if R == 1:
            continue
        x = res[R]
        rates = [(x["pos_miss"][b] / x["pos_tot"][b]) if x["pos_tot"][b] else 0.0 for b in range(N_POS_BUCKETS)]
        outer = [rates[0], rates[-1]]
        mid = rates[1:-1] or [0.0]
        outer_avg = sum(outer) / len(outer) if outer else 0.0
        mid_avg = sum(mid) / len(mid)
        ratio = (mid_avg / outer_avg) if outer_avg else float("nan")
        prows.append({"R": R, **{f"missrate_b{b}": round(rates[b], 3) for b in range(N_POS_BUCKETS)},
                      "mid_over_outer": round(ratio, 2)})
        print(f"{R:>4}  " + "  ".join(f"{rates[b]:.2f}" for b in range(N_POS_BUCKETS)) + f"    {ratio:.2f}")
    with open(out_dir / "rsweep_position.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pcols)
        w.writeheader()
        w.writerows(prows)

    # ---- 3. FLIP / SIGNAL test (C vs D) ------------------------------------
    # baseline: positives that R=1 gets right (the recall we later lose).
    base = res.get(1)
    print("\n== 3. FLIP test: positives correct at R=1, dropped at higher R ==")
    if base is None:
        print("  (no R=1 run -> skipped; add R=1 to the sweep for the baseline)")
    else:
        p1 = {rid for rid, v in base["verdicts"].items() if gold.get(rid, {}).get("gold") and v["pass"]}
        print(f"baseline |correct positives @R=1| = {len(p1)}")
        print(f"{'R':>4} {'dropped':>8} {'droprate':>9} {'mean_npos_drop':>15} {'len_drop':>9} {'len_kept':>9}")
        for R in Rs:
            if R == 1:
                continue
            v = res[R]["verdicts"]
            dropped = [rid for rid in p1 if rid in v and not v[rid]["pass"]]
            kept = [rid for rid in p1 if rid in v and v[rid]["pass"]]
            drop_rate = len(dropped) / len(p1) if p1 else 0.0
            npos_drop = [v[rid]["pos"] / (R - 1) for rid in dropped] if R > 1 else []
            mean_npos = (sum(npos_drop) / len(npos_drop)) if npos_drop else float("nan")
            len_drop = (sum(gold[rid]["len"] for rid in dropped) / len(dropped)) if dropped else float("nan")
            len_kept = (sum(gold[rid]["len"] for rid in kept) / len(kept)) if kept else float("nan")
            print(f"{R:>4} {len(dropped):>8} {drop_rate:>9.3f} {mean_npos:>15.3f} {len_drop:>9.0f} {len_kept:>9.0f}")

    # ---- verdict (largest R) -----------------------------------------------
    Rmax = max(Rs)
    x = res[Rmax]
    rates = [(x["pos_miss"][b] / x["pos_tot"][b]) if x["pos_tot"][b] else 0.0 for b in range(N_POS_BUCKETS)]
    outer_avg = (rates[0] + rates[-1]) / 2 if N_POS_BUCKETS >= 2 else rates[0]
    mid_avg = sum(rates[1:-1]) / max(1, len(rates[1:-1]))
    if outer_avg > 0:
        pos_ratio = mid_avg / outer_avg
        pos_str = f"mid/outer miss-rate ratio = {pos_ratio:.2f}"
        pos_c = pos_ratio > 1.3
    else:  # no misses at the batch edges: any middle misses => concentrated there
        pos_ratio = float("inf") if mid_avg > 0 else 0.0
        pos_str = f"edge miss-rate = 0, middle miss-rate = {mid_avg:.2f}"
        pos_c = mid_avg > 0
    print(f"\n== VERDICT (at R={Rmax}) ==")
    print(f"  recall {res[1]['recall'] if 1 in res else '?'} (R=1) -> {x['recall']} (R={Rmax}); precision "
          f"{res[1]['precision'] if 1 in res else '?'} -> {x['precision']}")
    print(f"  {pos_str}  "
          f"({'position-concentrated -> lost-in-the-middle (C)' if pos_c else 'roughly position-uniform -> not primarily C'})")
    if 1 in res:
        v = x["verdicts"]
        p1 = {rid for rid, vv in res[1]["verdicts"].items() if gold.get(rid, {}).get("gold") and vv["pass"]}
        dropped = [rid for rid in p1 if rid in v and not v[rid]["pass"]]
        kept = [rid for rid in p1 if rid in v and v[rid]["pass"]]
        if dropped and kept:
            ld = sum(gold[rid]["len"] for rid in dropped) / len(dropped)
            lk = sum(gold[rid]["len"] for rid in kept) / len(kept)
            rel = (ld - lk) / lk if lk else 0.0
            print(f"  dropped positives are {abs(rel)*100:.0f}% {'shorter' if rel < 0 else 'longer'} than kept "
                  f"(len {ld:.0f} vs {lk:.0f}) -> length/signal effect ({'consistent with' if abs(rel) > 0.1 else 'weak for'} conservative-default D)")
    print(f"\nwrote {out_dir/'rsweep_quality.csv'}, {out_dir/'rsweep_position.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

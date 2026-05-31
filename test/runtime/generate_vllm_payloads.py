#!/usr/bin/env python3
"""Dump N vLLM completions payloads to a JSONL file, matching the workload
shape sembench's vLLM performance analysis (E1 cell) sends.

Reuses sembench's own builders (build_prompt + make_body) so the wire-level
bytes are bit-identical to what scripts/vllm_perf_driver.py would send for
the N=128, R=32 cell. Each payload gets a different `unique_prefix_words`
prefix, so vLLM's prefix cache stays cold (prefix_cache_hit_rate ~ 0).

The output JSONL is consumed by flock_async_vllm_integration --payload-file,
which round-robins through the payloads.

Usage:
  python generate_vllm_payloads.py --count 1000 --rows-per-request 32 \\
      --output-tokens 64 --out /tmp/vllm_payloads.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

# Import the driver as a module so we use its exact builders.
# [POTENTIAL TODO] Do not rely on sys.path hack.
SEMBENCH_ROOT = Path("/local/home/dmicloiu/sembench")
sys.path.insert(0, str(SEMBENCH_ROOT / "scripts"))
import vllm_perf_driver as driver  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--rows-per-request", type=int, default=32)
    p.add_argument("--row-padding-words", type=int, default=0)
    p.add_argument("--unique-prefix-words", type=int, default=16)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--endpoint", choices=("completions", "chat"),
                   default="completions")
    p.add_argument("--response-schema",
                   choices=("none", "boolean_array"),
                   default="boolean_array")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # make_body needs a few extra fields it reads off args.
    body_args = SimpleNamespace(
        model=args.model,
        endpoint=args.endpoint,
        output_tokens=args.output_tokens,
        natural_output=False,
        response_schema=args.response_schema,
        rows_per_request=args.rows_per_request,
    )

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    with out_path.open("w") as f:
        for i in range(args.count):
            prompt = driver.build_prompt(
                rows=args.rows_per_request,
                row_padding_words=args.row_padding_words,
                unique_prefix_words=args.unique_prefix_words,
                rng=rng,
                base_id=i * args.rows_per_request,
            )
            body = driver.make_body(body_args, prompt)
            f.write(json.dumps(body, separators=(",", ":")))
            f.write("\n")

    print(f"wrote {args.count} payloads to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Dump N vLLM completions payloads to a JSONL file, matching the workload
shape sembench's vLLM performance analysis (E1 cell) sends.

Reuses sembench's own builders (build_prompt + make_body) so the wire-level
bytes are bit-identical to what scripts/vllm_perf_driver.py would send for
the N=128, R=32 cell. By default each payload gets a different
`unique_prefix_words` prefix, so vLLM's prefix cache stays cold
(prefix_cache_hit_rate ~ 0).

The output JSONL is consumed by the integration binaries --payload-file,
which round-robin through the payloads.

Two workload variants exist for the EndpointRouter experiment:

  --prefix-groups K   Assign each payload to one of K groups and give every
                      payload in a group the SAME long head prefix (cacheable).
                      Writes a parallel keys file (--keys-out) of "g<n>" tags.
                      sticky_by_prefix pins each group to one endpoint -> that
                      endpoint's prefix cache stays warm; round_robin scatters
                      each group across all endpoints -> cache thrash. This is
                      the signal that makes sticky's win visible.

  --skew-frac F       Make a fraction F of payloads request --skew-output-tokens
                      output tokens instead of --output-tokens, creating
                      per-request cost skew. least_loaded should ride out the
                      resulting load imbalance better than round_robin (tail
                      latency). The routing key is irrelevant for this variant.

Usage:
  python generate_vllm_payloads.py --count 1000 --rows-per-request 32 \\
      --output-tokens 64 --out /tmp/vllm_payloads.jsonl

  python generate_vllm_payloads.py --count 1024 --prefix-groups 8 \\
      --shared-prefix-words 128 --out /tmp/grouped.jsonl --keys-out /tmp/keys.txt

  python generate_vllm_payloads.py --count 1024 --skew-frac 0.2 \\
      --skew-output-tokens 512 --out /tmp/skewed.jsonl
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
    # -- EndpointRouter workload variants ------------------------------------
    p.add_argument("--prefix-groups", type=int, default=0,
                   help="K>0: assign payloads to K groups sharing a head prefix; "
                        "emit a parallel keys file via --keys-out")
    p.add_argument("--shared-prefix-words", type=int, default=128,
                   help="length of the per-group shared (cacheable) head prefix")
    p.add_argument("--keys-out", default=None,
                   help="where to write the parallel 'g<n>' key per payload "
                        "(required when --prefix-groups > 0)")
    p.add_argument("--skew-frac", type=float, default=0.0,
                   help="fraction of payloads that use --skew-output-tokens")
    p.add_argument("--skew-output-tokens", type=int, default=512,
                   help="output tokens for the skewed fraction of payloads")
    args = p.parse_args()

    if args.prefix_groups > 0 and not args.keys_out:
        p.error("--keys-out is required when --prefix-groups > 0")

    # make_body needs a few extra fields it reads off args. output_tokens is
    # mutated per-payload below when --skew-frac is set.
    body_args = SimpleNamespace(
        model=args.model,
        endpoint=args.endpoint,
        output_tokens=args.output_tokens,
        natural_output=False,
        response_schema=args.response_schema,
        rows_per_request=args.rows_per_request,
    )

    rng = random.Random(args.seed)

    # Pre-generate one shared head prefix per group. Reused verbatim across all
    # payloads in the group so the leading tokens are identical -> vLLM prefix
    # cache can hit when those requests land on the same endpoint.
    group_prefixes = []
    if args.prefix_groups > 0:
        for _ in range(args.prefix_groups):
            group_prefixes.append(
                " ".join(rng.choice(driver.WORDLIST)
                         for _ in range(args.shared_prefix_words)))

    out_path = Path(args.out)
    keys_f = open(args.keys_out, "w") if args.keys_out else None
    try:
        with out_path.open("w") as f:
            for i in range(args.count):
                if args.skew_frac > 0.0:
                    body_args.output_tokens = (
                        args.skew_output_tokens
                        if rng.random() < args.skew_frac
                        else args.output_tokens)

                if args.prefix_groups > 0:
                    g = i % args.prefix_groups
                    # No per-payload random head prefix; prepend the shared one.
                    core = driver.build_prompt(
                        rows=args.rows_per_request,
                        row_padding_words=args.row_padding_words,
                        unique_prefix_words=0,
                        rng=rng,
                        base_id=i * args.rows_per_request,
                    )
                    prompt = f"{group_prefixes[g]}\n\n{core}"
                    if keys_f:
                        keys_f.write(f"g{g}\n")
                else:
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
    finally:
        if keys_f:
            keys_f.close()

    msg = f"wrote {args.count} payloads to {out_path}"
    if args.prefix_groups > 0:
        msg += f" ({args.prefix_groups} prefix groups, keys -> {args.keys_out})"
    if args.skew_frac > 0.0:
        msg += f" (skew {args.skew_frac:.2f} -> {args.skew_output_tokens} tok)"
    print(msg, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

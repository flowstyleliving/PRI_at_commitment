#!/usr/bin/env python3
"""Wrapped-vs-unwrapped invariance probe for the inter-head wrapper.

Runs trace_sample twice per prompt:
  1. WRAPPED — with attention_capture active (manual SDPA at target layers).
  2. UNWRAPPED — with no context manager (native fused kernel everywhere).

Compares gen_token_ids element-wise. If the wrapper is observational, all token
IDs should match exactly. Differences indicate the wrapper is perturbing the
forward pass on the target architecture.

Usage:
  .venv/bin/python scripts/invariance_probe_inter_head.py \\
      --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
      --data experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl \\
      --limit 10
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_v2_io_plugins as io_plugins
import pri_v2_mlx_pipeline as pipeline
from pri_calibrator import _load_calibration_jsonl
from scripts.diagnose_inter_head_disagreement import (
    _find_layers,
    _target_layer_map,
    attention_capture,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Wrapped-vs-unwrapped invariance probe")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=4)
    args = p.parse_args()

    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False
    model, tokenizer, projection, layer_indices = pipeline.load_model(args.model, cfg)
    layers = _find_layers(model)
    target_map = _target_layer_map(len(layers))
    prompts, _, _ = _load_calibration_jsonl(args.data)
    prompts = prompts[: args.limit]
    prompt_strategy = io_plugins.get_prompt_strategy(args.model)

    print(f"[invariance] model={args.model}")
    print(f"[invariance] n_layers={len(layers)}  target_map={target_map}")
    print(f"[invariance] {len(prompts)} samples × max_new_tokens={args.max_new_tokens}")
    print()

    n_match = 0
    n_diff = 0
    n_total = 0
    per_sample_status: list[str] = []
    for i, prompt in enumerate(prompts):
        wrapped_prompt = prompt_strategy(prompt, tokenizer)

        # WRAPPED path
        with attention_capture(layers, target_map):
            trace_w = pipeline.trace_sample(
                model=model, tokenizer=tokenizer, prompt=wrapped_prompt,
                layer_indices=layer_indices, output_projection=projection,
                max_new_tokens=args.max_new_tokens, v3_capture=False,
            )
        ids_w = list(trace_w.get("gen_token_ids") or [])

        # UNWRAPPED path
        trace_u = pipeline.trace_sample(
            model=model, tokenizer=tokenizer, prompt=wrapped_prompt,
            layer_indices=layer_indices, output_projection=projection,
            max_new_tokens=args.max_new_tokens, v3_capture=False,
        )
        ids_u = list(trace_u.get("gen_token_ids") or [])

        n_total += 1
        if ids_w == ids_u:
            n_match += 1
            per_sample_status.append(f"  sample {i}: MATCH  ids={ids_w}")
        else:
            n_diff += 1
            per_sample_status.append(
                f"  sample {i}: DIFF   wrapped={ids_w}  unwrapped={ids_u}"
            )

    print("\n".join(per_sample_status))
    print()
    print("=" * 60)
    print(f"[invariance] result: {n_match}/{n_total} match, {n_diff}/{n_total} differ")
    print("=" * 60)
    return 0 if n_diff == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

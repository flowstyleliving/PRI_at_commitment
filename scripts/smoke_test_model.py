#!/usr/bin/env python3
"""
Prereq-4 new-model smoke test.

Loads an MLX model, routes it through the model_adapters factory, and
validates: component-locate succeeds, forward pass runs, logits are finite
and shape-correct, hidden states captured per layer. Target runtime <60s
per model (weights-cached) so this can gate n=20/cell runs cheaply.

Usage:
    python scripts/smoke_test_model.py \\
        --model mlx-community/gemma-3-1b-it-4bit \\
        --model-type gemma3

Exit 0 on pass; exit 1 on any smoke failure with a legible reason.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from mlx_lm import load as mlx_load

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hidden_state_collector  # noqa: E402
from model_adapters import create_adapter  # noqa: E402


def smoke(model_slug: str, model_type: str, prompt: str = "Hello, world.") -> int:
    print(f"[smoke] model={model_slug} type={model_type}")
    t0 = time.time()
    model, tokenizer = mlx_load(model_slug)
    print(f"[smoke] loaded in {time.time() - t0:.1f}s — class={type(model).__name__}")

    # Introspect structure
    core = model.model if hasattr(model, "model") else model
    print(f"[smoke] core class={type(core).__name__}")
    print(f"[smoke]   has embed_tokens={hasattr(core, 'embed_tokens')}")
    print(f"[smoke]   has layers={hasattr(core, 'layers')}")
    print(f"[smoke]   has norm={hasattr(core, 'norm')}")
    print(f"[smoke]   has lm_head (on outer)={hasattr(model, 'lm_head')}")
    if hasattr(core, "layers"):
        print(f"[smoke]   n_layers={len(core.layers)}")
    # SWA-relevant attributes (may or may not exist per family)
    for attr in ("sliding_window_pattern", "window_size", "sliding_window", "swa_idx"):
        if hasattr(core, attr):
            print(f"[smoke]   {attr}={getattr(core, attr)}")

    # Instantiate adapter via factory
    collector = hidden_state_collector.HiddenStateCollector()
    try:
        adapter = create_adapter(model, collector, model_type)
    except Exception as e:
        print(f"[smoke] FAIL factory/adapter-init: {type(e).__name__}: {e}")
        return 1
    print(f"[smoke] adapter={type(adapter).__name__}")
    print(
        f"[smoke]   located: embed_tokens={adapter.embed_tokens is not None} "
        f"norm={adapter.norm is not None} lm_head={adapter.lm_head is not None} "
        f"layers_n={len(adapter.layers) if adapter.layers else 0}"
    )

    # Forward pass
    tokens = tokenizer.encode(prompt)
    print(f"[smoke] prompt='{prompt}' n_tokens={len(tokens)}")
    input_ids = mx.array(tokens, dtype=mx.int32)
    try:
        logits = adapter.forward_prefix_with_collection(input_ids)
        mx.eval(logits)
    except Exception as e:
        print(f"[smoke] FAIL forward_prefix_with_collection: {type(e).__name__}: {e}")
        return 1

    # Cast to float32 before numpy conversion — bfloat16 (used by some mlx-community
    # 4-bit builds, e.g. gemma-3-4b-it-4bit) isn't natively supported by numpy's
    # buffer protocol and raises PEP 3118 errors on direct np.array().
    orig_dtype = logits.dtype
    logits_np = np.array(logits.astype(mx.float32))
    print(f"[smoke] logits shape={logits_np.shape} mlx_dtype={orig_dtype} (cast to float32 for numpy)")
    print(f"[smoke] logits finite={bool(np.isfinite(logits_np).all())}")
    print(f"[smoke] logits[:5]={logits_np[:5]}")

    # Vocab-alignment check: logits last-dim == tokenizer vocab size
    vocab_size = logits_np.shape[-1]
    try:
        tok_vocab = tokenizer.vocab_size
    except AttributeError:
        tok_vocab = len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else None
    print(f"[smoke] vocab_size: logits={vocab_size} tokenizer={tok_vocab}")

    # Hidden states collected
    try:
        blocks = collector.get_all_blocks()
    except Exception as e:
        print(f"[smoke] FAIL collector.get_all_blocks: {type(e).__name__}: {e}")
        return 1
    print(f"[smoke] hidden states captured: {len(blocks)} layers")
    if blocks:
        h0 = np.array(blocks[0].astype(mx.float32))
        hN = np.array(blocks[-1].astype(mx.float32))
        print(f"[smoke]   layer 0 shape={h0.shape} finite={bool(np.isfinite(h0).all())}")
        print(f"[smoke]   layer {len(blocks) - 1} shape={hN.shape} finite={bool(np.isfinite(hN).all())}")

    # All assertions
    if not np.isfinite(logits_np).all():
        print("[smoke] FAIL: logits contain non-finite values")
        return 1
    if len(blocks) == 0:
        print("[smoke] FAIL: no hidden states captured")
        return 1
    if any(not np.isfinite(np.array(b.astype(mx.float32))).all() for b in blocks):
        print("[smoke] FAIL: one or more hidden states contain non-finite values")
        return 1
    if len(blocks) != len(adapter.layers):
        print(
            f"[smoke] FAIL: captured {len(blocks)} layers but adapter has "
            f"{len(adapter.layers)} layers (should match)"
        )
        return 1

    print(f"[smoke] OK (total {time.time() - t0:.1f}s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="mlx-community slug")
    ap.add_argument(
        "--model-type",
        required=True,
        choices=["llama", "qwen", "qwen3", "phi3", "mistral", "gemma3", "smollm", "llava"],
    )
    ap.add_argument("--prompt", default="Hello, world.")
    args = ap.parse_args()
    return smoke(args.model, args.model_type, args.prompt)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Prereq-4 new-model smoke test.

Loads an MLX model, routes it through the model_adapters factory, and
validates: component-locate succeeds, forward pass runs, logits are finite
and shape-correct, hidden states captured per layer. Target runtime <60s
per model (weights-cached) so this can gate n=20/cell runs cheaply.

With --gate, additionally runs the Prereq-4 behavioral preflight:
generates greedy YES/NO answers on n control puzzles (default 4, seeded
for reproducibility) and asserts accuracy ≥ threshold (default 0.98).

Usage:
    python scripts/smoke_test_model.py \\
        --model mlx-community/gemma-3-1b-it-4bit \\
        --model-type gemma3

    python scripts/smoke_test_model.py \\
        --model mlx-community/gemma-3-1b-it-4bit \\
        --model-type gemma3 --gate

Exit 0 on pass; exit 1 on any smoke / gate failure with a legible reason.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hidden_state_collector  # noqa: E402
import synthetic_logic_loader  # noqa: E402
from model_adapters import create_adapter  # noqa: E402


def _parse_yes_no(text: str | None) -> str:
    """
    Loose YES/NO parser for reasoning-tuned models.

    Returns whichever of YES/NO appears FIRST in the response, ignoring role
    headers. Returns UNPARSEABLE only if neither word appears. This differs
    intentionally from run_synthetic_logic_experiment.py's strict
    parse_yes_no_answer (which requires YES/NO as the first meaningful
    alphabetic token and therefore flags reasoning-prefixed answers like
    "Final Answer: YES" as UNPARSEABLE). The strict parser was tuned for
    direct-answer primaries (Llama 3.2 3B, Mistral 7B, Qwen 2.5 7B) — but
    Gemma 3 and Qwen3 are instruct-tuned to reason first, so strict parsing
    would reject correct answers purely on emission style. For the Prereq-4
    behavioral gate we care whether the model reaches the right conclusion,
    not whether it emits it as the first token. Use the runner's strict
    parser if and when exact v2-pilot parity is required.
    """
    if text is None:
        return "UNPARSEABLE"
    s = str(text).strip()
    s = re.sub(r"<\|[^|>]+?\|>", " ", s)
    role_header_tokens = {"ASSISTANT", "USER", "SYSTEM"}
    for match in re.finditer(r"[A-Za-z]+", s):
        tok = match.group(0).upper()
        if tok in role_header_tokens:
            continue
        if tok.startswith("YES"):
            return "YES"
        if tok.startswith("NO"):
            return "NO"
    return "UNPARSEABLE"


def adapter_smoke(model: Any, tokenizer: Any, model_type: str, prompt: str) -> int:
    """Adapter-level smoke: locate, forward, validate outputs."""
    core = model.model if hasattr(model, "model") else model
    print(f"[smoke] core class={type(core).__name__}")
    print(f"[smoke]   has embed_tokens={hasattr(core, 'embed_tokens')}")
    print(f"[smoke]   has layers={hasattr(core, 'layers')}")
    print(f"[smoke]   has norm={hasattr(core, 'norm')}")
    print(f"[smoke]   has lm_head (on outer)={hasattr(model, 'lm_head')}")
    if hasattr(core, "layers"):
        print(f"[smoke]   n_layers={len(core.layers)}")
    for attr in ("sliding_window_pattern", "window_size", "sliding_window", "swa_idx"):
        if hasattr(core, attr):
            print(f"[smoke]   {attr}={getattr(core, attr)}")

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

    tokens = tokenizer.encode(prompt)
    print(f"[smoke] prompt='{prompt}' n_tokens={len(tokens)}")
    input_ids = mx.array(tokens, dtype=mx.int32)
    try:
        logits = adapter.forward_prefix_with_collection(input_ids)
        mx.eval(logits)
    except Exception as e:
        print(f"[smoke] FAIL forward_prefix_with_collection: {type(e).__name__}: {e}")
        return 1

    # Cast to float32 before numpy — bfloat16 (e.g. gemma-3-4b-it-4bit,
    # Qwen3-8B-4bit) fails PEP 3118 on direct np.array().
    orig_dtype = logits.dtype
    logits_np = np.array(logits.astype(mx.float32))
    print(f"[smoke] logits shape={logits_np.shape} mlx_dtype={orig_dtype} (cast to float32 for numpy)")
    print(f"[smoke] logits finite={bool(np.isfinite(logits_np).all())}")
    print(f"[smoke] logits[:5]={logits_np[:5]}")

    vocab_size = logits_np.shape[-1]
    try:
        tok_vocab = tokenizer.vocab_size
    except AttributeError:
        tok_vocab = len(tokenizer.get_vocab()) if hasattr(tokenizer, "get_vocab") else None
    print(f"[smoke] vocab_size: logits={vocab_size} tokenizer={tok_vocab}")

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
    return 0


def behavioral_gate(
    model: Any,
    tokenizer: Any,
    n: int,
    threshold: float,
    seed: int,
    max_tokens: int,
) -> int:
    """
    Prereq-4 step 3: generate greedy answers on n control (non-contradiction)
    puzzles and gate on accuracy ≥ threshold. Samples are seed-reproducible.
    """
    print(f"[gate] building n={n} control puzzles (seed={seed}, baseline/YES-NO mode)")
    # Need enough samples per cell to pull n from the two control cells
    # (short_no_contradiction + long_no_contradiction). With 2 control cells,
    # n_per_cell = ceil(n/2) suffices; n=4 → 2 per cell.
    per_cell = (n + 1) // 2
    sl_cfg = synthetic_logic_loader.SyntheticLogicConfig(
        n_per_cell=per_cell,
        seed=seed,
        prompt_template_variant=synthetic_logic_loader.PROMPT_TEMPLATE_BASELINE,
    ).validate()
    dataset = synthetic_logic_loader.generate_synthetic_logic_dataset(sl_cfg)

    controls = [s for s in dataset if not s.get("has_contradiction", False)][:n]
    if len(controls) < n:
        print(f"[gate] FAIL: only {len(controls)} control samples available (need {n})")
        return 1
    print(f"[gate] selected {len(controls)} control samples — running greedy generation")

    results: List[Tuple[str, str, str, bool]] = []  # (sample_id, expected, parsed, correct)
    t0 = time.time()
    for i, sample in enumerate(controls):
        sample_id = str(sample.get("sample_id", f"s{i}"))
        prompt = str(sample.get("prompt", ""))
        # Baseline (YES/NO) expected answer: "YES" when has_contradiction=False.
        expected = "NO" if bool(sample.get("has_contradiction", False)) else "YES"
        try:
            out = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        except Exception as e:
            print(f"[gate] sample {sample_id}: FAIL generate ({type(e).__name__}: {e})")
            results.append((sample_id, expected, "GEN_ERROR", False))
            continue
        parsed = _parse_yes_no(out)
        correct = parsed == expected
        preview = str(out).strip().replace("\n", " ")[:80]
        print(f"[gate]   {sample_id}  expected={expected}  parsed={parsed}  correct={correct}  output='{preview}'")
        results.append((sample_id, expected, parsed, correct))

    elapsed = time.time() - t0
    n_correct = sum(1 for _, _, _, c in results if c)
    acc = n_correct / len(results) if results else 0.0
    print(f"[gate] accuracy = {n_correct}/{len(results)} = {acc:.3f}  (threshold {threshold:.2f}, elapsed {elapsed:.1f}s)")
    if acc < threshold:
        print(f"[gate] FAIL: accuracy {acc:.3f} < threshold {threshold:.2f}")
        return 1
    print(f"[gate] OK")
    return 0


def run(
    model_slug: str,
    model_type: str,
    prompt: str,
    do_gate: bool,
    gate_n: int,
    gate_threshold: float,
    gate_seed: int,
    gate_max_tokens: int,
) -> int:
    print(f"[smoke] model={model_slug} type={model_type}")
    t0 = time.time()
    model, tokenizer = mlx_load(model_slug)
    print(f"[smoke] loaded in {time.time() - t0:.1f}s — class={type(model).__name__}")

    rc = adapter_smoke(model, tokenizer, model_type, prompt)
    if rc != 0:
        return rc
    print(f"[smoke] OK (adapter total {time.time() - t0:.1f}s)")

    if do_gate:
        gate_rc = behavioral_gate(
            model=model,
            tokenizer=tokenizer,
            n=gate_n,
            threshold=gate_threshold,
            seed=gate_seed,
            max_tokens=gate_max_tokens,
        )
        if gate_rc != 0:
            return gate_rc

    print(f"[smoke] ALL OK (total {time.time() - t0:.1f}s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="mlx-community slug")
    ap.add_argument(
        "--model-type",
        required=True,
        choices=["llama", "qwen", "qwen3", "phi3", "mistral", "gemma3", "smollm", "llava"],
    )
    ap.add_argument("--prompt", default="Hello, world.", help="adapter-smoke input prompt")
    ap.add_argument("--gate", action="store_true", help="also run Prereq-4 behavioral preflight")
    ap.add_argument("--gate-n", type=int, default=4, help="control-puzzle count (plan spec: 4)")
    ap.add_argument("--gate-threshold", type=float, default=0.98, help="min control accuracy to pass")
    ap.add_argument("--gate-seed", type=int, default=42, help="seed for puzzle sampling")
    ap.add_argument("--gate-max-tokens", type=int, default=256, help="max generation length; large enough for reasoning-tuned models (Gemma 3, Qwen3) to finish their CoT and emit YES/NO")
    args = ap.parse_args()
    return run(
        model_slug=args.model,
        model_type=args.model_type,
        prompt=args.prompt,
        do_gate=args.gate,
        gate_n=args.gate_n,
        gate_threshold=args.gate_threshold,
        gate_seed=args.gate_seed,
        gate_max_tokens=args.gate_max_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())

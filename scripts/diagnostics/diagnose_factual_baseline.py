#!/usr/bin/env python3
"""Unpaired TriviaQA-style baseline — just ask the question, capture p_t.

For each factual question, prompt the model with:
    "Question: <Q>\nAnswer:"
Then capture properties of the model's p_t at gen_step=1:
  - entropy(p_t)
  - top1 probability and top1 token
  - surprise on the *correct* answer's first token (if model knows fact, low)
  - surprise on the *wrong* distractor's first token

This gives a per-question scalar profile of "does the model know this fact?"
that we can correlate with paired Fisher-Rao distance later.

Outputs CSV with per-question metrics.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import mlx.core as mx
from mlx_lm import load as mlx_load

import pri_v2_mlx_pipeline as pipeline
from model_adapters import build_attention_masks, forward_layer, pick_layer_mask, post_embed_scale


MODEL_NAME = os.environ.get("DIAG_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
PAIRS_PATH = Path(__file__).resolve().parent.parent / "experiments" / "factual_pairs" / "factual_pairs.json"
PROMPT_TEMPLATE = "Question: {q}\nAnswer:"


def encode_text(tokenizer, text: str):
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text)["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def first_token_id(tokenizer, text: str) -> int:
    """Get the FIRST content token id when encoding `text`, skipping any
    auto-prepended BOS.

    HuggingFace tokenizers for Llama/Mistral auto-prepend their BOS token
    (`<|begin_of_text|>` id 128000 / `<s>` id 1) on every encode() call.
    Without this strip, both `correct_first_tok_id` and `wrong_first_tok_id`
    collapse to BOS for those models, making S(correct) and S(wrong)
    identical and the prefer-correct rate vacuously 0%. Qwen tokenizers
    don't auto-prepend BOS, so they were unaffected.
    """
    ids = encode_text(tokenizer, text)
    bos = getattr(tokenizer, "bos_token_id", None)
    if ids and bos is not None and ids[0] == bos:
        ids = ids[1:]
    return int(ids[0]) if ids else -1


def forward_get_pt(model, ids: list[int]) -> np.ndarray:
    core = model.model if hasattr(model, "model") else model
    layers = pipeline.find_layers(model)
    x = mx.array(np.array([ids], dtype=np.int32))
    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    else:
        raise RuntimeError("no embedding")
    h = post_embed_scale(core, h)
    fa, swa = build_attention_masks(core, h)
    for layer in layers:
        h = forward_layer(layer, h, pick_layer_mask(layer, fa, swa))
    if hasattr(core, "norm"):
        h = core.norm(h)
    elif hasattr(core, "final_layernorm"):
        h = core.final_layernorm(h)
    last = h[:, -1:, :]
    proj = pipeline.OutputProjection(model)
    if proj.mode == "tied_embed":
        logits = proj.layer.as_linear(last)
    else:
        logits = proj.layer(last)
    mx.eval(logits)
    logits_np = pipeline.to_numpy(logits).astype(np.float32)[0, 0]
    return pipeline.safe_softmax(logits_np)


def main() -> int:
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = mlx_load(MODEL_NAME)
    proj = pipeline.OutputProjection(model)
    print(f"  V={proj.vocab_size}, D={proj.hidden_size}")

    if not PAIRS_PATH.exists():
        print(f"ERROR: {PAIRS_PATH} not found.")
        return 2
    with PAIRS_PATH.open() as f:
        pairs = json.load(f)
    # We use the questions, but each "pair" gives us correct + wrong first-token surprises
    print(f"\nLoaded {len(pairs)} factual questions from {PAIRS_PATH.name}")
    print(f"Prompt format: {PROMPT_TEMPLATE!r}\n")

    rows = []
    for i, item in enumerate(pairs):
        prompt = PROMPT_TEMPLATE.format(q=item["question"])
        ids = encode_text(tokenizer, prompt)
        p_t = forward_get_pt(model, ids)

        # Token ids for the correct and wrong answer's FIRST token
        # We try with and without leading space because tokenizers split " Answer" vs "Answer" differently
        correct_with_space = " " + item["correct_answer"]
        wrong_with_space = " " + item["wrong_answer"]
        correct_id = first_token_id(tokenizer, correct_with_space)
        wrong_id = first_token_id(tokenizer, wrong_with_space)

        # Surprise = -log p_t at the relevant token
        eps = 1e-12
        surprise_correct = float(-np.log(p_t[correct_id] + eps)) if correct_id >= 0 else float("nan")
        surprise_wrong = float(-np.log(p_t[wrong_id] + eps)) if wrong_id >= 0 else float("nan")

        # Distribution properties
        entropy = float(-np.sum(p_t * np.log(p_t + eps)))
        top1_id = int(np.argmax(p_t))
        top1_prob = float(p_t[top1_id])
        try:
            top1_decoded = tokenizer.decode([top1_id])[:30]
        except Exception:
            top1_decoded = f"<id={top1_id}>"

        # Was the top-1 token the correct or wrong answer's first token?
        top1_is_correct_first = (top1_id == correct_id)
        top1_is_wrong_first = (top1_id == wrong_id)

        # log-ratio: which answer does the model prefer?
        # log(p_correct / p_wrong) > 0 means model prefers correct
        log_ratio_correct_over_wrong = (
            float(np.log((p_t[correct_id] + eps) / (p_t[wrong_id] + eps)))
            if (correct_id >= 0 and wrong_id >= 0) else float("nan")
        )

        rows.append({
            "question_id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "wrong_answer": item["wrong_answer"],
            "correct_first_tok_id": correct_id,
            "wrong_first_tok_id": wrong_id,
            "top1_token_id": top1_id,
            "top1_token_decoded": top1_decoded,
            "top1_prob": top1_prob,
            "entropy": entropy,
            "surprise_on_correct_first_tok": surprise_correct,
            "surprise_on_wrong_first_tok": surprise_wrong,
            "log_p_correct_over_p_wrong": log_ratio_correct_over_wrong,
            "top1_is_correct_first": top1_is_correct_first,
            "top1_is_wrong_first": top1_is_wrong_first,
        })
        prefer_correct = "✓" if log_ratio_correct_over_wrong > 0 else "✗"
        print(
            f"  [{i+1:>3}/{len(pairs)}] [{item['category']:<12}] {item['question'][:42]:<42} "
            f"top1={top1_decoded!r:<14} "
            f"S(correct)={surprise_correct:>5.2f}  S(wrong)={surprise_wrong:>5.2f}  "
            f"log_ratio={log_ratio_correct_over_wrong:>+5.2f} {prefer_correct}"
        )

    rdf = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parent.parent / "experiments" / "factual_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = MODEL_NAME.split("/")[-1]
    out_path = out_dir / f"factual_baseline_{model_tag}.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")

    # Summary
    print("=== Baseline metrics summary ===")
    print(f"  N questions: {len(rdf)}")
    print(f"  Entropy of p_t:           mean={rdf.entropy.mean():.4f}  median={rdf.entropy.median():.4f}  range=[{rdf.entropy.min():.3f}, {rdf.entropy.max():.3f}]")
    print(f"  Surprise on correct tok:  mean={rdf.surprise_on_correct_first_tok.mean():.4f}  median={rdf.surprise_on_correct_first_tok.median():.4f}")
    print(f"  Surprise on wrong tok:    mean={rdf.surprise_on_wrong_first_tok.mean():.4f}  median={rdf.surprise_on_wrong_first_tok.median():.4f}")
    print(f"  log(p_correct/p_wrong):   mean={rdf.log_p_correct_over_p_wrong.mean():.4f}  median={rdf.log_p_correct_over_p_wrong.median():.4f}")
    n_prefers_correct = (rdf.log_p_correct_over_p_wrong > 0).sum()
    print(f"  Model prefers correct over wrong: {n_prefers_correct}/{len(rdf)} = {n_prefers_correct/len(rdf):.0%}")
    print(f"  Top-1 == correct first tok:  {rdf.top1_is_correct_first.sum()}/{len(rdf)}")
    print(f"  Top-1 == wrong first tok:    {rdf.top1_is_wrong_first.sum()}/{len(rdf)}")

    print("\n=== By category ===")
    for cat in sorted(rdf.category.unique()):
        sub = rdf[rdf.category == cat]
        n = len(sub)
        n_pref_corr = (sub.log_p_correct_over_p_wrong > 0).sum()
        print(f"  {cat:<12}: prefers correct {n_pref_corr}/{n} ({n_pref_corr/n:.0%})  "
              f"S(correct)={sub.surprise_on_correct_first_tok.mean():.2f}  "
              f"S(wrong)={sub.surprise_on_wrong_first_tok.mean():.2f}  "
              f"H={sub.entropy.mean():.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

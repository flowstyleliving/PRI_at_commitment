#!/usr/bin/env python3
"""TriviaQA-style paired Fisher distance — apply paired-condition metric to
factual Q&A pairs.

For each factual pair (same Q, correct vs wrong proposed answer):
  - Run model on both prompts → p_t at gen_step=1
  - Compute KL/JSD/Hellinger/Fisher-Rao distance metrics

Usage:
  DIAG_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit python -u diagnose_factual_paired_fisher.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import mlx.core as mx
from mlx_lm import load as mlx_load

import pri_v2_mlx_pipeline as pipeline
from model_adapters import build_attention_masks, forward_layer, pick_layer_mask, post_embed_scale


MODEL_NAME = os.environ.get("DIAG_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
PAIRS_PATH = Path(__file__).resolve().parents[2] / "experiments" / "factual_pairs" / "factual_pairs.json"


def encode_text(tokenizer, text: str):
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text)["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


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


def distance_metrics(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> dict:
    p = p + eps; q = q + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    bc = min(float(np.sum(np.sqrt(p * q))), 1.0)
    hellinger = float(np.sqrt(max(1.0 - bc, 0.0)))
    fisher_rao = float(2.0 * np.arccos(bc))
    tv = 0.5 * float(np.sum(np.abs(p - q)))
    top1_p = int(np.argmax(p))
    top1_q = int(np.argmax(q))
    return {
        "kl_ctrl_to_contr": kl_pq, "kl_contr_to_ctrl": kl_qp,
        "jsd": jsd, "hellinger": hellinger, "fisher_rao": fisher_rao,
        "total_variation": tv, "bhattacharyya_coef": bc,
        "ctrl_top1_id": top1_p, "contr_top1_id": top1_q,
        "same_top1_token": (top1_p == top1_q),
        "p_ctrl_top1_prob": float(p[top1_p]), "p_contr_top1_prob": float(q[top1_q]),
    }


def main() -> int:
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = mlx_load(MODEL_NAME)
    proj = pipeline.OutputProjection(model)
    print(f"  V={proj.vocab_size}, D={proj.hidden_size}")

    if not PAIRS_PATH.exists():
        print(f"ERROR: {PAIRS_PATH} not found. Run scripts/diagnostics/build_factual_pairs.py first.")
        return 2
    with PAIRS_PATH.open() as f:
        pairs = json.load(f)
    print(f"\nLoaded {len(pairs)} factual pairs from {PAIRS_PATH.name}")

    print(f"\nProcessing {len(pairs)} pairs (2 forward passes each)...\n")
    rows = []
    for i, pair in enumerate(pairs):
        ids_ctrl = encode_text(tokenizer, pair["ctrl_prompt"])
        ids_contr = encode_text(tokenizer, pair["contr_prompt"])
        p_ctrl = forward_get_pt(model, ids_ctrl)
        p_contr = forward_get_pt(model, ids_contr)

        d = distance_metrics(p_ctrl, p_contr)

        try:
            ctrl_dec = tokenizer.decode([d["ctrl_top1_id"]])[:20]
        except Exception:
            ctrl_dec = f"<id={d['ctrl_top1_id']}>"
        try:
            contr_dec = tokenizer.decode([d["contr_top1_id"]])[:20]
        except Exception:
            contr_dec = f"<id={d['contr_top1_id']}>"

        rows.append({
            "pair_id": pair["id"],
            "category": pair["category"],
            "question": pair["question"],
            "correct_answer": pair["correct_answer"],
            "wrong_answer": pair["wrong_answer"],
            "ctrl_top1_decoded": ctrl_dec,
            "contr_top1_decoded": contr_dec,
            **d,
        })
        print(
            f"  [{i+1:>3}/{len(pairs)}] [{pair['category']:<12}] {pair['question'][:48]:<48} "
            f"ctrl→{ctrl_dec!r:>10} contr→{contr_dec!r:>10} | "
            f"JSD={d['jsd']:.4f}  FR={d['fisher_rao']:.4f}  KL(c→c̃)={d['kl_ctrl_to_contr']:.3f}"
        )

    rdf = pd.DataFrame(rows)
    out_dir = Path(__file__).resolve().parents[2] / "experiments" / "factual_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = MODEL_NAME.split("/")[-1]
    out_path = out_dir / f"factual_paired_fisher_{model_tag}.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")

    # Summary by category
    print("=== Distance distributions by category ===")
    for col in ["jsd", "hellinger", "fisher_rao", "kl_ctrl_to_contr", "kl_contr_to_ctrl"]:
        print(f"\n  {col}:")
        for cat in sorted(rdf.category.unique()):
            sub = rdf[rdf.category == cat][col]
            print(f"    {cat:<12}: n={len(sub)}  mean={sub.mean():.4f}  median={sub.median():.4f}  "
                  f"min={sub.min():.4f}  max={sub.max():.4f}")
        print(f"    {'OVERALL':<12}: mean={rdf[col].mean():.4f}  median={rdf[col].median():.4f}")

    print("\n=== Argmax-flip rate (contradiction changes the model's top token) ===")
    for cat in sorted(rdf.category.unique()):
        sub = rdf[rdf.category == cat]
        flip_rate = (1 - sub.same_top1_token.mean())
        print(f"  {cat:<12}: argmax flips in {flip_rate:.0%} of pairs ({int((1-sub.same_top1_token).sum())}/{len(sub)})")
    overall_flip = 1 - rdf.same_top1_token.mean()
    print(f"  {'OVERALL':<12}: {overall_flip:.0%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

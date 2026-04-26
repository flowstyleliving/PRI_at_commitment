#!/usr/bin/env python3
"""Paired-condition Fisher / KL / JSD / Hellinger / Fisher-Rao distance.

For each (chain_length, terms, subject) combination, generate TWO puzzles
that differ ONLY in the contradiction bit. Run the model on both prompts.
Measure how much p_t at gen_step=1 changes between control and contradiction
versions of the SAME prompt structure.

This sidesteps the JSD step1↔step2 category error — both distributions
predict the SAME slot (the (T+1)-th token of equivalently-structured
prompts), so JSD/KL/Hellinger/Fisher-Rao all have clean information-theoretic
interpretation.

Distance metrics computed per pair:
  - KL(p_ctrl || p_contr)             # forward
  - KL(p_contr || p_ctrl)             # backward
  - JSD(p_ctrl, p_contr)              # symmetric, bounded by log(2)
  - Hellinger(p_ctrl, p_contr)        # bounded [0,1], = sqrt(1 - BC)
  - Fisher-Rao(p_ctrl, p_contr)       # geodesic on simplex = 2*arccos(BC)

where BC(p,q) = Σ sqrt(p_i q_i) is the Bhattacharyya coefficient.
Fisher-Rao is the proper Riemannian distance on the probability simplex
under the Fisher information metric — finite-distance generalization
of √d_F² that v3 uses at infinitesimal scale.
"""

from __future__ import annotations

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


SEED = 20260423
N_PAIRS_PER_CL = int(os.environ.get("DIAG_N_PAIRS_PER_CL", "25"))  # 50 pairs total
CHAIN_LENGTHS = [2, 5]
MODEL_NAME = os.environ.get("DIAG_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")


def encode_text(tokenizer, text: str):
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text)["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def forward_get_pt(model, ids: list[int]) -> np.ndarray:
    """Forward pass on `ids`, return p_t at last position over full vocab."""
    core = model.model if hasattr(model, "model") else model
    layers = pipeline.find_layers(model)

    x = mx.array(np.array([ids], dtype=np.int32))
    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    else:
        raise RuntimeError("no embedding layer")
    h = post_embed_scale(core, h)
    fa, swa = build_attention_masks(core, h)
    for layer in layers:
        mask = pick_layer_mask(layer, fa, swa)
        h = forward_layer(layer, h, mask)
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
    """All paired distance metrics in one shot."""
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    bc = float(np.sum(np.sqrt(p * q)))  # Bhattacharyya coefficient
    bc = min(bc, 1.0)  # numerical floor
    hellinger = float(np.sqrt(max(1.0 - bc, 0.0)))
    fisher_rao = float(2.0 * np.arccos(bc))  # geodesic on simplex

    # Top-1 token shift
    top1_p = int(np.argmax(p))
    top1_q = int(np.argmax(q))
    same_top1 = (top1_p == top1_q)

    # Total variation
    tv = 0.5 * float(np.sum(np.abs(p - q)))

    return {
        "kl_ctrl_to_contr": kl_pq,
        "kl_contr_to_ctrl": kl_qp,
        "jsd": jsd,
        "hellinger": hellinger,
        "fisher_rao": fisher_rao,
        "total_variation": tv,
        "bhattacharyya_coef": bc,
        "ctrl_top1_id": top1_p,
        "contr_top1_id": top1_q,
        "same_top1_token": same_top1,
        "p_ctrl_top1_prob": float(p[top1_p]),
        "p_contr_top1_prob": float(q[top1_q]),
    }


def matched_pairs(seed: int, n_per_cl: int, chain_lengths: list[int]) -> list[tuple]:
    """Generate matched (ctrl, contr) pairs sharing terms/subject/cl."""
    gen = pipeline.PuzzleGenerator(seed=seed)
    pairs = []
    for cl in chain_lengths:
        for _ in range(n_per_cl):
            state = gen.rng.getstate()
            ctrl = gen.generate_puzzle(cl, contradiction=False)
            gen.rng.setstate(state)
            contr = gen.generate_puzzle(cl, contradiction=True)
            pairs.append((ctrl, contr))
    return pairs


def main() -> int:
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = mlx_load(MODEL_NAME)
    proj = pipeline.OutputProjection(model)
    print(f"  V={proj.vocab_size}, D={proj.hidden_size}")

    pairs = matched_pairs(SEED, N_PAIRS_PER_CL, CHAIN_LENGTHS)
    total = len(pairs)
    print(f"\nGenerated {total} matched pairs ({N_PAIRS_PER_CL}/cl × {len(CHAIN_LENGTHS)} cls), seed={SEED}")
    print(f"  Sample pair structure (cl=2): control prompts ends with '... is a target', contradiction with '... is not a target'\n")

    # Sanity: verify pair has identical structure except contradiction bit
    if pairs:
        c, k = pairs[0]
        ident = (c["chain_length"] == k["chain_length"] and
                 c["subject"] == k["subject"] and
                 c["target"] == k["target"] and
                 list(c["terms"]) == list(k["terms"]))
        print(f"  Pair-identity check: chain_length={c['chain_length']}, subject={c['subject']}, "
              f"terms-match={ident}, contradiction differ={c['contradiction']!=k['contradiction']}")

    print(f"\nProcessing {total} pairs (2 forward passes each)...\n")
    rows = []
    for i, (ctrl, contr) in enumerate(pairs):
        # Run both prompts
        ids_ctrl = encode_text(tokenizer, ctrl["prompt"])
        ids_contr = encode_text(tokenizer, contr["prompt"])
        p_ctrl = forward_get_pt(model, ids_ctrl)
        p_contr = forward_get_pt(model, ids_contr)

        d = distance_metrics(p_ctrl, p_contr)

        try:
            ctrl_decoded = tokenizer.decode([d["ctrl_top1_id"]])[:20]
        except Exception:
            ctrl_decoded = f"<id={d['ctrl_top1_id']}>"
        try:
            contr_decoded = tokenizer.decode([d["contr_top1_id"]])[:20]
        except Exception:
            contr_decoded = f"<id={d['contr_top1_id']}>"

        rows.append({
            "pair_id": i,
            "chain_length": ctrl["chain_length"],
            "subject": ctrl["subject"],
            "target": ctrl["target"],
            "ctrl_top1_decoded": ctrl_decoded,
            "contr_top1_decoded": contr_decoded,
            **d,
        })
        print(
            f"  [{i+1:>3}/{total}] cl={ctrl['chain_length']} "
            f"ctrl→{ctrl_decoded!r:>10} contr→{contr_decoded!r:>10} | "
            f"KL(c→c̃)={d['kl_ctrl_to_contr']:>6.3f}  KL(c̃→c)={d['kl_contr_to_ctrl']:>6.3f}  "
            f"JSD={d['jsd']:.4f}  Hellinger={d['hellinger']:.4f}  FR={d['fisher_rao']:.4f}"
        )

    rdf = pd.DataFrame(rows)
    model_tag = MODEL_NAME.split("/")[-1]
    out_path = Path(__file__).resolve().parents[2] / "experiments" / "v3-main-run" / "2026-04-24" / f"paired_fisher_{model_tag}.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")

    # Summary stats by chain_length
    print("=== Distance distributions by chain_length ===")
    for col in ["kl_ctrl_to_contr", "kl_contr_to_ctrl", "jsd", "hellinger", "fisher_rao", "total_variation"]:
        print(f"\n  {col}:")
        for cl in CHAIN_LENGTHS:
            sub = rdf[rdf.chain_length == cl][col]
            print(f"    cl={cl}: n={len(sub)}  mean={sub.mean():.4f}  median={sub.median():.4f}  "
                  f"min={sub.min():.4f}  max={sub.max():.4f}")
        # Difference between cl=5 and cl=2 — does longer chain produce bigger ctrl-vs-contr divergence?
        if all(cl in CHAIN_LENGTHS for cl in [2, 5]):
            cl2 = rdf[rdf.chain_length == 2][col].mean()
            cl5 = rdf[rdf.chain_length == 5][col].mean()
            print(f"    Δ(cl=5 − cl=2): {cl5 - cl2:+.4f}")

    print("\n=== Same top-1 token? (does contradiction flip argmax?) ===")
    for cl in CHAIN_LENGTHS:
        sub = rdf[rdf.chain_length == cl]
        same_frac = sub.same_top1_token.mean()
        print(f"  cl={cl}: argmax matches in {same_frac:.0%} of pairs ({int(sub.same_top1_token.sum())}/{len(sub)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

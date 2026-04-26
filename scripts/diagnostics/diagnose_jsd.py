#!/usr/bin/env python3
"""JSD per-step diagnostic — does distribution-shift between consecutive
commit moments discriminate contradictions?

Hypothesis: rank-free, norm-free metric that side-steps both the rank-pinning
issue and the J_n-correction issue. Compute:
    p1 = p(token | prefix)              # distribution at gen_step=1
    p2 = p(token | prefix + token1)     # distribution at gen_step=2
    JSD(p1, p2) = 0.5*KL(p1||M) + 0.5*KL(p2||M),  M = (p1+p2)/2

Larger JSD → bigger landscape shift after committing the first token.
Contradictions hypothesized to produce larger JSD than controls.

Default: N=50 stratified (12/13 per cell), Qwen 2.5 only. Toggle via
DIAG_MODEL env var. Prints AUROC for JSD/KL_fwd/KL_bwd as contradiction
discriminators; saves per-sample CSV.
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
from sklearn.metrics import roc_auc_score

import pri_v2_mlx_pipeline as pipeline
from model_adapters import build_attention_masks, forward_layer, pick_layer_mask, post_embed_scale


SEED = 20260423
N_PER_CELL = int(os.environ.get("DIAG_N_PER_CELL", "12"))  # 48 total stratified
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


def forward_get_pt(model, ids: list[int]) -> tuple[np.ndarray, int, np.ndarray]:
    """Forward pass on `ids`, return (p_t at last position, argmax token, all_logits_at_last)."""
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
    p = pipeline.safe_softmax(logits_np)
    return p, int(np.argmax(logits_np)), logits_np


def jsd_and_kls(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> tuple[float, float, float]:
    """Compute JSD (in nats) and KL(p||q) and KL(q||p). Inputs are dense vectors."""
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    jsd = 0.5 * kl_pm + 0.5 * kl_qm
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return jsd, kl_pq, kl_qp


def main() -> int:
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = mlx_load(MODEL_NAME)
    proj = pipeline.OutputProjection(model)
    print(f"  V={proj.vocab_size}, D={proj.hidden_size}")

    total = 4 * N_PER_CELL
    print(f"\nGenerating {N_PER_CELL}/cell × 2cl × 2contr = {total} samples (seed={SEED})")
    gen = pipeline.PuzzleGenerator(seed=SEED)
    df = gen.generate_dataset(N_PER_CELL, CHAIN_LENGTHS).reset_index(drop=True)

    print(f"\nProcessing {len(df)} samples (each does 2 forward passes)...\n")
    rows = []
    for i, row in df.iterrows():
        prompt = row["prompt"]
        ids = encode_text(tokenizer, prompt)

        # Step 1: p_t at last prefix position
        p1, tok1, _ = forward_get_pt(model, ids)
        try:
            decoded1 = tokenizer.decode([tok1])
        except Exception:
            decoded1 = f"<id={tok1}>"

        # Step 2: append tok1, forward again, get p_t at new last position
        ids_ext = ids + [tok1]
        p2, tok2, _ = forward_get_pt(model, ids_ext)
        try:
            decoded2 = tokenizer.decode([tok2])
        except Exception:
            decoded2 = f"<id={tok2}>"

        # Truncate to common vocab size (defensive — should be identical)
        V = min(p1.shape[0], p2.shape[0])
        jsd, kl_pq, kl_qp = jsd_and_kls(p1[:V], p2[:V])

        # Also compute step1 entropy and step2 entropy for context
        ent1 = float(-np.sum(p1 * np.log(p1 + 1e-12)))
        ent2 = float(-np.sum(p2 * np.log(p2 + 1e-12)))

        # Top-1 prob at each step
        top1_p1 = float(np.max(p1))
        top1_p2 = float(np.max(p2))

        rows.append({
            "sample_id": int(i),
            "contradiction": bool(row["contradiction"]),
            "chain_length": int(row.get("chain_length", -1)),
            "tok_step1": decoded1,
            "tok_step2": decoded2,
            "top1_p_step1": top1_p1,
            "top1_p_step2": top1_p2,
            "entropy_step1": ent1,
            "entropy_step2": ent2,
            "jsd_step1_step2": jsd,
            "kl_step1_step2": kl_pq,
            "kl_step2_step1": kl_qp,
        })
        print(
            f"  [{i+1:>3}/{len(df)}] contr={int(row['contradiction'])} cl={row.get('chain_length','?')} "
            f"step1={decoded1!r:>10} step2={decoded2!r:>10} | "
            f"JSD={jsd:.5f}  KL(1||2)={kl_pq:.4f}  KL(2||1)={kl_qp:.4f} | "
            f"H1={ent1:.3f} H2={ent2:.3f}"
        )

    rdf = pd.DataFrame(rows)
    model_tag = MODEL_NAME.split("/")[-1]
    out_path = Path(__file__).resolve().parents[2] / "experiments" / "v3-main-run" / "2026-04-24" / f"jsd_diagnostic_{model_tag}.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")

    # AUROC analysis
    y = rdf.contradiction.astype(int).to_numpy()

    def auroc_signed(score):
        a = roc_auc_score(y, score)
        return (a, +1) if a >= 0.5 else (1 - a, -1)

    print("=== AUROC for contradiction discrimination ===")
    metrics = [
        "jsd_step1_step2", "kl_step1_step2", "kl_step2_step1",
        "entropy_step1", "entropy_step2",
        "top1_p_step1", "top1_p_step2",
    ]
    for col in metrics:
        score = rdf[col].to_numpy()
        a, sign = auroc_signed(score)
        # sample-level bootstrap CI
        rng = np.random.default_rng(SEED)
        aucs = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), size=len(y))
            yi = y[idx]
            if yi.min() == yi.max(): continue
            au = roc_auc_score(yi, score[idx])
            au = au if sign == 1 else 1 - au
            aucs.append(au)
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        print(f"  {col:<25}  AUROC={a:.4f} sign={'+' if sign==1 else '-'}  CI=[{lo:.3f}, {hi:.3f}]")

    print("\n=== Mean by condition ===")
    for col in metrics:
        c = rdf[~rdf.contradiction][col].mean()
        k = rdf[rdf.contradiction][col].mean()
        print(f"  {col:<25}  ctrl={c:>10.5f}  contr={k:>10.5f}  Δ={k-c:>+10.5f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

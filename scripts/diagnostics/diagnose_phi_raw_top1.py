#!/usr/bin/env python3
"""Diagnose WHY Phi-3.5-mini's raw_top1 hits 0.9989 AUROC contradiction
discrimination at the sealed plane (final layer, gen_step=1, post-norm,
J_n-corrected).

Headline question: is the static W_u top-1 right singular vector a
content-axis (e.g. YES/NO/answer-token) on Phi specifically? If yes, that
mechanistically explains why HARP-style detection saturates here and
Fisher's per-sample reweighting cannot add. If no (as on Mistral, where
top-1 was code-domain tokens but raw_top1 still discriminated as a
rupture-magnitude axis), the explanation is more subtle.

Tests (mirrors scripts/diagnostics/diagnose_mistral_raw_top1.py):
  1. Token analysis: top positive / negative tokens projecting onto
     V_raw[0]; targeted YES / NO / Answer / colon / newline projections.
  2. Per-sample signed projection of Δh_jn onto V_raw[0..2] across
     N=100 stratified puzzles (50 ctrl, 50 contr).
  3. Distribution comparison: ctrl vs contr means / signs / magnitudes.

Output: stdout report + CSV with per-sample signed projections.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from mlx_lm import load as mlx_load

import pri_v2_mlx_pipeline as pipeline
from scripts.diagnostics.diagnose_norm_jacobian import (
    forward_with_pre_and_post_norm,
    apply_norm_jacobian,
    get_norm_gamma,
)


SEED = 20260423
N_PER_CELL = 25
CHAIN_LENGTHS = [2, 5]
MODEL_NAME = "mlx-community/Phi-3.5-mini-instruct-4bit"
TOP_K_TOKENS = 30


def main() -> int:
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = mlx_load(MODEL_NAME)
    proj = pipeline.OutputProjection(model)
    gamma = get_norm_gamma(model)
    print(f"  V={proj.vocab_size}, D={proj.hidden_size}, γ.mean={gamma.mean():.4f}, γ.std={gamma.std():.4f}")

    raw_basis_pkg = proj.raw_right_singular_vectors(8)
    Vt_raw, S_raw, _ = raw_basis_pkg
    print(f"  Raw SVD top-1 σ: {S_raw[0]:.4f} (top-8 σ: {S_raw[:8]})")

    # === TEST 1: token analysis on raw_top1 ===
    print("\n=== TEST 1: tokens with highest SIGNED projection onto raw_top1 ===")
    n_sample = min(16384, proj.vocab_size)
    idx_all = np.arange(n_sample)
    W_sample = proj.get_rows(idx_all)
    proj_top1 = W_sample @ Vt_raw[0]
    proj_top2 = W_sample @ Vt_raw[1]

    print(f"\n  TOP-{TOP_K_TOKENS} positive projections (largest +(W_u[t] · V_raw_top1)):")
    top_pos = np.argsort(-proj_top1)[:TOP_K_TOKENS]
    for ti in top_pos:
        tok_id = int(ti)
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            decoded = f"<id={tok_id}>"
        print(f"    +{proj_top1[ti]:>8.4f}  tok={decoded!r}")

    print(f"\n  TOP-{TOP_K_TOKENS} negative projections (largest -(W_u[t] · V_raw_top1)):")
    top_neg = np.argsort(proj_top1)[:TOP_K_TOKENS]
    for ti in top_neg:
        tok_id = int(ti)
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            decoded = f"<id={tok_id}>"
        print(f"    {proj_top1[ti]:>8.4f}  tok={decoded!r}")

    print("\n  Targeted: project key tokens onto raw_top1 (and top2):")
    target_strs = [" YES", " NO", "YES", "NO", " Yes", " No", "Yes", "No",
                   " Answer", "Answer", " answer", "answer",
                   ":", " ", "\n"]
    for s in target_strs:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False) if hasattr(tokenizer, "encode") else tokenizer(s)["input_ids"]
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            if not ids:
                continue
            for tok_id in ids:
                if tok_id >= n_sample:
                    continue
                rec = (
                    f"    {s!r:<12} → id={tok_id:<6}  "
                    f"top1={proj_top1[tok_id]:>+8.4f}  "
                    f"top2={proj_top2[tok_id]:>+8.4f}"
                )
                print(rec)
        except Exception:
            continue

    # === TEST 2: per-sample signed projection ===
    print(f"\n=== TEST 2: signed projection of Δh_jn onto raw_top1 (N={4 * N_PER_CELL}) ===")
    gen = pipeline.PuzzleGenerator(seed=SEED)
    df = gen.generate_dataset(N_PER_CELL, CHAIN_LENGTHS).reset_index(drop=True)
    rows = []
    for i, row in df.iterrows():
        out = forward_with_pre_and_post_norm(model, tokenizer, row["prompt"])
        dh_pre = out["h_t_pre"] - out["h_prev_pre"]
        dh_jn = apply_norm_jacobian(dh_pre, out["h_prev_pre"], gamma)
        signed_top1 = float(np.dot(dh_jn, Vt_raw[0]))
        signed_top2 = float(np.dot(dh_jn, Vt_raw[1]))
        signed_top3 = float(np.dot(dh_jn, Vt_raw[2]))
        rows.append({
            "sample_id": int(i),
            "contradiction": bool(row["contradiction"]),
            "next_token": out["decoded_next"][:20],
            "dh_jn_l2": float(np.linalg.norm(dh_jn)),
            "signed_proj_raw_top1": signed_top1,
            "signed_proj_raw_top2": signed_top2,
            "signed_proj_raw_top3": signed_top3,
            "abs_proj_top1_over_l2": abs(signed_top1) / (float(np.linalg.norm(dh_jn)) + 1e-10),
        })
        if i < 8 or i >= len(df) - 4:
            print(f"  [{i+1:>3}] contr={int(row['contradiction'])} next={out['decoded_next']!r:>10}  "
                  f"signed_proj(top1)={signed_top1:>+9.2f}  "
                  f"|signed|/|Δh|={abs(signed_top1) / (float(np.linalg.norm(dh_jn))+1e-10):.4f}")

    rdf = pd.DataFrame(rows)
    out_dir = Path("/Users/msrk/Documents/PRI_at_commitment/experiments/v3-main-run/2026-04-30")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phi_signed_proj.csv"
    rdf.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\n=== Signed projection distributions ===")
    for axis in ["signed_proj_raw_top1", "signed_proj_raw_top2", "signed_proj_raw_top3"]:
        c = rdf[~rdf.contradiction][axis]
        k = rdf[rdf.contradiction][axis]
        print(f"  {axis}:")
        print(f"    ctrl     : mean={c.mean():>+9.4f}  std={c.std():.4f}  median={c.median():>+9.4f}")
        print(f"    contr    : mean={k.mean():>+9.4f}  std={k.std():.4f}  median={k.median():>+9.4f}")
        print(f"    Δ(c-c)   : {k.mean() - c.mean():>+9.4f}")
        ctrl_pos_frac = (c > 0).mean()
        contr_pos_frac = (k > 0).mean()
        print(f"    fraction>0:  ctrl={ctrl_pos_frac:.2%}  contr={contr_pos_frac:.2%}  → axis-bipolarity")

    return 0


if __name__ == "__main__":
    sys.exit(main())

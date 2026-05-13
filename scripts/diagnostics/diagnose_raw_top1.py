#!/usr/bin/env python3
"""Generic raw_top1 diagnostic — replaces per-model clones.

Runs the W_u top-1 right-singular-vector token analysis + per-sample
signed projection of Δh_jn onto V_raw[0..2] for any model in the v3.1
lineup. Mirrors scripts/diagnostics/diagnose_{mistral,phi}_raw_top1.py
but parametrized via --model.

Critical: uses pri_v2_mlx_pipeline._extract_final_rmsnorm_gamma (NOT
the legacy get_norm_gamma in diagnose_norm_jacobian.py) so Gemma 3's
1+γ RMSNorm formulation is honored. Without that, Gemma γ would be the
raw (≈0) weight and Δh_jn would collapse.

Outputs:
  - CSV: {out_dir}/{model_short}_signed_proj.csv  (per-sample rows)
  - JSON: {out_dir}/{model_short}_top1_summary.json  (aggregation-ready)

Run:
  python scripts/diagnostics/diagnose_raw_top1.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --out-dir experiments/v3-main-run/2026-04-30
"""

from __future__ import annotations

import argparse
import json
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
)


SEED = 20260423
N_PER_CELL = 25
CHAIN_LENGTHS = [2, 5]
TOP_K_TOKENS = 30


def model_short(model_name: str) -> str:
    return model_name.split("/")[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="mlx-lm model id")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    p.add_argument("--top-k", type=int, default=TOP_K_TOKENS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    short = model_short(args.model)

    print(f"Loading {args.model}...")
    model, tokenizer = mlx_load(args.model)
    # Gemma 3 multimodal wrapper: top-level is gemma3.Model; the actual
    # text model is at model.language_model. OutputProjection / forward
    # helpers expect the text model directly, so unwrap one level.
    if hasattr(model, "language_model"):
        print(f"  detected multimodal wrapper; unwrapping model.language_model")
        model = model.language_model
    proj = pipeline.OutputProjection(model)
    gamma = pipeline._extract_final_rmsnorm_gamma(model)
    if gamma is None:
        print("  ERROR: _extract_final_rmsnorm_gamma returned None", file=sys.stderr)
        return 2
    print(f"  V={proj.vocab_size}, D={proj.hidden_size}, "
          f"γ.mean={gamma.mean():.4f}, γ.std={gamma.std():.4f}")

    raw_basis_pkg = proj.raw_right_singular_vectors(8)
    Vt_raw, S_raw, _ = raw_basis_pkg
    print(f"  Raw SVD top-1 σ: {S_raw[0]:.4f} (top-8 σ: {S_raw[:8]})")

    summary: dict = {
        "model": args.model,
        "model_short": short,
        "vocab_size": int(proj.vocab_size),
        "hidden_size": int(proj.hidden_size),
        "gamma_mean": float(gamma.mean()),
        "gamma_std": float(gamma.std()),
        "top1_sigma": float(S_raw[0]),
        "top8_sigma": [float(s) for s in S_raw[:8]],
    }

    n_sample = min(16384, proj.vocab_size)
    idx_all = np.arange(n_sample)
    W_sample = proj.get_rows(idx_all)
    proj_top1 = W_sample @ Vt_raw[0]
    proj_top2 = W_sample @ Vt_raw[1]

    print(f"\n=== TOP-{args.top_k} positive projections onto V_raw[0] ===")
    top_pos = np.argsort(-proj_top1)[: args.top_k]
    pos_records = []
    for ti in top_pos:
        tok_id = int(ti)
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            decoded = f"<id={tok_id}>"
        print(f"    +{proj_top1[ti]:>8.4f}  tok={decoded!r}")
        pos_records.append({"tok_id": tok_id, "proj": float(proj_top1[ti]), "decoded": decoded})
    summary["top_pos_tokens"] = pos_records

    print(f"\n=== TOP-{args.top_k} negative projections onto V_raw[0] ===")
    top_neg = np.argsort(proj_top1)[: args.top_k]
    neg_records = []
    for ti in top_neg:
        tok_id = int(ti)
        try:
            decoded = tokenizer.decode([tok_id])
        except Exception:
            decoded = f"<id={tok_id}>"
        print(f"    {proj_top1[ti]:>8.4f}  tok={decoded!r}")
        neg_records.append({"tok_id": tok_id, "proj": float(proj_top1[ti]), "decoded": decoded})
    summary["top_neg_tokens"] = neg_records

    print("\n=== Targeted: project key tokens onto V_raw[0] / [1] ===")
    target_strs = [" YES", " NO", "YES", "NO", " Yes", " No", "Yes", "No",
                   " Answer", "Answer", " answer", "answer",
                   ":", " ", "\n"]
    targeted: dict = {}
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
                rec = {
                    "tok_id": int(tok_id),
                    "top1": float(proj_top1[tok_id]),
                    "top2": float(proj_top2[tok_id]),
                }
                key = f"{s!r}_{tok_id}"
                targeted[key] = rec
                print(f"    {s!r:<12} → id={tok_id:<6}  "
                      f"top1={proj_top1[tok_id]:>+8.4f}  "
                      f"top2={proj_top2[tok_id]:>+8.4f}")
        except Exception:
            continue
    summary["targeted_tokens"] = targeted

    print(f"\n=== Per-sample signed Δh_jn · V_raw[0..2] (N={4 * args.n_per_cell}) ===")
    gen = pipeline.PuzzleGenerator(seed=SEED)
    df = gen.generate_dataset(args.n_per_cell, CHAIN_LENGTHS).reset_index(drop=True)
    rows = []
    for i, row in df.iterrows():
        out = forward_with_pre_and_post_norm(model, tokenizer, row["prompt"])
        dh_pre = out["h_t_pre"] - out["h_prev_pre"]
        dh_jn = apply_norm_jacobian(dh_pre, out["h_prev_pre"], gamma)
        signed_top1 = float(np.dot(dh_jn, Vt_raw[0]))
        signed_top2 = float(np.dot(dh_jn, Vt_raw[1]))
        signed_top3 = float(np.dot(dh_jn, Vt_raw[2]))
        l2 = float(np.linalg.norm(dh_jn))
        rows.append({
            "sample_id": int(i),
            "contradiction": bool(row["contradiction"]),
            "chain_length": int(row.get("chain_length", -1)),
            "next_token": out["decoded_next"][:20],
            "dh_jn_l2": l2,
            "signed_proj_raw_top1": signed_top1,
            "signed_proj_raw_top2": signed_top2,
            "signed_proj_raw_top3": signed_top3,
            "abs_proj_top1_over_l2": abs(signed_top1) / (l2 + 1e-10),
        })
        if i < 8 or i >= len(df) - 4:
            print(f"  [{i+1:>3}] contr={int(row['contradiction'])} next={out['decoded_next']!r:>10}  "
                  f"signed(top1)={signed_top1:>+9.2f}  "
                  f"|signed|/|Δh|={abs(signed_top1)/(l2+1e-10):.4f}")

    rdf = pd.DataFrame(rows)
    csv_path = out_dir / f"{short}_signed_proj.csv"
    rdf.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Distributions
    print("\n=== Signed projection distributions ===")
    dist_summary: dict = {}
    for axis in ["signed_proj_raw_top1", "signed_proj_raw_top2", "signed_proj_raw_top3"]:
        c = rdf[~rdf.contradiction][axis]
        k = rdf[rdf.contradiction][axis]
        rec = {
            "ctrl_mean": float(c.mean()),
            "ctrl_std": float(c.std()),
            "ctrl_median": float(c.median()),
            "contr_mean": float(k.mean()),
            "contr_std": float(k.std()),
            "contr_median": float(k.median()),
            "delta_mean": float(k.mean() - c.mean()),
            "ctrl_pos_frac": float((c > 0).mean()),
            "contr_pos_frac": float((k > 0).mean()),
        }
        dist_summary[axis] = rec
        print(f"  {axis}:")
        print(f"    ctrl     : mean={rec['ctrl_mean']:>+9.4f}  std={rec['ctrl_std']:.4f}  median={rec['ctrl_median']:>+9.4f}")
        print(f"    contr    : mean={rec['contr_mean']:>+9.4f}  std={rec['contr_std']:.4f}  median={rec['contr_median']:>+9.4f}")
        print(f"    Δ(c-c)   : {rec['delta_mean']:>+9.4f}")
        print(f"    fraction>0:  ctrl={rec['ctrl_pos_frac']:.2%}  contr={rec['contr_pos_frac']:.2%}")
    summary["distributions"] = dist_summary

    # Commit-token tally (what does this model actually emit at gen_step=1?)
    tok_counts = rdf["next_token"].value_counts().to_dict()
    summary["next_token_counts"] = {str(k): int(v) for k, v in tok_counts.items()}
    print("\n=== gen_step=1 commit tokens (next_token tally) ===")
    for tok, n in list(tok_counts.items())[:10]:
        print(f"    {n:>4}× {tok!r}")

    json_path = out_dir / f"{short}_top1_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nWrote {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

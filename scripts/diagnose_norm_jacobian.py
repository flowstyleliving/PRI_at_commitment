#!/usr/bin/env python3
"""J_n-corrected null_ratio diagnostic — cross-model E17b head-to-head.

Originated as a Qwen-only diagnostic (does pre-norm vs post-norm hidden
state choice flip Qwen's E17b verdict?), since promoted to the cross-
model Stage 2 driver in the overnight pipeline. Runs whichever model
DIAG_MODEL points at and captures three Δh variants:

  - Δh_pre   = h_pre[t] − h_pre[t-1]   (pre-final-RMSNorm)
  - Δh_post  = h_post[t] − h_post[t-1] (post-final-RMSNorm)
  - Δh_jn    = J_n(h_pre[t-1]) · Δh_pre  (Fisher-pullback proper)

For each Δh, computes null_ratio against both Fisher-basis (W_u-derived
under p_t) and raw W_u-SVD basis, across rank sweep ALL_RANKS. Sealed
E17b head-to-head is at rank=1 under Δh_jn — the geometrically correct
linear approximation of Δh_post that lives in the same space as the
W_u-derived basis without higher-order curvature.

Driver-level configuration via environment variables:
  DIAG_MODEL          mlx-lm model id (default: Qwen 2.5 7B 4bit)
  DIAG_N_PER_CELL     samples per (chain_length × contradiction) cell
                      (default: 50 → 200 total at sealed sample size)

Outputs a CSV (one row per sample, all rank/basis/Δh combinations as
columns) and a printed summary table.
"""

from __future__ import annotations

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


SEED = 20260423
# Default 50/cell × 4 cells = 200 total — matches sealed E17b sample size.
# Override with DIAG_N_PER_CELL=<int> for ad-hoc smaller/larger runs.
N_PER_CELL = int(os.environ.get("DIAG_N_PER_CELL", 50))
CHAIN_LENGTHS = [2, 5]
MODEL_NAME = os.environ.get("DIAG_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
ALL_RANKS = [1, 2, 3, 4, 5, 8, 13, 16, 21, 32]


def encode_text(tokenizer, text: str):
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer(text)["input_ids"]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def get_norm_gamma(model):
    """Extract RMSNorm γ (per-channel scale) from the final norm layer."""
    core = model.model if hasattr(model, "model") else model
    norm = None
    if hasattr(core, "norm"):
        norm = core.norm
    elif hasattr(core, "final_layernorm"):
        norm = core.final_layernorm
    if norm is None:
        return None
    if hasattr(norm, "weight"):
        w = norm.weight
        return pipeline.to_numpy(w).astype(np.float32)
    return None


def apply_norm_jacobian(dh: np.ndarray, h_prev: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply J_n(h_prev) · dh where n is RMSNorm.

    n(h) = γ ⊙ h / r(h),  r(h) = sqrt(mean(h²))
    J_n(h) · dh = (1/r) · γ ⊙ (dh - h · (h · dh) / (D · r²))
    """
    D = h_prev.shape[0]
    r = float(np.sqrt(np.mean(h_prev ** 2) + eps))
    h_dot_dh = float(np.dot(h_prev, dh))
    correction = h_prev * (h_dot_dh / (D * r * r))
    return (gamma * (dh - correction)) / r


def forward_with_pre_and_post_norm(model, tokenizer, prompt: str):
    """Run prefix → next-token forward, capture pre-norm and post-norm h
    at last-prefix position AND first-gen-token position."""
    core = model.model if hasattr(model, "model") else model
    layers = pipeline.find_layers(model)
    last_layer_idx = len(layers) - 1

    ids = encode_text(tokenizer, prompt)
    ids_arr = np.array([ids], dtype=np.int32)
    x = mx.array(ids_arr)

    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    else:
        raise RuntimeError("no embedding layer found")

    h = post_embed_scale(core, h)
    fa_mask, swa_mask = build_attention_masks(core, h)

    for li, layer in enumerate(layers):
        mask = pick_layer_mask(layer, fa_mask, swa_mask)
        h = forward_layer(layer, h, mask)

    # h is now [1, T, D] pre-norm at last layer
    mx.eval(h)
    h_pre = pipeline.to_numpy(h).astype(np.float32)[0]  # (T, D)

    # Apply final norm to get post-norm
    if hasattr(core, "norm"):
        h_post_mx = core.norm(h)
    elif hasattr(core, "final_layernorm"):
        h_post_mx = core.final_layernorm(h)
    else:
        raise RuntimeError("no final norm found")
    mx.eval(h_post_mx)
    h_post = pipeline.to_numpy(h_post_mx).astype(np.float32)[0]  # (T, D)

    # Compute logits at last-prefix position
    last_h_mx = h_post_mx[:, -1:, :]
    proj = pipeline.OutputProjection(model)
    if proj.mode == "tied_embed":
        logits = proj.layer.as_linear(last_h_mx)
    else:
        logits = proj.layer(last_h_mx)
    mx.eval(logits)
    logits_np = pipeline.to_numpy(logits).astype(np.float32)[0, 0]

    # Pick next token (greedy)
    next_id = int(np.argmax(logits_np))
    p_t = pipeline.safe_softmax(logits_np)

    # Now run forward with the next token appended (one extra token)
    ids_ext = ids + [next_id]
    x_ext = mx.array(np.array([ids_ext], dtype=np.int32))
    if hasattr(core, "embed_tokens"):
        h2 = core.embed_tokens(x_ext)
    elif hasattr(core, "wte"):
        h2 = core.wte(x_ext)
    h2 = post_embed_scale(core, h2)
    fa2, swa2 = build_attention_masks(core, h2)
    for li, layer in enumerate(layers):
        mask = pick_layer_mask(layer, fa2, swa2)
        h2 = forward_layer(layer, h2, mask)
    mx.eval(h2)
    h_pre2 = pipeline.to_numpy(h2).astype(np.float32)[0]
    if hasattr(core, "norm"):
        h_post2_mx = core.norm(h2)
    elif hasattr(core, "final_layernorm"):
        h_post2_mx = core.final_layernorm(h2)
    mx.eval(h_post2_mx)
    h_post2 = pipeline.to_numpy(h_post2_mx).astype(np.float32)[0]

    # Last-prefix position is index T-1 in original ids; first-gen-token
    # position is index T in extended ids
    last_pref_pre = h_pre[-1]
    last_pref_post = h_post[-1]
    first_gen_pre = h_pre2[len(ids)]
    first_gen_post = h_post2[len(ids)]

    return {
        "h_prev_pre": last_pref_pre,
        "h_prev_post": last_pref_post,
        "h_t_pre": first_gen_pre,
        "h_t_post": first_gen_post,
        "p_t": p_t,
        "next_id": next_id,
        "decoded_next": tokenizer.decode([next_id]) if hasattr(tokenizer, "decode") else str(next_id),
        "proj": proj,
    }


def null_ratio_against_basis(dh: np.ndarray, Vt: np.ndarray, rank: int) -> float:
    """null_ratio = ||dh - V_top V_topᵀ dh|| / ||dh||  for top-`rank` directions."""
    dh_norm = float(np.linalg.norm(dh))
    if dh_norm <= 0:
        return 0.0
    Vt_top = Vt[:rank]
    proj = Vt_top @ dh  # (rank,)
    null_sq = max(dh_norm * dh_norm - float(np.dot(proj, proj)), 0.0)
    return float(np.sqrt(null_sq) / dh_norm)


def fisher_basis(p_t: np.ndarray, proj: pipeline.OutputProjection, max_rank: int) -> np.ndarray:
    """Mirror the pipeline's null_ratio_and_energy support formula EXACTLY:
    support = min(max(256, 16*max_rank), V).
    This gives 512 at max_rank=32 (overnight value) — NOT 1024 as my earlier
    hard-coded version had. Caught by codex 2026-04-25.
    """
    V = p_t.shape[0]
    support = int(min(max(256, max_rank * 16), V))
    idx = np.argpartition(-p_t, kth=support - 1)[:support]
    p_s = p_t[idx]
    W_s = proj.get_rows(idx)  # (support, D)
    A = (np.sqrt(p_s + 1e-10)[:, None]) * W_s
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:max_rank]  # (max_rank, D)


def main() -> int:
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = mlx_load(MODEL_NAME)

    total = 4 * N_PER_CELL
    print(f"Generating 2x2 factorial — {N_PER_CELL}/cell, total {total} samples (cl={CHAIN_LENGTHS}, seed={SEED})")
    gen = pipeline.PuzzleGenerator(seed=SEED)
    df = gen.generate_dataset(N_PER_CELL, CHAIN_LENGTHS).reset_index(drop=True)

    # Pre-compute raw W_u top-r basis (cached basis from OutputProjection)
    proj_for_raw = pipeline.OutputProjection(model)
    max_rank = max(ALL_RANKS)
    raw_basis_pkg = proj_for_raw.raw_right_singular_vectors(max_rank)
    if raw_basis_pkg is None:
        print("ERROR: raw SVD basis unavailable")
        return 2
    Vt_raw, _, _ = raw_basis_pkg  # (max_rank, D)

    # Extract RMSNorm γ for J_n-corrected Fisher pullback
    gamma = get_norm_gamma(model)
    if gamma is None:
        print("WARNING: could not extract RMSNorm γ; J_n correction unavailable")
    else:
        print(f"  Extracted RMSNorm γ: shape={gamma.shape}, mean={gamma.mean():.4f}, std={gamma.std():.4f}")

    print(f"\nProcessing {len(df)} samples (rank sweep {ALL_RANKS})...\n")
    results = []
    for i, row in df.iterrows():
        out = forward_with_pre_and_post_norm(model, tokenizer, row["prompt"])

        dh_pre = out["h_t_pre"] - out["h_prev_pre"]
        dh_post = out["h_t_post"] - out["h_prev_post"]

        # J_n-corrected pre-norm Δh (the proper Fisher pullback object).
        # J_n(h_prev) · Δh_pre is the linear approximation of Δh_post around
        # h_prev. Same space as the W_u-derived basis, but without the
        # higher-order curvature that contaminates Δh_post.
        if gamma is not None:
            dh_jn = apply_norm_jacobian(dh_pre, out["h_prev_pre"], gamma)
        else:
            dh_jn = dh_post  # fallback

        cos_pre_post = float(
            np.dot(dh_pre, dh_post)
            / (np.linalg.norm(dh_pre) * np.linalg.norm(dh_post) + 1e-10)
        )
        cos_jn_post = float(
            np.dot(dh_jn, dh_post)
            / (np.linalg.norm(dh_jn) * np.linalg.norm(dh_post) + 1e-10)
        ) if gamma is not None else 0.0

        Vt_fisher = fisher_basis(out["p_t"], out["proj"], max_rank=max_rank)
        cos_fisher_raw_top1 = float(np.dot(Vt_fisher[0], Vt_raw[0]))

        record = {
            "sample_id": int(i),
            "contradiction": bool(row["contradiction"]),
            "chain_length": int(row.get("chain_length", -1)),
            "next_token_id": int(out["next_id"]),
            "next_token_decoded": out["decoded_next"][:30],
            "dh_pre_l2": float(np.linalg.norm(dh_pre)),
            "dh_post_l2": float(np.linalg.norm(dh_post)),
            "dh_jn_l2": float(np.linalg.norm(dh_jn)),
            "cos_dh_pre_post": cos_pre_post,
            "cos_dh_jn_post": cos_jn_post,
            "cos_fisher_raw_top1": cos_fisher_raw_top1,
        }
        for r in ALL_RANKS:
            record[f"nr_fisher_pre_r{r}"] = null_ratio_against_basis(dh_pre, Vt_fisher, r)
            record[f"nr_fisher_post_r{r}"] = null_ratio_against_basis(dh_post, Vt_fisher, r)
            record[f"nr_fisher_jn_r{r}"] = null_ratio_against_basis(dh_jn, Vt_fisher, r)
            record[f"nr_raw_pre_r{r}"] = null_ratio_against_basis(dh_pre, Vt_raw, r)
            record[f"nr_raw_post_r{r}"] = null_ratio_against_basis(dh_post, Vt_raw, r)
            record[f"nr_raw_jn_r{r}"] = null_ratio_against_basis(dh_jn, Vt_raw, r)
        results.append(record)
        print(
            f"  [{i+1:>3}/{len(df)}] contr={int(row['contradiction'])} cl={row.get('chain_length','?')} "
            f"next={out['decoded_next']!r:>10} | "
            f"|dh|: pre={float(np.linalg.norm(dh_pre)):.0f} post={float(np.linalg.norm(dh_post)):.0f} jn={float(np.linalg.norm(dh_jn)):.0f} | "
            f"cos(jn,post)={cos_jn_post:+.3f} | cos(F,R)_top1={cos_fisher_raw_top1:+.3f}"
        )

    rdf = pd.DataFrame(results)
    model_tag = MODEL_NAME.split("/")[-1]
    out_path = Path(__file__).resolve().parent.parent / "experiments" / "v3-main-run" / "2026-04-24" / f"norm_diagnostic_{model_tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")

    # AUROC table for each rank, both Fisher/raw, both pre/post norm
    from sklearn.metrics import roc_auc_score
    y = rdf.contradiction.astype(int).to_numpy()

    def auroc_signed(score):
        a = roc_auc_score(y, score)
        return (a, +1) if a >= 0.5 else (1.0 - a, -1)

    print("=== AUROC table — three Δh choices: pre-norm | J_n-corrected | post-norm ===")
    print(f"{'rank':>4} | {'F_pre':>8} {'F_jn':>8} {'F_post':>8} | {'R_pre':>8} {'R_jn':>8} {'R_post':>8}")
    for r in ALL_RANKS:
        f_pre, sf_pre = auroc_signed(rdf[f"nr_fisher_pre_r{r}"].to_numpy())
        f_jn, sf_jn = auroc_signed(rdf[f"nr_fisher_jn_r{r}"].to_numpy())
        f_post, sf_post = auroc_signed(rdf[f"nr_fisher_post_r{r}"].to_numpy())
        r_pre, sr_pre = auroc_signed(rdf[f"nr_raw_pre_r{r}"].to_numpy())
        r_jn, sr_jn = auroc_signed(rdf[f"nr_raw_jn_r{r}"].to_numpy())
        r_post, sr_post = auroc_signed(rdf[f"nr_raw_post_r{r}"].to_numpy())
        s = lambda a, sg: f"{a:.4f}{'+' if sg==1 else '-'}"
        print(f"{r:>4} | {s(f_pre,sf_pre):>8} {s(f_jn,sf_jn):>8} {s(f_post,sf_post):>8} | "
              f"{s(r_pre,sr_pre):>8} {s(r_jn,sr_jn):>8} {s(r_post,sr_post):>8}")

    print()
    print("=== HEAD-TO-HEAD ΔAUROC: null_ratio_fisher − null_ratio_raw  (sealed E17b is at rank=1; gate is Δ ≥ +0.02 with CI > 0) ===")
    for r in ALL_RANKS:
        f_pre, _ = auroc_signed(rdf[f"nr_fisher_pre_r{r}"].to_numpy())
        f_jn, _ = auroc_signed(rdf[f"nr_fisher_jn_r{r}"].to_numpy())
        f_post, _ = auroc_signed(rdf[f"nr_fisher_post_r{r}"].to_numpy())
        r_pre, _ = auroc_signed(rdf[f"nr_raw_pre_r{r}"].to_numpy())
        r_jn, _ = auroc_signed(rdf[f"nr_raw_jn_r{r}"].to_numpy())
        r_post, _ = auroc_signed(rdf[f"nr_raw_post_r{r}"].to_numpy())
        print(f"  rank={r:>2} | pre Δ(F-R)={f_pre-r_pre:+.4f} | "
              f"J_n Δ(F-R)={f_jn-r_jn:+.4f} | post Δ(F-R)={f_post-r_post:+.4f}")

    print()
    print("=== SAMPLE-LEVEL means ===")
    for col in ["dh_pre_l2", "dh_post_l2", "dh_jn_l2", "cos_dh_pre_post", "cos_dh_jn_post", "cos_fisher_raw_top1"]:
        if col not in rdf.columns:
            continue
        c = rdf[~rdf.contradiction][col].mean()
        k = rdf[rdf.contradiction][col].mean()
        print(f"  {col:<28}  ctrl={c:+.6f}  contr={k:+.6f}  Δ={k-c:+.6f}")

    print("\n=== INTERPRETATION GUIDE ===")
    print(" - cos(dh_jn, dh_post) ≈ 1.0 confirms J_n is the right linear approximation of Δh_post")
    print(" - If F_jn > F_pre AND F_jn beats R_jn → J_n correction recovers Fisher's edge → v3 saved")
    print(" - If F_jn ≈ F_pre (no improvement) → bug isn't about norm Jacobian; falsification stands")
    print(" - If F_jn improves Fisher across all primaries, keep the correction; if architecture-dependent, document it")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
E22 — direction-depth signature gate (pre-v3 confirmatory run).

For each model × puzzle × layer ℓ at step-1 commitment:
  1. Forward pass on prefix → argmax last-position logit = commit_token.
  2. Forward pass on [prefix + commit_token]; for every transformer block ℓ,
     capture hidden at positions T-1 (h_prev_ℓ) and T (h_t_ℓ).
     Causal mask → position T-1 in the extended pass equals prefix-only last
     (the step-0 h_prev binding in pri_v2_mlx_pipeline.py:1213–1214).
  3. Δh_ℓ = h_t_ℓ − h_prev_ℓ.
  4. Option A eigenspace (single, final-p): p_t = softmax(W_u · h_final_at_T).
     Support-truncate top-256 → W_s, p_s. SVD of A = sqrt(p_s)[:,None]·W_s → Vt.
  5. For r ∈ RANKS: null_ratio_ℓ^r = ||Δh_ℓ − V_topr^T V_topr Δh_ℓ|| / ||Δh_ℓ||,
     fisher_energy^r = Σ_{i<r} σ_i² / Σ_i σ_i².

Output: per-(sample, layer) row with null_ratio_rank{R} + fisher_energy_rank{R}
plus delta_h_norm, delta_h_cosine. r is sample-level; layer varies within sample.

Gate semantics: does null_ratio_ℓ show a cross-arch-reproducible depth profile
that λ_max/λ_mean (2026-04-14 spectral-band run) failed to show?
  Reproducible structure → keep every-layer × 12-step density for v3 main run.
  No structure        → narrow to 5-probe-layer × 12-step schedule.

Reuses: sup_spectral_band.py helpers (greedy_commit_token, forward_all_layers
modified to return two positions). OutputProjection + safe_softmax + find_layers
from pri_v2_mlx_pipeline.

Output: experiments/e22-direction-depth/<YYYY-MM-DD>/run-NN/{model_slug}_e22.parquet
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx
from mlx_lm import load as mlx_load

from model_adapters import build_attention_masks, forward_layer, pick_layer_mask

from pri_v2_mlx_pipeline import (
    OutputProjection,
    find_layers,
    safe_softmax,
    to_numpy,
    encode_text,
)
from synthetic_logic_loader import (
    SyntheticLogicConfig,
    generate_synthetic_logic_dataset,
)
from scripts._paths import experiment_run_dir


# ---- Config ----

MODELS = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
]
N_PER_CELL = 4
SUPPORT = 256
RANKS = (8, 16, 32, 64)

EXPERIMENT_SLUG = "e22-direction-depth"


# ---- Forward pass: per-layer hidden at last TWO positions ----


def forward_all_layers_two_pos(
    model: Any, token_ids: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """For each transformer block, return (h_at_minus2, h_at_minus1) as fp32 vectors.

    Single forward pass over token_ids of shape [1, T+1]; position T-1 = last prefix,
    position T = commit (assuming token_ids = prefix + [commit_token]).
    """
    core = model.model if hasattr(model, "model") else model
    layers = find_layers(model)

    x = mx.array(token_ids.astype(np.int32))
    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    else:
        raise RuntimeError("Could not locate token embedding layer.")

    fa_mask, swa_mask = build_attention_masks(core, h)

    per_layer_two: List[Tuple[np.ndarray, np.ndarray]] = []
    for layer in layers:
        mask = pick_layer_mask(layer, fa_mask, swa_mask)
        h = forward_layer(layer, h, mask)
        mx.eval(h)
        h_np = to_numpy(h)[0]  # [T+1, d]
        per_layer_two.append(
            (h_np[-2].astype(np.float32), h_np[-1].astype(np.float32))
        )

    return per_layer_two


def greedy_commit_token(
    model: Any, tokenizer: Any, prompt: str, projection: OutputProjection
) -> int:
    """Run prefix forward; return argmax of the last-position logits."""
    core = model.model if hasattr(model, "model") else model
    token_ids = np.array(encode_text(tokenizer, prompt), dtype=np.int32)[None, :]
    x = mx.array(token_ids)

    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    fa_mask, swa_mask = build_attention_masks(core, h)
    layers = find_layers(model)
    for layer in layers:
        mask = pick_layer_mask(layer, fa_mask, swa_mask)
        h = forward_layer(layer, h, mask)
    if hasattr(core, "norm"):
        h = core.norm(h)
    elif hasattr(core, "final_layernorm"):
        h = core.final_layernorm(h)
    if projection.mode == "tied_embed":
        logits = projection.layer.as_linear(h)
    else:
        logits = projection.layer(h)
    mx.eval(logits)
    return int(np.argmax(to_numpy(logits)[0, -1]))


# ---- Option A eigenspace + null_ratio ----


def final_p_t_eigenspace(
    h_final_commit: np.ndarray,
    projection: OutputProjection,
    support: int = SUPPORT,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Return (Vt [min(support,d) × d], S², diagnostics).

    SVD of A = sqrt(p_s)[:,None] · W_s where p = softmax(W_u · h_final_commit),
    support = indices of top-`support` entries of p.
    """
    logits = projection.project(h_final_commit.astype(np.float32))
    p = safe_softmax(logits)

    support_k = int(min(support, p.shape[0]))
    idx = np.argpartition(-p, kth=support_k - 1)[:support_k]
    p_s = p[idx].astype(np.float64)
    W_s = projection.get_rows(idx)
    if W_s is None or W_s.ndim != 2:
        raise RuntimeError("OutputProjection.get_rows returned invalid shape.")

    A = (np.sqrt(p_s + 1e-12)[:, None]) * W_s.astype(np.float64)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    lam = S ** 2
    diag = {
        "p_t_top1": float(p.max()),
        "p_t_entropy": float(-np.sum(p * np.log(p + 1e-12))),
        "support_rows": support_k,
    }
    return Vt, lam, diag


def null_ratio_all_ranks(
    dh: np.ndarray, Vt: np.ndarray, ranks: Tuple[int, ...]
) -> Dict[int, float]:
    """For each r in ranks, || dh − Vt[:r]^T Vt[:r] dh || / || dh ||."""
    dh = dh.astype(np.float64)
    dh_norm = float(np.linalg.norm(dh))
    out: Dict[int, float] = {}
    if dh_norm < 1e-12:
        return {r: float("nan") for r in ranks}
    for r in ranks:
        r_eff = int(min(r, Vt.shape[0]))
        V_r = Vt[:r_eff]  # [r, d]
        proj = V_r.T @ (V_r @ dh)  # [d]
        out[r] = float(np.linalg.norm(dh - proj) / dh_norm)
    return out


def fisher_energy_all_ranks(
    lam: np.ndarray, ranks: Tuple[int, ...]
) -> Dict[int, float]:
    total = float(lam.sum()) + 1e-30
    return {
        r: float(lam[: int(min(r, lam.shape[0]))].sum() / total) for r in ranks
    }


# ---- Run one model ----


def run_model(model_name: str, samples: List[Dict[str, Any]]) -> pd.DataFrame:
    print(f"\n[{model_name}] loading...")
    t0 = time.time()
    model, tokenizer = mlx_load(model_name)
    projection = OutputProjection(model)
    n_layers = len(find_layers(model))
    print(
        f"  layers={n_layers} hidden={projection.hidden_size} "
        f"vocab={projection.vocab_size}  load={time.time()-t0:.1f}s"
    )

    rows: List[Dict[str, Any]] = []
    sample_bar = tqdm(samples, desc=f"  {model_name.split('/')[-1]}", unit="sample")
    for sample in sample_bar:
        prompt = sample["prompt"]
        sample_id = sample["sample_id"]
        cell = sample["cell"]
        has_contradiction = bool(sample["has_contradiction"])

        commit_id = greedy_commit_token(model, tokenizer, prompt, projection)
        prefix_ids = encode_text(tokenizer, prompt)
        full_ids = np.array(prefix_ids + [commit_id], dtype=np.int32)[None, :]

        per_layer_two = forward_all_layers_two_pos(model, full_ids)

        # Option A eigenspace from FINAL layer's commit hidden.
        _, h_final_commit = per_layer_two[-1]
        Vt, lam, diag = final_p_t_eigenspace(h_final_commit, projection)
        fe = fisher_energy_all_ranks(lam, RANKS)

        layer_bar = tqdm(
            enumerate(per_layer_two),
            total=n_layers,
            desc=f"    layers ({sample_id})",
            unit="layer",
            leave=False,
        )
        for li, (h_prev_l, h_t_l) in layer_bar:
            dh = (h_t_l.astype(np.float64) - h_prev_l.astype(np.float64))
            dh_norm = float(np.linalg.norm(dh))
            # cosine(h_t, h_prev) — sanity check only.
            ht = h_t_l.astype(np.float64)
            hp = h_prev_l.astype(np.float64)
            denom = float(np.linalg.norm(ht) * np.linalg.norm(hp)) + 1e-30
            cos_ht_hp = float(np.dot(ht, hp) / denom)

            nr = null_ratio_all_ranks(dh, Vt, RANKS)

            row = {
                "model": model_name,
                "sample_id": sample_id,
                "cell": cell,
                "has_contradiction": has_contradiction,
                "commit_token_id": commit_id,
                "layer_index": li,
                "layer_normalized": li / max(n_layers - 1, 1),
                "n_layers": n_layers,
                "delta_h_norm": dh_norm,
                "h_t_h_prev_cos": cos_ht_hp,
                "p_t_top1": diag["p_t_top1"],
                "p_t_entropy": diag["p_t_entropy"],
                "support_rows": diag["support_rows"],
            }
            for r in RANKS:
                row[f"null_ratio_rank{r}"] = nr[r]
                row[f"fisher_energy_rank{r}"] = fe[r]
            rows.append(row)
        layer_bar.close()
        sample_bar.set_postfix(rows=len(rows))

    return pd.DataFrame(rows)


def model_slug(name: str) -> str:
    return name.split("/")[-1].replace(".", "_").lower()


def main() -> int:
    out_dir = experiment_run_dir(EXPERIMENT_SLUG)
    print("=" * 72)
    print("E22 direction-depth signature gate")
    print(f"  n={N_PER_CELL}/cell × 4 cells × 3 models × every layer × {len(RANKS)} ranks")
    print(f"  Output: {out_dir}")
    print("=" * 72)

    cfg = SyntheticLogicConfig(n_per_cell=N_PER_CELL, seed=42)
    samples = generate_synthetic_logic_dataset(cfg)
    print(f"\nGenerated {len(samples)} samples")

    for model_name in tqdm(MODELS, desc="models", unit="model"):
        df = run_model(model_name, samples)
        out_path = out_dir / f"{model_slug(model_name)}_e22.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} rows)")

        print(f"\n  null_ratio_rank32 by depth (median across samples):")
        summary = (
            df.groupby("layer_normalized")["null_ratio_rank32"]
              .agg(["median", "min", "max"])
              .reset_index()
        )
        for _, r in summary.iterrows():
            print(
                f"    depth={r['layer_normalized']:.2f}  "
                f"median={r['median']:.3f}  "
                f"[{r['min']:.3f}, {r['max']:.3f}]"
            )

    print("\nDone. Combine parquets and write the verdict into your research log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

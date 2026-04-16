#!/usr/bin/env python3
"""
SUP spectral-band validation: per-layer Fisher-pullback eigenspectrum at step-1 commitment.

Tests the SUP claim that λ_max/λ_mean ∈ [10², 10⁴] at semantic layers.
Outcome tags: [SUP-VALIDATED-IN-FURNACE], [SHIFTED], [FALSIFIED].

For each model × puzzle × layer ℓ:
  1. Forward pass on (prefix + first generated token) — capture h_ℓ at commitment position.
  2. Compute logit-lens distribution p^(ℓ) = softmax(W_u · h_ℓ).
  3. Support-truncate to top-256 probability rows → W_s, p_s.
  4. SVD of A = sqrt(p_s)[:,None] * W_s → singular values S.
  5. Record λ_max = S[0]², λ_mean = mean(S²), ratio, ε(r) for r ∈ {8,16,32,64}.

Reuses: OutputProjection + safe_softmax + find_layers from pri_v2_mlx_pipeline.

Writes:  furnace-research/raw/experiments/sup-spectral-band/2026-04-14/{model_slug}_spectrum.parquet
Schema:  [model, sample_id, cell, has_contradiction, layer_index, layer_normalized,
          lambda_max, lambda_mean, lambda_ratio, fisher_energy_r8, _r16, _r32, _r64,
          p_t_entropy, top1_prob, support_rows]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Make the main pipeline importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask

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


# ---- Config ----

MODELS = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
]
N_PER_CELL = 4
SUPPORT = 256
ENERGY_RANKS = (8, 16, 32, 64)

OUT_DIR = Path(
    "/Users/msrk/Desktop/furnace-research/raw/experiments/sup-spectral-band/2026-04-14"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- Per-layer capture forward pass ----


def forward_all_layers(model: Any, token_ids: np.ndarray) -> List[np.ndarray]:
    """Return list of hidden states at the FINAL token position, one per transformer block."""
    core = model.model if hasattr(model, "model") else model
    layers = find_layers(model)

    x = mx.array(token_ids.astype(np.int32))
    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    else:
        raise RuntimeError("Could not locate token embedding layer.")

    fa_mask = create_attention_mask(h, None)
    swa_mask = None
    if hasattr(core, "swa_idx") and getattr(core, "swa_idx") is not None:
        swa_mask = create_attention_mask(
            h, None, window_size=getattr(core, "sliding_window", None)
        )

    per_layer_last: List[np.ndarray] = []
    for layer in layers:
        mask = (
            swa_mask
            if (swa_mask is not None and hasattr(layer, "use_sliding") and layer.use_sliding)
            else fa_mask
        )
        try:
            h = layer(h, mask, cache=None)
        except TypeError:
            try:
                h = layer(h, mask, None)
            except TypeError:
                h = layer(h, mask)
        mx.eval(h)
        per_layer_last.append(to_numpy(h)[0, -1].astype(np.float32))

    return per_layer_last


def greedy_commit_token(
    model: Any, tokenizer: Any, prompt: str, projection: OutputProjection
) -> int:
    """Run prefix forward; return argmax of the last-position logits (= first generated token)."""
    core = model.model if hasattr(model, "model") else model
    token_ids = np.array(encode_text(tokenizer, prompt), dtype=np.int32)[None, :]
    x = mx.array(token_ids)

    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    fa_mask = create_attention_mask(h, None)
    layers = find_layers(model)
    for layer in layers:
        try:
            h = layer(h, fa_mask, cache=None)
        except TypeError:
            h = layer(h, fa_mask)
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


# ---- Per-layer spectrum ----


def layer_spectrum(
    h_layer: np.ndarray, projection: OutputProjection, support: int = SUPPORT
) -> Dict[str, float]:
    """Return lambda_max, lambda_mean, lambda_ratio, fisher_energy_r{R}, top1_prob, entropy."""
    logits = projection.project(h_layer.astype(np.float32))
    p = safe_softmax(logits)

    support = int(min(support, p.shape[0]))
    idx = np.argpartition(-p, kth=support - 1)[:support]
    p_s = p[idx].astype(np.float64)
    W_s = projection.get_rows(idx)
    if W_s is None or W_s.ndim != 2:
        return {
            "lambda_max": float("nan"),
            "lambda_mean": float("nan"),
            "lambda_ratio": float("nan"),
            "top1_prob": float(p.max()),
            "p_t_entropy": float(-np.sum(p * np.log(p + 1e-12))),
            **{f"fisher_energy_r{r}": float("nan") for r in ENERGY_RANKS},
        }

    A = (np.sqrt(p_s + 1e-12)[:, None]) * W_s.astype(np.float64)
    try:
        S = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "lambda_max": float("nan"),
            "lambda_mean": float("nan"),
            "lambda_ratio": float("nan"),
            "top1_prob": float(p.max()),
            "p_t_entropy": float(-np.sum(p * np.log(p + 1e-12))),
            **{f"fisher_energy_r{r}": float("nan") for r in ENERGY_RANKS},
        }

    lam = S ** 2
    total = float(lam.sum()) + 1e-30
    out = {
        "lambda_max": float(lam[0]),
        "lambda_mean": float(lam.mean()),
        "lambda_ratio": float(lam[0] / (lam.mean() + 1e-30)),
        "top1_prob": float(p.max()),
        "p_t_entropy": float(-np.sum(p * np.log(p + 1e-12))),
    }
    for r in ENERGY_RANKS:
        r_eff = int(min(r, lam.shape[0]))
        out[f"fisher_energy_r{r}"] = float(lam[:r_eff].sum() / total)
    return out


# ---- Run one model ----


def run_model(model_name: str, samples: List[Dict[str, Any]]) -> pd.DataFrame:
    print(f"\n[{model_name}] loading...")
    model, tokenizer = mlx_load(model_name)
    projection = OutputProjection(model)
    n_layers = len(find_layers(model))
    print(f"  layers={n_layers} hidden={projection.hidden_size} vocab={projection.vocab_size}")

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

        per_layer_h = forward_all_layers(model, full_ids)

        layer_bar = tqdm(
            enumerate(per_layer_h),
            total=n_layers,
            desc=f"    layers ({sample_id})",
            unit="layer",
            leave=False,
        )
        for li, h_vec in layer_bar:
            spec = layer_spectrum(h_vec, projection)
            rows.append({
                "model": model_name,
                "sample_id": sample_id,
                "cell": cell,
                "has_contradiction": has_contradiction,
                "commit_token_id": commit_id,
                "layer_index": li,
                "layer_normalized": li / max(n_layers - 1, 1),
                "n_layers": n_layers,
                "support_rows": SUPPORT,
                **spec,
            })
        layer_bar.close()
        sample_bar.set_postfix(rows=len(rows))

    df = pd.DataFrame(rows)
    return df


def model_slug(name: str) -> str:
    return name.split("/")[-1].replace(".", "_").lower()


def main() -> int:
    print("=" * 72)
    print("SUP spectral-band validation  |  n=4/cell × 3 models × every layer")
    print(f"Output: {OUT_DIR}")
    print("=" * 72)

    cfg = SyntheticLogicConfig(n_per_cell=N_PER_CELL, seed=42)
    samples = generate_synthetic_logic_dataset(cfg)
    print(f"\nGenerated {len(samples)} samples "
          f"(= {N_PER_CELL}/cell × 4 cells)")

    for model_name in tqdm(MODELS, desc="models", unit="model"):
        df = run_model(model_name, samples)
        out_path = OUT_DIR / f"{model_slug(model_name)}_spectrum.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} rows)")

        # Quick per-layer summary
        summary = (
            df.groupby("layer_normalized")["lambda_ratio"]
              .agg(["median", "min", "max"])
              .reset_index()
        )
        print(f"\n  λ_max/λ_mean by depth (median / min / max):")
        for _, row in summary.iterrows():
            print(f"    depth={row['layer_normalized']:.2f}  "
                  f"median={row['median']:.2e}  "
                  f"[{row['min']:.2e}, {row['max']:.2e}]")

    print("\nDone. Combine parquets in wiki/results/sup-spectral-band.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

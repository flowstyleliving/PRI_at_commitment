"""End-to-end verification of _extract_final_rmsnorm_gamma.

Invariant: applying the extracted γ via PRIComputer.rmsnorm in numpy must
reproduce the model's own final-RMSNorm forward pass (modulo dtype rounding).

If max-abs error < 1e-3 on a random h, the extraction is correct for that
family. The Gemma branch should pass when (1.0 + weight) is correctly
applied; without the +1 transform, Gemma's reproduction error would jump by
several orders of magnitude.

Usage: PYTHONUNBUFFERED=1 python -u scripts/verify_gamma_extraction.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_load

from pri_v2_mlx_pipeline import PRIComputer, _extract_final_rmsnorm_gamma, to_numpy


MODELS = [
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", "llama"),
    ("mlx-community/Mistral-7B-Instruct-v0.3-4bit", "mistral"),
    ("mlx-community/Qwen2.5-7B-Instruct-4bit", "qwen2"),
    ("mlx-community/Qwen3-8B-4bit", "qwen3"),
    ("mlx-community/gemma-3-1b-it-4bit", "gemma3-1b"),
    ("mlx-community/gemma-3-4b-it-4bit", "gemma3-4b"),
]


def find_final_norm(model):
    core = model.model if hasattr(model, "model") else model
    if hasattr(core, "norm"):
        return core.norm
    if hasattr(core, "final_layernorm"):
        return core.final_layernorm
    return None


def main() -> int:
    rng = np.random.default_rng(20260426)
    print(
        f"{'model':<48s} {'family':<11s} {'d':>5s} "
        f"{'mean(γ)':>10s} {'mean(|h_post|)':>15s} "
        f"{'max|err|':>12s}  verdict"
    )
    print("-" * 130)
    failures = 0

    for slug, family in MODELS:
        try:
            model, _ = mlx_load(slug)
        except Exception as e:
            print(f"{slug:<48s} LOAD FAIL: {e}")
            failures += 1
            continue

        if hasattr(model, "language_model") and not hasattr(model, "model"):
            model = model.language_model

        gamma_np = _extract_final_rmsnorm_gamma(model)
        if gamma_np is None:
            print(f"{slug:<48s} {family:<11s} extract FAILED")
            failures += 1
            continue
        d = gamma_np.shape[0]

        h_np = rng.standard_normal(d).astype(np.float32) * 5.0
        h_post_pipeline = PRIComputer.rmsnorm(h_np, gamma_np)

        norm = find_final_norm(model)
        if norm is None:
            print(f"{slug:<48s} {family:<11s} no norm module")
            failures += 1
            continue

        h_mx = mx.array(h_np[None, None, :])
        try:
            h_post_model_mx = norm(h_mx)
        except Exception as e:
            print(f"{slug:<48s} {family:<11s} forward FAILED: {e}")
            failures += 1
            continue
        h_post_model = to_numpy(h_post_model_mx).astype(np.float32).reshape(-1)

        max_err = float(np.max(np.abs(h_post_pipeline - h_post_model)))
        mean_gamma = float(np.mean(gamma_np))
        mean_abs_post = float(np.mean(np.abs(h_post_pipeline)))
        ok = max_err < 1e-3
        verdict = "OK" if ok else "WRONG"
        print(
            f"{slug:<48s} {family:<11s} {d:>5d} "
            f"{mean_gamma:>10.4f} {mean_abs_post:>15.4f} "
            f"{max_err:>12.6e}  {verdict}"
        )
        if not ok:
            failures += 1

    print("-" * 130)
    print(f"{'PASS' if failures == 0 else 'FAIL'}: {failures} failure(s)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnose Gemma 3-4B final-RMSNorm mismatch (max|err| = 0.305 vs <1e-5
on every other family including Gemma 3-1B). The +1 transform is right for
1B; what's different on 4B?

Hypotheses to discriminate:
  A. dtype precision (4B is bf16, 1B is fp16) — would explain ~0.4% rounding
     but not 3.6% error.
  B. Gemma 3-4B has additional norms (Q/K-norm, post-FF norm) outside `.norm`
     that the class-level RMSNorm forward applies — but core.norm itself
     should be just the final norm.
  C. mlx-lm gemma3 4B's RMSNorm has different eps or formula than gemma3_text.
  D. Multimodal wrapper has its own norm at a layer we haven't unwrapped to.
  E. mx.fast.rms_norm is doing internal casting that diverges from our
     fp32 numpy reproduction.

Usage: PYTHONUNBUFFERED=1 python -u scripts/diag_gemma_4b_norm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect

import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_load

from pri_v2_mlx_pipeline import PRIComputer, _extract_final_rmsnorm_gamma, to_numpy


def main() -> int:
    slug = "mlx-community/gemma-3-4b-it-4bit"
    print(f"loading {slug}...")
    model, _ = mlx_load(slug)

    print(f"\noutermost class: {type(model).__module__}.{type(model).__name__}")
    print(f"has language_model: {hasattr(model, 'language_model')}")
    print(f"has model: {hasattr(model, 'model')}")

    if hasattr(model, "language_model") and not hasattr(model, "model"):
        model = model.language_model
    print(f"after unwrap: {type(model).__module__}.{type(model).__name__}")
    print(f"  .model: {type(model.model).__module__}.{type(model.model).__name__}")

    core = model.model
    norm = core.norm
    print(f"\nfinal norm class: {type(norm).__module__}.{type(norm).__name__}")
    print(f"  norm.weight.dtype: {norm.weight.dtype}")
    print(f"  norm.weight.shape: {norm.weight.shape}")
    print(f"  norm.eps: {getattr(norm, 'eps', 'N/A')}")
    print(f"  has post_attention_layernorm sibling? {hasattr(core, 'post_attention_layernorm')}")
    print(f"  core attrs: {sorted([a for a in dir(core) if 'norm' in a.lower() and not a.startswith('_')])[:10]}")

    # Print the RMSNorm source
    print(f"\nRMSNorm.__call__ source:")
    src = inspect.getsource(type(norm))
    print(src)

    gamma = _extract_final_rmsnorm_gamma(model)
    d = gamma.shape[0]
    print(f"\nextracted γ_eff: shape={gamma.shape} mean={gamma.mean():.4f} "
          f"std={gamma.std():.4f} min={gamma.min():.4f} max={gamma.max():.4f}")

    raw_weight = to_numpy(norm.weight).astype(np.float32)
    print(f"raw weight:      mean={raw_weight.mean():.4f} std={raw_weight.std():.4f} "
          f"min={raw_weight.min():.4f} max={raw_weight.max():.4f}")

    rng = np.random.default_rng(20260426)
    h_np = rng.standard_normal(d).astype(np.float32) * 5.0

    # Path A: PRIComputer.rmsnorm in fp32
    h_post_pipeline = PRIComputer.rmsnorm(h_np, gamma)

    # Path B: model's own norm at native dtype (bf16)
    h_mx = mx.array(h_np[None, None, :])
    h_post_model = to_numpy(norm(h_mx)).astype(np.float32).reshape(-1)

    # Path C: model's own norm with input forced to bf16 (matches activation dtype)
    h_mx_bf = mx.array(h_np[None, None, :]).astype(mx.bfloat16)
    h_post_model_bf = to_numpy(norm(h_mx_bf)).astype(np.float32).reshape(-1)

    # Path D: numpy emulation but with γ matching what mx.fast.rms_norm sees
    # i.e. (1.0 + raw_weight) computed AT bf16 precision then cast to fp32
    raw_w_bf = mx.array(raw_weight).astype(mx.bfloat16)
    one_plus_w_bf = (mx.ones_like(raw_w_bf) + raw_w_bf).astype(mx.float32)
    gamma_bf = to_numpy(one_plus_w_bf).astype(np.float32)
    h_post_bf_emu = PRIComputer.rmsnorm(h_np, gamma_bf)

    # Path E: numpy emulation with eps=1e-6 (the PRIComputer.rmsnorm default)
    # vs eps=1e-5 (the actual Gemma 3 class default)
    h_post_eps5 = PRIComputer.rmsnorm(h_np, gamma, eps=1e-5)

    def err(a, b):
        return float(np.max(np.abs(a - b)))

    print("\n=== pairwise max|err| ===")
    print(f"  pipeline_fp32     vs  model_native_input(fp32):  {err(h_post_pipeline, h_post_model):.6e}")
    print(f"  pipeline_fp32     vs  model_bf16_input:          {err(h_post_pipeline, h_post_model_bf):.6e}")
    print(f"  pipeline_eps1e-5  vs  model_native_input(fp32):  {err(h_post_eps5, h_post_model):.6e}")
    print(f"  pipeline_bf16_γ   vs  model_native_input(fp32):  {err(h_post_bf_emu, h_post_model):.6e}")
    print(f"  model_native(fp32) vs  model_bf16_input:         {err(h_post_model, h_post_model_bf):.6e}")

    print("\n=== first 10 channels side-by-side ===")
    print(f"{'i':>4s} {'h_in':>12s} {'γ_eff':>12s} {'pipe(fp32)':>14s} {'model(fp32)':>14s} {'model(bf16)':>14s} {'pipe-model':>14s}")
    for i in range(10):
        print(f"{i:>4d} {h_np[i]:>12.4f} {gamma[i]:>12.4f} "
              f"{h_post_pipeline[i]:>14.4f} {h_post_model[i]:>14.4f} {h_post_model_bf[i]:>14.4f} "
              f"{h_post_pipeline[i] - h_post_model[i]:>14.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

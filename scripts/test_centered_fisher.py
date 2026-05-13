#!/usr/bin/env python3
"""v3.2 unit tests — PRIComputer.kl_discharged_and_centered.

Does NOT require MLX or a real model. Uses a numpy-only StubOutputProjection
that serves a deterministic synthetic W_u, then drives the production method
through PRIComputer to check:

  1. kl_discharged closed form == ½·∂hᵀ F_c ∂h (explicit centered Fisher).
  2. Per-rank centered eigvals match direct eigh of F_c on full vocab support.
  3. Σ_r kl_per_dir at full rank == kl_discharged (within numerical eps).
  4. null_ratio_centered_post_rank{r} ∈ [0, 1] and ↓ in r.
  5. Degenerate one-hot p ⇒ kl_discharged ≈ 0 ⇒ null_ratios all 0.0.
  6. Degenerate ∂h_post = 0 ⇒ kl_discharged == 0, null_ratios all 0.0.
  7. High-confidence p (p[k] ≈ 1) ⇒ centered top-eigval is orders of magnitude
     below uncentered (the Qwen 3 8B regime; preview check from amendment).
  8. Smoke: kl_discharged ≈ 0.5 · d_F_full² (existing v2 column identity).

Run with the venv python:
    .venv/bin/python scripts/test_centered_fisher.py
Exit 0 on pass, 1 on fail.

Cross-references:
  - method:  pri_v2_mlx_pipeline.py  PRIComputer.kl_discharged_and_centered
  - pre-reg: wiki/results/v3.2-amendment.md
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

import pri_v2_mlx_pipeline as pipeline  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
#  Stub OutputProjection — numpy-only; no MLX dependency
# ─────────────────────────────────────────────────────────────────────────────


class StubOutputProjection:
    """Minimal OutputProjection stand-in: serves a fixed (V, d) numpy W_u."""

    def __init__(self, W: np.ndarray):
        assert W.ndim == 2, "W must be (V, d)"
        self._W = W.astype(np.float64)
        self.vocab_size = int(W.shape[0])
        self.hidden_size = int(W.shape[1])
        self._raw_svd_cache: Optional[tuple] = None

    def project(self, hidden_vec: np.ndarray) -> np.ndarray:
        return (self._W @ hidden_vec.astype(np.float64)).astype(np.float32)

    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        return self._W[np.asarray(indices, dtype=np.int64)].astype(np.float32)


def make_pri(W: np.ndarray) -> pipeline.PRIComputer:
    """Build a PRIComputer with a stub OutputProjection. final_norm_gamma is
    None — kl_discharged_and_centered does not depend on RMSNorm γ."""
    return pipeline.PRIComputer(StubOutputProjection(W), final_norm_gamma=None)


# ─────────────────────────────────────────────────────────────────────────────
#  Checks
# ─────────────────────────────────────────────────────────────────────────────


def test_closed_form_matches_explicit_fisher() -> None:
    rng = np.random.default_rng(20260507)
    V, d = 64, 32
    W = rng.standard_normal((V, d)) * 0.5
    p = _softmax(rng.standard_normal(V) * 1.5)
    dh = rng.standard_normal(d) * 0.3

    pri = make_pri(W)
    out = pri.kl_discharged_and_centered(dh, p, rank_values=[1, 2, 4, 8, 16, 32])

    # Explicit centered Fisher.
    F_c = W.T @ (np.diag(p) - np.outer(p, p)) @ W
    F_c = 0.5 * (F_c + F_c.T)
    kl_explicit = 0.5 * float(dh @ F_c @ dh)

    assert np.isclose(out["kl_discharged"], kl_explicit, atol=1e-9), (
        f"(1) kl_discharged={out['kl_discharged']:.12f} vs "
        f"explicit ½ ∂hᵀ F_c ∂h={kl_explicit:.12f}"
    )
    print("(1) PASS  kl_discharged == ½ ∂hᵀ F_c ∂h")


def test_centered_eigvals_match_direct_eigh() -> None:
    """At full vocab support, the per-rank `fisher_energy_centered_rank{r}`
    cumulant should match the cumulative-eigvalue ratio of `eigh(F_c)`."""
    rng = np.random.default_rng(0xC0FFEE)
    V, d = 48, 24
    W = rng.standard_normal((V, d)) * 0.4
    p = _softmax(rng.standard_normal(V))
    dh = rng.standard_normal(d) * 0.2

    pri = make_pri(W)
    rank_list = [1, 2, 4, 8, 16, 24]
    out = pri.kl_discharged_and_centered(dh, p, rank_values=rank_list)

    # Direct centered eigh on full F_c (truncated support == full vocab here).
    F_c = W.T @ (np.diag(p) - np.outer(p, p)) @ W
    F_c = 0.5 * (F_c + F_c.T)
    eigvals = np.linalg.eigvalsh(F_c)
    eigvals = np.maximum(eigvals[::-1], 0.0)
    cum_eig = np.cumsum(eigvals) / (np.sum(eigvals) + 1e-12)

    for r in rank_list:
        method = out[f"fisher_energy_centered_rank{r}"]
        direct = float(cum_eig[min(r, len(eigvals)) - 1])
        assert np.isclose(method, direct, atol=1e-7), (
            f"(2) energy rank={r}: method {method:.6f} vs direct {direct:.6f}"
        )
    print("(2) PASS  centered eigvals match direct eigh(F_c)")


def test_per_direction_kl_sums_to_total() -> None:
    """At rank == support_size, KL_topr should equal kl_discharged."""
    rng = np.random.default_rng(2024)
    V, d = 40, 20
    W = rng.standard_normal((V, d)) * 0.4
    p = _softmax(rng.standard_normal(V))
    dh = rng.standard_normal(d) * 0.2

    pri = make_pri(W)
    # Use a rank ≥ support to force full-rank decomposition. Support
    # truncation in null_ratio_and_energy uses min(max(256, max_rank*16), V),
    # so picking max_rank=64 gives support=min(1024, 40)=40 (full vocab).
    out = pri.kl_discharged_and_centered(dh, p, rank_values=[64])
    null_ratio = out["null_ratio_centered_post_rank64"]
    # At full rank, null_ratio should be ≈ 0 (everything captured).
    assert null_ratio < 1e-6, (
        f"(3) null_ratio at full rank = {null_ratio} (expected ≈ 0)"
    )
    print(f"(3) PASS  Σ kl_per_dir at full rank ≈ kl_total  (null_ratio={null_ratio:.2e})")


def test_null_ratio_in_unit_interval_and_monotone() -> None:
    rng = np.random.default_rng(0x42)
    V, d = 64, 32
    W = rng.standard_normal((V, d)) * 0.5
    p = _softmax(rng.standard_normal(V) * 1.0)
    dh = rng.standard_normal(d)

    pri = make_pri(W)
    rank_list = [1, 2, 3, 4, 8, 16, 32]
    out = pri.kl_discharged_and_centered(dh, p, rank_values=rank_list)
    nrs = [out[f"null_ratio_centered_post_rank{r}"] for r in rank_list]
    for r, nr in zip(rank_list, nrs):
        assert 0.0 <= nr <= 1.0 + 1e-9, f"r={r}: null_ratio={nr} out of [0,1]"
    for a, b in zip(nrs, nrs[1:]):
        assert a >= b - 1e-9, f"non-monotone: {a} -> {b}"
    print(f"(4) PASS  null_ratio ∈ [0,1] and ↓ in r  (r=1: {nrs[0]:.4f}, r=32: {nrs[-1]:.4f})")


def test_one_hot_p_is_zero_kl() -> None:
    """One-hot p ⇒ diag(p) − ppᵀ = 0 ⇒ F_c = 0 ⇒ kl_discharged = 0."""
    rng = np.random.default_rng(7)
    V, d = 64, 32
    W = rng.standard_normal((V, d)) * 0.5
    p = np.zeros(V)
    p[3] = 1.0
    dh = rng.standard_normal(d)

    pri = make_pri(W)
    out = pri.kl_discharged_and_centered(dh, p, rank_values=[1, 4, 32])
    assert out["kl_discharged"] < 1e-10, (
        f"(5) one-hot p kl_discharged = {out['kl_discharged']} (expected ≈ 0)"
    )
    for r in [1, 4, 32]:
        assert out[f"null_ratio_centered_post_rank{r}"] == 0.0
        assert out[f"fisher_energy_centered_rank{r}"] == 0.0
    print(f"(5) PASS  one-hot p ⇒ kl_discharged = {out['kl_discharged']:.2e}, null_ratios = 0")


def test_zero_dh_is_zero_kl() -> None:
    rng = np.random.default_rng(8)
    V, d = 64, 32
    W = rng.standard_normal((V, d)) * 0.5
    p = _softmax(rng.standard_normal(V))
    dh = np.zeros(d)

    pri = make_pri(W)
    out = pri.kl_discharged_and_centered(dh, p, rank_values=[1, 8, 32])
    assert out["kl_discharged"] == 0.0
    for r in [1, 8, 32]:
        assert out[f"null_ratio_centered_post_rank{r}"] == 0.0
        assert out[f"fisher_energy_centered_rank{r}"] == 0.0
    print("(6) PASS  ∂h = 0 ⇒ kl_discharged = 0, null_ratios = 0")


def test_high_confidence_centered_vs_uncentered() -> None:
    """At p ≈ one-hot, top eigval of centered F_c should be orders of magnitude
    below top eigval of uncentered W_uᵀ diag(p) W_u. This is the Qwen 3 8B
    regime preview from the v3.2 amendment."""
    rng = np.random.default_rng(11)
    V, d = 64, 32
    W = rng.standard_normal((V, d)) * 0.5
    p = np.full(V, 1e-6)
    p[7] = 1.0 - (V - 1) * 1e-6
    p /= p.sum()

    F_uncentered = W.T @ np.diag(p) @ W
    F_centered = W.T @ (np.diag(p) - np.outer(p, p)) @ W
    top_uncen = float(np.linalg.eigvalsh(F_uncentered)[-1])
    top_cen = float(np.linalg.eigvalsh(F_centered)[-1])
    ratio = top_cen / max(top_uncen, 1e-20)
    assert ratio < 1e-2, (
        f"(7) high-conf ratio {ratio:.3e} not << 1 "
        f"(uncentered={top_uncen:.3e}, centered={top_cen:.3e})"
    )
    print(
        f"(7) PASS  high-conf p: top eigval centered/uncentered = "
        f"{ratio:.2e}  ({top_cen:.2e} / {top_uncen:.2e})"
    )


def test_kl_discharged_matches_d_F_full() -> None:
    """Identity check: kl_discharged ≡ 0.5 · d_F_full² (within numerical eps).
    Confirms the centered closed form is the same quantity in nats that
    fim_full_from_proj already computed in sqrt-Fisher units."""
    rng = np.random.default_rng(12)
    V, d = 64, 32
    W = rng.standard_normal((V, d)) * 0.5
    p = _softmax(rng.standard_normal(V) * 1.2)
    dh = rng.standard_normal(d) * 0.3

    pri = make_pri(W)
    out = pri.kl_discharged_and_centered(dh, p, rank_values=[1])
    z = (W @ dh).astype(np.float32)
    p32 = p.astype(np.float32)
    d_F_full = pipeline.PRIComputer.fim_full_from_proj(z, p32)
    kl_from_dF = 0.5 * float(d_F_full) ** 2
    assert np.isclose(out["kl_discharged"], kl_from_dF, atol=1e-5), (
        f"(8) kl_discharged={out['kl_discharged']:.6e} vs "
        f"½·d_F_full²={kl_from_dF:.6e}"
    )
    print(
        f"(8) PASS  kl_discharged ≡ ½·d_F_full²  "
        f"({out['kl_discharged']:.4e} vs {kl_from_dF:.4e})"
    )


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    tests = [
        test_closed_form_matches_explicit_fisher,
        test_centered_eigvals_match_direct_eigh,
        test_per_direction_kl_sums_to_total,
        test_null_ratio_in_unit_interval_and_monotone,
        test_one_hot_p_is_zero_kl,
        test_zero_dh_is_zero_kl,
        test_high_confidence_centered_vs_uncentered,
        test_kl_discharged_matches_d_F_full,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{failures} / {len(tests)} FAIL")
        return 1
    print(f"ALL {len(tests)} PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

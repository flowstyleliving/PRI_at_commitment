#!/usr/bin/env python3
"""E17b unit test — raw-W_u SVD cache + null_ratio_raw against numpy ground truth.

Does NOT require MLX or a real model. Builds a stand-in OutputProjection that
serves deterministic synthetic W_u rows, then verifies:

  1. `raw_right_singular_vectors(k)` recovers the same top-k right singular
     vectors (up to sign) as `np.linalg.svd(W, full_matrices=False)`.
  2. Cached basis is orthonormal: V_raw @ V_raw^T ≈ I_k.
  3. Eigenvalues ↔ singular values squared: S_raw^2 matches σ^2 from SVD.
  4. `null_ratio_raw_rank{r}` is in [0, 1] for any dh.
  5. `null_ratio_raw_rank{r}` monotonically non-increasing in r (more informed
     directions ⇒ less null content).
  6. For dh aligned with V_1: null_ratio_raw_rank1 ≈ 0 (all in informed).
  7. For dh orthogonal to V_top[:k]: null_ratio_raw_rank{k} ≈ 1 (all null).
  8. `raw_energy_rank{r}` is non-decreasing and reaches 1.0 at r=d.
  9. Chunked accumulation (batch < V) matches unchunked (batch ≥ V).

Run with the venv python:
    .venv/bin/python scripts/test_e17b_raw_svd.py
Exit 0 on pass, 1 on fail.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

# Late import — pipeline's module init runs np.random.seed + mkdir on cfg.save_dir.
import pri_v2_mlx_pipeline as pipeline  # noqa: E402


class StubOutputProjection:
    """Minimal OutputProjection stand-in: serves a fixed np.ndarray as W_u."""

    def __init__(self, W: np.ndarray):
        assert W.ndim == 2, "W must be (V, d)"
        self._W = W
        self.vocab_size = int(W.shape[0])
        self.hidden_size = int(W.shape[1])
        self._raw_svd_cache: Optional[
            Tuple[int, np.ndarray, np.ndarray, float]
        ] = None

    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        return self._W[np.asarray(indices, dtype=np.int64)].astype(np.float32)

    # Attach the pipeline method to this class so the cache logic runs as-is.
    raw_right_singular_vectors = pipeline.OutputProjection.raw_right_singular_vectors


# ─────────────────────────────────────────────────────────────────────────────
#  Checks
# ─────────────────────────────────────────────────────────────────────────────

def _abs_diff_top_subspace(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm of (AAᵀ − BBᵀ). Zero iff A and B span the same subspace
    up to sign / rotation within each singular-value group."""
    return float(np.linalg.norm(A @ A.T - B @ B.T))


def test_svd_matches_numpy_ground_truth() -> None:
    rng = np.random.default_rng(0)
    V, d, k = 1024, 32, 8
    W = rng.standard_normal((V, d)).astype(np.float32)

    proj = StubOutputProjection(W)
    basis = proj.raw_right_singular_vectors(k)
    assert basis is not None, "raw_right_singular_vectors returned None"
    Vt_got, S_got, total_sigma_sq_got = basis

    # Ground truth.
    _, S_ref, Vt_ref = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    Vt_ref_k = Vt_ref[:k]
    S_ref_k = S_ref[:k]
    total_sigma_sq_ref = float(np.sum(S_ref**2))
    # Total-σ² denominator is the full d-eigenvalue sum, used to normalize
    # raw_energy_rank{r} against HARP's 95%-cutoff convention — must match
    # the full SVD's Σ σ² up to numerical slack from chunked accumulation.
    rel_err = abs(total_sigma_sq_got - total_sigma_sq_ref) / (total_sigma_sq_ref + 1e-9)
    assert rel_err < 1e-4, (
        f"total σ² mismatch: got={total_sigma_sq_got:.4f} ref={total_sigma_sq_ref:.4f} "
        f"rel_err={rel_err:.2e}"
    )

    # 1. Same subspace (orthonormal bases — equal up to sign/rotation per
    #    singular-value group). We check the projector is identical.
    subspace_err = _abs_diff_top_subspace(Vt_got, Vt_ref_k)
    assert subspace_err < 1e-4, f"top-{k} subspace mismatch: {subspace_err}"

    # 2. Orthonormality of returned basis.
    gram = Vt_got @ Vt_got.T
    assert np.allclose(gram, np.eye(k), atol=1e-6), f"V_raw not orthonormal: {gram}"

    # 3. Singular values match.
    assert np.allclose(S_got, S_ref_k, atol=1e-3), f"S_raw mismatch: got={S_got} ref={S_ref_k}"

    print("  [pass] raw SVD matches numpy ground truth (subspace, orthonormality, σ)")


def test_null_ratio_range_and_monotonicity() -> None:
    rng = np.random.default_rng(1)
    V, d = 2048, 64
    W = rng.standard_normal((V, d)).astype(np.float32)
    proj = StubOutputProjection(W)
    pri = pipeline.PRIComputer(proj)

    ranks = [1, 2, 4, 8, 16, 32, 64]
    for trial in range(5):
        dh = rng.standard_normal(d).astype(np.float32)
        out = pri.null_ratio_raw_and_energy(dh, ranks)

        # 4. Range.
        for r in ranks:
            nr = out[f"null_ratio_raw_rank{r}"]
            assert 0.0 - 1e-9 <= nr <= 1.0 + 1e-9, f"null_ratio_raw_rank{r}={nr} out of [0,1]"

        # 5. Non-increasing in r.
        seq = [out[f"null_ratio_raw_rank{r}"] for r in ranks]
        for i in range(len(seq) - 1):
            assert seq[i] + 1e-9 >= seq[i + 1], f"non-monotone null_ratio: {seq}"

        # 8. Energy non-decreasing, reaches 1.0 at r=d (denominator is the
        # full d-eigenvalue sum, so energy at r=d must equal 1.0; at r<d
        # it's strictly less).
        energies = [out[f"raw_energy_rank{r}"] for r in ranks]
        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1] + 1e-9, f"non-monotone energy: {energies}"
        # ranks ends at 64 = d here, so the final value equals 1.0.
        assert ranks[-1] == 64 == d
        assert abs(energies[-1] - 1.0) < 1e-6, f"energy at r=d should be 1.0, got {energies[-1]}"
        # At r=d/2=32 we should be strictly less than 1.0 for a full-rank
        # random Gaussian matrix.
        mid = ranks.index(32)
        assert energies[mid] < 1.0 - 1e-6, (
            f"energy at r=32 < 1.0 expected for rank-64 random W; got {energies[mid]}"
        )

    print("  [pass] null_ratio_raw ∈ [0,1], non-increasing in r; energy non-decreasing, →1 at r=d")


def test_dh_aligned_and_orthogonal() -> None:
    rng = np.random.default_rng(2)
    V, d, k = 2048, 32, 16
    W = rng.standard_normal((V, d)).astype(np.float32)
    proj = StubOutputProjection(W)
    pri = pipeline.PRIComputer(proj)

    Vt_raw, _, _ = proj.raw_right_singular_vectors(k)

    # 6. dh aligned with V_1 → null_ratio_rank1 ≈ 0 (float32 cast limits us to
    # ~1e-3 in practice; the method casts to float64 inside, but dh comes in
    # float32 so the alignment projection leaks a tiny fraction).
    dh_aligned = Vt_raw[0] * 3.14
    out = pri.null_ratio_raw_and_energy(dh_aligned.astype(np.float32), [1, k])
    assert out["null_ratio_raw_rank1"] < 1e-3, f"aligned dh null_ratio_rank1 = {out['null_ratio_raw_rank1']}"

    # 7. dh in null complement of V_top[:k] → null_ratio_rank{k} ≈ 1.
    basis_top = Vt_raw  # (k, d)
    rand = rng.standard_normal(d)
    proj_onto_top = basis_top.T @ (basis_top @ rand)
    dh_null = (rand - proj_onto_top).astype(np.float32)
    dh_null /= np.linalg.norm(dh_null)
    out = pri.null_ratio_raw_and_energy(dh_null, [k])
    assert out[f"null_ratio_raw_rank{k}"] > 1.0 - 1e-4, (
        f"orthogonal dh null_ratio_rank{k} = {out[f'null_ratio_raw_rank{k}']}"
    )

    print("  [pass] dh-aligned → null≈0; dh-in-null-complement → null≈1")


def test_chunked_matches_unchunked() -> None:
    rng = np.random.default_rng(3)
    V, d, k = 5000, 24, 6
    W = rng.standard_normal((V, d)).astype(np.float32)

    proj_a = StubOutputProjection(W)
    basis_a = proj_a.raw_right_singular_vectors(k, batch=V + 1)  # single shot
    proj_b = StubOutputProjection(W)
    basis_b = proj_b.raw_right_singular_vectors(k, batch=128)    # chunked

    assert basis_a is not None and basis_b is not None
    Vt_a, S_a, tot_a = basis_a
    Vt_b, S_b, tot_b = basis_b

    # 9. Subspace identity regardless of chunk size.
    err = _abs_diff_top_subspace(Vt_a, Vt_b)
    assert err < 1e-4, f"chunk-size subspace drift: {err}"
    assert np.allclose(S_a, S_b, atol=1e-3), f"chunk-size S drift: a={S_a} b={S_b}"
    assert abs(tot_a - tot_b) / (tot_a + 1e-9) < 1e-6, (
        f"chunk-size total σ² drift: a={tot_a} b={tot_b}"
    )

    print("  [pass] chunked accumulation (batch=128) matches single-shot (batch=V+1)")


def test_compute_step_emits_both_null_ratios() -> None:
    """compute_step should emit null_ratio_rank{r} AND null_ratio_raw_rank{r}
    when v3_capture_raw=True and v3_rank_values is non-empty."""
    rng = np.random.default_rng(4)
    V, d = 1024, 16
    W = rng.standard_normal((V, d)).astype(np.float32)
    proj = StubOutputProjection(W)
    pri = pipeline.PRIComputer(proj)

    # Override .project to bypass the real projection (needs MLX); the method
    # is only used for the v2 Fisher path, which requires z = W_u · dh via
    # MLX. Monkey-patch _project for the test so compute_step doesn't touch MLX.
    def _np_project(dh: np.ndarray) -> np.ndarray:
        return (W.astype(np.float64) @ dh.astype(np.float64)).astype(np.float32)
    pri._project = _np_project  # type: ignore[assignment]

    h_t = rng.standard_normal(d).astype(np.float32)
    h_prev = rng.standard_normal(d).astype(np.float32)
    dh = h_t - h_prev
    logits = W @ dh
    p_t = np.exp(logits - logits.max())
    p_t /= p_t.sum()

    ranks = [1, 4, 8]
    out = pri.compute_step(
        h_t, h_prev, p_t, S_t=0.5, alpha=1.0,
        topk_values=[], lowrank_values=[],
        v3_rank_values=ranks, v3_capture_raw=True,
    )

    for r in ranks:
        assert f"null_ratio_rank{r}" in out, f"missing Fisher null_ratio_rank{r}"
        assert f"null_ratio_raw_rank{r}" in out, f"missing raw null_ratio_raw_rank{r}"
        assert f"fisher_energy_rank{r}" in out, f"missing fisher_energy_rank{r}"
        assert f"raw_energy_rank{r}" in out, f"missing raw_energy_rank{r}"
        # Neither should be NaN for a finite-W setup.
        assert np.isfinite(out[f"null_ratio_rank{r}"]), f"NaN Fisher null at r={r}"
        assert np.isfinite(out[f"null_ratio_raw_rank{r}"]), f"NaN raw null at r={r}"

    # v3_capture_raw=False must NOT emit the raw columns.
    out_off = pri.compute_step(
        h_t, h_prev, p_t, S_t=0.5, alpha=1.0,
        topk_values=[], lowrank_values=[],
        v3_rank_values=ranks, v3_capture_raw=False,
    )
    for r in ranks:
        assert f"null_ratio_raw_rank{r}" not in out_off, (
            f"raw null leaked when v3_capture_raw=False at r={r}"
        )

    print("  [pass] compute_step emits both Fisher and raw columns when flag is on; "
          "none when off")


def test_cache_reuse_returns_same_basis() -> None:
    """Second call with the same k reads the cache — should be byte-identical."""
    rng = np.random.default_rng(5)
    W = rng.standard_normal((1024, 20)).astype(np.float32)
    proj = StubOutputProjection(W)

    b1 = proj.raw_right_singular_vectors(8)
    b2 = proj.raw_right_singular_vectors(8)
    b3 = proj.raw_right_singular_vectors(4)  # subset of cached k
    assert b1 is not None and b2 is not None and b3 is not None
    assert np.array_equal(b1[0], b2[0]) and np.array_equal(b1[1], b2[1])
    assert abs(b1[2] - b2[2]) < 1e-9, "total σ² should be cache-identical on reuse"
    assert b3[0].shape == (4, 20) and b3[1].shape == (4,)
    assert np.array_equal(b3[0], b1[0][:4]) and np.array_equal(b3[1], b1[1][:4])
    # Subset slice shares the same total σ² (denominator is whole-model, not k).
    assert abs(b3[2] - b1[2]) < 1e-9, "total σ² should not depend on subset slice"

    print("  [pass] cache reuse returns identical basis; subset slice is a prefix")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\nE17b raw-W_u SVD unit tests\n" + "=" * 48)
    try:
        test_svd_matches_numpy_ground_truth()
        test_null_ratio_range_and_monotonicity()
        test_dh_aligned_and_orthogonal()
        test_chunked_matches_unchunked()
        test_cache_reuse_returns_same_basis()
        test_compute_step_emits_both_null_ratios()
    except AssertionError as e:
        print(f"\n  [FAIL] {e}")
        return 1
    except Exception as e:
        import traceback
        print(f"\n  [ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print("=" * 48 + "\nE17b tests — all 6 bundles passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

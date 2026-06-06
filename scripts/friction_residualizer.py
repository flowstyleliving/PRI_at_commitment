#!/usr/bin/env python3
"""Draft — friction residualizer for residual-stream sub-layer friction.

Turns ABSOLUTE friction into a deviation score: "how many sigma weirder than
normal refinement is this block, for its layer and route size?" Raw friction
cannot separate a constructive refinement from a destructive veto (proven in
test_residual_friction.py check 9); the residualizer removes the benign
"big route -> big trim" variance so the pathological veto stands out.

Two flavors, identical skeleton, differing only in feature set + target:
  - W_u-free : target = interference;
               nuisance feats = ‖a‖, log‖a‖, sink_fraction, gaze_entropy
               (‖m‖ deliberately dropped — see THE TRAP)
  - W_u      : target = directed_veto onto the answer axis û;
               + route_on_answer = |a·û|

THE TRAP (do not regress out the signal): regressors must be NUISANCE covariates
only — quantities that benignly drive trimming. Never include anything that
itself predicts the label, NOR any argument of the target. ‖m‖ is EXCLUDED for
the latter reason: interference = 1 − ‖a+m‖/(‖a‖+‖m‖) is mechanically a function
of ‖m‖, so regressing on it is circular and can delete real signal (Codex
2026-06-05). `route_on_answer` is defensible ("how much answer-content was on the
table"); the t=0 output margin is deliberately EXCLUDED — it is downstream of the
commitment under study and would absorb the signal.

DRAFT / not wired. numpy-only. At productionization the friction primitives
consolidate into pri_v2_mlx_pipeline.py and the cell into pri_calibrator.py.
Fit on GROUNDED (or held-out) commits only — Mahalanobis/ridge with d feats on
n samples is optimistically biased in-sample (use the calibrator's nested OOB).

Self-test:
    .venv/bin/python scripts/friction_residualizer.py
Exit 0 on pass, 1 on fail.

Cross-references:
  - sibling identity suite: scripts/test_residual_friction.py
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Friction primitives  (consolidate into pri_v2_mlx_pipeline.py at productionize)
# ─────────────────────────────────────────────────────────────────────────────


def interference(a: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> float:
    """Destructive-interference fraction 1 - ‖a+m‖/(‖a‖+‖m‖) ∈ [0, 1]."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    m = np.asarray(m, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nm = float(np.linalg.norm(m))
    if na + nm <= eps:
        return 0.0
    return 1.0 - float(np.linalg.norm(a + m)) / (na + nm)


def directed_veto(a: np.ndarray, m: np.ndarray, u: np.ndarray, eps: float = 1e-12) -> float:
    """Signed cancellation of a's push along consequential direction u:
    -(a·û)(m·û). > 0 ⇒ MLP vetoes the route on u; < 0 ⇒ endorses it.

    `u` is the W_u/Fisher dial (see candidate #9 handle (iii)): neighbour-block
    Δh (clean), activation top-PC, or the answer direction (W_u, Fisher-free)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    m = np.asarray(m, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    nu = float(np.linalg.norm(u))
    if nu <= eps:
        return 0.0
    uhat = u / nu
    return -float(a @ uhat) * float(m @ uhat)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction — NUISANCE covariates only (never label-carrying)
# ─────────────────────────────────────────────────────────────────────────────


def featurize_wu_free(a: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """W_u-free nuisance covariates of the ROUTE only. alpha = gaze over positions.

    Deliberately excludes ‖m‖: interference = 1 − ‖a+m‖/(‖a‖+‖m‖) is mechanically
    a function of ‖m‖, so regressing the target on ‖m‖ is circular and can delete
    real signal (Codex 2026-06-05). `m` is kept in the signature for API symmetry
    with featurize_wu but is not used as a covariate."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    sink = float(alpha[0]) if alpha.size else 0.0
    ent = float(-(alpha * np.log(alpha + 1e-12)).sum()) if alpha.size else 0.0
    return np.array([na, np.log1p(na), sink, ent])


def featurize_wu(a: np.ndarray, m: np.ndarray, alpha: np.ndarray, u_hat: np.ndarray) -> np.ndarray:
    """W_u variant: + how much attention routed toward the answer axis û.
    u_hat = unit answer direction (W_u[YES]-W_u[NO], or the belief-readout dir)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    route_on_answer = abs(float(a @ np.asarray(u_hat, dtype=np.float64).reshape(-1)))
    return np.concatenate([featurize_wu_free(a, m, alpha), [route_on_answer]])


# ─────────────────────────────────────────────────────────────────────────────
#  The residualizer — per-layer ridge → studentized residual
# ─────────────────────────────────────────────────────────────────────────────


class InterferenceResidualizer:
    """Removes benign 'big route → big trim' variance. transform() returns, per
    sample, (actual - predicted_normal) / sigma_LOO for its layer = "sigma above
    normal refinement". sigma is the leave-one-out (out-of-fold) residual spread,
    not in-sample, so it does not understate the deviation. Large positive =
    anomalous veto. Fit on GROUNDED (or held-out) commits only."""

    def __init__(self, lam: float = 1.0):
        self.lam = float(lam)
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.W: dict[int, np.ndarray] = {}
        self.sigma: dict[int, float] = {}

    def fit(self, layer_ids, X, y) -> "InterferenceResidualizer":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        layer_ids = np.asarray(layer_ids)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9  # ridge is scale-sensitive → standardize feats
        Xs = (X - self.mu) / self.sd
        # errstate silences macOS-Accelerate's spurious matmul FP flags (numpy 2.x);
        # the explicit finiteness assert below still catches a genuine blow-up.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for L in np.unique(layer_ids):
                sel = layer_ids == L
                D, yl = self._design(Xs[sel]), y[sel]
                # ridge closed form w = (DᵀD + P)⁻¹ Dᵀy, with P = λ·diag(0,1,…,1):
                # the intercept (col 0) is NOT penalized — it is the per-layer
                # baseline we want to estimate, so shrinking it biases small-n layers.
                P = self.lam * np.eye(D.shape[1])
                P[0, 0] = 0.0
                Ainv = np.linalg.inv(D.T @ D + P)
                w = Ainv @ (D.T @ yl)
                if not np.isfinite(w).all():
                    raise FloatingPointError(f"non-finite ridge weights at layer {int(L)}")
                # Honest out-of-fold sigma: exact ridge leave-one-out residuals
                # e_i/(1−h_ii), h_ii = leverage from the hat matrix H = D·Ainv·Dᵀ.
                # (In-sample resid.std understates → studentization was dishonest.)
                resid = yl - D @ w
                h = np.einsum("ij,jk,ik->i", D, Ainv, D)
                loo = resid / np.clip(1.0 - h, 1e-6, None)
                self.W[int(L)] = w
                self.sigma[int(L)] = float(loo.std(ddof=1)) or 1.0
        return self

    def transform(self, layer_ids, X, y) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        layer_ids = np.asarray(layer_ids)
        Xs = (X - self.mu) / self.sd
        out = np.empty(len(y))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i in range(len(y)):
                L = int(layer_ids[i])
                pred = float((self._design(Xs[i][None, :]) @ self.W[L])[0])
                out[i] = (y[i] - pred) / self.sigma[L]
        return out

    @staticmethod
    def _design(X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X])  # + intercept


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test — same raw friction, separated by route-size conditioning
# ─────────────────────────────────────────────────────────────────────────────


def _self_test() -> int:
    rng = np.random.default_rng(20260605)
    n = 300
    na = rng.uniform(0.5, 2.0, n)  # route size varies across calibration commits
    # NORMAL refinement scales with route size (the benign confound we residualize out).
    y = 0.10 + 0.30 * na + rng.normal(0.0, 0.02, n)
    X = np.column_stack(
        [
            na,
            np.log1p(na),
            rng.uniform(0.0, 0.3, n),  # sink_fraction (noise wrt y)
            rng.uniform(0.5, 1.5, n),  # gaze_entropy (noise wrt y)
        ]
    )
    layer = np.zeros(n, dtype=int)

    res = InterferenceResidualizer(lam=1.0).fit(layer, X, y)

    # Two probes with IDENTICAL raw interference but different route size.
    raw = 0.67
    x_benign = np.array([1.9, np.log1p(1.9), 0.1, 1.0])  # high route ⇒ 0.67 EXPECTED
    x_veto = np.array([0.6, np.log1p(0.6), 0.1, 1.0])    # low route  ⇒ 0.67 ANOMALOUS
    z = res.transform([0, 0], np.vstack([x_benign, x_veto]), [raw, raw])
    z_benign, z_veto = float(z[0]), float(z[1])

    failures = 0
    if abs(z_benign) >= 2.0:
        print(f"FAIL  benign not near normal: z={z_benign:.2f} (want |z| < 2)")
        failures += 1
    if z_veto <= 3.0:
        print(f"FAIL  veto not flagged: z={z_veto:.2f} (want > 3)")
        failures += 1
    if z_veto <= z_benign + 3.0:
        print(f"FAIL  no separation: benign {z_benign:.2f} vs veto {z_veto:.2f}")
        failures += 1

    # Featurizer shapes.
    a = np.array([1.0, 1.0, 0.0, 0.0])
    m = np.array([-0.5, 0.0, 1.0, 0.0])
    alpha = np.array([0.6, 0.2, 0.1, 0.1])
    fwf = featurize_wu_free(a, m, alpha)
    fw = featurize_wu(a, m, alpha, u_hat=np.array([1.0, 0.0, 0.0, 0.0]))
    if fwf.shape != (4,) or fw.shape != (5,):
        print(f"FAIL  featurizer shapes: W_u-free {fwf.shape}, W_u {fw.shape}")
        failures += 1

    print()
    if failures:
        print(f"{failures} checks FAIL")
        return 1
    print(
        f"PASS  same raw interference {raw}: benign z={z_benign:+.2f} (normal) vs "
        f"veto z={z_veto:+.2f} (anomalous) — route-conditioning isolates the veto"
    )
    print(f"PASS  featurizers: W_u-free dim {fwf.shape[0]}, W_u dim {fw.shape[0]}")
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())

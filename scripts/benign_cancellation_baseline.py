#!/usr/bin/env python3
"""Benign Cancellation Baseline for v5 residual friction.

This is an OFFLINE diagnostic for candidate #9. It does not load MLX models.

Two checks are intentionally separated:

1. A pure NumPy same-Delta synthetic harness. It constructs paired decompositions
   with exactly the same residual update Delta = A + M and the same cancellation
   magnitude, but different hidden disagreement directions. The real split has
   label-aligned disagreement on an epistemic direction u; the benign split puts
   the same norm in the u-orthogonal subspace. A valid Knowledge Veto statistic
   should beat the benign same-Delta split.

2. A persisted-dump panel. Schema v3 dumps contain an `Xbenign` block: the same
   raw cancellation/interference magnitudes with the directed-veto component
   replaced by the route-matched same-Delta benign split. Older schema v2 dumps
   contain only scalar friction features and a random-u control, so they fall back
   to the conservative random-u floor:

       net = Delta(real friction | null+route) - Delta(random-u | null+route)

The exact vector-level construction is A' = Delta/2 + r, M' = Delta/2 - r with
r rotated away from the consequential direction u while preserving the route
norm constraints when feasible. The pilot persists the sufficient projections
needed for this control; it does not need to store full A/M vectors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from friction_residualizer import directed_veto, interference  # noqa: E402
from pilot_residual_friction import _repeated_cv_delta  # noqa: E402


def _veto(a: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    return -float(a @ m) / na if na > eps else 0.0


def _unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= eps:
        raise ValueError("cannot normalize near-zero vector")
    return x / n


def _orthogonalize(x: np.ndarray, basis: Iterable[np.ndarray]) -> np.ndarray:
    out = np.asarray(x, dtype=np.float64).copy()
    for b in basis:
        b = np.asarray(b, dtype=np.float64)
        nb2 = float(b @ b)
        if nb2 > 1e-12:
            out -= float(out @ b) / nb2 * b
    return out


def _make_orthogonal_unit(
    rng: np.random.Generator,
    dim: int,
    basis: Iterable[np.ndarray],
) -> np.ndarray:
    for _ in range(1000):
        x = _orthogonalize(rng.standard_normal(dim), basis)
        n = float(np.linalg.norm(x))
        if n > 1e-8:
            return x / n
    raise RuntimeError("failed to sample an orthogonal unit vector")


def _friction_features(A: np.ndarray, M: np.ndarray, u: np.ndarray) -> np.ndarray:
    out = np.empty((len(A), 3), dtype=np.float64)
    for i, (a, m) in enumerate(zip(A, M)):
        out[i, 0] = interference(a, m)
        out[i, 1] = _veto(a, m)
        out[i, 2] = directed_veto(a, m, u)
    return out


def _synthetic_same_delta(
    *,
    n: int,
    dim: int,
    seed: int,
    repeats: int,
    folds: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2, dtype=np.int32), np.ones(n - n // 2, dtype=np.int32)]
    rng.shuffle(y)

    u = np.zeros(dim, dtype=np.float64)
    u[0] = 1.0

    Delta = np.empty((n, dim), dtype=np.float64)
    R_real = np.empty((n, dim), dtype=np.float64)
    R_benign = np.empty((n, dim), dtype=np.float64)
    rho = rng.lognormal(mean=0.0, sigma=0.15, size=n)

    for i in range(n):
        # Keep Delta orthogonal to u, so Delta itself cannot carry the epistemic
        # direction and A/M route norms are controlled by ||Delta|| and ||R||.
        delta_dir = _make_orthogonal_unit(rng, dim, [u])
        Delta[i] = rng.uniform(0.6, 1.4) * delta_dir

        # Same cancellation magnitude for both classes, but class 1 allocates
        # much more of that disagreement along u. Class 0 mostly refines in a
        # u-orthogonal direction. Both are orthogonal to Delta.
        q = 0.85 if y[i] == 1 else 0.25
        ortho_dir = _make_orthogonal_unit(rng, dim, [u, Delta[i]])
        R_real[i] = rho[i] * (q * u + np.sqrt(max(0.0, 1.0 - q * q)) * ortho_dir)

        # Benign control: identical Delta and identical ||R||, but no u component.
        R_benign[i] = rho[i] * _make_orthogonal_unit(rng, dim, [u, Delta[i]])

    A_real, M_real = 0.5 * Delta + R_real, 0.5 * Delta - R_real
    A_benign, M_benign = 0.5 * Delta + R_benign, 0.5 * Delta - R_benign

    Xroute = np.column_stack(
        [
            np.linalg.norm(A_real, axis=1),
            np.linalg.norm(M_real, axis=1),
            np.linalg.norm(Delta, axis=1),
            np.ones(n),
        ]
    )
    Xnull = rng.normal(size=(n, 1))
    base = np.column_stack([Xnull, Xroute])
    Xreal = _friction_features(A_real, M_real, u)
    Xbenign = _friction_features(A_benign, M_benign, u)

    real = _repeated_cv_delta(base, np.column_stack([base, Xreal]), y, repeats, folds, seed)
    benign = _repeated_cv_delta(base, np.column_stack([base, Xbenign]), y, repeats, folds, seed)

    delta_err = float(np.max(np.abs((A_real + M_real) - (A_benign + M_benign))))
    route_err = float(
        np.max(
            np.abs(
                np.column_stack(
                    [
                        np.linalg.norm(A_real, axis=1),
                        np.linalg.norm(M_real, axis=1),
                        np.linalg.norm(A_real + M_real, axis=1),
                    ]
                )
                - np.column_stack(
                    [
                        np.linalg.norm(A_benign, axis=1),
                        np.linalg.norm(M_benign, axis=1),
                        np.linalg.norm(A_benign + M_benign, axis=1),
                    ]
                )
            )
        )
    )

    return {
        "n": n,
        "dim": dim,
        "same_delta_max_abs": delta_err,
        "route_norm_max_abs": route_err,
        "real_delta": real["delta_median"],
        "real_lo": real["delta_lo"],
        "real_hi": real["delta_hi"],
        "benign_delta": benign["delta_median"],
        "benign_lo": benign["delta_lo"],
        "benign_hi": benign["delta_hi"],
        "net_delta": real["delta_median"] - benign["delta_median"],
        "auc_base": real["auc_a"],
        "auc_real": real["auc_b"],
        "auc_benign": benign["auc_b"],
    }


def _dump_panel(npz_paths: list[Path], *, seed: int, repeats: int, folds: int) -> str:
    lines: list[str] = []
    lines.append("Persisted dump panel: same-Delta benign floor when present, else random-u fallback")
    lines.append("model primary lo hi floor_kind floor lo hi net schema")
    for path in npz_paths:
        d = np.load(path, allow_pickle=True)
        y = np.asarray(d["y"], dtype=np.int32)
        Xnull = np.asarray(d["Xnull"], dtype=np.float64)
        Xroute = np.asarray(d["Xroute"], dtype=np.float64)
        Xfric = np.asarray(d["Xfric"], dtype=np.float64)
        base = np.column_stack([Xnull, Xroute])
        real = _repeated_cv_delta(base, np.column_stack([base, Xfric]), y, repeats, folds, seed)
        if "Xbenign" in d.files:
            Xfloor = np.asarray(d["Xbenign"], dtype=np.float64)
            floor_kind = "same_delta"
        else:
            Xfloor = np.asarray(d["Xrand"], dtype=np.float64)
            floor_kind = "random_u"
        floor = _repeated_cv_delta(base, np.column_stack([base, Xfloor]), y, repeats, folds, seed)
        meta = json.loads(str(d["metadata"])) if "metadata" in d.files else {}
        model = meta.get("model_slug", path.stem).replace("mlx-community/", "")
        schema = meta.get("schema", "unknown")
        lines.append(
            f"{model} "
            f"{real['delta_median']:+.4f} {real['delta_lo']:+.4f} {real['delta_hi']:+.4f} "
            f"{floor_kind} "
            f"{floor['delta_median']:+.4f} {floor['delta_lo']:+.4f} {floor['delta_hi']:+.4f} "
            f"{real['delta_median'] - floor['delta_median']:+.4f} {schema}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="*", help="optional residual_friction_features.npz dumps for random-u floor panel")
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--cv-repeats", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--synthetic-n", type=int, default=400)
    ap.add_argument("--synthetic-dim", type=int, default=128)
    args = ap.parse_args()

    syn = _synthetic_same_delta(
        n=args.synthetic_n,
        dim=args.synthetic_dim,
        seed=args.seed,
        repeats=args.cv_repeats,
        folds=args.folds,
    )
    print("Synthetic same-Delta harness")
    print(f"  n={syn['n']} dim={syn['dim']}")
    print(f"  max |Delta_real - Delta_benign| = {syn['same_delta_max_abs']:.3e}")
    print(f"  max route-norm mismatch          = {syn['route_norm_max_abs']:.3e}")
    print(
        "  real epistemic split: "
        f"Delta={syn['real_delta']:+.4f} [{syn['real_lo']:+.4f},{syn['real_hi']:+.4f}] "
        f"auc={syn['auc_real']:.3f}"
    )
    print(
        "  benign same-Delta:    "
        f"Delta={syn['benign_delta']:+.4f} [{syn['benign_lo']:+.4f},{syn['benign_hi']:+.4f}] "
        f"auc={syn['auc_benign']:.3f}"
    )
    print(f"  net Knowledge Veto floor-subtracted Delta = {syn['net_delta']:+.4f}")

    if args.npz:
        print()
        print(_dump_panel([Path(p) for p in args.npz], seed=args.seed, repeats=args.cv_repeats, folds=args.folds), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

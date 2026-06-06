#!/usr/bin/env python3
"""Residual-norm budget diagnostic for v5 residual friction.

OFFLINE only: consumes schema-v2/v3 residual_friction_features.npz dumps and
asks whether the apparent friction signal is better explained by norm budgeting:
attention writes a large route, MLP trims/counterweights it, and the block keeps
the net residual update in range.

The decisive checks are nested OOF AUROC increments:

  budget_norm | null+route
      Adds norm-budget features that are not the directed-veto axis:
      mean ||a+m||, path sum ||a||+||m||, balance | ||a||-||m|| |, and gain
      ratios.

  budget_trim | null+route
      Adds the above plus explicit cancellation/trim ratios. This is intentionally
      close to "benign cancellation" and should be read as a budget floor, not a
      veto.

  friction | null+route+budget_*
      If this is ~0, the veto block is not adding beyond budget/cancellation.

  same-Delta benign | null+route
      For schema-v3 dumps, the stricter paired floor: same Delta, same raw
      cancellation, disagreement rotated off the consequential direction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pilot_residual_friction import _repeated_cv_delta, _signfree_auroc  # noqa: E402


def _safe_div(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return num / np.maximum(np.abs(den), eps)


def _budget_features(d: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    LT = np.asarray(d["layer_tensor"], dtype=np.float64)
    names = [str(x) for x in d["layer_feature_names"]]
    idx = {name: i for i, name in enumerate(names)}
    na = LT[:, :, idx["na"]]
    nm = LT[:, :, idx["nm"]]
    hnorm = LT[:, :, idx["hnorm"]]
    if "delta_norm" in idx:
        delta = LT[:, :, idx["delta_norm"]]
    else:
        # v2 fallback: recover ||a+m|| from interference = 1 - ||a+m||/(||a||+||m||).
        delta = (na + nm) * (1.0 - LT[:, :, idx["interference"]])

    path = na + nm
    trim = path - delta
    balance = np.abs(na - nm)
    ratio_delta_path = _safe_div(delta, path)
    ratio_trim_path = _safe_div(trim, path)
    ratio_balance_path = _safe_div(balance, path)
    ratio_delta_h = _safe_div(delta, hnorm)

    def mean(x: np.ndarray) -> np.ndarray:
        return np.nanmean(x, axis=1)

    Xnorm = np.column_stack(
        [
            mean(delta),
            mean(path),
            mean(balance),
            mean(ratio_delta_path),
            mean(ratio_balance_path),
            mean(ratio_delta_h),
        ]
    )
    norm_names = [
        "mean_delta_norm",
        "mean_path_norm",
        "mean_balance_norm",
        "mean_delta_over_path",
        "mean_balance_over_path",
        "mean_delta_over_hnorm",
    ]

    Xtrim = np.column_stack([Xnorm, mean(trim), mean(ratio_trim_path)])
    trim_names = norm_names + ["mean_trim_norm", "mean_trim_over_path"]
    return Xnorm, norm_names, Xtrim, trim_names


def _line(label: str, d: dict[str, float]) -> str:
    return (
        f"{label:<34s} aucA={d['auc_a']:.3f} aucB={d['auc_b']:.3f} "
        f"Delta={d['delta_median']:+.4f} [{d['delta_lo']:+.4f},{d['delta_hi']:+.4f}]"
    )


def _top_marginals(X: np.ndarray, names: list[str], y: np.ndarray, k: int = 4) -> str:
    vals = [(name, _signfree_auroc(X[:, i], y)) for i, name in enumerate(names)]
    vals.sort(key=lambda kv: -abs(kv[1] - 0.5) if np.isfinite(kv[1]) else -1.0)
    return ", ".join(f"{name}={auc:.3f}" for name, auc in vals[:k])


def _analyze_one(path: Path, repeats: int, folds: int, seed: int) -> str:
    d = np.load(path, allow_pickle=True)
    y = np.asarray(d["y"], dtype=np.int32)
    Xnull = np.asarray(d["Xnull"], dtype=np.float64)
    Xroute = np.asarray(d["Xroute"], dtype=np.float64)
    Xfric = np.asarray(d["Xfric"], dtype=np.float64)
    Xbenign = np.asarray(d["Xbenign"], dtype=np.float64) if "Xbenign" in d.files else None
    Xnorm, norm_names, Xtrim, trim_names = _budget_features(d)
    base = np.column_stack([Xnull, Xroute])
    base_norm = np.column_stack([base, Xnorm])
    base_trim = np.column_stack([base, Xtrim])

    meta = json.loads(str(d["metadata"])) if "metadata" in d.files else {}
    model = meta.get("model_slug", path.stem).replace("mlx-community/", "")
    schema = meta.get("schema", "unknown")

    lines = [f"\n=== {model} ({schema}) ==="]
    lines.append(f"budget_norm marginals: {_top_marginals(Xnorm, norm_names, y)}")
    lines.append(f"budget_trim marginals: {_top_marginals(Xtrim, trim_names, y)}")

    tests: list[tuple[str, dict[str, float]]] = []
    tests.append(("friction | null+route", _repeated_cv_delta(base, np.column_stack([base, Xfric]), y, repeats, folds, seed)))
    tests.append(("budget_norm | null+route", _repeated_cv_delta(base, base_norm, y, repeats, folds, seed)))
    tests.append(("budget_trim | null+route", _repeated_cv_delta(base, base_trim, y, repeats, folds, seed)))
    tests.append(("friction | null+route+budget_norm", _repeated_cv_delta(base_norm, np.column_stack([base_norm, Xfric]), y, repeats, folds, seed)))
    tests.append(("friction | null+route+budget_trim", _repeated_cv_delta(base_trim, np.column_stack([base_trim, Xfric]), y, repeats, folds, seed)))
    tests.append(("budget_norm | null+route+friction", _repeated_cv_delta(np.column_stack([base, Xfric]), np.column_stack([base, Xfric, Xnorm]), y, repeats, folds, seed)))
    if Xbenign is not None:
        tests.append(("same-Delta benign | null+route", _repeated_cv_delta(base, np.column_stack([base, Xbenign]), y, repeats, folds, seed)))
        tests.append(("friction | null+route+same-Delta", _repeated_cv_delta(np.column_stack([base, Xbenign]), np.column_stack([base, Xbenign, Xfric]), y, repeats, folds, seed)))

    for label, res in tests:
        lines.append(_line(label, res))

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", help="residual_friction_features.npz dump(s)")
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--cv-repeats", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    for p in args.npz:
        print(_analyze_one(Path(p), args.cv_repeats, args.folds, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""v7 attention-route diagnostic over v6 projection-veto feature dumps.

This is OFFLINE: it consumes `projection_veto_features_v1` dumps and asks
whether the attention write alone points toward the YES/NO answer contrast before
the MLP responds.

The dump already contains per-layer:

    pa   = <u_NO-YES, a_L>
    pm   = <u_NO-YES, m_L>
    pnet = <u_NO-YES, a_L + m_L>

Positive values point toward NO / contradiction; negative values point toward
YES / entailment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pilot_residual_friction import _repeated_cv_delta, _signfree_auroc  # noqa: E402


def _fmt(v: float) -> str:
    return f"{v:+.4f}" if np.isfinite(v) else "nan"


def _auc(v: np.ndarray, y: np.ndarray) -> float:
    return _signfree_auroc(np.asarray(v, dtype=np.float64), y)


def _safe_sign(x: np.ndarray, deadzone: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.int8)
    out[x > deadzone] = 1
    out[x < -deadzone] = -1
    return out


def _third_means(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[1]
    a = max(1, n // 3)
    b = max(a + 1, (2 * n) // 3)
    return (
        np.nanmean(X[:, :a], axis=1),
        np.nanmean(X[:, a:b], axis=1),
        np.nanmean(X[:, b:], axis=1),
    )


def _signed_max_abs(X: np.ndarray) -> np.ndarray:
    rows = np.arange(X.shape[0])
    cols = np.nanargmax(np.abs(X), axis=1)
    return X[rows, cols]


def _load_dump(path: Path) -> Dict[str, object]:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["metadata"]))
    if meta.get("schema") != "projection_veto_features_v1":
        raise ValueError(f"{path}: expected projection_veto_features_v1, got {meta.get('schema')!r}")
    names = [str(x) for x in d["layer_feature_names"]]
    idx = {name: i for i, name in enumerate(names)}
    for required in ("pa", "pm", "pnet", "na"):
        if required not in idx:
            raise ValueError(f"{path}: missing layer feature {required!r}")
    LT = np.asarray(d["layer_tensor"], dtype=np.float64)
    return {
        "path": path,
        "meta": meta,
        "y": np.asarray(d["y"], dtype=np.int32),
        "base": np.column_stack([np.asarray(d["Xnull"], dtype=np.float64),
                                 np.asarray(d["Xroute"], dtype=np.float64)]),
        "pa": LT[:, :, idx["pa"]],
        "pm": LT[:, :, idx["pm"]],
        "pnet": LT[:, :, idx["pnet"]],
        "na": LT[:, :, idx["na"]],
    }


def _features(pa: np.ndarray, pm: np.ndarray, pnet: np.ndarray, na: np.ndarray, deadzone: float):
    pa_early, pa_mid, pa_late = _third_means(pa)
    pm_early, pm_mid, pm_late = _third_means(pm)
    pn_early, pn_mid, pn_late = _third_means(pnet)

    Xattn = np.column_stack([
        np.nanmean(pa, axis=1),
        pa_early,
        pa_mid,
        pa_late,
        _signed_max_abs(pa),
    ])
    attn_names = [
        "mean_pa",
        "early_mean_pa",
        "mid_mean_pa",
        "late_mean_pa",
        "signed_max_abs_pa",
    ]

    Xattn_abs = np.column_stack([
        np.nanmean(np.abs(pa), axis=1),
        np.nanmax(np.abs(pa), axis=1),
        np.nanmean(na, axis=1),
        np.nanmean(np.abs(pa) / np.maximum(na, 1e-12), axis=1),
    ])
    attn_abs_names = [
        "mean_abs_pa",
        "max_abs_pa",
        "mean_na",
        "mean_abs_pa_over_na",
    ]

    Xmlp = np.column_stack([
        np.nanmean(pm, axis=1),
        pm_early,
        pm_mid,
        pm_late,
        _signed_max_abs(pm),
    ])
    mlp_names = [
        "mean_pm",
        "early_mean_pm",
        "mid_mean_pm",
        "late_mean_pm",
        "signed_max_abs_pm",
    ]

    Xfinal = np.column_stack([
        np.nanmean(pnet, axis=1),
        pn_early,
        pn_mid,
        pn_late,
        _signed_max_abs(pnet),
    ])
    final_names = [
        "mean_pnet",
        "early_mean_pnet",
        "mid_mean_pnet",
        "late_mean_pnet",
        "signed_max_abs_pnet",
    ]

    sa = _safe_sign(pa, deadzone)
    sm = _safe_sign(pm, deadzone)
    sn = _safe_sign(pnet, deadzone)
    oppose = (sa != 0) & (sm != 0) & (sa != sm)
    final_follows_m = oppose & (sn == sm)
    final_follows_a = oppose & (sn == sa)
    signed_margin = np.where(oppose, np.abs(pm) - np.abs(pa), 0.0)
    Xoverride = np.column_stack([
        np.nanmean(-pa * pm, axis=1),
        np.nanmean(oppose.astype(np.float64), axis=1),
        np.nanmean(final_follows_m.astype(np.float64), axis=1),
        np.nanmean(final_follows_a.astype(np.float64), axis=1),
        np.nanmean(signed_margin, axis=1),
    ])
    override_names = [
        "mean_projection_conflict",
        "frac_attn_mlp_oppose",
        "frac_final_follows_mlp",
        "frac_final_follows_attn",
        "mean_override_margin",
    ]
    return {
        "attn": (Xattn, attn_names),
        "attn_abs": (Xattn_abs, attn_abs_names),
        "mlp": (Xmlp, mlp_names),
        "final": (Xfinal, final_names),
        "override": (Xoverride, override_names),
    }


def _line(label: str, d: dict[str, float]) -> str:
    return (
        f"    {label:<46s} aucA={d['auc_a']:.3f} aucB={d['auc_b']:.3f} "
        f"Delta={_fmt(d['delta_median'])} [{_fmt(d['delta_lo'])},{_fmt(d['delta_hi'])}]"
        f" (r={d['n_repeats']})"
    )


def _print_marginal(title: str, X: np.ndarray, names: Iterable[str], y: np.ndarray) -> None:
    print(f"\n  marginal sign-free AUROC: {title}")
    for name, col in zip(names, X.T):
        print(f"    {name:<30s} {_auc(col, y):.3f}")


def analyze(path: Path, repeats: int, folds: int, seed: int, deadzone: float) -> None:
    dump = _load_dump(path)
    y = dump["y"]
    base = dump["base"]
    pa = dump["pa"]
    pm = dump["pm"]
    pnet = dump["pnet"]
    feats = _features(pa, pm, pnet, dump["na"], deadzone)
    Xattn, attn_names = feats["attn"]
    Xattn_abs, attn_abs_names = feats["attn_abs"]
    Xmlp, mlp_names = feats["mlp"]
    Xfinal, final_names = feats["final"]
    Xoverride, override_names = feats["override"]

    meta = dump["meta"]
    contrast = meta.get("contrast", {})
    model = meta.get("model_slug", path.name)
    layer_span = f"[{meta.get('layer_lo_index')},{meta.get('layer_hi_index')})"
    print(f"\n{'=' * 78}\n  v7 attention-route: {model}\n{'=' * 78}")
    print(f"  dump: {path}")
    print(f"  n={len(y)} pos={int((y == 1).sum())} neg={int((y == 0).sum())}")
    print(f"  layer window: {layer_span}")
    print(f"  contrast: {contrast.get('contrast')}  positive=NO/contradiction")

    _print_marginal("attention route", Xattn, attn_names, y)
    _print_marginal("attention absolute budget", Xattn_abs, attn_abs_names, y)
    _print_marginal("MLP/readout response", Xmlp, mlp_names, y)
    _print_marginal("final projected update", Xfinal, final_names, y)
    _print_marginal("override descriptors", Xoverride, override_names, y)

    base_attn = np.column_stack([base, Xattn])
    base_attn_abs = np.column_stack([base, Xattn_abs])
    base_attn_budget = np.column_stack([base, Xattn, Xattn_abs])
    print("\n  incremental cross-fit OOF AUROC - repeated-CV split-sensitivity:")
    stats = {
        "attention-route | null+route-size": _repeated_cv_delta(
            base, np.column_stack([base, Xattn]), y, repeats, folds, seed),
        "abs-attn-budget | null+route-size": _repeated_cv_delta(
            base, base_attn_abs, y, repeats, folds, seed),
        "attention-route | null+route-size+abs-attn": _repeated_cv_delta(
            base_attn_abs, base_attn_budget, y, repeats, folds, seed),
        "MLP-readout | null+route-size+attention": _repeated_cv_delta(
            base_attn, np.column_stack([base_attn, Xmlp]), y, repeats, folds, seed),
        "final-readout | null+route-size+attention": _repeated_cv_delta(
            base_attn, np.column_stack([base_attn, Xfinal]), y, repeats, folds, seed),
        "override | null+route-size+attention+abs-attn": _repeated_cv_delta(
            base_attn_budget, np.column_stack([base_attn_budget, Xoverride]), y, repeats, folds, seed),
    }
    yshuf = y.copy()
    np.random.default_rng(seed + 17).shuffle(yshuf)
    stats["shuffled-labels attention-route"] = _repeated_cv_delta(
        base, np.column_stack([base, Xattn]), yshuf, repeats, folds, seed)
    for label, stat in stats.items():
        print(_line(label, stat))

    raw = stats["attention-route | null+route-size"]["delta_median"]
    abs_floor = stats["abs-attn-budget | null+route-size"]["delta_median"]
    signed_after_abs = stats["attention-route | null+route-size+abs-attn"]["delta_median"]
    override = stats["override | null+route-size+attention+abs-attn"]["delta_median"]
    print("\n  v7 route summary:")
    print(f"    signed attention-route raw Delta       {_fmt(raw)}")
    print(f"    absolute attention budget floor        {_fmt(abs_floor)}")
    print(f"    signed attention after abs budget      {_fmt(signed_after_abs)}")
    print(f"    override after route+abs budget        {_fmt(override)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dumps", nargs="+", help="projection_veto_features_v1 .npz files")
    ap.add_argument("--cv-repeats", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--deadzone", type=float, default=1e-8)
    args = ap.parse_args()
    for p in args.dumps:
        analyze(Path(p), args.cv_repeats, args.folds, args.seed, args.deadzone)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

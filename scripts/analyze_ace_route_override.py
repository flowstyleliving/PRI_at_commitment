#!/usr/bin/env python3
"""v8 ACE-route × MLP/readout override screen.

This runner combines:

  1. A sealed ACE profile, rescored per prompt at t=0.
  2. A v6 projection-veto feature dump containing p_a, p_m, p_net and controls.

It asks whether MLP/final/override features add beyond the sealed ACE route, and
whether projection-veto survives the same projection-budget / same-Delta floors
that deflated v6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pilot_residual_friction import _repeated_cv_delta, _signfree_auroc  # noqa: E402
from pri_detector import Detector  # noqa: E402


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


def _projection_blocks(d: np.lib.npyio.NpzFile, deadzone: float):
    names = [str(x) for x in d["layer_feature_names"]]
    idx = {name: i for i, name in enumerate(names)}
    for required in ("pa", "pm", "pnet"):
        if required not in idx:
            raise ValueError(f"projection dump missing layer feature {required!r}")
    LT = np.asarray(d["layer_tensor"], dtype=np.float64)
    pa = LT[:, :, idx["pa"]]
    pm = LT[:, :, idx["pm"]]
    pnet = LT[:, :, idx["pnet"]]

    pm_early, pm_mid, pm_late = _third_means(pm)
    pn_early, pn_mid, pn_late = _third_means(pnet)
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
    return (Xmlp, mlp_names), (Xfinal, final_names), (Xoverride, override_names)


def _load_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    prompts = [str(r["prompt"]) for r in rows]
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    hashes = np.asarray([hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts], dtype="U64")
    return prompts, y, hashes


def _score_ace(
    profile: Path,
    prompts: list[str],
    prompt_hash: np.ndarray,
    out_npz: Path | None,
) -> np.ndarray:
    if out_npz and out_npz.exists():
        z = np.load(out_npz, allow_pickle=True)
        if "ace_score" in z and len(z["ace_score"]) == len(prompts):
            if "prompt_sha256" in z and not np.array_equal(prompt_hash, z["prompt_sha256"]):
                raise ValueError(f"cached ACE scores do not match prompts: {out_npz}")
            print(f"  loaded cached ACE scores: {out_npz}", flush=True)
            return np.asarray(z["ace_score"], dtype=np.float64)

    detector = Detector.from_profile(str(profile))
    scores = np.full(len(prompts), np.nan, dtype=np.float64)
    print(f"  scoring ACE route for {len(prompts)} prompts ...", flush=True)
    for i, prompt in enumerate(prompts):
        try:
            scores[i] = float(detector.score(prompt))
        except Exception as exc:
            print(f"    sample {i}: ACE score failed: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(prompts):
            print(f"    [{i+1}/{len(prompts)}]", flush=True)
    if out_npz:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": "ace_route_scores_v1",
            "profile": str(profile),
            "n_samples": int(len(prompts)),
        }
        np.savez_compressed(
            out_npz,
            ace_score=scores,
            prompt_sha256=prompt_hash,
            metadata=json.dumps(meta, sort_keys=True),
        )
        print(f"  wrote cached ACE scores: {out_npz}", flush=True)
    return scores


def _line(label: str, d: dict[str, float]) -> str:
    return (
        f"    {label:<50s} aucA={d['auc_a']:.3f} aucB={d['auc_b']:.3f} "
        f"Delta={_fmt(d['delta_median'])} [{_fmt(d['delta_lo'])},{_fmt(d['delta_hi'])}]"
        f" (r={d['n_repeats']})"
    )


def _print_marginal(title: str, X: np.ndarray, names: Iterable[str], y: np.ndarray) -> None:
    print(f"\n  marginal sign-free AUROC: {title}")
    for name, col in zip(names, X.T):
        print(f"    {name:<30s} {_auc(col, y):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True, help="sealed ACE profile JSON")
    ap.add_argument("--data", default="experiments/v4-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl")
    ap.add_argument("--projection-dump", required=True, help="v6 projection_veto_features_v1 .npz")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--cv-repeats", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--deadzone", type=float, default=1e-8)
    args = ap.parse_args()

    profile = Path(args.profile)
    data_path = Path(args.data)
    dump_path = Path(args.projection_dump)
    out_dir = Path(args.out_dir) if args.out_dir else None

    prompts, y, prompt_hash = _load_rows(data_path)
    d = np.load(dump_path, allow_pickle=True)
    meta = json.loads(str(d["metadata"]))
    if meta.get("schema") != "projection_veto_features_v1":
        raise ValueError(f"{dump_path}: expected projection_veto_features_v1")
    if not np.array_equal(y, np.asarray(d["y"], dtype=np.int32)):
        raise ValueError("label vector mismatch between data and projection dump")
    if "prompt_sha256" in d and not np.array_equal(prompt_hash, d["prompt_sha256"]):
        raise ValueError("prompt hash mismatch between data and projection dump")

    ace_cache = (out_dir / "ace_scores_qwen25.npz") if out_dir else None
    ace = _score_ace(profile, prompts, prompt_hash, ace_cache)
    Xace = ace[:, None]
    finite = np.isfinite(ace)
    if finite.mean() < 0.98:
        raise RuntimeError(f"only {finite.mean():.1%} finite ACE scores")

    Xnull = np.asarray(d["Xnull"], dtype=np.float64)
    Xroute = np.asarray(d["Xroute"], dtype=np.float64)
    Xproj = np.asarray(d["Xproj"], dtype=np.float64)
    Xsame = np.asarray(d["Xsame"], dtype=np.float64)
    Xbudget = np.asarray(d["Xbudget"], dtype=np.float64)
    (Xmlp, mlp_names), (Xfinal, final_names), (Xoverride, override_names) = _projection_blocks(d, args.deadzone)

    with profile.open() as f:
        prof = json.load(f)
    det = prof.get("detector", {})
    metric = det.get("metric", {})

    print(f"\n{'=' * 78}\n  v8 ACE-route override screen\n{'=' * 78}")
    print(f"  profile: {profile}")
    print(f"  ACE cell: {metric.get('column_name')} @ step {det.get('gen_step')} sign={det.get('sign'):+d}")
    print(f"  profile AUROC={prof['calibration_stats']['auroc']:.3f} "
          f"OOB={prof['calibration_stats']['oob_auroc_median']:.3f} "
          f"[{prof['calibration_stats']['oob_auroc_ci_lo']:.3f},"
          f"{prof['calibration_stats']['oob_auroc_ci_hi']:.3f}] "
          f"stability={prof['calibration_stats']['winner_stability']:.2f}")
    print(f"  projection dump: {dump_path}")
    print(f"  n={len(y)} pos={int((y == 1).sum())} neg={int((y == 0).sum())}")

    _print_marginal("ACE route", Xace, ["ace_signed_score"], y)
    _print_marginal("MLP/readout response", Xmlp, mlp_names, y)
    _print_marginal("final projected update", Xfinal, final_names, y)
    _print_marginal("override descriptors", Xoverride, override_names, y)

    base = np.column_stack([Xnull, Xroute])
    base_ace = np.column_stack([base, Xace])
    base_ace_budget = np.column_stack([base_ace, Xbudget])
    base_ace_same = np.column_stack([base_ace, Xsame])
    r, k, s = args.cv_repeats, args.folds, args.seed
    stats = {
        "ACE-route | null+route-size": _repeated_cv_delta(base, base_ace, y, r, k, s),
        "MLP-readout | null+route-size+ACE": _repeated_cv_delta(base_ace, np.column_stack([base_ace, Xmlp]), y, r, k, s),
        "final-readout | null+route-size+ACE": _repeated_cv_delta(base_ace, np.column_stack([base_ace, Xfinal]), y, r, k, s),
        "override | null+route-size+ACE": _repeated_cv_delta(base_ace, np.column_stack([base_ace, Xoverride]), y, r, k, s),
        "projection-veto | null+route-size+ACE": _repeated_cv_delta(base_ace, np.column_stack([base_ace, Xproj]), y, r, k, s),
        "same-Delta projection | null+route-size+ACE": _repeated_cv_delta(base_ace, base_ace_same, y, r, k, s),
        "budget | null+route-size+ACE": _repeated_cv_delta(base_ace, base_ace_budget, y, r, k, s),
        "projection-veto | null+route-size+ACE+budget": _repeated_cv_delta(base_ace_budget, np.column_stack([base_ace_budget, Xproj]), y, r, k, s),
        "projection-veto | null+route-size+ACE+same-Delta": _repeated_cv_delta(base_ace_same, np.column_stack([base_ace_same, Xproj]), y, r, k, s),
    }
    yshuf = y.copy()
    np.random.default_rng(s + 23).shuffle(yshuf)
    stats["shuffled-labels ACE-route"] = _repeated_cv_delta(base, base_ace, yshuf, r, k, s)
    stats["shuffled-labels override"] = _repeated_cv_delta(base_ace, np.column_stack([base_ace, Xoverride]), yshuf, r, k, s)

    print("\n  incremental cross-fit OOF AUROC - repeated-CV split-sensitivity:")
    for label, stat in stats.items():
        print(_line(label, stat))

    raw = stats["projection-veto | null+route-size+ACE"]["delta_median"]
    same = stats["same-Delta projection | null+route-size+ACE"]["delta_median"]
    after_budget = stats["projection-veto | null+route-size+ACE+budget"]["delta_median"]
    after_same = stats["projection-veto | null+route-size+ACE+same-Delta"]["delta_median"]
    print("\n  v8 summary:")
    print(f"    projection-veto beyond ACE raw       {_fmt(raw)}")
    print(f"    same-Delta floor beyond ACE          {_fmt(same)}")
    print(f"    raw-minus-same floor                 {_fmt(raw - same)}")
    print(f"    projection-veto after ACE+budget     {_fmt(after_budget)}")
    print(f"    projection-veto after ACE+same-Delta {_fmt(after_same)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

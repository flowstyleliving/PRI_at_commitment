#!/usr/bin/env python3
"""Score the sealed E18 + E17b gates on a v3.1 run directory.

Sealed spec (pri-v3-plan.md §Magnitude-independence, 2026-04-18 + 2026-04-23
rank-pin + 2026-04-24 three-phase amendment + 2026-04-26 J_n geometry fix):
- Analysis plane: final layer, gen_step == 1 (1-indexed commit step)
- Residualization: null_ratio_rank1_resid = null_ratio_rank1 - OLS(~ d_F_lowrank32)
  via linear regression on d_F alone (NOT logistic). Robustness check at
  d_F_topk32.
- Rank pinned at r = 1 (top-1 Fisher direction = commit direction).
- Acceptance: AUROC(null_ratio_resid) >= 0.60 with non-overlap bootstrap 95%
  CI vs 0.5 on 2-of-3 primaries. Direction pre-registered: higher ->
  contradiction.
- Bootstrap: 1000 resamples, sample-level (not token-level).

E17b head-to-head (pre-registered 2026-04-23, columns pinned 2026-04-26):
- Sealed E17b reads from null_ratio_post_rank{r} and null_ratio_raw_post_rank{r}
  (post-norm Δh on post-norm basis — geometry-consistent). Threshold:
    AUROC(null_ratio_post_rank1) - AUROC(null_ratio_raw_post_rank1) >= 0.02,
    non-overlap 95% CI on Qwen 2.5 7B. Same analysis plane, 1000 sample-level
    resamples.
- Legacy null_ratio_rank{r} columns (pre-norm Δh on post-norm basis — known
  geometric mismatch) are still read when the post-norm columns are absent
  (pre-2026-04-26 parquets). Use --columns {legacy,post,auto} to override.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

SEALED_RANK = 1
SEALED_LAYER = "final"
SEALED_STEP = 1
SEALED_BOOTSTRAP_N = 1000
SEALED_THRESHOLD = 0.60
PRIMARIES = [
    "Llama-3.2-3B-Instruct-4bit",
    "Mistral-7B-Instruct-v0.3-4bit",
    "Qwen2.5-7B-Instruct-4bit",
]
QWEN25 = "Qwen2.5-7B-Instruct-4bit"


def _column_names(geometry: str) -> dict:
    """Return the null_ratio column-name templates for the chosen geometry.

    geometry ∈ {"legacy", "post"}.
      legacy: null_ratio_rank{r} / null_ratio_raw_rank{r}
              (pre-norm Δh on post-norm basis — known mismatch, pre-2026-04-26
              parquets only carry these).
      post:   null_ratio_post_rank{r} / null_ratio_raw_post_rank{r}
              (post-norm Δh on post-norm basis — geometry-consistent; PR #11
              and later parquets emit these alongside legacy).
    """
    if geometry == "post":
        return {
            "fisher": "null_ratio_post_rank{r}",
            "raw": "null_ratio_raw_post_rank{r}",
            "label": "post-norm Δh / post-norm basis (J_n-consistent)",
        }
    return {
        "fisher": "null_ratio_rank{r}",
        "raw": "null_ratio_raw_rank{r}",
        "label": "pre-norm Δh / post-norm basis (legacy, geometric mismatch)",
    }


def _resolve_geometry(df: pd.DataFrame, requested: str) -> str:
    """Resolve --columns auto/legacy/post against actually-present columns."""
    has_post = "null_ratio_post_rank1" in df.columns
    if requested == "post":
        if not has_post:
            raise SystemExit(
                "ERROR: --columns post requested but null_ratio_post_rank1 not "
                "found in parquet. Re-run pipeline against a fix/null-ratio-"
                "post-norm-geometry build (PR #11) or use --columns legacy."
            )
        return "post"
    if requested == "legacy":
        return "legacy"
    # auto
    return "post" if has_post else "legacy"


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    reg = LinearRegression().fit(x.reshape(-1, 1), y)
    return y - reg.predict(x.reshape(-1, 1))


def _auroc_signed(score: np.ndarray, label: np.ndarray) -> tuple[float, int]:
    """AUROC with sign recovery — return (|auc - 0.5| oriented score, sign)."""
    auc = roc_auc_score(label, score)
    if auc >= 0.5:
        return float(auc), 1
    return float(1.0 - auc), -1


def _bootstrap_ci(
    score: np.ndarray,
    label: np.ndarray,
    n_resamples: int = SEALED_BOOTSTRAP_N,
    seed: int = 20260423,
    use_sign: int | None = None,
) -> tuple[float, float]:
    """Sample-level bootstrap CI for AUROC.

    If use_sign is set (from the point estimate), orient each resample to that
    sign so CIs don't flip around 0.5. This matches the pre-reg's non-overlap
    test against 0.5.
    """
    rng = np.random.default_rng(seed)
    n = len(score)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y = label[idx]
        if y.min() == y.max():
            continue  # degenerate resample
        s = score[idx]
        auc = roc_auc_score(y, s)
        if use_sign == -1:
            auc = 1.0 - auc
        aucs.append(auc)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def _score(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "contradiction",
    seed: int = 20260423,
) -> dict:
    score = df[score_col].to_numpy()
    label = df[label_col].astype(int).to_numpy()
    auc, sign = _auroc_signed(score, label)
    lo, hi = _bootstrap_ci(score, label, use_sign=sign, seed=seed)
    return {"auroc": auc, "ci": [lo, hi], "sign": sign}


def _load_model_df(run_dir: Path, model_tag: str) -> pd.DataFrame | None:
    matches = list(run_dir.glob(f"{model_tag}_results.parquet"))
    if not matches:
        return None
    df = pd.read_parquet(matches[0])
    df = df[(df["layer"] == SEALED_LAYER) & (df["gen_step"] == SEALED_STEP)].copy()
    return df.reset_index(drop=True)


def _paired_auc_delta_ci(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    label_col: str = "contradiction",
    n_resamples: int = SEALED_BOOTSTRAP_N,
    seed: int = 20260423,
) -> dict:
    """Bootstrap delta AUROC(a) - AUROC(b) with paired resampling.

    Used for E17b gate (null_ratio_rank1 vs null_ratio_raw_rank1).
    Signs are recovered per metric on the point estimate, then locked
    through the bootstrap so we're testing the magnitude head-to-head.
    """
    sa = df[col_a].to_numpy()
    sb = df[col_b].to_numpy()
    y = df[label_col].astype(int).to_numpy()
    auc_a, sign_a = _auroc_signed(sa, y)
    auc_b, sign_b = _auroc_signed(sb, y)
    delta = auc_a - auc_b
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        a = roc_auc_score(yi, sa[idx])
        b = roc_auc_score(yi, sb[idx])
        if sign_a == -1:
            a = 1.0 - a
        if sign_b == -1:
            b = 1.0 - b
        deltas.append(a - b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "auroc_a": auc_a,
        "auroc_b": auc_b,
        "delta": float(delta),
        "delta_ci": [float(lo), float(hi)],
        "sign_a": sign_a,
        "sign_b": sign_b,
    }


def _analyze_model(df: pd.DataFrame, tag: str, geometry: str = "legacy") -> dict:
    cols = _column_names(geometry)
    fisher_col_r1 = cols["fisher"].format(r=1)
    fisher_col_r32 = cols["fisher"].format(r=32)
    raw_col_r1 = cols["raw"].format(r=1)

    df = df.copy()

    # E18 primary: null_ratio_(post_)rank1 residualized against d_F_lowrank32
    resid_col_lr = f"{fisher_col_r1}_resid_lowrank32"
    resid_col_tk = f"{fisher_col_r1}_resid_topk32"
    df[resid_col_lr] = _residualize(
        df[fisher_col_r1].to_numpy(),
        df["d_F_lowrank32"].to_numpy(),
    )
    df[resid_col_tk] = _residualize(
        df[fisher_col_r1].to_numpy(),
        df["d_F_topk32"].to_numpy(),
    )

    out: dict = {"model": tag, "n": int(len(df)), "geometry": geometry}

    # Sealed E18 (rank 1, d_F=lowrank32)
    out["E18_sealed_rank1_lowrank32"] = _score(df, resid_col_lr)
    # E18 robustness (rank 1, d_F=topk32)
    out["E18_robust_rank1_topk32"] = _score(df, resid_col_tk)

    # E17 (null_bare at rank 32; rupture without residualization)
    if fisher_col_r32 in df.columns:
        out["E17_null_bare_rank32"] = _score(df, fisher_col_r32)
    out["E17_null_bare_rank1"] = _score(df, fisher_col_r1)

    # E17b HARP head-to-head — sealed columns pinned by 2026-04-26 amendment.
    # Under post geometry, this is null_ratio_post_rank1 vs null_ratio_raw_post_rank1.
    # Under legacy geometry, this is the pre-2026-04-26 reading retained for
    # forensic comparison.
    if raw_col_r1 in df.columns:
        out["E17b_null_raw_rank1"] = _score(df, raw_col_r1)
        out["E17b_head_to_head"] = _paired_auc_delta_ci(df, fisher_col_r1, raw_col_r1)
        out["E17b_columns"] = {"fisher": fisher_col_r1, "raw": raw_col_r1}
    else:
        out["E17b_null_raw_rank1"] = None
        out["E17b_head_to_head"] = None
        out["E17b_columns"] = None

    # Baselines
    out["baseline_surprise"] = _score(df, "surprise")
    out["baseline_v1_cosine"] = _score(df, "pri_v1_cosine")
    out["baseline_v2_lowrank32"] = _score(df, "pri_v2_lowrank32")
    out["baseline_v2_topk32"] = _score(df, "pri_v2_topk32")

    # E18 rank sweep (diagnostic only; sealed reading is rank 1)
    rank_sweep = []
    for r in [1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64]:
        col = cols["fisher"].format(r=r)
        if col not in df.columns:
            continue
        resid = _residualize(df[col].to_numpy(), df["d_F_lowrank32"].to_numpy())
        auc, sign = _auroc_signed(resid, df["contradiction"].astype(int).to_numpy())
        rank_sweep.append((r, float(auc), int(sign)))
    out["E18_rank_sweep_lowrank32"] = rank_sweep

    return out


def _gate_pass(entry: dict) -> bool:
    auc = entry["auroc"]
    lo, _ = entry["ci"]
    return auc >= SEALED_THRESHOLD and lo > 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="run-NN dir with *_results.parquet")
    parser.add_argument("--out", default=None, help="output JSON path (default: run-dir/sealed_gate.json)")
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument(
        "--columns",
        choices=["legacy", "post", "auto"],
        default="auto",
        help=(
            "which null_ratio column family to read. "
            "'post' = null_ratio_post_rank{r} / null_ratio_raw_post_rank{r} "
            "(post-norm Δh on post-norm basis — sealed E17b spec, 2026-04-26 amendment). "
            "'legacy' = null_ratio_rank{r} / null_ratio_raw_rank{r} "
            "(pre-norm Δh on post-norm basis — known geometric mismatch, retained "
            "for forensic comparison with run-05 and earlier parquets). "
            "'auto' (default) = post if present, else legacy."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run-dir not found: {run_dir}")
        return 2

    out_path = Path(args.out) if args.out else run_dir / "sealed_gate.json"

    # Resolve geometry from the FIRST present primary's columns. All primaries in a
    # given run use the same pipeline, so they'll have the same column families.
    geometry: str | None = None
    per_model: list[dict] = []
    missing: list[str] = []
    for tag in PRIMARIES:
        df = _load_model_df(run_dir, tag)
        if df is None or df.empty:
            missing.append(tag)
            continue
        if geometry is None:
            geometry = _resolve_geometry(df, args.columns)
        per_model.append(_analyze_model(df, tag, geometry=geometry))
    if geometry is None:
        geometry = args.columns if args.columns != "auto" else "legacy"
    cols_label = _column_names(geometry)["label"]

    # Summary header
    print("=" * 84)
    print(f"SEALED E18 + E17b GATE — run_dir = {run_dir}")
    print(f"  rank=1  layer={SEALED_LAYER}  step={SEALED_STEP}  "
          f"bootstrap_n={SEALED_BOOTSTRAP_N}  seed={args.seed}")
    print(f"  columns={geometry}  ({cols_label})")
    print("=" * 84)
    if missing:
        print(f"MISSING PRIMARIES (skipped): {', '.join(missing)}")
        print()

    # E18 per-model
    print(f"{'Model':<36}  {'AUROC':>8}  {'95% CI':<20}  {'sign':>4}  {'Gate':<5}")
    print("-" * 84)
    passes = 0
    for m in per_model:
        e = m["E18_sealed_rank1_lowrank32"]
        gate = "PASS" if _gate_pass(e) else "FAIL"
        if gate == "PASS":
            passes += 1
        ci_s = f"[{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]"
        print(f"{m['model']:<36}  {e['auroc']:>8.4f}  {ci_s:<20}  {e['sign']:>4}  {gate:<5}")

    print("-" * 84)
    n_present = len(per_model)
    bar_met = passes >= 2 and n_present == 3
    verdict = "SEALED E18 GATE: PASS" if bar_met else "SEALED E18 GATE: INCOMPLETE/FAIL"
    print(f"{verdict}  ({passes}/{n_present} primaries cleared AUROC>=0.60 w/ CI>0.5)")
    print()

    # E17b Qwen 2.5 head-to-head
    qwen_entry = next((m for m in per_model if m["model"] == QWEN25), None)
    if qwen_entry is not None and qwen_entry.get("E17b_head_to_head") is not None:
        h2h = qwen_entry["E17b_head_to_head"]
        ci_s = f"[{h2h['delta_ci'][0]:.3f}, {h2h['delta_ci'][1]:.3f}]"
        e17b_pass = h2h["delta"] >= 0.02 and h2h["delta_ci"][0] > 0.0
        print("E17b (Qwen 2.5):")
        print(f"  null_ratio_rank1      : {h2h['auroc_a']:.4f} (sign {h2h['sign_a']})")
        print(f"  null_ratio_raw_rank1  : {h2h['auroc_b']:.4f} (sign {h2h['sign_b']})")
        print(f"  delta                 : {h2h['delta']:+.4f}  95% CI {ci_s}")
        print(f"  E17b gate             : {'PASS' if e17b_pass else 'FAIL'}  "
              f"(need delta>=0.02 w/ CI>0)")
        print()

    # Baselines summary
    print("Baselines (AUROC at rank-1 plane):")
    print(f"  {'Model':<36}  {'surprise':>9}  {'v1_cos':>8}  {'v2_topk32':>10}  {'v2_lr32':>9}")
    for m in per_model:
        print(
            f"  {m['model']:<36}  "
            f"{m['baseline_surprise']['auroc']:>9.4f}  "
            f"{m['baseline_v1_cosine']['auroc']:>8.4f}  "
            f"{m['baseline_v2_topk32']['auroc']:>10.4f}  "
            f"{m['baseline_v2_lowrank32']['auroc']:>9.4f}"
        )
    print()

    # Full JSON dump
    payload = {
        "run_dir": str(run_dir),
        "sealed_spec": {
            "rank": SEALED_RANK,
            "layer": SEALED_LAYER,
            "step": SEALED_STEP,
            "bootstrap_n": SEALED_BOOTSTRAP_N,
            "threshold": SEALED_THRESHOLD,
            "d_F": "lowrank32",
            "residualization": "linear",
            "geometry": geometry,
            "columns_label": cols_label,
            "fisher_col_template": _column_names(geometry)["fisher"],
            "raw_col_template": _column_names(geometry)["raw"],
        },
        "seed": args.seed,
        "per_model": per_model,
        "missing": missing,
        "e18_gate": {
            "passes": passes,
            "n_primaries": n_present,
            "verdict": "PASS" if bar_met else "INCOMPLETE_OR_FAIL",
        },
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Wrote {out_path}")
    return 0 if bar_met else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

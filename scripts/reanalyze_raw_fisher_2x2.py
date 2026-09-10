#!/usr/bin/env python3
"""Raw-vs-Fisher 2x2 re-analyzer.

Tests the 2x2 decomposition of the v3 null_ratio family:

                       uncentered             centered
    raw      W_u    |  null_ratio_raw_post  | null_ratio_raw_centered_post
    sqrt(p) W_u     |  null_ratio_post      | null_ratio_centered_post
                       (sealed E18)           (v3.2 descriptive)

The three currently-emitted variants isolate the two principled corrections
the sealed metric makes against E17b raw:

    delta_sqrtp   = AUROC(post)          - AUROC(raw_post)
                    -- contribution of sqrt(p) row-weighting (state-conditioning)

    delta_center  = AUROC(centered_post) - AUROC(post)
                    -- contribution of -p p^T rank-1 correction (DC removal)

The fourth cell (centered raw -- UNIFORM W_u row-mean subtraction without
sqrt(p) row-weighting) requires a new pipeline column. See the proposed
PRIComputer.null_ratio_raw_centered_and_energy method at the bottom of this
file -- one-time model load adds the centered raw SVD basis next to the
existing raw_right_singular_vectors cache; per-sample cost is one matvec
(same shape as the existing E17b raw projection).

Until the fourth cell is regenerated, this analyzer reports the three-way
decomposition and flags the missing column with a NaN row so downstream
plotting still works.

Inputs:  v3-main-run/<date>/run-NN/<MODEL>_results.parquet  (post-2026-04-26
         schema with null_ratio_*_post_rank{r} columns + null_ratio_centered).

Analysis plane: same as the sealed gate -- (layer == 'final', gen_step == 1,
alpha == alpha_default to dedup). Bootstrap: 1000 sample-level resamples,
paired across variants so delta CIs use the same indices per resample.

Usage:
    python scripts/reanalyze_raw_fisher_2x2.py \
        --run-dir experiments/v3-main-run/2026-05-10/run-01 \
        --models Qwen2.5-7B-Instruct-4bit Llama-3.2-3B-Instruct-4bit \
                 Mistral-7B-Instruct-v0.3-4bit \
        --ranks 1 32 \
        --out experiments/_analysis/raw_fisher_2x2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEALED_LAYER = "final"
SEALED_STEP = 1
DEFAULT_ALPHA = 1.0
DEFAULT_RANKS = (1, 32)
DEFAULT_BOOTSTRAP_N = 1000
DEFAULT_SEED = 20260423

# Column families. The fourth (centered raw) is reserved for the upcoming
# pipeline addition; today it resolves to a missing-column NaN row.
VARIANTS = {
    "raw_post":             "null_ratio_raw_post_rank{r}",            # E17b
    "post":                 "null_ratio_post_rank{r}",                # sealed E18 (sqrt(p), uncentered)
    "centered_post":        "null_ratio_centered_post_rank{r}",       # v3.2 (sqrt(p), centered = Fisher)
    "raw_centered_post":    "null_ratio_raw_centered_post_rank{r}",   # PROPOSED (uniform-centered raw)
}

# Pairs we always compute a paired delta for (B - A).
PAIRS = [
    ("post",          "raw_post",      "delta_sqrtp"),    # weighting contribution
    ("centered_post", "post",          "delta_center"),   # centering contribution
    # Diagonal of the 2x2 (only meaningful once raw_centered_post is regenerated):
    ("centered_post", "raw_centered_post", "delta_sqrtp_given_centered"),
    ("raw_centered_post", "raw_post",      "delta_center_given_raw"),
]


def _signed_auroc(score: np.ndarray, label: np.ndarray) -> tuple[float, int]:
    """AUROC oriented so the reported value is >= 0.5. Returns (auc, sign).

    The sealed pre-reg orientation is "higher score -> contradiction". We
    keep that convention but record the empirical sign per variant so a
    flipped variant doesn't masquerade as failure -- a stable -1 sign means
    the geometric direction is informative but mirrored, which is itself a
    finding worth surfacing.
    """
    auc = roc_auc_score(label, score)
    if auc >= 0.5:
        return float(auc), 1
    return float(1.0 - auc), -1


def _bootstrap_paired(
    scores: dict[str, np.ndarray],
    label: np.ndarray,
    signs: dict[str, int],
    n_resamples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample-level paired bootstrap.

    Returns per-variant arrays of resampled AUROCs (length n_resamples,
    minus degenerate resamples). All variants are resampled on identical
    index draws so deltas can be computed pairwise per resample without
    contaminating the CI with independent sampling noise.
    """
    rng = np.random.default_rng(seed)
    n = len(label)
    aucs: dict[str, list[float]] = {k: [] for k in scores}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y = label[idx]
        if y.min() == y.max():
            continue
        for k, s in scores.items():
            a = roc_auc_score(y, s[idx])
            if signs[k] == -1:
                a = 1.0 - a
            aucs[k].append(a)
    return {k: np.asarray(v) for k, v in aucs.items()}


def _ci(arr: np.ndarray) -> tuple[float, float]:
    if arr.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def _load_model(run_dir: Path, model: str, alpha: float) -> pd.DataFrame | None:
    path = run_dir / f"{model}_results.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    mask = (df["layer"] == SEALED_LAYER) & (df["gen_step"] == SEALED_STEP)
    if "alpha" in df.columns:
        mask &= np.isclose(df["alpha"].to_numpy(dtype=float), alpha)
    df = df.loc[mask].reset_index(drop=True)
    if df.empty:
        return None
    return df


def _score_one(
    df: pd.DataFrame, col: str, label: np.ndarray, n_boot: int, seed: int,
) -> dict | None:
    if col not in df.columns:
        return None
    score = df[col].to_numpy(dtype=float)
    finite = np.isfinite(score)
    if not finite.all():
        score = score[finite]
        sub_label = label[finite]
    else:
        sub_label = label
    if len(np.unique(sub_label)) < 2:
        return None
    auc, sign = _signed_auroc(score, sub_label)
    # CI from per-variant resample (not paired -- paired comes later).
    rng = np.random.default_rng(seed)
    n = len(score)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = sub_label[idx]
        if y.min() == y.max():
            continue
        a = roc_auc_score(y, score[idx])
        boot.append(1.0 - a if sign == -1 else a)
    lo, hi = _ci(np.asarray(boot))
    return {
        "auroc": auc,
        "ci": [lo, hi],
        "sign": sign,
        "n": int(len(score)),
    }


def analyze_model(
    df: pd.DataFrame,
    ranks: tuple[int, ...],
    n_boot: int,
    seed: int,
) -> dict:
    """Per-rank dict: {rank: {variant: {auroc, ci, sign}, deltas: {...}}}."""
    label = df["contradiction"].astype(int).to_numpy()
    out: dict = {}
    for r in ranks:
        cols = {k: tmpl.format(r=r) for k, tmpl in VARIANTS.items()}
        variant_results: dict = {}
        present_scores: dict[str, np.ndarray] = {}
        signs: dict[str, int] = {}
        for k, col in cols.items():
            res = _score_one(df, col, label, n_boot, seed)
            variant_results[k] = res  # None if column missing
            if res is not None:
                present_scores[k] = df[col].to_numpy(dtype=float)
                signs[k] = res["sign"]

        # Paired bootstrap across all variants that share a finite mask.
        if present_scores:
            stack = np.stack(list(present_scores.values()), axis=0)
            finite_row = np.isfinite(stack).all(axis=0)
            label_f = label[finite_row]
            present_scores = {k: v[finite_row] for k, v in present_scores.items()}
            paired = _bootstrap_paired(
                present_scores, label_f, signs, n_boot, seed
            )
        else:
            paired = {}

        deltas: dict = {}
        for a, b, name in PAIRS:
            if a not in paired or b not in paired:
                deltas[name] = None
                continue
            d = paired[a] - paired[b]
            lo, hi = _ci(d)
            deltas[name] = {
                "point": float(variant_results[a]["auroc"] - variant_results[b]["auroc"]),
                "ci": [lo, hi],
                "non_overlap_zero": bool(lo > 0.0 or hi < 0.0),
            }

        out[str(r)] = {"variants": variant_results, "deltas": deltas}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--ranks", nargs="+", type=int, default=list(DEFAULT_RANKS))
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, default=None,
                   help="JSON output path. If omitted, prints to stdout only.")
    args = p.parse_args()

    payload: dict = {
        "run_dir": str(args.run_dir),
        "alpha": args.alpha,
        "ranks": args.ranks,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "per_model": {},
        "missing_columns": [],
    }

    for model in args.models:
        df = _load_model(args.run_dir, model, args.alpha)
        if df is None:
            payload["per_model"][model] = {"error": "no rows after filter"}
            continue
        # Surface which variant columns are missing so the 4th-cell story is
        # explicit, not buried in a NaN row.
        for r in args.ranks:
            for k, tmpl in VARIANTS.items():
                col = tmpl.format(r=r)
                if col not in df.columns:
                    payload["missing_columns"].append(
                        {"model": model, "variant": k, "column": col}
                    )
        payload["per_model"][model] = analyze_model(
            df, tuple(args.ranks), args.n_bootstrap, args.seed
        )

    _print_human(payload)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote: {args.out}")


def _print_human(payload: dict) -> None:
    print(f"\nrun_dir: {payload['run_dir']}")
    print(f"alpha: {payload['alpha']} | ranks: {payload['ranks']} | "
          f"bootstrap n: {payload['n_bootstrap']}\n")
    for model, body in payload["per_model"].items():
        print(f"=== {model} ===")
        if "error" in body:
            print(f"  {body['error']}\n")
            continue
        for r_str, rb in body.items():
            print(f"  rank {r_str}:")
            for k, res in rb["variants"].items():
                if res is None:
                    print(f"    {k:>22}: MISSING COLUMN")
                    continue
                sgn = "+" if res["sign"] == 1 else "-"
                print(f"    {k:>22}: AUROC = {res['auroc']:.4f}  "
                      f"CI=[{res['ci'][0]:.3f}, {res['ci'][1]:.3f}]  "
                      f"sign={sgn}  n={res['n']}")
            for dname, d in rb["deltas"].items():
                if d is None:
                    print(f"    {dname:>22}:  (missing column -- not computable)")
                    continue
                star = "  *" if d["non_overlap_zero"] else "   "
                print(f"    {dname:>22}:  delta = {d['point']:+.4f}  "
                      f"CI=[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]{star}")
        print()
    if payload["missing_columns"]:
        cells = sorted({m["variant"] for m in payload["missing_columns"]})
        print(f"NOTE: missing variants in this run: {cells}")
        print("      Regenerate with the proposed `v3_capture_raw_centered` "
              "flag to complete the 2x2 (see bottom of this file).")


# ---------------------------------------------------------------------------
# PROPOSED PRIComputer addition (paste into pri_v2_mlx_pipeline.py near the
# existing null_ratio_raw_and_energy and wire through compute_step via a
# Config.v3_capture_raw_centered flag). One-time model load adds the
# centered raw SVD basis; per-sample cost is identical to E17b raw.
# ---------------------------------------------------------------------------
#
# In OutputProjection (whichever class owns raw_right_singular_vectors):
#
#     def raw_centered_right_singular_vectors(self, max_rank):
#         """Top-k right singular vectors of (W_u - row_mean(W_u))."""
#         W = self.materialize()                          # (V, d) once, cached
#         mu = W.mean(axis=0, keepdims=True)              # uniform DC direction
#         Wc = W - mu                                     # (V, d), centered
#         # Truncated SVD; reuse the same code path as raw_right_singular_vectors
#         # but on Wc. Cache the (Vt, S, total_sigma_sq) triple identically.
#         ...
#         return (Vt_c, S_c, total_sigma_sq_c)
#
# In PRIComputer:
#
#     def null_ratio_raw_centered_and_energy(self, dh_post, rank_values):
#         """Uniform-centered raw null_ratio. Same shape as E17b raw.
#
#         Basis is (W_u - row_mean(W_u))'s top-r right singular vectors --
#         removes the DC direction (constant logit shift, softmax-invariant)
#         without state-conditioning. Sits in the 2x2 as the missing fourth
#         cell next to:
#           raw_post           = SVD(W_u)                  -- E17b
#           post               = SVD(sqrt(p) W_u)          -- sealed
#           centered_post      = SVD(sqrt(p)(W_u - p^T W_u))-- v3.2 Fisher
#           raw_centered_post  = SVD(W_u - row_mean(W_u))   -- THIS
#         """
#         # ... mirror null_ratio_raw_and_energy, swapping the basis source.
#
# In Config:  v3_capture_raw_centered: bool = False  (descriptive-only at
# introduction; flip True only after a fresh run-NN folder so the schema
# delta is auditable per the no-silent-override rule).
#
# In compute_step:  if v3_capture_raw_centered: out.update(self.null_ratio_raw_centered_and_energy(dh_post, v3_list))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v3.2 rank-swept AUROC landscape — Fisher / Raw / Centered across all ranks.

Per the audit-operating-point convention, sweep AUROC at every emitted rank
before writing [FALSIFIED]. Reports:
  * Per-rank AUROC table (rows: rank × family; columns: per model + mean)
  * Best rank per (model, family)
  * Centered − Fisher Δ landscape (positive ⇒ centered helps)
  * Per-model best/worst Δ

Usage:
    .venv/bin/python scripts/diagnostics/diagnose_v3_2_rank_sweep.py \\
        --run-dir experiments/v3-main-run/<DATE>/run-NN
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def auroc_signed(labels, scores):
    finite = np.isfinite(scores)
    if finite.sum() < 4 or len(np.unique(labels[finite])) < 2:
        return float("nan"), 0
    auc = roc_auc_score(labels[finite], scores[finite])
    return float(max(auc, 1 - auc)), 1 if auc >= 0.5 else -1


RANKS = [1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    parquet = args.run_dir / "all_results.parquet"
    if not parquet.exists():
        print(f"ERROR: {parquet} missing"); return 1
    df = pd.read_parquet(parquet)
    sealed = df[(df["gen_step"] == 1) & (df["layer"] == "final") & (df["alpha"] == 1.0)].copy()
    models = sorted(sealed["model"].unique())
    short = [m.split("/")[-1].replace("-Instruct", "").replace("-4bit", "") for m in models]

    # Build per-(rank, family) AUROC matrix
    matrix = {}
    for r in RANKS:
        for family, prefix in [("Fisher", "null_ratio_post_rank"),
                                 ("Raw", "null_ratio_raw_post_rank"),
                                 ("Centered", "null_ratio_centered_post_rank")]:
            col = f"{prefix}{r}"
            aucs = []
            for m in models:
                g = sealed[sealed["model"] == m]
                if col not in g.columns:
                    aucs.append(float("nan")); continue
                labels = g["contradiction"].astype(int).to_numpy()
                auc, _ = auroc_signed(labels, g[col].to_numpy())
                aucs.append(auc)
            matrix[(r, family)] = aucs

    # Rank-by-rank table
    print("=" * 110)
    print("RANK SWEEP — AUROC at the sealed plane (final/step=1, alpha=1.0)")
    print("=" * 110)
    header = f"\n{'rank':>4s}  {'metric':<10s}"
    for s in short:
        header += f"  {s[:9]:>9s}"
    header += f"  {'mean':>8s}"
    print(header)
    for r in RANKS:
        for family in ["Fisher", "Raw", "Centered"]:
            aucs = matrix[(r, family)]
            mean = np.nanmean(aucs)
            cells = "  ".join(f"{a:>9.4f}" for a in aucs)
            print(f"{r:>4d}  {family:<10s}  {cells}  {mean:>8.4f}")
        print()

    # Per-model headlines
    print("=" * 110)
    print("PER-MODEL HEADLINES — best rank for each family")
    print("=" * 110)
    for i, sname in enumerate(short):
        print(f"\n  {sname}:")
        for family in ["Fisher", "Raw", "Centered"]:
            ranked = [(r, matrix[(r, family)][i]) for r in RANKS]
            best_r, best_auc = max(ranked, key=lambda x: x[1] if not np.isnan(x[1]) else -1)
            print(f"    {family:<10s}  best rank={best_r:>3d}  AUROC={best_auc:.4f}")

    # Centered − Fisher landscape
    print("\n" + "=" * 110)
    print("CENTERED − FISHER  (positive ⇒ centered helps)")
    print("=" * 110)
    header = f"\n{'rank':>4s}"
    for s in short:
        header += f"  {s[:10]:>10s}"
    header += f"  {'mean':>10s}"
    print(header)
    for r in RANKS:
        deltas = [c - f for c, f in zip(matrix[(r, "Centered")], matrix[(r, "Fisher")])]
        mean = np.nanmean(deltas)
        cells = "  ".join(f"{d:>+10.4f}" for d in deltas)
        print(f"{r:>4d}  {cells}  {mean:>+10.4f}")

    # Per-model best/worst
    print("\n" + "=" * 110)
    print("PER-MODEL  best Δ (centered helps most)  vs  worst Δ (centered hurts most)")
    print("=" * 110)
    for i, sname in enumerate(short):
        deltas = [(r, matrix[(r, "Centered")][i] - matrix[(r, "Fisher")][i]) for r in RANKS]
        best_r, best_d = max(deltas, key=lambda x: x[1])
        worst_r, worst_d = min(deltas, key=lambda x: x[1])
        print(f"  {sname:<22s}  best Δ = {best_d:+.4f} (rank {best_r:>3d})  worst Δ = {worst_d:+.4f} (rank {worst_r:>3d})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

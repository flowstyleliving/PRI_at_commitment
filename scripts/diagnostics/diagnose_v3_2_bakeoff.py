#!/usr/bin/env python3
"""v3.2 three-way bake-off — Fisher / Raw / Centered AUROC at the sealed plane.

Sealed analysis plane: layer == 'final', gen_step == 1, alpha == 1.0.
Reports per-model AUROC + sign for each metric family at rank=1, plus the
three v3.2 decision criteria (Qwen 3 recovery, no regression, kl_discharged
competitiveness).

Usage:
    .venv/bin/python scripts/diagnostics/diagnose_v3_2_bakeoff.py \\
        --run-dir experiments/v3-main-run/<DATE>/run-NN

Cross-references:
  - wiki/results/v3.2-amendment.md — pre-reg + decision criteria
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


PRIMARY_METRICS = [
    ("Fisher",   "null_ratio_post_rank1"),
    ("Raw",      "null_ratio_raw_post_rank1"),
    ("Centered", "null_ratio_centered_post_rank1"),
    ("KL_total", "kl_discharged"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    parquet = args.run_dir / "all_results.parquet"
    if not parquet.exists():
        print(f"ERROR: {parquet} missing"); return 1
    df = pd.read_parquet(parquet)
    sealed = df[(df["gen_step"] == 1) & (df["layer"] == "final") & (df["alpha"] == 1.0)].copy()

    print(f"Total rows in run: {len(df)}")
    print(f"Sealed plane (final/step=1/alpha=1.0): {len(sealed)} rows")
    print("Rows per model at sealed plane:")
    for m, n in sealed.groupby("model").size().items():
        print(f"  {m.split('/')[-1]:<35s}  {n:>4d}")

    print(f"\n{'=' * 90}")
    print("v3.2 BAKE-OFF — sealed plane (final/step=1, rank=1, alpha=1.0)")
    print(f"{'=' * 90}")
    print(f"  {'Model':<26s}  {'Fisher':>14s}  {'Raw':>14s}  {'Centered':>14s}  {'KL_total':>10s}")

    results = {}
    for m, g in sealed.groupby("model"):
        short = m.split("/")[-1].replace("-Instruct", "").replace("-4bit", "")
        labels = g["contradiction"].astype(int).to_numpy()
        row = {}
        line = f"  {short:<26s}"
        for name, col in PRIMARY_METRICS:
            if col not in g.columns:
                line += f"  {'-':>14s}"
                continue
            auc, sign = auroc_signed(labels, g[col].to_numpy())
            row[name] = (auc, sign)
            sign_str = "+" if sign > 0 else "-"
            cell = f"{auc:.4f} ({sign_str})" if name != "KL_total" else f"{auc:.4f}"
            line += f"  {cell:>14s}" if name != "KL_total" else f"  {cell:>10s}"
        print(line)
        results[short] = row

    # Decision criterion 1
    print(f"\n{'=' * 90}\nDECISION CRITERION 1 — Qwen 3 recovery\n{'=' * 90}")
    for short, row in results.items():
        if "Qwen3" in short:
            f_auc, _ = row.get("Fisher", (float("nan"), 0))
            c_auc, _ = row.get("Centered", (float("nan"), 0))
            delta = c_auc - f_auc
            verdict = ("PASS — promote to v3.3 sealed candidate" if delta >= 0.10 else
                       "PARTIAL — investigate other geometric corrections" if delta > 0.02 else
                       "FAIL at sealed step=1 (audit operating point before declaring falsified)")
            print(f"  {short}:  Δ = {c_auc:.4f} − {f_auc:.4f} = {delta:+.4f}  →  {verdict}")

    # Decision criterion 2
    print(f"\n{'=' * 90}\nDECISION CRITERION 2 — no regression on Fisher-wins primaries\n{'=' * 90}")
    for short in ["Llama-3.2-3B", "Mistral-7B-v0.3", "Qwen2.5-7B"]:
        if short in results:
            f_auc, _ = results[short].get("Fisher", (float("nan"), 0))
            c_auc, _ = results[short].get("Centered", (float("nan"), 0))
            delta = c_auc - f_auc
            flag = "OK" if delta >= -0.05 else "REGRESSION"
            print(f"  {short:<22s}  Δ = {delta:+.4f}  ({flag})")

    # Decision criterion 3
    print(f"\n{'=' * 90}\nDECISION CRITERION 3 — kl_discharged vs Fisher rank-r\n{'=' * 90}")
    for short, row in results.items():
        f_auc, _ = row.get("Fisher", (float("nan"), 0))
        kl_auc, _ = row.get("KL_total", (float("nan"), 0))
        delta = kl_auc - f_auc
        flag = "competitive" if abs(delta) <= 0.05 else "diverges"
        print(f"  {short:<22s}  Δ(kl_discharged − Fisher) = {delta:+.4f}  ({flag})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

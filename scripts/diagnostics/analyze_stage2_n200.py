#!/usr/bin/env python3
"""Stage 2 N=200 analyzer: per-rank Δ AUROC = AUROC(nr_fisher_jn_rN) − AUROC(nr_raw_jn_rN),
with paired bootstrap CI (1000 resamples, seed 20260423).

Same metric used in the 2026-04-25 overnight summary so the N=200 numbers are
like-for-like comparable to the N=100-buggy snapshot in wiki/log.md.

Usage:
    .venv/bin/python scripts/analyze_stage2_n200.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RANKS = [1, 2, 3, 4, 5, 8, 13, 16, 21, 32]
BOOT = 1000
SEED = 20260423
DIAG_DIR = Path(__file__).resolve().parent.parent / "experiments" / "v3-main-run" / "2026-04-24"

MODELS = [
    ("Llama 3.2 3B", "norm_diagnostic_Llama-3.2-3B-Instruct-4bit.csv"),
    ("Mistral 7B", "norm_diagnostic_Mistral-7B-Instruct-v0.3-4bit.csv"),
    ("Qwen 2.5 7B", "norm_diagnostic_Qwen2.5-7B-Instruct-4bit.csv"),
    ("Qwen3 8B", "norm_diagnostic_Qwen3-8B-4bit.csv"),
]


def auc(y, s):
    if np.unique(y).size < 2 or np.all(np.isnan(s)):
        return np.nan
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(s)
    if mask.sum() < 4:
        return np.nan
    return roc_auc_score(y[mask], s[mask])


def delta_with_ci(df, fisher_col, raw_col, label_col="contradiction", oriented=True):
    """If oriented=True (matches scripts/overnight_summary.py), each AUROC is
    flipped to max(auc, 1-auc) so basis direction is treated as a free sign;
    Δ then measures separability irrespective of which class scores higher.
    If oriented=False, Δ = AUROC(F) − AUROC(R) directly (sign carries direction).
    """
    y = df[label_col].astype(int).to_numpy()
    sf = df[fisher_col].to_numpy(dtype=float)
    sr = df[raw_col].to_numpy(dtype=float)
    m = np.isfinite(sf) & np.isfinite(sr)
    y, sf, sr = y[m], sf[m], sr[m]

    af = auc(y, sf); ar = auc(y, sr)
    if oriented:
        sign_f = 1 if af >= 0.5 else -1
        sign_r = 1 if ar >= 0.5 else -1
        af_o = af if sign_f == 1 else 1 - af
        ar_o = ar if sign_r == 1 else 1 - ar
    else:
        sign_f = sign_r = 1
        af_o, ar_o = af, ar
    point = af_o - ar_o

    rng = np.random.default_rng(SEED)
    n = len(y)
    deltas = np.empty(BOOT)
    for b in range(BOOT):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            deltas[b] = np.nan
            continue
        afi = auc(y[idx], sf[idx])
        ari = auc(y[idx], sr[idx])
        if oriented:
            afi = afi if sign_f == 1 else 1 - afi
            ari = ari if sign_r == 1 else 1 - ari
        deltas[b] = afi - ari
    deltas = deltas[np.isfinite(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, lo, hi, af, ar, sign_f, sign_r


def fmt(v, lo, hi):
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v):.3f} [{lo:+.2f},{hi:+.2f}]"


def verdict_at_r1(lo, hi, threshold=0.02):
    if lo > threshold:
        return f"Fisher PASS (CI lo > +{threshold:.2f})"
    if hi < -threshold:
        return f"raw PASS (CI hi < -{threshold:.2f})"
    if lo > 0:
        return "Fisher lean (CI > 0, < +0.02 bar)"
    if hi < 0:
        return "raw lean (CI < 0, > -0.02 bar)"
    return "indeterminate"


def per_model_table(both, label):
    print(f"\n{'='*100}")
    print(f"Stage 2 N=200 — {label} (1000-bootstrap, seed={SEED})")
    print(f"Sealed E17b bar: |Δ| ≥ +0.02 with non-overlap CI")
    print(f"{'='*100}\n")
    rows = []
    for name, fname in MODELS:
        df = pd.read_csv(DIAG_DIR / fname)
        n = len(df)
        per_rank = {}
        for r in RANKS:
            fc, rc = f"nr_fisher_jn_r{r}", f"nr_raw_jn_r{r}"
            if fc in df.columns and rc in df.columns:
                per_rank[r] = delta_with_ci(df, fc, rc, oriented=both)
        best_r = max(per_rank, key=lambda r: per_rank[r][0])
        d_best = per_rank[best_r]
        d_r1 = per_rank[1]
        rows.append((name, n, best_r, d_best, d_r1, per_rank))
        print(f"{name:14s}  N={n}  AUROC(F,R)@r1=({d_r1[3]:.3f},{d_r1[4]:.3f}) signs=({d_r1[5]:+d},{d_r1[6]:+d})  best=r{best_r:<2} Δ@best={fmt(*d_best[:3])}  Δ@r1={fmt(*d_r1[:3])} → {verdict_at_r1(d_r1[1], d_r1[2])}")
    return rows


def main():
    rows_oriented = per_model_table(True, "ORIENTED Δ AUROC = max(F,1-F) − max(R,1-R) [matches overnight_summary.py / log.md]")
    rows_directed = per_model_table(False, "DIRECTED Δ AUROC = AUROC(F) − AUROC(R) [sign carries direction-of-discrimination]")

    print(f"\n{'='*100}")
    print(f"Per-rank ORIENTED Δ AUROC landscape (point estimate)")
    print(f"{'='*100}\n")
    header = "rank   " + "  ".join(f"{m[0]:>14s}" for m in MODELS)
    print(header)
    print("-" * len(header))
    for r in RANKS:
        line = f"r{r:<3}  "
        for name, fname in MODELS:
            df = pd.read_csv(DIAG_DIR / fname)
            fc, rc = f"nr_fisher_jn_r{r}", f"nr_raw_jn_r{r}"
            if fc in df.columns and rc in df.columns:
                point, _, _, _, _, _, _ = delta_with_ci(df, fc, rc, oriented=True)
                sign = "+" if point >= 0 else "−"
                line += f"        {sign}{abs(point):>5.3f}"
            else:
                line += f"  {'-':>14s}"
        print(line)

    print(f"\n{'='*100}")
    print(f"Side-by-side vs N=100-buggy (ORIENTED metric, like-for-like)")
    print(f"log.md 2026-04-25 overnight table reported these N=100-buggy values:")
    print(f"{'='*100}\n")
    n100 = {
        "Llama 3.2 3B": (+0.054, "indeterminate", 16, +0.142),
        "Mistral 7B": (-0.184, "Δ<0 with CI [-0.27,-0.11]", 21, +0.434),
        "Qwen 2.5 7B": (+0.015, "indeterminate", 5, +0.108),
        "Qwen3 8B": (+0.206, "Δ>0 with CI [+0.03,+0.39] PASS", 32, +0.390),
    }
    print(f"{'model':14s}  {'r1 N=100':<11s}  {'r1 N=200':<28s}  {'best N=100':<14s}  {'best N=200':<28s}  Δ@r1 sign-flip?")
    print("-" * 130)
    for name, n, br, db, dr1, _ in rows_oriented:
        old_pt, old_v, old_br, old_dbest = n100[name]
        flip = "→ FLIP" if (old_pt > 0) != (dr1[0] > 0) else ""
        print(f"{name:14s}  {old_pt:+.3f}      {fmt(*dr1[:3]):<28s}  r{old_br:<2} {old_dbest:+.3f}    r{br:<2} {fmt(*db[:3]):<25s}  {flip}")
    print()


if __name__ == "__main__":
    main()

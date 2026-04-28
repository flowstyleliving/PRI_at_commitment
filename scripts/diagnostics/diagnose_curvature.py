#!/usr/bin/env python3
"""Curvature diagnostic — does the divergence between Δh_post and J_n·Δh_pre
carry signal beyond what null_ratio already captures?

For each sample at the sealed plane (gen_step=1, final layer):
  D := Δh_post - J_n(h_prev)·Δh_pre   (Taylor remainder of RMSNorm)
  |D| derived via law of cosines from existing CSV columns:
      |D|² = |Δh_post|² + |Δh_jn|² - 2·|Δh_post|·|Δh_jn|·cos(jn, post)
  κ  := |D| / |Δh_pre|     (relative curvature per pre-step)
  ρ  := cos(Δh_jn, Δh_post) (linearization alignment; 1.0 = no curvature)

Tests:
  - Per-class κ means (contradiction vs control)
  - κ AUROC on contradiction (directed + oriented)
  - Correlation of κ with null_ratio at sealed r=1 (jn, post, pre bases)
  - Magnitude-confound: κ vs |Δh_pre|
  - Cross-model architecture comparison: where do models land in κ-space?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DIAG_DIR = Path(__file__).resolve().parents[2] / "experiments" / "v3-main-run" / "2026-04-24"
MODELS = [
    ("Llama 3.2 3B", "norm_diagnostic_Llama-3.2-3B-Instruct-4bit.csv"),
    ("Mistral 7B", "norm_diagnostic_Mistral-7B-Instruct-v0.3-4bit.csv"),
    ("Qwen 2.5 7B", "norm_diagnostic_Qwen2.5-7B-Instruct-4bit.csv"),
    ("Qwen3 8B", "norm_diagnostic_Qwen3-8B-4bit.csv"),
]


def auc_oriented(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a), a


def compute_kappa(df):
    pre = df.dh_pre_l2.to_numpy()
    post = df.dh_post_l2.to_numpy()
    jn = df.dh_jn_l2.to_numpy()
    cos_jn_post = df.cos_dh_jn_post.to_numpy()
    D2 = post**2 + jn**2 - 2 * post * jn * cos_jn_post
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)
    kappa = D / np.maximum(pre, 1e-12)
    rho = cos_jn_post
    return D, kappa, rho


def main():
    print("\n" + "=" * 100)
    print("PER-MODEL κ AND ρ DESCRIPTIVES")
    print("=" * 100)
    rows = []
    for name, fname in MODELS:
        df = pd.read_csv(DIAG_DIR / fname)
        n = len(df)
        contr = df.contradiction.astype(bool).to_numpy()
        D, kappa, rho = compute_kappa(df)
        print(f"\n--- {name}  N={n} ---")
        print(f"  κ overall:        mean={kappa.mean():.4f}   std={kappa.std():.4f}   median={np.median(kappa):.4f}")
        print(f"  κ contradiction:  mean={kappa[contr].mean():.4f}  (n={int(contr.sum())})")
        print(f"  κ control:        mean={kappa[~contr].mean():.4f}  (n={int((~contr).sum())})")
        print(f"  ρ overall:        mean={rho.mean():.4f}   std={rho.std():.4f}")
        print(f"  ρ contradiction:  mean={rho[contr].mean():.4f}")
        print(f"  ρ control:        mean={rho[~contr].mean():.4f}")
        try:
            ka_o, ka_d = auc_oriented(contr.astype(int), kappa)
            ra_o, ra_d = auc_oriented(contr.astype(int), rho)
            print(f"  κ AUROC: directed={ka_d:.4f}  oriented={ka_o:.4f}")
            print(f"  ρ AUROC: directed={ra_d:.4f}  oriented={ra_o:.4f}")
        except Exception as e:
            print(f"  AUROC failed: {e}")
        rows.append((name, n, kappa, rho, contr, df))

    print("\n" + "=" * 100)
    print("CORRELATION WITH NULL_RATIO AT SEALED r=1 (orthogonality check)")
    print("=" * 100)
    print(f"{'model':14s}  {'corr(κ, nr_jn_r1)':>20s}  {'corr(κ, nr_post_r1)':>22s}  {'corr(κ, nr_pre_r1)':>20s}  {'corr(κ, |Δh_pre|)':>20s}")
    print("-" * 100)
    for name, n, kappa, rho, contr, df in rows:
        nr_jn = df.nr_fisher_jn_r1.to_numpy()
        nr_post = df.nr_fisher_post_r1.to_numpy()
        nr_pre = df.nr_fisher_pre_r1.to_numpy()
        pre_mag = df.dh_pre_l2.to_numpy()
        c1 = np.corrcoef(kappa, nr_jn)[0, 1]
        c2 = np.corrcoef(kappa, nr_post)[0, 1]
        c3 = np.corrcoef(kappa, nr_pre)[0, 1]
        c4 = np.corrcoef(kappa, pre_mag)[0, 1]
        print(f"{name:14s}  {c1:+20.4f}  {c2:+22.4f}  {c3:+20.4f}  {c4:+20.4f}")

    print("\n" + "=" * 100)
    print("CROSS-MODEL κ DISTRIBUTION (does architecture-dependence live in curvature?)")
    print("=" * 100)
    print(f"{'model':14s}  {'κ mean':>9s}  {'κ std':>9s}  {'κ p25':>9s}  {'κ p50':>9s}  {'κ p75':>9s}  {'ρ mean':>9s}")
    print("-" * 80)
    for name, n, kappa, rho, contr, df in rows:
        print(f"{name:14s}  {kappa.mean():9.4f}  {kappa.std():9.4f}  {np.percentile(kappa,25):9.4f}  {np.percentile(kappa,50):9.4f}  {np.percentile(kappa,75):9.4f}  {rho.mean():9.4f}")

    print("\n" + "=" * 100)
    print("κ DISCRIMINATIVE POWER vs null_ratio_post_r1 (does κ add or duplicate?)")
    print("=" * 100)
    print(f"{'model':14s}  {'κ AUROC':>10s}  {'nr_post_r1 AUROC':>18s}  {'nr_jn_r1 AUROC':>16s}  {'lift if κ adds new info':>25s}")
    print("-" * 100)
    for name, n, kappa, rho, contr, df in rows:
        y = contr.astype(int)
        ka_o, _ = auc_oriented(y, kappa)
        np_o, _ = auc_oriented(y, df.nr_fisher_post_r1.to_numpy())
        nj_o, _ = auc_oriented(y, df.nr_fisher_jn_r1.to_numpy())
        nr_post = df.nr_fisher_post_r1.to_numpy()
        # Residualize κ on null_ratio_post — does what's LEFT discriminate?
        nrm = (nr_post - nr_post.mean()) / nr_post.std()
        km = (kappa - kappa.mean()) / kappa.std()
        beta = np.sum(km * nrm) / np.sum(nrm * nrm)
        kappa_residual = km - beta * nrm
        try:
            kr_o, _ = auc_oriented(y, kappa_residual)
        except Exception:
            kr_o = np.nan
        print(f"{name:14s}  {ka_o:10.4f}  {np_o:18.4f}  {nj_o:16.4f}  κ⊥null_ratio_post AUROC = {kr_o:.4f}")


if __name__ == "__main__":
    main()

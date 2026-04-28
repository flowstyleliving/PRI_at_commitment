"""Render the v3 paper's body figures from canonical parquets.

Produces Fig 1, 2, 3, 4, 8, 9 from the figure inventory in
`wiki/paper/scaffold.md`. Output: PNG (300 dpi) under
`experiments/_analysis/paper_figures/`. All numbers trace to the n=150
powered sweep at run-09 + 2026-04-27/run-{01,02} (post-norm geometry only).

Bootstrap config: 1000 sample-level paired resamples, seed=20260423 — matches
the analyzer's sealed config.

Usage:
    PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ---- config ----------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "_analysis" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

RUN_PRIMARIES = REPO / "experiments" / "v3-main-run" / "2026-04-26" / "run-09"
RUN_PHI = REPO / "experiments" / "v3-main-run" / "2026-04-27" / "run-01"
RUN_GEMMA = REPO / "experiments" / "v3-main-run" / "2026-04-27" / "run-02"
RUN_BUGGY = REPO / "experiments" / "v3-main-run" / "2026-04-24" / "run-05"

MODELS = [
    ("Llama 3.2 3B", RUN_PRIMARIES / "Llama-3.2-3B-Instruct-4bit_results.parquet", "#1f77b4"),
    ("Mistral 7B", RUN_PRIMARIES / "Mistral-7B-Instruct-v0.3-4bit_results.parquet", "#ff7f0e"),
    ("Qwen 2.5 7B", RUN_PRIMARIES / "Qwen2.5-7B-Instruct-4bit_results.parquet", "#2ca02c"),
    ("Qwen 3 8B", RUN_PRIMARIES / "Qwen3-8B-4bit_results.parquet", "#d62728"),
    ("Phi-3.5-mini", RUN_PHI / "Phi-3.5-mini-instruct-4bit_results.parquet", "#9467bd"),
    ("Gemma 3-4B", RUN_GEMMA / "gemma-3-4b-it-4bit_results.parquet", "#8c564b"),
]
PRIMARIES = [name for name, _, _ in MODELS[:3]]
RANKS = [1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64]
SEED = 20260423
B = 1000

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---- bootstrap helpers ----------------------------------------------------

def boot_auroc(y, scores, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y); aus = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            aus.append(roc_auc_score(y[idx], scores[idx]))
        except Exception:
            pass
    a = np.array(aus)
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def boot_oriented(y, scores, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y); out = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            au = roc_auc_score(y[idx], scores[idx])
            out.append(max(au, 1 - au))
        except Exception:
            pass
    a = np.array(out)
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

def boot_h2h_oriented(y, a, b, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y); deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        ya, sa, sb = y[idx], a[idx], b[idx]
        try:
            au_a = roc_auc_score(ya, sa); o_a = max(au_a, 1 - au_a)
            au_b = roc_auc_score(ya, sb); o_b = max(au_b, 1 - au_b)
            deltas.append(o_a - o_b)
        except Exception:
            pass
    d = np.array(deltas)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def load_filtered(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[(df["layer"] == "final") & (df["gen_step"] == 1)].copy()

# ---- Fig 1: sealed E18 3/3 PASS ------------------------------------------

def fig1_sealed_e18(out: Path):
    sg = json.loads((RUN_PRIMARIES / "sealed_gate.json").read_text())
    rows = sg["per_model"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    names = [r["model"].replace("-Instruct-4bit", "").replace("-4bit", "") for r in rows]
    aurocs = [r["E18_sealed_rank1_lowrank32"]["auroc"] for r in rows]
    cis = [r["E18_sealed_rank1_lowrank32"]["ci"] for r in rows]
    err_lo = [a - c[0] for a, c in zip(aurocs, cis)]
    err_hi = [c[1] - a for a, c in zip(aurocs, cis)]
    x = np.arange(len(names))
    ax.bar(x, aurocs, yerr=[err_lo, err_hi], color="#2ca02c", alpha=0.8, capsize=6, width=0.55)
    ax.axhline(0.60, ls="--", color="black", alpha=0.6, lw=1, label="sealed threshold (0.60)")
    ax.axhline(0.50, ls=":", color="grey", alpha=0.5, lw=1, label="chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("AUROC (residualized\nnull_ratio_post_rank1)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("Fig 1 — Sealed E18 verdict: 3 of 3 primaries PASS (n=600, post-norm)",
                 pad=12)
    for xi, a, c in zip(x, aurocs, cis):
        ax.text(xi, c[1] + 0.015, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.savefig(out / "fig1_sealed_e18.png")
    plt.close(fig)
    print(f"  wrote {out / 'fig1_sealed_e18.png'}")


def _set_rank_ticks(ax):
    """Apply rotated, log-spaced rank ticks that don't squish 32/34 or 55/64."""
    # Major ticks (always shown clearly): 1, 2, 4, 8, 16, 32, 64
    # Minor ticks: 3, 5, 13, 21, 34, 55 (irregular ranks in the sweep)
    major = [1, 2, 4, 8, 16, 32, 64]
    minor = [3, 5, 13, 21, 34, 55]
    ax.set_xscale("log")
    ax.set_xlim(0.9, 75)
    ax.set_xticks(major)
    ax.set_xticklabels([str(r) for r in major], fontsize=9)
    ax.set_xticks(minor, minor=True)
    ax.set_xticklabels([str(r) for r in minor], minor=True, fontsize=7, rotation=0,
                        color="#555555")
    ax.tick_params(axis="x", which="minor", pad=14)

# ---- Fig 2: sealed E17b head-to-head on Qwen 2.5 -------------------------

def fig2_sealed_e17b(out: Path):
    sg = json.loads((RUN_PRIMARIES / "sealed_gate.json").read_text())
    qwen = next(r for r in sg["per_model"] if "Qwen2.5" in r["model"])
    h2h = qwen["E17b_head_to_head"]
    f_au = h2h["auroc_a"]; f_o = max(f_au, 1 - f_au); f_sign = h2h["sign_a"]
    r_au = h2h["auroc_b"]; r_o = max(r_au, 1 - r_au); r_sign = h2h["sign_b"]
    delta = h2h["delta"]; ci = h2h["delta_ci"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), gridspec_kw={"width_ratios": [2, 1]})
    # left: oriented AUROCs side by side
    bars = ax1.bar([0, 1], [f_o, r_o],
                   color=["#1f77b4", "#7f7f7f"], alpha=0.85, width=0.5)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([f"Fisher\nnull_ratio_post_rank1\n(sign {f_sign:+d})",
                         f"Raw\nnull_ratio_raw_post_rank1\n(sign {r_sign:+d})"])
    ax1.set_ylabel("oriented AUROC = max(AUROC, 1−AUROC)")
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(0.5, ls=":", color="grey", alpha=0.5, lw=1)
    for xi, v in zip([0, 1], [f_o, r_o]):
        ax1.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_title("Qwen 2.5 7B sealed-r=1 oriented AUROCs")

    # right: delta with CI
    ax2.bar([0], [delta], yerr=[[delta - ci[0]], [ci[1] - delta]],
            color="#2ca02c", alpha=0.85, capsize=8, width=0.4)
    ax2.axhline(0.02, ls="--", color="black", alpha=0.6, lw=1, label="sealed bar (+0.02)")
    ax2.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Δ = Fisher − Raw\n(oriented)"])
    ax2.set_ylim(-0.01, 0.25)
    ax2.set_title("Δ AUROC + 95% CI")
    ax2.text(0, ci[1] + 0.008, f"+{delta:.3f}\n[+{ci[0]:.3f}, +{ci[1]:.3f}]",
             ha="center", fontsize=9, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True, fontsize=8)
    plt.suptitle("Fig 2 — Sealed E17b head-to-head: Fisher beats Raw on Qwen 2.5 (PASS)", y=1.02)
    plt.savefig(out / "fig2_sealed_e17b.png")
    plt.close(fig)
    print(f"  wrote {out / 'fig2_sealed_e17b.png'}")

# ---- Fig 3: J_n correction effect ---------------------------------------

def fig3_jn_correction(out: Path):
    # Buggy reading from the 2026-04-24 run-05 forensic parquet — analyzer can't read it
    # post-cleanup, so we recompute here using the legacy column path directly.
    df = pd.read_parquet(RUN_BUGGY / "Qwen2.5-7B-Instruct-4bit_results.parquet")
    df = df[(df["layer"] == "final") & (df["gen_step"] == 1)].copy()
    y = df["contradiction"].astype(int).values
    f_buggy = df["null_ratio_rank1"].values  # legacy column = pre-norm Δh / post-norm basis
    r_buggy = df["null_ratio_raw_rank1"].values
    f_au_b = roc_auc_score(y, f_buggy); f_o_b = max(f_au_b, 1 - f_au_b)
    r_au_b = roc_auc_score(y, r_buggy); r_o_b = max(r_au_b, 1 - r_au_b)
    delta_b, lo_b, hi_b = boot_h2h_oriented(y, f_buggy, r_buggy)

    # Corrected reading from run-09 sealed_gate.json
    sg = json.loads((RUN_PRIMARIES / "sealed_gate.json").read_text())
    qwen = next(r for r in sg["per_model"] if "Qwen2.5" in r["model"])
    h2h = qwen["E17b_head_to_head"]
    f_au_c = h2h["auroc_a"]; f_o_c = max(f_au_c, 1 - f_au_c); f_sign_c = h2h["sign_a"]
    r_au_c = h2h["auroc_b"]; r_o_c = max(r_au_c, 1 - r_au_c); r_sign_c = h2h["sign_b"]
    delta_c, lo_c, hi_c = h2h["delta"], h2h["delta_ci"][0], h2h["delta_ci"][1]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.array([0, 1])
    deltas = np.array([delta_b, delta_c])
    err_lo = np.array([deltas[0] - lo_b, deltas[1] - lo_c])
    err_hi = np.array([hi_b - deltas[0], hi_c - deltas[1]])
    colors = ["#d62728", "#2ca02c"]
    ax.bar(x, deltas, yerr=[err_lo, err_hi], color=colors, alpha=0.85, capsize=8, width=0.45)
    ax.axhline(0.02, ls="--", color="black", alpha=0.6, lw=1, label="sealed bar (+0.02)")
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Buggy 2026-04-24\n(pre-norm Δh / post-norm basis)\nrun-05",
                        "Corrected 2026-04-27\n(post-norm Δh / post-norm basis)\nrun-09"])
    ax.set_ylabel("Δ AUROC (oriented Fisher − Raw)")
    ax.set_title("Fig 3 — J_n correction flipped the sealed E17b verdict on Qwen 2.5\n"
                 "(same data, same spec, different basis-coordinate-frame implementation)")
    for xi, d, lo, hi in zip(x, deltas, [lo_b, lo_c], [hi_b, hi_c]):
        sym = "+" if d > 0 else ""
        ax.text(xi, hi + 0.012, f"{sym}{d:.3f}\n[{lo:+.3f}, {hi:+.3f}]",
                ha="center", fontsize=9, fontweight="bold")
    # annotate verdict
    ax.text(0, lo_b - 0.04, "FAIL\n(Raw decisive)", ha="center", fontsize=10,
            color="#d62728", fontweight="bold")
    ax.text(1, hi_c + 0.06, "PASS\n(Fisher decisive)", ha="center", fontsize=10,
            color="#2ca02c", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    plt.savefig(out / "fig3_jn_correction.png")
    plt.close(fig)
    print(f"  wrote {out / 'fig3_jn_correction.png'}")

# ---- Fig 4: cross-model rank landscape ----------------------------------

def fig4_rank_landscape(out: Path):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    axes = axes.flatten()
    for ax, (name, path, color) in zip(axes, MODELS):
        df = load_filtered(path)
        y = df["contradiction"].astype(int).values
        deltas, lows, highs = [], [], []
        for r in RANKS:
            f = df[f"null_ratio_post_rank{r}"].values
            rw = df[f"null_ratio_raw_post_rank{r}"].values
            m, lo, hi = boot_h2h_oriented(y, f, rw)
            deltas.append(m); lows.append(lo); highs.append(hi)
        deltas = np.array(deltas); lows = np.array(lows); highs = np.array(highs)
        ax.fill_between(RANKS, lows, highs, color=color, alpha=0.25)
        ax.plot(RANKS, deltas, color=color, lw=2, marker="o", markersize=4)
        ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8)
        ax.axvline(1, ls=":", color="black", alpha=0.4, lw=0.8)
        _set_rank_ticks(ax)
        ax.set_ylim(-0.55, 0.55)
        ax.set_title(name, fontsize=10)
        ax.grid(True, alpha=0.3, which="major")
    for ax in axes[3:]:
        ax.set_xlabel("rank r (log scale, major + minor ticks)")
    for i, ax in enumerate(axes):
        if i % 3 == 0:
            ax.set_ylabel("oriented Δ\n(Fisher − Raw)")
    fig.suptitle("Fig 4 — Cross-architecture rank landscape (n=600, post-norm; sealed pin r=1 dotted)",
                 y=1.00, fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "fig4_rank_landscape.png")
    plt.close(fig)
    print(f"  wrote {out / 'fig4_rank_landscape.png'}")

# ---- Fig 8: Gemma 4B Motif 2 (within-model rank flip) -------------------

def fig8_gemma_rankflip(out: Path):
    df = load_filtered(RUN_GEMMA / "gemma-3-4b-it-4bit_results.parquet")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    series = [
        ("pool", df, "#8c564b", 2.5, "-"),
        ("cl=2", df[df["chain_length"] == 2], "#ff7f0e", 1.5, "--"),
        ("cl=5", df[df["chain_length"] == 5], "#2ca02c", 1.5, "--"),
    ]
    for label, sub, color, lw, ls in series:
        y = sub["contradiction"].astype(int).values
        deltas, lows, highs = [], [], []
        for r in RANKS:
            f = sub[f"null_ratio_post_rank{r}"].values
            rw = sub[f"null_ratio_raw_post_rank{r}"].values
            m, lo, hi = boot_h2h_oriented(y, f, rw)
            deltas.append(m); lows.append(lo); highs.append(hi)
        deltas = np.array(deltas); lows = np.array(lows); highs = np.array(highs)
        if label == "pool":
            ax.fill_between(RANKS, lows, highs, color=color, alpha=0.18)
        ax.plot(RANKS, deltas, color=color, lw=lw, ls=ls, marker="o",
                markersize=5 if label == "pool" else 3.5, label=label)
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8)
    ax.axvline(2.5, ls=":", color="red", alpha=0.5, lw=1.0)
    # Place the flip annotation in the empty mid-band to the right of the transition,
    # not in the upper-right where the legend sits or the upper margin where the title runs.
    ax.annotate("F → R flip\n(both strata,\nr=2 → r=3)",
                xy=(2.5, 0.0), xytext=(5.5, 0.20),
                color="red", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1, alpha=0.7),
                ha="left")
    _set_rank_ticks(ax)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel("rank r (log scale, major + minor ticks)")
    ax.set_ylabel("oriented Δ AUROC (Fisher − Raw)")
    ax.set_title("Fig 8 — Motif 2: within-model rank flip robust to chain length (Gemma 3-4B)\n"
                 "Both strata transition F → R at r=2 → r=3 — pure SVD-spectrum effect",
                 pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3, which="major")
    plt.savefig(out / "fig8_gemma_rankflip.png")
    plt.close(fig)
    print(f"  wrote {out / 'fig8_gemma_rankflip.png'}")

# ---- Fig 9: Mistral Motif 3 (chain-length × rank interaction) -----------

def fig9_mistral_simpsons(out: Path):
    df = load_filtered(RUN_PRIMARIES / "Mistral-7B-Instruct-v0.3-4bit_results.parquet")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    series = [
        ("pool", df, "#ff7f0e", 2.5, "-"),
        ("cl=2", df[df["chain_length"] == 2], "#1f77b4", 1.8, "--"),
        ("cl=5", df[df["chain_length"] == 5], "#d62728", 1.8, "--"),
    ]
    for label, sub, color, lw, ls in series:
        y = sub["contradiction"].astype(int).values
        deltas, lows, highs = [], [], []
        for r in RANKS:
            f = sub[f"null_ratio_post_rank{r}"].values
            rw = sub[f"null_ratio_raw_post_rank{r}"].values
            m, lo, hi = boot_h2h_oriented(y, f, rw)
            deltas.append(m); lows.append(lo); highs.append(hi)
        deltas = np.array(deltas); lows = np.array(lows); highs = np.array(highs)
        if label == "pool":
            ax.fill_between(RANKS, lows, highs, color=color, alpha=0.18)
        ax.plot(RANKS, deltas, color=color, lw=lw, ls=ls, marker="o",
                markersize=5 if label == "pool" else 4, label=label)
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8)
    # Simpson's-paradox vertical guides
    ax.axvline(1, ls=":", color="purple", alpha=0.4, lw=1.0)
    ax.axvline(32, ls=":", color="purple", alpha=0.4, lw=1.0)
    # Annotations placed in OPEN regions of the plot (not on top of data lines or legend).
    # Top-right is empty (data sits in mid-band); bottom-center is empty (data sits ±0.4).
    ax.annotate(
        "Simpson's site #1\nr=1: pool R, cl=2 F\n(both at non-overlap CI)",
        xy=(1, -0.14), xytext=(2.3, -0.55),
        color="purple", fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="purple", lw=1, alpha=0.7),
        ha="left",
    )
    ax.annotate(
        "Simpson's site #2 — Δ_cross = −0.575\nr=32: pool F (+0.18), cl=2 R (−0.20), cl=5 F (+0.38)\n(largest spread in 156-cell landscape)",
        xy=(32, 0.18), xytext=(4.5, 0.55),
        color="purple", fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="purple", lw=1, alpha=0.7),
        ha="left",
    )
    _set_rank_ticks(ax)
    ax.set_ylim(-0.70, 0.70)
    ax.set_xlabel("rank r (log scale, major + minor ticks)")
    ax.set_ylabel("oriented Δ AUROC (Fisher − Raw)")
    ax.set_title("Fig 9 — Motif 3: chain-length × rank interaction (Mistral 7B)\n"
                 "Two Simpson's-paradox sites — pooled verdict dissolves under stratification")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3, which="major")
    plt.savefig(out / "fig9_mistral_simpsons.png")
    plt.close(fig)
    print(f"  wrote {out / 'fig9_mistral_simpsons.png'}")

# ---- entry -----------------------------------------------------------------

def main() -> int:
    print(f"Output dir: {OUT}")
    fig1_sealed_e18(OUT)
    fig2_sealed_e17b(OUT)
    fig3_jn_correction(OUT)
    fig4_rank_landscape(OUT)
    fig8_gemma_rankflip(OUT)
    fig9_mistral_simpsons(OUT)
    print("\nDone — 6 body figures rendered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

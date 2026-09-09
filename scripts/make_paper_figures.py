"""Render the v3 paper's body figures from canonical parquets.

Produces Fig 1, 2, 3, 4, 8, 9 from the figure inventory in
`wiki/paper/scaffold.md`. Output: vector PDF (for the manuscript) plus PNG
(300 dpi, for quick viewing) under `experiments/_analysis/paper_figures/`.
All numbers trace to the n=150 powered sweep at run-09 +
2026-04-27/run-{01,02} (post-norm geometry only). Fig 3 additionally reads
2026-04-24/run-05 and 2026-04-26/run-02 for the matched n=200 J_n pair.

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
# Corrected geometry at the SAME n=200 as run-05 — the matched partner that
# isolates the J_n effect. run-09 is the later powered replication at n=600.
RUN_PRELIM = REPO / "experiments" / "v3-main-run" / "2026-04-26" / "run-02"

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
    # TrueType, not Type 3 — arXiv rejects Type-3 fonts in figures.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _save(out: Path, stem: str) -> None:
    """Emit vector PDF for the manuscript and PNG for quick viewing."""
    for ext in ("pdf", "png"):
        plt.savefig(out / f"{stem}.{ext}")

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
    # Headroom above the tallest value label so the legend can sit clear of the
    # bars; at lower right it printed on top of the Qwen 2.5 bar.
    ax.set_ylim(0.45, 1.06)
    ax.set_title("Sealed E18 verdict: 3 of 3 primaries PASS (n=600, post-norm)",
                 pad=12)
    for xi, a, c in zip(x, aurocs, cis):
        ax.text(xi, c[1] + 0.015, f"{a:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    _save(out, "fig1_sealed_e18")
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
    # Headroom for the legend above the value label; at lower right it printed
    # on top of the bar and the sealed-bar line.
    ax2.set_ylim(-0.01, 0.32)
    ax2.set_title("Δ AUROC + 95% CI")
    ax2.text(0, ci[1] + 0.008, f"+{delta:.3f}\n[+{ci[0]:.3f}, +{ci[1]:.3f}]",
             ha="center", fontsize=9, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=8)
    plt.suptitle("Sealed E17b head-to-head: Fisher beats Raw on Qwen 2.5 (PASS)", y=1.02)
    _save(out, "fig2_sealed_e17b")
    plt.close(fig)
    print(f"  wrote {out / 'fig2_sealed_e17b.png'}")

# ---- Fig 3: J_n correction effect ---------------------------------------

def _e17b_reading(run_dir: Path):
    """Read the sealed analyzer's Qwen 2.5 E17b head-to-head from a run's
    sealed_gate.json. Both readings come from the same authority the paper's
    Table 2 quotes, so the figure cannot drift from the table."""
    sg = json.loads((run_dir / "sealed_gate.json").read_text())
    qwen = next(r for r in sg["per_model"] if "Qwen2.5" in r["model"])
    h2h = qwen["E17b_head_to_head"]
    return {
        "n": qwen["n"],
        "delta": h2h["delta"],
        "lo": h2h["delta_ci"][0],
        "hi": h2h["delta_ci"][1],
    }


def fig3_jn_correction(out: Path):
    # Every bar reads the archived sealed_gate.json for its run — the same
    # authority Table 2 quotes. Recomputing the buggy bar from the legacy
    # parquet column instead produced -0.165 against the table's -0.166.
    #
    # Three readings, not two. run-05 and run-02 are the MATCHED n=200 pair
    # that isolates the J_n effect; pairing run-05 against the n=600 run-09
    # confounds the correction with a 3x sample-size increase.
    buggy = _e17b_reading(RUN_BUGGY)
    fixed = _e17b_reading(RUN_PRELIM)
    powered = _e17b_reading(RUN_PRIMARIES)
    readings = [buggy, fixed, powered]

    # Assert against the sealed analyzer output before drawing (DC house rule:
    # a figure states what the scored artifact says, or it does not ship).
    for r, want_d, want_n in [(buggy, -0.1658, 200), (fixed, 0.1495, 200),
                              (powered, 0.1571, 600)]:
        assert abs(r["delta"] - want_d) < 5e-4, r
        assert r["n"] == want_n, r

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(3)
    deltas = np.array([r["delta"] for r in readings])
    los = np.array([r["lo"] for r in readings])
    his = np.array([r["hi"] for r in readings])
    colors = ["#d62728", "#2ca02c", "#2ca02c"]
    ax.bar(x, deltas, yerr=[deltas - los, his - deltas], color=colors,
           alpha=0.85, capsize=8, width=0.55, zorder=2)
    # The powered replication is the same verdict on more data, not a third
    # condition — hatch it so it does not read as another experimental arm.
    # Sparse hatch: dense hatching cuts through the white verdict label.
    ax.patches[2].set_hatch("/")
    ax.patches[2].set_edgecolor("white")
    ax.axhline(0.02, ls="--", color="black", alpha=0.6, lw=1,
               label="sealed bar (+0.02)", zorder=1)
    ax.axhline(0, ls="-", color="grey", alpha=0.5, lw=0.8, zorder=1)

    # Fixed limits with headroom at both ends so no annotation can escape the
    # axes and overprint the title or the tick labels.
    # Top headroom clears the two-line value labels so the swing bracket can
    # sit above them; the basis is post-norm throughout and is stated in the
    # caption, so the tick labels carry only what differs between runs.
    ax.set_ylim(-0.34, 0.40)
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Buggy 2026-04-24\npre-norm $\\Delta$h\nrun-05, n={buggy['n']}",
         f"Corrected 2026-04-26\npost-norm $\\Delta$h\nrun-02, n={fixed['n']}",
         f"Powered replication\npost-norm $\\Delta$h\nrun-09, n={powered['n']}"],
        fontsize=9)
    ax.set_ylabel("Δ AUROC (oriented Fisher − Raw)")
    ax.set_title("$J_n$ correction flipped the sealed E17b verdict on Qwen 2.5\n"
                 "left pair is matched at n=200 and isolates the correction; "
                 "right bar is the powered replication",
                 fontsize=10.5, pad=10)

    # Bracket the matched pair so the comparison the paper claims is the one
    # the eye makes.
    swing = fixed["delta"] - buggy["delta"]
    ax.annotate("", xy=(0, 0.335), xytext=(1, 0.335),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.2))
    ax.text(0.5, 0.348, f"$J_n$ correction: {swing:+.3f} swing", ha="center",
            va="bottom", fontsize=9, fontweight="bold", color="#333333")

    # Value + CI outside the whisker; verdict inside the bar body. Neither can
    # collide with the other, with the title, or with the tick labels.
    for xi, d, lo, hi in zip(x, deltas, los, his):
        above = d > 0
        y = hi + 0.022 if above else lo - 0.022
        ax.text(xi, y, f"{d:+.3f}\n[{lo:+.3f}, {hi:+.3f}]", ha="center",
                va="bottom" if above else "top", fontsize=9, fontweight="bold")
    # Verdict sits in the stretch of the bar the CI whisker does not cross,
    # i.e. between the baseline and the whisker cap nearest to it.
    for xi, d, lo, hi, verdict in zip(x, deltas, los, his,
                                      ["FAIL\nRaw decisive",
                                       "PASS\nFisher decisive",
                                       "PASS\nFisher decisive"]):
        inner_cap = lo if d > 0 else hi
        ax.text(xi, inner_cap / 2, verdict, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold", zorder=3)

    # Lower right is the only quadrant free of a bar or a value label.
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    _save(out, "fig3_jn_correction")
    plt.close(fig)
    print(f"  wrote {out / 'fig3_jn_correction.png'} ("
          + ", ".join(f"{r['delta']:+.4f} n={r['n']}" for r in readings) + ")")

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
    fig.suptitle("Cross-architecture rank landscape (n=600, post-norm; sealed pin r=1 dotted)",
                 y=1.00, fontsize=11)
    plt.tight_layout()
    _save(out, "fig4_rank_landscape")
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
    ax.set_title("Motif 2: within-model rank flip robust to chain length (Gemma 3-4B)\n"
                 "Both strata transition F → R at r=2 → r=3 — pure SVD-spectrum effect",
                 pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3, which="major")
    _save(out, "fig8_gemma_rankflip")
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
    ax.set_title("Motif 3: chain-length × rank interaction (Mistral 7B)\n"
                 "Two Simpson's-paradox sites — pooled verdict dissolves under stratification")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3, which="major")
    _save(out, "fig9_mistral_simpsons")
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

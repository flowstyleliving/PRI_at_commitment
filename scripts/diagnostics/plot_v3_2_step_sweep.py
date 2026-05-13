#!/usr/bin/env python3
"""Plot the v3.2 step-sweep results — heatmap grid + best-per-model trajectory.

Reads the CSV emitted by `diagnose_v3_2_step_sweep.py` (long-format with
`model, gen_step, family, rank, n, n_min_class, auroc` columns) and produces
two PNG figures:

  1. step_sweep_heatmaps.png — 6×3 grid (rows=models, cols=families). Each
     subplot is rank × gen_step, AUROC color. Class-imbalanced cells
     (n_min_class < 20) hatched diagonally to flag selection-bias artifacts.

  2. step_sweep_best_trajectory.png — line plot. x=gen_step, y=AUROC, one
     line per model at its per-model balanced-best metric. Best peak
     annotated. Caps trajectory at last gen_step where n_min_class >= 20
     for that model so we don't draw imbalanced AUROCs as if they were
     trustworthy.

Usage:
    .venv/bin/python scripts/diagnostics/plot_v3_2_step_sweep.py \\
        --csv /tmp/v3_2_step_sweep_balanced.csv \\
        --out-dir experiments/_analysis/v3_2_step_sweep_plots

Style matches `scripts/make_paper_figures.py` — DPI 300, tight bbox, hex
palette per model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Match make_paper_figures.py per-model palette where possible
MODEL_COLORS: Dict[str, str] = {
    "Llama-3.2-3B-Instruct-4bit":     "#1f77b4",
    "Mistral-7B-Instruct-v0.3-4bit":  "#ff7f0e",
    "Qwen2.5-7B-Instruct-4bit":       "#2ca02c",
    "Qwen3-8B-4bit":                  "#9467bd",
    "Phi-3.5-mini-instruct-4bit":     "#8c564b",
    "gemma-3-4b-it-4bit":             "#e377c2",
}
MODEL_ORDER = list(MODEL_COLORS.keys())
FAMILIES = ["Fisher", "Raw", "Centered"]
RANK_ORDER = ["r=1", "r=2", "r=3", "r=4", "r=5", "r=8", "r=13",
              "r=16", "r=21", "r=32", "r=34", "r=55", "r=64"]
MIN_CLASS_THRESHOLD = 20


def short_model(m: str) -> str:
    return m.split("/")[-1]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["short"] = df["model"].map(short_model)
    return df


def heatmap_grid(df: pd.DataFrame, out_path: Path) -> None:
    """6×3 subplot grid: rows = models (in MODEL_ORDER), cols = families.
    Each subplot heatmap: rows = gen_step (1..24), cols = rank (RANK_ORDER).
    Color = AUROC; class-imbalanced cells hatched."""
    fig, axes = plt.subplots(
        len(MODEL_ORDER), len(FAMILIES),
        figsize=(13, 16),
        sharex=True, sharey=True,
    )
    if len(MODEL_ORDER) == 1:
        axes = np.array([axes])

    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(vmin=0.5, vmax=1.0)

    for i, m_short in enumerate(MODEL_ORDER):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i, j]
            sub = df[(df["short"] == m_short) & (df["family"] == fam)]
            if sub.empty:
                ax.set_visible(False)
                continue
            piv_auc = sub.pivot_table(
                index="gen_step", columns="rank", values="auroc", aggfunc="first"
            ).reindex(columns=RANK_ORDER)
            piv_imb = sub.pivot_table(
                index="gen_step", columns="rank", values="n_min_class", aggfunc="first"
            ).reindex(columns=RANK_ORDER)
            arr = piv_auc.to_numpy()
            steps = piv_auc.index.tolist()
            im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm, origin="lower")
            # Hatch class-imbalanced cells
            for r_idx, step in enumerate(steps):
                for c_idx, rank in enumerate(RANK_ORDER):
                    n_min = piv_imb.iloc[r_idx, c_idx] if c_idx < piv_imb.shape[1] else None
                    if n_min is not None and not np.isnan(n_min) and n_min < MIN_CLASS_THRESHOLD:
                        ax.add_patch(plt.Rectangle(
                            (c_idx - 0.5, r_idx - 0.5), 1, 1,
                            fill=False, hatch="///", edgecolor="black",
                            linewidth=0.0, alpha=0.6,
                        ))
            ax.set_xticks(range(len(RANK_ORDER)))
            ax.set_xticklabels([r.replace("r=", "") for r in RANK_ORDER], fontsize=7, rotation=0)
            ax.set_yticks(range(0, len(steps), 4))
            ax.set_yticklabels([str(steps[k]) for k in range(0, len(steps), 4)], fontsize=7)
            if i == 0:
                ax.set_title(fam, fontsize=11)
            if j == 0:
                ax.set_ylabel(m_short.replace("-Instruct", "").replace("-4bit", "") + "\ngen_step",
                              fontsize=8)
            if i == len(MODEL_ORDER) - 1:
                ax.set_xlabel("rank", fontsize=8)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="AUROC (sign-agnostic)")

    legend_elements = [
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label=f"class imbalance (min_class < {MIN_CLASS_THRESHOLD})"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=1,
               bbox_to_anchor=(0.5, 0.95), fontsize=9)
    fig.suptitle("v3.2 step sweep — AUROC heatmap (gen_step × rank, per family per model)",
                 fontsize=12, y=0.97)
    plt.tight_layout(rect=[0, 0, 0.91, 0.94])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved: {out_path}")


def best_trajectory(df: pd.DataFrame, out_path: Path) -> None:
    """Line plot: per-model best metric AUROC across gen_steps.
    Trims trajectories where n_min_class < threshold to avoid plotting
    selection-biased AUROCs alongside trustworthy ones."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # For each model, pick the best (family, rank) on class-balanced cells
    balanced = df[df["n_min_class"] >= MIN_CLASS_THRESHOLD].dropna(subset=["auroc"])
    annotations: List[str] = []
    for m_short in MODEL_ORDER:
        sub_bal = balanced[balanced["short"] == m_short]
        if sub_bal.empty:
            continue
        best = sub_bal.loc[sub_bal["auroc"].idxmax()]
        fam, rank = best["family"], best["rank"]
        # Now plot the FULL trajectory for that (fam, rank), but mark imbalanced steps
        full = df[(df["short"] == m_short) & (df["family"] == fam) & (df["rank"] == rank)]
        full = full.sort_values("gen_step")
        steps = full["gen_step"].to_numpy()
        aucs = full["auroc"].to_numpy()
        n_min = full["n_min_class"].to_numpy()
        bal_mask = n_min >= MIN_CLASS_THRESHOLD
        color = MODEL_COLORS[m_short]
        # Solid line on balanced steps
        ax.plot(steps[bal_mask], aucs[bal_mask],
                color=color, linewidth=2.0, marker="o", markersize=4,
                label=f"{m_short.replace('-Instruct','').replace('-4bit','')}  ({fam} {rank})")
        # Dashed continuation through imbalanced steps
        ax.plot(steps[~bal_mask], aucs[~bal_mask],
                color=color, linewidth=1.0, linestyle="--", marker="x", markersize=4,
                alpha=0.5)
        # Annotate peak (balanced only)
        if bal_mask.any():
            peak_idx = np.nanargmax(np.where(bal_mask, aucs, np.nan))
            ax.annotate(f"step {steps[peak_idx]}",
                        xy=(steps[peak_idx], aucs[peak_idx]),
                        xytext=(3, 5), textcoords="offset points",
                        fontsize=8, color=color)

    ax.set_xlabel("gen_step", fontsize=11)
    ax.set_ylabel("AUROC (sign-agnostic) at per-model best (family, rank)", fontsize=11)
    ax.set_title("v3.2 step sweep — per-model best metric trajectory across gen_steps",
                 fontsize=12)
    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)
    ax.axhline(0.9, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)
    ax.text(24.3, 0.51, "chance", fontsize=8, color="grey", ha="right", va="bottom")
    ax.text(24.3, 0.91, "0.90 bar", fontsize=8, color="grey", ha="right", va="bottom")
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.text(0.5, 0.01,
             "Solid line + dot = class-balanced (n_min_class ≥ {}). "
             "Dashed line + × = class-imbalanced step (Mistral / Gemma EOS truncation).".format(
                 MIN_CLASS_THRESHOLD),
             ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True, type=Path,
                    help="long-format CSV from diagnose_v3_2_step_sweep.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.csv)
    if "n_min_class" not in df.columns:
        print(
            "ERROR: CSV missing `n_min_class` column. "
            "Re-run `diagnose_v3_2_step_sweep.py` after the 2026-05-10 patch."
        )
        return 1

    heatmap_grid(df, args.out_dir / "step_sweep_heatmaps.png")
    best_trajectory(df, args.out_dir / "step_sweep_best_trajectory.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

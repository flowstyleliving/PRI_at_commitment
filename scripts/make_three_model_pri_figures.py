#!/usr/bin/env python3
"""
Generate the three final PRI-at-commitment figures from raw trace files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


C_CONTROL = "#2a6f97"   # blue
C_CONTRA = "#c44536"    # red
C_CORRECT = "#f4a261"   # orange


def _load_records(path: Path) -> List[dict]:
    rows: List[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pri_step(rec: dict, step: int) -> float:
    traj = (rec.get("generation_trace") or {}).get("trajectory") or []
    if len(traj) < step:
        return float("nan")
    v = float(traj[step - 1].get("pri", np.nan))
    return v if math.isfinite(v) else float("nan")


def _finite(arr: List[float]) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    return x[np.isfinite(x)]


def _mean_ci95(arr: List[float]) -> tuple[float, float]:
    x = _finite(arr)
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(x))
    if n == 1:
        return mean, 0.0
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    return mean, 1.96 * se


def _hedges_g(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan"), float("nan")
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    dof = nx + ny - 2
    if dof <= 0:
        return float("nan"), float("nan")
    sp2 = (((nx - 1) * vx) + ((ny - 1) * vy)) / dof
    if sp2 <= 0:
        return float("nan"), float("nan")
    d = (mx - my) / math.sqrt(sp2)
    j = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0))
    g = j * d
    var_g = (nx + ny) / (nx * ny) + (g * g) / (2.0 * dof)
    return float(g), float(var_g)


def _step1_g_stratified(records: List[dict]) -> float:
    gs: List[float] = []
    vars_: List[float] = []
    for chain in ("short", "long"):
        c_vals: List[float] = []
        x_vals: List[float] = []
        for r in records:
            is_short = str(r.get("cell", "")).startswith("short_")
            if (chain == "short") != is_short:
                continue
            v = _pri_step(r, 1)
            if not math.isfinite(v):
                continue
            if bool(r.get("has_contradiction", False)):
                x_vals.append(v)
            else:
                c_vals.append(v)
        g, var_g = _hedges_g(_finite(x_vals), _finite(c_vals))
        if math.isfinite(g) and math.isfinite(var_g) and var_g > 0:
            gs.append(g)
            vars_.append(var_g)
    if not gs:
        return float("nan")
    w = 1.0 / np.asarray(vars_, dtype=float)
    g_meta = float(np.sum(w * np.asarray(gs, dtype=float)) / np.sum(w))
    return g_meta


def _prefix_mean_by_offset(records: List[dict], signal: str, contradiction: bool) -> np.ndarray:
    offsets = np.arange(-5, 6, dtype=int)
    buckets: Dict[int, List[float]] = {int(o): [] for o in offsets}
    for r in records:
        if bool(r.get("has_contradiction", False)) != contradiction:
            continue
        for p in r.get("prefix_trace", []):
            rel = int(round(float(p.get("relative_offset", 0))))
            if rel < -5 or rel > 5:
                continue
            v = float(p.get(signal, np.nan))
            if math.isfinite(v):
                buckets[rel].append(v)
    out = []
    for off in offsets:
        vals = buckets[int(off)]
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return np.asarray(out, dtype=float)


def _plot_three_model_pri_step1(records_by_model: Dict[str, List[dict]], out: Path) -> None:
    models = ["Llama", "Mistral", "Qwen"]
    centers = np.arange(len(models), dtype=float)
    rng = np.random.default_rng(42)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.patch.set_facecolor("white")

    # Subplot A: paired violins + jitter
    width = 0.32
    for i, model in enumerate(models):
        records = records_by_model[model]
        control = [_pri_step(r, 1) for r in records if not bool(r.get("has_contradiction", False))]
        contra = [_pri_step(r, 1) for r in records if bool(r.get("has_contradiction", False))]
        control = _finite(control)
        contra = _finite(contra)
        pos_c = centers[i] - 0.18
        pos_x = centers[i] + 0.18

        c_parts = ax1.violinplot([control], positions=[pos_c], widths=width, showmedians=True, showextrema=False)
        for b in c_parts["bodies"]:
            b.set_facecolor(C_CONTROL)
            b.set_edgecolor("black")
            b.set_alpha(0.45)
        c_parts["cmedians"].set_color("black")

        x_parts = ax1.violinplot([contra], positions=[pos_x], widths=width, showmedians=True, showextrema=False)
        for b in x_parts["bodies"]:
            b.set_facecolor(C_CONTRA)
            b.set_edgecolor("black")
            b.set_alpha(0.45)
        x_parts["cmedians"].set_color("black")

        ax1.scatter(rng.normal(pos_c, 0.03, size=control.size), control, s=8, color=C_CONTROL, alpha=0.30, edgecolors="none")
        ax1.scatter(rng.normal(pos_x, 0.03, size=contra.size), contra, s=8, color=C_CONTRA, alpha=0.30, edgecolors="none")

        g = _step1_g_stratified(records)
        top = float(np.nanmax(np.concatenate([control, contra])))
        ax1.text(centers[i], top + 0.12, f"g = {g:.2f}", ha="center", va="bottom", fontsize=9)

    ax1.set_title("PRI at First Generated Token")
    ax1.set_ylabel("PRI step 1 value")
    ax1.set_xticks(centers)
    ax1.set_xticklabels(models)
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Subplot B: outcome bars with CI
    labels = ["Control", "Contradiction-Correct", "Contradiction-Incorrect"]
    keys = ["control", "contra_correct", "contra_incorrect"]
    colors = [C_CONTROL, C_CORRECT, C_CONTRA]
    bar_w = 0.22
    offsets = [-bar_w, 0.0, bar_w]

    for label, key, color, off in zip(labels, keys, colors, offsets):
        means = []
        cis = []
        for model in models:
            records = records_by_model[model]
            vals: List[float] = []
            for r in records:
                is_contra = bool(r.get("has_contradiction", False))
                is_correct = bool(r.get("answer_correct", False))
                if key == "control" and not is_contra:
                    vals.append(_pri_step(r, 1))
                elif key == "contra_correct" and is_contra and is_correct:
                    vals.append(_pri_step(r, 1))
                elif key == "contra_incorrect" and is_contra and (not is_correct):
                    vals.append(_pri_step(r, 1))
            m, ci = _mean_ci95(vals)
            means.append(m)
            cis.append(ci)
        x = centers + off
        ax2.bar(x, means, width=bar_w, color=color, edgecolor="black", linewidth=0.6, label=label)
        ax2.errorbar(x, means, yerr=cis, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)

    ax2.set_title("PRI Step 1 by Behavioral Outcome")
    ax2.set_ylabel("Mean PRI step 1")
    ax2.set_xticks(centers)
    ax2.set_xticklabels(models)
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_generation_pri_steps(records_by_model: Dict[str, List[dict]], out: Path) -> None:
    models = ["Llama", "Mistral", "Qwen"]
    steps = np.array([1, 2, 3], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300, sharey=True)

    for ax, model in zip(axes, models):
        records = records_by_model[model]
        ctrl_mean = []
        ctrl_lo = []
        ctrl_hi = []
        con_mean = []
        con_lo = []
        con_hi = []

        for s in [1, 2, 3]:
            c_vals = [_pri_step(r, s) for r in records if not bool(r.get("has_contradiction", False))]
            x_vals = [_pri_step(r, s) for r in records if bool(r.get("has_contradiction", False))]
            cm, cci = _mean_ci95(c_vals)
            xm, xci = _mean_ci95(x_vals)
            ctrl_mean.append(cm)
            ctrl_lo.append(cm - cci)
            ctrl_hi.append(cm + cci)
            con_mean.append(xm)
            con_lo.append(xm - xci)
            con_hi.append(xm + xci)

        ctrl_mean = np.asarray(ctrl_mean, dtype=float)
        con_mean = np.asarray(con_mean, dtype=float)
        ax.plot(steps, ctrl_mean, color=C_CONTROL, lw=2.0, marker="o", label="Control")
        ax.fill_between(steps, np.asarray(ctrl_lo), np.asarray(ctrl_hi), color=C_CONTROL, alpha=0.18)
        ax.plot(steps, con_mean, color=C_CONTRA, lw=2.0, marker="o", label="Contradiction")
        ax.fill_between(steps, np.asarray(con_lo), np.asarray(con_hi), color=C_CONTRA, alpha=0.18)

        delta1 = float(con_mean[0] - ctrl_mean[0])
        y = float(max(con_mean[0], ctrl_mean[0]))
        ax.annotate(f"Delta1={delta1:.2f}", xy=(1, y), xytext=(1.12, y + 0.08), fontsize=8, color="#333")

        g = _step1_g_stratified(records)
        ax.set_title(f"{model} (step1 g={g:.2f})")
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Generation Step")
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("PRI")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_prefix_null(records_by_model: Dict[str, List[dict]], out: Path) -> None:
    models = ["Llama", "Mistral", "Qwen"]
    signals = [("pri", "PRI"), ("delta_sigma_jsd", "delta_sigma_jsd"), ("acr_mid_mean", "acr_mid_mean")]
    offsets = np.arange(-5, 6, dtype=int)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), dpi=300, sharex=True)
    for r, model in enumerate(models):
        records = records_by_model[model]
        for c, (signal_key, title) in enumerate(signals):
            ax = axes[r, c]
            y_ctrl = _prefix_mean_by_offset(records, signal_key, contradiction=False)
            y_con = _prefix_mean_by_offset(records, signal_key, contradiction=True)
            ax.plot(offsets, y_ctrl, color=C_CONTROL, lw=1.8, label="Control")
            ax.plot(offsets, y_con, color=C_CONTRA, lw=1.8, label="Contradiction")
            ax.axvline(0, linestyle="--", color="#666", lw=1.0, alpha=0.9)
            ax.grid(axis="y", alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"{model}\nSignal Value")
            if r == 2:
                ax.set_xlabel("Relative Offset from Anchor")
            if r == 0 and c == 2:
                ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Prefix-Phase Encoding: No Differential Signal (3 Models x 3 Signals)", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the three PRI-at-commitment publication figures.")
    parser.add_argument("--results-dir", default="./results", help="Directory containing raw *_samples.jsonl.gz files.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    file_map = {
        "Llama": results_dir / "synthetic_logic_promptfix_full_llama_samples.jsonl.gz",
        "Mistral": results_dir / "synthetic_logic_promptfix_full_mistral_7b_samples.jsonl.gz",
        "Qwen": results_dir / "synthetic_logic_promptfix_full_qwen25_7b_samples.jsonl.gz",
    }
    for p in file_map.values():
        if not p.exists():
            raise FileNotFoundError(f"Missing input trace file: {p}")

    records_by_model = {k: _load_records(v) for k, v in file_map.items()}

    out1 = results_dir / "synthetic_logic_three_model_pri_step1_comparison.png"
    out2 = results_dir / "fig1_generation_pri_steps.png"
    out3 = results_dir / "fig2_prefix_null_three_model.png"

    _plot_three_model_pri_step1(records_by_model, out1)
    _plot_generation_pri_steps(records_by_model, out2)
    _plot_prefix_null(records_by_model, out3)

    print(json.dumps({"figures": [str(out1), str(out2), str(out3)]}, indent=2))


if __name__ == "__main__":
    main()

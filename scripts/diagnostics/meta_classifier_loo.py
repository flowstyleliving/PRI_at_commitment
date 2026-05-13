#!/usr/bin/env python3
"""Meta-classifier LOO-CV — v4-candidate #4 first pass.

Concatenates the N=10 model parquets (6 primaries + 4 expansion), runs the
v3.2 step sweep, and evaluates three candidate selection strategies under
leave-one-out cross-validation:

  A. universal_default — always predict (scalar, kl_discharged, gen_step=1).
     Single fixed cell, no features used. Baseline.

  B. family_rule — for held-out model M with family F, pick the
     (gen_step, family, rank) cell with highest *mean AUROC across the
     OTHER models in family F*. If F has no siblings in training set,
     fall back to universal_default. Tests "family-level generalization."

  C. nn_terminal_commit — pick the held-out model's predicted cell from
     its single nearest neighbor in feature space (terminal_commit_rate,
     mean_gen_steps). Tests "output-style-similar models share optimal
     cells."

Oracle = best cell on the held-out model itself (upper bound).

Cells are gated by min_n=50 and n_min_class=20 to drop class-imbalance
artifacts (consistent with diagnose_v3_2_step_sweep.py).

Usage:
    .venv/bin/python scripts/diagnostics/meta_classifier_loo.py \\
        [--min-n 50] [--min-n-per-class 20] \\
        [--out-json /tmp/meta_classifier_loo.json]

Cross-references:
  - wiki/v4-candidates.md §4 — meta-classifier design
  - scripts/diagnostics/diagnose_v3_2_step_sweep.py — reused sweep helpers
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

from analyze_adaptive_step import auroc_signed, detect_rank_columns  # noqa: E402
from diagnose_v3_2_step_sweep import run_sweep, short_model_name  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Data assembly — the N=10 model set
# ─────────────────────────────────────────────────────────────────────────────


RUN_PARQUETS = [
    ROOT / "experiments/v3-main-run/2026-05-08/run-01/all_results.parquet",  # 6 primaries
    ROOT / "experiments/v3-main-run/2026-05-11/run-01/all_results.parquet",  # Llama 3.1 8B
    ROOT / "experiments/v3-main-run/2026-05-11/run-02/all_results.parquet",  # Phi-4-mini
    ROOT / "experiments/v3-main-run/2026-05-11/run-04/all_results.parquet",  # Mistral-Nemo + Gemma-3-1B
    ROOT / "experiments/v3-main-run/2026-05-12/run-01/all_results.parquet",  # Qwen3-1.7B (reasoning)
]


def load_all_models() -> pd.DataFrame:
    parts = []
    for p in RUN_PARQUETS:
        if not p.exists():
            raise SystemExit(f"missing {p}")
        df = pd.read_parquet(p)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    print(f"Loaded {len(out)} rows across {out['model'].nunique()} models from {len(parts)} runs.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-model feature extraction
# ─────────────────────────────────────────────────────────────────────────────


def family_of(model_slug: str) -> str:
    s = model_slug.lower()
    if "llama" in s:
        return "Llama"
    if "mistral" in s and "dolphin" not in s:
        # Includes Mistral-Nemo (which is Mistral-family but newer generation)
        return "Mistral"
    if "qwen" in s:
        return "Qwen"
    if "phi" in s:
        return "Phi"
    if "gemma" in s:
        return "Gemma"
    return "other"


def is_reasoning_tuned(model_slug: str) -> bool:
    """Detect reasoning-tuned variants by name pattern. As of 2026-05 our N=10
    set has only Qwen3-8B in this category (no DeepSeek-R1-distill / o1-style
    yet). Detection rule: 'qwen3' OR explicit 'reasoning' / 'r1' tokens."""
    s = model_slug.lower()
    if "qwen3" in s:  # Qwen3 is reasoning-tuned by default (think-tags etc.)
        return True
    if any(t in s for t in ("-r1-", "reasoning", "thinking")):
        return True
    return False


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model observable features. Cheap — derived from results.parquet."""
    sealed = df[(df.get("layer") == "final") & (df.get("alpha") == 1.0)]
    step1 = sealed[sealed["gen_step"] == 1] if "gen_step" in sealed.columns else sealed
    rows = []
    for m, g in df.groupby("model"):
        n_samples = g["sample_id"].nunique() if "sample_id" in g.columns else len(g)
        steps_per_sample = g.groupby("sample_id")["gen_step"].max() if "sample_id" in g.columns else None
        if steps_per_sample is not None:
            mean_gen_steps = float(steps_per_sample.mean())
            terminal_commit_rate = float((steps_per_sample == 1).mean())
        else:
            mean_gen_steps = float(g["gen_step"].mean())
            terminal_commit_rate = float((g["gen_step"] == 1).mean())
        # Step-1 surprise: a cheap proxy for output-style regime.
        # Low (<0.05)  → committed-immediately (Mistral, Phi-3.5, Gemma)
        # Mid (0.1-1)  → format-prefix (Llama "Answer:", Qwen 2.5)
        # High (>1.0)  → reasoning-tuned (Qwen 3)
        s1 = step1[step1["model"] == m]
        mean_surprise_step1 = float(s1["surprise"].mean()) if len(s1) and "surprise" in s1.columns else float("nan")
        rows.append({
            "model": m,
            "short": short_model_name(m),
            "family": family_of(m),
            "is_reasoning_tuned": is_reasoning_tuned(m),
            "n_samples": n_samples,
            "mean_gen_steps": mean_gen_steps,
            "terminal_commit_rate": terminal_commit_rate,
            "mean_surprise_step1": mean_surprise_step1,
        })
    return pd.DataFrame(rows).set_index("model")


# ─────────────────────────────────────────────────────────────────────────────
# Sweep + cell-lookup helpers
# ─────────────────────────────────────────────────────────────────────────────


def filtered_sweep(
    sweep_df: pd.DataFrame, min_n: int, min_n_per_class: int
) -> pd.DataFrame:
    """Drop cells with too-few samples or class-imbalance artifacts."""
    f = sweep_df.copy()
    f = f.dropna(subset=["auroc"])
    f = f[f["n"] >= min_n]
    f = f[f["n_min_class"] >= min_n_per_class]
    return f


CellKey = Tuple[int, str, str]  # (gen_step, family, rank)


def cell_key(row) -> CellKey:
    return (int(row["gen_step"]), str(row["family"]), str(row["rank"]))


def best_cell_per_model(filt: pd.DataFrame) -> Dict[str, Tuple[CellKey, float]]:
    out: Dict[str, Tuple[CellKey, float]] = {}
    for m, g in filt.groupby("model"):
        if g.empty:
            continue
        top = g.sort_values("auroc", ascending=False).iloc[0]
        out[m] = (cell_key(top), float(top["auroc"]))
    return out


def lookup_auroc(filt: pd.DataFrame, model: str, key: CellKey) -> Optional[float]:
    """SIGN-AGNOSTIC AUROC lookup (legacy). Returns the abs-flipped AUROC
    that auroc_signed produced. Used for oracle-best computation only.
    Do NOT use this in the LOO scoring path — it lets each cell pick its
    sign after seeing held-out labels."""
    step, fam, rank = key
    g = filt[
        (filt["model"] == model)
        & (filt["gen_step"] == step)
        & (filt["family"] == fam)
        & (filt["rank"] == rank)
    ]
    if g.empty:
        return None
    return float(g["auroc"].iloc[0])


def lookup_signed_score(
    sealed: pd.DataFrame, model: str, key: CellKey, fixed_sign: int
) -> Optional[float]:
    """DIRECTION-PRESERVING AUROC: score the cell on `model` with a sign
    that was fixed BEFORE looking at the model's labels (i.e., learned
    from training-fold models). Returns roc_auc_score on `scores *
    fixed_sign`, NOT max(auc, 1-auc).

    This is the right metric for v4-candidate #4's acceptance bar: a
    deployable classifier needs to know which direction to score on a new
    model BEFORE seeing the labels. Sign-after-the-fact peeks at the
    answer key.

    Returns None if the cell column isn't present, or NaN if there are
    fewer than 4 finite samples / fewer than 2 distinct labels.
    """
    from sklearn.metrics import roc_auc_score

    if fixed_sign not in (-1, 1):
        return float("nan")
    step, fam, rank = key
    col = COL_BY_FAMILY_RANK.get((fam, rank))
    if col is None or col not in sealed.columns:
        return None
    g = sealed[(sealed["model"] == model) & (sealed["gen_step"] == step)]
    if g.empty:
        return None
    scores = g[col].to_numpy() * fixed_sign
    labels = g["contradiction"].astype(int).to_numpy()
    finite = np.isfinite(scores)
    if finite.sum() < 4 or len(np.unique(labels[finite])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[finite], scores[finite]))


# Populated in main() before run_loo_cv. Maps (family, rank) -> raw
# parquet column name. Read once from the sweep DataFrame.
COL_BY_FAMILY_RANK: Dict[Tuple[str, str], str] = {}
filt_global: pd.DataFrame = pd.DataFrame()


def majority_sign_from_train(
    train_filt: pd.DataFrame, key: CellKey
) -> int:
    """Fix the sign for a predicted cell from training-fold models only.
    Returns +1 if a majority of training models give sign +1 on this cell,
    -1 otherwise. Ties → +1."""
    step, fam, rank = key
    g = train_filt[
        (train_filt["gen_step"] == step)
        & (train_filt["family"] == fam)
        & (train_filt["rank"] == rank)
    ]
    if g.empty:
        return +1
    pos = (g["sign"] == 1).sum()
    neg = (g["sign"] == -1).sum()
    return +1 if pos >= neg else -1


# ─────────────────────────────────────────────────────────────────────────────
# Selection strategies (predict a cell key from training set)
# ─────────────────────────────────────────────────────────────────────────────


UNIVERSAL_DEFAULT_CELL: CellKey = (1, "scalar", "kl_discharged")


def strategy_universal(train_models, features, train_filt, target_model) -> CellKey:
    return UNIVERSAL_DEFAULT_CELL


def strategy_family_rule(
    train_models, features, train_filt, target_model
) -> CellKey:
    """For target model's family, pick the cell with highest *mean AUROC*
    across other members of the same family. Fall back to universal if no
    family siblings in training set."""
    target_family = features.loc[target_model, "family"]
    siblings = [m for m in train_models if features.loc[m, "family"] == target_family]
    if not siblings:
        return UNIVERSAL_DEFAULT_CELL
    g = train_filt[train_filt["model"].isin(siblings)]
    if g.empty:
        return UNIVERSAL_DEFAULT_CELL
    # Mean AUROC across siblings per (step, family, rank).
    grp = g.groupby(["gen_step", "family", "rank"])["auroc"].agg(["mean", "count"])
    # Only cells present in EVERY sibling (count == len(siblings)).
    grp = grp[grp["count"] == len(siblings)]
    if grp.empty:
        return UNIVERSAL_DEFAULT_CELL
    best = grp.sort_values("mean", ascending=False).index[0]
    return (int(best[0]), str(best[1]), str(best[2]))


def strategy_nn_terminal_commit(
    train_models, features, train_filt, target_model
) -> CellKey:
    """Find single nearest training model by terminal_commit_rate; use its
    own best cell as the prediction."""
    target_rate = features.loc[target_model, "terminal_commit_rate"]
    best_nn = None
    best_dist = float("inf")
    for m in train_models:
        d = abs(features.loc[m, "terminal_commit_rate"] - target_rate)
        if d < best_dist:
            best_dist = d
            best_nn = m
    if best_nn is None:
        return UNIVERSAL_DEFAULT_CELL
    g = train_filt[train_filt["model"] == best_nn]
    if g.empty:
        return UNIVERSAL_DEFAULT_CELL
    top = g.sort_values("auroc", ascending=False).iloc[0]
    return cell_key(top)


def strategy_handcrafted(
    train_models, features, train_filt, target_model
) -> CellKey:
    """v1 hand-crafted rule (2026-05-12 first pass): 2 branches + fallback.
    Preserved as a baseline; v2 below adds reasoning + step-bucket branches.
    """
    if features.loc[target_model, "terminal_commit_rate"] >= 0.5:
        return (1, "Centered", "r=4")
    if features.loc[target_model, "family"] == "Llama":
        return (4, "Raw", "r=2")
    return UNIVERSAL_DEFAULT_CELL


def strategy_handcrafted_v4(
    train_models, features, train_filt, target_model
) -> CellKey:
    """v4 tree (2026-05-12, post-Qwen3-1.7B). Same shape as v3 but the
    Qwen-family branch is unified with the reasoning branch — Raw r=21
    @ step 3 hits ≥ 0.967 on ALL 3 Qwen models (Qwen 2.5 0.967, Qwen 3
    8B 0.976, Qwen 3 1.7B 1.000). Beats v3's split (Fisher r=2 step 3
    for non-reasoning Qwen + Raw r=21 step 3 for reasoning) because Qwen3-1.7B
    has low surprise (0.302) — it answers directly even though it's a
    reasoning-family model, so v3's reasoning branch missed it.

    Decision order:
      1. mean_surprise_step1 > 1.0 OR family == Qwen   → Raw r=21 @ step 3
      2. terminal_commit_rate ≥ 0.5                    → Centered r=4 @ step 1
      3. family == Llama                               → Raw r=2 @ step 4
      4. family == Phi                                 → d_F_full @ step 2
      5. family == Gemma                               → Raw r=1 @ step 3
      6. mean_gen_steps < 12                           → d_F_full @ step 6
      7. fallback                                      → kl_discharged @ step 1
    """
    surprise_s1 = features.loc[target_model, "mean_surprise_step1"]
    fam = features.loc[target_model, "family"]
    if (pd.notnull(surprise_s1) and surprise_s1 > 1.0) or fam == "Qwen":
        return (3, "Raw", "r=21")
    if features.loc[target_model, "terminal_commit_rate"] >= 0.5:
        return (1, "Centered", "r=4")
    if fam == "Llama":
        return (4, "Raw", "r=2")
    if fam == "Phi":
        return (2, "scalar", "d_F_full")
    if fam == "Gemma":
        return (3, "Raw", "r=1")
    if features.loc[target_model, "mean_gen_steps"] < 12:
        return (6, "scalar", "d_F_full")
    return UNIVERSAL_DEFAULT_CELL


def strategy_handcrafted_v3(
    train_models, features, train_filt, target_model
) -> CellKey:
    """v3 tree (2026-05-12, post-v2). Adds Qwen-family branch + uses
    step-1 surprise as a secondary discriminator.

    Decision order:
      1. is_reasoning_tuned (or step-1 surprise > 1.0)
                                 → Fisher r=2 @ step 3      (Qwen 3 branch)
      2. terminal_commit ≥ 0.5   → Centered r=4 @ step 1    (Mistral-Nemo)
      3. family == Llama         → Raw r=2 @ step 4
      4. family == Phi           → d_F_full @ step 2
      5. family == Gemma         → Raw r=1 @ step 3
      6. family == Qwen          → Fisher r=2 @ step 3      (NEW — Qwen 2.5 catch)
      7. mean_gen_steps < 12     → d_F_full @ step 6        (Mistral 7B v0.3)
      8. fallback                → kl_discharged @ step 1

    Branch 1 is broadened: either explicit reasoning_tuned tag OR step-1
    surprise > 1.0 (a future-proof signal for reasoning-tuned models we
    haven't named yet).
    """
    # v3 → v4 (2026-05-12): reasoning-branch corrected from Fisher r=2 step 3
    # to Raw r=21 step 3 after the Qwen3-1.7B run revealed that within
    # reasoning-tuned models, the optimal cell is Raw r=21 @ step 3
    # (Qwen 3 8B: 0.976, Qwen 3 1.7B: 1.000 → min = 0.976). Fisher r=64
    # @ step 2 (Qwen 3 8B's oracle) is NOT scale-transferable (0.576 on 1.7B).
    surprise_s1 = features.loc[target_model, "mean_surprise_step1"]
    if pd.notnull(surprise_s1) and surprise_s1 > 1.0:
        return (3, "Raw", "r=21")
    if features.loc[target_model, "terminal_commit_rate"] >= 0.5:
        return (1, "Centered", "r=4")
    fam = features.loc[target_model, "family"]
    if fam == "Llama":
        return (4, "Raw", "r=2")
    if fam == "Phi":
        return (2, "scalar", "d_F_full")
    if fam == "Gemma":
        return (3, "Raw", "r=1")
    if fam == "Qwen":
        return (3, "Fisher", "r=2")
    if features.loc[target_model, "mean_gen_steps"] < 12:
        return (6, "scalar", "d_F_full")
    return UNIVERSAL_DEFAULT_CELL


def strategy_handcrafted_v2(
    train_models, features, train_filt, target_model
) -> CellKey:
    """v2 hand-crafted tree (2026-05-12, post first-pass failure analysis).

    Decision order:
      1. reasoning_tuned   → Fisher r=2 @ step 3            (Qwen 3 branch)
      2. terminal_commit ≥ 0.5 → Centered r=4 @ step 1      (Mistral-Nemo)
      3. family == Llama   → Raw r=2 @ step 4               (within-Llama universal)
      4. family == Phi     → d_F_full @ step 2              (both Phi-versions peak at step 2)
      5. family == Gemma   → Raw r=1 @ step 3               (Gemma 4B's cell, transfers)
      6. mean_gen_steps < 12 → d_F_full @ step 6            (Mistral 7B v0.3 short-CoT)
      7. fallback          → kl_discharged @ step 1         (Qwen 2.5 catch-all)

    Each branch maps to features observable BEFORE running the experiment.
    The Qwen 3 branch picks Fisher r=2 @ step 3 (not r=64) because r=2 is
    more likely to transfer to future reasoning-tuned models than r=64.
    """
    if features.loc[target_model, "is_reasoning_tuned"]:
        return (3, "Fisher", "r=2")
    if features.loc[target_model, "terminal_commit_rate"] >= 0.5:
        return (1, "Centered", "r=4")
    fam = features.loc[target_model, "family"]
    if fam == "Llama":
        return (4, "Raw", "r=2")
    if fam == "Phi":
        return (2, "scalar", "d_F_full")
    if fam == "Gemma":
        return (3, "Raw", "r=1")
    if features.loc[target_model, "mean_gen_steps"] < 12:
        return (6, "scalar", "d_F_full")
    return UNIVERSAL_DEFAULT_CELL


STRATEGIES = {
    "universal_kl_discharged": strategy_universal,
    "family_rule": strategy_family_rule,
    "nn_terminal_commit": strategy_nn_terminal_commit,
    "handcrafted_v1": strategy_handcrafted,
    "handcrafted_v2": strategy_handcrafted_v2,
    "handcrafted_v3": strategy_handcrafted_v3,
    "handcrafted_v4": strategy_handcrafted_v4,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main LOO-CV loop
# ─────────────────────────────────────────────────────────────────────────────


def run_loo_cv(
    filt: pd.DataFrame,
    features: pd.DataFrame,
    sealed: pd.DataFrame,
) -> pd.DataFrame:
    """LOO-CV with BOTH the legacy sign-agnostic scoring (`_auroc_abs`,
    historical) and the direction-preserving scoring (`_auroc_signed`,
    per Codex review 2026-05-12). The signed score uses a sign fixed
    from training-fold models only — no held-out label peek.

    Oracle is computed sign-agnostically (it's an upper bound, not a
    deployable classifier), so its values match prior reports.
    """
    models = sorted(filt["model"].unique().tolist())
    oracle = best_cell_per_model(filt)

    rows = []
    for held_out in models:
        train_models = [m for m in models if m != held_out]
        train_filt = filt[filt["model"].isin(train_models)]
        oracle_key, oracle_auc = oracle[held_out]

        row = {
            "model": short_model_name(held_out),
            "family": features.loc[held_out, "family"],
            "terminal_commit_rate": features.loc[held_out, "terminal_commit_rate"],
            "mean_gen_steps": features.loc[held_out, "mean_gen_steps"],
            "oracle_cell": f"{oracle_key[1]} {oracle_key[2]} @ step {oracle_key[0]}",
            "oracle_auroc": oracle_auc,
        }
        for sname, strategy_fn in STRATEGIES.items():
            pred_key = strategy_fn(train_models, features, train_filt, held_out)
            pred_auc_abs = lookup_auroc(filt, held_out, pred_key)
            # Fallback if predicted cell doesn't exist for held-out.
            fallback_used = False
            if pred_auc_abs is None:
                pred_key = UNIVERSAL_DEFAULT_CELL
                pred_auc_abs = lookup_auroc(filt, held_out, pred_key)
                fallback_used = True
            cell_label = f"{pred_key[1]} {pred_key[2]} @ step {pred_key[0]}"
            if fallback_used:
                cell_label += " (FALLBACK)"

            # DIRECTION-PRESERVING SCORE — sign fixed from training folds.
            fixed_sign = majority_sign_from_train(train_filt, pred_key)
            pred_auc_signed = lookup_signed_score(sealed, held_out, pred_key, fixed_sign)
            if pred_auc_signed is None:
                pred_auc_signed = float("nan")

            row[f"{sname}_cell"] = cell_label
            row[f"{sname}_fixed_sign"] = fixed_sign
            row[f"{sname}_auroc_abs"] = pred_auc_abs if pred_auc_abs is not None else float("nan")
            row[f"{sname}_auroc_signed"] = pred_auc_signed
            row[f"{sname}_gap_abs"] = (
                (oracle_auc - pred_auc_abs) if pred_auc_abs is not None else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_report(loo_df: pd.DataFrame, features: pd.DataFrame) -> None:
    print("\n" + "=" * 130)
    print("PER-MODEL FEATURES")
    print("=" * 130)
    cols = ["family", "is_reasoning_tuned", "terminal_commit_rate", "mean_gen_steps", "mean_surprise_step1", "n_samples"]
    pretty = features[cols].copy()
    pretty.index = [short_model_name(m) for m in pretty.index]
    pretty["terminal_commit_rate"] = pretty["terminal_commit_rate"].map("{:.3f}".format)
    pretty["mean_gen_steps"] = pretty["mean_gen_steps"].map("{:.2f}".format)
    pretty["mean_surprise_step1"] = pretty["mean_surprise_step1"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "  NA ")
    print(pretty.to_string())

    print("\n" + "=" * 130)
    print("LOO-CV (SIGN-AGNOSTIC, legacy) — predicted-cell AUROC vs oracle. Note: this uses held-out labels to flip sign.")
    print("=" * 130)
    cols = ["model", "family", "oracle_auroc"]
    for sname in STRATEGIES.keys():
        cols += [f"{sname}_auroc_abs"]
    show = loo_df[cols].copy()
    for c in show.columns:
        if c.endswith("_auroc") or c.endswith("_abs"):
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "  NA ")
    print(show.to_string(index=False))

    print("\n" + "=" * 130)
    print("LOO-CV (DIRECTION-PRESERVING, sign fixed from training folds) — the deployable-classifier metric")
    print("=" * 130)
    cols = ["model", "family", "oracle_auroc"]
    for sname in STRATEGIES.keys():
        cols += [f"{sname}_auroc_signed"]
    show = loo_df[cols].copy()
    for c in show.columns:
        if c.endswith("_auroc") or c.endswith("_signed"):
            show[c] = show[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "  NA ")
    print(show.to_string(index=False))

    print("\n" + "=" * 130)
    print("LOO-CV — predicted cell vs oracle cell per model")
    print("=" * 130)
    cols = ["model", "oracle_cell"]
    for sname in STRATEGIES.keys():
        cols.append(f"{sname}_cell")
    print(loo_df[cols].to_string(index=False))

    print("\n" + "=" * 130)
    print("AGGREGATE SUMMARY — strategy means across N=10")
    print("=" * 130)
    # SIGN-AGNOSTIC summary (legacy — measures separability after label peek)
    rows = [{
        "strategy": "oracle (upper bound)",
        "mean_auroc_abs": loo_df["oracle_auroc"].mean(),
        "min_auroc_abs": loo_df["oracle_auroc"].min(),
        "n_above_0.90_abs": int((loo_df["oracle_auroc"] >= 0.90).sum()),
        "n_above_0.95_abs": int((loo_df["oracle_auroc"] >= 0.95).sum()),
    }]
    for sname in STRATEGIES.keys():
        ac = loo_df[f"{sname}_auroc_abs"]
        rows.append({
            "strategy": sname,
            "mean_auroc_abs": ac.mean(),
            "min_auroc_abs": ac.min(),
            "n_above_0.90_abs": int((ac >= 0.90).sum()),
            "n_above_0.95_abs": int((ac >= 0.95).sum()),
        })
    summary_abs = pd.DataFrame(rows)
    for c in ["mean_auroc_abs", "min_auroc_abs"]:
        summary_abs[c] = summary_abs[c].map(lambda x: f"{x:.3f}")
    print("\n--- SIGN-AGNOSTIC (legacy, label-peeking) ---")
    print(summary_abs.to_string(index=False))

    # DIRECTION-PRESERVING summary (the deployable-classifier metric)
    rows = []
    for sname in STRATEGIES.keys():
        ac = loo_df[f"{sname}_auroc_signed"]
        rows.append({
            "strategy": sname,
            "mean_auroc_signed": ac.mean(),
            "min_auroc_signed": ac.min(),
            "n_above_0.90_signed": int((ac >= 0.90).sum()),
            "n_above_0.95_signed": int((ac >= 0.95).sum()),
        })
    summary_signed = pd.DataFrame(rows)
    for c in ["mean_auroc_signed", "min_auroc_signed"]:
        summary_signed[c] = summary_signed[c].map(lambda x: f"{x:.3f}")
    print("\n--- DIRECTION-PRESERVING (sign fixed from training folds) ---")
    print(summary_signed.to_string(index=False))


def main() -> int:
    p = argparse.ArgumentParser(description="v4-candidate #4 LOO-CV meta-classifier")
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument("--min-n-per-class", type=int, default=20)
    p.add_argument("--out-json", type=str, default="")
    args = p.parse_args()

    df = load_all_models()
    features = extract_features(df)
    sealed = df[(df["layer"] == "final") & (df["alpha"] == 1.0)].copy()
    sweep_df = run_sweep(df)
    filt = filtered_sweep(sweep_df, args.min_n, args.min_n_per_class)
    print(
        f"Sweep: {len(sweep_df)} cells total; {len(filt)} after filtering "
        f"(min_n={args.min_n}, min_n_per_class={args.min_n_per_class})"
    )

    # Populate the (family, rank) → raw-column map for direction-preserving
    # scoring. Each (family, rank) maps to a single parquet column.
    global COL_BY_FAMILY_RANK, filt_global
    COL_BY_FAMILY_RANK = {
        (str(r["family"]), str(r["rank"])): str(r["column"])
        for _, r in filt[["family", "rank", "column"]].drop_duplicates().iterrows()
    }
    filt_global = filt

    loo_df = run_loo_cv(filt, features, sealed)
    print_report(loo_df, features)

    # Filter-sensitivity diagnostic (medium-severity finding from Codex review)
    print("\n" + "=" * 130)
    print("FILTER-SENSITIVITY — per-model oracle cell under different filter settings")
    print("=" * 130)
    print(f"{'model':<35} {'(min_n=50,cls=20) cell':<35} {'AUROC':<7} {'(no filter) cell':<35} {'AUROC':<7}")
    unfilt = sweep_df.dropna(subset=["auroc"])
    for m in sorted(filt["model"].unique().tolist()):
        f_best = filt[filt["model"] == m].sort_values("auroc", ascending=False).head(1)
        u_best = unfilt[unfilt["model"] == m].sort_values("auroc", ascending=False).head(1)
        f_lbl = f"{f_best.iloc[0]['family']} {f_best.iloc[0]['rank']} @ step {int(f_best.iloc[0]['gen_step'])}"
        u_lbl = f"{u_best.iloc[0]['family']} {u_best.iloc[0]['rank']} @ step {int(u_best.iloc[0]['gen_step'])}"
        match = "✓" if f_lbl == u_lbl else "✗"
        print(f"{short_model_name(m):<35} {f_lbl:<35} {f_best.iloc[0]['auroc']:.4f}  "
              f"{u_lbl:<35} {u_best.iloc[0]['auroc']:.4f}  {match}")

    if args.out_json:
        out = {
            "min_n": args.min_n,
            "min_n_per_class": args.min_n_per_class,
            "n_models": len(loo_df),
            "features": features.reset_index().assign(
                model=lambda d: d["model"].map(short_model_name)
            ).to_dict(orient="records"),
            "loo": loo_df.to_dict(orient="records"),
            "universal_default_cell": list(UNIVERSAL_DEFAULT_CELL),
        }
        Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nJSON written: {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

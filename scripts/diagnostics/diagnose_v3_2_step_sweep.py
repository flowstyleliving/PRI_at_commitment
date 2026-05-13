#!/usr/bin/env python3
"""v3.2 full per-(gen_step × metric × rank) AUROC sweep.

Reads `all_results.parquet` (post-2026-05-08 patch — must contain the v3.2
metric columns and `gen_token_id` for adaptive cross-reference). Filters to
the sealed analysis plane (layer == 'final', alpha == 1.0), then sweeps
AUROC across every (model, gen_step, family, rank) cell.

Outputs four computed sections:
  1. Per-model best cells (top-K by AUROC)
  2. Universal-winner search — cells where MIN AUROC across all models
     is highest, dropping cells where any model has n < min_n (EOS-truncation
     guard for Mistral / Gemma 4B at gen_step >= 6)
  3. Per-step AUROC trajectory for the best-per-model metric
  4. Optional best-fixed-step vs adaptive comparison (--adaptive-json)

Reuses `auroc_signed` and `detect_rank_columns` from analyze_adaptive_step.py
for convention parity.

Usage:
    .venv/bin/python scripts/diagnostics/diagnose_v3_2_step_sweep.py \\
        --run-dir experiments/v3-main-run/<DATE>/run-NN \\
        [--out-csv /tmp/v3_2_step_sweep.csv] \\
        [--out-json /tmp/v3_2_step_sweep.json] \\
        [--adaptive-json /tmp/v3_2_adaptive_run2.json] \\
        [--top-k-universal 10] [--top-k-per-model 5] [--min-n 50]

Cross-references:
  - wiki/v4-candidates.md §3 — adaptive-step rupture detection
  - wiki/results/v3.2-amendment.md — pre-reg context
  - scripts/analyze_adaptive_step.py — auroc_signed, detect_rank_columns
  - scripts/diagnostics/diagnose_v3_2_rank_sweep.py — rank-axis parallel
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT / "scripts"))

# Convention parity: reuse the AUROC + family-detection helpers.
from analyze_adaptive_step import auroc_signed, detect_rank_columns  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def short_model_name(m: str) -> str:
    return m.split("/")[-1]


def short_metric_name(family: str, rank_label: str) -> str:
    """Match the convention used by analyze_adaptive_step.py:
        Fisher r=1, Raw r=1, Centered r=1, kl_discharged, d_F_full
    """
    if family == "scalar":
        return rank_label
    return f"{family} {rank_label}"


# ─────────────────────────────────────────────────────────────────────────────
# Sweep core
# ─────────────────────────────────────────────────────────────────────────────


def run_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Sweep AUROC over (model × gen_step × family × rank). Returns a long-
    format DataFrame; one row per cell. Includes per-class sample counts
    (n_control, n_contradiction) so downstream selection-bias filters can
    drop class-imbalanced cells."""
    sealed = df[(df["layer"] == "final") & (df["alpha"] == 1.0)].copy()
    families = detect_rank_columns(sealed)

    # Flat list of (family, rank_label, col)
    flat = []
    for fam in ("Fisher", "Raw", "Centered", "scalar"):
        for rank_label, col in families.get(fam, []):
            flat.append((fam, rank_label, col))

    rows = []
    for m in sorted(sealed["model"].unique().tolist()):
        gm = sealed[sealed["model"] == m]
        for step, gs in gm.groupby("gen_step"):
            n = len(gs)
            labels = gs["contradiction"].astype(int).to_numpy()
            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())
            n_min_class = min(n_pos, n_neg)
            for fam, rank_label, col in flat:
                if col not in gs.columns:
                    continue
                scores = gs[col].to_numpy()
                auc, sign = auroc_signed(labels, scores)
                rows.append({
                    "model": m,
                    "gen_step": int(step),
                    "family": fam,
                    "rank": rank_label,
                    "metric_name": short_metric_name(fam, rank_label),
                    "column": col,
                    "n": int(n),
                    "n_control": n_neg,
                    "n_contradict": n_pos,
                    "n_min_class": n_min_class,
                    "auroc": auc,
                    "sign": int(sign),
                })
    return pd.DataFrame(rows)


def per_model_best(sweep_df: pd.DataFrame, top_k: int) -> Dict[str, List[Dict]]:
    """Top-K (gen_step, family, rank) by AUROC per model. Includes per-class
    sample counts so the report can flag class-imbalance artifacts (e.g.,
    Gemma 4B at step 12 = 80 controls vs 5 contradictions)."""
    out: Dict[str, List[Dict]] = {}
    for m, g in sweep_df.groupby("model"):
        finite = g.dropna(subset=["auroc"]).sort_values("auroc", ascending=False)
        out[short_model_name(m)] = [
            {
                "gen_step": int(r["gen_step"]),
                "family": str(r["family"]),
                "rank": str(r["rank"]),
                "metric": str(r["metric_name"]),
                "auroc": float(r["auroc"]),
                "sign": int(r["sign"]),
                "n": int(r["n"]),
                "n_control": int(r.get("n_control", -1)),
                "n_contradict": int(r.get("n_contradict", -1)),
                "n_min_class": int(r.get("n_min_class", -1)),
            }
            for _, r in finite.head(top_k).iterrows()
        ]
    return out


def universal_winners(
    sweep_df: pd.DataFrame, top_k: int, min_n: int, min_n_per_class: int
) -> List[Dict]:
    """For each (gen_step, family, rank), report min AUROC across all models.
    Drops cells where any model has n < min_n (total-sample guard) OR
    n_min_class < min_n_per_class (class-balance guard, prevents Mistral/Gemma
    EOS-driven imbalance from generating spurious universal winners). Sorted
    by min_auroc descending."""
    n_models = sweep_df["model"].nunique()
    grouped = sweep_df.groupby(["gen_step", "family", "rank"])
    results = []
    for (step, fam, rank), g in grouped:
        if len(g) < n_models:
            continue
        if g["auroc"].isna().any():
            continue
        if (g["n"] < min_n).any():
            continue
        if "n_min_class" in g.columns and (g["n_min_class"] < min_n_per_class).any():
            continue
        per_model = {short_model_name(r["model"]): float(r["auroc"]) for _, r in g.iterrows()}
        per_model_n_min_class = {
            short_model_name(r["model"]): int(r.get("n_min_class", -1))
            for _, r in g.iterrows()
        }
        results.append({
            "gen_step": int(step),
            "family": str(fam),
            "rank": str(rank),
            "metric": short_metric_name(str(fam), str(rank)),
            "min_auroc": float(g["auroc"].min()),
            "max_auroc": float(g["auroc"].max()),
            "min_n": int(g["n"].min()),
            "min_n_per_class": int(g["n_min_class"].min()) if "n_min_class" in g.columns else -1,
            "per_model": per_model,
            "per_model_n_min_class": per_model_n_min_class,
        })
    results.sort(key=lambda x: x["min_auroc"], reverse=True)
    return results[:top_k]


def per_step_trajectory(
    sweep_df: pd.DataFrame, pm_best: Dict[str, List[Dict]]
) -> Dict[str, Dict]:
    """For each model's top-1 metric, report AUROC across all gen_steps."""
    out: Dict[str, Dict] = {}
    full_models = sweep_df["model"].unique()
    short_to_full = {short_model_name(m): m for m in full_models}
    for short_m, bests in pm_best.items():
        if not bests:
            continue
        best = bests[0]
        full_m = short_to_full.get(short_m)
        if full_m is None:
            continue
        sub = sweep_df[
            (sweep_df["model"] == full_m)
            & (sweep_df["family"] == best["family"])
            & (sweep_df["rank"] == best["rank"])
        ].sort_values("gen_step")
        out[short_m] = {
            "metric": best["metric"],
            "family": best["family"],
            "rank": best["rank"],
            "trajectory": [
                {
                    "gen_step": int(r["gen_step"]),
                    "auroc": float(r["auroc"]) if not pd.isna(r["auroc"]) else None,
                    "n": int(r["n"]),
                }
                for _, r in sub.iterrows()
            ],
        }
    return out


def adaptive_vs_fixed(
    sweep_df: pd.DataFrame, adaptive_json_path: Path
) -> Dict[str, Dict[str, Dict]]:
    """Compare per-sample adaptive AUROC (from analyze_adaptive_step.py output)
    to the best fixed-gen_step AUROC, per (model, metric)."""
    adaptive = json.loads(adaptive_json_path.read_text())
    out: Dict[str, Dict[str, Dict]] = {}
    full_models = sweep_df["model"].unique()
    for full_m in full_models:
        short_m = short_model_name(full_m)
        adap_metrics = adaptive.get("auroc", {}).get(short_m, {})
        if not adap_metrics:
            continue
        m_rows: Dict[str, Dict] = {}
        gm = sweep_df[sweep_df["model"] == full_m]
        for metric_name, adap_data in adap_metrics.items():
            adap_auc = adap_data.get("adaptive")
            if adap_auc is None or not isinstance(adap_auc, (int, float)) or np.isnan(adap_auc):
                continue
            sub = gm[gm["metric_name"] == metric_name].dropna(subset=["auroc"])
            if sub.empty:
                continue
            best_row = sub.loc[sub["auroc"].idxmax()]
            best_step = int(best_row["gen_step"])
            best_auc = float(best_row["auroc"])
            m_rows[metric_name] = {
                "best_fixed_step": best_step,
                "best_fixed_auroc": best_auc,
                "adaptive_auroc": float(adap_auc),
                "delta": float(adap_auc - best_auc),
            }
        out[short_m] = m_rows
    return out


def sample_count_summary(sweep_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Per-(model, gen_step) sample count. All metric rows at the same
    (model, gen_step) share n, so de-dup before pivoting."""
    out: Dict[str, Dict[str, int]] = {}
    for m, g in sweep_df.groupby("model"):
        short_m = short_model_name(m)
        n_by = g.drop_duplicates("gen_step")[["gen_step", "n"]]
        out[short_m] = {str(int(r["gen_step"])): int(r["n"]) for _, r in n_by.iterrows()}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def print_report(
    sweep_df: pd.DataFrame,
    pm_best: Dict[str, List[Dict]],
    universals: List[Dict],
    trajectory: Dict[str, Dict],
    adaptive: Optional[Dict],
    min_n: int,
) -> None:
    n_models = int(sweep_df["model"].nunique())
    n_cells = len(sweep_df)
    n_finite = int(sweep_df["auroc"].notna().sum())
    print("=" * 110)
    print(f"v3.2 STEP SWEEP — {n_models} models, {n_cells} cells, {n_finite} finite AUROCs")
    print("=" * 110)

    # 1. Sample-count pivot
    print(f"\nSAMPLE COUNTS — (gen_step × model). Threshold for universal-winner inclusion: n>={min_n}.")
    pivot = sweep_df.drop_duplicates(["model", "gen_step"])[["model", "gen_step", "n"]].copy()
    pivot["short"] = pivot["model"].map(short_model_name)
    pv = pivot.pivot(index="gen_step", columns="short", values="n").fillna(0).astype(int)
    print(pv.to_string())

    # 2. Per-model best
    print("\n" + "=" * 110)
    print("PER-MODEL BEST CELLS  (⚠ = n < min_n;  ⚖ = class imbalance, min_class < 20)")
    print("=" * 110)
    for short_m, rows in pm_best.items():
        print(f"\n  [{short_m}]")
        print(f"    {'#':>2s}  {'metric':<22s}  {'step':>4s}  {'AUROC':>8s}  {'sign':>4s}  {'n':>5s}  {'ctrl':>4s}  {'cntr':>4s}")
        for i, r in enumerate(rows, 1):
            warn_n = " ⚠" if r["n"] < min_n else ""
            warn_class = " ⚖" if r.get("n_min_class", -1) >= 0 and r["n_min_class"] < 20 else ""
            sign_str = "+" if r["sign"] >= 0 else "-"
            print(
                f"    {i:>2d}  {r['metric']:<22s}  {r['gen_step']:>4d}  "
                f"{r['auroc']:>8.4f}  {sign_str:>4s}  {r['n']:>5d}  "
                f"{r.get('n_control', -1):>4d}  {r.get('n_contradict', -1):>4d}{warn_n}{warn_class}"
            )

    # 3. Universal winners
    print("\n" + "=" * 110)
    print(
        f"UNIVERSAL WINNERS — top-{len(universals)} by min(AUROC) across {n_models} models, "
        f"min n>={min_n}"
    )
    print("=" * 110)
    if not universals:
        print(f"  (no cells meet the n>={min_n} threshold for all {n_models} models)")
    else:
        model_keys = sorted(universals[0]["per_model"].keys())
        header = (
            f"  {'#':>2s}  {'metric':<22s}  {'step':>4s}  {'min':>8s}  {'max':>8s}  "
            f"{'min_n':>6s}  " + "  ".join(f"{k[:10]:>10s}" for k in model_keys)
        )
        print(header)
        for i, r in enumerate(universals, 1):
            cells = "  ".join(f"{r['per_model'][k]:>10.4f}" for k in model_keys)
            print(
                f"  {i:>2d}  {r['metric']:<22s}  {r['gen_step']:>4d}  "
                f"{r['min_auroc']:>8.4f}  {r['max_auroc']:>8.4f}  {r['min_n']:>6d}  {cells}"
            )

    # 4. Trajectory
    print("\n" + "=" * 110)
    print("PER-MODEL TRAJECTORY — AUROC at each gen_step for the best-per-model metric")
    print("=" * 110)
    for short_m, data in trajectory.items():
        print(f"\n  [{short_m}]  best metric: {data['metric']}")
        print(f"    {'step':>4s}  {'AUROC':>8s}  {'n':>5s}")
        for r in data["trajectory"]:
            auc_str = f"{r['auroc']:>8.4f}" if r["auroc"] is not None else f"{'n/a':>8s}"
            warn = " ⚠" if r["n"] < min_n else ""
            print(f"    {r['gen_step']:>4d}  {auc_str}  {r['n']:>5d}{warn}")

    # 5. Adaptive comparison
    if adaptive:
        print("\n" + "=" * 110)
        print(
            "BEST-FIXED-STEP vs ADAPTIVE — Δ = adaptive_auroc − best_fixed_auroc.  "
            "Δ ≈ 0 ⇒ adaptive is decorative; Δ > 0 ⇒ adaptive is essential."
        )
        print("=" * 110)
        for short_m, metrics in adaptive.items():
            if not metrics:
                continue
            print(f"\n  [{short_m}]")
            print(
                f"    {'metric':<22s}  {'best_fixed_step':>15s}  "
                f"{'best_fixed':>11s}  {'adaptive':>9s}  {'delta':>10s}"
            )
            sorted_metrics = sorted(metrics.items(), key=lambda kv: -abs(kv[1]["delta"]))
            for metric_name, d in sorted_metrics[:8]:
                print(
                    f"    {metric_name:<22s}  {d['best_fixed_step']:>15d}  "
                    f"{d['best_fixed_auroc']:>11.4f}  {d['adaptive_auroc']:>9.4f}  "
                    f"{d['delta']:>+10.4f}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--adaptive-json",
        type=Path,
        default=None,
        help="output JSON from analyze_adaptive_step.py for adaptive comparison",
    )
    ap.add_argument("--top-k-universal", type=int, default=10)
    ap.add_argument("--top-k-per-model", type=int, default=5)
    ap.add_argument(
        "--min-n",
        type=int,
        default=50,
        help="minimum total samples per (model, gen_step) for universal-winner inclusion",
    )
    ap.add_argument(
        "--min-n-per-class",
        type=int,
        default=20,
        help="minimum per-class samples (control AND contradiction) for universal-winner inclusion. "
             "Drops EOS-imbalance cells (Mistral step ≥ 7: 3 controls / 33 contradicts; Gemma step ≥ 6: 80 / 5)",
    )
    args = ap.parse_args()

    parquet = args.run_dir / "all_results.parquet"
    if not parquet.exists():
        print(f"ERROR: {parquet} missing")
        return 1

    df = pd.read_parquet(parquet)
    sweep_df = run_sweep(df)

    pm_best = per_model_best(sweep_df, top_k=args.top_k_per_model)
    universals = universal_winners(
        sweep_df,
        top_k=args.top_k_universal,
        min_n=args.min_n,
        min_n_per_class=args.min_n_per_class,
    )
    traj = per_step_trajectory(sweep_df, pm_best)
    adaptive = None
    if args.adaptive_json and args.adaptive_json.exists():
        adaptive = adaptive_vs_fixed(sweep_df, args.adaptive_json)
    elif args.adaptive_json:
        print(f"WARN: --adaptive-json {args.adaptive_json} not found; skipping comparison")

    print_report(sweep_df, pm_best, universals, traj, adaptive, args.min_n)

    if args.out_csv:
        sweep_df.to_csv(args.out_csv, index=False)
        print(f"\nCSV written: {args.out_csv} ({len(sweep_df)} rows)")

    if args.out_json:
        out = {
            "run_dir": str(args.run_dir),
            "n_models": int(sweep_df["model"].nunique()),
            "n_gen_steps": int(sweep_df["gen_step"].nunique()),
            "min_n": int(args.min_n),
            "sample_count_summary": sample_count_summary(sweep_df),
            "per_model_best": pm_best,
            "universal_winners": universals,
            "per_model_trajectory": traj,
        }
        if adaptive is not None:
            out["adaptive_vs_fixed"] = adaptive
        args.out_json.write_text(json.dumps(out, indent=2, default=str) + "\n")
        print(f"JSON written: {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

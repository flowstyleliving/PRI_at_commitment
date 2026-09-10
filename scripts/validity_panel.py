#!/usr/bin/env python3
"""Validity panel for the PRI v3 run artifacts.

Five diagnostics that bear on whether the measured quantity supports the
claimed one. Read-only over banked parquets; runs no model and re-reads no
gate.

  1. Outcome ceiling   -- can this dataset support error-prediction claims at
                          all, or is the model already correct almost always?
  2. Folding incidence -- where does max(AUROC, 1-AUROC) actually fire? A cell
                          that never folds is orientation-proof by inspection.
  3. Sign stability    -- fit the orientation on one half, evaluate on the
                          other. Distinguishes "discriminates" from
                          "deployable with a fixed sign".
  4. Error prediction  -- does the score predict actual incorrect answers, or
                          only the prompt manipulation? Reported pooled and
                          within the contradiction arm, which holds prompt
                          type fixed.
  5. Residualizer CI   -- the sealed analyzer fits OLS once and bootstraps the
                          fixed residuals. Refitting inside each resample gives
                          an interval that includes that fitting uncertainty.

Usage:
    python scripts/validity_panel.py --runs-dir experiments/v3-main-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

SEED = 20260423
NBOOT = 1000
RANKS = [1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64]
CEILING_MIN_ERROR_RATE = 0.05
SPLITS = 200

CELLS = [
    ("2026-04-26/run-09", "Llama-3.2-3B-Instruct-4bit"),
    ("2026-04-26/run-09", "Mistral-7B-Instruct-v0.3-4bit"),
    ("2026-04-26/run-09", "Qwen2.5-7B-Instruct-4bit"),
    ("2026-04-26/run-09", "Qwen3-8B-4bit"),
    ("2026-04-27/run-01", "Phi-3.5-mini-instruct-4bit"),
    ("2026-04-27/run-02", "gemma-3-4b-it-4bit"),
]
FISHER = "null_ratio_post_rank{}"
RAW = "null_ratio_raw_post_rank{}"


def load(runs_dir: Path, run: str, model: str) -> pd.DataFrame | None:
    path = runs_dir / run / f"{model}_results.parquet"
    if not path.exists():
        return None
    d = pd.read_parquet(path)
    return d[d.gen_step == 1]


def auc(y, s):
    return roc_auc_score(y, s)


def ceiling(frames):
    print("=" * 76)
    print("1. OUTCOME CEILING -- is there any error variance to predict?")
    print("=" * 76)
    print(f"{'model':30s} {'n':>5s} {'errors':>7s} {'rate':>8s}  testable?")
    for model, d in frames:
        err = (~d.is_correct.astype(bool))
        rate = err.mean()
        ok = "yes" if rate >= CEILING_MIN_ERROR_RATE else "NO -- at ceiling"
        print(f"{model[:29]:30s} {len(d):5d} {err.sum():7d} {rate:7.1%}  {ok}")
    print(f"\n  Bar: error rate >= {CEILING_MIN_ERROR_RATE:.0%}. Below it, an "
          f"error-prediction AUROC rests on too few positives to interpret,\n"
          f"  and the dataset cannot support a hallucination-detection claim "
          f"for that model regardless of what the geometry shows.\n")


def folding(frames):
    print("=" * 76)
    print("2. FOLDING INCIDENCE -- where does max(AUROC, 1-AUROC) fire?")
    print("=" * 76)
    print(f"{'model':30s} {'Fisher folded':>14s} {'Raw folded':>12s}   sealed r=1")
    for model, d in frames:
        y = d.contradiction.astype(int).values
        f_fold = r_fold = n = 0
        r1 = ""
        for rank in RANKS:
            fc, rc = FISHER.format(rank), RAW.format(rank)
            if fc not in d.columns or rc not in d.columns:
                continue
            af, ar = auc(y, d[fc].values), auc(y, d[rc].values)
            f_fold += af < 0.5
            r_fold += ar < 0.5
            n += 1
            if rank == 1:
                tags = []
                if af < 0.5:
                    tags.append("Fisher folded")
                if ar < 0.5:
                    tags.append("Raw folded")
                r1 = ", ".join(tags) if tags else "neither folded"
        print(f"{model[:29]:30s} {f_fold:>10d}/{n:<3d} {r_fold:>8d}/{n:<3d}   {r1}")
    print("\n  A cell that never folds cannot have an orientation artifact; a "
          "cell that folds\n  reports an inverted score at its magnitude.\n")


def sign_stability(frames, rng_seed=SEED):
    print("=" * 76)
    print("3. SIGN STABILITY -- fit orientation on half, evaluate on the other")
    print("=" * 76)
    print(f"{'model':30s} {'metric':8s} {'agree':>7s} {'held-out AUROC':>16s}")
    for model, d in frames:
        y_all = d.contradiction.astype(int).values
        for label, tpl in (("Fisher", FISHER), ("Raw", RAW)):
            col = tpl.format(1)
            if col not in d.columns:
                continue
            s_all = d[col].values
            rng = np.random.default_rng(rng_seed)
            n = len(y_all)
            agree, held = 0, []
            full_sign = 1 if auc(y_all, s_all) >= 0.5 else -1
            for _ in range(SPLITS):
                idx = rng.permutation(n)
                a, b = idx[: n // 2], idx[n // 2:]
                if len(set(y_all[a])) < 2 or len(set(y_all[b])) < 2:
                    continue
                sa = 1 if auc(y_all[a], s_all[a]) >= 0.5 else -1
                ab = auc(y_all[b], s_all[b])
                agree += (sa == full_sign)
                held.append(ab if sa > 0 else 1 - ab)
            print(f"{model[:29]:30s} {label:8s} {agree/SPLITS:6.1%} "
                  f"{np.mean(held):15.4f}")
    print("\n  'agree' = how often a sign fitted on a random half matches the "
          "full-sample sign.\n  'held-out AUROC' orients by the fitted half and "
          "scores the other, so it is\n  free of in-sample orientation "
          "advantage. Low agreement means the sign is not\n  transferable and "
          "the cell is not deployable at a fixed polarity.\n")


def error_prediction(frames):
    print("=" * 76)
    print("4. ERROR PREDICTION -- actual mistakes, not the prompt manipulation")
    print("=" * 76)
    any_testable = False
    for model, d in frames:
        err = (~d.is_correct.astype(bool)).astype(int).values
        if err.mean() < CEILING_MIN_ERROR_RATE:
            continue
        any_testable = True
        print(f"\n{model}  (errors {err.sum()}/{len(d)} = {err.mean():.1%})")
        con = d.contradiction.astype(bool).values
        for label, col in (("Fisher r1", FISHER.format(1)),
                           ("Raw r1", RAW.format(1)),
                           ("surprise", "surprise")):
            if col not in d.columns:
                continue
            a_pool = auc(err, d[col].values)
            line = (f"  {label:10s} pooled vs error "
                    f"{max(a_pool, 1 - a_pool):.4f}")
            sub = d[con]
            e_sub = (~sub.is_correct.astype(bool)).astype(int).values
            if 0 < e_sub.sum() < len(e_sub):
                a_sub = auc(e_sub, sub[col].values)
                line += (f"   within-contradiction "
                         f"{max(a_sub, 1 - a_sub):.4f}")
                # Saturation guard. null_ratio is a ratio bounded above by 1,
                # so the meaningful test is distance from that bound, not
                # spread: a score pinned at 0.996-1.000 separates in its fifth
                # decimal, which is numerical, not substantive. Spread alone
                # misses this because the pinned range is still nonzero.
                v = sub[col].values
                headroom = 1.0 - float(np.min(v))
                if headroom < 0.01:
                    line += (f"   [SATURATED: all values within "
                             f"{headroom:.1e} of 1.0]")
            print(line)
    if not any_testable:
        print("\n  No model clears the error-rate bar. Error prediction is "
              "untestable on this dataset.")
    print("\n  The within-contradiction column holds prompt type fixed, so it "
          "cannot be\n  satisfied by detecting the manipulation alone. It is "
          "the construct-relevant\n  number.\n")


def residualizer_ci(frames):
    print("=" * 76)
    print("5. RESIDUALIZER CI -- refit OLS inside each bootstrap resample")
    print("=" * 76)
    print(f"{'model':30s} {'fixed-resid CI':>22s} {'refit CI':>22s}")
    for model, d in frames:
        fc = FISHER.format(1)
        if fc not in d.columns or "d_F_lowrank32" not in d.columns:
            continue
        y = d.contradiction.astype(int).values
        s = d[fc].values
        x = d["d_F_lowrank32"].values.reshape(-1, 1)
        fixed_resid = s - LinearRegression().fit(x, s).predict(x)
        rng = np.random.default_rng(SEED)
        n = len(y)
        a_fixed, a_refit = [], []
        for _ in range(NBOOT):
            i = rng.integers(0, n, n)
            if len(set(y[i])) < 2:
                continue
            a_fixed.append(auc(y[i], fixed_resid[i]))
            rs = s[i] - LinearRegression().fit(x[i], s[i]).predict(x[i])
            a_refit.append(auc(y[i], rs))
        lo1, hi1 = np.percentile(a_fixed, [2.5, 97.5])
        lo2, hi2 = np.percentile(a_refit, [2.5, 97.5])
        print(f"{model[:29]:30s} [{lo1:.4f}, {hi1:.4f}]  w={hi1-lo1:.4f}"
              f"   [{lo2:.4f}, {hi2:.4f}]  w={hi2-lo2:.4f}")
    print("\n  The sealed analyzer reports the fixed-residual interval. If the "
          "refit interval\n  is wider, the published CI omits the uncertainty "
          "in fitting the residualizer.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--emit-json", type=Path,
                    help="Also write the panel's numbers here, so figures "
                         "quoted from it are machine-checkable rather than "
                         "living only in this script's stdout.")
    args = ap.parse_args()

    frames = []
    for run, model in CELLS:
        d = load(args.runs_dir, run, model)
        if d is not None:
            frames.append((model, d))
    if not frames:
        raise SystemExit(f"no run parquets found under {args.runs_dir}")

    ceiling(frames)
    folding(frames)
    sign_stability(frames)
    error_prediction(frames)
    residualizer_ci(frames)

    if args.emit_json:
        payload = {
            "note": "Validity-panel outputs. DESCRIPTIVE, computed from banked "
                    "parquets; not a sealed gate and not part of any "
                    "pre-registered endpoint. Emitted so published figures are "
                    "checkable against a file rather than against stdout.",
            "seed": SEED, "nboot": NBOOT, "splits": SPLITS,
            "ceiling_min_error_rate": CEILING_MIN_ERROR_RATE,
            "models": {},
        }
        for model, d in frames:
            err = (~d.is_correct.astype(bool))
            payload["models"][model] = {
                "n": int(len(d)),
                "errors": int(err.sum()),
                "error_rate": float(err.mean()),
                "error_prediction_testable":
                    bool(err.mean() >= CEILING_MIN_ERROR_RATE),
            }
        args.emit_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[wrote] {args.emit_json}")


if __name__ == "__main__":
    main()

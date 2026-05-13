#!/usr/bin/env python3
"""Sweep PRI calibration across every cached model × ANLI R1/R2/R3, n=50.

Each (model, round) pair produces one v1.1 CalibrationProfile JSON. Each
model is loaded ONCE and reused across the three rounds (saves ~30s × 22 =
~11 min vs reloading per round). After all profiles land, the runner emits
two summary tables to stdout + writes the same as CSV:

  1. Winners table: model × round → (winning cell, sign, in-sample AUROC,
     OOB median, OOB CI, winner_stability, warnings count)
  2. **Fisher r=2 @ step 3 focus table**: same axes but reports the
     AUROC + sign + n_evaluated of THAT specific cell on every (model,
     round) profile — answers "is `Fisher r=2 @ step 3` a stable signed
     direction across models, or did Phi-4 + Qwen 2.5 coincide on it by
     accident?"

Resumable: profiles already on disk are skipped (delete to force re-run).

Usage:
    .venv/bin/python scripts/anli_full_sweep.py \\
        --out-dir experiments/anli-sweep/$(date +%Y-%m-%d) \\
        --n-per-class 25 --seed 20260513 --n-bootstrap 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_calibrator  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Model + round presets
# ─────────────────────────────────────────────────────────────────────────────


MODEL_PRESETS: Dict[str, List[str]] = {
    "all": [
        # 11 models that have been calibrated or smoke-tested in this session.
        # Order chosen to fail fast: smaller/faster models first.
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        "mlx-community/Phi-4-mini-instruct-4bit",
        "mlx-community/gemma-3-1b-it-4bit",
        "mlx-community/gemma-3-4b-it-4bit",
        "mlx-community/Qwen3-1.7B-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    ],
    # Subsets for partial runs
    "smoke": [
        "mlx-community/gemma-3-1b-it-4bit",  # smallest, fast iteration
    ],
    "primaries": [
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    ],
}

ROUNDS = ["R1", "R2", "R3"]


# ─────────────────────────────────────────────────────────────────────────────
# ANLI dataset construction
# ─────────────────────────────────────────────────────────────────────────────


# Matches scripts/run_v3_anli.py / scripts/anli_smoke.py for consistency.
ANLI_PROMPT_TEMPLATE = (
    "Instruction: Read the premise and decide whether the hypothesis is "
    "entailed by the premise. Answer YES if the premise entails the "
    "hypothesis, NO if the premise contradicts the hypothesis.\n"
    "\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Answer:"
)


def build_anli_jsonl(
    round_id: str,
    n_per_class: int,
    seed: int,
    out_path: Path,
) -> int:
    """Build a calibration.jsonl from `facebook/anli` `dev_r{1,2,3}`. Drops
    neutral (label=1), balances entailment (label=0 → ANLI YES → calibrator
    label=0) and contradiction (label=2 → ANLI NO → calibrator label=1) at
    `n_per_class` each. Deterministic given seed.

    Returns the number of rows written.
    """
    from datasets import load_dataset

    split = f"dev_{round_id.lower()}"
    ds = load_dataset("facebook/anli", split=split)
    rng = np.random.RandomState(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)

    pos: List[Dict[str, Any]] = []  # entailment → calibrator label 0
    neg: List[Dict[str, Any]] = []  # contradiction → calibrator label 1
    for idx in order:
        ex = ds[idx]
        if ex["label"] == 1:
            continue
        prompt = ANLI_PROMPT_TEMPLATE.format(
            premise=ex["premise"], hypothesis=ex["hypothesis"]
        )
        if ex["label"] == 0 and len(pos) < n_per_class:
            pos.append({"prompt": prompt, "label": 0})
        elif ex["label"] == 2 and len(neg) < n_per_class:
            neg.append({"prompt": prompt, "label": 1})
        if len(pos) == n_per_class and len(neg) == n_per_class:
            break

    if len(pos) < n_per_class or len(neg) < n_per_class:
        raise SystemExit(
            f"ANLI {split}: insufficient class fill — pos={len(pos)}, "
            f"neg={len(neg)}, target={n_per_class}; bump --n-per-class down "
            f"or expand the dataset slice."
        )

    rows = pos + neg
    rng.shuffle(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────


def short_model_name(slug: str) -> str:
    return slug.split("/")[-1]


def profile_path(out_dir: Path, model_slug: str, round_id: str,
                 seed: int, n_per_class: int) -> Path:
    """Profile filename includes seed + n_per_class so re-running with
    different sampling parameters can't silently collide with an old run's
    profiles. Codex review (2026-05-13) flagged the prior identity-by-
    (model, round) alone as a real risk for skip-existing + summary mixing.
    """
    return out_dir / (
        f"{short_model_name(model_slug)}__anli_{round_id}"
        f"_seed{seed}_n{n_per_class}.profile.json"
    )


def jsonl_path(out_dir: Path, round_id: str, seed: int, n_per_class: int) -> Path:
    """Per-round jsonl path; the (seed, n_per_class) suffix makes
    re-running with different params produce different files."""
    return out_dir / f"anli_{round_id}_seed{seed}_n{n_per_class}.jsonl"


def calibrate_model_on_rounds(
    model_slug: str,
    rounds: List[str],
    out_dir: Path,
    *,
    n_per_class: int,
    seed: int,
    n_bootstrap: int,
    max_new_tokens: int,
    skip_existing: bool,
) -> Dict[str, Optional[Path]]:
    """Load the model once, calibrate on each round, save profiles.
    Returns dict mapping round_id → profile path (or None if skipped/failed).
    """
    results: Dict[str, Optional[Path]] = {}

    # Skip-existing fast path: if all profiles already exist (matching THIS
    # run's seed + n_per_class), don't load the model. Filename-encoded
    # identity guards against silent reuse of profiles built with a
    # different sampling configuration (Codex review 2026-05-13).
    all_present = all(
        profile_path(out_dir, model_slug, r, seed, n_per_class).exists()
        for r in rounds
    )
    if skip_existing and all_present:
        print(f"[sweep] {short_model_name(model_slug)}: all profiles present, skipping load")
        for r in rounds:
            results[r] = profile_path(out_dir, model_slug, r, seed, n_per_class)
        return results

    print(f"\n{'=' * 80}")
    print(f"[sweep] MODEL: {model_slug}")
    print(f"{'=' * 80}")

    try:
        state = pri_calibrator.load_calibration_state(model_slug, seed=seed)
    except Exception as e:
        # Model load failures are isolated — log and skip every round.
        print(f"[sweep] FAILED to load {model_slug}: {type(e).__name__}: {e}")
        for r in rounds:
            results[r] = None
        return results

    for round_id in rounds:
        prof_path = profile_path(out_dir, model_slug, round_id, seed, n_per_class)
        if skip_existing and prof_path.exists():
            print(f"[sweep]   {round_id}: profile exists, skipping")
            results[round_id] = prof_path
            continue

        # 2026-05-13 Codex review: wrap BOTH dataset-build and calibration
        # in the same try, so a per-round failure (bad ANLI fetch, missing
        # gen step coverage, etc.) doesn't terminate the whole sweep. The
        # calibrator now raises RuntimeError (no longer SystemExit) so a
        # plain `except Exception` actually catches its failure modes.
        try:
            jp = jsonl_path(out_dir, round_id, seed, n_per_class)
            if not jp.exists():
                print(f"[sweep]   {round_id}: building calibration jsonl "
                      f"({n_per_class}+{n_per_class})")
                build_anli_jsonl(round_id, n_per_class, seed, jp)

            profile = pri_calibrator.calibrate_with_state(
                state,
                str(jp),
                task_label=f"anli_{round_id}_dev_n{2*n_per_class}",
                n_bootstrap=n_bootstrap,
                max_new_tokens=max_new_tokens,
            )
            profile.to_json(str(prof_path))
            results[round_id] = prof_path
        except Exception as e:
            print(f"[sweep]   {round_id}: FAILED ({type(e).__name__}: {e})")
            results[round_id] = None

    # Release model memory before loading the next one.
    del state
    try:
        pri_calibrator.pipeline.clear_mlx_cache()
    except Exception:
        pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary + focus tables
# ─────────────────────────────────────────────────────────────────────────────


def _panel_lookup(profile: pri_calibrator.CalibrationProfile, family: str, rank_label: str, step: int) -> Optional[Dict[str, Any]]:
    """Find a specific cell in profile.calibration_stats.candidate_panel.
    Returns the panel entry dict or None if the cell isn't in this profile's
    panel (shouldn't happen for the default panel)."""
    for entry in profile.calibration_stats.get("candidate_panel", []):
        if (
            entry.get("family") == family
            and entry.get("rank_label") == rank_label
            and entry.get("step") == step
        ):
            return entry
    return None


def collect_profiles(
    out_dir: Path,
    *,
    expected_seed: Optional[int] = None,
    expected_n_calibration: Optional[int] = None,
) -> Dict[Tuple[str, str], pri_calibrator.CalibrationProfile]:
    """Read all *__anli_R?_seed*_n*.profile.json under out_dir; key by
    (model_short, round_id). Validates each profile's `provenance.calibration_seed`
    and `task.n_calibration` against the expected values when supplied —
    mismatched profiles are skipped with a warning rather than silently
    mixed into the summary (Codex review 2026-05-13).

    Also collects legacy-named profiles (`*__anli_R?.profile.json`) for back-
    compat but they're flagged as unverified.
    """
    out: Dict[Tuple[str, str], pri_calibrator.CalibrationProfile] = {}
    # New-style filenames first (preferred)
    candidates = sorted(out_dir.glob("*__anli_R*_seed*_n*.profile.json"))
    # Plus legacy-format fallback
    candidates += sorted(out_dir.glob("*__anli_R[0-9].profile.json"))
    for p in candidates:
        stem = p.name.removesuffix(".profile.json")
        # Parse the model short + round; tolerate both legacy and new formats.
        # New: {short}__anli_{Rn}_seed{seed}_n{n}
        # Legacy: {short}__anli_{Rn}
        parts = stem.split("__anli_")
        if len(parts) != 2:
            print(f"[summary]   skipping {p.name}: unparseable filename")
            continue
        model_short = parts[0]
        round_chunk = parts[1].split("_seed")[0]  # "R2_seed42_n50" → "R2"; "R2" → "R2"
        try:
            profile = pri_calibrator.CalibrationProfile.from_json(str(p))
        except Exception as e:
            print(f"[summary]   skipping {p.name}: {type(e).__name__}: {e}")
            continue
        # Validate against expected config — refuses to silently mix runs.
        if expected_seed is not None:
            actual_seed = profile.provenance.get("calibration_seed")
            if actual_seed != expected_seed:
                print(f"[summary]   skipping {p.name}: "
                      f"seed mismatch (expected {expected_seed}, got {actual_seed})")
                continue
        if expected_n_calibration is not None:
            actual_n = profile.task.get("n_calibration")
            if actual_n != expected_n_calibration:
                print(f"[summary]   skipping {p.name}: "
                      f"n_calibration mismatch (expected {expected_n_calibration}, got {actual_n})")
                continue
        out[(model_short, round_chunk)] = profile
    return out


def emit_summary(profiles: Dict[Tuple[str, str], pri_calibrator.CalibrationProfile], out_dir: Path) -> None:
    """Print and write the two summary tables."""
    if not profiles:
        print("[summary] no profiles found")
        return

    rounds_seen = sorted({r for _, r in profiles.keys()})
    models_seen = sorted({m for m, _ in profiles.keys()})

    # ─── Winners table ───────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  WINNING CELL per (model, round)")
    print("=" * 110)
    header = f"{'model':<38s}  {'round':<5s}  {'winning cell':<30s}  {'sign':>4s}  {'in-AUROC':>9s}  {'OOB med':>8s}  {'OOB CI':<18s}  {'stab':>5s}  {'warn':>4s}"
    print(header)
    winners_csv = [["model", "round", "winning_cell", "sign", "in_sample_auroc",
                    "in_sample_ci_lo", "in_sample_ci_hi",
                    "oob_median", "oob_ci_lo", "oob_ci_hi",
                    "winner_stability", "n_warnings", "warnings"]]
    for m in models_seen:
        for r in rounds_seen:
            p = profiles.get((m, r))
            if not p:
                print(f"{m:<38s}  {r:<5s}  {'-- missing --':<30s}")
                continue
            d = p.detector
            s = p.calibration_stats
            cell = f"{d['metric']['family']} {d['metric']['label']} @ st {d['gen_step']}"
            sign = d["sign"]
            in_auc = s["auroc"]
            oob = s["oob_auroc_median"]
            ci_lo = s["oob_auroc_ci_lo"]
            ci_hi = s["oob_auroc_ci_hi"]
            stab = s["winner_stability"]
            n_warn = len(p.warnings)
            print(f"{m:<38s}  {r:<5s}  {cell:<30s}  {sign:>+4d}  "
                  f"{in_auc:>9.4f}  {oob:>8.4f}  "
                  f"[{ci_lo:.3f}, {ci_hi:.3f}]  "
                  f"{stab:>5.2f}  {n_warn:>4d}")
            winners_csv.append([m, r, cell, sign, in_auc,
                                s["auroc_bootstrap_ci_lo"], s["auroc_bootstrap_ci_hi"],
                                oob, ci_lo, ci_hi, stab, n_warn,
                                "; ".join(p.warnings)])

    (out_dir / "summary_winners.csv").write_text(
        "\n".join(",".join(str(c) for c in row) for row in winners_csv) + "\n"
    )

    # ─── Fisher r=2 @ step 3 focus table ─────────────────────────────────
    print("\n" + "=" * 110)
    print("  FOCUS: Fisher r=2 @ step 3 per (model, round)")
    print("  — does this specific cell carry a stable signed direction across models, or is it noise?")
    print("=" * 110)
    header = f"{'model':<38s}  {'round':<5s}  {'AUROC':>7s}  {'sign':>4s}  {'n_eval':>6s}  {'is_winner':>9s}"
    print(header)
    focus_csv = [["model", "round", "fisher_r2_step3_auroc", "fisher_r2_step3_sign",
                  "fisher_r2_step3_n_evaluated", "is_winner_cell"]]
    for m in models_seen:
        for r in rounds_seen:
            p = profiles.get((m, r))
            if not p:
                continue
            entry = _panel_lookup(p, family="Fisher", rank_label="r=2", step=3)
            if not entry:
                print(f"{m:<38s}  {r:<5s}  {'NA':>7s}  {'NA':>4s}  {'NA':>6s}  {'NA':>9s}")
                focus_csv.append([m, r, "", "", "", ""])
                continue
            is_winner = (
                p.detector["metric"]["family"] == "Fisher"
                and p.detector["metric"]["label"] == "r=2"
                and p.detector["gen_step"] == 3
            )
            auc = entry["auroc"]
            sign = entry["sign"]
            n_eval = entry["n_evaluated"]
            print(f"{m:<38s}  {r:<5s}  {auc:>7.4f}  {sign:>+4d}  {n_eval:>6d}  "
                  f"{'✓' if is_winner else ' ':>9s}")
            focus_csv.append([m, r, auc, sign, n_eval, is_winner])

    (out_dir / "summary_fisher_r2_step3.csv").write_text(
        "\n".join(",".join(str(c) for c in row) for row in focus_csv) + "\n"
    )

    # ─── Sign-consistency rollup for Fisher r=2 @ step 3 ─────────────────
    sign_counts = {-1: 0, 0: 0, 1: 0}
    finite_aurocs = []
    for (m, r), p in profiles.items():
        entry = _panel_lookup(p, family="Fisher", rank_label="r=2", step=3)
        if not entry or not np.isfinite(entry.get("auroc", float("nan"))):
            continue
        sign_counts[entry["sign"]] += 1
        finite_aurocs.append(entry["auroc"])
    print("\n" + "=" * 110)
    print(f"  Sign distribution for Fisher r=2 @ step 3 across {sum(sign_counts.values())} profiles:")
    print(f"     positive (+1): {sign_counts[1]}")
    print(f"     negative (-1): {sign_counts[-1]}")
    print(f"     zero (degen.): {sign_counts[0]}")
    if finite_aurocs:
        print(f"  AUROC: min={min(finite_aurocs):.3f}  median={np.median(finite_aurocs):.3f}  "
              f"max={max(finite_aurocs):.3f}  (sign-agnostic)")

    print(f"\n[summary] CSVs written to {out_dir}/summary_*.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="ANLI R1/R2/R3 sweep — full PRI calibration")
    p.add_argument("--models", default="all",
                   help="comma-separated model slugs OR a preset tag "
                        f"({', '.join(MODEL_PRESETS)}). Default: all.")
    p.add_argument("--rounds", default="R1,R2,R3",
                   help="comma-separated ANLI rounds (default: all three)")
    p.add_argument("--n-per-class", type=int, default=25,
                   help="samples per class (entailment + contradiction). "
                        "n_calibration = 2 × n_per_class. Default 25 → n=50.")
    p.add_argument("--seed", type=int, default=20260513)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--out-dir", type=str, default=None,
                   help="output directory (default: experiments/anli-sweep/<today>/run-NN)")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="skip calibrations whose profile already exists (resumable; default True)")
    p.add_argument("--force", action="store_true",
                   help="re-calibrate even if profile exists")
    p.add_argument("--summary-only", action="store_true",
                   help="skip calibration; just rebuild summary tables from existing profiles")
    args = p.parse_args()

    if args.force:
        args.skip_existing = False

    if args.models in MODEL_PRESETS:
        models = MODEL_PRESETS[args.models]
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    rounds = [r.strip() for r in args.rounds.split(",") if r.strip()]
    for r in rounds:
        if r not in ROUNDS:
            raise SystemExit(f"unknown round {r!r}; choose from {ROUNDS}")

    # Resolve out-dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        base = REPO_ROOT / "experiments" / "anli-sweep" / date_str
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted(base.glob("run-*"))
        run_num = (int(existing[-1].name.split("-")[1]) + 1) if existing else 1
        out_dir = base / f"run-{run_num:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"ANLI sweep — {len(models)} models × {len(rounds)} rounds = {len(models)*len(rounds)} profiles")
    print(f"  n_per_class={args.n_per_class}  (n_calibration={2*args.n_per_class}/profile)")
    print(f"  seed={args.seed}  n_bootstrap={args.n_bootstrap}")
    print(f"  out_dir={out_dir}")
    print(f"  models: {[short_model_name(m) for m in models]}")
    print(f"  rounds: {rounds}")
    print("=" * 80)

    if not args.summary_only:
        for slug in models:
            calibrate_model_on_rounds(
                slug, rounds, out_dir,
                n_per_class=args.n_per_class,
                seed=args.seed,
                n_bootstrap=args.n_bootstrap,
                max_new_tokens=args.max_new_tokens,
                skip_existing=args.skip_existing,
            )

    print("\n" + "=" * 80)
    print("  Building summary tables...")
    print("=" * 80)
    profiles = collect_profiles(
        out_dir,
        expected_seed=args.seed,
        expected_n_calibration=2 * args.n_per_class,
    )
    emit_summary(profiles, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

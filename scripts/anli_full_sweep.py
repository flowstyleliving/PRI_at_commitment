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
from scripts.sweep_locking import claim_next_run_dir, hold_out_dir_lock  # noqa: E402


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

WINNERS_FULL_FILENAME = "summary_winners_full.csv"
WINNERS_PUBLISHABLE_FILENAME = "summary_winners_publishable.csv"
WINNERS_BLOCKED_FILENAME = "summary_winners_blocked.csv"
DEPRECATED_WINNERS_FILENAME = "summary_winners.csv"

PUBLISHABLE_MIN_IN_SAMPLE_AUROC = 0.65
PUBLISHABLE_MAX_CI_WIDTH = 0.30
PUBLISHABLE_MIN_OOB_AUROC = 0.60
PUBLISHABLE_MAX_OOB_GAP = 0.15
PUBLISHABLE_MIN_WINNER_STABILITY = 0.70
PUBLISHABLE_MIN_WINNER_COVERAGE = 0.80


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


def _winner_panel_entry(profile: pri_calibrator.CalibrationProfile) -> Optional[Dict[str, Any]]:
    metric = profile.detector["metric"]
    return _panel_lookup(
        profile,
        family=metric["family"],
        rank_label=metric["label"],
        step=profile.detector["gen_step"],
    )


def _winner_coverage_ratio(profile: pri_calibrator.CalibrationProfile) -> Optional[float]:
    entry = _winner_panel_entry(profile)
    if not entry:
        return None
    n_calibration = profile.task.get("n_calibration")
    n_eval = entry.get("n_evaluated")
    if not n_calibration or n_eval is None:
        return None
    return float(n_eval) / float(n_calibration)


def publishability_reasons(profile: pri_calibrator.CalibrationProfile) -> List[str]:
    """Winner-specific deployability gate derived from structured stats only."""
    reasons: List[str] = []
    stats = profile.calibration_stats

    in_auc = float(stats.get("auroc", float("nan")))
    if not np.isfinite(in_auc) or in_auc < PUBLISHABLE_MIN_IN_SAMPLE_AUROC:
        reasons.append(
            f"low_auroc (best={in_auc:.3f}; <{PUBLISHABLE_MIN_IN_SAMPLE_AUROC:.2f} likely not deployable)"
            if np.isfinite(in_auc)
            else "low_auroc (best=nan; in-sample AUROC missing)"
        )

    ci_lo = float(stats.get("auroc_bootstrap_ci_lo", float("nan")))
    ci_hi = float(stats.get("auroc_bootstrap_ci_hi", float("nan")))
    if np.isfinite(ci_lo) and np.isfinite(ci_hi):
        width = ci_hi - ci_lo
        if width > PUBLISHABLE_MAX_CI_WIDTH:
            reasons.append(
                f"wide_ci (95% CI width={width:.2f}; >{PUBLISHABLE_MAX_CI_WIDTH:.2f} implies n too small)"
            )

    oob = float(stats.get("oob_auroc_median", float("nan")))
    if not np.isfinite(oob):
        reasons.append("oob_estimate_missing (winner OOB deployment estimate unavailable)")
    elif oob < PUBLISHABLE_MIN_OOB_AUROC:
        reasons.append(
            f"oob_low_auroc (oob_median={oob:.3f}; <{PUBLISHABLE_MIN_OOB_AUROC:.2f} means the cell selection didn't generalize)"
        )
    if np.isfinite(in_auc) and np.isfinite(oob):
        gap = in_auc - oob
        if gap > PUBLISHABLE_MAX_OOB_GAP:
            reasons.append(
                f"large_oob_in_sample_gap (in_sample={in_auc:.3f}, oob_median={oob:.3f}, gap={gap:.3f}; in-sample AUROC is materially over-stated)"
            )

    winner_stability = float(stats.get("winner_stability", float("nan")))
    if not np.isfinite(winner_stability):
        reasons.append("winner_stability_missing (winner stability unavailable)")
    elif winner_stability < PUBLISHABLE_MIN_WINNER_STABILITY:
        reasons.append(
            f"winner_unstable (winner_stability={winner_stability:.2f}; <{PUBLISHABLE_MIN_WINNER_STABILITY:.2f} means the selected cell is noise-driven at this n)"
        )

    coverage_ratio = _winner_coverage_ratio(profile)
    entry = _winner_panel_entry(profile)
    if coverage_ratio is None or entry is None:
        reasons.append("winner_coverage_missing (could not recover winning-cell coverage)")
    elif coverage_ratio < PUBLISHABLE_MIN_WINNER_COVERAGE:
        n_eval = entry.get("n_evaluated")
        n_calibration = profile.task.get("n_calibration")
        reasons.append(
            f"insufficient_coverage_at_winner (n_evaluated={n_eval}/{n_calibration}; <{int(PUBLISHABLE_MIN_WINNER_COVERAGE * 100)}% of calibration rows reached the winning cell)"
        )
    return reasons


def is_publishable_winner(profile: pri_calibrator.CalibrationProfile) -> bool:
    return len(publishability_reasons(profile)) == 0


def _winner_summary_row(
    model_short: str,
    round_id: str,
    profile: pri_calibrator.CalibrationProfile,
) -> Dict[str, Any]:
    d = profile.detector
    s = profile.calibration_stats
    cell = f"{d['metric']['family']} {d['metric']['label']} @ st {d['gen_step']}"
    reasons = publishability_reasons(profile)
    return {
        "model": model_short,
        "round": round_id,
        "winning_cell": cell,
        "sign": d["sign"],
        "in_sample_auroc": s["auroc"],
        "in_sample_ci_lo": s["auroc_bootstrap_ci_lo"],
        "in_sample_ci_hi": s["auroc_bootstrap_ci_hi"],
        "oob_median": s["oob_auroc_median"],
        "oob_ci_lo": s["oob_auroc_ci_lo"],
        "oob_ci_hi": s["oob_auroc_ci_hi"],
        "winner_stability": s["winner_stability"],
        "n_warnings": len(profile.warnings),
        "warnings": "; ".join(profile.warnings),
        "publishable": "yes" if not reasons else "no",
        "publishability_reasons": "; ".join(reasons),
    }


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


def emit_summary(
    profiles: Dict[Tuple[str, str], pri_calibrator.CalibrationProfile],
    out_dir: Path,
) -> None:
    """Print and write the summary tables.

    New runs emit explicit winner-table artifacts:
      - summary_winners_full.csv
      - summary_winners_publishable.csv
      - summary_winners_blocked.csv

    `summary_winners.csv` is deprecated and intentionally NOT emitted for new
    runs so readers must opt into one of the explicit contracts.
    """
    if not profiles:
        print("[summary] no profiles found")
        return

    rounds_seen = sorted({r for _, r in profiles.keys()})
    models_seen = sorted({m for m, _ in profiles.keys()})

    # ─── Winners table ───────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  WINNING CELL per (model, round)")
    print("=" * 110)
    header = (
        f"{'model':<38s}  {'round':<5s}  {'winning cell':<30s}  {'sign':>4s}  "
        f"{'in-AUROC':>9s}  {'OOB med':>8s}  {'OOB CI':<18s}  {'stab':>5s}  "
        f"{'warn':>4s}  {'pub':>4s}"
    )
    print(header)
    winner_header = [
        "model", "round", "winning_cell", "sign", "in_sample_auroc",
        "in_sample_ci_lo", "in_sample_ci_hi",
        "oob_median", "oob_ci_lo", "oob_ci_hi",
        "winner_stability", "n_warnings", "warnings",
        "publishable", "publishability_reasons",
    ]
    winners_full_rows: List[Dict[str, Any]] = []
    winners_publishable_rows: List[Dict[str, Any]] = []
    winners_blocked_rows: List[Dict[str, Any]] = []
    missing_pairs = 0
    for m in models_seen:
        for r in rounds_seen:
            p = profiles.get((m, r))
            if not p:
                print(f"{m:<38s}  {r:<5s}  {'-- missing --':<30s}")
                missing_pairs += 1
                continue
            row = _winner_summary_row(m, r, p)
            in_auc = float(row["in_sample_auroc"])
            oob = float(row["oob_median"])
            ci_lo = float(row["oob_ci_lo"])
            ci_hi = float(row["oob_ci_hi"])
            stab = float(row["winner_stability"])
            sign = int(row["sign"])
            n_warn = len(p.warnings)
            is_publishable = row["publishable"] == "yes"
            print(
                f"{m:<38s}  {r:<5s}  {row['winning_cell']:<30s}  {sign:>+4d}  "
                f"{in_auc:>9.4f}  {oob:>8.4f}  "
                f"[{ci_lo:.3f}, {ci_hi:.3f}]  "
                f"{stab:>5.2f}  {n_warn:>4d}  {('yes' if is_publishable else 'no'):>4s}"
            )
            winners_full_rows.append(row)
            (winners_publishable_rows if is_publishable else winners_blocked_rows).append(row)

    # 2026-05-15 Codex fix: use csv.writer for proper quoting. Warning
    # strings contain commas (e.g. "large_oob_in_sample_gap (gap=0.151;
    # in-sample AUROC is materially over-stated by selection bias)"),
    # which the prior naive join broke into extra columns and made the
    # most-problematic rows unparseable downstream.
    for fname, rows in (
        (WINNERS_FULL_FILENAME, winners_full_rows),
        (WINNERS_PUBLISHABLE_FILENAME, winners_publishable_rows),
        (WINNERS_BLOCKED_FILENAME, winners_blocked_rows),
    ):
        with (out_dir / fname).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=winner_header, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(rows)

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

    with (out_dir / "summary_fisher_r2_step3.csv").open("w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for row in focus_csv:
            w.writerow(row)

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

    publishable_winners = len(winners_publishable_rows)
    blocked_winners = len(winners_blocked_rows)
    total_winners = len(winners_full_rows)
    deprecated_exists = (out_dir / DEPRECATED_WINNERS_FILENAME).exists()
    print(f"\n[summary] publishable winners: {publishable_winners}")
    print(f"[summary] blocked winners: {blocked_winners}")
    print(f"[summary] total winner rows: {total_winners}")
    print(f"[summary] missing model/round pairs: {missing_pairs}")
    print(f"[summary] CSVs written to {out_dir}/summary_*.csv")
    print(
        f"[summary] canonical winner tables: {WINNERS_FULL_FILENAME}, "
        f"{WINNERS_PUBLISHABLE_FILENAME}, {WINNERS_BLOCKED_FILENAME}"
    )
    if deprecated_exists:
        print(
            f"[summary] note: existing {DEPRECATED_WINNERS_FILENAME} left untouched "
            "for backward inspection; new runs no longer emit it."
        )
    if blocked_winners > 0 or missing_pairs > 0:
        print(
            f"[summary] note: inspect {WINNERS_PUBLISHABLE_FILENAME} for the "
            f"filtered view and {WINNERS_BLOCKED_FILENAME} for excluded rows."
        )


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
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        base = REPO_ROOT / "experiments" / "anli-sweep" / date_str
        out_dir = claim_next_run_dir(base).resolve()

    try:
        with hold_out_dir_lock(out_dir):
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
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""PRI calibrator — produce a deployable (model, task) detector profile from
a small labeled set.

Why this exists
---------------
The 2026-05-12 Codex adversarial review + the ANLI cross-task pilot showed
PRI rupture detection cannot be deployed label-free: both the *which metric*
and the *which sign* questions are heterogeneous across models (synthetic
N=11: 8 +/3 −) and across tasks for the same model (Mistral-Nemo: synthetic
+, ANLI R2 −). The rupture is real — every (model, task) we measured had
AUROC ≈ 0.70–1.0 on some cell at gen_step=1 — but each (model, task) pair
needs a small labeled calibration set to *pick the cell* and *lock the sign*.

This module is the calibration harness. Usage:

    python pri_calibrator.py \\
        --model mlx-community/Mistral-Nemo-Instruct-2407-4bit \\
        --data calibration.jsonl \\
        --out my_model_my_task.profile.json

It loads the model once, runs the model on each labeled sample, computes a
short, curated panel of candidate rupture metrics at fixed gen_steps,
selects the best (metric, sign) by direction-preserving AUROC on the
calibration set, bootstraps a confidence interval, and persists a versioned
`CalibrationProfile` JSON that downstream consumers (a future `pri_detector.py`)
can load to score new prompts.

Schema is v1.0 and frozen. Migrations land in `pri_profile_migrations/` once
we have a v2.0.

Input format (jsonl):
    {"prompt": "Premise: ... Hypothesis: ... Answer:", "label": 0}
    {"prompt": "Premise: ... Hypothesis: ... Answer:", "label": 1}

`label` ∈ {0, 1}: 1 = the target class (contradiction/positive), 0 = consistent/
non-contradiction. Prompts are PRE-built by the researcher; the calibrator
does NOT inject a prompt template — it only applies the model's chat-template
strategy via `pri_v2_io_plugins.get_prompt_strategy`.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse existing pipeline primitives.
import pri_v2_mlx_pipeline as pipeline
import pri_v2_io_plugins as io_plugins
from analyze_adaptive_step import auroc_signed


SCHEMA_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Metric panel — fixed-step per candidate.
# ─────────────────────────────────────────────────────────────────────────────

# `(parquet_gen_step, family, rank_label)` — these match the column conventions
# the existing pipeline + step-sweep analyses use. The calibrator only
# evaluates these 8 cells (multiple-testing burden at n=10-50 is tractable).
PanelCell = Tuple[int, str, str]

DEFAULT_PANEL: List[PanelCell] = [
    (1, "scalar", "d_F_full"),         # most task-stable across N=11 (per 2026-05-12 cross-task analysis)
    (1, "scalar", "kl_discharged"),    # closed-form KL-grounded scalar
    (1, "Fisher", "r=1"),              # sealed v3 primary
    (3, "Fisher", "r=2"),              # Qwen 2.5 oracle (synthetic)
    (1, "Centered", "r=2"),            # Mistral-Nemo / Phase B oracle (synthetic)
    (1, "Centered", "r=4"),            # alternate centered low-rank
    (4, "Raw", "r=2"),                 # cross-Llama universal (3B + 8B)
    (3, "Raw", "r=21"),                # cross-Qwen universal (N=11 LOO finding)
]


def _column_name(cell: PanelCell) -> str:
    """Map a PanelCell to the `compute_step` output dict key."""
    step, fam, label = cell
    if fam == "scalar":
        return label  # "d_F_full" / "kl_discharged"
    rank = int(label.split("=")[1])
    if fam == "Fisher":
        return f"null_ratio_post_rank{rank}"
    if fam == "Raw":
        return f"null_ratio_raw_post_rank{rank}"
    if fam == "Centered":
        return f"null_ratio_centered_post_rank{rank}"
    raise ValueError(f"unknown family: {fam}")


def _cell_label(cell: PanelCell) -> str:
    """Human-readable cell name for reports + provenance."""
    step, fam, label = cell
    if fam == "scalar":
        return f"{label} @ step {step}"
    return f"{fam} {label} @ step {step}"


# ─────────────────────────────────────────────────────────────────────────────
# CalibrationProfile (frozen schema v1.0)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationProfile:
    """Versioned profile that fully specifies a deployable PRI detector.

    Fields are deliberately flat-friendly so the JSON form is easy to inspect
    in a text editor. `schema_version` MUST be `"1.0"` until a breaking
    change lands (then bump + add a migration under pri_profile_migrations/).
    """

    schema_version: str
    model: Dict[str, Any]                 # {slug, output_projection_kind}
    task: Dict[str, Any]                  # {label, n_calibration, n_pos, n_neg, data_hash}
    detector: Dict[str, Any]              # {gen_step, layer, alpha, metric, sign, threshold}
    calibration_stats: Dict[str, Any]     # {auroc, ci_lo, ci_hi, candidate_panel, n_evaluated_per_cell}
    provenance: Dict[str, Any]            # {calibration_seed, pipeline_module_hash, calibrated_at_iso, n_bootstrap}
    warnings: List[str] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str) -> "CalibrationProfile":
        d = json.loads(Path(path).read_text())
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"profile schema {d.get('schema_version')} != supported {SCHEMA_VERSION}; "
                f"see pri_profile_migrations/ once v2.0 lands"
            )
        return cls(**d)


# ─────────────────────────────────────────────────────────────────────────────
# Data ingestion
# ─────────────────────────────────────────────────────────────────────────────


def _load_calibration_jsonl(path: str) -> Tuple[List[str], np.ndarray, str]:
    """Read calibration.jsonl → (prompts, labels, sha256_hash).
    `data_hash` is the sha256 of `<label>\\t<prompt>\\n` rows in input order so
    re-running on the same input produces an identical hash.
    """
    prompts: List[str] = []
    labels: List[int] = []
    h = hashlib.sha256()
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p = str(row["prompt"])
            y = int(row["label"])
            if y not in (0, 1):
                raise ValueError(f"label must be 0 or 1, got {y!r}")
            prompts.append(p)
            labels.append(y)
            h.update(f"{y}\t{p}\n".encode("utf-8"))
    if not prompts:
        raise SystemExit(f"no calibration samples loaded from {path}")
    return prompts, np.array(labels, dtype=np.int32), h.hexdigest()


def _hash_file(path: Path) -> str:
    """sha256 of a file's contents, used for code-version provenance."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Trace + metric computation for one calibration sample
# ─────────────────────────────────────────────────────────────────────────────


def _compute_panel_scores_for_sample(
    pri_computer: pipeline.PRIComputer,
    trace: Dict[str, Any],
    layer_name: str,
    panel: List[PanelCell],
    alpha: float = 1.0,
    v3_rank_values: Tuple[int, ...] = (1, 2, 4, 21),
) -> Dict[PanelCell, Optional[float]]:
    """For one calibration sample's trace, compute every panel cell's value.
    Returns dict mapping panel cell → score (or None if the model EOS'd
    before the panel step). The same `compute_step` invocation at a given
    step emits ALL column families, so we cache by step rather than
    re-computing.
    """
    gen_hidden = trace["gen_hidden"][layer_name]
    n_gen = len(gen_hidden)
    gen_probs = trace["gen_probs"]
    gen_surprises = trace["gen_surprises"]
    last_prefix = trace["last_prefix_hidden"][layer_name]

    # Cache per-step compute_step output. parquet gen_step 1 → idx 0.
    steps_needed = sorted(set(step for step, _, _ in panel))
    step_to_result: Dict[int, Optional[Dict[str, float]]] = {}
    for step in steps_needed:
        idx = step - 1
        if idx >= n_gen:
            step_to_result[step] = None
            continue
        h_t = gen_hidden[idx]
        h_prev = gen_hidden[idx - 1] if idx >= 1 else last_prefix
        p_t = gen_probs[idx]
        S_t = float(gen_surprises[idx]) if np.isfinite(gen_surprises[idx]) else 0.0
        result = pri_computer.compute_step(
            h_t=h_t,
            h_prev=h_prev,
            p_t=p_t,
            S_t=S_t,
            alpha=alpha,
            topk_values=[32],
            lowrank_values=[32],
            v3_rank_values=list(v3_rank_values),
            v3_capture_raw=True,
            v3_capture_centered=True,
        )
        step_to_result[step] = result

    out: Dict[PanelCell, Optional[float]] = {}
    for cell in panel:
        step, _, _ = cell
        res = step_to_result.get(step)
        if res is None:
            out[cell] = None
            continue
        col = _column_name(cell)
        v = res.get(col)
        if v is None or not np.isfinite(v):
            out[cell] = None
        else:
            out[cell] = float(v)
    return out


def _trace_one_prompt(
    model: Any,
    tokenizer: Any,
    projection: pipeline.OutputProjection,
    layer_indices: Dict[str, int],
    prompt: str,
    prompt_strategy,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """Apply the model's chat-template strategy then call trace_sample."""
    wrapped = prompt_strategy(prompt, tokenizer)
    return pipeline.trace_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=wrapped,
        layer_indices=layer_indices,
        output_projection=projection,
        max_new_tokens=max_new_tokens,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scoring + bootstrap
# ─────────────────────────────────────────────────────────────────────────────


def _score_candidate(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, int, int]:
    """Return (auroc, sign, n_evaluated). Sign is locked from THIS data —
    that's the whole point of calibration. Drops NaN scores; if fewer than
    4 samples with both labels survive, returns (nan, 0, n_finite)."""
    finite = np.isfinite(scores)
    n_eval = int(finite.sum())
    s = scores[finite]
    y = labels[finite]
    if n_eval < 4 or len(np.unique(y)) < 2:
        return float("nan"), 0, n_eval
    auc, sign = auroc_signed(y, s)
    return float(auc), int(sign), n_eval


def _bootstrap_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    sign: int,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float]:
    """Resample with replacement n_bootstrap times, scoring with the LOCKED
    sign (no re-flipping per round). Returns (2.5%, 97.5%) percentiles.
    With locked sign, AUROC < 0.5 is meaningful — it means the sign was
    wrong for this resample (chance noise on small calibration sets).
    """
    from sklearn.metrics import roc_auc_score

    finite = np.isfinite(scores)
    s = scores[finite] * sign
    y = labels[finite]
    n = len(s)
    if n < 4 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        ys, ss = y[idx], s[idx]
        if len(np.unique(ys)) < 2:
            continue
        aucs.append(roc_auc_score(ys, ss))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ─────────────────────────────────────────────────────────────────────────────
# Warnings
# ─────────────────────────────────────────────────────────────────────────────


def _emit_warnings(
    n_calibration: int,
    n_pos: int,
    n_neg: int,
    best_auroc: float,
    ci_lo: float,
    ci_hi: float,
    panel_eval_counts: Dict[PanelCell, int],
) -> List[str]:
    """Deployability warnings, baked into the profile so downstream consumers
    see them at load time. Don't raise — these are advisory; the researcher
    decides whether to deploy."""
    w: List[str] = []
    if n_calibration < 20:
        w.append(f"small_calibration_n (n={n_calibration}; rule of thumb: >= 20)")
    if n_pos + n_neg > 0:
        pos_rate = n_pos / (n_pos + n_neg)
        if pos_rate < 0.3 or pos_rate > 0.7:
            w.append(f"class_imbalance (pos_rate={pos_rate:.2f}; aim for [0.3, 0.7])")
    if np.isfinite(best_auroc) and best_auroc < 0.65:
        w.append(f"low_auroc (best={best_auroc:.3f}; <0.65 likely not deployable)")
    if np.isfinite(ci_lo) and np.isfinite(ci_hi):
        width = ci_hi - ci_lo
        if width > 0.30:
            w.append(f"wide_ci (95%% CI width={width:.2f}; >0.30 implies n too small)")
    for cell, n_eval in panel_eval_counts.items():
        if n_eval < n_calibration * 0.8:
            w.append(
                f"insufficient_coverage_at_{_cell_label(cell)} "
                f"(n_evaluated={n_eval}/{n_calibration}; model EOS'd before this step too often)"
            )
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration entry point
# ─────────────────────────────────────────────────────────────────────────────


def calibrate(
    model_slug: str,
    calibration_jsonl_path: str,
    *,
    task_label: str = "",
    panel: Optional[List[PanelCell]] = None,
    seed: int = 20260512,
    n_bootstrap: int = 1000,
    max_new_tokens: int = 8,
    layer_name: str = "final",
    alpha: float = 1.0,
) -> CalibrationProfile:
    """Calibrate a PRI detector for (model_slug, calibration_data). Returns
    a CalibrationProfile with the best (cell, sign) locked, plus bootstrap
    CI and deployability warnings.

    `max_new_tokens` defaults to 8 — enough to cover gen_step ∈ {1..5} for the
    default panel (max panel step is 4; one extra for safety) while keeping
    per-sample cost low.
    """
    panel = list(panel or DEFAULT_PANEL)
    prompts, labels, data_hash = _load_calibration_jsonl(calibration_jsonl_path)
    n_calibration = len(prompts)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    print(f"[calibrate] model={model_slug}")
    print(f"[calibrate] n_calibration={n_calibration} (pos={n_pos}, neg={n_neg})")
    print(f"[calibrate] panel={[_cell_label(c) for c in panel]}")

    # Build a Config that mirrors the v3.2 sealed protocol (final-layer only).
    cfg = pipeline.Config()
    cfg.layers_to_probe = [layer_name]
    cfg.seed = seed

    # Load model + extract gamma + build PRIComputer.
    model, tokenizer, projection, layer_indices = pipeline.load_model(model_slug, cfg)
    gamma = pipeline._extract_final_rmsnorm_gamma(model)
    if gamma is None:
        raise SystemExit(
            f"could not extract final-RMSNorm gamma for {model_slug}; "
            f"check pri_v2_mlx_pipeline._extract_final_rmsnorm_gamma logs."
        )
    pri_computer = pipeline.PRIComputer(projection, final_norm_gamma=gamma)
    prompt_strategy = io_plugins.get_prompt_strategy(model_slug)

    # Per-sample × per-cell score matrix.
    n_cells = len(panel)
    score_matrix = np.full((n_calibration, n_cells), np.nan, dtype=np.float64)
    print(f"[calibrate] tracing {n_calibration} samples...")
    for i, prompt in enumerate(prompts):
        trace = _trace_one_prompt(
            model, tokenizer, projection, layer_indices, prompt,
            prompt_strategy, max_new_tokens,
        )
        per_cell = _compute_panel_scores_for_sample(
            pri_computer, trace, layer_name, panel, alpha=alpha,
        )
        for j, cell in enumerate(panel):
            v = per_cell.get(cell)
            if v is not None:
                score_matrix[i, j] = v
        if (i + 1) % 10 == 0 or i + 1 == n_calibration:
            print(f"[calibrate]   {i+1}/{n_calibration}")

    # Score every candidate; pick best by |AUROC - 0.5|.
    candidate_results = []
    panel_eval_counts: Dict[PanelCell, int] = {}
    best_idx = -1
    best_distance = -1.0
    for j, cell in enumerate(panel):
        scores_j = score_matrix[:, j]
        auc, sign, n_eval = _score_candidate(scores_j, labels)
        panel_eval_counts[cell] = n_eval
        candidate_results.append({
            "cell": _cell_label(cell),
            "step": cell[0],
            "family": cell[1],
            "rank_label": cell[2],
            "column_name": _column_name(cell),
            "auroc": auc,
            "sign": sign,
            "n_evaluated": n_eval,
        })
        if np.isfinite(auc):
            d = abs(auc - 0.5)
            if d > best_distance:
                best_distance = d
                best_idx = j

    if best_idx < 0:
        raise SystemExit(
            "no candidate cell produced a finite AUROC — calibration data "
            "may be too small or all samples EOS'd before reaching panel steps."
        )

    best_cell = panel[best_idx]
    best_auroc = float(candidate_results[best_idx]["auroc"])
    best_sign = int(candidate_results[best_idx]["sign"])

    print(f"[calibrate] best cell: {_cell_label(best_cell)}  "
          f"AUROC={best_auroc:.3f}  sign={best_sign:+d}")

    # Bootstrap CI on the locked sign.
    ci_lo, ci_hi = _bootstrap_auroc(
        score_matrix[:, best_idx], labels, best_sign, n_bootstrap, seed,
    )
    print(f"[calibrate] bootstrap 95%% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  "
          f"(n_bootstrap={n_bootstrap})")

    warnings_list = _emit_warnings(
        n_calibration, n_pos, n_neg, best_auroc, ci_lo, ci_hi, panel_eval_counts,
    )
    for w in warnings_list:
        print(f"[calibrate]   WARNING: {w}")

    pipeline_path = REPO_ROOT / "pri_v2_mlx_pipeline.py"
    profile = CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        model={
            "slug": model_slug,
            "output_projection_kind": projection.mode,
        },
        task={
            "label": task_label,
            "n_calibration": n_calibration,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "data_hash_sha256": data_hash,
        },
        detector={
            "gen_step": int(best_cell[0]),
            "layer": layer_name,
            "alpha": float(alpha),
            "metric": {
                "family": best_cell[1],
                "label": best_cell[2],
                "column_name": _column_name(best_cell),
            },
            "sign": best_sign,
            "threshold": None,  # researcher chooses at deploy time (Youden's J etc.)
        },
        calibration_stats={
            "auroc": best_auroc,
            "auroc_bootstrap_ci_lo": ci_lo,
            "auroc_bootstrap_ci_hi": ci_hi,
            "candidate_panel": candidate_results,
        },
        provenance={
            "calibration_seed": int(seed),
            "n_bootstrap": int(n_bootstrap),
            "pipeline_module_hash_sha256": _hash_file(pipeline_path),
            "calibrator_module_hash_sha256": _hash_file(REPO_ROOT / "pri_calibrator.py"),
            "calibrated_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "max_new_tokens": int(max_new_tokens),
        },
        warnings=warnings_list,
    )
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="PRI calibrator (v1.0 schema)")
    p.add_argument("--model", required=True, help="model slug, e.g. mlx-community/Mistral-Nemo-Instruct-2407-4bit")
    p.add_argument("--data", required=True, help="calibration jsonl path")
    p.add_argument("--out", required=True, help="output profile json path")
    p.add_argument("--task-label", default="", help="task identifier for provenance (e.g. 'anli_r2_dev')")
    p.add_argument("--seed", type=int, default=20260512)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--max-new-tokens", type=int, default=8,
                   help="generation budget per sample (default 8 — covers panel steps 1..4 + safety)")
    p.add_argument("--layer", default="final", help="capture layer (default: final)")
    p.add_argument("--alpha", type=float, default=1.0)
    args = p.parse_args()

    profile = calibrate(
        model_slug=args.model,
        calibration_jsonl_path=args.data,
        task_label=args.task_label,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        max_new_tokens=args.max_new_tokens,
        layer_name=args.layer,
        alpha=args.alpha,
    )
    profile.to_json(args.out)
    print(f"[calibrate] wrote profile: {args.out}")
    if profile.warnings:
        print(f"[calibrate] {len(profile.warnings)} warning(s) — see profile['warnings']")
    return 0


if __name__ == "__main__":
    sys.exit(main())

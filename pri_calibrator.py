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


SCHEMA_VERSION = "1.1"
# v1.0 → v1.1 (2026-05-13): Codex adversarial review fixes.
#   * calibration_stats gains oob_auroc_median + oob_auroc_ci_{lo,hi} +
#     oob_n_bootstrap_used + winner_stability + winner_counts. The in-sample
#     auroc/CI are still recorded (legacy semantics, useful for inspection),
#     but the OOB stats are the honest "deployment-ready" estimate.
#   * provenance gains io_plugins_module_hash_sha256 +
#     model_adapters_module_hash_sha256 + model_snapshot_sha. Detector's
#     strict mode validates ALL hashes, not just the pipeline file.
#   * v1.0 profiles are rejected — re-calibrate.


# ─────────────────────────────────────────────────────────────────────────────
# Metric panel — fixed-step per candidate.
# ─────────────────────────────────────────────────────────────────────────────

# `(parquet_gen_step, family, rank_label)` — these match the column conventions
# the existing pipeline + step-sweep analyses use. The calibrator only
# evaluates these 8 cells (multiple-testing burden at n=10-50 is tractable).
PanelCell = Tuple[int, str, str]

DEFAULT_PANEL: List[PanelCell] = [
    # ─── Direct (raw) cells from compute_step ──────────────────────────
    (1, "scalar", "d_F_full"),         # most task-stable across N=11
    (1, "scalar", "kl_discharged"),    # closed-form KL-grounded scalar
    (1, "Fisher", "r=1"),              # `pri_v3_null_bare` — decomposition control
    (3, "Fisher", "r=2"),              # Qwen 2.5 oracle (synthetic)
    (1, "Centered", "r=2"),            # Mistral-Nemo oracle (synthetic)
    (1, "Centered", "r=4"),            # alternate centered low-rank
    (4, "Raw", "r=2"),                 # cross-Llama universal (3B + 8B)
    (3, "Raw", "r=21"),                # cross-Qwen universal
    # ─── Residualized (E18 sealed primary form) ────────────────────────
    # `null_ratio_resid = null_ratio − predicted(null_ratio | d_F_full)`
    # via linear regression on d_F_full alone. Per pri-v3-plan.md §
    # Magnitude-independence test: this is the SEALED E18 acceptance
    # variable — strips the magnitude confound the raw null_ratio carries.
    # Added 2026-05-13 after Codex/user found the calibrator was scoring
    # only the decomposition control (E17 null_bare), not the sealed primary.
    (1, "Fisher_resid", "r=1"),
    (1, "Centered_resid", "r=2"),
    (4, "Raw_resid", "r=2"),
    (3, "Raw_resid", "r=21"),
    # ─── Composites (E18 additive + E19 multiplicative) ────────────────
    # `pri_v3_null_ratio = S_t + α · null_ratio_final` (additive, E18)
    # `pri_v3_null_gated = d_F · null_ratio` (multiplicative, E19)
    (1, "Composite", "additive_S_fisher_r=1"),     # surprise + null_ratio_post_rank1
    (1, "Composite", "gated_dF_fisher_r=1"),       # d_F_full * null_ratio_post_rank1
    (1, "Composite", "additive_S_centered_r=2"),
    (1, "Composite", "gated_dF_centered_r=2"),
]


# Families that are DERIVED (not present as a direct compute_step column).
# Residualized cells use the base family's column + a regression against
# d_F_full. Composite cells combine compute_step columns (S_t / d_F_full
# with null_ratio). Both are computed in `_compute_panel_scores_for_sample`
# and (for residuals) post-processed across the full sample set.
DERIVED_RESID_FAMILIES = {"Fisher_resid", "Raw_resid", "Centered_resid"}
DERIVED_COMPOSITE_FAMILY = "Composite"


def _resid_base_family(family: str) -> str:
    """Map a `*_resid` family to the underlying compute_step family."""
    return family.removesuffix("_resid")


def _column_name(cell: PanelCell) -> str:
    """Map a PanelCell to the `compute_step` output dict key.
    For derived cells (residualized / composite), this returns the column
    of the *base* signal — the residualization or composition logic is
    applied by `_compute_panel_scores_for_sample` and the post-loop
    residualization pass in `calibrate_with_state`.
    """
    step, fam, label = cell
    if fam == "scalar":
        return label  # "d_F_full" / "kl_discharged"
    if fam == DERIVED_COMPOSITE_FAMILY:
        # Composite labels encode their own base column reference. Return
        # the label itself; the dispatcher in _compute_panel_scores_for_sample
        # parses it and composes from primitive columns.
        return f"composite::{label}"
    base_fam = _resid_base_family(fam) if fam in DERIVED_RESID_FAMILIES else fam
    rank = int(label.split("=")[1])
    if base_fam == "Fisher":
        return f"null_ratio_post_rank{rank}"
    if base_fam == "Raw":
        return f"null_ratio_raw_post_rank{rank}"
    if base_fam == "Centered":
        return f"null_ratio_centered_post_rank{rank}"
    raise ValueError(f"unknown family: {fam}")


def _cell_label(cell: PanelCell) -> str:
    """Human-readable cell name for reports + provenance."""
    step, fam, label = cell
    if fam == "scalar":
        return f"{label} @ step {step}"
    if fam == DERIVED_COMPOSITE_FAMILY:
        return f"composite[{label}] @ step {step}"
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
        raise RuntimeError(f"no calibration samples loaded from {path}")
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
        step, fam, label = cell
        res = step_to_result.get(step)
        if res is None:
            out[cell] = None
            continue

        # ── Composite: assemble from primitive columns at THIS sample ──
        if fam == DERIVED_COMPOSITE_FAMILY:
            v = _compose_score(res, label)
            out[cell] = v if (v is not None and np.isfinite(v)) else None
            continue

        # ── Residualized: emit the BASE column value here; the residual
        # ── pass after all samples land will overwrite this with
        # ── (raw − predicted_from_d_F).
        if fam in DERIVED_RESID_FAMILIES:
            base_fam = _resid_base_family(fam)
            base_col = _column_name((step, base_fam, label))
            v = res.get(base_col)
            out[cell] = float(v) if (v is not None and np.isfinite(v)) else None
            continue

        # ── Direct cells (scalar / Fisher / Raw / Centered) ────────────
        col = _column_name(cell)
        v = res.get(col)
        if v is None or not np.isfinite(v):
            out[cell] = None
        else:
            out[cell] = float(v)
    return out


def _compose_score(result: Dict[str, float], label: str) -> Optional[float]:
    """Build a composite score from primitive compute_step columns.

    Label conventions (extend by appending to this dispatcher):
      additive_S_fisher_r=N    → result["surprise"] + result["null_ratio_post_rankN"]
      gated_dF_fisher_r=N      → result["d_F_full"] * result["null_ratio_post_rankN"]
      additive_S_centered_r=N  → result["surprise"] + result["null_ratio_centered_post_rankN"]
      gated_dF_centered_r=N    → result["d_F_full"] * result["null_ratio_centered_post_rankN"]
      additive_S_raw_r=N       → result["surprise"] + result["null_ratio_raw_post_rankN"]
      gated_dF_raw_r=N         → result["d_F_full"] * result["null_ratio_raw_post_rankN"]
    """
    parts = label.split("_")
    if len(parts) < 4:
        return None
    op = parts[0]                # "additive" or "gated"
    scalar_key = parts[1]        # "S" or "dF"
    base_family = parts[2]       # "fisher" / "centered" / "raw"
    rank_part = parts[3]         # "r=N"
    if not rank_part.startswith("r="):
        return None
    try:
        rank = int(rank_part[2:])
    except ValueError:
        return None

    if base_family == "fisher":
        base_col = f"null_ratio_post_rank{rank}"
    elif base_family == "centered":
        base_col = f"null_ratio_centered_post_rank{rank}"
    elif base_family == "raw":
        base_col = f"null_ratio_raw_post_rank{rank}"
    else:
        return None

    base = result.get(base_col)
    if base is None or not np.isfinite(base):
        return None

    if scalar_key == "S":
        scalar = result.get("surprise")
    elif scalar_key == "dF":
        scalar = result.get("d_F_full")
    else:
        return None
    if scalar is None or not np.isfinite(scalar):
        return None

    if op == "additive":
        return float(scalar + base)
    if op == "gated":
        return float(scalar * base)
    return None


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


def _nested_bootstrap_oob_auroc(
    score_matrix: np.ndarray,
    labels: np.ndarray,
    panel: List[PanelCell],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    """Nested bootstrap: at each round, resample the calibration set with
    replacement to form an in-bag set; re-run the WHOLE cell selection
    (best cell + sign-lock) on the in-bag samples; then evaluate the
    selected (cell, sign) on the out-of-bag samples. The OOB AUROC
    distribution is the honest deployment estimate — it accounts for the
    selection bias that contaminates in-sample stats.

    Also tracks how often each cell is selected across rounds; if one cell
    wins ≪ 100% of resamples, the selection is noisy at this n and the
    profile gets a `winner_unstable` warning.

    Returns a dict with the OOB summary stats. If no round produced a
    valid OOB AUROC (degenerate small-n case), returns a dict full of
    NaNs.
    """
    from sklearn.metrics import roc_auc_score

    n, n_cells = score_matrix.shape
    rng = np.random.RandomState(seed + 1)  # +1 so it doesn't clash with _bootstrap_auroc's stream
    oob_aurocs: List[float] = []
    winner_counts: Dict[int, int] = {j: 0 for j in range(n_cells)}

    for _ in range(n_bootstrap):
        in_bag = rng.randint(0, n, size=n)
        in_bag_set = set(in_bag.tolist())
        oob = np.array([i for i in range(n) if i not in in_bag_set], dtype=np.int64)
        if len(oob) < 4 or len(np.unique(labels[oob])) < 2:
            continue

        # Re-run cell selection inside this resample.
        best_j = -1
        best_distance = -1.0
        best_sign = 0
        for j in range(n_cells):
            s_in = score_matrix[in_bag, j]
            y_in = labels[in_bag]
            auc, sign, _ = _score_candidate(s_in, y_in)
            if np.isfinite(auc):
                d = abs(auc - 0.5)
                if d > best_distance:
                    best_distance = d
                    best_j = j
                    best_sign = sign
        if best_j < 0:
            continue
        winner_counts[best_j] += 1

        # Evaluate the in-bag-selected cell on OOB with the in-bag-locked sign.
        s_oob = score_matrix[oob, best_j] * best_sign
        y_oob = labels[oob]
        finite = np.isfinite(s_oob)
        if finite.sum() < 4 or len(np.unique(y_oob[finite])) < 2:
            continue
        oob_aurocs.append(float(roc_auc_score(y_oob[finite], s_oob[finite])))

    total_winners = sum(winner_counts.values())
    if total_winners > 0:
        max_count = max(winner_counts.values())
        winner_stability = max_count / total_winners
    else:
        winner_stability = float("nan")
    winner_counts_labeled = {
        _cell_label(panel[j]): c for j, c in winner_counts.items() if c > 0
    }
    if not oob_aurocs:
        return {
            "oob_auroc_median": float("nan"),
            "oob_auroc_ci_lo": float("nan"),
            "oob_auroc_ci_hi": float("nan"),
            "oob_n_bootstrap_used": 0,
            "winner_stability": float(winner_stability) if np.isfinite(winner_stability) else float("nan"),
            "winner_counts": winner_counts_labeled,
        }
    arr = np.array(oob_aurocs)
    return {
        "oob_auroc_median": float(np.median(arr)),
        "oob_auroc_ci_lo": float(np.percentile(arr, 2.5)),
        "oob_auroc_ci_hi": float(np.percentile(arr, 97.5)),
        "oob_n_bootstrap_used": int(len(oob_aurocs)),
        "winner_stability": float(winner_stability),
        "winner_counts": winner_counts_labeled,
    }


def _resolve_model_snapshot_sha(model_slug: str) -> Optional[str]:
    """Resolve the HuggingFace cache snapshot SHA for the model the calibrator
    will load. This pins the exact model artifact (not just the slug). Returns
    None if the model isn't cached locally or the path can't be parsed.

    The HF cache layout puts each downloaded snapshot under
        ~/.cache/huggingface/hub/models--{owner}--{repo}/snapshots/{commit_sha}/
    so the parent directory of any file we resolve gives us the SHA directly.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    # config.json is the most reliable sentinel — present in every HF repo.
    path = try_to_load_from_cache(repo_id=model_slug, filename="config.json")
    if not path:
        return None
    try:
        sha = Path(path).parent.name
    except Exception:
        return None
    # Sanity: HF commit SHAs are 40-char hex.
    if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha):
        return None
    return sha


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
    *,
    oob_auroc_median: Optional[float] = None,
    winner_stability: Optional[float] = None,
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
    # OOB-flavored warnings — these reflect the honest deployment estimate,
    # not the optimistically-biased in-sample stats. Added 2026-05-13 to
    # address the Codex review's selection-bias finding.
    if oob_auroc_median is not None and np.isfinite(oob_auroc_median):
        if oob_auroc_median < 0.60:
            w.append(
                f"oob_low_auroc (oob_median={oob_auroc_median:.3f}; <0.60 means the cell "
                f"selection didn't generalize beyond the calibration sample)"
            )
        if np.isfinite(best_auroc):
            gap = best_auroc - oob_auroc_median
            if gap > 0.15:
                w.append(
                    f"large_oob_in_sample_gap (in_sample={best_auroc:.3f}, "
                    f"oob_median={oob_auroc_median:.3f}, gap={gap:.3f}; "
                    f"in-sample AUROC is materially over-stated by selection bias)"
                )
    if winner_stability is not None and np.isfinite(winner_stability):
        if winner_stability < 0.70:
            w.append(
                f"winner_unstable (winner_stability={winner_stability:.2f}; "
                f"a different panel cell wins on >30% of bootstrap resamples — "
                f"the chosen cell is noise-driven at this n)"
            )
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration entry point
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CalibrationState:
    """Loaded model + everything needed to run calibrate_with_state(). Build
    once via load_calibration_state(model_slug, ...); reuse across multiple
    calibrations (e.g. ANLI R1/R2/R3 on the same model) to skip the model
    load cost. Not part of the persisted profile."""
    model_slug: str
    model: Any
    tokenizer: Any
    projection: Any
    layer_indices: Dict[str, int]
    pri_computer: Any
    prompt_strategy: Any
    layer_name: str
    seed: int


def _residualize_in_place(
    score_matrix: np.ndarray,
    panel: List[PanelCell],
    *,
    prompts_n: int,
) -> None:
    """For each `*_resid` cell in the panel, replace its column in the
    score matrix with `raw − predicted(raw | d_F_full)`. Linear regression
    is fit on (d_F_full, raw) pairs across the n calibration samples. NaNs
    in either column drop the sample from the fit AND from the residual
    output (residual stays NaN for that row).

    This is the E18 sealed acceptance variable from pri-v3-plan.md
    §Magnitude-independence test (lines 178-181). It strips the magnitude
    confound that raw `null_ratio` carries.
    """
    # Build column-index map for direct cells we need to reference.
    col_index: Dict[PanelCell, int] = {cell: j for j, cell in enumerate(panel)}
    # Find d_F_full column (must exist for residualization)
    d_F_cell: Optional[PanelCell] = None
    for cell in panel:
        if cell[1] == "scalar" and cell[2] == "d_F_full":
            d_F_cell = cell
            break
    if d_F_cell is None:
        # Without d_F_full in the panel we can't residualize. Leave the
        # resid columns as their base values; downstream cell scoring will
        # behave as if they're duplicates of the raw cells.
        return
    d_F_col_idx = col_index[d_F_cell]
    d_F_values = score_matrix[:, d_F_col_idx]

    for j, cell in enumerate(panel):
        if cell[1] not in DERIVED_RESID_FAMILIES:
            continue
        raw_values = score_matrix[:, j].copy()
        finite_mask = np.isfinite(raw_values) & np.isfinite(d_F_values)
        if finite_mask.sum() < 3:
            # Too few finite pairs to fit a regression; mark all as NaN.
            score_matrix[:, j] = np.nan
            continue
        x = d_F_values[finite_mask]
        y = raw_values[finite_mask]
        # OLS: y ≈ b0 + b1 * x. Use np.polyfit (degree=1) for numerical
        # stability + tiny dependency surface.
        b1, b0 = np.polyfit(x, y, 1)
        predicted = b0 + b1 * d_F_values
        residuals = raw_values - predicted
        # Preserve NaN where either input was NaN — don't fabricate residuals.
        residuals = np.where(finite_mask, residuals, np.nan)
        score_matrix[:, j] = residuals


def load_calibration_state(
    model_slug: str,
    *,
    layer_name: str = "final",
    seed: int = 20260512,
) -> CalibrationState:
    """Load the model + tokenizer + PRIComputer once. The returned state can
    be passed to `calibrate_with_state(state, ...)` repeatedly with different
    calibration jsonls without paying the model-load cost each time.
    """
    cfg = pipeline.Config()
    cfg.layers_to_probe = [layer_name]
    cfg.seed = seed
    model, tokenizer, projection, layer_indices = pipeline.load_model(model_slug, cfg)
    gamma = pipeline._extract_final_rmsnorm_gamma(model)
    if gamma is None:
        raise RuntimeError(
            f"could not extract final-RMSNorm gamma for {model_slug}; "
            f"check pri_v2_mlx_pipeline._extract_final_rmsnorm_gamma logs."
        )
    pri_computer = pipeline.PRIComputer(projection, final_norm_gamma=gamma)
    prompt_strategy = io_plugins.get_prompt_strategy(model_slug)
    return CalibrationState(
        model_slug=model_slug,
        model=model,
        tokenizer=tokenizer,
        projection=projection,
        layer_indices=layer_indices,
        pri_computer=pri_computer,
        prompt_strategy=prompt_strategy,
        layer_name=layer_name,
        seed=seed,
    )


def calibrate_with_state(
    state: CalibrationState,
    calibration_jsonl_path: str,
    *,
    task_label: str = "",
    panel: Optional[List[PanelCell]] = None,
    n_bootstrap: int = 1000,
    max_new_tokens: int = 8,
    alpha: float = 1.0,
) -> CalibrationProfile:
    """Run the calibration pass using a pre-loaded model state. This is the
    inner work — see `calibrate()` for the single-shot wrapper that loads +
    runs in one call.

    Use this when you want to calibrate the same model on multiple datasets
    (e.g. ANLI R1/R2/R3) without reloading the model each time.
    """
    panel = list(panel or DEFAULT_PANEL)
    prompts, labels, data_hash = _load_calibration_jsonl(calibration_jsonl_path)
    n_calibration = len(prompts)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    print(f"[calibrate] model={state.model_slug}  task={task_label or '(unset)'}")
    print(f"[calibrate] n_calibration={n_calibration} (pos={n_pos}, neg={n_neg})")
    print(f"[calibrate] panel={[_cell_label(c) for c in panel]}")

    # Per-sample × per-cell score matrix.
    n_cells = len(panel)
    score_matrix = np.full((n_calibration, n_cells), np.nan, dtype=np.float64)
    print(f"[calibrate] tracing {n_calibration} samples...")
    for i, prompt in enumerate(prompts):
        trace = _trace_one_prompt(
            state.model, state.tokenizer, state.projection, state.layer_indices,
            prompt, state.prompt_strategy, max_new_tokens,
        )
        per_cell = _compute_panel_scores_for_sample(
            state.pri_computer, trace, state.layer_name, panel, alpha=alpha,
        )
        for j, cell in enumerate(panel):
            v = per_cell.get(cell)
            if v is not None:
                score_matrix[i, j] = v
        if (i + 1) % 10 == 0 or i + 1 == n_calibration:
            print(f"[calibrate]   {i+1}/{n_calibration}")

    # Post-loop residualization pass (E18 sealed primary form). For each
    # `*_resid` cell, regress the cell's CURRENT score_matrix column
    # (which holds the raw null_ratio values at this point) against the
    # d_F_full column AT THE SAME STEP across all calibration samples, and
    # replace the column with residuals. Cells whose corresponding d_F_full
    # or base column doesn't exist on this sample are left as NaN.
    _residualize_in_place(score_matrix, panel, prompts_n=n_calibration)

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
        raise RuntimeError(
            "no candidate cell produced a finite AUROC — calibration data "
            "may be too small or all samples EOS'd before reaching panel steps."
        )

    best_cell = panel[best_idx]
    best_auroc = float(candidate_results[best_idx]["auroc"])
    best_sign = int(candidate_results[best_idx]["sign"])

    print(f"[calibrate] best cell: {_cell_label(best_cell)}  "
          f"AUROC={best_auroc:.3f}  sign={best_sign:+d}")

    # In-sample bootstrap CI on the locked sign (legacy semantics).
    ci_lo, ci_hi = _bootstrap_auroc(
        score_matrix[:, best_idx], labels, best_sign, n_bootstrap, state.seed,
    )
    print(f"[calibrate] in-sample 95%% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  "
          f"(n_bootstrap={n_bootstrap})")

    # Nested OOB bootstrap — the honest deployment estimate. Re-runs cell
    # selection inside each resample, evaluates on the out-of-bag samples.
    # 2026-05-13: added in response to the Codex adversarial review's
    # post-selection-bias finding.
    print(f"[calibrate] nested OOB bootstrap...")
    oob_stats = _nested_bootstrap_oob_auroc(
        score_matrix, labels, panel, n_bootstrap, state.seed,
    )
    if oob_stats["oob_n_bootstrap_used"] > 0:
        print(
            f"[calibrate] OOB median AUROC: {oob_stats['oob_auroc_median']:.3f}  "
            f"CI [{oob_stats['oob_auroc_ci_lo']:.3f}, {oob_stats['oob_auroc_ci_hi']:.3f}]  "
            f"(used {oob_stats['oob_n_bootstrap_used']}/{n_bootstrap} rounds)"
        )
        print(
            f"[calibrate] winner stability: {oob_stats['winner_stability']:.2f}  "
            f"counts: {oob_stats['winner_counts']}"
        )
    else:
        print("[calibrate] OOB bootstrap: 0/N usable rounds (calibration set "
              "too small or degenerate); deployment estimate unavailable")

    warnings_list = _emit_warnings(
        n_calibration, n_pos, n_neg, best_auroc, ci_lo, ci_hi, panel_eval_counts,
        oob_auroc_median=oob_stats["oob_auroc_median"],
        winner_stability=oob_stats["winner_stability"],
    )
    for w in warnings_list:
        print(f"[calibrate]   WARNING: {w}")

    pipeline_path = REPO_ROOT / "pri_v2_mlx_pipeline.py"
    io_plugins_path = REPO_ROOT / "pri_v2_io_plugins.py"
    model_adapters_path = REPO_ROOT / "model_adapters.py"
    model_snapshot_sha = _resolve_model_snapshot_sha(state.model_slug)
    profile = CalibrationProfile(
        schema_version=SCHEMA_VERSION,
        model={
            "slug": state.model_slug,
            "output_projection_kind": state.projection.mode,
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
            "layer": state.layer_name,
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
            # In-sample post-selection — kept for inspection but NOT deployable.
            "auroc": best_auroc,
            "auroc_bootstrap_ci_lo": ci_lo,
            "auroc_bootstrap_ci_hi": ci_hi,
            # OOB stats — the honest deployment estimate.
            "oob_auroc_median": oob_stats["oob_auroc_median"],
            "oob_auroc_ci_lo": oob_stats["oob_auroc_ci_lo"],
            "oob_auroc_ci_hi": oob_stats["oob_auroc_ci_hi"],
            "oob_n_bootstrap_used": oob_stats["oob_n_bootstrap_used"],
            "winner_stability": oob_stats["winner_stability"],
            "winner_counts": oob_stats["winner_counts"],
            "candidate_panel": candidate_results,
        },
        provenance={
            "calibration_seed": int(state.seed),
            "n_bootstrap": int(n_bootstrap),
            "pipeline_module_hash_sha256": _hash_file(pipeline_path),
            "io_plugins_module_hash_sha256": _hash_file(io_plugins_path),
            "model_adapters_module_hash_sha256": _hash_file(model_adapters_path),
            "calibrator_module_hash_sha256": _hash_file(REPO_ROOT / "pri_calibrator.py"),
            "model_snapshot_sha": model_snapshot_sha,  # may be None if uncached
            "calibrated_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "max_new_tokens": int(max_new_tokens),
        },
        warnings=warnings_list,
    )
    return profile


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
    """Single-shot calibration: load the model then run the calibration pass.

    Thin wrapper around `load_calibration_state` + `calibrate_with_state`.
    Use the two-step form directly when you want to calibrate the same model
    on multiple datasets without reloading.

    `max_new_tokens` defaults to 8 — enough to cover gen_step ∈ {1..5} for the
    default panel (max panel step is 4; one extra for safety).
    """
    state = load_calibration_state(model_slug, layer_name=layer_name, seed=seed)
    return calibrate_with_state(
        state,
        calibration_jsonl_path,
        task_label=task_label,
        panel=panel,
        n_bootstrap=n_bootstrap,
        max_new_tokens=max_new_tokens,
        alpha=alpha,
    )


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

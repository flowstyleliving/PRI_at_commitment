#!/usr/bin/env python3
"""Chord-vs-Path Fisher diagnostic — is the chord-based d_F losing signal?

For each calibration sample at gen_step=1, computes:

  * d_F_chord = √((h_final - h_layer_0)^T F_final (h_final - h_layer_0))
       — magnitude of the cumulative residual under the FINAL-layer Fisher
         metric F_final.

  * d_F_path_fixed = Σ_ℓ √(Δh_ℓ^T F_final Δh_ℓ)        ← PRIMARY METRIC
       — path integral of the SAME final-layer metric F_final along the
         depth trajectory. Since chord and path use the same quadratic form,
         the triangle inequality applies: `d_F_path_fixed ≥ d_F_chord`,
         with equality iff the trajectory is collinear in F_final-space.
         `curvature_fixed = d_F_path_fixed - d_F_chord ≥ 0` is then a
         well-defined "wandering" measure.

  * d_F_path_varying = Σ_ℓ √(Δh_ℓ^T F_ℓ Δh_ℓ)         ← DESCRIPTIVE ONLY
       — path integral with the Fisher metric F_ℓ varying per layer (logit-
         lens p_ℓ at each intermediate hidden state). NOT directly comparable
         to d_F_chord because they're different quadratic forms; reported
         alongside for inspection but the decision rule does NOT apply.
         2026-05-15 fix in response to Codex review: the previous version
         used varying metric for path but compared against fixed-metric
         chord, which violated the triangle-inequality framing.

Reports:
  * Pearson correlation(chord, path_fixed) — the headline diagnostic
  * Mean ratio path_fixed/chord (always ≥ 1 by triangle inequality)
  * AUROC of each (chord, path_fixed, path_varying, curvature) vs label

Decision rule (sealed BEFORE looking at the numbers; applies to FIXED-metric
path only, NOT path_varying):
  * corr(chord, path_fixed) > 0.95: path collapses to chord — chord-based
    primitives are fine.
  * 0.7 < corr ≤ 0.95: path adds independent information — evaluate
    `curvature_fixed` as a single sealed cell at calibrator-level.
  * corr ≤ 0.7: chord throws away significant signal — replacement-grade
    question opens; consider re-deriving the panel on path quantities.

Requires v3_capture=True (every-layer hidden states at gen_step=0).
Approximation: logit-lens at intermediate layers uses the final-layer
RMSNorm γ.

Usage:
    .venv/bin/python scripts/diagnose_chord_vs_path.py \\
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
        --data /tmp/calibration_n30.jsonl \\
        --out /tmp/chord_vs_path.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_v2_mlx_pipeline as pipeline
import pri_v2_io_plugins as io_plugins
from pri_calibrator import _load_calibration_jsonl


# ─────────────────────────────────────────────────────────────────────────────
# Logit-lens probability at an intermediate layer
# ─────────────────────────────────────────────────────────────────────────────


def _logit_lens_probs(
    h_layer: np.ndarray,
    projection: pipeline.OutputProjection,
    final_norm_gamma: np.ndarray,
) -> np.ndarray:
    """softmax(W_u · rmsnorm(h_layer)). Applies the FINAL-layer RMSNorm
    γ to every intermediate hidden state — standard logit-lens trick.
    Approximation; the correlation diagnostic is robust to small per-layer
    scaling differences."""
    h_norm = pipeline.PRIComputer.rmsnorm(h_layer, final_norm_gamma)
    logits = projection.project(h_norm)  # (V,)
    return pipeline.safe_softmax(logits.astype(np.float32))


def _fisher_d_F_full(
    pri_computer: pipeline.PRIComputer,
    h_t: np.ndarray,
    h_prev: np.ndarray,
    p_t: np.ndarray,
) -> float:
    """Compute `d_F_full` magnitude — √(Δh_post^T F Δh_post) at layer
    where F is the uncentered Fisher pullback W_u^T diag(p) W_u.
    Mirrors PRIComputer.compute_step's d_F_full path but bypasses the
    per-step cosine/L2/null_ratio/etc. baggage so we can call it L times
    per sample cheaply."""
    gamma = pri_computer.final_norm_gamma
    h_t_post = pri_computer.rmsnorm(h_t, gamma)
    h_prev_post = pri_computer.rmsnorm(h_prev, gamma)
    dh_post = h_t_post - h_prev_post
    z = pri_computer._project(h_t - h_prev)  # raw dh through projection
    if z.shape[0] != p_t.shape[0]:
        m = min(z.shape[0], p_t.shape[0])
        z = z[:m]
        p_t = p_t[:m]
        p_t = p_t / (np.sum(p_t) + 1e-10)
    return pri_computer.fim_full_from_proj(z, p_t)


# ─────────────────────────────────────────────────────────────────────────────
# Path/chord computation for one sample
# ─────────────────────────────────────────────────────────────────────────────


def _chord_and_path_for_sample(
    trace: Dict[str, Any],
    pri_computer: pipeline.PRIComputer,
    projection: pipeline.OutputProjection,
    final_norm_gamma: np.ndarray,
) -> Optional[Dict[str, float]]:
    """Extract gen_step=0's depth trajectory from a v3_capture trace,
    compute chord d_F and path d_F. Returns None if the model EOS'd
    before producing any generated token or the capture is missing.
    """
    captures = trace.get("gen_captures_by_step", [])
    if not captures:
        return None
    step0_captures = captures[0]  # gen_step=1 in parquet convention
    # Layer names: "layer_00", "layer_01", ..., plus paper aliases.
    # Filter to "layer_NN" canonical entries and sort by index.
    layer_items: List[tuple[int, np.ndarray]] = []
    for name, payload in step0_captures.items():
        if not name.startswith("layer_"):
            continue
        try:
            idx = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        layer_items.append((idx, payload["h_t"]))
    if len(layer_items) < 2:
        return None
    layer_items.sort(key=lambda x: x[0])
    h_by_layer = [vec for _, vec in layer_items]
    L = len(h_by_layer)  # number of layer captures (e.g., 28 for Llama)

    # ── Resolve the FINAL-layer Fisher metric ONCE (canonical p_final) ──
    h_0 = h_by_layer[0]
    h_L = h_by_layer[-1]
    p_final = _logit_lens_probs(h_L, projection, final_norm_gamma)

    # ── d_F_path_fixed: SAME metric F_final integrated along the trajectory.
    # ── This is the math-honest "is the chord a faithful summary?" comparison.
    # ── Triangle inequality applies: d_F_path_fixed >= d_F_chord by construction.
    path_fixed: List[float] = []
    for ell in range(1, L):
        h_prev = h_by_layer[ell - 1]
        h_t = h_by_layer[ell]
        try:
            d_F_ell = _fisher_d_F_full(pri_computer, h_t, h_prev, p_final)
        except Exception:
            d_F_ell = float("nan")
        path_fixed.append(d_F_ell)
    path_fixed_arr = np.array(path_fixed, dtype=np.float64)
    if not np.all(np.isfinite(path_fixed_arr)):
        return None
    d_F_path_fixed = float(np.sum(path_fixed_arr))

    # ── d_F_path_varying: per-layer Fisher (logit-lens at each layer).
    # ── DESCRIPTIVE ONLY — different quadratic form per segment, NO triangle
    # ── inequality. Reported for inspection but NOT used by the decision rule.
    path_varying: List[float] = []
    for ell in range(1, L):
        h_prev = h_by_layer[ell - 1]
        h_t = h_by_layer[ell]
        p_ell = _logit_lens_probs(h_t, projection, final_norm_gamma)
        try:
            d_F_ell = _fisher_d_F_full(pri_computer, h_t, h_prev, p_ell)
        except Exception:
            d_F_ell = float("nan")
        path_varying.append(d_F_ell)
    path_varying_arr = np.array(path_varying, dtype=np.float64)
    d_F_path_varying = (
        float(np.sum(path_varying_arr))
        if np.all(np.isfinite(path_varying_arr))
        else float("nan")
    )

    # ── Chord d_F: same F_final on the (h_L - h_0) displacement ──
    try:
        d_F_chord = _fisher_d_F_full(pri_computer, h_L, h_0, p_final)
    except Exception:
        return None
    if not np.isfinite(d_F_chord):
        return None

    return {
        "d_F_chord": d_F_chord,
        "d_F_path_fixed": d_F_path_fixed,
        "d_F_path_varying": d_F_path_varying,
        "curvature_fixed": d_F_path_fixed - d_F_chord,   # ≥ 0 by triangle ineq
        "n_layers": L,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="chord-vs-path Fisher diagnostic")
    p.add_argument("--model", required=True, help="model slug to load")
    p.add_argument("--data", required=True, help="calibration jsonl (uses same format as pri_calibrator)")
    p.add_argument("--out", default="/tmp/chord_vs_path.csv", help="output CSV path")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="cap n samples (default: all)")
    args = p.parse_args()

    prompts, labels, _ = _load_calibration_jsonl(args.data)
    if args.limit:
        prompts, labels = prompts[: args.limit], labels[: args.limit]
    print(f"[diagnose] {len(prompts)} samples, model={args.model}")

    # Load model + v3-capture-enabled Config
    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = True  # critical — without this, no every-layer captures
    model, tokenizer, projection, layer_indices = pipeline.load_model(args.model, cfg)
    gamma = pipeline._extract_final_rmsnorm_gamma(model)
    if gamma is None:
        raise RuntimeError(f"no final-RMSNorm gamma for {args.model}")
    pri_computer = pipeline.PRIComputer(projection, final_norm_gamma=gamma)
    prompt_strategy = io_plugins.get_prompt_strategy(args.model)

    rows: List[Dict[str, Any]] = []
    print(f"[diagnose] tracing {len(prompts)} samples with v3_capture=True ...")
    for i, prompt in enumerate(prompts):
        wrapped = prompt_strategy(prompt, tokenizer)
        try:
            trace = pipeline.trace_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=wrapped,
                layer_indices=layer_indices,
                output_projection=projection,
                max_new_tokens=args.max_new_tokens,
                v3_capture=True,
            )
        except Exception as e:
            print(f"[diagnose]   sample {i}: trace FAILED ({e})")
            continue
        result = _chord_and_path_for_sample(trace, pri_computer, projection, gamma)
        if result is None:
            continue
        result["sample_idx"] = i
        result["label"] = int(labels[i])
        rows.append(result)
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            print(f"[diagnose]   {i+1}/{len(prompts)}")

    if not rows:
        raise SystemExit("no usable samples")

    # ── Write CSV (proper quoting via csv module) ──
    import csv as _csv
    with open(args.out, "w", newline="") as f:
        w = _csv.writer(f, quoting=_csv.QUOTE_MINIMAL)
        w.writerow([
            "sample_idx", "label", "n_layers",
            "d_F_chord", "d_F_path_fixed", "d_F_path_varying",
            "curvature_fixed", "path_fixed_over_chord",
        ])
        for r in rows:
            ratio = (
                r["d_F_path_fixed"] / r["d_F_chord"]
                if r["d_F_chord"] > 0 else float("nan")
            )
            w.writerow([
                r["sample_idx"], r["label"], r["n_layers"],
                f"{r['d_F_chord']:.6f}",
                f"{r['d_F_path_fixed']:.6f}",
                f"{r['d_F_path_varying']:.6f}",
                f"{r['curvature_fixed']:.6f}",
                f"{ratio:.6f}",
            ])
    print(f"[diagnose] wrote {len(rows)} rows to {args.out}")

    # ── Summary stats ──
    chords = np.array([r["d_F_chord"] for r in rows])
    paths_fixed = np.array([r["d_F_path_fixed"] for r in rows])
    paths_varying = np.array([r["d_F_path_varying"] for r in rows])
    curvs = np.array([r["curvature_fixed"] for r in rows])
    labels_arr = np.array([r["label"] for r in rows])

    corr_fixed = float(np.corrcoef(chords, paths_fixed)[0, 1])
    finite_pv = np.isfinite(paths_varying)
    corr_varying = (
        float(np.corrcoef(chords[finite_pv], paths_varying[finite_pv])[0, 1])
        if finite_pv.sum() >= 2 else float("nan")
    )
    ratio_fixed = float(np.mean(paths_fixed / np.maximum(chords, 1e-12)))
    min_ratio_fixed = float(np.min(paths_fixed / np.maximum(chords, 1e-12)))

    print()
    print("=" * 80)
    print(f"  Chord-vs-path summary (n={len(rows)})")
    print("=" * 80)
    print(f"  PRIMARY (fixed final-layer F, triangle-inequality applies):")
    print(f"    Pearson correlation(chord, path_fixed): {corr_fixed:.4f}")
    print(f"    Mean path_fixed / chord:                {ratio_fixed:.4f}  (≥1 by triangle ineq)")
    print(f"    Min  path_fixed / chord:                {min_ratio_fixed:.4f}  (should be ≥1; <1 = numerical bug)")
    print()
    print(f"  DESCRIPTIVE (per-layer logit-lens F_ℓ varying — different quadratic forms per segment):")
    print(f"    Pearson correlation(chord, path_varying): {corr_varying:.4f}")
    print()
    print(f"  chord:           mean={chords.mean():.3f}  std={chords.std():.3f}")
    print(f"  path_fixed:      mean={paths_fixed.mean():.3f}   std={paths_fixed.std():.3f}")
    print(f"  path_varying:    mean={paths_varying[finite_pv].mean():.3f}   std={paths_varying[finite_pv].std():.3f}")
    print(f"  curvature_fixed: mean={curvs.mean():.3f}   std={curvs.std():.3f}  (must be ≥ 0 per sample)")

    # Triangle-inequality sanity check on fixed-metric path
    n_violations = int(np.sum(curvs < -1e-6))
    if n_violations > 0:
        print(f"  WARNING: {n_violations}/{len(rows)} samples have curvature_fixed < 0 "
              f"— numerical issue (NOT a real triangle-inequality violation)")

    # ── AUROC of each vs label (sign-agnostic) ──
    if len(np.unique(labels_arr)) > 1:
        from sklearn.metrics import roc_auc_score
        def auroc_signed(s):
            mask = np.isfinite(s)
            auc = roc_auc_score(labels_arr[mask], s[mask])
            return max(auc, 1 - auc), (1 if auc >= 0.5 else -1)
        c_a, c_s = auroc_signed(chords)
        pf_a, pf_s = auroc_signed(paths_fixed)
        pv_a, pv_s = auroc_signed(paths_varying)
        cv_a, cv_s = auroc_signed(curvs)
        print()
        print(f"  AUROC (sign-agnostic) vs contradiction label:")
        print(f"    d_F_chord:        {c_a:.4f}  sign={c_s:+d}")
        print(f"    d_F_path_fixed:   {pf_a:.4f}  sign={pf_s:+d}")
        print(f"    d_F_path_varying: {pv_a:.4f}  sign={pv_s:+d}")
        print(f"    curvature_fixed:  {cv_a:.4f}  sign={cv_s:+d}")

    print()
    print("Decision rule (applies to FIXED-metric path only):")
    if corr_fixed > 0.95:
        print(f"  → corr_fixed > 0.95 — path collapses to chord; current chord-based")
        print(f"    primitives retain primacy. Path proposal is decorative.")
    elif corr_fixed > 0.7:
        print(f"  → 0.7 < corr_fixed ≤ 0.95 — path adds independent information.")
        print(f"    Worth evaluating `curvature_fixed` as a single sealed cell.")
    else:
        print(f"  → corr_fixed ≤ 0.7 — chord throws away significant signal.")
        print(f"    Replacement-grade question opens: re-derive panel on path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

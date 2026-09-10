#!/usr/bin/env python3
"""Δσ_onaxis diagnostic — Shannon entropy of the on-axis energy distribution.

For each calibration sample at gen_step=1, computes the standard sealed-v3
quantities (dh_post, p_t, SVD of √p_t·W_s on top-`support` rows) and emits:

  * null_ratio_post_rank{r}     — fraction of ‖dh_post‖² OFF the top-r axes.
                                  (Identical to PRIComputer.null_ratio_and_energy.)

  * delta_sigma_onaxis_rank{r}  — Shannon entropy of the per-axis energy
                                  distribution INSIDE the top-r axes:
                                    e_i = (V_i · dh_post)²,   i = 0..r-1
                                    q_i = e_i / Σ_{j<r} e_j
                                    H_r = -Σ q_i log(q_i + eps)
                                  Theoretically motivated by the SUP frame
                                  ℏ = √(Δμ · Δσ): Δμ ~ (1 − null_ratio) is the
                                  on-axis concentration, Δσ ~ H_r is the on-axis
                                  flexibility/dispersion.

  * delta_sigma_onaxis_norm_rank{r} — H_r / log(r), in [0, 1] for direct
                                       cross-rank comparison.

  * fisher_energy_rank{r}        — Σ_{i≤r} σ_i² / Σ_i σ_i² (same as sealed v3).

The bivariate (null_ratio, Δσ_onaxis) is the headline. Predicted signatures:
  · truthful           — low null_ratio + low Δσ_onaxis (one dominant axis)
  · uncertain          — low null_ratio + high Δσ_onaxis (spread across axes)
  · contradiction      — high null_ratio + high Δσ_onaxis (off-axis + thrashing)
  · confident-hallu.   — low null_ratio + low Δσ_onaxis (sharp, but wrong axis)

This panel cannot separate truthful from confident-hallucination (no label),
but it CAN test:
  · does Δσ_onaxis carry signal vs contradiction-label by itself?
  · does the bivariate (null_ratio, Δσ_onaxis) beat null_ratio alone?
  · what's corr(null_ratio, Δσ_onaxis)? — orthogonality check.

Rank 1 is degenerate (single axis → H=0); reported but uninformative.
Default rank sweep: {2, 4, 8, 16, 32}.

Usage:
    .venv/bin/python scripts/diagnose_delta_sigma_onaxis.py \\
        --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
        --data /tmp/calibration_n100.jsonl \\
        --out /tmp/delta_sigma_onaxis.csv
"""
from __future__ import annotations

import argparse
import csv as _csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_v2_mlx_pipeline as pipeline
import pri_v2_io_plugins as io_plugins
from pri_calibrator import _load_calibration_jsonl

DEFAULT_RANKS: Tuple[int, ...] = (2, 4, 8, 16, 32)
MIN_AUROC_SAMPLES = 5


def _prepare_output_path(output_arg: str) -> Path:
    out_path = Path(output_arg).expanduser().resolve()
    parent = out_path.parent
    if out_path.exists() and out_path.is_dir():
        raise SystemExit(f"output path is a directory, expected a file: {out_path}")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise SystemExit(f"output parent is not a directory: {parent}")
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=parent, prefix=".write_probe.", suffix=".tmp", delete=True):
            pass
    except OSError as exc:
        raise SystemExit(f"output path is not writable: {out_path} ({exc})") from exc
    return out_path


def _compute_delta_sigma_per_sample(
    dh_post: np.ndarray,
    p_t: np.ndarray,
    projection: pipeline.OutputProjection,
    ranks: Tuple[int, ...],
) -> Optional[Dict[str, float]]:
    """Run the sealed-v3 SVD once and emit per-rank
    {null_ratio_post, delta_sigma_onaxis, delta_sigma_onaxis_norm, fisher_energy}.

    Mirrors PRIComputer.null_ratio_and_energy's support-truncated SVD so the
    null_ratio columns are bit-equivalent (modulo float reordering) to the
    sealed pipeline. Per-axis projection energies are reused for entropy
    rather than discarded.
    """
    dh_norm_sq = float(np.dot(dh_post, dh_post))
    if dh_norm_sq <= 0.0:
        return None
    dh_norm = float(np.sqrt(dh_norm_sq))

    max_rank = max(ranks)
    support = int(min(max(256, max_rank * 16), p_t.shape[0]))
    idx = np.argpartition(-p_t, kth=support - 1)[:support]
    p_s = p_t[idx]
    W_s = projection.get_rows(idx)
    if W_s is None or W_s.ndim != 2:
        return None

    A = (np.sqrt(p_s + 1e-10)[:, None]) * W_s
    try:
        _, S, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    s_sq = S ** 2
    total_energy = float(np.sum(s_sq)) + 1e-10
    cum_energy = np.cumsum(s_sq)

    proj = Vt @ dh_post                # shape: (max_available,)
    proj_sq = proj ** 2                 # per-axis energies
    cum_proj_sq = np.cumsum(proj_sq)
    max_available = Vt.shape[0]

    eps = 1e-12
    out: Dict[str, float] = {}
    for r in ranks:
        r_eff = int(min(r, max_available))
        # null_ratio_post_rank{r}
        top_proj_sq = float(cum_proj_sq[r_eff - 1]) if r_eff > 0 else 0.0
        null_sq = max(dh_norm_sq - top_proj_sq, 0.0)
        out[f"null_ratio_post_rank{r}"] = float(np.sqrt(null_sq) / dh_norm)

        # fisher_energy_rank{r}
        out[f"fisher_energy_rank{r}"] = (
            float(cum_energy[r_eff - 1] / total_energy) if r_eff > 0 else 0.0
        )

        # Δσ_onaxis: entropy over the top-r axes' projection energies
        if r_eff < 2:
            out[f"delta_sigma_onaxis_rank{r}"] = 0.0
            out[f"delta_sigma_onaxis_norm_rank{r}"] = 0.0
            continue
        e = proj_sq[:r_eff]
        on_axis_sum = float(e.sum())
        if on_axis_sum <= 0.0:
            out[f"delta_sigma_onaxis_rank{r}"] = float("nan")
            out[f"delta_sigma_onaxis_norm_rank{r}"] = float("nan")
            continue
        q = e / on_axis_sum
        H = float(-np.sum(q * np.log(q + eps)))
        out[f"delta_sigma_onaxis_rank{r}"] = H
        out[f"delta_sigma_onaxis_norm_rank{r}"] = float(H / math.log(r_eff))

    return out


def _trace_sample_dh_p(
    trace: Dict[str, Any],
    final_layer_name: str,
    final_norm_gamma: np.ndarray,
    projection: pipeline.OutputProjection,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract gen_step=1 final-layer (dh_post, p_t).

    h_t  = first generated token's final-layer raw hidden state
    h_prev = last prefix final-layer raw hidden state
    dh_post = rmsnorm(h_t, γ) - rmsnorm(h_prev, γ)
    p_t = softmax(W_u · rmsnorm(h_t, γ))
    """
    try:
        h_t = trace["gen_hidden"][final_layer_name][0]
        h_prev = trace["last_prefix_hidden"][final_layer_name]
    except (KeyError, IndexError):
        return None
    if h_t is None or h_prev is None:
        return None

    h_t_post = pipeline.PRIComputer.rmsnorm(h_t, final_norm_gamma)
    h_prev_post = pipeline.PRIComputer.rmsnorm(h_prev, final_norm_gamma)
    dh_post = (h_t_post - h_prev_post).astype(np.float32)

    logits = projection.project(h_t_post).astype(np.float32)
    p_t = pipeline.safe_softmax(logits)
    return dh_post, p_t


def _write_rows_csv(out_path: Path, rows: List[Dict[str, Any]], ranks: Tuple[int, ...]) -> None:
    header = ["sample_idx", "label"]
    for r in ranks:
        header.extend([
            f"null_ratio_post_rank{r}",
            f"delta_sigma_onaxis_rank{r}",
            f"delta_sigma_onaxis_norm_rank{r}",
            f"fisher_energy_rank{r}",
        ])
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", dir=out_path.parent,
            prefix=f".{out_path.stem}.", suffix=f"{out_path.suffix or '.csv'}.tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            w = _csv.writer(f, quoting=_csv.QUOTE_MINIMAL)
            w.writerow(header)
            for row in rows:
                rec = [row["sample_idx"], row["label"]]
                for r in ranks:
                    for col in (
                        f"null_ratio_post_rank{r}",
                        f"delta_sigma_onaxis_rank{r}",
                        f"delta_sigma_onaxis_norm_rank{r}",
                        f"fisher_energy_rank{r}",
                    ):
                        v = row.get(col, float("nan"))
                        rec.append(f"{v:.6f}" if isinstance(v, float) and math.isfinite(v) else "nan")
                w.writerow(rec)
        os.replace(tmp_path, out_path)
    except OSError as exc:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise SystemExit(f"failed to write {out_path}: {exc}") from exc


def _auroc(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, str]:
    """Sign-agnostic AUROC. Returns (auroc, sign) where sign='+' or '-'."""
    mask = np.isfinite(scores) & np.isfinite(labels)
    if mask.sum() < MIN_AUROC_SAMPLES:
        return float("nan"), "?"
    s = scores[mask]
    y = labels[mask].astype(int)
    if len(np.unique(y)) < 2 or np.isclose(s.std(), 0.0):
        return float("nan"), "?"
    a = pipeline.safe_auroc(y, s)
    return (a, "+") if a >= 0.5 else (float(1 - a), "-")


def _corr(xs: np.ndarray, ys: np.ndarray) -> float:
    m = np.isfinite(xs) & np.isfinite(ys)
    if m.sum() < 3:
        return float("nan")
    a, b = xs[m], ys[m]
    if np.isclose(a.std(), 0) or np.isclose(b.std(), 0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    p = argparse.ArgumentParser(description="Δσ_onaxis diagnostic — single-pass entropy of on-axis energy")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="/tmp/delta_sigma_onaxis.csv")
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="cap n samples (default: all)")
    p.add_argument("--ranks", default=",".join(str(r) for r in DEFAULT_RANKS),
                   help="comma-separated rank sweep (default: 2,4,8,16,32)")
    args = p.parse_args()

    ranks = tuple(int(x) for x in args.ranks.split(",") if x.strip())
    if not ranks or min(ranks) < 1:
        raise SystemExit("--ranks must be positive integers")

    out_path = _prepare_output_path(args.out)

    prompts, labels, _ = _load_calibration_jsonl(args.data)
    if args.limit:
        prompts, labels = prompts[: args.limit], labels[: args.limit]
    print(f"[Δσ_onaxis] {len(prompts)} samples, ranks={ranks}, model={args.model}")

    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False  # only need final-layer h_t + last_prefix_hidden
    model, tokenizer, projection, layer_indices = pipeline.load_model(args.model, cfg)
    gamma = pipeline._extract_final_rmsnorm_gamma(model)
    if gamma is None:
        raise RuntimeError(f"no final-RMSNorm gamma for {args.model}")
    final_name = "final"
    prompt_strategy = io_plugins.get_prompt_strategy(args.model)

    rows: List[Dict[str, Any]] = []
    print(f"[Δσ_onaxis] tracing {len(prompts)} samples ...")
    for i, prompt in enumerate(prompts):
        wrapped = prompt_strategy(prompt, tokenizer)
        try:
            trace = pipeline.trace_sample(
                model=model, tokenizer=tokenizer, prompt=wrapped,
                layer_indices=layer_indices, output_projection=projection,
                max_new_tokens=args.max_new_tokens, v3_capture=False,
            )
        except Exception as e:
            print(f"[Δσ_onaxis]   sample {i}: trace FAILED ({e})")
            continue

        dhp = _trace_sample_dh_p(trace, final_name, gamma, projection)
        if dhp is None:
            continue
        dh_post, p_t = dhp

        result = _compute_delta_sigma_per_sample(dh_post, p_t, projection, ranks)
        if result is None:
            continue
        result["sample_idx"] = i
        result["label"] = int(labels[i])
        rows.append(result)
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            print(f"[Δσ_onaxis]   {i+1}/{len(prompts)}")

    if not rows:
        raise SystemExit("no usable samples")

    _write_rows_csv(out_path, rows, ranks)
    print(f"[Δσ_onaxis] wrote {len(rows)} rows to {out_path}")

    # ── Summary ──
    y = np.array([r["label"] for r in rows], dtype=np.float64)
    print()
    print("=" * 80)
    print(f"  Δσ_onaxis summary (n={len(rows)})")
    print("=" * 80)
    print(f"  rank | AUROC null  AUROC Δσ_n  AUROC null·Δσ  corr(null,Δσ_n)  fisher_energy")
    print(f"  -----+----------------------------------------------------------------------")
    for r in ranks:
        nr = np.array([row[f"null_ratio_post_rank{r}"] for row in rows])
        ds = np.array([row[f"delta_sigma_onaxis_norm_rank{r}"] for row in rows])
        fe = np.array([row[f"fisher_energy_rank{r}"] for row in rows])
        biv = nr * ds  # unsupervised SUP-flavored composite
        au_nr, sgn_nr = _auroc(y, nr)
        au_ds, sgn_ds = _auroc(y, ds)
        au_bv, sgn_bv = _auroc(y, biv)
        c = _corr(nr, ds)
        print(f"  {r:>4} | "
              f"{au_nr:.4f} ({sgn_nr})  "
              f"{au_ds:.4f} ({sgn_ds})  "
              f"{au_bv:.4f} ({sgn_bv})    "
              f"{c:+.4f}          "
              f"{float(fe.mean()):.4f}")
    print()
    print("  Bivariate (null·Δσ) is the SUP-flavored unsupervised composite.")
    print("  AUROC sign: '+' means HIGHER value predicts contradiction; '-' is flipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

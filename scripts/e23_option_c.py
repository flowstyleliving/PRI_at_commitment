#!/usr/bin/env python3
"""
E23 — sharpness-aware Option C prototype (pre-v3 Prereq 6).

Motivation (from SUP spectral-band verdict + E22):
  Option B (per-layer logit-lens eigenspace with sqrt(p^(ℓ)) weighting) is
  dominated by p-sharpness — as p^(ℓ) concentrates, A_ℓ = sqrt(p_s^(ℓ))·W_s^(ℓ)
  becomes near-rank-1 by construction and the SVD measures p-sharpness, not
  Fisher geometry. Option A (fixed final-p eigenspace, E22 default) avoids this
  but throws away per-layer distribution information.

  Option C softens the weighting: A_ℓ_α = (p_s^(ℓ))^α · W_s^(ℓ) with α < 1.
  At α=0 the weighting is uniform-over-support (pure W_u geometry); at α=1 we
  reproduce Option B; in between we interpolate.

For each Llama sample × layer ℓ × α × support:
  1. Apply model final norm to raw block output before logit-lens:
         p^(ℓ) = softmax(W_u · norm(h_t^(ℓ))).
  2. Support = top-`support` of p^(ℓ); swept over SUPPORTS to separate cutoff
     artefacts from genuine α effects.
  3. A_α = (p_s^(ℓ))^α [:,None] · W_s^(ℓ); SVD → V_α[:rank].
  4. null_ratio_C_α = ||Δh_ℓ − V_α[:r]^T V_α[:r] Δh_ℓ|| / ||Δh_ℓ||.

Also carried through for direct comparison:
  - null_ratio_A: Option A (single final-p eigenspace, built from NORMED final
    hidden — i.e. the real commit-distribution eigenspace, not the raw-block
    one used by the unpatched E22 helper).
  - p_ell_entropy, p_ell_top1 (per-layer, post-norm).
  - support_mass_supp{S}, support_sig_supp{S} (audit trail for support choice).

Success criterion (reported for three partitions: all / layer=0 / layer>0):
  at least one α ∈ {0.0, 0.25, 0.5} × support yields
    |corr(null_ratio_C_α,ℓ , H[p^(ℓ)])| < 0.3
  across layers (ideally in layer>0), with depth structure preserved.

Scope: Llama 3B only (prototype). n=4/cell. rank=32 only. Replicate on Mistral/Qwen
only if a winner emerges.

Output: experiments/e23-option-c/<YYYY-MM-DD>/run-NN/llama_e23_fixed.parquet
  (underscore-fixed suffix indicates post-review build that applies final norm
  before logit-lens, sweeps support, and persists support metadata.)
Reuses: e22_direction_depth.forward_all_layers_two_pos, greedy_commit_token,
final_p_t_eigenspace, synthetic_logic_loader, pri_v2_mlx_pipeline primitives.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx
from mlx_lm import load as mlx_load

from pri_v2_mlx_pipeline import (
    OutputProjection,
    find_layers,
    safe_softmax,
    to_numpy,
    encode_text,
)
from synthetic_logic_loader import (
    SyntheticLogicConfig,
    generate_synthetic_logic_dataset,
)
from scripts.e22_direction_depth import (
    forward_all_layers_two_pos,
    greedy_commit_token,
    final_p_t_eigenspace,
    model_slug,
)
from scripts._paths import experiment_run_dir


# ---- Config ----

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
N_PER_CELL = 4
SUPPORTS = (128, 256, 512)  # Sweep — fixed SUPPORT=256 was the pre-fix cutoff.
RANK = 32
ALPHAS = (0.0, 0.25, 0.5, 1.0)  # α=1.0 reproduces Option B (known-bad reference).

EXPERIMENT_SLUG = "e23-option-c"


def alpha_key(a: float) -> str:
    return f"{a:g}".replace(".", "p")  # 0.25 -> "0p25"


# ---- Final-norm helper ----


def apply_final_norm(model: Any, h_np: np.ndarray) -> np.ndarray:
    """Apply the model's final norm to a raw block-output hidden vector.

    Real next-token logits go through core.norm(h) before W_u (see
    e22_direction_depth.greedy_commit_token). The unpatched E23 built its
    per-layer logit-lens on raw block output, so the support set was chosen
    in the wrong space. This helper fixes that for both Option A (final-p
    eigenspace) and per-layer Option C.
    """
    core = model.model if hasattr(model, "model") else model
    h_mx = mx.array(h_np.astype(np.float32)[None, None, :])  # [1, 1, d]
    if hasattr(core, "norm"):
        h_mx = core.norm(h_mx)
    elif hasattr(core, "final_layernorm"):
        h_mx = core.final_layernorm(h_mx)
    else:
        raise RuntimeError("Model core has no norm / final_layernorm attribute.")
    mx.eval(h_mx)
    return to_numpy(h_mx)[0, 0].astype(np.float32)  # [d]


def support_signature(idx: np.ndarray) -> int:
    """Stable 31-bit hash of a support-index set, order-invariant."""
    return int(hash(np.sort(idx).tobytes()) % (2**31 - 1))


# ---- Per-layer Option C eigenspaces + null_ratio, swept over support ----


def per_layer_option_c(
    dh: np.ndarray,
    h_t_l_normed: np.ndarray,
    projection: OutputProjection,
    alphas: Tuple[float, ...],
    rank: int,
    supports: Tuple[int, ...],
) -> Dict[str, float]:
    """Compute null_ratio_C_α at a single layer for every α × support.

    Per-layer logit-lens p^(ℓ) drives support selection (top-`support` of p^(ℓ))
    and p_s^(ℓ)^α weighting of the SVD. One SVD per (α, support). `h_t_l_normed`
    must already be post-norm (see apply_final_norm).
    """
    logits = projection.project(h_t_l_normed.astype(np.float32))
    p_l = safe_softmax(logits)

    H_l = float(-np.sum(p_l * np.log(p_l + 1e-12)))
    top1 = float(p_l.max())

    dh_f = dh.astype(np.float64)
    dh_norm = float(np.linalg.norm(dh_f))

    out: Dict[str, float] = {
        "p_ell_entropy": H_l,
        "p_ell_top1": top1,
    }

    for support in supports:
        support_k = int(min(support, p_l.shape[0]))
        idx = np.argpartition(-p_l, kth=support_k - 1)[:support_k]
        p_s = p_l[idx].astype(np.float64)
        W_s = projection.get_rows(idx).astype(np.float64)

        out[f"support_mass_supp{support}"] = float(p_s.sum())
        out[f"support_sig_supp{support}"] = support_signature(idx)

        if dh_norm < 1e-12:
            for a in alphas:
                key = f"alpha{alpha_key(a)}_supp{support}_rank{rank}"
                out[f"null_ratio_C_{key}"] = float("nan")
                out[f"fisher_energy_C_{key}"] = float("nan")
            continue

        for a in alphas:
            w = (p_s ** a) if a > 0 else np.ones_like(p_s)
            A = w[:, None] * W_s
            _, S, Vt = np.linalg.svd(A, full_matrices=False)
            V_r = Vt[:rank]
            proj = V_r.T @ (V_r @ dh_f)
            nr = float(np.linalg.norm(dh_f - proj) / dh_norm)
            lam = S ** 2
            fe = float(lam[:rank].sum() / (lam.sum() + 1e-30))
            key = f"alpha{alpha_key(a)}_supp{support}_rank{rank}"
            out[f"null_ratio_C_{key}"] = nr
            out[f"fisher_energy_C_{key}"] = fe

    return out


def option_a_null_ratio(
    dh: np.ndarray, Vt_A: np.ndarray, rank: int
) -> float:
    """Project Δh into the single fixed final-p eigenspace (Option A)."""
    dh_f = dh.astype(np.float64)
    dh_norm = float(np.linalg.norm(dh_f))
    if dh_norm < 1e-12:
        return float("nan")
    V_r = Vt_A[:rank]
    proj = V_r.T @ (V_r @ dh_f)
    return float(np.linalg.norm(dh_f - proj) / dh_norm)


# ---- Run ----


def run_model(model_name: str, samples: List[Dict[str, Any]]) -> pd.DataFrame:
    print(f"\n[{model_name}] loading...")
    t0 = time.time()
    model, tokenizer = mlx_load(model_name)
    projection = OutputProjection(model)
    n_layers = len(find_layers(model))
    print(
        f"  layers={n_layers} hidden={projection.hidden_size} "
        f"vocab={projection.vocab_size}  load={time.time()-t0:.1f}s"
    )

    rows: List[Dict[str, Any]] = []
    sample_bar = tqdm(samples, desc=f"  {model_name.split('/')[-1]}", unit="sample")
    for sample in sample_bar:
        prompt = sample["prompt"]
        sample_id = sample["sample_id"]
        cell = sample["cell"]
        has_contradiction = bool(sample["has_contradiction"])

        commit_id = greedy_commit_token(model, tokenizer, prompt, projection)
        prefix_ids = encode_text(tokenizer, prompt)
        full_ids = np.array(prefix_ids + [commit_id], dtype=np.int32)[None, :]

        per_layer_two = forward_all_layers_two_pos(model, full_ids)

        # Option A eigenspace — build from NORMED final hidden so it reflects
        # the real commit distribution (matches greedy_commit_token's logits).
        _, h_final_commit_raw = per_layer_two[-1]
        h_final_commit_normed = apply_final_norm(model, h_final_commit_raw)
        Vt_A, _lam_A, diag_A = final_p_t_eigenspace(
            h_final_commit_normed, projection
        )

        layer_bar = tqdm(
            enumerate(per_layer_two),
            total=n_layers,
            desc=f"    layers ({sample_id})",
            unit="layer",
            leave=False,
        )
        for li, (h_prev_l, h_t_l) in layer_bar:
            dh = h_t_l.astype(np.float64) - h_prev_l.astype(np.float64)
            dh_norm = float(np.linalg.norm(dh))

            h_t_l_normed = apply_final_norm(model, h_t_l)

            nr_A = option_a_null_ratio(dh, Vt_A, RANK)
            c_results = per_layer_option_c(
                dh, h_t_l_normed, projection, ALPHAS, RANK, SUPPORTS
            )

            row = {
                "model": model_name,
                "sample_id": sample_id,
                "cell": cell,
                "has_contradiction": has_contradiction,
                "commit_token_id": commit_id,
                "layer_index": li,
                "layer_normalized": li / max(n_layers - 1, 1),
                "n_layers": n_layers,
                "delta_h_norm": dh_norm,
                f"null_ratio_A_rank{RANK}": nr_A,
                "p_t_entropy_final": diag_A["p_t_entropy"],
                "p_t_top1_final": diag_A["p_t_top1"],
            }
            row.update(c_results)
            rows.append(row)
        layer_bar.close()
        sample_bar.set_postfix(rows=len(rows))

    return pd.DataFrame(rows)


def _safe_corr(df: pd.DataFrame, col_a: str, col_b: str) -> float:
    sub = df[[col_a, col_b]].dropna()
    if len(sub) < 3 or sub[col_a].nunique() < 2 or sub[col_b].nunique() < 2:
        return float("nan")
    return float(sub.corr().iloc[0, 1])


def _fmt_corr(c: float) -> str:
    return f"{c:>+9.4f}" if not np.isnan(c) else f"{'nan':>9}"


def print_split_diagnostics(df: pd.DataFrame) -> None:
    """Emit corr(null_ratio, p_ell_entropy) for all / layer=0 / layer>0."""
    partitions = [
        ("all",      df),
        ("layer=0",  df[df["layer_index"] == 0]),
        ("layer>0",  df[df["layer_index"] > 0]),
    ]

    def report(variant: str, col: str) -> None:
        cells = []
        for label, sub in partitions:
            cells.append(_fmt_corr(_safe_corr(sub, col, "p_ell_entropy")))
        mean_nr = df[col].mean()
        min_nr = df[col].min()
        print(
            f"  {variant:>34}  {cells[0]}  {cells[1]}  {cells[2]}  "
            f"{mean_nr:>9.4f}  {min_nr:>9.4f}"
        )

    print(
        f"\n  {'variant':>34}  {'all':>9}  {'layer=0':>9}  {'layer>0':>9}  "
        f"{'mean_nr':>9}  {'min_nr':>9}"
    )
    print("  " + "-" * 96)

    report("Option A (fixed, normed)", f"null_ratio_A_rank{RANK}")
    for support in SUPPORTS:
        for a in ALPHAS:
            tag = f"C α={a:g} supp={support}" + (" (=B)" if a == 1.0 else "")
            col = f"null_ratio_C_alpha{alpha_key(a)}_supp{support}_rank{RANK}"
            report(tag, col)


def main() -> int:
    out_dir = experiment_run_dir(EXPERIMENT_SLUG)
    print("=" * 72)
    print("E23 sharpness-aware Option C prototype (post-review fixed build)")
    print(
        f"  {MODEL.split('/')[-1]} × n={N_PER_CELL}/cell × every layer "
        f"× α∈{ALPHAS} × supp∈{SUPPORTS}"
    )
    print(f"  Output: {out_dir}")
    print("=" * 72)

    cfg = SyntheticLogicConfig(n_per_cell=N_PER_CELL, seed=42)
    samples = generate_synthetic_logic_dataset(cfg)
    print(f"\nGenerated {len(samples)} samples")

    df = run_model(MODEL, samples)
    out_path = out_dir / f"{model_slug(MODEL)}_e23_fixed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n  wrote {out_path} ({len(df)} rows)")

    print("\n  Primary diagnostic — corr(null_ratio, p_ell_entropy):")
    print_split_diagnostics(df)

    print(
        "\n  Success criterion (per Codex adversarial review):\n"
        "    |corr| < 0.3 in the layer>0 partition, with structure preserved.\n"
        "    If layer=0 dominates pooled result, the artifact is isolated — \n"
        "    the layer>0 column is then the honest success signal."
    )
    print("  See your research log for the E23 verdict writeup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

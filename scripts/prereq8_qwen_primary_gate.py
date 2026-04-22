#!/usr/bin/env python3
"""
Prereq 8 — primary gate: normed Option A Qwen rerun (post-E23-fix).

Replaces the stale `scripts/e22_direction_depth.py` reading for Qwen, which
built the Option A subspace from raw block output (no final norm) and would
clear or condemn Qwen on a subspace that doesn't match production.

For each Qwen puzzle × layer at step-1 commitment:
  1. Prefix forward → greedy commit token.
  2. [prefix + commit] forward → per-layer (h_prev, h_t) at positions T-1, T.
  3. Apply model final norm to h_final_commit_raw before building Option A.
  4. Vt_A = SVD eigenspace of sqrt(p_t)[:,None] · W_s with p_t from normed
     final hidden; support = top-256 by p_t.
  5. For each layer ℓ, null_ratio_A_ℓ = ||Δh_ℓ − V_r^T V_r Δh_ℓ|| / ||Δh_ℓ||
     at rank 32.

Gate condition (vs random baseline √((d−r)/d) ≈ 0.9955 for d=3584, r=32):
  max |median(null_ratio_A_ℓ) − baseline| over layers ≥ 0.020
    → Qwen is NOT anomalous (E22 flat reading was a norm artifact).
       Lift outlier flag, proceed to v3 main run.
  max |dev| < 0.020
    → Qwen flatness survives the norm fix; escalate to rank sweep
       (step 2) or deferred fp16 replication (step 3).

Artifacts (auto-incremented run-NN per date):
  experiments/prereq8-qwen-gate/<YYYY-MM-DD>/run-NN/
    qwen2_5-7b-instruct-4bit_prereq8.parquet
    manifest.json   (model, config, git SHA, baseline, verdict)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from mlx_lm import load as mlx_load

from pri_v2_mlx_pipeline import (
    OutputProjection,
    find_layers,
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
from scripts.e23_option_c import apply_final_norm, option_a_null_ratio
from scripts._paths import experiment_run_dir


# ---- Config ----

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
N_PER_CELL = 4
SUPPORT = 256
RANK = 32
SEED = 42
GATE_THRESHOLD = 0.020

EXPERIMENT_SLUG = "prereq8-qwen-gate"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "-dirty"
            return sha
    except Exception:
        pass
    return "unknown"


# ---- Run ----


def run_model(model_name: str, samples: List[Dict[str, Any]]) -> pd.DataFrame:
    print(f"\n[{model_name}] loading...")
    t0 = time.time()
    model, tokenizer = mlx_load(model_name)
    projection = OutputProjection(model)
    n_layers = len(find_layers(model))
    d = projection.hidden_size
    print(
        f"  layers={n_layers} hidden={d} "
        f"vocab={projection.vocab_size}  load={time.time()-t0:.1f}s"
    )

    rows: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc=f"  {model_name.split('/')[-1]}", unit="sample"):
        prompt = sample["prompt"]
        sample_id = sample["sample_id"]
        cell = sample["cell"]
        has_contradiction = bool(sample["has_contradiction"])

        commit_id = greedy_commit_token(model, tokenizer, prompt, projection)
        prefix_ids = encode_text(tokenizer, prompt)
        full_ids = np.array(prefix_ids + [commit_id], dtype=np.int32)[None, :]

        per_layer_two = forward_all_layers_two_pos(model, full_ids)

        # Option A eigenspace from NORMED final hidden (E23 fix).
        _, h_final_commit_raw = per_layer_two[-1]
        h_final_commit_normed = apply_final_norm(model, h_final_commit_raw)
        Vt_A, _lam_A, diag_A = final_p_t_eigenspace(
            h_final_commit_normed, projection, support=SUPPORT
        )

        for li, (h_prev_l, h_t_l) in enumerate(per_layer_two):
            dh = h_t_l.astype(np.float64) - h_prev_l.astype(np.float64)
            dh_norm = float(np.linalg.norm(dh))
            nr_A = option_a_null_ratio(dh, Vt_A, RANK)

            rows.append({
                "model": model_name,
                "sample_id": sample_id,
                "cell": cell,
                "has_contradiction": has_contradiction,
                "commit_token_id": commit_id,
                "layer_index": li,
                "layer_normalized": li / max(n_layers - 1, 1),
                "n_layers": n_layers,
                "hidden_dim": d,
                "delta_h_norm": dh_norm,
                f"null_ratio_A_rank{RANK}": nr_A,
                "p_t_entropy_final": diag_A["p_t_entropy"],
                "p_t_top1_final": diag_A["p_t_top1"],
                "support_rows": diag_A["support_rows"],
                "h_prev_source": "causal_T_minus_1",
                "final_norm_applied": True,
                "rank": RANK,
                "support": SUPPORT,
                "seed": SEED,
            })

    return pd.DataFrame(rows)


def main() -> int:
    # Auto-incremented run-NN per date. No --run-id / --force: each invocation
    # gets a fresh run directory; nothing can be overwritten.
    argparse.ArgumentParser(description="Prereq 8 primary gate").parse_args()

    out_dir = experiment_run_dir(EXPERIMENT_SLUG)
    run_id = out_dir.name

    print("=" * 72)
    print("Prereq 8 — primary gate: normed Option A Qwen rerun")
    print(f"  model={MODEL}")
    print(f"  n={N_PER_CELL}/cell × 4 cells = {N_PER_CELL * 4} samples, seed={SEED}")
    print(f"  rank={RANK}, support={SUPPORT}, every layer")
    print(f"  gate threshold: max |dev from baseline| ≥ {GATE_THRESHOLD:.3f}")
    print(f"  output: {out_dir}")
    print("=" * 72)

    cfg = SyntheticLogicConfig(n_per_cell=N_PER_CELL, seed=SEED)
    samples = generate_synthetic_logic_dataset(cfg)
    print(f"\nGenerated {len(samples)} samples")

    df = run_model(MODEL, samples)
    parquet_path = out_dir / f"{model_slug(MODEL)}_prereq8.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\n  wrote {parquet_path} ({len(df)} rows)")

    # Per-layer summary + gate evaluation.
    d = int(df["hidden_dim"].iloc[0])
    baseline = float(np.sqrt((d - RANK) / d))
    col = f"null_ratio_A_rank{RANK}"
    summary = (
        df.groupby("layer_index")[col]
          .agg(["median", "min", "max"])
          .reset_index()
    )
    summary["dev_from_baseline"] = summary["median"] - baseline
    summary["abs_dev"] = summary["dev_from_baseline"].abs()

    print(f"\n  Random baseline (d={d}, r={RANK}): √((d-r)/d) = {baseline:.4f}")
    print(f"\n  null_ratio_A_rank{RANK} by layer (median across samples):")
    print(f"  {'layer':>6}  {'median':>8}  {'dev':>8}  {'[min, max]'}")
    for _, r in summary.iterrows():
        print(
            f"  {int(r['layer_index']):>6}  "
            f"{r['median']:>8.4f}  "
            f"{r['dev_from_baseline']:>+8.4f}  "
            f"[{r['min']:.4f}, {r['max']:.4f}]"
        )

    max_abs_dev = float(summary["abs_dev"].max())
    argmax_layer = int(summary.loc[summary["abs_dev"].idxmax(), "layer_index"])
    passed = max_abs_dev >= GATE_THRESHOLD

    print(f"\n  max |dev from baseline| = {max_abs_dev:.4f} at layer {argmax_layer}")
    print(f"  gate threshold          = {GATE_THRESHOLD:.4f}")
    if passed:
        print(f"  VERDICT: GATE PASSED — Qwen shows structure (not flat).")
        print(f"           E22 flat reading was a norm artifact.")
        print(f"           Lift Qwen outlier flag; proceed to v3 main run.")
    else:
        print(f"  VERDICT: GATE FAILED — Qwen flatness survives norm fix.")
        print(f"           Escalate to step 2 (rank sweep 32/64/128/256).")

    # Sidecar manifest.
    manifest = {
        "run_id": run_id,
        "utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(_REPO_ROOT)),
        "git_sha": _git_sha(),
        "model": MODEL,
        "n_per_cell": N_PER_CELL,
        "n_samples": int(N_PER_CELL * 4),
        "seed": SEED,
        "support": SUPPORT,
        "rank": RANK,
        "gate_threshold": GATE_THRESHOLD,
        "baseline_random_null_ratio": baseline,
        "n_layers": int(df["n_layers"].iloc[0]),
        "hidden_dim": d,
        "final_norm_applied": True,
        "max_abs_dev": max_abs_dev,
        "argmax_layer": argmax_layer,
        "gate_passed": passed,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  manifest: {manifest_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

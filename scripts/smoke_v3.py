#!/usr/bin/env python3
"""
Smoke test for v3 null_ratio / fisher_energy emission.

Runs run_experiment with n_samples_per_cell=1, chain_lengths=[2], one model.
Confirms:
  - pipeline executes end-to-end without exception
  - parquet rows include null_ratio_rank{R} + fisher_energy_rank{R} columns
  - null_ratio values are finite and in [0, 1]
  - fisher_energy values are finite and in [0, 1]

Usage:
    .venv/bin/python scripts/smoke_v3.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

import pri_v2_mlx_pipeline as pipeline  # noqa: E402


def main() -> int:
    cfg = pipeline.Config()
    cfg.n_samples_per_cell = 1
    cfg.chain_lengths = [2]
    cfg.pilot_n = 1
    cfg.pilot_threshold = 0.0
    cfg.models = ["mlx-community/Llama-3.2-3B-Instruct-4bit"]
    cfg.alpha_values = [1.0]
    cfg.topk_values = [32]
    cfg.lowrank_values = [16]
    cfg.v3_rank_values = [1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64]
    cfg.max_new_tokens = 4
    cfg.n_trace_dumps = 0
    cfg.save_dir = os.path.join(ROOT, "pri_smoke_v3_results")
    cfg.seed = 0

    results_df, _ = pipeline.run_experiment(cfg)

    print("\n=== smoke-v3 assertions ===")
    expected = [f"null_ratio_rank{r}" for r in cfg.v3_rank_values] + [
        f"fisher_energy_rank{r}" for r in cfg.v3_rank_values
    ]
    missing = [c for c in expected if c not in results_df.columns]
    assert not missing, f"missing columns: {missing}"
    print(f"all {len(expected)} v3 columns present")

    for c in expected:
        vals = results_df[c].to_numpy()
        assert np.all(np.isfinite(vals)), f"{c} has non-finite values"
        assert np.all(vals >= -1e-6), f"{c} has negative values: min={vals.min()}"
        assert np.all(vals <= 1.0 + 1e-6), f"{c} has values > 1: max={vals.max()}"
    print("all v3 values finite and in [0, 1]")

    s1 = results_df[(results_df["gen_step"] == 1) & (results_df["layer"] == "final")]
    for r in cfg.v3_rank_values:
        print(
            f"  rank {r}: null_ratio mean={s1[f'null_ratio_rank{r}'].mean():.3f} "
            f"fisher_energy mean={s1[f'fisher_energy_rank{r}'].mean():.3f}"
        )

    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

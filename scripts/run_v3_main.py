#!/usr/bin/env python3
"""v3 confirmatory main-run launcher.

Thin wrapper around pri_v2_mlx_pipeline.run_experiment with Config overrides
tailored to the pre-registered confirmatory experiments:

  E17   pri_v3_null_bare      — null_ratio alone
  E17b  pri_v3_null_raw       — null_ratio from raw W_u (HARP baseline)
  E18   pri_v3_null_ratio     — additive S_t + alpha * null_ratio
  E19   pri_v3_null_gated     — multiplicative d_F * null_ratio

Sample size (50/cell × 4 cells = 200/model) comes from the 2026-04-22
power-fix amendment to pri-v3-plan.md: at n=20/cell the E18 AUROC ≥ 0.60
gate is undecidable at its own threshold; at n=50/cell the 95% CI lower
bound clears 0.5.

Usage:
    .venv/bin/python scripts/run_v3_main.py --scope all
    .venv/bin/python scripts/run_v3_main.py --scope primaries --n-per-cell 10   # pilot
    .venv/bin/python scripts/run_v3_main.py --scope all --v3-capture            # + E21 depth data

Exit 0 on completion; the summary parquet and per-model parquets land under
  experiments/v3-main-run/<YYYY-MM-DD>/run-NN/
"""

from __future__ import annotations

import argparse
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

import pri_v2_mlx_pipeline as pipeline  # noqa: E402
from scripts._paths import experiment_run_dir  # noqa: E402


PRIMARIES = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
]
GEMMAS = [
    "mlx-community/gemma-3-1b-it-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
]
NON_GEMMA_EXTENDED = [
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
]
EXTENDED = GEMMAS + NON_GEMMA_EXTENDED
# Gemma-isolated scopes let a Gemma-specific failure (bf16, interleaved SWA,
# embed scale, multimodal wrapper) quarantine without killing the rest of the
# run. "non_gemmas" = primaries + Qwen3 + Phi-3.5; "gemmas" = Gemma 1B + 4B.
SCOPES = {
    "primaries": PRIMARIES,
    "extended": EXTENDED,
    "gemmas": GEMMAS,
    "non_gemmas": PRIMARIES + NON_GEMMA_EXTENDED,
    "all": PRIMARIES + EXTENDED,
}

EXPERIMENT_SLUG = "v3-main-run"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="v3 confirmatory main run (E17/E17b/E18/E19)"
    )
    parser.add_argument(
        "--scope",
        choices=sorted(SCOPES.keys()),
        default="all",
        help="which model set to run (default: all 7)",
    )
    parser.add_argument(
        "--n-per-cell",
        type=int,
        default=50,
        help="samples per 2x2 cell (default 50 per 2026-04-22 power fix)",
    )
    parser.add_argument(
        "--v3-capture",
        action="store_true",
        help="every-layer capture for E21 depth data (confirmatory doesn't need it)",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=14,
        help="generation budget (default 14 — enough to cross into probe_4 regime)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default 42)"
    )
    args = parser.parse_args()

    out_dir = experiment_run_dir(EXPERIMENT_SLUG)

    cfg = pipeline.Config()
    cfg.n_samples_per_cell = args.n_per_cell
    cfg.models = SCOPES[args.scope]

    # Confirmatory-only variant set: α = 1.0, topk = 32, lowrank = 32 are the
    # per-model best variants from the 2026-04-13 v2 baseline (validated in
    # wiki/results/summary.md). A wider sweep is out of scope for the sealed
    # confirmatory run — exploratory ranks can be rerun separately.
    cfg.alpha_values = [1.0]
    cfg.topk_values = [32]
    cfg.lowrank_values = [32]
    # v3 rank sweep retained for E17/E17b/E18/E19 rank-sensitivity analysis.
    cfg.v3_rank_values = [1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64]

    cfg.max_new_tokens = args.max_gen_tokens
    cfg.seed = args.seed
    cfg.save_dir = str(out_dir)

    # v3 capture schedule — default off. On only for E21 depth-profile work;
    # E17/E17b/E18/E19 operate on final-layer null_ratio which the v2 path
    # already emits via v3_rank_values + PRIComputer.
    cfg.v3_capture = bool(args.v3_capture)

    print("=" * 72)
    print(f"v3 main run — scope={args.scope} ({len(cfg.models)} models)")
    print(f"  n_per_cell={cfg.n_samples_per_cell}  "
          f"cells=2x2  total={cfg.n_samples_per_cell * 4}/model")
    print(f"  alpha={cfg.alpha_values}  topk={cfg.topk_values}  "
          f"lowrank={cfg.lowrank_values}")
    print(f"  v3_rank_values={cfg.v3_rank_values}")
    print(f"  v3_capture={cfg.v3_capture}  max_new_tokens={cfg.max_new_tokens}")
    print(f"  save_dir={cfg.save_dir}")
    print("=" * 72)
    for m in cfg.models:
        print(f"  - {m}")
    print()

    results_df, _ = pipeline.run_experiment(cfg)

    print()
    print("=" * 72)
    print(f"v3 main run COMPLETE — {len(results_df)} rows")
    print(f"  artifacts: {cfg.save_dir}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())

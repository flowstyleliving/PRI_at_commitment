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
    "non_gemma_extended": NON_GEMMA_EXTENDED,
    "non_gemmas": PRIMARIES + NON_GEMMA_EXTENDED,
    "all": PRIMARIES + EXTENDED,
    # v3.1-main-run scope (2026-04-24 pre-reg amendment): primaries + Qwen3 only.
    # Excludes Phi-3.5-mini (gate-fails at n=20 per 2026-04-23 evidence; not
    # worth the 2-3 min gate-skip cost) and Gemma 3-1B / 3-4B (full run_experiment
    # path not yet proven end-to-end on a Gemma checkpoint with v3_capture_raw=True;
    # isolating to a separate v3_1_gemmas launch keeps the main gate run clean if
    # a Gemma-specific adapter bug trips the raw-W_u SVD path).
    "v3_1_main": PRIMARIES + [
        "mlx-community/Qwen3-8B-4bit",
    ],
    # v3.1-gemmas companion scope: Gemma 1B + 4B only. Run AFTER v3_1_main
    # completes so the sealed-gate data is already checkpointed and cannot be
    # contaminated by a Gemma-side adapter regression.
    "v3_1_gemmas": GEMMAS,
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
        "--no-e17b",
        action="store_true",
        help="disable E17b HARP-style raw-W_u null_ratio capture. Default is ON — "
             "emits null_ratio_raw_rank{r} and raw_energy_rank{r} columns alongside "
             "the Fisher-weighted v3 columns for the same rank sweep. One-time "
             "model-load cost (~5–30s per model); per-sample cost is a matvec.",
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
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="bypass the behavioral gate (sets pilot_threshold=0.0). Use when"
             " models have already been verified via smoke_test_model.py --gate"
             " — avoids re-verifying reasoning-tuned models whose phrasing can"
             " push check_answer below the 0.80 threshold in rare cases.",
    )
    parser.add_argument(
        "--gate-verbose",
        action="store_true",
        help="print per-sample gate diagnostic (expected vs parsed + output "
             "preview). Useful to diagnose gate failures on reasoning models.",
    )
    parser.add_argument(
        "--pilot-threshold",
        type=float,
        default=None,
        help="override control-accuracy gate threshold (default from Config: "
             "0.80). Lower for models with legitimately mixed reasoning output.",
    )
    parser.add_argument(
        "--gate-max-tokens",
        type=int,
        default=None,
        help="override gate generation budget (default from Config: 256).",
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
    # E17b capture — default ON so v3.1 runs produce the head-to-head
    # against HARP's static raw-W_u subspace. See pri-v3-plan.md §E17b for
    # the falsification criterion (AUROC(null_bare) − AUROC(null_raw) ≥ 0.02
    # with non-overlap CI on Qwen).
    cfg.v3_capture_raw = not bool(args.no_e17b)

    # Gate controls. --skip-gate short-circuits by forcing pilot_threshold=0.0
    # (every run passes); --pilot-threshold lets you dial a lower bar (e.g. 0.6
    # for reasoning models with legitimately mixed output). --gate-max-tokens
    # overrides the Config default (256 matches smoke's working budget).
    cfg.gate_verbose = bool(args.gate_verbose)
    if args.skip_gate:
        cfg.pilot_threshold = 0.0
    elif args.pilot_threshold is not None:
        cfg.pilot_threshold = float(args.pilot_threshold)
    if args.gate_max_tokens is not None:
        cfg.gate_max_new_tokens = int(args.gate_max_tokens)

    print("=" * 72)
    print(f"v3 main run — scope={args.scope} ({len(cfg.models)} models)")
    print(f"  n_per_cell={cfg.n_samples_per_cell}  "
          f"cells=2x2  total={cfg.n_samples_per_cell * 4}/model")
    print(f"  alpha={cfg.alpha_values}  topk={cfg.topk_values}  "
          f"lowrank={cfg.lowrank_values}")
    print(f"  v3_rank_values={cfg.v3_rank_values}")
    print(f"  v3_capture={cfg.v3_capture}  v3_capture_raw (E17b)={cfg.v3_capture_raw}  "
          f"max_new_tokens={cfg.max_new_tokens}")
    print(
        f"  gate: threshold={cfg.pilot_threshold:.0%}  "
        f"max_new_tokens={cfg.gate_max_new_tokens}  "
        f"verbose={cfg.gate_verbose}"
        + ("  (bypassed via --skip-gate)" if args.skip_gate else "")
    )
    print(f"  save_dir={cfg.save_dir}")
    print("=" * 72)

    # Prominent, numbered queue so you know exactly what's on deck before
    # run_experiment's own banner drowns it out mid-run.
    print()
    print("=" * 72)
    print(f"  MODEL QUEUE ({len(cfg.models)} total, processed in this order)")
    print("=" * 72)
    width = len(str(len(cfg.models)))
    for i, m in enumerate(cfg.models, 1):
        print(f"  [{i:>{width}}/{len(cfg.models)}] {m}")
    print("=" * 72)
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

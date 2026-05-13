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
    # v3.1 three-phase launch (amended 2026-04-24): each axis as its own scope
    # so any phase can be run / skipped / re-run independently without touching
    # the others' checkpoints. Sealed gate authority stays with v3_1_primaries.
    #
    # v3_1_primaries  — Phase 1 — 3 primaries alone. Sealed E18 + E17b authority;
    #                   pass/fail verdict lives here. ~60-80 min on Mac mini M4.
    "v3_1_primaries": PRIMARIES,
    # v3_1_qwen3      — Phase 2 (optional) — Qwen3-8B alone. Cross-generation
    #                   within the Qwen family (Qwen 2.5 → Qwen 3); same seed so
    #                   puzzle draws match Phase 1. ~15-20 min. Fresh Qwen3 data
    #                   with E17b columns at pinned rank 1 — complements the
    #                   2026-04-23 main-run capture (which has no raw-W_u SVD).
    "v3_1_qwen3": [
        "mlx-community/Qwen3-8B-4bit",
    ],
    # v3_1_mistral_only / v3_1_qwen25_only — single-primary scopes for
    # fresh-process per-model runs. Added 2026-04-24 after codex 2nd-rescue
    # verdict: compressor state carried across models in the same process
    # slows the later ones (observed 6× throughput drop on Mistral after
    # Llama 3B gate-failed + cleanup in same process). Use these when you
    # need to reset the MLX allocator between primaries. Not a sealed-gate
    # re-specification — just a process-boundary tool.
    "v3_1_mistral_only": [
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    ],
    "v3_1_qwen25_only": [
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    ],
    # v3_1_phi_only   — single-primary scope for re-validating Phi-3.5-mini
    # under the v3.1 gate fixes (PR #7 stratified preflight + 3-tier parser,
    # plus operational --gate-max-tokens 12 from 2026-04-24). Phi was
    # excluded from the 2026-04-23 sealed run after gate-failing at
    # 12/20=60% under the original parser + 256-token gate budget.
    # Recovery here lifts Phi from "excluded" to descriptive companion;
    # a clean fail confirms the model-specific finding. Not a sealed primary.
    "v3_1_phi_only": [
        "mlx-community/Phi-3.5-mini-instruct-4bit",
    ],
    # v3_1_gemmas     — Phase 3 (optional) — Gemma 3-1B + Gemma 3-4B. Within-
    #                   family scale axis held fixed at architecture. Isolated
    #                   from Phase 1 because the full run_experiment loop with
    #                   v3_capture_raw=True has never executed end-to-end on a
    #                   Gemma checkpoint (Prereq 4 dryrun validated only
    #                   trace_sample + SVD at n=4/cell). ~40-60 min.
    "v3_1_gemmas": GEMMAS,
    # v3_1_gemma4b_only — Phase 3 narrowed to Gemma 3-4B alone (2026-04-26).
    #                   Gemma 1B was excluded after gate-fail at 11/20 = 55%
    #                   on n=20 stratified controls (model-capability, not
    #                   parser — defaults to "Answer: NO" on YES controls).
    #                   Pipeline pre-validated end-to-end on both Gemmas at
    #                   n=2/cell smoke (run-04) and n=10/cell pilot (run-05).
    #                   Gemma 4B descriptive-only — cross-architecture point,
    #                   not within-family scale (axis collapsed). ~25-30 min.
    "v3_1_gemma4b_only": [
        "mlx-community/gemma-3-4b-it-4bit",
    ],
    # v3_1_main       — convenience alias: primaries + Qwen3 as a single run.
    #                   Preserved for backward compat with the 2026-04-24 plan
    #                   amendment; equivalent to running v3_1_primaries followed
    #                   by v3_1_qwen3 but in one launch. Gemma still isolated.
    "v3_1_main": PRIMARIES + [
        "mlx-community/Qwen3-8B-4bit",
    ],
    # v3_2_centered_bakeoff — pre-reg amendment 2026-05-07. Three-way
    #                   Fisher / Raw / Centered-Fisher comparison across the
    #                   six v3.1 cross-architecture models. Adds two
    #                   descriptive column families (kl_discharged,
    #                   null_ratio_centered_post_rank{r}) and one capture
    #                   family (gen_p_t_topk_*) — sealed E18/E17b primaries
    #                   unchanged. See wiki/results/v3.2-amendment.md for
    #                   the pre-reg, decision criteria, and seed (20260507).
    "v3_2_centered_bakeoff": PRIMARIES + [
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        "mlx-community/gemma-3-4b-it-4bit",
    ],
    # v3_2_expansion_llama_8b — 2026-05-11 expansion run. Only Llama-3.1-8B
    # passed smoke (Mistral-Nemo, Phi-4-mini, Gemma-3-1B, Dolphin-Nemo all
    # failed the behavioral gate or adapter pipeline; see
    # wiki/v4-candidates.md §4 "Planned model-set expansion" for the
    # diagnoses and follow-up tasks). This single-model scope builds the
    # within-family scale axis for Llama (3B → 8B).
    "v3_2_expansion_llama_8b": [
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
    ],
    # v3_2_expansion_phi_4 — 2026-05-11. Phi-4-mini-instruct added after the
    # Phi3Adapter received a tied-embedding fix (model_adapters.py:622:
    # falls back to embed_tokens.as_linear() when lm_head is None). Builds
    # the within-family generation axis for Phi (3.5 → 4).
    "v3_2_expansion_phi_4": [
        "mlx-community/Phi-4-mini-instruct-4bit",
    ],
    # v3_2_expansion_phase_b — 2026-05-11. Mistral-Nemo (12B) + Gemma-3-1B
    # added after the pri_v2_io_plugins module landed: chat-template is
    # now applied per-model via PROMPT_STRATEGY_BY_MODEL, parser handles
    # bare YES/NO via Tier 0. Both models re-smoke 4/4 = 100% with the
    # new pipeline. Adds the within-family pair for Mistral (7B → Nemo 12B)
    # and Gemma (1B + 4B). Mistral-Nemo runs first because it's larger
    # (~90 min) and dominates total wall time; Gemma-3-1B is the smaller
    # quick add (~30 min).
    "v3_2_expansion_phase_b": [
        "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "mlx-community/gemma-3-1b-it-4bit",
    ],
    # v3_2_expansion_reasoning — 2026-05-12. Second reasoning-tuned model
    # added to unblock v4-candidate #4's strict-bar failure on Qwen 3 8B
    # (oracle Fisher r=64 step 2 vs predicted Fisher r=2 step 3). Qwen3-1.7B
    # is a same-family scale variant — if its oracle cell lands at the same
    # (family, rank) as Qwen 3 8B, the reasoning-branch's rank is pinned;
    # if not, the within-reasoning rank choice is model-specific and the
    # branch needs further work. DeepSeek-R1-Distill-Qwen-7B was attempted
    # first but smoke-failed (R1 thinks-before-answer; 0/4 at 24 tokens).
    # Smoke pass 4/4 = 100% after the 2026-05-12 tied-embedding patch in
    # QwenAdapter (lm_head=None → embed_tokens.as_linear()).
    "v3_2_expansion_reasoning": [
        "mlx-community/Qwen3-1.7B-4bit",
    ],
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
        "--no-centered",
        action="store_true",
        help="disable v3.2 centered-Fisher null_ratio capture. Default is ON — "
             "emits kl_discharged + null_ratio_centered_post_rank{r} + "
             "fisher_energy_centered_rank{r} columns alongside the sealed "
             "Fisher / Raw pair (descriptive-only, no sealed-gate authority). "
             "Per-sample cost is one extra eigh on a (≤support × ≤support) "
             "matrix, ~50ms at support=512.",
    )
    parser.add_argument(
        "--p-t-topk",
        type=int,
        default=None,
        help="override Config.v3_capture_p_t_topk (default 512). Persists "
             "per-step top-K probability indices+values in trace_dumps for "
             "post-hoc KL-grounded analysis without replay. Set 0 to disable.",
    )
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=14,
        help="generation budget (default 14 — enough to cross into probe_4 regime)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="override Config.layers_to_probe (default: final mid quarter). "
             "Use `--layers final` for sealed-gate runs where only the final "
             "layer is analyzed; drops main-run metric compute by ~3× "
             "(skips mid+quarter layer iterations in the inner loop).",
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

    # --layers override (2026-04-24): cuts main-run inner loop from 3 layers
    # to 1 when the sealed analysis plane is final-layer only. Not a sealed
    # re-specification — the sealed E18/E17b spec pins layer=final already;
    # this just skips capturing mid/quarter data the sealed analyzer ignores.
    # Keep None (default) to preserve full landscape data for diagnostics.
    if args.layers is not None:
        valid = {"final", "mid", "quarter"}
        bad = [l for l in args.layers if l not in valid]
        if bad:
            raise SystemExit(
                f"--layers got unknown value(s) {bad}; "
                f"choose from {sorted(valid)}"
            )
        cfg.layers_to_probe = list(args.layers)

    # v3 capture schedule — default off. On only for E21 depth-profile work;
    # E17/E17b/E18/E19 operate on final-layer null_ratio which the v2 path
    # already emits via v3_rank_values + PRIComputer.
    cfg.v3_capture = bool(args.v3_capture)
    # E17b capture — default ON so v3.1 runs produce the head-to-head
    # against HARP's static raw-W_u subspace. See pri-v3-plan.md §E17b for
    # the falsification criterion (AUROC(null_bare) − AUROC(null_raw) ≥ 0.02
    # with non-overlap CI on Qwen).
    cfg.v3_capture_raw = not bool(args.no_e17b)
    # v3.2 centered-Fisher capture — default ON. Adds kl_discharged +
    # null_ratio_centered_post_rank{r} + fisher_energy_centered_rank{r}
    # columns alongside the sealed Fisher/Raw pair. DESCRIPTIVE-ONLY: no
    # sealed-gate authority. See wiki/results/v3.2-amendment.md.
    cfg.v3_capture_centered = not bool(args.no_centered)
    if args.p_t_topk is not None:
        if args.p_t_topk < 0:
            raise SystemExit(f"--p-t-topk must be ≥ 0, got {args.p_t_topk}")
        cfg.v3_capture_p_t_topk = int(args.p_t_topk)

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
    print(f"  layers_to_probe={cfg.layers_to_probe}")
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

#!/usr/bin/env python3
"""v3.2 PRI pipeline on ANLI R2 — natural-language NLI replacement for the
synthetic logic puzzles.

Mirrors `run_v3_main.py` but injects an ANLI dataset via Config.task_dataset
instead of using PuzzleGenerator. Uses the same v3.2 capture protocol
(layer=final, alpha=1.0, max_gen_tokens=24, gate_max_tokens=12, all v3
capture flags on).

Mapping:
  ANLI label 0 (entailment)    → contradiction=False, correct_value="YES"
  ANLI label 2 (contradiction) → contradiction=True,  correct_value="NO"
  ANLI label 1 (neutral)       → DROPPED for binary PRI signal

  chain_length: bucketed by premise length:
    < 300 chars → 2 (short)
    ≥ 300 chars → 5 (long)
  Matches the synthetic puzzle 2x2 factorial structure so stratified
  preflight gate behavior is preserved.

Usage:
    .venv/bin/python scripts/run_v3_anli.py \\
        --scope mistral_nemo_llama_3b \\
        --n-per-cell 50 \\
        --round dev_r2 \\
        --seed 20260512 \\
        --max-gen-tokens 24 --gate-max-tokens 12 --layers final
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from datasets import load_dataset

import pri_v2_mlx_pipeline as pipeline


# v3.2 + ANLI prompt template — matches the smoke that worked 8/10 on
# Mistral-Nemo. Same instruction shape as logic puzzles + "Answer:" suffix
# so commit-step lands on the YES/NO token.
ANLI_PROMPT_TEMPLATE = (
    "Instruction: Read the premise and decide whether the hypothesis is "
    "entailed by the premise. Answer YES if the premise entails the "
    "hypothesis, NO if the premise contradicts the hypothesis.\n"
    "\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Answer:"
)


SCOPES = {
    # Phase-1 smoke: 2 models that span behavioral regimes
    #   * Mistral-Nemo: terminal-commit (single-token YES/NO on synthetic);
    #     does it preserve that on natural language, or does the longer
    #     premise break the pattern? ANLI smoke showed it now emits
    #     'YES. The premise states...' — answer-then-justify, like Phi-4.
    #   * Llama 3.2-3B: format-completion ("Answer: YES\nNow solve...");
    #     same on ANLI?
    "mistral_nemo_llama_3b": [
        "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
    ],
    "mistral_nemo_only": [
        "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    ],
}

EXPERIMENT_SLUG = "v3-anli-run"


def build_anli_dataset(
    split: str,
    n_per_cell: int,
    seed: int,
) -> pd.DataFrame:
    """Load ANLI samples, drop neutral, balance entailment vs contradiction,
    stratify by premise length, return a synthetic-puzzle-schema DataFrame.

    Returns 4 cells: (chain_length=2, contradiction=False/True) and
    (chain_length=5, contradiction=False/True), n_per_cell each.
    """
    print(f"  Loading ANLI split: {split}")
    ds = load_dataset("facebook/anli", split=split)
    print(f"  ANLI {split}: {len(ds)} raw samples")

    # Filter neutrals; balance binary; stratify by premise length
    rng = np.random.RandomState(seed)
    buckets = {(2, False): [], (2, True): [], (5, False): [], (5, True): []}
    target = n_per_cell

    # Shuffle deterministically before allocating to cells
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    for idx in indices:
        ex = ds[idx]
        if ex["label"] == 1:
            continue  # drop neutral
        contradiction = (ex["label"] == 2)
        cl = 2 if len(ex["premise"]) < 300 else 5
        key = (cl, contradiction)
        if len(buckets[key]) >= target:
            continue
        buckets[key].append({
            "anli_uid": ex["uid"],
            "premise": ex["premise"],
            "hypothesis": ex["hypothesis"],
            "label": ex["label"],
            "chain_length": cl,
            "contradiction": contradiction,
            "correct_value": "NO" if contradiction else "YES",
            "prompt": ANLI_PROMPT_TEMPLATE.format(
                premise=ex["premise"], hypothesis=ex["hypothesis"]
            ),
        })

    # Check fills
    fills = {k: len(v) for k, v in buckets.items()}
    print(f"  Cell fills: {fills}")
    short_e = fills[(2, False)]
    short_c = fills[(2, True)]
    long_e = fills[(5, False)]
    long_c = fills[(5, True)]
    if any(c < target for c in (short_e, short_c, long_e, long_c)):
        print(f"  WARN: not all cells reached {target} — {fills}")

    # Combine
    rows = []
    sid = 0
    for key, items in buckets.items():
        for r in items:
            r["sample_id"] = sid
            # Pad missing PuzzleGenerator fields with None for schema compat
            r.setdefault("subject", None)
            r.setdefault("target", None)
            r.setdefault("terms", None)
            r.setdefault("injected_statement", None)
            rows.append(r)
            sid += 1
    df = pd.DataFrame(rows)
    # Shuffle so models don't process all-entail-then-all-contradict
    df = df.sample(frac=1, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)
    print(f"  Final ANLI dataset: {len(df)} samples")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="v3.2 PRI pipeline on ANLI (R1/R2/R3)"
    )
    parser.add_argument(
        "--scope",
        choices=sorted(SCOPES.keys()),
        default="mistral_nemo_llama_3b",
        help="which model set to run",
    )
    parser.add_argument(
        "--n-per-cell", type=int, default=50,
        help="samples per (chain_length, contradiction) cell. 4 cells total.",
    )
    parser.add_argument(
        "--round",
        choices=["dev_r1", "dev_r2", "dev_r3", "test_r2"],
        default="dev_r2",
        help="ANLI split (default dev_r2)",
    )
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--max-gen-tokens", type=int, default=24)
    parser.add_argument("--gate-max-tokens", type=int, default=12)
    parser.add_argument("--layers", nargs="+", default=["final"])
    parser.add_argument("--gate-verbose", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    # Set up output dir
    date_str = datetime.now().strftime("%Y-%m-%d")
    base = REPO_ROOT / "experiments" / EXPERIMENT_SLUG / date_str
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("run-*"))
    run_num = (int(existing[-1].name.split("-")[1]) + 1) if existing else 1
    out_dir = base / f"run-{run_num:02d}"
    out_dir.mkdir()

    print("=" * 72)
    print(f"v3 ANLI run — scope={args.scope}")
    print(f"  round={args.round}  n_per_cell={args.n_per_cell}  "
          f"total={args.n_per_cell * 4}/model")
    print(f"  out_dir={out_dir}")
    print("=" * 72)

    # Build ANLI dataset
    print("\nSECTION 0: ANLI DATASET CONSTRUCTION")
    print("-" * 72)
    anli_df = build_anli_dataset(args.round, args.n_per_cell, args.seed)

    # Build Config
    cfg = pipeline.Config()
    cfg.task_dataset = anli_df
    cfg.task_label = f"anli_{args.round}"
    cfg.n_samples_per_cell = args.n_per_cell
    cfg.models = SCOPES[args.scope]
    cfg.alpha_values = [1.0]
    cfg.topk_values = [32]
    cfg.lowrank_values = [32]
    cfg.v3_rank_values = [1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64]
    cfg.max_new_tokens = args.max_gen_tokens
    cfg.seed = args.seed
    cfg.save_dir = str(out_dir)
    cfg.layers_to_probe = list(args.layers)
    cfg.v3_capture = False
    cfg.v3_capture_raw = True
    cfg.v3_capture_centered = True
    cfg.v3_capture_p_t_topk = 512
    cfg.gate_verbose = args.gate_verbose
    if args.skip_gate:
        cfg.pilot_threshold = 0.0
    cfg.gate_max_new_tokens = args.gate_max_tokens

    # MODEL QUEUE banner
    print()
    print("=" * 72)
    print(f"  MODEL QUEUE ({len(cfg.models)} total)")
    print("=" * 72)
    for i, m in enumerate(cfg.models, 1):
        print(f"  [{i}/{len(cfg.models)}] {m}")
    print()

    # Run!
    all_results, all_traces = pipeline.run_experiment(cfg)

    # Save consolidated outputs
    all_results.to_parquet(out_dir / "all_results.parquet")
    all_traces.to_parquet(out_dir / "all_trace_dumps.parquet")
    anli_df.to_parquet(out_dir / "anli_dataset.parquet")
    print(f"\nv3 ANLI run COMPLETE — {len(all_results)} rows")
    print(f"  artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Per-sample diagnostic for the 2026-05-10 step-sweep anomalies.

Four model-specific oddities surfaced in the step sweep:

  * Llama 3.2 3B — best fixed step = 2 (model emits "Answer:" at steps 1-4
    and YES/NO at step ~5; rupture peaks BEFORE the answer is emitted)
  * Phi-3.5-mini — best fixed step = 2 (same pattern, pre-Answer)
  * Qwen 3 8B — best fixed step = 2 (mid-COT, before answer reasoning starts)
  * Gemma 3 4B — best fixed step = 12 (LONG after commit_step=4; model is
    deep into format-completion / fabricated next-puzzle by step 12)

For each, this script dumps the trace_dump samples with a window of decoded
tokens + top-K probabilities + Fisher r=1 metric value, control vs
contradiction side-by-side. Goal: humanely-readable evidence that explains
*what the model is doing* at the anomalous best step, so we can decide
whether the rupture signal is catching:
  (a) early commitment to YES/NO (interpretability-positive)
  (b) format-completion artifact (interpretability-negative)
  (c) something else entirely

Usage:
    .venv/bin/python scripts/diagnostics/diagnose_v3_2_anomalies.py \\
        --run-dir experiments/v3-main-run/2026-05-08/run-01

Cross-references:
  - wiki/v4-candidates.md §3 (open question on pre-commit) and §4 (Gemma anomaly risk)
  - wiki/results/v3.2-results.md §4 — the step-sweep that surfaced these
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Model → anomalous best step (from diagnose_v3_2_step_sweep.py output on
# 2026-05-08/run-01). Step is inclusive; we display a window around it.
ANOMALIES = [
    ("mlx-community/Llama-3.2-3B-Instruct-4bit", 2, "pre-Answer (commit ~step 5)"),
    ("mlx-community/Phi-3.5-mini-instruct-4bit", 2, "pre-Answer (commit ~step 3)"),
    ("mlx-community/Qwen3-8B-4bit", 2, "pre-COT-answer (commit ~step 3)"),
    # NOTE: Gemma 4B's prior step=12 anomaly was a CLASS-IMBALANCE ARTIFACT
    # (n=85: 80 controls vs 5 contradictions — only the format-completing
    # samples reach step 12). With class-balance filtering, Gemma's true
    # balanced best is Raw r=1 @ step 3 (AUROC 0.9998, n=200, 100/100).
    ("mlx-community/gemma-3-4b-it-4bit", 3, "balanced best (Raw r=1; commit ~step 4)"),
]
WINDOW = 2  # show step ± WINDOW
TOPK_DISPLAY = 5
N_PER_CONDITION = 3  # how many trace dumps to display per condition


def decode_step_window(tokenizer, gen_token_ids: List[int], best: int, window: int):
    """Return list of (step, token_str, cumulative_str) for steps in window."""
    out = []
    cumulative = ""
    for i, tid in enumerate(gen_token_ids):
        decoded = tokenizer.decode([int(tid)])
        cumulative += decoded
        step = i + 1  # 1-indexed
        if best - window <= step <= best + window:
            out.append((step, decoded, cumulative))
    return out


def fmt_topk(indices: List[int], values: List[float], tokenizer, k: int) -> str:
    """One-line top-K probability summary."""
    pairs = []
    for tid, p in list(zip(indices, values))[:k]:
        tok = tokenizer.decode([int(tid)])
        # Escape whitespace and format compactly
        tok_repr = tok.replace("\n", "\\n").replace("\t", "\\t")
        if len(tok_repr) > 16:
            tok_repr = tok_repr[:14] + "…"
        pairs.append(f"{tok_repr!r}={float(p):.3f}")
    return "  ".join(pairs)


def diagnose_model(
    df_dumps: pd.DataFrame,
    df_results: pd.DataFrame,
    tokenizer,
    model: str,
    best_step: int,
    note: str,
) -> None:
    short = model.split("/")[-1]
    gm = df_dumps[df_dumps["model"] == model]
    gres = df_results[df_results["model"] == model]
    print("\n" + "═" * 110)
    print(f"  [{short}]  best fixed step = {best_step}  ({note})")
    print("═" * 110)

    # Look up per-sample Fisher r=1 trajectory for these dump samples
    sample_ids = gm["sample_id"].astype(int).tolist()
    fisher_r1 = gres[
        (gres["sample_id"].isin(sample_ids))
        & (gres["layer"] == "final")
        & (gres["alpha"] == 1.0)
    ][["sample_id", "gen_step", "null_ratio_post_rank1"]]

    for cond, label in [(False, "CONTROL    (expected YES)"),
                         (True,  "CONTRADICT (expected NO)")]:
        rows = gm[gm["contradiction"] == cond].head(N_PER_CONDITION)
        if rows.empty:
            continue
        print(f"\n  ── {label} ──")
        for _, row in rows.iterrows():
            sid = int(row["sample_id"])
            gen_token_ids = list(row["gen_token_ids"])
            gen_surprises = list(row["gen_surprises"])
            raw_idx = row.get("gen_p_t_topk_indices", None)
            raw_vals = row.get("gen_p_t_topk_values", None)
            topk_idx = list(raw_idx) if raw_idx is not None else []
            topk_vals = list(raw_vals) if raw_vals is not None else []
            generated_text = (row["generated_text"] or "").strip()

            print(f"\n  sample_id={sid}  cl={row['chain_length']}  is_correct={row['is_correct']}")
            # Show the FULL generated text (compact)
            text_compact = generated_text[:200].replace("\n", "\\n")
            if len(generated_text) > 200:
                text_compact += "…"
            print(f"    gen_text: {text_compact}")

            # Decode the window around best_step
            window_decoded = decode_step_window(tokenizer, gen_token_ids, best_step, WINDOW)

            # Build the per-step report
            print(f"    {'step':>4s}  {'token':<15s}  {'S_t':>7s}  {'F_r=1':>6s}  top-{TOPK_DISPLAY} probabilities")
            for step, token_str, _cumulative in window_decoded:
                idx0 = step - 1  # 0-indexed
                if idx0 >= len(gen_surprises):
                    break
                S_t = gen_surprises[idx0]
                # Look up Fisher r=1 from all_results
                fr1_row = fisher_r1[(fisher_r1["sample_id"] == sid) & (fisher_r1["gen_step"] == step)]
                if len(fr1_row) == 1:
                    fr1 = float(fr1_row["null_ratio_post_rank1"].iloc[0])
                    fr1_str = f"{fr1:>6.3f}"
                else:
                    fr1_str = f"{'n/a':>6s}"
                # Top-K probs
                if idx0 < len(topk_idx) and idx0 < len(topk_vals):
                    topk_str = fmt_topk(list(topk_idx[idx0]), list(topk_vals[idx0]), tokenizer, TOPK_DISPLAY)
                else:
                    topk_str = "(no topk capture)"
                tok_repr = repr(token_str)
                if len(tok_repr) > 15:
                    tok_repr = tok_repr[:13] + "…'"
                marker = "  ←" if step == best_step else ""
                print(f"    {step:>4d}  {tok_repr:<15s}  {S_t:>7.3f}  {fr1_str}  {topk_str}{marker}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    df_dumps = pd.read_parquet(args.run_dir / "all_trace_dumps.parquet")
    df_results = pd.read_parquet(args.run_dir / "all_results.parquet")

    from transformers import AutoTokenizer
    tokenizers: Dict[str, object] = {}
    for model, _, _ in ANOMALIES:
        try:
            tokenizers[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as exc:
            print(f"FAILED to load tokenizer for {model}: {exc}")

    print("=" * 110)
    print("v3.2 ANOMALY DIAGNOSTIC — per-sample step window + token decoding + top-K + Fisher r=1")
    print("=" * 110)
    print(
        "Window: best_step ± {} steps.  Top-K: {}.  Trace dumps: n={}/condition.\n"
        "Look for: at the best step, are control and contradiction samples clearly\n"
        "separated by Fisher r=1?  Does the top-K already encode the answer commitment?"
        .format(WINDOW, TOPK_DISPLAY, N_PER_CONDITION)
    )

    for model, best_step, note in ANOMALIES:
        if model not in tokenizers:
            continue
        diagnose_model(df_dumps, df_results, tokenizers[model], model, best_step, note)

    return 0


if __name__ == "__main__":
    sys.exit(main())

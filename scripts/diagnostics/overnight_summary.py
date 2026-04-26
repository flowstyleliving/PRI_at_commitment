#!/usr/bin/env python3
"""Overnight summary aggregator — reads all results from the overnight chain
and produces a single markdown summary the user can read first thing.

Reads:
  - paired_fisher_*.csv  (4 models, synthetic 2x2)
  - norm_diagnostic_*.csv at N=200 (4 models, J_n correction at sealed sample size)
  - factual_baseline_*.csv (4 models, unpaired TriviaQA-style)
  - factual_paired_fisher_*.csv (4 models, paired TriviaQA-style)

Writes:
  wiki/results/overnight-2026-04-26.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO = Path("/Users/msrk/Documents/PRI_at_commitment")
EXP = REPO / "experiments" / "v3-main-run" / "2026-04-24"
FACTUAL = REPO / "experiments" / "factual_pairs"
VAULT_OUT = Path("/Users/msrk/Desktop/the_GOAT/wiki/results/overnight-2026-04-26.md")

MODELS = [
    ("Llama 3.2 3B", "Llama-3.2-3B-Instruct-4bit"),
    ("Mistral 7B", "Mistral-7B-Instruct-v0.3-4bit"),
    ("Qwen 2.5 7B", "Qwen2.5-7B-Instruct-4bit"),
    ("Qwen 3 8B", "Qwen3-8B-4bit"),
]


def safe_read(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            return None
    return None


def paired_delta_ci(df, col_a, col_b, n_resamples=1000, seed=20260423):
    if df is None or col_a not in df.columns or col_b not in df.columns:
        return None
    sa = df[col_a].to_numpy()
    sb = df[col_b].to_numpy()
    y = df.contradiction.astype(int).to_numpy()
    auc_a = roc_auc_score(y, sa)
    auc_b = roc_auc_score(y, sb)
    sign_a = +1 if auc_a >= 0.5 else -1
    sign_b = +1 if auc_b >= 0.5 else -1
    a = auc_a if sign_a == 1 else 1 - auc_a
    b = auc_b if sign_b == 1 else 1 - auc_b
    delta = a - b
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if yi.min() == yi.max(): continue
        ai = roc_auc_score(yi, sa[idx])
        bi = roc_auc_score(yi, sb[idx])
        ai = ai if sign_a == 1 else 1 - ai
        bi = bi if sign_b == 1 else 1 - bi
        deltas.append(ai - bi)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return delta, (float(lo), float(hi)), a, b, sign_a, sign_b


def main() -> int:
    out = []
    out.append("# Overnight 2026-04-26 — auto-generated summary")
    out.append("")
    out.append(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `scripts/diagnostics/overnight_summary.py`._")
    out.append("")
    out.append("Companion to [v3.1-replicate](v3.1-replicate.md). Stages run sequentially overnight; each stage has its own CSVs in `experiments/v3-main-run/2026-04-24/` (synthetic) or `experiments/factual_pairs/` (factual).")
    out.append("")

    # ============ STAGE 1: paired Fisher synthetic, cross-model ============
    out.append("## Stage 1 — Paired Fisher distance, synthetic 2×2 puzzles, cross-model")
    out.append("")
    out.append("Same prompt structure within pair, only contradiction bit flipped. Per-pair Fisher-Rao distance / KL / JSD / Hellinger.")
    out.append("")
    out.append("| Model | N pairs | mean FR | median FR | mean JSD | mean KL(c→c̃) | mean KL(c̃→c) | KL asym ratio | argmax-flip rate |")
    out.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for label, tag in MODELS:
        df = safe_read(EXP / f"paired_fisher_{tag}.csv")
        if df is None:
            out.append(f"| {label} | — | (file missing) | | | | | | |")
            continue
        kl_fwd = df.kl_ctrl_to_contr.mean()
        kl_bwd = df.kl_contr_to_ctrl.mean()
        flip_rate = 1 - df.same_top1_token.mean()
        out.append(
            f"| {label} | {len(df)} | {df.fisher_rao.mean():.3f} | {df.fisher_rao.median():.3f} | "
            f"{df.jsd.mean():.3f} | {kl_fwd:.3f} | {kl_bwd:.3f} | {kl_fwd/kl_bwd:.2f} | {flip_rate:.0%} |"
        )
    out.append("")

    # ============ STAGE 2: J_n at N=200 sealed sample size ============
    out.append("## Stage 2 — J_n-corrected null_ratio at N=200 (sealed sample size), cross-model")
    out.append("")
    out.append("Sealed-equivalent N. Δ AUROC(Fisher) − AUROC(raw) at rank=1 under proper Fisher pullback (J_n correction). Sealed E17b bar: Δ ≥ +0.02 with non-overlap CI.")
    out.append("")
    out.append("| Model | N | Δ(F-R) at rank=1 | 95% CI | sealed verdict |")
    out.append("|---|:---:|:---:|:---|:---:|")
    for label, tag in MODELS:
        df = safe_read(EXP / f"norm_diagnostic_{tag}.csv")
        if df is None:
            out.append(f"| {label} | — | (file missing) | | |")
            continue
        n = len(df)
        if n < 150:
            note = f"(only N={n}; expected N=200)"
        else:
            note = ""
        result = paired_delta_ci(df, "nr_fisher_jn_r1", "nr_raw_jn_r1")
        if result is None:
            out.append(f"| {label} | {n} | (cols missing) {note} | | |")
            continue
        delta, ci, a, b, sa, sb = result
        verdict = "✅ PASS" if (delta >= 0.02 and ci[0] > 0) else ("FAIL" if delta < 0 else "borderline")
        out.append(f"| {label} | {n} {note} | {delta:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {verdict} |")
    out.append("")
    out.append("Per-rank table for each model in the underlying CSVs at `experiments/v3-main-run/2026-04-24/norm_diagnostic_<MODEL>.csv`.")
    out.append("")

    # ============ STAGE 3: factual baseline (unpaired) ============
    out.append("## Stage 3 — Factual baseline (unpaired TriviaQA-style), cross-model")
    out.append("")
    out.append("60 hand-curated factual questions, prompt format `\"Question: <Q>\\nAnswer:\"`. Per-question: surprise on correct first-token, surprise on wrong first-token, log-ratio (>0 means model prefers correct).")
    out.append("")
    out.append("| Model | N | mean S(correct) | mean S(wrong) | mean log(p_corr/p_wrong) | prefers correct | top1 = correct first tok |")
    out.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for label, tag in MODELS:
        df = safe_read(FACTUAL / f"factual_baseline_{tag}.csv")
        if df is None:
            out.append(f"| {label} | — | (file missing) | | | | |")
            continue
        prefers = (df.log_p_correct_over_p_wrong > 0).mean()
        out.append(
            f"| {label} | {len(df)} | {df.surprise_on_correct_first_tok.mean():.2f} | "
            f"{df.surprise_on_wrong_first_tok.mean():.2f} | "
            f"{df.log_p_correct_over_p_wrong.mean():+.2f} | "
            f"{prefers:.0%} | {df.top1_is_correct_first.mean():.0%} |"
        )
    out.append("")

    # ============ STAGE 4: factual paired Fisher ============
    out.append("## Stage 4 — Factual paired Fisher distance (TriviaQA-style)")
    out.append("")
    out.append("Same factual pairs as Stage 3, but now BOTH the correct-answer-proposed and wrong-answer-proposed prompts are run; paired-Fisher metric per pair.")
    out.append("")
    out.append("| Model | N pairs | mean FR | median FR | mean JSD | mean KL(c→c̃) | argmax-flip rate |")
    out.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for label, tag in MODELS:
        df = safe_read(FACTUAL / f"factual_paired_fisher_{tag}.csv")
        if df is None:
            out.append(f"| {label} | — | (file missing) | | | | |")
            continue
        flip_rate = 1 - df.same_top1_token.mean()
        out.append(
            f"| {label} | {len(df)} | {df.fisher_rao.mean():.3f} | {df.fisher_rao.median():.3f} | "
            f"{df.jsd.mean():.3f} | {df.kl_ctrl_to_contr.mean():.3f} | {flip_rate:.0%} |"
        )
    out.append("")

    # Cross-stage comparison: synthetic vs factual
    out.append("## Headline comparison — does paired Fisher generalize from synthetic to factual?")
    out.append("")
    out.append("| Model | Synth FR (Stage 1) | Factual FR (Stage 4) | Δ (factual − synth) |")
    out.append("|---|:---:|:---:|:---:|")
    for label, tag in MODELS:
        synth_df = safe_read(EXP / f"paired_fisher_{tag}.csv")
        fact_df = safe_read(FACTUAL / f"factual_paired_fisher_{tag}.csv")
        synth_fr = synth_df.fisher_rao.mean() if synth_df is not None else None
        fact_fr = fact_df.fisher_rao.mean() if fact_df is not None else None
        if synth_fr is None or fact_fr is None:
            out.append(f"| {label} | {'—' if synth_fr is None else f'{synth_fr:.3f}'} | {'—' if fact_fr is None else f'{fact_fr:.3f}'} | (incomplete) |")
        else:
            out.append(f"| {label} | {synth_fr:.3f} | {fact_fr:.3f} | {fact_fr - synth_fr:+.3f} |")
    out.append("")
    out.append("If factual FR is meaningfully > 0 across models AND tracks synthetic FR direction, the paired Fisher metric generalizes to natural language and is a candidate v3-paper second-pillar metric. If factual FR is near 0 or noise, the metric is template-bound (claim narrowly).")
    out.append("")

    out.append("## Underlying CSVs (paths)")
    out.append("")
    out.append("Synthetic 2×2:")
    for label, tag in MODELS:
        out.append(f"- `experiments/v3-main-run/2026-04-24/paired_fisher_{tag}.csv` — Stage 1")
        out.append(f"- `experiments/v3-main-run/2026-04-24/norm_diagnostic_{tag}.csv` — Stage 2 (N=200)")
    out.append("")
    out.append("Factual:")
    for label, tag in MODELS:
        out.append(f"- `experiments/factual_pairs/factual_baseline_{tag}.csv` — Stage 3 (unpaired)")
        out.append(f"- `experiments/factual_pairs/factual_paired_fisher_{tag}.csv` — Stage 4 (paired)")
    out.append("")

    VAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    VAULT_OUT.write_text("\n".join(out))
    print(f"Wrote {VAULT_OUT} ({len(out)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

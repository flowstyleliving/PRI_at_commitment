#!/usr/bin/env python3
"""Pilot: do v3 residual-stream metrics (null_ratio, d_F_full, kl_discharged)
work at the t=0 prefix-last-position locus?

v4 showed attention metrics at t=0 discriminate YES/NO reliably. v3's
residual-stream cells (Fisher/Raw null_ratio, d_F_full, kl_discharged) were
only ever measured at gen_step=1. This checks whether the same signal lives
at t=0 — the last prefix-token hidden state, before any generation — and
compares AUROC head-to-head with gen_step=1 on the same samples.

t=0 inputs:
  h_t    = last_prefix_hidden[layer]          (last prompt-token hidden state)
  h_prev = prefix_hidden[layer][-2]           (second-to-last prompt token)
  p_t    = prefix_probs[-1]                   (logit distribution at last prefix pos)
  S_t    = gen_surprises[0]                   (surprise of first generated token)

gen_step=1 inputs (standard calibrator):
  h_t    = gen_hidden[0]                      (hidden state after first gen token)
  h_prev = last_prefix_hidden[layer]
  p_t    = gen_probs[0]
  S_t    = gen_surprises[0]

Both measurements use the same trace (max_new_tokens=1) and the same compute_step
call — no pipeline changes needed. AUROC is sign-free (max(auc, 1-auc)) because
this is exploratory; sign direction is not pre-registered.

Usage:
  .venv/bin/python scripts/pilot_t0_residual.py \\
      --data experiments/v4-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl

Models (default: Mistral-7B, Qwen2.5-7B, Gemma-3-4B):
  --models Mistral-7B Qwen2.5-7B Gemma-3-4B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pri_calibrator as cal
import pri_runtime as pipeline  # noqa: F401 — needed for side-effects on import path

SLUG_MAP: Dict[str, str] = {
    "Mistral-7B":  "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "Qwen2.5-7B":  "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Gemma-3-4B":  "mlx-community/gemma-3-4b-it-4bit",
}

# (display_label, key in compute_step output dict)
CELLS: List[Tuple[str, str]] = [
    ("d_F_full",      "d_F_full"),
    ("kl_discharged", "kl_discharged"),
    ("Fisher_r1",     "null_ratio_post_rank1"),
    ("Fisher_r2",     "null_ratio_post_rank2"),
    ("Raw_r21",       "null_ratio_raw_post_rank21"),
]


def _auroc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return (signed_auroc, signfree_auroc). signed is raw; signfree = max(auc, 1-auc)."""
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(scores)
    n_ok = int(ok.sum())
    if n_ok < 10:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(labels[ok], scores[ok]))
    return auc, max(auc, 1.0 - auc)


def _call_compute_step(
    pri_computer,
    h_t: np.ndarray,
    h_prev: np.ndarray,
    p_t: np.ndarray,
    S_t: float,
) -> Dict[str, float]:
    return pri_computer.compute_step(
        h_t=h_t,
        h_prev=h_prev,
        p_t=p_t,
        S_t=S_t,
        alpha=1.0,
        topk_values=[32],
        lowrank_values=[32],
        v3_rank_values=[1, 2, 4, 21],
        v3_capture_raw=True,
        v3_capture_centered=True,
    )


def run_model(
    slug: str,
    prompts: List[str],
    labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Return {cell_label: {"t0": auroc, "s1": auroc}} for all CELLS."""
    print(f"  loading {slug} ...", flush=True)
    state = cal.load_calibration_state(slug)
    lname = state.layer_name  # "final"

    t0_raw: Dict[str, List[float]] = {c[0]: [] for c in CELLS}
    s1_raw: Dict[str, List[float]] = {c[0]: [] for c in CELLS}

    for i, prompt in enumerate(prompts):
        trace = cal._trace_one_prompt(
            state.model,
            state.tokenizer,
            state.projection,
            state.layer_indices,
            prompt,
            state.prompt_strategy,
            max_new_tokens=1,
        )

        prefix_seq = trace["prefix_hidden"][lname]   # list[np.ndarray], length T
        gen_hid    = trace["gen_hidden"][lname]       # list[np.ndarray], length 0 or 1
        gen_probs  = trace["gen_probs"]
        gen_surps  = trace["gen_surprises"]

        # Shared S_t: surprise of the first generated token (= surprise of
        # committing, measured from prefix_probs[-1]).
        S_commit = float(gen_surps[0]) if gen_surps else 0.0

        # ── t=0: last prefix-token hidden state ─────────────────────────
        h_t0    = trace["last_prefix_hidden"][lname]
        h_prev0 = prefix_seq[-2] if len(prefix_seq) >= 2 else h_t0
        p_t0    = trace["prefix_probs"][-1]

        r0 = _call_compute_step(state.pri_computer, h_t0, h_prev0, p_t0, S_commit)

        # ── gen_step=1: first generated token's hidden state ────────────
        r1: Optional[Dict] = None
        if gen_hid and gen_probs:
            r1 = _call_compute_step(
                state.pri_computer,
                h_t=gen_hid[0],
                h_prev=h_t0,         # last_prefix IS h_prev at gen_step=1
                p_t=gen_probs[0],
                S_t=S_commit,
            )

        for label, key in CELLS:
            t0_raw[label].append(r0.get(key, float("nan")))
            s1_raw[label].append(r1.get(key, float("nan")) if r1 else float("nan"))

        if (i + 1) % 20 == 0 or i + 1 == len(prompts):
            print(f"    [{i+1}/{len(prompts)}]", flush=True)

    results: Dict[str, Dict[str, float]] = {}
    for label, _ in CELLS:
        t0_arr = np.array(t0_raw[label], dtype=np.float32)
        s1_arr = np.array(s1_raw[label], dtype=np.float32)
        t0_signed, t0_sf = _auroc(t0_arr, labels)
        s1_signed, s1_sf = _auroc(s1_arr, labels)
        results[label] = {
            "t0":        t0_sf,
            "t0_signed": t0_signed,
            "s1":        s1_sf,
            "s1_signed": s1_signed,
            "n_t0_finite": int(np.isfinite(t0_arr).sum()),
            "n_s1_finite": int(np.isfinite(s1_arr).sum()),
        }
    return results


def _fmt(v: float) -> str:
    return f"{v:.3f}" if not np.isnan(v) else "  nan"


def _sign_tag(t0_signed: float, s1_signed: float) -> str:
    """⚠ if the two sides have opposite signs — a sign flip in the discriminant direction."""
    if np.isnan(t0_signed) or np.isnan(s1_signed):
        return ""
    if (t0_signed > 0.5) != (s1_signed > 0.5):
        return " ⚠sign"
    return ""


def _print_table(short_name: str, slug: str, results: Dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {short_name}  ({slug})")
    print(f"{'=' * 72}")
    print(f"  {'Cell':<18s}  {'t=0(sf)':>8s}  {'t=0(sgn)':>9s}  {'s=1(sf)':>8s}  {'s=1(sgn)':>9s}  {'Δsf':>6s}  {'n_s1':>5s}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*5}")
    for label, _ in CELLS:
        r = results[label]
        t0, s1 = r["t0"], r["s1"]
        delta = (t0 - s1) if (not np.isnan(t0) and not np.isnan(s1)) else float("nan")
        tag = _sign_tag(r["t0_signed"], r["s1_signed"])
        print(
            f"  {label:<18s}  {_fmt(t0):>8s}  {_fmt(r['t0_signed']):>9s}"
            f"  {_fmt(s1):>8s}  {_fmt(r['s1_signed']):>9s}"
            f"  {_fmt(delta):>6s}  {r['n_s1_finite']:>5d}{tag}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to calibration .jsonl")
    ap.add_argument(
        "--models", nargs="+", default=list(SLUG_MAP.keys()),
        help="short model names to run (default: all three)",
    )
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    prompts = [r["prompt"] for r in rows]
    labels  = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    print(f"Loaded {len(prompts)} samples  (pos={int((labels==1).sum())}, neg={int((labels==0).sum())})")
    print(f"Data: {args.data}\n")

    all_results: Dict[str, Dict] = {}
    for short_name in args.models:
        slug = SLUG_MAP.get(short_name, short_name)
        print(f"\n>>> {short_name}", flush=True)
        results = run_model(slug, prompts, labels)
        all_results[short_name] = results
        _print_table(short_name, slug, results)
        # Release model weights + MLX buffer cache before loading the next model.
        # Pattern from anli_full_sweep.py — prevents silent OOM on sequential loads.
        import gc
        gc.collect()
        try:
            cal.pipeline.clear_mlx_cache()
        except Exception:
            pass

    # Cross-model summary: which cells improve at t=0?
    print(f"\n{'=' * 60}")
    print("  Summary: cells where t=0 beats s=1 across models")
    print(f"{'=' * 60}")
    print(f"  {'Cell':<18s}", end="")
    for name in args.models:
        print(f"  {name[:10]:>10s}", end="")
    print()
    print(f"  {'-'*18}", end="")
    for _ in args.models:
        print(f"  {'-'*10}", end="")
    print()
    for label, _ in CELLS:
        print(f"  {label:<18s}", end="")
        for name in args.models:
            r = all_results[name][label]
            t0, s1 = r["t0"], r["s1"]
            if np.isnan(t0) or np.isnan(s1):
                sym = "     nan"
            else:
                delta = t0 - s1
                tag = _sign_tag(all_results[name][label]["t0_signed"], all_results[name][label]["s1_signed"])
                sym = f"{delta:+.3f} {'↑' if delta > 0.01 else ('↓' if delta < -0.01 else '≈')}{tag}"
            print(f"  {sym:>10s}", end="")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

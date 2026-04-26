#!/usr/bin/env python3
"""Diagnose what tokens dominate the top-r right singular vectors of W_u
for each primary model.

Hypothesis (2026-04-24, after Qwen 3 norm diagnostic): Qwen's
multilingual training (heavy CJK + English) produces a W_u with
fundamentally different SVD geometry than Mistral/Llama. The "Fisher
loses to raw" finding on Qwen E17b might trace back to the static raw
W_u top-r being CJK-aligned, while Δh at English-prompt gen_step=1
lives in English-aligned subspaces.

For each model:
  1. Load and compute raw W_u SVD via OutputProjection._raw_svd_cache
  2. For each of top-r right singular vectors V_i:
     - Compute |W_u[t] · V_i| for all token IDs t
     - Take top-K tokens with highest projection magnitude
     - Decode them via tokenizer
     - Classify each token: ASCII-only / CJK-containing / mixed / punctuation / other
  3. Report the language-class composition of each top-r direction.

Strong CJK presence in Qwen's top-r would support the multilingual-training
hypothesis. Mistral/Llama should show English/format-token dominance.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from mlx_lm import load as mlx_load

import pri_v2_mlx_pipeline as pipeline


MODELS = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Qwen3-8B-4bit",
]
TOP_RANKS = [1, 2, 3, 5, 8, 13, 32]
TOP_K_PER_VECTOR = 20  # how many top-projecting tokens to inspect per V_i
MAX_RANK = max(TOP_RANKS)


def classify_token(s: str) -> str:
    """Classify a decoded token by character composition."""
    if not s or not s.strip():
        return "whitespace"
    has_cjk = any(
        ('一' <= ch <= '鿿')         # CJK unified
        or ('㐀' <= ch <= '䶿')      # CJK extension A
        or ('぀' <= ch <= 'ヿ')      # Hiragana + Katakana
        or ('가' <= ch <= '힯')      # Hangul
        for ch in s
    )
    has_ascii_letter = any(ch.isascii() and ch.isalpha() for ch in s)
    has_other_unicode = any(
        (not ch.isascii()) and not (
            ('一' <= ch <= '鿿')
            or ('㐀' <= ch <= '䶿')
            or ('぀' <= ch <= 'ヿ')
            or ('가' <= ch <= '힯')
        )
        for ch in s
    )
    if has_cjk:
        return "CJK_mixed" if has_ascii_letter else "CJK_only"
    if has_other_unicode and not has_ascii_letter:
        return "other_unicode"
    if has_ascii_letter:
        return "ascii_word"
    return "ascii_punct"


def get_W_u_rows(proj: pipeline.OutputProjection, n_rows: int = 8192) -> np.ndarray:
    """Sample n_rows from W_u (via get_rows) — covering token IDs 0..n_rows-1.
    For analysis purposes we sample enough rows to find top-projectors;
    full-V is too expensive but 8192 catches the high-prob English subset
    plus a chunk of the lower-prob tail."""
    idx = np.arange(n_rows)
    rows = proj.get_rows(idx)
    return rows  # (n_rows, D)


def analyze_model(model_name: str) -> dict:
    print(f"\n{'='*76}\n  Loading {model_name}\n{'='*76}")
    model, tokenizer = mlx_load(model_name)
    proj = pipeline.OutputProjection(model)
    print(f"  V={proj.vocab_size}, D={proj.hidden_size}, mode={proj.mode}")

    print(f"  Computing raw W_u SVD top-{MAX_RANK}...")
    basis_pkg = proj.raw_right_singular_vectors(MAX_RANK)
    if basis_pkg is None:
        print("  [skip] raw SVD basis unavailable")
        return {"model": model_name, "skipped": True}
    Vt_raw, S_raw, total_sigma_sq = basis_pkg  # (max_rank, D), (max_rank,), scalar

    # Sample W_u rows for analysis. Full V is expensive (152K for Qwen).
    # Sample tokens 0..16K — covers most-common English + first chunk of vocab.
    n_sample = min(16384, proj.vocab_size)
    print(f"  Sampling top-{n_sample} token IDs for projection analysis...")
    W_sample = get_W_u_rows(proj, n_rows=n_sample)  # (n_sample, D)
    if W_sample is None:
        print("  [skip] W_u row fetch failed")
        return {"model": model_name, "skipped": True}

    print(f"  Computing |W_u[t] · V_i| for each rank up to r={MAX_RANK}...")
    # projections: (n_sample, max_rank)
    projs = W_sample @ Vt_raw.T

    out_per_rank = {}
    for r in TOP_RANKS:
        col = projs[:, r - 1]  # the r-th singular vector's projection
        abs_col = np.abs(col)
        top_idx = np.argpartition(-abs_col, kth=TOP_K_PER_VECTOR - 1)[:TOP_K_PER_VECTOR]
        top_idx = top_idx[np.argsort(-abs_col[top_idx])]
        toks = []
        for ti in top_idx:
            tok_id = int(ti)
            try:
                decoded = tokenizer.decode([tok_id]) if hasattr(tokenizer, "decode") else str(tok_id)
            except Exception:
                decoded = f"<id={tok_id}>"
            cls = classify_token(decoded)
            toks.append({
                "token_id": tok_id,
                "decoded": decoded,
                "projection": float(col[tok_id]),
                "abs_projection": float(abs_col[tok_id]),
                "class": cls,
            })

        # Class composition
        class_counts = {}
        for t in toks:
            class_counts[t["class"]] = class_counts.get(t["class"], 0) + 1

        # Σ singular value contribution
        sigma_r = float(S_raw[r - 1])
        sigma_total = float(np.sqrt(total_sigma_sq))

        out_per_rank[r] = {
            "rank": r,
            "sigma_r": sigma_r,
            "sigma_r_frac_of_total": sigma_r / (sigma_total + 1e-10),
            "class_counts": class_counts,
            "top_tokens": toks,
        }

    return {
        "model": model_name,
        "vocab_size": proj.vocab_size,
        "hidden_size": proj.hidden_size,
        "n_sample": n_sample,
        "per_rank": out_per_rank,
    }


def main() -> int:
    selected = sys.argv[1:] if len(sys.argv) > 1 else MODELS
    results = []
    for model_name in selected:
        try:
            r = analyze_model(model_name)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR on {model_name}: {exc}")
            results.append({"model": model_name, "error": str(exc)})

    # Summary table
    print("\n" + "=" * 76)
    print("  SUMMARY: language-class composition of top-K projecting tokens per rank")
    print("=" * 76)
    print(
        f"{'Model':<48} {'rank':>4} {'σ_r/σ_total':>11}  "
        f"{'ascii_word':>10} {'ascii_punct':>11} {'CJK_only':>9} {'CJK_mixed':>10} "
        f"{'other':>6} {'whitespace':>10}"
    )
    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        short = r["model"].split("/")[-1][:48]
        for rank in TOP_RANKS:
            d = r["per_rank"][rank]
            cc = d["class_counts"]
            print(
                f"{short:<48} {rank:>4} {d['sigma_r_frac_of_total']:>11.4f}  "
                f"{cc.get('ascii_word', 0):>10} "
                f"{cc.get('ascii_punct', 0):>11} "
                f"{cc.get('CJK_only', 0):>9} "
                f"{cc.get('CJK_mixed', 0):>10} "
                f"{cc.get('other_unicode', 0):>6} "
                f"{cc.get('whitespace', 0):>10}"
            )
        print()

    # Sample tokens at rank=1 per model, for color
    print("=" * 76)
    print("  TOP-10 PROJECTING TOKENS at rank=1 per model (decoded, |projection|)")
    print("=" * 76)
    for r in results:
        if r.get("skipped") or r.get("error"):
            continue
        short = r["model"].split("/")[-1]
        print(f"\n  {short}:")
        for t in r["per_rank"][1]["top_tokens"][:10]:
            preview = repr(t["decoded"])[:36]
            print(f"    [{t['class']:<14}] |proj|={t['abs_projection']:>9.4f}  tok={preview}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe Mistral's V_raw[0] direction directly: is it a YES/NO bipolar axis
or a 'rupture-magnitude' axis?

Key tests:
  1. cos(W_u[YES_tok], V_raw_top1) vs cos(W_u[NO_tok], V_raw_top1)
  2. cos(W_u[YES_tok], V_raw_top2/3/4)
  3. Same for ' Yes', ' No' (different tokenizations) and ' Answer'
  4. Project top-prob tokens onto V_raw_top1 — what's the spectrum?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from mlx_lm import load as mlx_load

import pri_v2_mlx_pipeline as pipeline


MODEL_NAME = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"


def main() -> int:
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = mlx_load(MODEL_NAME)
    proj = pipeline.OutputProjection(model)
    raw_basis_pkg = proj.raw_right_singular_vectors(8)
    Vt_raw, S_raw, _ = raw_basis_pkg
    print(f"  V={proj.vocab_size}, D={proj.hidden_size}")
    print(f"  σ_raw[0..7]: {S_raw[:8]}")

    # Get token IDs for YES/NO variants
    targets = [
        " YES", " NO", "YES", "NO", " Yes", " No", "Yes", "No",
        " yes", " no", " answer", " Answer", "Answer", "answer",
        ":", " ", "\n", " A", " B"
    ]
    token_ids = {}
    for s in targets:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False) if hasattr(tokenizer, "encode") else tokenizer(s)["input_ids"]
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            if not ids:
                continue
            # Use the LAST token if multi-token (e.g. ' Yes' might be [' ', 'Yes'])
            token_ids[s] = (int(ids[0]), int(ids[-1]))
        except Exception:
            pass

    # Fetch the W_u rows for these tokens
    print("\n=== Per-token: W_u row projection onto V_raw[0..3], normalized ===")
    print(f"{'token':<10} {'first_id':>9} {'last_id':>9} | {'||W_u row||':>11} | "
          f"{'cos(top1)':>10} {'cos(top2)':>10} {'cos(top3)':>10} {'cos(top4)':>10}")
    for s, (first_id, last_id) in token_ids.items():
        rows = proj.get_rows(np.array([last_id]))  # use last_id (more semantically meaningful for multi-token strings)
        if rows is None or rows.shape[0] == 0:
            continue
        w = rows[0]
        w_norm = float(np.linalg.norm(w))
        if w_norm == 0:
            continue
        cosines = [float(np.dot(w, Vt_raw[k]) / w_norm) for k in range(4)]
        print(f"{repr(s):<10} {first_id:>9} {last_id:>9} | {w_norm:>11.4f} | "
              f"{cosines[0]:>+10.4f} {cosines[1]:>+10.4f} {cosines[2]:>+10.4f} {cosines[3]:>+10.4f}")

    # Compute symmetry: is YES vs NO antisymmetric?
    print("\n=== YES/NO antisymmetry test ===")
    pairs = [(" YES", " NO"), ("YES", "NO"), (" Yes", " No"), ("Yes", "No")]
    for yes_s, no_s in pairs:
        if yes_s not in token_ids or no_s not in token_ids:
            continue
        y_id = token_ids[yes_s][1]
        n_id = token_ids[no_s][1]
        rows = proj.get_rows(np.array([y_id, n_id]))
        if rows is None or rows.shape[0] != 2:
            continue
        wy, wn = rows[0], rows[1]
        # Project both onto V_raw[0]
        py = float(np.dot(wy, Vt_raw[0]))
        pn = float(np.dot(wn, Vt_raw[0]))
        # Antisymmetry: if YES axis = -NO axis on top1, py ≈ -pn (sign flip)
        # If both same direction (magnitude axis), py and pn have same sign
        sign_match = "OPPOSITE (bipolar)" if py * pn < 0 else "SAME (magnitude)"
        print(f"  {yes_s!r} vs {no_s!r}: proj(top1) = {py:+.4f} vs {pn:+.4f}  →  {sign_match}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

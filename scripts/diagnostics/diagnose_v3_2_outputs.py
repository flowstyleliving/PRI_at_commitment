#!/usr/bin/env python3
"""v3.2 model-output viewer — print actual generated text per (model, sample).

Reads `all_trace_dumps.parquet` and prints up to N control + N contradiction
trace dumps per model, with the YES/NO answer + correctness + raw text.
Useful for diagnosing parser brittleness and model-specific format quirks
(Mistral's `Answer: YES` clean exit vs Qwen-family `Answer: YES Now solve
the following…` continuation vs Qwen 3's chain-of-thought-before-answer).

Usage:
    .venv/bin/python scripts/diagnostics/diagnose_v3_2_outputs.py \\
        --run-dir experiments/v3-main-run/<DATE>/run-NN \\
        [--per-cell 3] [--width 100]
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--per-cell", type=int, default=3,
                    help="number of dumps to show per (model, condition)")
    ap.add_argument("--width", type=int, default=100,
                    help="text wrap width")
    args = ap.parse_args()

    parquet = args.run_dir / "all_trace_dumps.parquet"
    if not parquet.exists():
        print(f"ERROR: {parquet} missing"); return 1
    df = pd.read_parquet(parquet)
    print(f"Trace dumps: {len(df)} rows")
    print("rows per (model, condition):")
    print(df.groupby(["model", "contradiction"]).size())

    print("\n" + "=" * 110)
    for m, g in df.groupby("model"):
        short = m.split("/")[-1]
        print(f"\n[{short}]")
        for cond in [False, True]:
            label = "CONTROL    (expected YES)" if not cond else "CONTRADICT (expected NO)"
            samples = g[g["contradiction"] == cond].head(args.per_cell)
            if samples.empty:
                print(f"  [no {label} samples]"); continue
            for _, row in samples.iterrows():
                txt = (row["generated_text"] or "").strip()
                wrapped = "\n             ".join(textwrap.wrap(txt, width=args.width)) \
                    if txt else "(empty)"
                mark = "OK" if row["is_correct"] else "MISS"
                print(f"  {label}  id={row['sample_id']:<3d}  cl={row['chain_length']}  {mark}")
                print(f"             {wrapped}")
                print()
    print("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())

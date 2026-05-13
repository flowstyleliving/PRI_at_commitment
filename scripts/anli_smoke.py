#!/usr/bin/env python3
"""ANLI R2 smoke on Mistral-Nemo-Instruct-2407-4bit.

Pulls 5 entailment + 5 contradiction examples from ANLI R2 dev, wraps each
with the Mistral-Nemo chat template (via pri_v2_io_plugins), and reports
the raw generated text + parser verdict at max_gen_tokens=24 and =64.

Purpose: sanity-check whether the rupture-signal pipeline can be moved from
synthetic logic puzzles (which have AUROC-saturated at 1.000 with stereotyped
output) to natural-language NLI. Key questions:

  1. Does Mistral-Nemo still emit clean YES/NO on natural-language NLI, or
     does the longer/varied premise break the terminal-commit pattern?
  2. Does it commit at step 1 (matching the synthetic-puzzle pattern) or
     does it now reason for several tokens first?
  3. What's the parsing surface — same Tier-0 (bare first word) hit, or
     do we get new failure modes?

This is a TEXT-ONLY smoke, NOT a full PRI pipeline run. Just generation.

Usage:
    .venv/bin/python scripts/anli_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from mlx_lm import load as mlx_load
from mlx_lm import generate as mlx_generate

import pri_v2_io_plugins as io_plugins


MODEL_SLUG = "mlx-community/Mistral-Nemo-Instruct-2407-4bit"

# Same instruction shape as v3.2 logic puzzles — premise(s) + question + Answer:
PROMPT_TEMPLATE = (
    "Instruction: Read the premise and decide whether the hypothesis is "
    "entailed by the premise. Answer YES if the premise entails the "
    "hypothesis, NO if the premise contradicts the hypothesis.\n"
    "\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Answer:"
)


def pick_samples(ds, n_per_class: int = 5):
    """Return [(label_int, premise, hypothesis, expected), ...] ordered
    entailment-then-contradiction. label semantics: 0=entail (YES),
    2=contradict (NO), 1=neutral (skipped)."""
    out = []
    seen_e = 0
    seen_c = 0
    for ex in ds:
        if ex["label"] == 0 and seen_e < n_per_class:
            out.append((0, ex["premise"], ex["hypothesis"], "YES"))
            seen_e += 1
        elif ex["label"] == 2 and seen_c < n_per_class:
            out.append((2, ex["premise"], ex["hypothesis"], "NO"))
            seen_c += 1
        if seen_e == n_per_class and seen_c == n_per_class:
            break
    return out


def run_one(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate text from the chat-template-wrapped prompt."""
    strategy = io_plugins.get_prompt_strategy(MODEL_SLUG)
    wrapped = strategy(prompt, tokenizer)
    out = mlx_generate(
        model,
        tokenizer,
        prompt=wrapped,
        max_tokens=max_tokens,
        verbose=False,
    )
    return out


def main() -> int:
    print(f"Loading ANLI R2 dev split...")
    ds = load_dataset("facebook/anli", split="dev_r2")
    samples = pick_samples(ds, n_per_class=5)
    print(f"Loaded {len(samples)} samples (5 entailment + 5 contradiction)")
    print()

    print(f"Loading {MODEL_SLUG}...")
    model, tokenizer = mlx_load(MODEL_SLUG)
    print(f"Loaded.")
    print()

    correct_24 = 0
    correct_64 = 0
    for i, (label, premise, hypothesis, expected) in enumerate(samples):
        kind = "ENTAILMENT (YES)" if label == 0 else "CONTRADICTION (NO)"
        print("=" * 110)
        print(f"  Sample {i+1}/10  [{kind}]  premise_len={len(premise)} chars  hyp_len={len(hypothesis)}")
        print("=" * 110)
        print(f"  premise:    {premise[:280]!r}{'...' if len(premise) > 280 else ''}")
        print(f"  hypothesis: {hypothesis!r}")
        prompt = PROMPT_TEMPLATE.format(premise=premise, hypothesis=hypothesis)

        # max_tokens=24 (v3.2 protocol)
        out24 = run_one(model, tokenizer, prompt, max_tokens=24)
        parsed24 = io_plugins.parse_yes_no(out24)
        c24 = (parsed24 == expected)
        if c24:
            correct_24 += 1
        print(f"\n  --- max_tokens=24  ({'OK' if c24 else 'FAIL'}) ---")
        print(f"    generated: {out24!r}")
        print(f"    parsed:    {parsed24!r}  expected={expected!r}")

        # max_tokens=64 — does giving more budget change anything?
        out64 = run_one(model, tokenizer, prompt, max_tokens=64)
        parsed64 = io_plugins.parse_yes_no(out64)
        c64 = (parsed64 == expected)
        if c64:
            correct_64 += 1
        print(f"\n  --- max_tokens=64  ({'OK' if c64 else 'FAIL'}) ---")
        print(f"    generated: {out64!r}")
        print(f"    parsed:    {parsed64!r}  expected={expected!r}")
        print()

    print("=" * 110)
    print(f"  SUMMARY")
    print("=" * 110)
    print(f"  max_tokens=24:  {correct_24}/10 = {correct_24/10:.0%}")
    print(f"  max_tokens=64:  {correct_64}/10 = {correct_64/10:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

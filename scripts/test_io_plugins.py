#!/usr/bin/env python3
"""Test pri_v2_io_plugins.parse_yes_no against the n=140 collected outputs.

Loads:
  /tmp/n20_outputs.json          — 60 outputs from 3 failed-smoke models
                                   (Mistral-Nemo, Gemma-3-1B, Dolphin-Nemo)
                                   generated 2026-05-11 with chat-template
                                   wrapping.
  /tmp/working_models_outputs.json — 80 outputs from 8 working models'
                                     existing trace_dumps.

For each output, compute:
  * NEW parser:    pri_v2_io_plugins.parse_yes_no (4-tier)
  * OLD parser:    inline reproduction of pri_v2_mlx_pipeline.check_answer's
                   3-tier logic (without the io_plugins module)

Reports per-model accuracy + per-model deltas. Validates the fix without
re-running any models.

Usage:
    .venv/bin/python scripts/test_io_plugins.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pri_v2_io_plugins import parse_yes_no, DEFAULT_TIERS  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Old 3-tier parser (inline reproduction from pri_v2_mlx_pipeline.check_answer)
# ─────────────────────────────────────────────────────────────────────────────


def old_parser(generated: str) -> str | None:
    """Reproduce the pre-2026-05-11 pipeline parser without the io_plugins
    Tier-0 (bare first word). Used as the baseline to measure the fix's gain."""
    if not generated:
        return None
    text = re.sub(r"<\|[^|>]+?\|>", " ", str(generated).strip())

    last_answer = None
    for m in re.finditer(
        r"(?:(?:^|\n)\s*|[\.\:]\s+)answer\s*[:=]?\s*[\"']?(YES|NO)\b",
        text, re.IGNORECASE,
    ):
        last_answer = m.group(1).upper()
    if last_answer is not None:
        return last_answer

    for ln in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        m = re.fullmatch(
            r"[\s\*\"'\.\!\?\-\:\(\)]*(YES|NO)[\s\*\"'\.\!\?\-\:\(\)]*",
            ln, re.IGNORECASE,
        )
        if m:
            return m.group(1).upper()

    last_token = None
    for m in re.finditer(r"[A-Za-z]+", text):
        tok = m.group(0).upper()
        if tok in {"YES", "NO"}:
            last_token = tok
    return last_token


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(outputs_dict: dict, source_label: str) -> None:
    print(f"\n{'=' * 95}")
    print(f"  {source_label}")
    print(f"{'=' * 95}")
    print(
        f"  {'model':<38s}  {'n':>3s}  {'OLD acc':>8s}  {'NEW acc':>8s}  "
        f"{'Δ':>6s}  {'tier_0_fires':>12s}"
    )
    total_old, total_new, total_n = 0, 0, 0
    for model, info in outputs_dict.items():
        outs = info["outputs"]
        old_correct = 0
        new_correct = 0
        new_via_tier_0 = 0
        for o in outs:
            expected = o["expected"]  # "YES" or "NO"
            gen = o["generated"]
            old_p = old_parser(gen)
            new_p = parse_yes_no(gen)
            # Check whether Tier 0 (bare_first_word) is what produced the
            # new result — fires for outputs that previously hit Tier 3 / fell
            # through the old logic.
            from pri_v2_io_plugins import tier_answer_prefix, tier_bare_first_word
            t1 = tier_answer_prefix(gen)
            t0 = tier_bare_first_word(gen) if t1 is None else None
            if t1 is None and t0 is not None:
                new_via_tier_0 += 1
            if old_p == expected:
                old_correct += 1
            if new_p == expected:
                new_correct += 1
        old_acc = old_correct / len(outs) if outs else 0
        new_acc = new_correct / len(outs) if outs else 0
        delta = new_acc - old_acc
        d_str = f"{delta:+.3f}" if delta != 0 else "  0.00"
        print(
            f"  {model:<38s}  {len(outs):>3d}  {old_acc:>7.2%}  {new_acc:>7.2%}  "
            f"{d_str:>5s}  {new_via_tier_0:>12d}"
        )
        total_old += old_correct
        total_new += new_correct
        total_n += len(outs)
    delta_total = (total_new - total_old) / total_n
    d_total_str = f"{delta_total:+.3f}"
    print(
        f"  {'TOTAL':<38s}  {total_n:>3d}  "
        f"{total_old / total_n:>7.2%}  {total_new / total_n:>7.2%}  "
        f"{d_total_str:>6s}"
    )


def synthetic_case_variations() -> int:
    """Targeted tests for case-variation coverage across every tier + the
    emphatic-closing prefix list. These are unit-style assertions on
    contrived inputs — not on real model outputs."""
    from pri_v2_io_plugins import parse_yes_no, EMPHATIC_CLOSING_PREFIXES

    print(f"\n{'=' * 95}")
    print(f"  SYNTHETIC CASE-VARIATION TESTS  (emphatic prefixes: {EMPHATIC_CLOSING_PREFIXES})")
    print(f"{'=' * 95}")

    cases = [
        # Tier 1: Answer: prefix — case variations
        ("Answer: Yes",              "YES",  "Tier 1 case: Title"),
        ("Answer: NO",               "NO",   "Tier 1 case: Upper"),
        ("ANSWER: YES",              "YES",  "Tier 1 case: All-caps prefix"),
        ("answer: yes",              "YES",  "Tier 1 case: All-lower"),
        ("aNsWeR: yEs",              "YES",  "Tier 1 case: Mixed"),
        ("Answer = YES",             "YES",  "Tier 1: equals delimiter"),
        ("Answer YES",               "YES",  "Tier 1: no delimiter"),
        # Tier 0.5: emphatic-closing
        ("Final Answer: YES",        "YES",  "Tier 0.5: Final Answer"),
        ("Final Answer: yes",        "YES",  "Tier 0.5: Final Answer case"),
        ("FINAL ANSWER: NO",         "NO",   "Tier 0.5: Final Answer all-caps"),
        ("Final Answer is yes",      "YES",  "Tier 0.5: Final Answer is"),
        ("Conclusion: YES",          "YES",  "Tier 0.5: Conclusion"),
        ("Conclusion: no",           "NO",   "Tier 0.5: Conclusion lower"),
        ("CONCLUSION = YES",         "YES",  "Tier 0.5: Conclusion equals"),
        # Precedence: Tier 0.5 wins over Tier 1
        ("Answer: NO. Final Answer: YES.",  "YES",  "Tier 0.5 overrides Tier 1"),
        ("Answer: YES. Conclusion: no.",    "NO",   "Tier 0.5 overrides Tier 1"),
        # Precedence: Tier 0.5 takes LAST occurrence
        ("Final Answer: NO. Final Answer: YES.",  "YES",  "Tier 0.5: last wins"),
        # Tier 0: bare first word — case variations
        ("YES",                      "YES",  "Tier 0: bare upper"),
        ("yes",                      "YES",  "Tier 0: bare lower"),
        ("Yes",                      "YES",  "Tier 0: bare title"),
        ("NO.",                      "NO",   "Tier 0: bare with period"),
        ("no",                       "NO",   "Tier 0: bare lower NO"),
        # Tier 2: trailing-line case variations
        ("Some reasoning\nyes",      "YES",  "Tier 2: trailing lower"),
        ("Reasoning here\n\nNO.",    "NO",   "Tier 2: trailing NO"),
        # Tier 3: last-match fallback
        ("I think it could be NO but actually YES seems right",  "YES",  "Tier 3: last match"),
        # Edge cases — false positives we must NOT trigger
        ("My response is incomplete",        None,   "no YES/NO present"),
        ("",                                  None,   "empty"),
    ]

    fails = []
    for input_text, expected, label in cases:
        got = parse_yes_no(input_text)
        mark = "OK  " if got == expected else "FAIL"
        if got != expected:
            fails.append((label, input_text, expected, got))
        txt_short = input_text[:60].replace("\n", "\\n")
        print(f"  {mark}  expected={str(expected):<5s}  got={str(got):<5s}  {label:<35s}  in={txt_short!r}")

    print(f"\n  {len(cases) - len(fails)}/{len(cases)} pass")
    return 1 if fails else 0


def main() -> int:
    n20 = Path("/tmp/n20_outputs.json")
    working = Path("/tmp/working_models_outputs.json")

    print("Plugin loaded:", [t.__name__ for t in DEFAULT_TIERS])

    rc_synth = synthetic_case_variations()

    if not n20.exists():
        print(f"missing {n20}; skipping corpus tests")
        return rc_synth
    if not working.exists():
        print(f"missing {working}; skipping corpus tests")
        return rc_synth

    evaluate(json.loads(n20.read_text()),
             "FAILED-SMOKE MODELS — n=20 each, chat-template applied")
    evaluate(json.loads(working.read_text()),
             "WORKING MODELS — 10 trace_dumps each (raw-prompt regime)")

    return rc_synth


if __name__ == "__main__":
    sys.exit(main())

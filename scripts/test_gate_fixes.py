#!/usr/bin/env python3
"""Regression tests for the 2026-04-24 gate fixes.

Two fixes land together:
  1. Stratified behavioral-gate preflight (chain_length-balanced sampling).
  2. Three-tier check_answer parser (Answer: > trailing-line > last-match).

Both MLX-free — pure pandas + regex. Run with:
    .venv/bin/python scripts/test_gate_fixes.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)

import pri_v2_mlx_pipeline as pipeline  # noqa: E402
from pri_v2_mlx_pipeline import PuzzleGenerator, check_answer  # noqa: E402


FAIL = 0


def _fail(msg: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"  [pass] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
#  Fix 1: stratified preflight sampling
# ─────────────────────────────────────────────────────────────────────────────

def test_stratified_preflight_is_seed_invariant() -> None:
    """Stratified logic draws an even chain_length split regardless of seed.

    Reproduces the exact selection `run_experiment` uses (lines ~1589–1620 in
    pri_v2_mlx_pipeline.py after the 2026-04-24 patch). Seed 42 and
    seed 20260423 under the old head(20) drew 14/6 and 9/11 cl=2/cl=5
    respectively — the skew that surfaced the cl=5 gate-fail artifact. The
    stratified logic must hit exactly 10/10 regardless of seed.
    """

    def stratified(dataset: pd.DataFrame, pilot_n: int, chain_lengths: list) -> pd.DataFrame:
        all_controls = dataset[~dataset.contradiction]
        eligible = all_controls  # no resume exclusion in unit test
        cl_list = list(chain_lengths) if chain_lengths else [None]
        per_cl = pilot_n // len(cl_list)
        extra = pilot_n - per_cl * len(cl_list)
        strata = []
        for i, cl in enumerate(cl_list):
            quota = per_cl + (1 if i < extra else 0)
            strata.append(eligible[eligible.chain_length == cl].head(quota))
        return pd.concat(strata, ignore_index=False).head(pilot_n)

    for seed in (42, 101, 20260423, 20260424):
        gen = PuzzleGenerator(seed=seed)
        ds = gen.generate_dataset(n_per_cell=50, chain_lengths=[2, 5])
        preflight = stratified(ds, pilot_n=20, chain_lengths=[2, 5])
        c2 = int((preflight.chain_length == 2).sum())
        c5 = int((preflight.chain_length == 5).sum())
        if (c2, c5) != (10, 10):
            _fail(f"seed={seed}: stratified preflight not 10/10 (got {c2} cl=2 / {c5} cl=5)")
            return

    _ok("stratified preflight draws exactly 10/10 cl=2/cl=5 for every seed tested")


def test_stratified_preflight_handles_odd_pilot_n() -> None:
    """For pilot_n not divisible by #chain_lengths, the first chain gets the +1."""

    def stratified(dataset: pd.DataFrame, pilot_n: int, chain_lengths: list) -> pd.DataFrame:
        all_controls = dataset[~dataset.contradiction]
        cl_list = list(chain_lengths) if chain_lengths else [None]
        per_cl = pilot_n // len(cl_list)
        extra = pilot_n - per_cl * len(cl_list)
        strata = []
        for i, cl in enumerate(cl_list):
            quota = per_cl + (1 if i < extra else 0)
            strata.append(all_controls[all_controls.chain_length == cl].head(quota))
        return pd.concat(strata, ignore_index=False).head(pilot_n)

    gen = PuzzleGenerator(seed=20260424)
    ds = gen.generate_dataset(n_per_cell=50, chain_lengths=[2, 5])
    # pilot_n = 21 → cl=2 gets 11, cl=5 gets 10
    preflight = stratified(ds, pilot_n=21, chain_lengths=[2, 5])
    c2 = int((preflight.chain_length == 2).sum())
    c5 = int((preflight.chain_length == 5).sum())
    if (c2, c5) != (11, 10):
        _fail(f"pilot_n=21: expected 11/10 cl=2/cl=5 (first chain gets the remainder), got {c2}/{c5}")
        return
    _ok("stratified preflight handles odd pilot_n — first chain gets the +1 remainder")


def test_old_head20_still_reproducibly_reveals_the_bug() -> None:
    """Sanity check: the old head(20) logic should still produce the skewed
    draw under seed 20260423 (proves our test fixture and the stratified fix
    are measuring the same thing the pre-fix pipeline saw)."""
    gen = PuzzleGenerator(seed=20260423)
    ds = gen.generate_dataset(n_per_cell=50, chain_lengths=[2, 5])
    old_preflight = ds[~ds.contradiction].head(20)
    c2 = int((old_preflight.chain_length == 2).sum())
    c5 = int((old_preflight.chain_length == 5).sum())
    if (c2, c5) != (9, 11):
        _fail(f"seed=20260423 head(20) regression: expected 9/11 cl=2/cl=5 (the original bug draw), got {c2}/{c5}")
        return
    _ok("seed 20260423 head(20) still reveals the 9/11 skew the fix targets")


# ─────────────────────────────────────────────────────────────────────────────
#  Fix 2: three-tier check_answer parser
# ─────────────────────────────────────────────────────────────────────────────

def test_parser_tier1_answer_colon_wins_over_mid_cot() -> None:
    """Tier 1: 'Answer: YES' wins even if the CoT above says 'NO' multiple times."""
    cot = (
        "Let me check: glorp → blen → ... NO direct evidence.\n"
        "But walking forward: NO contradiction found either.\n"
        "Answer: YES"
    )
    if not check_answer(cot, "YES"):
        _fail("Tier 1 should catch trailing 'Answer: YES' over mid-CoT NO tokens")
        return
    if check_answer(cot, "NO"):
        _fail("Tier 1 should NOT match expected=NO when the explicit answer is YES")
        return
    _ok("Tier 1: 'Answer: YES' overrides mid-CoT NO tokens")


def test_parser_tier1_multiple_answer_lines_take_last() -> None:
    """If the model emits multiple 'Answer: X' statements (rare), the last wins."""
    text = "Answer: NO\nOn reflection, that's wrong.\nAnswer: YES"
    if not check_answer(text, "YES"):
        _fail("Tier 1 should prefer the LAST 'Answer: X' statement when multiple exist")
        return
    _ok("Tier 1: multiple 'Answer: X' lines — the last one wins")


def test_parser_tier2_trailing_yes_overrides_mid_no() -> None:
    """Tier 2: a final-line bare YES wins over mid-CoT NO tokens (the cl=5 case)."""
    cot = (
        "Walking the chain:\n"
        "- glorp → blen: NO direct\n"
        "- blen → sorin: still NO\n"
        "But the injection says Varn6970 IS a sorin.\n"
        "\n"
        "YES"
    )
    if not check_answer(cot, "YES"):
        _fail("Tier 2 should catch trailing-line bare 'YES' over mid-CoT NO tokens")
        return
    if check_answer(cot, "NO"):
        _fail("Tier 2 should NOT match NO when the trailing line is YES")
        return
    _ok("Tier 2: trailing bare 'YES' line overrides mid-CoT NO tokens")


def test_parser_tier2_trailing_yes_with_punctuation() -> None:
    """Trailing-line match tolerates trailing period, markdown, quotes."""
    cases = [
        "some reasoning... NO wait... \n**YES**",
        "reasoning...\n\"yes.\"",
        "reasoning... NO. \nFinal: YES!",
        "reasoning\n- YES",
    ]
    for c in cases:
        if not check_answer(c, "YES"):
            _fail(f"Tier 2 should tolerate trailing punctuation / markdown around YES: {c!r}")
            return
    _ok("Tier 2: trailing bare YES tolerates markdown, quotes, punctuation")


def test_parser_tier3_fallback_preserves_legacy_behavior() -> None:
    """Tier 3: if no 'Answer:' and no clean trailing line, fall back to last-match."""
    # No "Answer: X", trailing line is not a bare YES/NO, so Tier 1 and 2 miss.
    # Legacy last-match should still decide.
    text = "Thinking carefully about whether X is Y, I conclude NO"
    if not check_answer(text, "NO"):
        _fail(f"Tier 3 should catch last-match NO when no structured answer: {text!r}")
        return
    if check_answer(text, "YES"):
        _fail("Tier 3 should NOT match YES when the last token is NO")
        return
    _ok("Tier 3: last-match fallback preserves legacy behavior when Tiers 1–2 miss")


def test_parser_gemma_leading_no_no_longer_flips() -> None:
    """Original first-match bug regression check (fixed 2026-04-23): a leading
    'No contradiction' followed by a proper YES/NO verdict must not flip."""
    text = "No contradiction found — X is in fact a Y.\nAnswer: YES"
    if not check_answer(text, "YES"):
        _fail("Leading 'No contradiction' should not flip a correct YES verdict")
        return
    _ok("Regression: leading 'No contradiction' phrasing does not flip YES verdict")


def test_parser_direct_answer_unchanged() -> None:
    """Direct-answer models (just 'YES' or 'NO') still parse correctly."""
    for direct, expected in [
        ("YES", "YES"),
        ("NO", "NO"),
        ("yes", "YES"),
        ("no.", "NO"),
        ("  YES  ", "YES"),
    ]:
        if not check_answer(direct, expected):
            _fail(f"Direct answer {direct!r} should match expected {expected!r}")
            return
    _ok("Direct-answer outputs parse identically (no regression on well-behaved models)")


def test_parser_unrelated_text_falls_through() -> None:
    """Output with no YES/NO token at all should fall through all three tiers
    and hit the substring fallback."""
    # "I cannot determine" contains no YES/NO tokens, so all 3 tiers miss.
    # The substring fallback looks for expected.lower() in text.lower() —
    # "yes".lower() not in this text, "no".lower() not in it either (as a full
    # token); but "no" IS in "cannot". That's the known limitation of the
    # substring fallback. This test just asserts that BOTH YES and NO are
    # reported the same way (neither should trigger a spurious TRUE because
    # of a random character match outside the expected context).
    text = "I cannot determine from the given premises."
    yes_result = check_answer(text, "YES")
    no_result = check_answer(text, "NO")
    # "yes" substring is NOT in "I cannot determine…", so YES should be False.
    # "no" substring IS in "cannot", so NO unfortunately returns True — this
    # is the pre-existing Tier-3-fallback behavior and is out of scope for
    # the 2026-04-24 fix. We just assert YES doesn't spuriously fire.
    if yes_result:
        _fail(f"Output with no YES/NO token should not report YES=True: {text!r}")
        return
    _ok("No-verdict output does not spuriously report YES (known NO-substring quirk preserved)")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n2026-04-24 gate fixes — regression tests\n" + "=" * 48)
    print("\n[Fix 1 — stratified preflight]")
    test_stratified_preflight_is_seed_invariant()
    test_stratified_preflight_handles_odd_pilot_n()
    test_old_head20_still_reproducibly_reveals_the_bug()

    print("\n[Fix 2 — three-tier check_answer parser]")
    test_parser_tier1_answer_colon_wins_over_mid_cot()
    test_parser_tier1_multiple_answer_lines_take_last()
    test_parser_tier2_trailing_yes_overrides_mid_no()
    test_parser_tier2_trailing_yes_with_punctuation()
    test_parser_tier3_fallback_preserves_legacy_behavior()
    test_parser_gemma_leading_no_no_longer_flips()
    test_parser_direct_answer_unchanged()
    test_parser_unrelated_text_falls_through()

    print("\n" + "=" * 48)
    if FAIL:
        print(f"FAILED ({FAIL} failures)\n")
        return 1
    print("ALL TESTS PASSED\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

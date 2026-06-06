#!/usr/bin/env python3
"""Identity stub — residual-stream sub-layer friction (attention vs MLP).

Candidate: "residual-stream sub-layer friction" — the hypothesis that the
hallucination tell lives in the FRICTION between a transformer block's two
writes to the residual stream (the attention contribution `a`, post-W_o, and
the MLP contribution `m`), not in either alone. The block update is
`dh = a + m`, so any metric that reads only `dh` is structurally blind to the
friction. This file pins the metric contract BEFORE the production code exists.

Numpy-only; no MLX, no model. Defines the reference friction metrics inline
(`residual_friction`) and asserts the identities the eventual production method
(`PRIComputer.residual_friction`, not yet implemented) must satisfy. When that
method lands, swap the `residual_friction` calls below for the production call
and the same checks become the acceptance test.

Checks:
  1. HEADLINE — same-sum-different-friction: two (a, m) pairs with IDENTICAL
     `a + m` (so identical dh, identical to v3) but opposite friction. Proves
     the signal is invisible in the sum.
  2. cos(a, m) ∈ [-1, 1]; parallel → +1, antiparallel → -1, orthogonal → 0.
  3. interference = 1 - ‖a+m‖/(‖a‖+‖m‖) ∈ [0, 1]; aligned → 0, full cancel → 1,
     orthogonal equal-norm → 1 - √2/2.
  4. veto = -(a·m)/‖a‖ sign: m opposes a → > 0 (knowledge vetoes the route);
     m aligns → < 0; orthogonal → 0.
  5. magnitude confound — cos is invariant under independent positive scaling;
     interference is invariant only under COMMON scaling, not independent
     (documents the "report both" caveat).
  6. cos and interference are symmetric in (a, m); veto is intentionally NOT
     (a is the designated "router").
  7. degenerate guards — zero `a` and/or zero `m` (the BOS-sink extreme) yield
     finite, sane values, never NaN.
  8. schema — `residual_friction` returns {cos, interference, veto}, all finite.
  9. ISOLATION — a benign refinement and a destructive veto can share IDENTICAL
     raw friction (cos, interference, veto). Only a direction-aware veto onto the
     consequential axis `u` (production: the top-Fisher stream direction) tells
     them apart. Proves raw friction is insufficient → the isolation baseline.

Run with the venv python:
    .venv/bin/python scripts/test_residual_friction.py
Exit 0 on pass, 1 on fail.

Cross-references:
  - future method: pri_v2_mlx_pipeline.py  PRIComputer.residual_friction (TODO)
  - future capture: per-block (a, m) sub-layer-output hook (TODO)
  - template:      scripts/test_centered_fisher.py
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Reference implementation — the contract the production method must match
# ─────────────────────────────────────────────────────────────────────────────


def residual_friction(a: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> dict:
    """Friction between two residual-stream writes `a` (attention) and `m` (MLP).

    All three metrics are W_u-free (they touch only the residual vectors):
      cos          — magnitude-normalized alignment, ∈ [-1, 1].
      interference — destructive-interference fraction, ∈ [0, 1]; how much of
                     the two writes' length cancels in the sum.
      veto         — signed magnitude of m's component along -a; positive when
                     the MLP pushes against what attention routed in.

    Degenerate inputs (zero `a` or `m`) return 0.0 for the affected metric
    rather than NaN — the sink-saturated extreme must not produce non-finite
    values.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    m = np.asarray(m, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nm = float(np.linalg.norm(m))
    nsum = float(np.linalg.norm(a + m))
    dot = float(a @ m)

    cos = dot / (na * nm) if na > eps and nm > eps else 0.0
    interference = 1.0 - nsum / (na + nm) if (na + nm) > eps else 0.0
    veto = -dot / na if na > eps else 0.0
    return {"cos": cos, "interference": interference, "veto": veto}


def directed_veto(a: np.ndarray, m: np.ndarray, u: np.ndarray, eps: float = 1e-12) -> float:
    """Signed cancellation of attention's push along a CONSEQUENTIAL direction `u`.

    > 0  ⇒  m vetoes a on `u` (destructive on the direction that matters).
    < 0  ⇒  m endorses a on `u` (constructive — pushes the same way).

    Raw friction (cos / interference) averages over all directions and is blind
    to this. In production, `u` is the top-Fisher / top-SVD stream direction
    (W_u-free proxy for "consequential"), or the answer direction (W_u-dependent,
    sharper). This is the seed of the isolation baseline.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    m = np.asarray(m, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    nu = float(np.linalg.norm(u))
    if nu <= eps:
        return 0.0
    uhat = u / nu
    return -float(a @ uhat) * float(m @ uhat)


# ─────────────────────────────────────────────────────────────────────────────
#  Checks
# ─────────────────────────────────────────────────────────────────────────────


def test_same_sum_different_friction() -> None:
    """The crown jewel. Two pairs whose writes sum to the SAME dh = [1, 0, 0, 0]
    but whose friction is opposite. v3 sees only `a + m`, so it cannot tell them
    apart; friction can."""
    # Cooperating: a and m point the same way.
    a1 = np.array([0.5, 0.0, 0.0, 0.0])
    m1 = np.array([0.5, 0.0, 0.0, 0.0])
    # Fighting: a and m nearly cancel.
    a2 = np.array([5.0, 0.0, 0.0, 0.0])
    m2 = np.array([-4.0, 0.0, 0.0, 0.0])

    # What v3 sees is identical.
    assert np.allclose(a1 + m1, a2 + m2), "(1) dh differs — example is broken"

    f1 = residual_friction(a1, m1)
    f2 = residual_friction(a2, m2)

    assert np.isclose(f1["cos"], +1.0) and np.isclose(f2["cos"], -1.0), (
        f"(1) cos: cooperating={f1['cos']:.4f} (want +1), fighting={f2['cos']:.4f} (want -1)"
    )
    assert f1["interference"] < 1e-9, f"(1) cooperating interference {f1['interference']} (want 0)"
    assert f2["interference"] > 0.8, f"(1) fighting interference {f2['interference']:.4f} (want ≫ 0)"
    print(
        f"(1) PASS  same dh={(a1 + m1).tolist()}, different friction "
        f"(cos {f1['cos']:+.2f} vs {f2['cos']:+.2f}; "
        f"interf {f1['interference']:.3f} vs {f2['interference']:.3f})"
    )


def test_cos_known_values_and_range() -> None:
    assert np.isclose(residual_friction([1, 2, 3], [2, 4, 6])["cos"], 1.0), "(2) parallel"
    assert np.isclose(residual_friction([1, 0], [-3, 0])["cos"], -1.0), "(2) antiparallel"
    assert np.isclose(residual_friction([1, 0], [0, 5])["cos"], 0.0), "(2) orthogonal"

    rng = np.random.default_rng(20260605)
    for _ in range(200):
        a = rng.standard_normal(16)
        m = rng.standard_normal(16)
        c = residual_friction(a, m)["cos"]
        assert -1.0 - 1e-9 <= c <= 1.0 + 1e-9, f"(2) cos out of range: {c}"
    print("(2) PASS  cos ∈ [-1, 1]; ∥→+1, anti→-1, ⊥→0")


def test_interference_range_and_endpoints() -> None:
    assert np.isclose(residual_friction([2, 0], [3, 0])["interference"], 0.0), "(3) aligned → 0"
    assert np.isclose(residual_friction([2, 0], [-2, 0])["interference"], 1.0), "(3) cancel → 1"
    orth = residual_friction([1, 0], [0, 1])["interference"]
    assert np.isclose(orth, 1.0 - np.sqrt(2) / 2), f"(3) orthogonal equal-norm {orth:.4f}"

    rng = np.random.default_rng(0xF00D)
    for _ in range(200):
        a = rng.standard_normal(16)
        m = rng.standard_normal(16)
        x = residual_friction(a, m)["interference"]
        assert -1e-9 <= x <= 1.0 + 1e-9, f"(3) interference out of range: {x}"
    print(f"(3) PASS  interference ∈ [0, 1]; aligned→0, cancel→1, ⊥→{orth:.3f}")


def test_veto_sign() -> None:
    assert residual_friction([1, 0], [-2, 0])["veto"] > 0, "(4) opposing m should veto (>0)"
    assert residual_friction([1, 0], [2, 0])["veto"] < 0, "(4) aligned m should endorse (<0)"
    assert np.isclose(residual_friction([1, 0], [0, 3])["veto"], 0.0), "(4) orthogonal → 0"
    print("(4) PASS  veto > 0 when MLP opposes the route, < 0 when it endorses")


def test_magnitude_confound() -> None:
    """cos normalizes away magnitude; interference does not (under independent
    scaling). This is the 'report both' caveat made executable."""
    rng = np.random.default_rng(7)
    a = rng.standard_normal(8)
    m = rng.standard_normal(8)

    # cos: invariant under independent positive scaling.
    assert np.isclose(
        residual_friction(a, m)["cos"], residual_friction(3.0 * a, 0.5 * m)["cos"]
    ), "(5) cos not scale-invariant"

    # interference: invariant under COMMON scaling.
    assert np.isclose(
        residual_friction(a, m)["interference"],
        residual_friction(4.0 * a, 4.0 * m)["interference"],
    ), "(5) interference not invariant under common scale"

    # interference: NOT invariant under independent scaling (orthogonal case).
    base = residual_friction([1, 0], [0, 1])["interference"]
    skew = residual_friction([2, 0], [0, 1])["interference"]
    assert not np.isclose(base, skew), "(5) interference unexpectedly scale-free"
    print(
        f"(5) PASS  cos magnitude-free; interference common-scale-free but "
        f"independent-scale-sensitive ({base:.3f} → {skew:.3f})"
    )


def test_symmetry() -> None:
    rng = np.random.default_rng(99)
    for _ in range(100):
        a = rng.standard_normal(12)
        m = rng.standard_normal(12)
        fab = residual_friction(a, m)
        fba = residual_friction(m, a)
        assert np.isclose(fab["cos"], fba["cos"]), "(6) cos not symmetric"
        assert np.isclose(fab["interference"], fba["interference"]), "(6) interference not symmetric"
    print("(6) PASS  cos & interference symmetric in (a, m); veto kept asymmetric by design")


def test_degenerate_guards() -> None:
    """The BOS-sink extreme: a (or m) collapses to ~0. Must stay finite."""
    for a, m in (
        ([0.0, 0.0], [1.0, 2.0]),
        ([1.0, 2.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ):
        out = residual_friction(a, m)
        for k, v in out.items():
            assert np.isfinite(v), f"(7) non-finite {k}={v} for a={a}, m={m}"
    print("(7) PASS  zero-vector inputs (sink extreme) stay finite, never NaN")


def test_schema() -> None:
    rng = np.random.default_rng(123)
    out = residual_friction(rng.standard_normal(32), rng.standard_normal(32))
    assert set(out.keys()) == {"cos", "interference", "veto"}, f"(8) keys {set(out.keys())}"
    assert all(isinstance(v, float) and np.isfinite(v) for v in out.values()), "(8) values"
    print("(8) PASS  schema = {cos, interference, veto}, all finite floats")


def test_direction_separates_benign_from_destructive() -> None:
    """The isolation hurdle, made executable. A benign refinement and a
    destructive veto are built to have IDENTICAL raw friction (same ‖a‖, ‖m‖,
    a·m ⇒ same cos, interference, veto). Only the direction-aware `directed_veto`
    onto the consequential axis `u` separates them — benign endorses (< 0),
    destructive vetoes (> 0)."""
    a = np.array([1.0, 1.0, 0.0, 0.0])    # attention routes; e0 = consequential axis
    m_b = np.array([0.5, -1.0, 0.0, 0.0])  # benign: trims in bulk, AGREES on e0
    m_d = np.array([-0.5, 0.0, 1.0, 0.0])  # destructive: VETOES on e0, trims in bulk
    u = np.array([1.0, 0.0, 0.0, 0.0])     # consequential dir (prod: top-Fisher dir)

    fb = residual_friction(a, m_b)
    fd = residual_friction(a, m_d)

    # Raw friction is identical by construction — it cannot separate them.
    for k in ("cos", "interference", "veto"):
        assert np.isclose(fb[k], fd[k]), (
            f"(9) raw {k} differs ({fb[k]:.6f} vs {fd[k]:.6f}) — example is broken"
        )

    # The direction-aware metric flips sign: benign endorses, destructive vetoes.
    vb = directed_veto(a, m_b, u)
    vd = directed_veto(a, m_d, u)
    assert vb < 0 < vd, f"(9) directed_veto failed to separate: benign={vb:.3f}, destructive={vd:.3f}"
    print(
        f"(9) PASS  raw friction identical (cos {fb['cos']:.3f}, interf "
        f"{fb['interference']:.3f}, veto {fb['veto']:.3f}) but directed_veto "
        f"separates ({vb:+.2f} benign vs {vd:+.2f} destructive)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    tests = [
        test_same_sum_different_friction,
        test_cos_known_values_and_range,
        test_interference_range_and_endpoints,
        test_veto_sign,
        test_magnitude_confound,
        test_symmetry,
        test_degenerate_guards,
        test_schema,
        test_direction_separates_benign_from_destructive,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{failures} / {len(tests)} FAIL")
        return 1
    print(f"ALL {len(tests)} PASS")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

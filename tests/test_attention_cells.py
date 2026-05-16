"""Tests for the Attention cell family (v4-candidate #5, landed 2026-05-15).

Fast unit tests cover the helpers (ATTENTION_PANEL shape, label-split, column-name
and cell-label dispatching, _compute_attention_score on synthetic captures).

Slow integration test (@pytest.mark.slow) runs an end-to-end calibration on
Gemma-3-1B with --attention-only panel + the existing puzzle_calibration_jsonl
fixture and verifies that (a) the calibrator can pick an Attention winner,
(b) the detector reloads the profile and self-tests within numerical
tolerance (|reported - deployed AUROC| < 1e-3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pri_calibrator import (
    ATTENTION_FAMILY,
    ATTENTION_LAYERS,
    ATTENTION_METRICS,
    ATTENTION_PANEL,
    DEFAULT_PANEL,
    _build_derivation,
    _cell_label,
    _column_name,
    _compute_attention_score,
    _is_attention_cell,
    _requires_attention_capture,
    _split_attention_label,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fast unit tests — no model load
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionPanelShape:
    """The ATTENTION_PANEL constant is exactly 12 cells, gen_step=1, all unique."""

    def test_panel_has_twelve_cells(self):
        assert len(ATTENTION_PANEL) == 12

    def test_all_cells_are_attention_family(self):
        assert all(cell[1] == ATTENTION_FAMILY for cell in ATTENTION_PANEL)

    def test_all_cells_at_gen_step_1(self):
        assert all(cell[0] == 1 for cell in ATTENTION_PANEL)

    def test_labels_are_unique(self):
        labels = [cell[2] for cell in ATTENTION_PANEL]
        assert len(set(labels)) == len(labels)

    def test_panel_covers_full_layer_metric_cross_product(self):
        expected = {f"{layer}_{metric}" for layer in ATTENTION_LAYERS for metric in ATTENTION_METRICS}
        actual = {cell[2] for cell in ATTENTION_PANEL}
        assert expected == actual


class TestAttentionPredicates:
    def test_is_attention_cell_positive(self):
        for cell in ATTENTION_PANEL:
            assert _is_attention_cell(cell)

    def test_is_attention_cell_negative(self):
        for cell in DEFAULT_PANEL:
            assert not _is_attention_cell(cell)

    def test_requires_attention_capture_true_when_present(self):
        panel = list(DEFAULT_PANEL) + [ATTENTION_PANEL[0]]
        assert _requires_attention_capture(panel)

    def test_requires_attention_capture_false_for_default_panel(self):
        assert not _requires_attention_capture(DEFAULT_PANEL)

    def test_requires_attention_capture_false_for_empty(self):
        assert not _requires_attention_capture([])


class TestSplitAttentionLabel:
    def test_all_panel_labels_split_cleanly(self):
        for cell in ATTENTION_PANEL:
            layer_metric = cell[2]
            parsed = _split_attention_label(layer_metric)
            assert parsed is not None
            layer, metric = parsed
            assert layer in ATTENTION_LAYERS
            assert metric in ATTENTION_METRICS
            assert f"{layer}_{metric}" == layer_metric

    def test_invalid_label_returns_none(self):
        assert _split_attention_label("garbage") is None
        assert _split_attention_label("final_unknown_metric") is None
        assert _split_attention_label("nonexistent_layer_js") is None
        assert _split_attention_label("") is None

    def test_metric_underscores_handled(self):
        # `js_kv_groups`, `js_no_bos`, `bos_mass` all contain underscores
        # and the splitter must anchor on layer prefixes, not naive rsplit.
        for layer in ATTENTION_LAYERS:
            for metric in ("js_kv_groups", "js_no_bos", "bos_mass"):
                parsed = _split_attention_label(f"{layer}_{metric}")
                assert parsed == (layer, metric)


class TestColumnNameAndCellLabel:
    def test_column_name_for_attention(self):
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        assert _column_name(cell) == "attention::final_js_kv_groups"

    def test_cell_label_for_attention(self):
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        assert _cell_label(cell) == "attention[final_js_kv_groups] @ step 1"

    def test_attention_doesnt_break_existing_dispatcher(self):
        # Existing cell types should keep their old labels/columns.
        scalar_cell = (1, "scalar", "d_F_full")
        assert _column_name(scalar_cell) == "d_F_full"
        assert _cell_label(scalar_cell) == "d_F_full @ step 1"
        composite_cell = (1, "Composite", "additive_S_fisher_r=1")
        assert _column_name(composite_cell) == "composite::additive_S_fisher_r=1"


class TestBuildDerivation:
    def test_attention_winners_have_no_derivation(self):
        # The detector reads attention scores from captures directly, not
        # via the derivation payload. _build_derivation must return None so
        # the schema doesn't carry a stale composite/residualized payload.
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        assert _build_derivation(cell, resid_coeffs={}) is None


class TestComputeAttentionScore:
    """Synthetic-captures unit tests. No model load."""

    @staticmethod
    def _synthetic_captures(weights_by_layer):
        """Build a captures dict in the shape attention_capture would produce.

        Each value is a list of attention-weight arrays (one per forward
        call); the calibrator reads index 1 (first gen forward). We pad
        index 0 with a placeholder to mirror real-world layout (prefix
        forward at idx 0, first gen forward at idx 1).
        """
        return {
            layer: [np.zeros((1, 1), dtype=np.float64), w]
            for layer, w in weights_by_layer.items()
        }

    def test_js_metric_on_uniform_heads(self):
        # All heads identical → js_radius == 0 (no head disagreement).
        w = np.ones((4, 8), dtype=np.float64) / 8
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js")
        score = _compute_attention_score(cell, captures, {"final": 4})
        assert score is not None
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_js_metric_on_divergent_heads(self):
        # Two heads concentrate on opposite positions → js_radius > 0.
        w = np.zeros((2, 4), dtype=np.float64)
        w[0, 0] = 1.0
        w[1, 3] = 1.0
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js")
        score = _compute_attention_score(cell, captures, {"final": 2})
        assert score is not None
        assert score > 0.0

    def test_bos_mass_metric(self):
        # Half the heads on BOS, half spread; mean BOS mass = 0.5.
        w = np.zeros((4, 4), dtype=np.float64)
        w[0, 0] = 1.0  # full BOS
        w[1, 0] = 1.0  # full BOS
        w[2] = 0.25    # uniform → 0.25 BOS
        w[3] = 0.25
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_bos_mass")
        score = _compute_attention_score(cell, captures, {"final": 4})
        assert score is not None
        assert score == pytest.approx(0.625, abs=1e-6)  # (1 + 1 + 0.25 + 0.25)/4

    def test_js_no_bos_drops_bos_column(self):
        # BOS column dominates; after dropping it + renorming, heads agree.
        w = np.zeros((2, 3), dtype=np.float64)
        w[0] = [0.99, 0.005, 0.005]
        w[1] = [0.99, 0.005, 0.005]
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js_no_bos")
        score = _compute_attention_score(cell, captures, {"final": 2})
        assert score is not None
        assert score == pytest.approx(0.0, abs=1e-6)  # identical post-drop

    def test_js_kv_groups_requires_n_kv_heads(self):
        w = np.ones((4, 8), dtype=np.float64) / 8
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        assert _compute_attention_score(cell, captures, {}) is None
        assert _compute_attention_score(cell, captures, {"final": 2}) is not None

    def test_js_kv_groups_collapses_repeats(self):
        # 4 q-heads sharing 2 kv-groups; pairs are identical pre-collapse.
        # Post-collapse: still 2 distinct distributions → js_radius matches
        # the un-collapsed value (because grouping doesn't change the
        # distinct positions). Sanity check: doesn't return NaN/None.
        w = np.zeros((4, 4), dtype=np.float64)
        w[0] = [1, 0, 0, 0]
        w[1] = [1, 0, 0, 0]
        w[2] = [0, 0, 0, 1]
        w[3] = [0, 0, 0, 1]
        captures = self._synthetic_captures({"final": w})
        cell = (1, ATTENTION_FAMILY, "final_js_kv_groups")
        score = _compute_attention_score(cell, captures, {"final": 2})
        assert score is not None
        assert score > 0.0  # heads disagree post-grouping

    def test_missing_captures_returns_none(self):
        cell = (1, ATTENTION_FAMILY, "final_js")
        assert _compute_attention_score(cell, {}, {}) is None

    def test_too_few_capture_entries_returns_none(self):
        # captures[tag] must have ≥ 2 entries (prefix + first gen forward).
        captures = {"final": [np.zeros((1, 1))]}
        cell = (1, ATTENTION_FAMILY, "final_js")
        assert _compute_attention_score(cell, captures, {"final": 1}) is None

    def test_wrong_step_returns_none(self):
        w = np.ones((2, 4), dtype=np.float64) / 4
        captures = self._synthetic_captures({"final": w})
        cell = (3, ATTENTION_FAMILY, "final_js")  # step 3, not 1
        assert _compute_attention_score(cell, captures, {"final": 2}) is None

    def test_non_attention_family_returns_none(self):
        captures = self._synthetic_captures({"final": np.ones((2, 4)) / 4})
        # Calling with a non-Attention cell is defensive — should refuse.
        cell = (1, "scalar", "d_F_full")
        assert _compute_attention_score(cell, captures, {"final": 2}) is None


# ─────────────────────────────────────────────────────────────────────────────
# Slow integration test — Gemma-3-1B end-to-end
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestAttentionEndToEnd:
    """End-to-end: calibrate Gemma-3-1B with the attention panel, then
    reload the profile in a Detector and confirm byte-exact self-test
    parity. Mirrors test_pri_detector.py::TestRuntime."""

    def test_calibrate_then_self_test(
        self, gemma_3_1b_slug, puzzle_calibration_jsonl, tmp_path
    ):
        from pri_calibrator import calibrate
        from pri_detector import Detector, _self_test

        # Use the attention panel only to keep the smoke focused.
        profile = calibrate(
            model_slug=gemma_3_1b_slug,
            calibration_jsonl_path=str(puzzle_calibration_jsonl),
            task_label="attention_e2e_smoke",
            panel=list(ATTENTION_PANEL),
            n_bootstrap=50,
            max_new_tokens=4,
        )

        # Profile should record the attention wrapper hash since the panel
        # contained Attention cells.
        assert profile.provenance.get("attention_wrapper_module_hash_sha256") is not None

        # Winner should be an Attention cell.
        assert profile.detector["metric"]["family"] == ATTENTION_FAMILY

        # Persist + reload + self-test.
        profile_path = tmp_path / "gemma_3_1b_attention.profile.json"
        profile.to_json(str(profile_path))
        rc = _self_test(str(profile_path), str(puzzle_calibration_jsonl))
        assert rc == 0, "detector self-test failed for attention winner profile"

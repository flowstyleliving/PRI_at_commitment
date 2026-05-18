from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.step0_belief_readout as belief
from scripts.sweep_locking import hold_out_dir_lock


class _FakeTokenizer:
    def __init__(self, mapping: dict[int, str]):
        self.mapping = mapping

    def decode(self, token_ids):
        if len(token_ids) != 1:
            raise ValueError("fake tokenizer only supports single-token decode")
        return self.mapping[int(token_ids[0])]


def test_literal_yes_no_buckets_are_literal_only():
    tok = _FakeTokenizer({
        0: " yes",
        1: " NO",
        2: "Yes.",
        3: "correct",
        4: "▁yes",
        5: "ĠNo",
    })
    buckets = belief.literal_yes_no_token_buckets(tok, 6)
    assert buckets["yes_token_ids"] == [0, 4]
    assert buckets["no_token_ids"] == [1, 5]
    assert 2 not in buckets["yes_token_ids"]
    assert 3 not in buckets["yes_token_ids"]


def test_semantic_and_control_bundles_follow_frozen_rules():
    tok = _FakeTokenizer({
        0: " correct",
        1: " false",
        2: "\n",
        3: "<|assistant|>",
        4: " To",
        5: " yes",
        6: " sure",
        7: " wrong",
    })
    semantic = belief.semantic_shortlist_token_buckets(tok, 8)
    control = belief.control_marker_token_bundle(tok, 8)
    assert semantic["semantic_yes_token_ids"] == [0, 5]
    assert semantic["semantic_no_token_ids"] == [1, 7]
    assert control["control_token_ids"] == [2, 3]
    assert 4 not in control["control_token_ids"]
    assert 6 not in semantic["semantic_yes_token_ids"]


def test_score_row_uses_control_floor_and_semantic_mass():
    probs = np.array([0.12, 0.08, 0.15, 0.01, 0.64], dtype=np.float64)
    row = belief._score_row_from_probs(  # noqa: SLF001
        probs,
        yes_token_ids=[0],
        no_token_ids=[1],
        semantic_yes_token_ids=[0, 2],
        semantic_no_token_ids=[1],
        control_token_ids=[3],
    )
    assert row["decidedness"] == 0.20
    assert row["semantic_decidedness"] == pytest.approx(0.35)
    assert row["semantic_offliteral_mass"] == pytest.approx(0.15)
    assert row["decidedness_floor"] == pytest.approx(belief.CONTROL_FLOOR_MULTIPLIER * 0.01)
    assert row["above_floor"] is True


def test_coverage_curve_filters_rows_below_floor():
    labels = np.array([0, 1, 0], dtype=np.int32)
    lean = np.array([2.0, -2.0, 0.1], dtype=np.float64)
    decidedness = np.array([0.9, 0.8, 1e-9], dtype=np.float64)
    eligible = np.array([True, True, False], dtype=bool)
    curve = belief.build_coverage_curve(
        labels=labels,
        lean_scores=lean,
        decidedness=decidedness,
        eligible_mask=eligible,
        n_bootstrap=20,
        seed=123,
    )
    assert len(curve) == 1
    assert curve[0]["coverage"] == 2 / 3
    assert curve[0]["auroc_b_signed"] > 0.99


def test_classify_verdict_distinguishes_low_decidedness_and_undetermined():
    empty_curve: list[dict[str, object]] = []
    assert belief.classify_verdict(
        empty_curve,
        eligible_coverage=0.40,
        undetermined_coverage=0.85,
    ) == "Undetermined-for-M"
    assert belief.classify_verdict(
        empty_curve,
        eligible_coverage=0.40,
        undetermined_coverage=0.10,
    ) == "Low-decidedness-for-M"
    assert belief.classify_verdict(
        empty_curve,
        eligible_coverage=0.90,
        undetermined_coverage=0.10,
    ) == "Decided-but-non-B-for-M"


def test_validate_panel_specs_requires_identical_hash(tmp_path):
    spec_a = tmp_path / "a.json"
    spec_b = tmp_path / "b.json"
    spec_a.write_text(json.dumps({"model_slug": "a", "data_hash_sha256": "hash-a"}))
    spec_b.write_text(json.dumps({"model_slug": "b", "data_hash_sha256": "hash-b"}))
    try:
        belief.validate_panel_specs(
            spec_paths=[spec_a, spec_b],
            expected_data_hash_sha256="hash-a",
        )
    except RuntimeError as exc:
        assert "spec data hashes do not match" in str(exc)
    else:
        raise AssertionError("expected validate_panel_specs to reject mismatched hashes")


def test_validate_panel_specs_requires_locked_model_panel(tmp_path):
    spec_a = tmp_path / "a.json"
    spec_b = tmp_path / "b.json"
    spec_a.write_text(json.dumps({"model_slug": belief.LOCKED_MODEL_PANEL[0], "data_hash_sha256": "hash"}))
    spec_b.write_text(json.dumps({"model_slug": "mlx-community/Not-Locked-4bit", "data_hash_sha256": "hash"}))
    with pytest.raises(RuntimeError, match="locked 10-model panel mismatch"):
        belief.validate_panel_specs(
            spec_paths=[spec_a, spec_b],
            expected_data_hash_sha256="hash",
            expected_models=[belief.LOCKED_MODEL_PANEL[0], "mlx-community/Not-Locked-4bit"],
        )


def test_validate_spec_redecodes_literal_token_ids_and_prompt_identity():
    tok = _FakeTokenizer({0: " yes", 1: " no"})
    prompt_identity = {
        "prompt_strategy_name": "raw_passthrough",
        "prompt_strategy_source_sha256": "source-hash",
        "prompt_probe_input_sha256": "probe-in",
        "prompt_probe_output_sha256": "probe-out",
    }
    spec = {
        "schema_version": "belief_spec_v1",
        "scoring_mode": belief.SCORING_MODE,
        "model_slug": "test/model",
        "data_hash_sha256": "hash",
        "prompt_strategy_name": "raw_passthrough",
        "prompt_strategy_source_sha256": "source-hash",
        "prompt_probe_input_sha256": "probe-in",
        "prompt_probe_output_sha256": "probe-out",
        "tokenizer_fix_flags": {},
        "control_floor_multiplier": belief.CONTROL_FLOOR_MULTIPLIER,
        "yes_token_ids": [0],
        "no_token_ids": [1],
        "semantic_yes_token_ids": [],
        "semantic_no_token_ids": [],
        "control_token_ids": [],
    }
    validated = belief._validate_spec(  # noqa: SLF001
        spec,
        model_slug="test/model",
        data_hash_sha256="hash",
        prompt_identity=prompt_identity,
        tokenizer_fix_flags={},
        tokenizer=tok,
    )
    assert validated["yes_tokens_live"] == [" yes"]
    assert validated["no_tokens_live"] == [" no"]

    bad_tok = _FakeTokenizer({0: " maybe", 1: " no"})
    with pytest.raises(RuntimeError, match="re-decodes"):
        belief._validate_spec(  # noqa: SLF001
            spec,
            model_slug="test/model",
            data_hash_sha256="hash",
            prompt_identity=prompt_identity,
            tokenizer_fix_flags={},
            tokenizer=bad_tok,
        )

    with pytest.raises(RuntimeError, match="prompt identity mismatch"):
        belief._validate_spec(  # noqa: SLF001
            spec,
            model_slug="test/model",
            data_hash_sha256="hash",
            prompt_identity={**prompt_identity, "prompt_probe_output_sha256": "different"},
            tokenizer_fix_flags={},
            tokenizer=tok,
        )


def test_finite_guards_fail_loudly():
    with pytest.raises(RuntimeError, match="non-finite values in last_probs"):
        belief._require_finite_array(  # noqa: SLF001
            "last_probs",
            np.array([0.2, np.nan, 0.8], dtype=np.float64),
            model_slug="test/model",
            sample_idx=4,
        )
    with pytest.raises(RuntimeError, match="non-finite lean"):
        belief._require_finite_scalars(  # noqa: SLF001
            {"lean": float("nan")},
            model_slug="test/model",
            sample_idx=4,
        )


def test_runner_missing_prereg_fails_before_model_execution(tmp_path, tmp_calibration_jsonl):
    data_path = tmp_calibration_jsonl([("prompt", 0), ("prompt2", 1)], filename="data.jsonl")
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    proc = subprocess.run(
        [
            "bash",
            str(Path("scripts/run_step0_belief_panel.sh")),
            "canary",
            str(data_path),
            str(out_dir),
            str(tmp_path / "missing-prereg.md"),
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "prereg doc not found" in (proc.stderr + proc.stdout)


def test_runner_lock_fails_closed(tmp_path, tmp_calibration_jsonl):
    data_path = tmp_calibration_jsonl([("prompt", 0), ("prompt2", 1)], filename="data.jsonl")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("locked prereg\n")
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    with hold_out_dir_lock(out_dir):
        proc = subprocess.run(
            [
                "bash",
                str(Path("scripts/run_step0_belief_panel.sh")),
                "canary",
                str(data_path),
                str(out_dir),
                str(prereg),
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode != 0
    assert "already holds" in (proc.stderr + proc.stdout)


def test_runner_rejects_model_override_env(tmp_path, tmp_calibration_jsonl):
    data_path = tmp_calibration_jsonl([("prompt", 0), ("prompt2", 1)], filename="data.jsonl")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("locked prereg\n")
    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    env["STEP0_BELIEF_MODELS"] = "mlx-community/Fake-4bit"
    proc = subprocess.run(
        [
            "bash",
            str(Path("scripts/run_step0_belief_panel.sh")),
            "canary",
            str(data_path),
            str(out_dir),
            str(prereg),
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "override is disabled" in (proc.stderr + proc.stdout)

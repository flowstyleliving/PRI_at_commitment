from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.diag_commit_step as diag_commit


PINNED_DATA = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "anli-sweep"
    / "2026-05-15"
    / "run-02"
    / "anli_R1_seed20260513_n100.jsonl"
)
PINNED_HASH = "94825f3d2029c0049f2a087b0093117edc576ada84f2a073b4eccdbf8e3fe3d5"


def _load_calibration_jsonl_fast(path: str):
    source = (Path(__file__).resolve().parent.parent / "pri_calibrator.py").read_text()
    module_ast = ast.parse(source, filename="pri_calibrator.py")
    fn_node = next(
        node for node in module_ast.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_calibration_jsonl"
    )
    mini_module = ast.Module(body=[fn_node], type_ignores=[])
    ast.fix_missing_locations(mini_module)
    ns = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "List": list,
        "Tuple": tuple,
    }
    exec(compile(mini_module, "pri_calibrator.py", "exec"), ns)  # noqa: S102
    return ns["_load_calibration_jsonl"](path)


def _load_locked_panel_helpers():
    source = (Path(__file__).resolve().parent.parent / "scripts" / "step0_belief_readout.py").read_text()
    module_ast = ast.parse(source, filename="step0_belief_readout.py")
    selected = []
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "LOCKED_MODEL_PANEL":
                    selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "validate_locked_model_panel":
            selected.append(node)
    mini_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mini_module)
    ns = {"Sequence": tuple}
    exec(compile(mini_module, "step0_belief_readout.py", "exec"), ns)  # noqa: S102
    return ns["LOCKED_MODEL_PANEL"], ns["validate_locked_model_panel"]


@pytest.mark.parametrize(
    ("commit_step", "expected_bucket"),
    [
        (1, "step1"),
        (2, "step2_4"),
        (4, "step2_4"),
        (5, "step5_16"),
        (16, "step5_16"),
        (17, "step17_64"),
        (64, "step17_64"),
        (65, "step65_cap"),
        (128, "step65_cap"),
    ],
)
def test_bucket_boundaries(commit_step: int, expected_bucket: str):
    assert diag_commit.bucket(commit_step) == expected_bucket


def test_locked_panel_order_validation_rejects_permuted_list():
    locked_model_panel, validate_locked_model_panel = _load_locked_panel_helpers()
    permuted = list(locked_model_panel)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    with pytest.raises(RuntimeError, match="locked 10-model panel mismatch"):
        validate_locked_model_panel(permuted)


def test_pinned_slice_hash_matches_locked_value():
    _prompts, _labels, data_hash = _load_calibration_jsonl_fast(str(PINNED_DATA))
    assert data_hash == PINNED_HASH

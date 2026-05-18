from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.anli_full_sweep as anli_sweep
import scripts.run_chord_vs_path_sweep as chord_sweep
from pri_calibrator import CalibrationProfile
from scripts.sweep_locking import claim_next_run_dir, hold_out_dir_lock


def _make_candidate_entry(
    *,
    family: str,
    rank_label: str,
    step: int,
    auroc: float,
    sign: int,
    n_evaluated: int,
) -> dict[str, object]:
    return {
        "cell": f"{family} {rank_label} @ step {step}",
        "step": step,
        "family": family,
        "rank_label": rank_label,
        "column_name": f"{family}_{rank_label}_{step}",
        "auroc": auroc,
        "sign": sign,
        "n_evaluated": n_evaluated,
    }


def _make_profile(
    synthetic_profile_dict: dict[str, object],
    *,
    model_slug: str = "test/model",
    n_calibration: int = 100,
    winner_family: str = "Centered",
    winner_label: str = "r=2",
    winner_step: int = 1,
    in_auc: float = 0.82,
    in_ci_lo: float = 0.72,
    in_ci_hi: float = 0.92,
    oob: float = 0.76,
    oob_ci_lo: float = 0.62,
    oob_ci_hi: float = 0.88,
    winner_stability: float = 0.91,
    winner_n_eval: int = 100,
    warnings: list[str] | None = None,
    extra_candidate_entries: list[dict[str, object]] | None = None,
) -> CalibrationProfile:
    data = copy.deepcopy(synthetic_profile_dict)
    data["model"]["slug"] = model_slug
    data["task"]["n_calibration"] = n_calibration
    data["task"]["n_pos"] = n_calibration // 2
    data["task"]["n_neg"] = n_calibration // 2
    data["detector"]["gen_step"] = winner_step
    data["detector"]["metric"]["family"] = winner_family
    data["detector"]["metric"]["label"] = winner_label
    data["calibration_stats"]["auroc"] = in_auc
    data["calibration_stats"]["auroc_bootstrap_ci_lo"] = in_ci_lo
    data["calibration_stats"]["auroc_bootstrap_ci_hi"] = in_ci_hi
    data["calibration_stats"]["oob_auroc_median"] = oob
    data["calibration_stats"]["oob_auroc_ci_lo"] = oob_ci_lo
    data["calibration_stats"]["oob_auroc_ci_hi"] = oob_ci_hi
    data["calibration_stats"]["winner_stability"] = winner_stability
    winner_entry = _make_candidate_entry(
        family=winner_family,
        rank_label=winner_label,
        step=winner_step,
        auroc=in_auc,
        sign=int(data["detector"]["sign"]),
        n_evaluated=winner_n_eval,
    )
    data["calibration_stats"]["candidate_panel"] = [winner_entry] + list(extra_candidate_entries or [])
    data["warnings"] = list(warnings or [])
    return CalibrationProfile(**data)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_chord_csv(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_idx", "label", "n_layers",
            "d_F_chord", "d_F_path_fixed", "d_F_path_varying",
            "curvature_fixed", "path_fixed_over_chord",
        ])
        writer.writerow([0, 0, 4, 1.0, 1.2, 1.1, 0.2, 1.2])
        writer.writerow([1, 1, 4, 2.0, 2.3, 2.2, 0.3, 1.15])
        writer.writerow([2, 0, 4, 3.0, 3.4, 3.1, 0.4, 1.133333])


class TestAnliWinnerSummary:
    def test_publishability_ignores_non_winner_coverage_warning(self, synthetic_profile_dict):
        profile = _make_profile(
            synthetic_profile_dict,
            warnings=[
                "insufficient_coverage_at_Fisher r=2 @ step 3 (n_evaluated=60/100; model EOS'd before this step too often)"
            ],
            extra_candidate_entries=[
                _make_candidate_entry(
                    family="Fisher",
                    rank_label="r=2",
                    step=3,
                    auroc=0.61,
                    sign=1,
                    n_evaluated=60,
                )
            ],
        )
        assert anli_sweep.publishability_reasons(profile) == []
        assert anli_sweep.is_publishable_winner(profile) is True

    def test_publishability_blocks_low_winner_coverage(self, synthetic_profile_dict):
        profile = _make_profile(
            synthetic_profile_dict,
            n_calibration=100,
            winner_n_eval=70,
        )
        reasons = anli_sweep.publishability_reasons(profile)
        assert any("insufficient_coverage_at_winner" in reason for reason in reasons)
        assert anli_sweep.is_publishable_winner(profile) is False

    def test_emit_summary_writes_explicit_winner_tables(self, tmp_path, synthetic_profile_dict):
        publishable = _make_profile(
            synthetic_profile_dict,
            model_slug="test/publishable",
            warnings=[
                "insufficient_coverage_at_Fisher r=2 @ step 3 (n_evaluated=60/100; model EOS'd before this step too often)"
            ],
            extra_candidate_entries=[
                _make_candidate_entry(
                    family="Fisher",
                    rank_label="r=2",
                    step=3,
                    auroc=0.58,
                    sign=1,
                    n_evaluated=60,
                )
            ],
        )
        blocked = _make_profile(
            synthetic_profile_dict,
            model_slug="test/blocked",
            in_auc=0.63,
            winner_stability=0.62,
            winner_n_eval=75,
        )
        profiles = {
            ("publishable", "R1"): publishable,
            ("blocked", "R1"): blocked,
        }

        anli_sweep.emit_summary(profiles, tmp_path)

        full_rows = _read_csv_rows(tmp_path / anli_sweep.WINNERS_FULL_FILENAME)
        publishable_rows = _read_csv_rows(tmp_path / anli_sweep.WINNERS_PUBLISHABLE_FILENAME)
        blocked_rows = _read_csv_rows(tmp_path / anli_sweep.WINNERS_BLOCKED_FILENAME)

        assert not (tmp_path / anli_sweep.DEPRECATED_WINNERS_FILENAME).exists()
        assert len(full_rows) == 2
        assert len(publishable_rows) == 1
        assert len(blocked_rows) == 1
        assert {row["publishable"] for row in full_rows} == {"yes", "no"}
        assert publishable_rows[0]["model"] == "publishable"
        assert publishable_rows[0]["publishability_reasons"] == ""
        assert blocked_rows[0]["model"] == "blocked"
        assert "low_auroc" in blocked_rows[0]["publishability_reasons"]
        assert "winner_unstable" in blocked_rows[0]["publishability_reasons"]
        assert "insufficient_coverage_at_winner" in blocked_rows[0]["publishability_reasons"]


class TestChordSweepRunner:
    def test_failure_rows_serialize_to_strict_json(self, tmp_path):
        failure = chord_sweep._failure_result(
            "mlx-community/Fake-4bit",
            tmp_path / "fake.csv",
            tmp_path / "fake.log",
            tmp_path / "fake.manifest.json",
            exit_code=2,
            reason="diagnostic exited with code 2",
            decision="run-failed",
        )
        text = json.dumps(chord_sweep._sanitize_for_json([failure]), allow_nan=False)
        payload = json.loads(text)
        assert payload[0]["corr_fixed"] is None
        assert payload[0]["corr_fixed_reason"] == "diagnostic exited with code 2"

    def test_skip_existing_reuses_only_exact_manifest(self, tmp_path, tmp_calibration_jsonl, monkeypatch):
        data_path = tmp_calibration_jsonl([("a", 0), ("b", 1), ("c", 0)], filename="data.jsonl").resolve()
        data_identity = chord_sweep._load_data_identity(data_path)
        script_hash = chord_sweep._hash_file(chord_sweep.SCRIPT_PATH)
        csv_path = chord_sweep.csv_path_for(tmp_path, "mlx-community/Fake-4bit")
        manifest_path = chord_sweep.manifest_path_for(tmp_path, "mlx-community/Fake-4bit")
        _write_chord_csv(csv_path)
        chord_sweep._write_manifest(
            manifest_path,
            chord_sweep._diagnostic_manifest(
                model_slug="mlx-community/Fake-4bit",
                data_identity=data_identity,
                max_new_tokens=8,
                limit=0,
                script_hash_sha256=script_hash,
            ),
        )

        called = {"count": 0}

        def _boom(*args, **kwargs):
            called["count"] += 1
            raise AssertionError("subprocess should not run when manifest matches")

        monkeypatch.setattr(chord_sweep.subprocess, "run", _boom)
        result = chord_sweep.run_one_model(
            sys.executable,
            "mlx-community/Fake-4bit",
            data_path,
            data_identity,
            tmp_path,
            script_hash_sha256=script_hash,
            max_new_tokens=8,
            limit=0,
            skip_existing=True,
        )
        assert called["count"] == 0
        assert result["status"] == "skipped-existing"

    def test_manifest_mismatch_forces_fresh_run(self, tmp_path, tmp_calibration_jsonl, monkeypatch):
        data_path = tmp_calibration_jsonl([("a", 0), ("b", 1), ("c", 0)], filename="data.jsonl").resolve()
        data_identity = chord_sweep._load_data_identity(data_path)
        script_hash = chord_sweep._hash_file(chord_sweep.SCRIPT_PATH)
        csv_path = chord_sweep.csv_path_for(tmp_path, "mlx-community/Fake-4bit")
        manifest_path = chord_sweep.manifest_path_for(tmp_path, "mlx-community/Fake-4bit")
        _write_chord_csv(csv_path)
        chord_sweep._write_manifest(
            manifest_path,
            chord_sweep._diagnostic_manifest(
                model_slug="mlx-community/Fake-4bit",
                data_identity={"data_path": str(data_path), "data_hash_sha256": "wrong"},
                max_new_tokens=8,
                limit=0,
                script_hash_sha256=script_hash,
            ),
        )

        called = {"count": 0}

        def _fake_run(cmd, cwd, stdout, stderr, text, check):
            called["count"] += 1
            _write_chord_csv(csv_path)
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(chord_sweep.subprocess, "run", _fake_run)
        result = chord_sweep.run_one_model(
            sys.executable,
            "mlx-community/Fake-4bit",
            data_path,
            data_identity,
            tmp_path,
            script_hash_sha256=script_hash,
            max_new_tokens=8,
            limit=0,
            skip_existing=True,
        )
        assert called["count"] == 1
        assert result["status"] == "ok"
        written_manifest = json.loads(manifest_path.read_text())
        assert written_manifest["data_hash_sha256"] == data_identity["data_hash_sha256"]

    def test_cleanup_failure_marks_model_failed(self, tmp_path, tmp_calibration_jsonl, monkeypatch):
        data_path = tmp_calibration_jsonl([("a", 0), ("b", 1), ("c", 0)], filename="data.jsonl").resolve()
        data_identity = chord_sweep._load_data_identity(data_path)
        script_hash = chord_sweep._hash_file(chord_sweep.SCRIPT_PATH)
        monkeypatch.setattr(chord_sweep, "_remove_stale_artifacts", lambda *args, **kwargs: "permission denied")
        result = chord_sweep.run_one_model(
            sys.executable,
            "mlx-community/Fake-4bit",
            data_path,
            data_identity,
            tmp_path,
            script_hash_sha256=script_hash,
            max_new_tokens=8,
            limit=0,
            skip_existing=False,
        )
        assert result["status"] == "failed"
        assert result["decision"] == "stale-artifact-cleanup-failed"
        assert result["corr_fixed_reason"] == "permission denied"

    def test_write_summary_meta_is_strict_json(self, tmp_path):
        failure = chord_sweep._failure_result(
            "mlx-community/Fake-4bit",
            tmp_path / "fake.csv",
            tmp_path / "fake.log",
            tmp_path / "fake.manifest.json",
            exit_code=2,
            reason="diagnostic exited with code 2",
            decision="run-failed",
        )
        meta = chord_sweep.write_summary_meta(tmp_path / "summary_meta.json", [failure])
        loaded = json.loads((tmp_path / "summary_meta.json").read_text())
        assert loaded["complete"] is False
        assert loaded["failed_models"] == ["mlx-community/Fake-4bit"]
        assert meta == loaded


class TestSweepLocking:
    def test_hold_out_dir_lock_fails_on_second_acquire(self, tmp_path):
        with hold_out_dir_lock(tmp_path):
            with pytest.raises(RuntimeError, match="another sweep already holds"):
                with hold_out_dir_lock(tmp_path):
                    pass
        with hold_out_dir_lock(tmp_path):
            pass

    def test_claim_next_run_dir_counts_up(self, tmp_path):
        first = claim_next_run_dir(tmp_path)
        second = claim_next_run_dir(tmp_path)
        assert first.name == "run-01"
        assert second.name == "run-02"

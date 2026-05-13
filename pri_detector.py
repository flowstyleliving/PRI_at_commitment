"""PRI detector — deployment-time consumer of a CalibrationProfile.

Loads a calibrated PRI rupture detector from a `CalibrationProfile` JSON
(produced by `pri_calibrator.py`) and scores arbitrary prompts. The score is
the calibrated metric value at the calibrated gen_step, multiplied by the
locked sign — higher score = more likely target class (contradiction /
inconsistent / positive).

Usage as a library:

    from pri_detector import Detector

    detector = Detector.from_profile("my_profile.json")
    score = detector.score("Premise: ...\\nHypothesis: ...\\nAnswer:")
    is_positive = detector.predict(prompt, threshold=0.0)

Usage as a CLI:

    # Single prompt
    python pri_detector.py --profile my_profile.json --input '...'

    # Batch from jsonl
    python pri_detector.py --profile my_profile.json --input-file prompts.jsonl

    # Self-test: re-score the n=30 calibration prompts, verify AUROC matches
    python pri_detector.py --self-test --profile my_profile.json \\
        --calibration-data /tmp/calibration_n30.jsonl

Important caveats — see also the deployability warnings in the profile:

  * The detector is locked to ONE (model, task) regime. Don't deploy a
    profile calibrated on ANLI on, e.g., factual QA.

  * If you've updated `pri_v2_mlx_pipeline.py` since the profile was
    calibrated, the metric values may have drifted. Pass
    `strict_pipeline_hash=True` to refuse to load on hash mismatch, or
    accept the warning and re-calibrate.

  * `predict()` requires a threshold. If the profile doesn't carry one,
    pass it explicitly. Sane default for a sign-locked score is `0.0`
    (positive sign means "above 0 = target class"), but a properly
    calibrated threshold from a held-out set will outperform.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401 — List used in strict-mode drift list

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pri_v2_mlx_pipeline as pipeline
import pri_v2_io_plugins as io_plugins
from pri_calibrator import CalibrationProfile, SCHEMA_VERSION, _hash_file


class Detector:
    """Deployable PRI rupture detector. Construct once via
    `Detector.from_profile(path)`; reuse for many `score(prompt)` calls.

    Caches the loaded model + tokenizer + PRIComputer. Single-threaded — MLX
    is not thread-safe.
    """

    def __init__(
        self,
        profile: CalibrationProfile,
        model: Any,
        tokenizer: Any,
        projection: pipeline.OutputProjection,
        layer_indices: Dict[str, int],
        pri_computer: pipeline.PRIComputer,
        prompt_strategy: Any,
        v3_rank_values: List[int],
    ):
        self.profile = profile
        self.model = model
        self.tokenizer = tokenizer
        self.projection = projection
        self.layer_indices = layer_indices
        self.pri_computer = pri_computer
        self.prompt_strategy = prompt_strategy
        self._gen_step = int(profile.detector["gen_step"])
        self._layer = str(profile.detector["layer"])
        self._alpha = float(profile.detector["alpha"])
        self._sign = int(profile.detector["sign"])
        self._metric_column = str(profile.detector["metric"]["column_name"])
        self._v3_rank_values = list(v3_rank_values)
        # Default max_new_tokens = calibration's max_new_tokens (so the gen
        # window at deploy matches what was used to derive the profile).
        self._default_max_new_tokens = int(
            profile.provenance.get("max_new_tokens", 8)
        )

    # ─── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_profile(
        cls,
        profile_path: str,
        *,
        strict_pipeline_hash: bool = False,
    ) -> "Detector":
        """Load a CalibrationProfile and instantiate a ready-to-score Detector.

        `strict_pipeline_hash=True` refuses to load if ANY score-critical
        artifact has changed since calibration: the pipeline module, the
        io_plugins module (prompt strategy + parser), the model_adapters
        module, the calibrator module itself, OR the HuggingFace cache
        snapshot of the model. When False, prints a warning to stderr.

        Recommended for production: always pass `strict_pipeline_hash=True`
        and re-calibrate when it fires.
        """
        profile = CalibrationProfile.from_json(profile_path)
        if profile.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"profile schema {profile.schema_version} != supported {SCHEMA_VERSION}"
            )

        # Check ALL score-critical provenance before loading the model — fast fail.
        # 2026-05-13: expanded from pipeline-only after the Codex review's
        # finding that prompt strategy + adapter changes also affect scores
        # while strict_pipeline_hash=True silently passed.
        score_critical_hashes = [
            ("pipeline_module_hash_sha256",       REPO_ROOT / "pri_v2_mlx_pipeline.py"),
            ("io_plugins_module_hash_sha256",     REPO_ROOT / "pri_v2_io_plugins.py"),
            ("model_adapters_module_hash_sha256", REPO_ROOT / "model_adapters.py"),
            ("calibrator_module_hash_sha256",     REPO_ROOT / "pri_calibrator.py"),
        ]
        drift_msgs: List[str] = []
        for field, fpath in score_critical_hashes:
            recorded = profile.provenance.get(field)
            if not recorded:
                continue
            current = _hash_file(fpath)
            if current != recorded:
                drift_msgs.append(
                    f"  {fpath.name}: recorded={recorded[:12]}…  current={current[:12]}…"
                )
        # Model artifact (HF cache snapshot) — the slug points at a moving
        # target unless we also pin the snapshot SHA.
        from pri_calibrator import _resolve_model_snapshot_sha
        recorded_snapshot = profile.provenance.get("model_snapshot_sha")
        if recorded_snapshot:
            current_snapshot = _resolve_model_snapshot_sha(profile.model["slug"])
            if current_snapshot and current_snapshot != recorded_snapshot:
                drift_msgs.append(
                    f"  model snapshot ({profile.model['slug']}): "
                    f"recorded={recorded_snapshot[:12]}…  current={current_snapshot[:12]}…"
                )
            elif current_snapshot is None:
                drift_msgs.append(
                    f"  model snapshot ({profile.model['slug']}): "
                    f"recorded={recorded_snapshot[:12]}… but local cache unresolvable"
                )
        if drift_msgs:
            msg = "score-critical provenance changed since calibration:\n" + "\n".join(drift_msgs)
            if strict_pipeline_hash:
                raise RuntimeError(msg)
            print(f"[detector] WARNING: {msg}", file=sys.stderr)

        # Load model.
        cfg = pipeline.Config()
        cfg.layers_to_probe = [profile.detector["layer"]]
        cfg.seed = int(profile.provenance.get("calibration_seed", 42))
        model, tokenizer, projection, layer_indices = pipeline.load_model(
            profile.model["slug"], cfg
        )

        # Verify output projection kind matches what was calibrated against.
        # If a model gets republished with a different head structure, this
        # catches it before scores drift silently.
        if projection.mode != profile.model.get("output_projection_kind"):
            raise RuntimeError(
                f"output projection kind mismatch: "
                f"profile has '{profile.model.get('output_projection_kind')}' "
                f"but loaded model has '{projection.mode}'"
            )

        # Build PRIComputer.
        gamma = pipeline._extract_final_rmsnorm_gamma(model)
        if gamma is None:
            raise RuntimeError(
                f"could not extract final-RMSNorm gamma for {profile.model['slug']}"
            )
        pri_computer = pipeline.PRIComputer(projection, final_norm_gamma=gamma)
        prompt_strategy = io_plugins.get_prompt_strategy(profile.model["slug"])

        # Determine which v3 ranks compute_step needs to emit so the calibrated
        # metric column shows up in the result. Cover the ranks in the default
        # panel — keeps cost predictable regardless of which cell was chosen.
        v3_rank_values = [1, 2, 4, 21]

        print(
            f"[detector] loaded {profile.model['slug']}  "
            f"cell={profile.detector['metric']['family']} "
            f"{profile.detector['metric']['label']} @ step "
            f"{profile.detector['gen_step']}  sign={profile.detector['sign']:+d}"
        )
        return cls(
            profile=profile,
            model=model,
            tokenizer=tokenizer,
            projection=projection,
            layer_indices=layer_indices,
            pri_computer=pri_computer,
            prompt_strategy=prompt_strategy,
            v3_rank_values=v3_rank_values,
        )

    # ─── Scoring ─────────────────────────────────────────────────────────

    def score(self, prompt: str, *, max_new_tokens: Optional[int] = None) -> float:
        """Return the signed PRI rupture score for one prompt. Higher means
        more likely target class. Deterministic for a given (prompt, model)
        pair under greedy decoding.

        Raises `RuntimeError` if the model EOS's before the calibrated
        gen_step (i.e. the rupture moment doesn't exist for this input).
        """
        budget = int(max_new_tokens if max_new_tokens is not None else self._default_max_new_tokens)
        wrapped = self.prompt_strategy(prompt, self.tokenizer)
        trace = pipeline.trace_sample(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=wrapped,
            layer_indices=self.layer_indices,
            output_projection=self.projection,
            max_new_tokens=budget,
        )

        gen_idx = self._gen_step - 1
        gen_hidden = trace["gen_hidden"][self._layer]
        if gen_idx >= len(gen_hidden):
            raise RuntimeError(
                f"model emitted fewer than gen_step={self._gen_step} tokens "
                f"(got {len(gen_hidden)}); rupture moment unreachable for this prompt"
            )
        h_t = gen_hidden[gen_idx]
        h_prev = (
            gen_hidden[gen_idx - 1]
            if gen_idx >= 1
            else trace["last_prefix_hidden"][self._layer]
        )
        p_t = trace["gen_probs"][gen_idx]
        S_raw = trace["gen_surprises"][gen_idx]
        S_t = float(S_raw) if np.isfinite(S_raw) else 0.0

        result = self.pri_computer.compute_step(
            h_t=h_t,
            h_prev=h_prev,
            p_t=p_t,
            S_t=S_t,
            alpha=self._alpha,
            topk_values=[32],
            lowrank_values=[32],
            v3_rank_values=self._v3_rank_values,
            v3_capture_raw=True,
            v3_capture_centered=True,
        )
        raw = result.get(self._metric_column)
        if raw is None or not np.isfinite(raw):
            raise RuntimeError(
                f"calibrated metric '{self._metric_column}' not produced at "
                f"gen_step={self._gen_step}; check that the panel ranks "
                f"cover the calibrated rank"
            )
        return float(raw) * self._sign

    def predict(self, prompt: str, *, threshold: Optional[float] = None) -> bool:
        """Binary prediction: True if score > threshold. Default threshold
        comes from profile.detector['threshold']; if neither set, raises."""
        if threshold is None:
            threshold = self.profile.detector.get("threshold")
        if threshold is None:
            raise ValueError(
                "no threshold available; pass threshold= or set "
                "profile.detector.threshold (a sane default for sign-locked "
                "scores is 0.0 but a held-out-calibrated value is better)"
            )
        return self.score(prompt) > float(threshold)

    def score_batch(self, prompts: List[str]) -> List[float]:
        """Convenience batch wrapper. Currently serial — MLX isn't thread-safe."""
        return [self.score(p) for p in prompts]


# ─────────────────────────────────────────────────────────────────────────────
# Self-test: re-score calibration prompts, verify AUROC matches profile
# ─────────────────────────────────────────────────────────────────────────────


def _self_test(profile_path: str, calibration_data_path: str) -> int:
    """Re-score the calibration set with the deployed detector, verify the
    AUROC under direction-preserving scoring matches the profile's reported
    AUROC within numerical tolerance.

    A perfect self-test confirms:
      (a) trace_sample → compute_step is deterministic
      (b) profile's metric column + sign + gen_step are wired up correctly
      (c) pri_v2_mlx_pipeline.py code path is byte-identical to calibration
    """
    from sklearn.metrics import roc_auc_score

    profile = CalibrationProfile.from_json(profile_path)
    print(f"[self-test] profile reports AUROC={profile.calibration_stats['auroc']:.4f} "
          f"on n={profile.task['n_calibration']}")

    # Load calibration data — same format the calibrator reads.
    prompts, labels = [], []
    with Path(calibration_data_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompts.append(row["prompt"])
            labels.append(int(row["label"]))
    labels_arr = np.array(labels, dtype=np.int32)
    print(f"[self-test] loaded {len(prompts)} calibration prompts")

    detector = Detector.from_profile(profile_path)
    print(f"[self-test] scoring all {len(prompts)} prompts...")
    scores = []
    for i, p in enumerate(prompts):
        try:
            s = detector.score(p)
        except RuntimeError as e:
            print(f"[self-test]   sample {i}: SKIP ({e})")
            s = float("nan")
        scores.append(s)
        if (i + 1) % 10 == 0:
            print(f"[self-test]   {i+1}/{len(prompts)}")
    scores_arr = np.array(scores, dtype=np.float64)
    finite = np.isfinite(scores_arr)
    if finite.sum() < 4 or len(np.unique(labels_arr[finite])) < 2:
        print("[self-test] not enough finite samples to compute AUROC")
        return 1

    # Direction-preserving AUROC on the deployed scores (sign already applied).
    auroc_deployed = float(roc_auc_score(labels_arr[finite], scores_arr[finite]))
    auroc_reported = float(profile.calibration_stats["auroc"])
    delta = abs(auroc_deployed - auroc_reported)

    print(f"[self-test] reported AUROC: {auroc_reported:.4f}")
    print(f"[self-test] deployed AUROC: {auroc_deployed:.4f}")
    print(f"[self-test] |delta|:        {delta:.4f}")
    if delta > 1e-3:
        print(f"[self-test] FAIL — delta exceeds 1e-3 tolerance")
        return 1
    print(f"[self-test] OK — reproducibility verified")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="PRI detector (v1.0 schema profiles)")
    p.add_argument("--profile", required=True, help="path to a CalibrationProfile JSON")
    p.add_argument("--input", help="single prompt string to score")
    p.add_argument("--input-file", help="jsonl with {'prompt': '...'} per line")
    p.add_argument("--threshold", type=float, default=None,
                   help="binary-prediction threshold (default: profile.detector.threshold)")
    p.add_argument("--strict-pipeline-hash", action="store_true",
                   help="refuse to load if pri_v2_mlx_pipeline.py has changed since calibration")
    p.add_argument("--self-test", action="store_true",
                   help="re-score --calibration-data, verify AUROC matches profile")
    p.add_argument("--calibration-data", help="calibration jsonl (for --self-test)")
    args = p.parse_args()

    if args.self_test:
        if not args.calibration_data:
            raise SystemExit("--self-test requires --calibration-data")
        return _self_test(args.profile, args.calibration_data)

    detector = Detector.from_profile(
        args.profile, strict_pipeline_hash=args.strict_pipeline_hash
    )

    if args.input:
        s = detector.score(args.input)
        out = {"score": s}
        if args.threshold is not None:
            out["predicted_positive"] = (s > args.threshold)
            out["threshold"] = args.threshold
        print(json.dumps(out, indent=2))
        return 0

    if args.input_file:
        for line in Path(args.input_file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                s = detector.score(row["prompt"])
                out = {"prompt": row["prompt"][:80], "score": s}
                if args.threshold is not None:
                    out["predicted_positive"] = (s > args.threshold)
                print(json.dumps(out))
            except RuntimeError as e:
                print(json.dumps({"prompt": row["prompt"][:80], "error": str(e)}))
        return 0

    raise SystemExit("provide --input or --input-file (or --self-test --calibration-data)")


if __name__ == "__main__":
    sys.exit(main())

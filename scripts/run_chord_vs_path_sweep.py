#!/usr/bin/env python3
"""Run the chord-vs-path diagnostic across a model panel.

Sequential, resumable runner around `scripts/diagnose_chord_vs_path.py`.
Each model gets its own CSV + log, then the runner writes an aggregated
summary CSV/JSON so cross-model results are easy to compare.

Usage:
    .venv/bin/python scripts/run_chord_vs_path_sweep.py \
        --data experiments/anli-sweep/2026-05-15/run-01/anli_R1_seed20260513_n75.jsonl \
        --model-preset broad7 \
        --out-dir experiments/chord-vs-path/2026-05-15/run-01
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnose_chord_vs_path.py"

from pri_calibrator import _load_calibration_jsonl
from scripts.diagnose_chord_vs_path import (
    EVALUATE_CURVATURE_CORR_THRESHOLD,
    PATH_COLLAPSE_CORR_THRESHOLD,
)
from scripts.sweep_locking import hold_out_dir_lock


MODEL_PRESETS: Dict[str, List[str]] = {
    "all": [
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        "mlx-community/Phi-4-mini-instruct-4bit",
        "mlx-community/gemma-3-1b-it-4bit",
        "mlx-community/gemma-3-4b-it-4bit",
        "mlx-community/Qwen3-1.7B-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/Qwen3-8B-4bit",
        "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    ],
    "broad7": [
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        "mlx-community/Phi-4-mini-instruct-4bit",
        "mlx-community/gemma-3-4b-it-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/Qwen3-8B-4bit",
    ],
    "primaries": [
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
    ],
    "smoke": [
        "mlx-community/gemma-3-1b-it-4bit",
    ],
}

MIN_CORR_SAMPLES = 3
REQUIRED_COLUMNS = {
    "sample_idx",
    "label",
    "n_layers",
    "d_F_chord",
    "d_F_path_fixed",
    "d_F_path_varying",
    "curvature_fixed",
}


def short_model_name(slug: str) -> str:
    return slug.split("/")[-1]


def csv_path_for(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_chord_vs_path.csv"


def log_path_for(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_chord_vs_path.log"


def manifest_path_for(out_dir: Path, model_slug: str) -> Path:
    return out_dir / f"{short_model_name(model_slug)}_chord_vs_path.manifest.json"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_data_identity(data_path: Path) -> Dict[str, Any]:
    _prompts, _labels, data_hash = _load_calibration_jsonl(str(data_path))
    return {
        "data_path": str(data_path),
        "data_hash_sha256": data_hash,
    }


def _diagnostic_manifest(
    *,
    model_slug: str,
    data_identity: Dict[str, Any],
    max_new_tokens: int,
    limit: int,
    script_hash_sha256: str,
) -> Dict[str, Any]:
    return {
        "model": model_slug,
        "data_path": data_identity["data_path"],
        "data_hash_sha256": data_identity["data_hash_sha256"],
        "max_new_tokens": int(max_new_tokens),
        "limit": int(limit),
        "diagnostic_script_hash_sha256": script_hash_sha256,
    }


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    return json.loads(manifest_path.read_text())


def _manifest_matches(actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    return actual == expected


def _write_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp_path, manifest_path)


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    return value


def _failure_result(
    model_slug: str,
    csv_path: Path,
    log_path: Path,
    manifest_path: Path,
    *,
    exit_code: int,
    reason: str,
    decision: str,
) -> Dict[str, Any]:
    return {
        "model": model_slug,
        "status": "failed",
        "exit_code": exit_code,
        "csv_path": str(csv_path),
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
        "n_rows": 0,
        "corr_fixed": float("nan"),
        "corr_fixed_reason": reason,
        "corr_varying": float("nan"),
        "corr_varying_reason": reason,
        "ratio_fixed": float("nan"),
        "min_ratio_fixed": float("nan"),
        "n_triangle_violations": 0,
        "decision": decision,
    }


def _remove_stale_artifacts(csv_path: Path, manifest_path: Path) -> Optional[str]:
    for path in (csv_path, manifest_path):
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            return f"failed to remove stale artifact {path}: {exc}"
    return None


def _corrcoef_or_reason(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    min_samples: int = MIN_CORR_SAMPLES,
) -> tuple[float, Optional[str]]:
    finite_mask = np.isfinite(xs) & np.isfinite(ys)
    usable = int(np.sum(finite_mask))
    if usable < min_samples:
        return float("nan"), f"need at least {min_samples} usable samples (got {usable})"

    xs_usable = xs[finite_mask]
    ys_usable = ys[finite_mask]
    if np.isclose(xs_usable.std(), 0.0) or np.isclose(ys_usable.std(), 0.0):
        return float("nan"), "correlation is undefined when either series has zero variance"

    corr = float(np.corrcoef(xs_usable, ys_usable)[0, 1])
    if not np.isfinite(corr):
        return float("nan"), "correlation remained undefined after filtering usable samples"
    return corr, None


def load_rows(csv_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        for row in reader:
            rows.append({
                "sample_idx": int(row["sample_idx"]),
                "label": int(row["label"]),
                "n_layers": int(row["n_layers"]),
                "d_F_chord": float(row["d_F_chord"]),
                "d_F_path_fixed": float(row["d_F_path_fixed"]),
                "d_F_path_varying": float(row["d_F_path_varying"]),
                "curvature_fixed": float(row["curvature_fixed"]),
            })
    return rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n_rows": 0,
            "corr_fixed": float("nan"),
            "corr_fixed_reason": "no usable rows",
            "corr_varying": float("nan"),
            "corr_varying_reason": "no usable rows",
            "ratio_fixed": float("nan"),
            "min_ratio_fixed": float("nan"),
            "decision": "no-usable-rows",
        }

    chords = np.array([r["d_F_chord"] for r in rows], dtype=np.float64)
    paths_fixed = np.array([r["d_F_path_fixed"] for r in rows], dtype=np.float64)
    paths_varying = np.array([r["d_F_path_varying"] for r in rows], dtype=np.float64)
    curvs = np.array([r["curvature_fixed"] for r in rows], dtype=np.float64)

    finite_pv = np.isfinite(paths_varying)
    corr_fixed, corr_fixed_reason = _corrcoef_or_reason(chords, paths_fixed)
    corr_varying, corr_varying_reason = _corrcoef_or_reason(
        chords[finite_pv], paths_varying[finite_pv]
    )
    ratio_fixed = float(np.mean(paths_fixed / np.maximum(chords, 1e-12)))
    min_ratio_fixed = float(np.min(paths_fixed / np.maximum(chords, 1e-12)))
    n_violations = int(np.sum(curvs < -1e-6))

    if corr_fixed_reason is not None:
        decision = "inconclusive"
    elif corr_fixed > PATH_COLLAPSE_CORR_THRESHOLD:
        decision = "path-collapses-to-chord"
    elif corr_fixed > EVALUATE_CURVATURE_CORR_THRESHOLD:
        decision = "evaluate-curvature"
    else:
        decision = "replacement-question-opens"

    return {
        "n_rows": len(rows),
        "corr_fixed": corr_fixed,
        "corr_fixed_reason": corr_fixed_reason,
        "corr_varying": corr_varying,
        "corr_varying_reason": corr_varying_reason,
        "ratio_fixed": ratio_fixed,
        "min_ratio_fixed": min_ratio_fixed,
        "n_triangle_violations": n_violations,
        "decision": decision,
    }


def summarize_csv(csv_path: Path) -> Dict[str, Any]:
    try:
        rows = load_rows(csv_path)
    except Exception as exc:
        return {
            "n_rows": 0,
            "corr_fixed": float("nan"),
            "corr_fixed_reason": f"failed to parse CSV: {exc}",
            "corr_varying": float("nan"),
            "corr_varying_reason": f"failed to parse CSV: {exc}",
            "ratio_fixed": float("nan"),
            "min_ratio_fixed": float("nan"),
            "n_triangle_violations": 0,
            "decision": "csv-parse-failed",
        }
    return summarize_rows(rows)


def expand_models(args: argparse.Namespace) -> List[str]:
    if args.models:
        return [m.strip() for m in args.models.split(",") if m.strip()]
    return MODEL_PRESETS[args.model_preset]


def run_one_model(
    python_bin: str,
    model_slug: str,
    data_path: Path,
    data_identity: Dict[str, Any],
    out_dir: Path,
    *,
    script_hash_sha256: str,
    max_new_tokens: int,
    limit: int,
    skip_existing: bool,
) -> Dict[str, Any]:
    csv_path = csv_path_for(out_dir, model_slug)
    log_path = log_path_for(out_dir, model_slug)
    manifest_path = manifest_path_for(out_dir, model_slug)
    expected_manifest = _diagnostic_manifest(
        model_slug=model_slug,
        data_identity=data_identity,
        max_new_tokens=max_new_tokens,
        limit=limit,
        script_hash_sha256=script_hash_sha256,
    )

    if skip_existing and csv_path.exists() and manifest_path.exists():
        try:
            actual_manifest = _load_manifest(manifest_path)
        except Exception:
            actual_manifest = {}
        if _manifest_matches(actual_manifest, expected_manifest):
            summary = summarize_csv(csv_path)
            if summary["decision"] != "csv-parse-failed":
                return {
                    "model": model_slug,
                    "status": "skipped-existing",
                    "exit_code": 0,
                    "csv_path": str(csv_path),
                    "log_path": str(log_path),
                    "manifest_path": str(manifest_path),
                    **summary,
                }

    # Fresh runs must not inherit a stale success artifact from an earlier
    # attempt. Remove the previous CSV before launching the child diagnostic.
    cleanup_error = _remove_stale_artifacts(csv_path, manifest_path)
    if cleanup_error is not None:
        return _failure_result(
            model_slug,
            csv_path,
            log_path,
            manifest_path,
            exit_code=1,
            reason=cleanup_error,
            decision="stale-artifact-cleanup-failed",
        )

    cmd = [
        python_bin,
        str(SCRIPT_PATH),
        "--model",
        model_slug,
        "--data",
        str(data_path),
        "--out",
        str(csv_path),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        return _failure_result(
            model_slug,
            csv_path,
            log_path,
            manifest_path,
            exit_code=proc.returncode,
            reason=f"diagnostic exited with code {proc.returncode}",
            decision="run-failed",
        )

    summary = summarize_csv(csv_path)
    if summary["decision"] != "csv-parse-failed":
        try:
            _write_manifest(manifest_path, expected_manifest)
        except Exception as exc:
            return _failure_result(
                model_slug,
                csv_path,
                log_path,
                manifest_path,
                exit_code=1,
                reason=f"failed to write manifest: {exc}",
                decision="manifest-write-failed",
            )
    return {
        "model": model_slug,
        "status": "ok" if summary["decision"] != "csv-parse-failed" else "failed",
        "exit_code": 0 if summary["decision"] != "csv-parse-failed" else 1,
        "csv_path": str(csv_path),
        "log_path": str(log_path),
        "manifest_path": str(manifest_path),
        **summary,
    }


def write_summary_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "model",
        "status",
        "exit_code",
        "n_rows",
        "corr_fixed",
        "corr_fixed_reason",
        "corr_varying",
        "corr_varying_reason",
        "ratio_fixed",
        "min_ratio_fixed",
        "n_triangle_violations",
        "decision",
        "csv_path",
        "log_path",
        "manifest_path",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 80)
    print("  Chord-vs-path sweep summary")
    print("=" * 80)
    for row in rows:
        corr_text = (
            f"{row['corr_fixed']:.4f}"
            if np.isfinite(row["corr_fixed"])
            else f"undefined ({row['corr_fixed_reason']})"
        )
        print(
            f"  {short_model_name(row['model'])}: status={row['status']}  "
            f"decision={row['decision']}  corr_fixed={corr_text}  n={row['n_rows']}"
        )


def write_summary_meta(out_path: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    failed = [row["model"] for row in rows if row["status"] == "failed"]
    meta = {
        "complete": len(failed) == 0,
        "n_models": len(rows),
        "n_failed_models": len(failed),
        "failed_models": failed,
    }
    out_path.write_text(json.dumps(_sanitize_for_json(meta), indent=2, allow_nan=False) + "\n")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description="run chord-vs-path diagnostic across a model panel")
    p.add_argument("--data", required=True, help="shared calibration jsonl for every model")
    p.add_argument("--out-dir", required=True, help="directory for per-model CSVs/logs + sweep summary")
    p.add_argument(
        "--model-preset",
        default="broad7",
        choices=sorted(MODEL_PRESETS),
        help="named model panel to run (ignored if --models is supplied)",
    )
    p.add_argument("--models", default="", help="comma-separated explicit model slugs")
    p.add_argument("--python-bin", default=sys.executable, help="python executable to use for each child run")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="forwarded to the per-model diagnostic")
    p.add_argument("--skip-existing", action="store_true", help="reuse existing per-model CSVs")
    args = p.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise SystemExit(f"data file not found: {data_path}")
    data_identity = _load_data_identity(data_path)
    script_hash_sha256 = _hash_file(SCRIPT_PATH)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = expand_models(args)
    if not models:
        raise SystemExit("no models selected")

    results: List[Dict[str, Any]] = []
    try:
        with hold_out_dir_lock(out_dir):
            print(f"[sweep] data={data_path}")
            print(f"[sweep] data_hash={data_identity['data_hash_sha256']}")
            print(f"[sweep] models={len(models)}")
            print(f"[sweep] out_dir={out_dir}")

            for i, model_slug in enumerate(models, start=1):
                print()
                print(f"[sweep] ({i}/{len(models)}) {model_slug}")
                result = run_one_model(
                    args.python_bin,
                    model_slug,
                    data_path,
                    data_identity,
                    out_dir,
                    script_hash_sha256=script_hash_sha256,
                    max_new_tokens=args.max_new_tokens,
                    limit=args.limit,
                    skip_existing=args.skip_existing,
                )
                print(
                    f"[sweep]   status={result['status']}  decision={result['decision']}  "
                    f"csv={result['csv_path']}"
                )
                if result["status"] == "failed":
                    print(f"[sweep]   see log: {result['log_path']}")
                results.append(result)

            summary_csv = out_dir / "summary.csv"
            summary_json = out_dir / "summary.json"
            summary_meta = out_dir / "summary_meta.json"
            write_summary_csv(summary_csv, results)
            summary_json.write_text(
                json.dumps(_sanitize_for_json(results), indent=2, allow_nan=False) + "\n"
            )
            meta = write_summary_meta(summary_meta, results)
            print_summary(results)
            print(f"\n[sweep] wrote {summary_csv}")
            print(f"[sweep] wrote {summary_json}")
            print(f"[sweep] wrote {summary_meta}")
            if not meta["complete"]:
                print(f"[sweep] INCOMPLETE panel — failed models: {meta['failed_models']}")
                return 1
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())

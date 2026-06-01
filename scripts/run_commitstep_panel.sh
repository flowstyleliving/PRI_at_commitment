#!/bin/bash
# Run the 10-model commit-step panel in one subprocess per model.
#
# Usage:
#   scripts/run_commitstep_panel.sh [data.jsonl] [out_dir] [log_file] [cap] [limit]
#
# Expects to be invoked from the PRI_at_commitment repo root.
set -euo pipefail

DATA="${1:-experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl}"
OUT_DIR="${2:-experiments/v4-mech-prep/2026-05-18/commitstep-panel/run-01}"
LOG="${3:-$OUT_DIR/panel_commitstep.log}"
CAP="${4:-128}"
LIMIT="${5:-0}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
PINNED_DATA_HASH_SHA256="94825f3d2029c0049f2a087b0093117edc576ada84f2a073b4eccdbf8e3fe3d5"

if [ ! -f "$DATA" ]; then
  echo "[commitstep-panel] data file not found: $DATA" >&2
  exit 1
fi

MODELS=(
  "mlx-community/Qwen3-1.7B-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
  "mlx-community/Phi-3.5-mini-instruct-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
  "mlx-community/Llama-3.1-8B-Instruct-4bit"
  "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
)
if [ -n "${COMMITSTEP_PANEL_MODELS:-}" ]; then
  echo "[commitstep-panel] COMMITSTEP_PANEL_MODELS override is disabled for the locked panel" >&2
  exit 1
fi

"$PYTHON_BIN" - "${MODELS[@]}" "$DATA" "$PINNED_DATA_HASH_SHA256" <<'PY'
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

repo_root = Path.cwd()

def _load_function(path: Path, *, fn_name: str, assign_names=()):
    module_ast = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in assign_names:
                    selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            selected.append(node)
    mini_module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mini_module)
    ns = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "np": np,
        "List": list,
        "Tuple": tuple,
        "Sequence": tuple,
    }
    exec(compile(mini_module, str(path), "exec"), ns)  # noqa: S102
    return ns

models = sys.argv[1:-2]
data_path = Path(sys.argv[-2]).expanduser().resolve()
expected_hash = sys.argv[-1]

belief_ns = _load_function(
    repo_root / "scripts" / "step0_belief_readout.py",
    fn_name="validate_locked_model_panel",
    assign_names=("LOCKED_MODEL_PANEL",),
)
belief_ns["validate_locked_model_panel"](models)

cal_ns = _load_function(
    repo_root / "pri_calibrator.py",
    fn_name="_load_calibration_jsonl",
)
_prompts, _labels, data_hash = cal_ns["_load_calibration_jsonl"](str(data_path))
if data_hash != expected_hash:
    raise SystemExit(
        f"pinned data hash mismatch: expected {expected_hash}, got {data_hash}"
    )
PY

mkdir -p "$OUT_DIR"
N=${#MODELS[@]}
FAILURES=0
STATUS_TSV="$OUT_DIR/.commitstep_panel_status.tsv"

LOCK_PATH=$(
  "$PYTHON_BIN" - "$OUT_DIR" <<'PY'
from pathlib import Path
import sys
from scripts.sweep_locking import acquire_out_dir_lock
print(acquire_out_dir_lock(Path(sys.argv[1]).expanduser().resolve()))
PY
)
cleanup() {
  "$PYTHON_BIN" - "$LOCK_PATH" <<'PY'
from pathlib import Path
import sys
from scripts.sweep_locking import release_out_dir_lock
release_out_dir_lock(Path(sys.argv[1]))
PY
}
trap cleanup EXIT

printf 'model\tstatus\tjson_path\tpersample_csv_path\treason\n' > "$STATUS_TSV"

{
  echo "[commitstep-panel] data=$DATA"
  echo "[commitstep-panel] out_dir=$OUT_DIR"
  echo "[commitstep-panel] cap=$CAP"
  echo "[commitstep-panel] limit=$LIMIT"
  echo "[commitstep-panel] models=$N"
  echo "[commitstep-panel] start: $(date)"
} | tee -a "$LOG"

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  NAME="${M##*/}"
  IDX=$((i+1))
  JSON_OUT="$OUT_DIR/${NAME}.commitstep-val.json"
  CSV_OUT="$OUT_DIR/${NAME}.commitstep-val.persample.csv"
  PER_LOG="$OUT_DIR/${NAME}.commitstep.log"

  {
    echo ""
    echo "[commitstep-panel] ($IDX/$N) $M started at $(date)"
  } | tee -a "$LOG"

  if [ -e "$JSON_OUT" ] || [ -e "$CSV_OUT" ] || [ -e "$PER_LOG" ]; then
    FAILURES=$((FAILURES + 1))
    printf '%s\tfailed\t%s\t%s\tartifact already exists in fresh run dir\n' "$M" "$JSON_OUT" "$CSV_OUT" >> "$STATUS_TSV"
    {
      echo "[commitstep-panel] ($IDX/$N) $NAME status=failed (artifact already exists in fresh run dir)"
    } | tee -a "$LOG"
    continue
  fi

  if PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/diag_commit_step.py \
      --model "$M" \
      --data "$DATA" \
      --out "$JSON_OUT" \
      --cap "$CAP" \
      --limit "$LIMIT" \
      > "$PER_LOG" 2>&1; then
    if [ ! -s "$JSON_OUT" ] || [ ! -s "$CSV_OUT" ]; then
      FAILURES=$((FAILURES + 1))
      printf '%s\tfailed\t%s\t%s\tmissing json or persample csv artifact\n' "$M" "$JSON_OUT" "$CSV_OUT" >> "$STATUS_TSV"
      {
        echo "[commitstep-panel] ($IDX/$N) $NAME status=failed (missing json or persample csv artifact) — see $PER_LOG"
      } | tee -a "$LOG"
    else
      printf '%s\tok\t%s\t%s\t\n' "$M" "$JSON_OUT" "$CSV_OUT" >> "$STATUS_TSV"
      {
        echo "[commitstep-panel] ($IDX/$N) $NAME status=ok  json=$JSON_OUT csv=$CSV_OUT"
      } | tee -a "$LOG"
    fi
  else
    RC=$?
    FAILURES=$((FAILURES + 1))
    printf '%s\tfailed\t%s\t%s\tchild exit %s\n' "$M" "$JSON_OUT" "$CSV_OUT" "$RC" >> "$STATUS_TSV"
    {
      echo "[commitstep-panel] ($IDX/$N) $NAME status=failed (exit $RC) — see $PER_LOG"
    } | tee -a "$LOG"
  fi
done

"$PYTHON_BIN" - "$OUT_DIR" "$STATUS_TSV" "$DATA" "$PINNED_DATA_HASH_SHA256" <<'PY'
from pathlib import Path
import csv
import json
import sys

out_dir = Path(sys.argv[1]).expanduser().resolve()
status_tsv = Path(sys.argv[2]).expanduser().resolve()
data_path = Path(sys.argv[3]).expanduser().resolve()
expected_hash = sys.argv[4]

summary_csv = out_dir / "commitstep_panel_summary.csv"
rows = list(csv.DictReader(status_tsv.open(newline=""), delimiter="\t"))
summary_rows = []
failed_models = []

for status in rows:
    model = status["model"]
    if status["status"] != "ok":
        failed_models.append(model)
        continue

    payload = json.loads(Path(status["json_path"]).read_text())
    buckets = payload.get("commit_step_buckets") or {}
    joint = payload.get("joint_2x2") or {}
    row = {
        "model": model,
        "n_total": payload.get("n_total"),
        "frac_immediate_step1": payload.get("frac_immediate_step1"),
        "frac_cot_step_gt1": payload.get("frac_cot_step_gt1"),
        "frac_abstain": payload.get("frac_abstain"),
        "median_commit_step": payload.get("median_commit_step"),
        "max_commit_step": payload.get("max_commit_step"),
        "n_model_error": payload.get("n_model_error"),
        "step1": buckets.get("step1", 0),
        "step2_4": buckets.get("step2_4", 0),
        "step5_16": buckets.get("step5_16", 0),
        "step17_64": buckets.get("step17_64", 0),
        "step65_cap": buckets.get("step65_cap", 0),
        "weak_only": buckets.get("weak_only", 0),
        "joint_correct_label0": joint.get("correct|label0", 0),
        "joint_correct_label1": joint.get("correct|label1", 0),
        "joint_wrong_label0": joint.get("wrong|label0", 0),
        "joint_wrong_label1": joint.get("wrong|label1", 0),
        "data_hash_sha256": payload.get("data_hash"),
    }
    if row["data_hash_sha256"] != expected_hash:
        failed_models.append(model)
        continue
    summary_rows.append(row)

if any(int(r["n_total"]) != 200 for r in summary_rows):
    mismatched = [(r["model"], r["n_total"]) for r in summary_rows if int(r["n_total"]) != 200]
    raise SystemExit(f"n_total mismatch (expected 200): {mismatched}")

fields = [
    "model", "n_total", "frac_immediate_step1", "frac_cot_step_gt1",
    "frac_abstain", "median_commit_step", "max_commit_step", "n_model_error",
    "step1", "step2_4", "step5_16", "step17_64", "step65_cap", "weak_only",
    "joint_correct_label0", "joint_correct_label1", "joint_wrong_label0",
    "joint_wrong_label1", "data_hash_sha256",
]
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow(row)

print(f"[commitstep-panel] wrote {summary_csv}")
for row in summary_rows:
    print(
        "[commitstep-panel] "
        f"{row['model']} "
        f"step1={row['frac_immediate_step1']} "
        f"cot_gt1={row['frac_cot_step_gt1']} "
        f"abstain={row['frac_abstain']} "
        f"median={row['median_commit_step']}"
    )
print(f"[commitstep-panel] data_hash_sha256={expected_hash}")
print(f"[commitstep-panel] n_rows={len(summary_rows)}")
if failed_models:
    print(f"[commitstep-panel] failed_models={failed_models}")
    raise SystemExit(1)
PY

echo "" | tee -a "$LOG"
if [ "$FAILURES" -gt 0 ]; then
  echo "[commitstep-panel] INCOMPLETE: $FAILURES/$N models failed at $(date)" | tee -a "$LOG"
  exit 1
fi
echo "[commitstep-panel] all $N models finished successfully at $(date)" | tee -a "$LOG"

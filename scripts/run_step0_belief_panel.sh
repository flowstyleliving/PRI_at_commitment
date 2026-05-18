#!/bin/bash
# Run the compact t=0 belief-readout panel in strict canary/score phases.
#
# Usage:
#   scripts/run_step0_belief_panel.sh canary <data.jsonl> <out_dir> <prereg.md> [log_file]
#   scripts/run_step0_belief_panel.sh score  <data.jsonl> <out_dir> <prereg.md> [log_file]
#
# prereg.md is required (it lives in the vault; repo files carry no
# vault-path defaults — repo<->wiki separation).
#
# Expects to be invoked from the PRI_at_commitment repo root.
set -euo pipefail

PHASE="$1"
DATA="$2"
OUT_DIR="$3"
PREREG="${4:?[belief-panel] prereg doc path required as arg 4 (no vault default — repo<->wiki separation)}"
LOG="${5:-/dev/stdout}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [ "$PHASE" != "canary" ] && [ "$PHASE" != "score" ]; then
  echo "[belief-panel] phase must be 'canary' or 'score'" >&2
  exit 1
fi
if [ ! -f "$DATA" ]; then
  echo "[belief-panel] data file not found: $DATA" >&2
  exit 1
fi
if [ ! -f "$PREREG" ]; then
  echo "[belief-panel] prereg doc not found: $PREREG" >&2
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
if [ -n "${STEP0_BELIEF_MODELS:-}" ]; then
  echo "[belief-panel] STEP0_BELIEF_MODELS override is disabled for the locked panel" >&2
  exit 1
fi

"$PYTHON_BIN" - "${MODELS[@]}" <<'PY'
import sys
from scripts.step0_belief_readout import validate_locked_model_panel

validate_locked_model_panel(sys.argv[1:])
PY

mkdir -p "$OUT_DIR"
N=${#MODELS[@]}
FAILURES=0
STATUS_TSV="$OUT_DIR/.belief_panel_status.tsv"

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

if [ "$PHASE" = "score" ]; then
  "$PYTHON_BIN" - "$OUT_DIR" "$DATA" "${MODELS[@]}" <<'PY'
from pathlib import Path
import sys
from pri_calibrator import _load_calibration_jsonl
from scripts.step0_belief_readout import spec_path_for, validate_panel_specs

out_dir = Path(sys.argv[1]).expanduser().resolve()
data_path = Path(sys.argv[2]).expanduser().resolve()
models = sys.argv[3:]
_prompts, _labels, data_hash = _load_calibration_jsonl(str(data_path))
spec_paths = [spec_path_for(out_dir, model) for model in models]
missing = [str(path) for path in spec_paths if not path.exists()]
if missing:
    raise SystemExit(f"missing frozen spec(s): {missing}")
validate_panel_specs(
    spec_paths=spec_paths,
    expected_data_hash_sha256=data_hash,
    expected_models=models,
)
PY
fi

{
  echo "[belief-panel] phase=$PHASE"
  echo "[belief-panel] data=$DATA"
  echo "[belief-panel] prereg=$PREREG"
  echo "[belief-panel] out_dir=$OUT_DIR"
  echo "[belief-panel] models=$N"
  echo "[belief-panel] start: $(date)"
} | tee -a "$LOG"

if [ "$PHASE" = "score" ]; then
  printf 'model\tstatus\tsummary_json_path\treason\n' > "$STATUS_TSV"
fi

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  NAME="${M##*/}"
  IDX=$((i+1))
  PER_LOG="$OUT_DIR/${NAME}_belief_${PHASE}.log"

  {
    echo ""
    echo "[belief-panel] ($IDX/$N) $M started at $(date)"
  } | tee -a "$LOG"

  if [ "$PHASE" = "canary" ]; then
    CANARY_JSON="$OUT_DIR/${NAME}_belief_canary.json"
    SPEC_JSON="$OUT_DIR/${NAME}_belief_spec.json"
    rm -f "$CANARY_JSON"
    if PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/step0_belief_readout.py canary \
        --model "$M" \
        --data "$DATA" \
        --out-dir "$OUT_DIR" \
        --prereg "$PREREG" \
        > "$PER_LOG" 2>&1; then
      if [ ! -s "$CANARY_JSON" ] || [ ! -s "$SPEC_JSON" ]; then
        FAILURES=$((FAILURES + 1))
        {
          echo "[belief-panel] ($IDX/$N) $NAME status=failed (missing canary/spec artifact) — see $PER_LOG"
        } | tee -a "$LOG"
      else
        {
          echo "[belief-panel] ($IDX/$N) $NAME status=ok  canary=$CANARY_JSON spec=$SPEC_JSON"
        } | tee -a "$LOG"
      fi
    else
      RC=$?
      FAILURES=$((FAILURES + 1))
      {
        echo "[belief-panel] ($IDX/$N) $NAME status=failed (exit $RC) — see $PER_LOG"
      } | tee -a "$LOG"
    fi
  else
    READOUT_CSV="$OUT_DIR/${NAME}_belief_readout.csv"
    READOUT_JSON="$OUT_DIR/${NAME}_belief_readout.json"
    rm -f "$READOUT_CSV" "$READOUT_JSON"
    if PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/step0_belief_readout.py score \
        --model "$M" \
        --data "$DATA" \
        --out-dir "$OUT_DIR" \
        --prereg "$PREREG" \
        > "$PER_LOG" 2>&1; then
      if [ ! -s "$READOUT_CSV" ] || [ ! -s "$READOUT_JSON" ]; then
        FAILURES=$((FAILURES + 1))
        printf '%s\tfailed\t\tmissing readout artifact\n' "$M" >> "$STATUS_TSV"
        {
          echo "[belief-panel] ($IDX/$N) $NAME status=failed (missing readout artifact) — see $PER_LOG"
        } | tee -a "$LOG"
      else
        printf '%s\tok\t%s\t\n' "$M" "$READOUT_JSON" >> "$STATUS_TSV"
        {
          echo "[belief-panel] ($IDX/$N) $NAME status=ok  csv=$READOUT_CSV json=$READOUT_JSON"
        } | tee -a "$LOG"
      fi
    else
      RC=$?
      FAILURES=$((FAILURES + 1))
      printf '%s\tfailed\t\tchild exit %s\n' "$M" "$RC" >> "$STATUS_TSV"
      {
        echo "[belief-panel] ($IDX/$N) $NAME status=failed (exit $RC) — see $PER_LOG"
      } | tee -a "$LOG"
    fi
  fi
done

if [ "$PHASE" = "score" ]; then
  "$PYTHON_BIN" - "$OUT_DIR" "$STATUS_TSV" <<'PY'
from pathlib import Path
import csv
import sys
from scripts.step0_belief_readout import write_panel_summary_from_status

out_dir = Path(sys.argv[1]).expanduser().resolve()
status_tsv = Path(sys.argv[2]).expanduser().resolve()
with status_tsv.open(newline="") as f:
    status_rows = list(csv.DictReader(f, delimiter="\t"))
meta = write_panel_summary_from_status(out_dir=out_dir, status_rows=status_rows)
print(f"[belief-panel] wrote {out_dir / 'panel_summary.csv'}")
print(f"[belief-panel] wrote {out_dir / 'panel_summary.json'}")
print(f"[belief-panel] complete={meta['complete']} failed_models={meta['failed_models']}")
PY
fi

echo "" | tee -a "$LOG"
if [ "$FAILURES" -gt 0 ]; then
  echo "[belief-panel] INCOMPLETE: $FAILURES/$N models failed at $(date)" | tee -a "$LOG"
  exit 1
fi
echo "[belief-panel] all $N models finished successfully at $(date)" | tee -a "$LOG"

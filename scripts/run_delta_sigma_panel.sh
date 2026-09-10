#!/bin/bash
# Run diagnose_delta_sigma_onaxis.py across the broad7 panel on one shared slice.
# Usage: scripts/run_delta_sigma_panel.sh <data.jsonl> <out_dir> [log_file]
# Expects to be invoked from the PRI_at_commitment repo root.
set -euo pipefail
DATA="$1"
OUT_DIR="$2"
LOG="${3:-/dev/stdout}"

MODELS=(
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/Phi-3.5-mini-instruct-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
)

mkdir -p "$OUT_DIR"
N=${#MODELS[@]}
FAILURES=0

{
  echo "[delta_sigma-panel] data=$DATA"
  echo "[delta_sigma-panel] out_dir=$OUT_DIR"
  echo "[delta_sigma-panel] models=$N"
} | tee -a "$LOG"

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  NAME="${M##*/}"
  IDX=$((i+1))
  CSV="$OUT_DIR/${NAME}_delta_sigma_onaxis.csv"
  PER_LOG="$OUT_DIR/${NAME}_delta_sigma_onaxis.log"

  {
    echo ""
    echo "[delta_sigma-panel] ($IDX/$N) $M started at $(date)"
  } | tee -a "$LOG"

  rm -f "$CSV"
  if .venv/bin/python -u scripts/diagnose_delta_sigma_onaxis.py \
      --model "$M" \
      --data "$DATA" \
      --out "$CSV" \
      > "$PER_LOG" 2>&1; then
    if [ ! -s "$CSV" ]; then
      FAILURES=$((FAILURES + 1))
      {
        echo "[delta_sigma-panel] ($IDX/$N) $NAME status=failed (missing or empty CSV) — see $PER_LOG"
      } | tee -a "$LOG"
      continue
    fi
    SUMMARY=$(grep -E '^\s+[0-9]+ \|' "$PER_LOG" | tail -5 || true)
    {
      echo "[delta_sigma-panel] ($IDX/$N) $NAME status=ok  csv=$CSV"
      [ -n "$SUMMARY" ] && echo "$SUMMARY"
    } | tee -a "$LOG"
  else
    RC=$?
    FAILURES=$((FAILURES + 1))
    {
      echo "[delta_sigma-panel] ($IDX/$N) $NAME status=failed (exit $RC) — see $PER_LOG"
    } | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
if [ "$FAILURES" -gt 0 ]; then
  echo "[delta_sigma-panel] INCOMPLETE: $FAILURES/$N models failed at $(date)" | tee -a "$LOG"
  exit 1
fi
echo "[delta_sigma-panel] all $N models finished successfully at $(date)" | tee -a "$LOG"

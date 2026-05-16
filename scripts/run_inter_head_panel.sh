#!/bin/bash
# Run diagnose_inter_head_disagreement.py across the 9-model panel on one
# shared slice. Mirrors run_delta_sigma_panel.sh shape so artifacts are
# directly comparable per-sample (same 200 ANLI R1 rows, same gen_step=1).
#
# Usage: scripts/run_inter_head_panel.sh <data.jsonl> <out_dir> [log_file]
# Expects to be invoked from the PRI_at_commitment repo root.
set -euo pipefail
DATA="$1"
OUT_DIR="$2"
LOG="${3:-/dev/stdout}"

# Ordered small → large to fail fast on small models if the wrapper has a
# regression. Mistral-Nemo last because it's the heaviest (12B).
MODELS=(
  "mlx-community/Qwen3-1.7B-4bit"
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/gemma-3-4b-it-4bit"
  "mlx-community/Phi-3.5-mini-instruct-4bit"
  "mlx-community/Phi-4-mini-instruct-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
  "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
)

mkdir -p "$OUT_DIR"
N=${#MODELS[@]}
FAILURES=0

{
  echo "[head-disagree-panel] data=$DATA"
  echo "[head-disagree-panel] out_dir=$OUT_DIR"
  echo "[head-disagree-panel] models=$N"
  echo "[head-disagree-panel] start: $(date)"
} | tee -a "$LOG"

for i in "${!MODELS[@]}"; do
  M="${MODELS[$i]}"
  NAME="${M##*/}"
  IDX=$((i+1))
  CSV="$OUT_DIR/${NAME}_head_disagree.csv"
  PER_LOG="$OUT_DIR/${NAME}_head_disagree.log"

  {
    echo ""
    echo "[head-disagree-panel] ($IDX/$N) $M started at $(date)"
  } | tee -a "$LOG"

  rm -f "$CSV"
  if PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/diagnose_inter_head_disagreement.py \
      --model "$M" \
      --data "$DATA" \
      --out "$CSV" \
      > "$PER_LOG" 2>&1; then
    if [ ! -s "$CSV" ]; then
      FAILURES=$((FAILURES + 1))
      {
        echo "[head-disagree-panel] ($IDX/$N) $NAME status=failed (missing or empty CSV) — see $PER_LOG"
      } | tee -a "$LOG"
      continue
    fi
    SUMMARY=$(grep -E '^\s+(final|mid|last_minus_1)' "$PER_LOG" | tail -3 || true)
    {
      echo "[head-disagree-panel] ($IDX/$N) $NAME status=ok  csv=$CSV"
      [ -n "$SUMMARY" ] && echo "$SUMMARY"
    } | tee -a "$LOG"
  else
    RC=$?
    FAILURES=$((FAILURES + 1))
    {
      echo "[head-disagree-panel] ($IDX/$N) $NAME status=failed (exit $RC) — see $PER_LOG"
    } | tee -a "$LOG"
  fi
done

echo "" | tee -a "$LOG"
if [ "$FAILURES" -gt 0 ]; then
  echo "[head-disagree-panel] INCOMPLETE: $FAILURES/$N models failed at $(date)" | tee -a "$LOG"
  exit 1
fi
echo "[head-disagree-panel] all $N models finished successfully at $(date)" | tee -a "$LOG"

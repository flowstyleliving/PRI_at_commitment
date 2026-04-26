#!/usr/bin/env bash
# Stage 2 J_n diagnostic rerun at N=200 (50/cell × 4 cells).
# Sequential across 4 primaries to fit M4 16GB. Per-model logs in logs/stage2_n200_<tag>_<ts>.log.

set -u

cd "$(dirname "$0")/.."

TS=$(date -u +%Y%m%d-%H%M%SZ)
SUMMARY="logs/stage2_n200_${TS}_summary.log"

MODELS=(
  "mlx-community/Llama-3.2-3B-Instruct-4bit"
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
)

echo "=== Stage 2 N=200 rerun starting at $(date -u) ===" | tee -a "$SUMMARY"

for MODEL in "${MODELS[@]}"; do
  TAG=$(echo "$MODEL" | awk -F/ '{print $2}')
  LOG="logs/stage2_n200_${TAG}_${TS}.log"
  echo "" | tee -a "$SUMMARY"
  echo "[$(date -u)] >>> START $MODEL" | tee -a "$SUMMARY"
  DIAG_MODEL="$MODEL" \
    DIAG_N_PER_CELL=50 \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python -u scripts/diagnose_norm_jacobian.py >"$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    echo "[$(date -u)] <<< OK    $MODEL  (log: $LOG)" | tee -a "$SUMMARY"
  else
    echo "[$(date -u)] <<< FAIL  $MODEL  rc=$RC  (log: $LOG)" | tee -a "$SUMMARY"
  fi
done

echo "" | tee -a "$SUMMARY"
echo "=== Stage 2 N=200 rerun finished at $(date -u) ===" | tee -a "$SUMMARY"

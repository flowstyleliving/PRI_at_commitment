set -u
cd /Users/msrk/Documents/PRI_at_commitment
BASE=experiments/residual-friction/2026-06-06/run-06
PAIRS="
qwen3-8b|mlx-community/Qwen3-8B-4bit
qwen3-1.7b|mlx-community/Qwen3-1.7B-4bit
llama31-8b|mlx-community/Llama-3.1-8B-Instruct-4bit
mistral-nemo-12b|mlx-community/Mistral-Nemo-Instruct-2407-4bit
dolphin-nemo-12b|mlx-community/dolphin-2.9.3-mistral-nemo-12b-4bit
deepseek-distill-qwen-7b|mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit
gemma-3-1b|mlx-community/gemma-3-1b-it-4bit
"
echo "START $(date)" > "$BASE/_progress.log"
echo "$PAIRS" | while IFS='|' read -r name slug; do
  [ -z "$name" ] && continue
  echo ">>> $name ($slug) $(date)" >> "$BASE/_progress.log"
  if .venv/bin/python scripts/pilot_residual_friction.py \
        --models "$slug" \
        --feature-dump-dir "$BASE/features" \
        > "$BASE/$name.log" 2>&1; then
    echo "    OK $name $(date)" >> "$BASE/_progress.log"
  else
    echo "    FAILED $name (exit $?) $(date)" >> "$BASE/_progress.log"
  fi
done
echo "DONE $(date)" >> "$BASE/_progress.log"

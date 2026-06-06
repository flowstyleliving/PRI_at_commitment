set -u
cd /Users/msrk/Documents/PRI_at_commitment
BASE=experiments/residual-friction/2026-06-06/run-06
declare -a MAP=(
 "qwen3-8b|Qwen3-8B-4bit"
 "qwen3-1.7b|Qwen3-1.7B-4bit"
 "llama31-8b|Llama-3.1-8B-Instruct-4bit"
 "mistral-nemo-12b|Mistral-Nemo-Instruct-2407-4bit"
 "deepseek-distill-qwen-7b|DeepSeek-R1-Distill-Qwen-7B-4bit"
)
for pair in "${MAP[@]}"; do
  name="${pair%%|*}"; slug="${pair##*|}"
  npz="$BASE/features/mlx-community_${slug}.residual_friction_features.npz"
  .venv/bin/python scripts/analyze_friction_layer_profile.py "$npz" \
     --out "$BASE/${name}.layer_profile.txt" && echo "profiled $name"
done
echo "PROFILES DONE"

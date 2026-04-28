#!/usr/bin/env bash
# Sequential n=150/cell runs across the 6 v3.1 models for chain_length-stratified
# analysis. Each run lands in its own experiments/v3-main-run/<DATE>/run-NN/ dir.
# At n=150/cell × 4 cells = 600 samples per model; per chain_length stratum =
# 300 samples (150 ctrl × 150 contr) — well-powered for the +0.02 sealed bar.
#
# Wall-clock estimate (Mac mini M4, 16 GB):
#   v3_1_main (4 models)   ~11.5h: Llama 1.8h + Mistral 1.8h + Qwen2.5 3.7h + Qwen3 4.2h
#   v3_1_phi_only           ~1.2h
#   v3_1_gemma4b_only       ~1.8h
#   Total                  ~14.5h overnight
#
# Sealed analysis plane unchanged: final layer, gen_step=1, rank=1 pinned.
# Sealed E18/E17b discipline preserved — primaries are the gate authority.
# Higher n is operational (sample size) not a sealed-spec change.
#
# Usage:
#   bash scripts/run_v3_1_stratified_n150.sh
# Or:
#   nohup bash scripts/run_v3_1_stratified_n150.sh > logs/v3_1_strat_n150_$(date +%Y%m%d-%H%M%SZ).log 2>&1 &

set +e  # don't abort if one model gate-fails — continue to the next
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d-%H%M%SZ)
LOGDIR=logs
mkdir -p "$LOGDIR"

PYBIN=.venv/bin/python
SEED=20260423
NPC=150
MGT=14
GMT=12
LAYERS=final

echo "===== v3.1 stratified n=150 sweep — start ${STAMP} ====="

# --- Run 1: primaries + Qwen3 (4 models) ----------------------------------
LOG1="${LOGDIR}/v3_1_main_n150_${STAMP}.log"
echo "[$(date +%H:%M:%S)] launching v3_1_main (Llama, Mistral, Qwen 2.5, Qwen3) → ${LOG1}"
PYTHONUNBUFFERED=1 "$PYBIN" -u scripts/run_v3_main.py \
    --scope v3_1_main \
    --n-per-cell "$NPC" \
    --seed "$SEED" \
    --max-gen-tokens "$MGT" \
    --gate-max-tokens "$GMT" \
    --layers "$LAYERS" \
    2>&1 | tee "$LOG1"

# --- Run 2: Phi-3.5-mini (1 model) ----------------------------------------
LOG2="${LOGDIR}/v3_1_phi_n150_${STAMP}.log"
echo "[$(date +%H:%M:%S)] launching v3_1_phi_only → ${LOG2}"
PYTHONUNBUFFERED=1 "$PYBIN" -u scripts/run_v3_main.py \
    --scope v3_1_phi_only \
    --n-per-cell "$NPC" \
    --seed "$SEED" \
    --max-gen-tokens "$MGT" \
    --gate-max-tokens "$GMT" \
    --layers "$LAYERS" \
    2>&1 | tee "$LOG2"

# --- Run 3: Gemma 3-4B (1 model) ------------------------------------------
LOG3="${LOGDIR}/v3_1_gemma4b_n150_${STAMP}.log"
echo "[$(date +%H:%M:%S)] launching v3_1_gemma4b_only → ${LOG3}"
PYTHONUNBUFFERED=1 "$PYBIN" -u scripts/run_v3_main.py \
    --scope v3_1_gemma4b_only \
    --n-per-cell "$NPC" \
    --seed "$SEED" \
    --max-gen-tokens "$MGT" \
    --gate-max-tokens "$GMT" \
    --layers "$LAYERS" \
    2>&1 | tee "$LOG3"

echo "===== v3.1 stratified n=150 sweep — done $(date +%Y%m%d-%H%M%SZ) ====="

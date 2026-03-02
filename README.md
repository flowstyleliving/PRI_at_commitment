# PRI_at_commitment

Standalone synthetic contradiction benchmark focused on **Predictive Rupture Index (PRI)** at generation commitment.

This repository is extracted from `/Users/mstrkttt/Documents/anthropic-ai-safety` and intentionally removes semantic-uncertainty / `hbar_s` logic.

## Included Signals

- `pri`
- `delta_sigma_jsd`
- `acr_mid_mean`
- SVD diagnostics (`pc1_ratio`, `effective_rank`, `spectral_entropy`)

## Reproduce the Full 3-Model Trial

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full experiment (same settings) for each model:

```bash
python scripts/run_synthetic_logic_experiment.py \
  --model-name llama_3.2_3b \
  --n-per-cell 200 \
  --preflight-per-cell 20 \
  --short-chain-steps 1 \
  --long-chain-steps 2 \
  --prompt-template-variant worked_example_v2 \
  --control-accuracy-gate 0.80 \
  --enforce-control-accuracy-gate \
  --window-tight 5 \
  --window-wide 12 \
  --max-gen-tokens 12 \
  --max-gen-tokens-escalated 24 \
  --n-permutations 10000 \
  --output-prefix synthetic_logic_promptfix_full_llama \
  --results-dir ./results
```

```bash
python scripts/run_synthetic_logic_experiment.py \
  --model-name mistral_7b \
  --n-per-cell 200 \
  --preflight-per-cell 20 \
  --short-chain-steps 1 \
  --long-chain-steps 2 \
  --prompt-template-variant worked_example_v2 \
  --control-accuracy-gate 0.80 \
  --enforce-control-accuracy-gate \
  --window-tight 5 \
  --window-wide 12 \
  --max-gen-tokens 12 \
  --max-gen-tokens-escalated 24 \
  --n-permutations 10000 \
  --output-prefix synthetic_logic_promptfix_full_mistral_7b \
  --results-dir ./results
```

```bash
python scripts/run_synthetic_logic_experiment.py \
  --model-name qwen_2.5_7b \
  --n-per-cell 200 \
  --preflight-per-cell 20 \
  --short-chain-steps 1 \
  --long-chain-steps 2 \
  --prompt-template-variant worked_example_v2 \
  --control-accuracy-gate 0.80 \
  --enforce-control-accuracy-gate \
  --window-tight 5 \
  --window-wide 12 \
  --max-gen-tokens 12 \
  --max-gen-tokens-escalated 24 \
  --n-permutations 10000 \
  --output-prefix synthetic_logic_promptfix_full_qwen25_7b \
  --results-dir ./results
```

Generate the three final publication plots:

```bash
python scripts/make_three_model_pri_figures.py --results-dir ./results
```

Expected output files:

- `./results/synthetic_logic_three_model_pri_step1_comparison.png`
- `./results/fig1_generation_pri_steps.png`
- `./results/fig2_prefix_null_three_model.png`

Optional generic summary plotting (single model):

```bash
python scripts/plot_synthetic_logic_results.py \
  --summary ./results/synthetic_logic_promptfix_full_llama_summary.json \
  --output-dir ./figures
```

## Repository Layout

- `synthetic_logic_loader.py`: synthetic puzzle generation with contradiction injection and anchor indexing.
- `synthetic_trace.py`: prefix and generation trace collection, event-window summaries, permutation tests.
- `pri_metrics.py`: PRI, surprise, cosine distance, cross-layer JSD, and SVD spectrum metrics.
- `scripts/run_synthetic_logic_experiment.py`: end-to-end runner with preflight behavioral gate.
- `scripts/make_three_model_pri_figures.py`: generates the 3 final three-model PRI figures.
- `scripts/plot_synthetic_logic_results.py`: plotting utility for event-aligned trajectories and peak offsets.
- `model_adapters.py`, `hidden_state_collector.py`, `attention_contribution.py`: MLX instrumentation stack.
- `config.py`: model registry and PRI-related numeric config.

## Notes

- Decoding defaults to greedy (`temperature=0`).
- The preflight gate enforces control-accuracy sanity before the full run.
- No semantic-uncertainty (`hbar_s`) or delta-mu reporting is included.

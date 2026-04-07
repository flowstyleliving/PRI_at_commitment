# PRI_at_commitment

Standalone synthetic contradiction benchmark focused on **Predictive Rupture Index (PRI)** at generation commitment, with an Apple Silicon / MLX-first PRI v2 pipeline.

This repository is extracted from a larger internal research workspace and intentionally excludes the earlier semantic-uncertainty / `hbar_s` logic.

## Current Status

The primary entrypoint is now [`pri_v2_mlx_pipeline.py`](./pri_v2_mlx_pipeline.py). It runs the full PRI v2 experiment, writes per-model checkpoints, skips models that already have completed results, resumes incomplete models from checkpoints, and then produces combined analysis tables and figures.

The most recent local full run completed across:

- `mlx-community/Llama-3.2-3B-Instruct-4bit`
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- `mlx-community/Qwen2.5-7B-Instruct-4bit`

Current step-1 / final-layer / `alpha=1.0` snapshot from `./pri_v2_results/`:

| Model | Control Acc. | Contradiction Acc. | Best Variant | AUROC |
| --- | ---: | ---: | --- | ---: |
| Llama-3.2-3B-Instruct-4bit | 1.00 | 1.00 | `pri_v2_topk32` | 0.7666 |
| Mistral-7B-Instruct-v0.3-4bit | 1.00 | 1.00 | `pri_v2_topk32` | 0.6715 |
| Qwen2.5-7B-Instruct-4bit | 0.98 | 1.00 | `pri_v2_lowrank32` | 0.7858 |

Across the completed run, the `pri_v2` family outperforms simpler baselines such as `surprise` and `pri_v1` on all three models.

## Quick Start

This workflow targets Apple Silicon with `mlx` / `mlx-lm`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Hugging Face authentication is needed on your machine:

```bash
hf auth login
```

Run the full PRI v2 pipeline:

```bash
python pri_v2_mlx_pipeline.py
```

The script will:

- generate the synthetic contradiction dataset
- run the three MLX models
- save rolling `*_checkpoint.parquet` files during incomplete runs
- skip models that already have `*_results.parquet`
- write combined analysis tables and figures once all models are finished

## Outputs

The PRI v2 pipeline writes results to `./pri_v2_results/`:

- `Llama-3.2-3B-Instruct-4bit_results.parquet`
- `Mistral-7B-Instruct-v0.3-4bit_results.parquet`
- `Qwen2.5-7B-Instruct-4bit_results.parquet`
- `*_trace_dumps.parquet`
- `all_results.parquet`
- `all_trace_dumps.parquet`
- `summary.parquet`
- `failure_cases.parquet`
- `fig1_v1_vs_v2_auroc.png` and `.pdf`
- `fig2_step_trajectory.png`
- `fig3_violins.png`
- `fig4_outcome_independence.png`
- `fig5_alpha_sweep.png`

These generated artifacts are local outputs and are gitignored by default.

## Repository Layout

- `pri_v2_mlx_pipeline.py`: primary MLX PRI v2 experiment runner, checkpoint/resume logic, analysis, and figure generation.
- `synthetic_logic_loader.py`: synthetic puzzle generation with contradiction injection and anchor indexing.
- `synthetic_trace.py`: prefix and generation trace collection, event-window summaries, and permutation tests.
- `pri_metrics.py`: PRI, surprise, cosine distance, cross-layer JSD, and related metrics.
- `model_adapters.py`, `hidden_state_collector.py`, `attention_contribution.py`: MLX instrumentation stack.
- `config.py`: model registry and numeric config.
- `scripts/run_synthetic_logic_experiment.py`: older end-to-end synthetic logic runner retained for legacy comparison.
- `scripts/make_three_model_pri_figures.py`: older three-model plotting utility for the legacy results layout.
- `scripts/plot_synthetic_logic_results.py`: older single-model plotting utility for the legacy results layout.

## Notes

- Decoding defaults to greedy generation (`temperature=0`).
- The preflight behavioral gate enforces control-accuracy sanity before the full run.
- The current MLX dequantization path is compatible with the `mlx` API shipped by recent Apple Silicon wheels.
- No semantic-uncertainty (`hbar_s`) or delta-mu reporting is included in this repository.

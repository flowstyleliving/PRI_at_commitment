# PRI v3 — Predictive Rupture Index at Commitment

> Token-level rupture detection for LLM generation, measured at the first-generated token, with a Fisher-pullback geometry and a HARP-style static-SVD baseline as the head-to-head control.

**v3 active line.** PRI v3 decomposes the hidden-state jump `Δh = h_t − h_prev` at commitment into a **direction** observable (`null_ratio`, the fraction of `Δh` that lies outside the top-r right singular vectors of `sqrt(p_t) · W_u`) rather than v2's magnitude scalar `d_F`. The hypothesis: contradictions push `Δh` *off* the commit direction, independent of how far it moved.

**Current status (2026-04-24).** Sealed E18 test passes 3/3 primaries at rank 1 on the 2026-04-23 main run with non-overlapping bootstrap CIs — Llama 3.2 3B 0.8593 [0.806, 0.908], Mistral 7B 0.8638 [0.814, 0.910], Qwen 2.5 7B 0.7274 [0.656, 0.795]. The sealed E19 interpretation gate (`null_gated = d_F · null_ratio` beats both null_bare and `v2_lowrank32` by non-overlap CI) is **FALSIFIED** on all 4 tested models — multiplicative interaction carries no signal beyond its components. Rank was not pinned in the original sealed block; v3.1 pre-registers rank 1 + seed 20260423 and replicates on fresh puzzles before any external claim.

**E17b companion.** The pipeline emits HARP-style raw-`W_u` null_ratio (`null_ratio_raw_rank{r}` + `raw_energy_rank{r}`) alongside the Fisher-weighted version, at the same rank sweep, so the head-to-head is `AUROC(null_ratio_rank1) − AUROC(null_ratio_raw_rank1)` on identical samples. The E17b sealed gate is on Qwen 2.5 with non-overlap 95% CI, margin ≥ 0.02.

---

## 🎯 What PRI measures, in one paragraph

For each generated token, take the hidden state right before (`h_prev`) and right after (`h_t`) the commitment. Their difference `Δh` points in some direction of hidden-state space. Project it onto the **top-r right singular vectors of `sqrt(p_t) · W_u`** (the output head, weighted by the current token distribution). Those top-r directions are the **commit directions** — moving the hidden state along them changes the output probability the most. Ordinary generation moves `Δh` along the commit direction. Contradiction-commitment tokens move `Δh` *off* the commit direction, into the null complement. `null_ratio_rank1` is the clean form: how much of `Δh` lives off the single most decisive commit axis. v3's bet is that this number separates contradictions from controls *independent* of how big `Δh` is.

## 🚀 Quick start

Apple Silicon with `mlx` / `mlx-lm`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# If HF auth is needed on this machine:
# hf auth login
```

### v3.1 main run — three-phase lean launch

Each axis is an independent scope so you can run, skip, or re-run any phase without touching the others' checkpoints. Sealed E18 + E17b gate authority lives in Phase 1.

```bash
# Phase 1 — sealed gate, primaries only. The verdict lands here.
# ~60-80 min on Mac mini M4.
.venv/bin/python scripts/run_v3_main.py \
  --scope v3_1_primaries \
  --n-per-cell 50 \
  --seed 20260423 \
  --max-gen-tokens 14

# Phase 2 (optional) — cross-generation companion: Qwen3-8B alone.
# Same seed → puzzle draws match Phase 1. ~15-20 min.
# Skip entirely if the 2026-04-23 Qwen3 data is sufficient for your framing.
.venv/bin/python scripts/run_v3_main.py \
  --scope v3_1_qwen3 \
  --n-per-cell 50 \
  --seed 20260423 \
  --max-gen-tokens 14

# Phase 3 (optional) — within-family-scale companion: Gemma 1B + 4B.
# Isolated because the full run_experiment loop with v3_capture_raw=True
# has never executed end-to-end on a Gemma checkpoint. ~40-60 min.
.venv/bin/python scripts/run_v3_main.py \
  --scope v3_1_gemmas \
  --n-per-cell 50 \
  --seed 20260423 \
  --max-gen-tokens 14
```

Prefer running Phase 1 first and verifying it clean before launching Phase 2 or Phase 3 — that way a Gemma-side adapter regression or a Qwen3 anomaly cannot contaminate the sealed-gate data. A convenience alias `--scope v3_1_main` combines Phase 1 + Phase 2 in one launch if you want primaries + Qwen3 without the manual sequencing.

### Overnight run — all three phases, unattended

For an overnight launch that runs all three phases in sequence with the right isolation semantics (Phase 1 fail-stop; Phase 2 and 3 best-effort-independent), use the bundled shell script:

```bash
# Foreground (progress visible in the terminal):
bash scripts/run_v3_1_overnight.sh

# Background / detached (close the terminal, let the display sleep):
nohup bash scripts/run_v3_1_overnight.sh >/dev/null 2>&1 &
```

The script:
- Logs merged stdout + stderr to `logs/v3_1_<timestamp>.log`.
- Touches `logs/v3_1_<timestamp>.done` on completion with a per-phase PASS/FAIL summary.
- Exits non-zero only if Phase 1 (the sealed-gate run) fails. Phase 2 / 3 failures are logged but don't fail the run.
- Uses the same sealed parameters (seed 20260423, rank 1 via Config defaults, n=50/cell, 80% gate threshold). Does NOT pass `--skip-gate` — the no-silent-override rule is enforced.
- Does NOT call `caffeinate`. Disable macOS sleep at the system level before launching (System Settings → Lock Screen / Battery → "Prevent automatic sleeping..."), or wrap the command with `caffeinate -disu bash scripts/run_v3_1_overnight.sh` if you prefer the launcher to handle it.

Artifacts land under `experiments/v3-main-run/<YYYY-MM-DD>/run-NN/`. E17b capture is on by default; pass `--no-e17b` to disable. Behavioral gate defaults to 80% control accuracy at n=20; `--skip-gate` bypasses for already-verified checkpoints (use sparingly — see Pre-reg discipline below).

### Legacy v2 run

```bash
python pri_v2_mlx_pipeline.py
```

Runs the v2 baseline (paper pipeline) with `v3_capture_raw=False` — Fisher-only columns, same three primaries, 200 samples/cell.

### Unit tests

```bash
.venv/bin/python scripts/test_e17b_raw_svd.py
```

Six bundles on synthetic fixtures: raw-SVD ground-truth match vs `np.linalg.svd`, range + monotonicity, dh-aligned / dh-orthogonal edge cases, chunked vs single-shot identity, cache reuse, compute_step flag parity.

## 🧭 Repo map

- **`pri_v2_mlx_pipeline.py`** — primary MLX pipeline. PRI v1 / v2 / v3 metric dispatcher in `PRIComputer`. Raw-`W_u` SVD cache lives on `OutputProjection.raw_right_singular_vectors` (chunked `W_uᵀ W_u` accumulation + `np.linalg.eigh`, static per model). `Config.v3_capture_raw = True` enables E17b by default.
- **`scripts/run_v3_main.py`** — v3 main-run launcher. Scopes: `primaries`, `extended`, `gemmas`, `non_gemma_extended`, `non_gemmas`, `all`, `v3_1_main` (= primaries + Qwen3-8B), `v3_1_gemmas` (= Gemma 1B + 4B). Flags: `--n-per-cell` / `--seed` / `--max-gen-tokens` / `--skip-gate` / `--gate-verbose` / `--pilot-threshold` / `--gate-max-tokens` / `--v3-capture` / `--no-e17b`.
- **`scripts/test_e17b_raw_svd.py`** — E17b unit tests (MLX-free; synthetic fixtures only).
- **`scripts/v3_capture_dryrun.py`** — Prereq 4 dryrun: 8 assertion bundles the shared pipeline must clear before a main-run launch. Covers per-row schema, capture schedule, provenance, finiteness, tripwire_healthy, tripwire_fault, dict_collision, consumer_audit.
- **`scripts/_paths.py`** — `experiment_run_dir(slug)` helper; new `experiments/<slug>/<YYYY-MM-DD>/run-NN/` layout with auto-increment per date.
- **`model_adapters.py`** — forward / hidden-state / vocab alignment adapters for the 7-model suite: primaries (Llama 3.2 3B, Mistral 7B v0.3, Qwen 2.5 7B) + extended (Gemma 3-1B, Gemma 3-4B, Qwen3-8B, Phi-3.5-mini). Handles tied-embed vs lm_head, Gemma 3 sqrt(hidden_size) post-embed scale + `sliding_window_pattern` masks, Gemma 3-4B multimodal wrapper reach-through, bfloat16 to_numpy casting.
- **`pri_metrics.py`** — PRI v1 cosine, surprise, L2, cross-layer JSD. v2 and v3 metrics live on `PRIComputer` in the pipeline module.
- **`synthetic_logic_loader.py`** — 2×2 factorial puzzle generator (chain_length ∈ {2, 5} × contradiction ∈ {False, True}), deterministic on seed.
- **`synthetic_trace.py`, `hidden_state_collector.py`, `attention_contribution.py`** — MLX instrumentation stack.
- **`config.py`** — model registry, gate thresholds (`GATE_THRESHOLDS` dict + `gate_threshold_for(model_type)` helper), numeric constants.
- **`experiments/`** — experiment artifacts. Structure: `<slug>/<YYYY-MM-DD>/run-NN/`. Slugs in use: `v3-main-run`, `v3-capture-dryrun`, `e22-direction-depth`, `e23-option-c`, `sup-spectral-band`, `prereq8-qwen-gate`. `.gitignore` skips `*.parquet` (large binaries) and tracks manifests + analysis JSON.
- **`scripts/run_synthetic_logic_experiment.py`, `scripts/make_three_model_pri_figures.py`, `scripts/plot_synthetic_logic_results.py`** — legacy v2-era utilities retained for reproduction.

## 🛡️ Pre-reg discipline and audit trail

This repo operates on pre-registration. Sealed parameters for the confirmatory v3 tests (E17 / E17b / E18 / E19) were frozen 2026-04-18 after an adversarial-review pass; rank and seed for the v3.1 replicate were frozen 2026-04-23 before any v3.1 data generation. Every commit referencing a sealed parameter is auditable against the plan's Amendments section.

**Sealed parameters for E18 (magnitude-independence test):**
- Unit of analysis: one row per `(sample, model, step=1, layer=final)`
- Per-model linear residualization of `null_ratio` against `d_F` (no pooling, no post-hoc re-spec)
- Threshold: `AUROC(null_ratio_resid) ≥ 0.60` with non-overlap 95% bootstrap CI vs 0.5 on ≥ 2 of 3 primary models
- Bootstrap: 1000 resamples at the sample level, not the token level
- **Pinned in v3.1 (2026-04-23):** rank `r = 1`, seed `20260423`

**Sealed parameters for E17b (Fisher-vs-static-SVD head-to-head):**
- Same analysis plane as E18
- Test statistic: `AUROC(null_ratio_rank1) − AUROC(null_ratio_raw_rank1)` on Qwen 2.5
- Acceptance: margin ≥ 0.02 with non-overlap 95% bootstrap CI
- Falsification: if raw ≥ Fisher on Qwen 2.5 by any margin or CIs overlap, v3 collapses toward HARP's static formulation

**No-silent-override rule.** Any gate-fail during launch must file a new Amendments entry in the plan before `--skip-gate` / `--pilot-threshold` overrides are applied. This protects the pre-reg from the "well, if we just lower the bar..." pattern.

**What's been settled vs what's open:**
- `[FALSIFIED]` E19 null_gated interpretation gate — sealed at `v2_lowrank32` rank, so rank 32 IS the sealed operating point here; failed on all 4 tested models in the 2026-04-23 main run. Final.
- `[PRIMARY-PASS / rank 1]` E18 null-space discharge hypothesis — 3/3 primaries pass at rank 1 on 2026-04-23; v3.1 replicate required before external claim.
- `[HYPOTHESIS / V3.1-READY]` E17b Fisher-vs-HARP head-to-head — capture shipped in pipeline, test pending on v3.1 data.
- `[OPEN]` Qwen3 8B weak signal at rank 1 final (AUROC ≈ 0.38 on 2026-04-23) and v2 collapse on Qwen3 (v2_lowrank32 ≈ 0.50 while surprise hits 0.96). Qwen-family diagnostic; not a v3 question.
- `[DEFERRED]` Phi-3.5-mini behavioral gate fails at n=20 (12/20 = 60% control accuracy) — reasoning-tuned string-match artifact suspected; follow-up is a `--gate-verbose` diagnostic in a separate launch.

## Legacy v2 baseline

Kept for reference and as the baseline to beat. Step-1 / final-layer / `alpha=1.0` on the original n=200/cell run (seed 42):

| Model | Control Acc. | Contradiction Acc. | Best Variant | AUROC |
| --- | ---: | ---: | --- | ---: |
| Llama-3.2-3B-Instruct-4bit | 1.00 | 1.00 | `pri_v2_topk32` | 0.7666 |
| Mistral-7B-Instruct-v0.3-4bit | 1.00 | 1.00 | `pri_v2_topk32` | 0.6715 |
| Qwen2.5-7B-Instruct-4bit | 0.98 | 1.00 | `pri_v2_lowrank32` | 0.7858 |

v2 beats v1 (cosine) on all three primaries. v3 passes its sealed E18 test at rank 1 with stronger CIs than v2 on the same models; v3.1 replicates to confirm on fresh puzzles.

## Notes

- Decoding defaults to greedy generation (`temperature=0`) so `Δh` at commitment is deterministic given the prompt.
- Raw-`W_u` SVD for E17b is one-time per model: ~30s for Llama 3B / Mistral 7B, ~2-3 min for Qwen 2.5 7B (152k-row lm_head is the heaviest). Per-sample cost is a single matvec against the cached basis.
- The MLX dequantization path is compatible with the `mlx` API shipped by recent Apple Silicon wheels; `to_numpy` handles bfloat16 via `mx.float32` cast before `np.array` (Gemma 3-4B and Qwen3-8B activations are bf16).
- This repository intentionally excludes the earlier semantic-uncertainty / `hbar_s` logic. PRI v3 is Fisher-geometric, not information-theoretic.

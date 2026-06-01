# PRI v3 — Predictive Rupture Index at Commitment

> Token-level rupture detection for LLM generation, measured at the first-generated token, with a Fisher-pullback geometry and a HARP-style static-SVD baseline as the head-to-head control.

**v3 active line.** PRI v3 decomposes the hidden-state jump `Δh = h_t − h_prev` at commitment into a **direction** observable (`null_ratio`, the fraction of `Δh` that lies outside the top-r right singular vectors of `sqrt(p_t) · W_u`) rather than v2's magnitude scalar `d_F`. The hypothesis: contradictions push `Δh` *off* the commit direction, independent of how far it moved.

**HARP** (Hu et al. 2025, [arXiv:2509.11536](https://arxiv.org/abs/2509.11536)) is the static-`W_u`-SVD baseline this repo tests Fisher pullback against. It decomposes the unembedding matrix once per model and uses the orthogonal complement of the top-r right singular vectors as a fixed "reasoning subspace" for hallucination detection. Fisher pullback is the natural per-sample generalization: same SVD machinery, but on `√p_t · W_u` (the unembedding re-weighted by the current token distribution), so the basis is locally tailored to where THIS prompt's prediction is sensitive instead of model-global.

**Current status (2026-04-27).** Sealed E18 test **passes 3/3 primaries at rank 1** on the 2026-04-26/27 main run at n=600 per model under J_n-corrected post-norm geometry — Llama 3.2 3B 0.8713 [0.842, 0.896], Mistral 7B 0.8707 [0.845, 0.897], Qwen 2.5 7B 0.6468 [0.603, 0.691]. Sealed E17b head-to-head **passes on Qwen 2.5** with Fisher decisive: Δ AUROC = +0.157 [+0.125, +0.190] — Fisher 0.8967 (sign +1) vs Raw 0.7396 (sign −1), clearing the sealed +0.02 bar by 7.9×. The sealed E19 interpretation gate (`null_gated = d_F · null_ratio` beats both null_bare and `v2_lowrank32` by non-overlap CI) remains **FALSIFIED** on all 4 tested models — multiplicative interaction carries no signal beyond its components.

**Cross-architecture (n=600, 6 models, 13 ranks).** At sealed rank=1, three architectures favor Fisher (Llama 3.2 3B, Qwen 2.5 7B, Gemma 3-4B) and three favor Raw (Mistral 7B, Qwen3 8B, Phi-3.5-mini); vendor and parameter-count are not predictive. The 6×13×2 model × rank × chain-length landscape exposes three motifs: (1) **Phi-3.5** is stable Raw across all 13 ranks (the canonical "HARP works as advertised" case, Raw_post_rank1 = 0.999); (2) **Gemma 4B** has a sharp within-model rank flip Fisher → Raw at r=2→r=3 robust to chain length; (3) **Mistral 7B** has a chain-length × rank Simpson's-paradox at sealed r=1 and at r=32, with cross-stratum spread Δ_cross = −0.575 at r=32 (the largest in the 156-cell landscape).

**E17b companion.** The pipeline emits HARP-style raw-`W_u` null_ratio (`null_ratio_raw_post_rank{r}` + `raw_energy_rank{r}`) alongside the Fisher-weighted version, at the same rank sweep, so the head-to-head is `AUROC(null_ratio_post_rank1) − AUROC(null_ratio_raw_post_rank1)` on identical samples. Both columns use J_n-corrected post-norm geometry — the legacy pre-norm column path was deleted 2026-04-26. The E17b sealed gate is on Qwen 2.5 with non-overlap 95% CI, margin ≥ 0.02.

**J_n geometry correction.** A coordinate-mismatch in the Fisher pullback — the pre-2026-04-26 pipeline projected raw pre-norm `Δh = h_t − h_prev` onto a basis derived from `√p_t · W_u` that lives in post-norm h-space — was identified, fixed, and the legacy code path deleted on 2026-04-26. Sealed E17b on Qwen 2.5 flipped from −0.166 (FAIL, Raw decisive) to +0.157 (PASS, Fisher decisive) under correction. Sealed E18 is unaffected because residualization against `d_F_lowrank32` (computed in the same buggy frame) absorbs the bias. Pre-registered _spec_ unchanged across the correction; only the implementation was revised.

**Workshop draft.** A workshop submission draft of the v3.1 results lives in the parent vault at `wiki/paper/draft.md` (status `[DRAFT]`, dated 2026-04-27). Frozen pre-reg snapshot: `PRI_V3_PRE_REGISTRATION_PLAN.md` at the repo root.

---

## 🎯 What PRI measures, in one paragraph

For each generated token, take the hidden state right before (`h_prev`) and right after (`h_t`) the commitment. Apply the model's final RMSNorm to both, take their difference `Δh_post = h_t_post − h_prev_post` (post-RMSNorm — the J_n-corrected geometry the unembedding `W_u` was trained against). Project it onto the **top-r right singular vectors of `sqrt(p_t) · W_u`** (the output head, weighted by the current token distribution). Those top-r directions are the **commit directions** — moving the hidden state along them changes the output probability the most. Ordinary generation moves `Δh_post` along the commit direction. Contradiction-commitment tokens move `Δh_post` *off* the commit direction, into the null complement. `null_ratio_post_rank1` is the clean form: how much of `Δh_post` lives off the single most decisive commit axis. v3's bet is that this number separates contradictions from controls *independent* of how far `Δh_post` moved.

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

### v3.1 main run — phased launch (powered, n=150/cell → n=600/model)

Each axis is an independent scope so you can run, skip, or re-run any phase without touching the others' checkpoints. Sealed E18 + E17b gate authority lives in Phase 1.

```bash
# Phase 1 — sealed gate, primaries + Qwen3 (the powered v3.1 main run).
# ~3-4 hours on Mac mini M4 at n=150/cell.
.venv/bin/python scripts/run_v3_main.py \
  --scope v3_1_main \
  --n-per-cell 150 \
  --seed 20260423 \
  --max-gen-tokens 14 \
  --gate-max-tokens 12 \
  --layers final

# Phase 2 — Phi-3.5-mini standalone (cross-vendor reasoning-tuned).
# ~30-45 min. Excluded from primaries (gate-pass via --gate-max-tokens 12 only).
.venv/bin/python scripts/run_v3_main.py \
  --scope v3_1_phi_only \
  --n-per-cell 150 \
  --seed 20260423 \
  --max-gen-tokens 14 \
  --gate-max-tokens 12 \
  --layers final

# Phase 3 — Gemma 3-4B standalone (within-family scale companion).
# Gemma 3-1B was excluded after gate-failing at 11/20=55% (model capability,
# not parser — defaults to "Answer: NO" on YES controls). ~30-45 min.
.venv/bin/python scripts/run_v3_main.py \
  --scope v3_1_gemma4b_only \
  --n-per-cell 150 \
  --seed 20260423 \
  --max-gen-tokens 14 \
  --gate-max-tokens 12 \
  --layers final

# Sealed-gate analyzer (post-norm geometry only after 2026-04-26 cleanup):
.venv/bin/python scripts/analyze_sealed_gate.py \
  --run-dir experiments/v3-main-run/<DATE>/run-NN
```

The 2026-04-26/27 powered runs land in `experiments/v3-main-run/2026-04-26/run-09/` (4 primaries + Qwen3) and `experiments/v3-main-run/2026-04-27/run-{01,02}/` (Phi-3.5-mini + Gemma 3-4B). Smaller per-cell counts (n=50, n=20) remain available via `--n-per-cell` overrides for smoke runs.

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
- Test statistic: `AUROC(null_ratio_post_rank1) − AUROC(null_ratio_raw_post_rank1)` on Qwen 2.5 (post-norm geometry; legacy pre-norm path deleted 2026-04-26)
- Acceptance: margin ≥ 0.02 with non-overlap 95% bootstrap CI
- Falsification: if raw ≥ Fisher on Qwen 2.5 by any margin or CIs overlap, v3 collapses toward HARP's static formulation

**No-silent-override rule.** Any gate-fail during launch must file a new Amendments entry in the plan before `--skip-gate` / `--pilot-threshold` overrides are applied. This protects the pre-reg from the "well, if we just lower the bar..." pattern.

**What's been settled vs what's open:**
- `[FALSIFIED]` E19 null_gated interpretation gate — sealed at `v2_lowrank32` rank, so rank 32 IS the sealed operating point here; failed on all 4 tested models in the 2026-04-23 main run. Final.
- `[PASS / 3-of-3]` E18 null-space discharge hypothesis (sealed) — 3/3 primaries pass at rank 1 on the 2026-04-26/27 powered run (n=600/model, J_n-corrected post-norm geometry). Bootstrap CIs ~33% tighter than the n=200 prelim.
- `[PASS / Qwen 2.5]` E17b Fisher-vs-HARP head-to-head (sealed) — Δ AUROC = +0.157 [+0.125, +0.190] on Qwen 2.5 7B at n=600, Fisher decisive. Replicates the corrected-geometry n=200 prelim within bootstrap noise.
- `[DESCRIPTIVE]` Cross-architecture rank landscape — 6 models × 13 ranks × 2 chain-length strata = 156 cells. Three motifs documented (Phi stable Raw, Gemma 4B rank flip, Mistral chain-length × rank Simpson's-paradox). Not pre-registered; reported descriptively in §4.3 of the workshop draft.
- `[OPEN]` Qwen3 8B weak signal at rank 1 final (AUROC ≈ 0.38 on 2026-04-23) and v2 collapse on Qwen3 (v2_lowrank32 ≈ 0.50 while surprise hits 0.96). Qwen-family diagnostic; not a v3 question.
- `[EXCLUDED]` Gemma 3-1B excluded from v3.1 after gate-failing at 11/20 = 55% under the post-PR#7 stratified-sampling and three-tier `check_answer` parser fixes. `--gate-verbose` confirmed model-capability rather than parser failure (defaults to `Answer: NO` on YES controls regardless of premises). Within-family scale axis collapses to a single point; left to v4.
- `[INCLUDED VIA RESCUE]` Phi-3.5-mini gate-passes with `--gate-max-tokens 12` operational rescue (front-loads `Answer: YES` then continues with format-completion). Filed in pre-reg amendments.

## v3 sealed vs baselines (n=600 powered, J_n-corrected)

| Primary | surprise | PRI v1 cosine | PRI v2 topk32 | PRI v2 lowrank32 | v3 null_ratio_post_rank1 (sealed) |
|---|:---:|:---:|:---:|:---:|:---:|
| Llama 3.2 3B | 0.6347 | 0.6224 | 0.7528 | 0.7500 | **0.8975** |
| Mistral 7B | 0.5187 | 0.5309 | 0.6623 | 0.6618 | **0.7849** |
| Qwen 2.5 7B | 0.8947 | 0.9155 | 0.7906 | 0.7948 | **0.8967** |

v3 outperforms PRI v1 and v2 on Llama and Mistral by clear margins. On Qwen 2.5, surprise (0.8947) and PRI v1 cosine (0.9155) are competitive with the sealed v3 metric (0.8967) — see §5.1 of the workshop draft for the qualitative-commit-token discussion.

## Legacy v2 baseline

Kept for reference. Step-1 / final-layer / `alpha=1.0` on the original n=200/cell run (seed 42):

| Model | Control Acc. | Contradiction Acc. | Best Variant | AUROC |
| --- | ---: | ---: | --- | ---: |
| Llama-3.2-3B-Instruct-4bit | 1.00 | 1.00 | `pri_v2_topk32` | 0.7666 |
| Mistral-7B-Instruct-v0.3-4bit | 1.00 | 1.00 | `pri_v2_topk32` | 0.6715 |
| Qwen2.5-7B-Instruct-4bit | 0.98 | 1.00 | `pri_v2_lowrank32` | 0.7858 |

## 🛠️ Tools

### Commitment Tracker — `scripts/commitment_tracker.py`

Dogfooding PRI's commitment-architecture lens on personal infrastructure. Scans Hermes session transcripts for future-tense declarations, stores them in a local SQLite DB (`~/.hermes/commitments.db`), and surfaces overdue commitments for closure.

```bash
# Scan recent sessions for commitments
python scripts/commitment_tracker.py scan --days 7

# Check pending commitments (cron-friendly)
python scripts/commitment_tracker.py check --report

# Interactive check (terminal)
python scripts/commitment_tracker.py check

# Manually add a commitment
python scripts/commitment_tracker.py add "I'll ship the paper by Friday" --check 2026-06-05

# Resolve non-interactively
python scripts/commitment_tracker.py resolve 3 --kept|--broken|--acknowledged

# Show all tracked commitments
python scripts/commitment_tracker.py status
```

Scheduled daily via `hermes cron`: scans for new declarations, surfaces anything past its check date. The extraction is regex-based (not LLM) — fast, deterministic, zero-cost.

ACE ≠ PRI — this tracker is a meta-tool, not part of the detector.

## Notes

- Decoding defaults to greedy generation (`temperature=0`) so `Δh` at commitment is deterministic given the prompt.
- Raw-`W_u` SVD for E17b is one-time per model: ~30s for Llama 3B / Mistral 7B, ~2-3 min for Qwen 2.5 7B (152k-row lm_head is the heaviest). Per-sample cost is a single matvec against the cached basis.
- The MLX dequantization path is compatible with the `mlx` API shipped by recent Apple Silicon wheels; `to_numpy` handles bfloat16 via `mx.float32` cast before `np.array` (Gemma 3-4B and Qwen3-8B activations are bf16).
- This repository intentionally excludes the earlier semantic-uncertainty / `hbar_s` logic. PRI v3 is Fisher-geometric, not information-theoretic.

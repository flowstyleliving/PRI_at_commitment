# PRI — Predictive Rupture Index at Commitment

> Pre-generation commitment signals read from a language model's **own internal state** at the moment it commits to its first generated token — increasingly without ever touching the output vocabulary head.

The through-line of this project is a single question: **at the instant a model commits to an answer, does its internal state already carry a readable signature of *how* it is committing** — confidently, against its own knowledge, off-axis? Each version (v1 → v5) is a different place to listen for that signature, and the arc moves steadily *away* from the output projection `W_u` (the "answer key") and *toward* the model's own routing and update dynamics.

---

## 🔥 v5 (current branch) — residual-stream sub-layer friction

Inside a transformer block the residual update is `Δh = a + m`, where `a` is the attention write (post-`W_o`, pre-add) and `m` is the MLP write (pre-add). v5's bet is that a commitment tell lives in the **friction between `a` and `m`** — not in either write alone, and crucially **not in their sum `Δh`**.

- 🧭 **Mechanism.** Attention *routes* (where to pull information from); the MLP acts as *key-value memory* (Geva et al. 2021). Under pre-norm the MLP reads `norm(h + a)`, so `m` is a **response** to `a`: friction = the knowledge layer **endorsing vs vetoing** the route attention just committed to.
- 🧬 **Orthogonal by construction.** `Δh = a + m` is invariant to the friction (same sum, opposite fight), so any earlier signal that reads `Δh` is blind to it. Whether friction *carries* a tell is the open empirical question this branch tests.
- 🆓 **`W_u`-free.** Friction is measured from `a` and `m` directly (angle / destructive-interference / directed veto) — no output vocabulary head required.

### Pilot status — 9-model screen (ANLI R1, n=200, t=0)

A screening pilot asks the decisive question: **does friction add discrimination *on top of* the prior direction signal (and beyond mere magnitude / route size)?** Results so far point to a **late-layer-localized** signal, present on capable Qwen and Llama models and absent on Mistral / Gemma:

| Model | Family | Residual-friction screen (incremental over the prior signal + route size) |
|---|---|---|
| Qwen2.5-7B | Qwen | ✅ positive (late-layer; one control runs warm — true effect slightly lower) |
| Qwen3-8B | Qwen | ✅ positive (clean controls) |
| Llama-3.2-3B | Llama | ✅ positive |
| Llama-3.1-8B | Llama | ✅ positive **at a late-layer window** — diluted to null by a full-window mean |
| Qwen3-1.7B | Qwen | — null (below the capability/scale threshold) |
| Mistral-7B · Mistral-Nemo-12B | Mistral | — null (both scales; sink/magnitude-dominated) |
| Gemma-3-4B | Gemma | — null |

📌 **Key finding is about the operating point, not just the models.** The signal concentrates in the **last 3–4 layers**; averaging friction across a wide mid-band *dilutes* it in proportion to model depth. That is exactly why the deeper Llama-3.1-8B first read as a non-replication before a per-layer audit surfaced its strong late-layer spike. The next step pins a late-layer window and runs the production calibrator's nested-OOB bootstrap for a real confidence interval.

> ⚠️ These are **screening** intervals (split + training variance), not inferential CIs. "Positive" means *clears the screen → eligible to promote to a sealed run*, not *validated*. Two models also exposed control leaks (one warm random-direction control; one broken shuffled-labels control on a reasoning-distill), flagged for the sealed pass.

### Run the v5 pilot

```bash
# 1) Score a model: capture a/m at the commit locus, dump per-sample features.
.venv/bin/python scripts/pilot_residual_friction.py \
  --models mlx-community/Qwen3-8B-4bit \
  --feature-dump-dir experiments/residual-friction/<DATE>/run-NN/features

# 2) Offline (no MLX): per-layer / sliding-window profile from the dumped features.
.venv/bin/python scripts/analyze_friction_layer_profile.py \
  experiments/residual-friction/<DATE>/run-NN/features/*.npz
```

The pilot self-protects on unfamiliar architectures: an allowlist gate → a sub-layer-component gate → a per-layer `a + m` reconstruction check → a length-spanning native-logits parity check (≤ 5e-3) → a finite-fraction check. Anything that does not decompose cleanly as `h + a + m` **aborts loudly** rather than emitting wrong features. The `.npz` dumps are the durable handoff: all downstream statistics (profiles, paired/null-centered tests, the sealed calibrator) run offline without re-loading any model.

**v5 artifacts & code**
- `scripts/pilot_residual_friction.py` — the friction screen (a/m capture, locked primary endpoint, controls, feature dump).
- `scripts/analyze_friction_layer_profile.py` — offline per-layer / sliding-window profiler over a feature dump.
- `scripts/friction_residualizer.py` — friction primitives (interference, directed veto) + the conditional-normal residualizer for the future correctness-labelled variant.
- `scripts/test_residual_friction.py` — identity/algebra unit suite (incl. the byte-identical-raw-friction-separated-only-by-direction contract).
- `model_adapters.py` — sub-layer capture helpers (`layer_supports_sublayer_capture`, pre-norm `a`/`m` split with a verify path).
- `experiments/residual-friction/` — run logs, feature `.npz` dumps, layer profiles.

---

## 🧭 The line so far — v1 → v5 (overview)

Each step listens for the same commitment signature in a different place; the trend is toward signals that need **less of the output head** and **more of the model's own dynamics**.

| Version | What it reads | Needs `W_u`? | Status |
|---|---|:---:|---|
| **v1** | Token surprise coupled to a representation-space cosine rupture (multiplicative). | no | superseded by v2 |
| **v2** | A **magnitude** of the commit-time update under a Fisher-pullback geometry. | partial | superseded by v3 |
| **v3** | A **direction** observable at the output head — *where* the update points relative to the most decisive commit axes, independent of how far it moved. Validated across a model panel and hardened into a production calibration library. | yes | sealed; **internals & sealed parameters are not in this repo** (live in the pre-registration / paper) |
| **v4 — ACE** | Reads the **attention landscape itself** (a *pre-generation belief readout*), `W_u`-free. Spine of the current paper. | no | sealed / paper |
| **v5** | **Residual-stream sub-layer friction** (attention write vs MLP write), `W_u`-free and orthogonal to the v3 sum. | no | this branch — pilot/screen |

🔒 **On v3:** the precise direction metric, the sealed gate parameters, the geometry correction, and the per-model confirmatory numbers are deliberately **kept out of this public branch**. They belong to the frozen pre-registration and the paper. What matters for the v3→v5 story is only the shape: v3 established that a *direction*-based commit signal beats a *magnitude*-based one and is deployable via per-(model, distribution) calibration — and that motivated pushing the readout off the output head entirely (v4, v5).

🏗️ **Production library (model-agnostic).** `pri_calibrator.py` fits a per-(model, deployment-distribution) profile against a labeled `.jsonl` and persists a versioned, selection-bias-corrected (nested out-of-bag bootstrap) calibration profile; `pri_detector.py` is the byte-reproducible scoring side with pipeline-hash drift checks. v5's positive cluster is promoted into this same machinery for its real confidence intervals — the friction feature dumps plug straight in, no model re-run.

---

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

Decoding defaults to greedy (`temperature=0`) so the commit-time update `Δh` is deterministic given the prompt. The legacy v1/v2 baseline pipeline (`pri_v2_mlx_pipeline.py`) and the v3-era launchers remain in the tree for reproduction but are not the focus of this branch.

---

## 🗺️ Repo map (v5-forward)

- **`scripts/pilot_residual_friction.py`** — v5 friction screen (see above).
- **`scripts/analyze_friction_layer_profile.py`** — v5 offline layer profiler.
- **`scripts/friction_residualizer.py`**, **`scripts/test_residual_friction.py`** — v5 primitives + tests.
- **`pri_calibrator.py`**, **`pri_detector.py`** — production calibration + detection library (model-agnostic; the sealed-run path for any promoted signal).
- **`model_adapters.py`** — forward / hidden-state / vocab alignment across the model suite (tied-embed vs lm_head, post-embed scaling, sliding-window masks, bf16 casting) + the v5 sub-layer capture helpers.
- **`pri_runtime.py`** — shared tracing runtime (prefix/commit hidden-state capture, single-pass forward).
- **`pri_v2_mlx_pipeline.py`**, **`pri_metrics.py`** — legacy v1/v2 metric pipeline, retained for reproduction.
- **`synthetic_logic_loader.py`**, **`synthetic_trace.py`**, **`hidden_state_collector.py`**, **`attention_contribution.py`** — MLX instrumentation stack.
- **`config.py`** — model registry, gate thresholds, numeric constants.
- **`experiments/`** — artifacts, `<slug>/<YYYY-MM-DD>/run-NN/`. `.gitignore` skips large binaries, tracks manifests + analysis JSON.

---

## 🛡️ Pre-registration discipline

This repo operates on pre-registration: confirmatory tests have their parameters frozen *before* data generation, and every result carries an explicit `[OPEN] / [PASS] / [FALSIFIED] / [EXCLUDED]` status. Two standing rules:

- **No silent gate overrides.** Any behavioral-gate failure during a launch must file an Amendments entry before any threshold/skip flag is applied — this protects the pre-reg from the "just lower the bar" pattern.
- **Audit the operating point before falsifying.** Sweep the unpinned-parameter neighborhood (layer window, rank, residualizer, …) before writing any `[FALSIFIED]` tag. Localized nulls are common — v5's Llama-3.1-8B "non-replication" was exactly one of these (a full-window mean hiding a late-layer signal).

Screening results (like the v5 pilot) are explicitly **not** confirmatory: they gate promotion to a sealed nested-OOB run, which is where real confidence intervals come from.

---

## 🛠️ Tools

### Commitment Tracker — `scripts/commitment_tracker.py`

Dogfoods PRI's commitment lens on personal infrastructure: scans session transcripts for future-tense declarations, stores them in a local SQLite DB, and surfaces overdue commitments. Regex-based extraction (deterministic, zero-cost), schedulable via cron. A meta-tool — not part of the detector.

```bash
python scripts/commitment_tracker.py scan --days 7
python scripts/commitment_tracker.py check --report
python scripts/commitment_tracker.py status
```

---

## Notes

- The MLX dequantization path matches the `mlx` API on recent Apple Silicon wheels; `to_numpy` handles bf16 via an `mx.float32` cast (Gemma-3 / Qwen3 activations are bf16).
- This repository intentionally excludes the earlier semantic-uncertainty / `hbar_s` logic. PRI's line is geometric/dynamical, not information-theoretic.

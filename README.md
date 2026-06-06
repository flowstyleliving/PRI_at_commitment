# Internal Knowledge Veto

> Catching a language model **vetoing its own commitment from the inside** — the moment its knowledge layer overrides the answer its attention just routed to — read directly from the residual stream at the commit token, without ever touching the output vocabulary head.

**Internal Knowledge Veto (IKV)** is the current name, and the **v5** incarnation, of a line of work developed through v4 as **PRI (Predictive Rupture Index)**. The through-line is a single question: **at the instant a model commits to an answer, does its internal state already carry a readable signature of *how* it is committing** — confidently, against its own knowledge, off-axis? The arc moves steadily *away* from the output projection `W_u` (the "answer key") and *toward* the model's own routing and update dynamics. v5 reads the commitment tell **entirely from inside the transformer block**.

---

## 🎛️ v6 — Projection Veto (readout-space conflict)

v5 taught the hard lesson: `a` and `m` can fight in residual space for benign reasons. The same-`Δh` and residual-budget baselines deflated the Qwen/Llama signal, meaning much of the apparent friction was norm budgeting rather than epistemic conflict.

v6 moves the lens to the readout: **we do not care if attention and MLP fight in the hallway; we care if they fight over the steering wheel.** For each attention write `a` and MLP write `m`, project both through frozen `W_u` contrast directions and ask whether they oppose each other on answer-relevant axes:

```text
p_c(x) = <u_c, x>
projection_veto_c = -p_c(a) * p_c(m)
```

The primary projection should use `W_u` directly as a **contrast bank**, not a single raw top-1 logit: start with frozen YES-vs-NO / entail-vs-contradict token-bucket contrasts, then use top-1-vs-runner-up as an unsupervised secondary. If final norm matters, use the local norm-linearized direction `u_c(h) = J_norm(h)^T (W_u[i] - W_u[j])`.

The same-`Δh` benign baseline remains mandatory: a v6 signal only counts if projection-space conflict survives identical net residual update, matched cancellation, projection-budget controls, random directions, and shuffled labels.

First Qwen2.5-7B screen (ANLI R1, n=200, t=0) is a useful falsification check: raw projection-veto clears strongly (`Δ=+0.110 [0.100,0.124]` over `null+route`), but the same-`Δh` projection baseline clears by the same amount (`Δ=+0.111`) and the net statistic is `-0.0012`. Projection-budget also explains the lift. So v6 does see answer-axis conflict, but on this first run it reads as projection-space norm/budget structure, not a clean Knowledge Veto.

Spec: [docs/v6_projection_veto.md](docs/v6_projection_veto.md)

---

## 🔥 v5 — Internal Knowledge Veto (residual-stream sub-layer friction)

Inside a transformer block the residual update is `Δh = a + m`, where `a` is the attention write (post-`W_o`, pre-add) and `m` is the MLP write (pre-add). IKV's bet is that a commitment tell lives in the **friction between `a` and `m`** — the knowledge layer's **veto** of the route attention just committed to — and that this lives in neither write alone, and crucially **not in their sum `Δh`**.

- 🧭 **Mechanism = the veto.** Attention *routes* (where to pull information from); the MLP acts as *key-value memory* (Geva et al. 2021). Under pre-norm the MLP reads `norm(h + a)`, so `m` is a **response** to `a`. A large `m` reversing `a` is the model's **internal knowledge vetoing** the route it just committed to — the signal IKV is named for.
- 🧬 **Orthogonal by construction.** `Δh = a + m` is invariant to the friction (same sum, opposite fight), so any signal that reads `Δh` is blind to the veto. Whether the veto *carries* a commitment tell is the open empirical question this branch tests.
- 🆓 **`W_u`-free.** The veto is measured from `a` and `m` directly (angle / destructive-interference / directed veto) — no output vocabulary head required.

### Pilot status — same-Delta corrected screen (ANLI R1, n=200, t=0)

A screening pilot asks the decisive question: **does the veto add discrimination *on top of* the prior direction signal (and beyond mere magnitude / route size)?** The first random-û-controlled screen looked positive on capable Qwen and Llama models. The stricter schema-v3 rerun adds the **Benign Cancellation Baseline**: hold `Δh = a + m` fixed, match raw cancellation, and rotate only the hidden disagreement channel off the consequential direction. Under that baseline, most of the original Qwen/Llama signal is explained as benign cancellation / geometry rather than a clean directed knowledge veto.

| Model | Family | Same-Delta corrected status |
|---|---|---|
| Qwen2.5-7B | Qwen | raw full-window +0.1205, same-Δ floor +0.1129 → net +0.0076; best selected 3-layer net +0.0366 |
| Qwen3-8B | Qwen | raw full-window +0.0955, same-Δ floor +0.1235 → net -0.0280; best selected 3-layer net +0.0062 |
| Llama-3.2-3B | Llama | raw full-window +0.0458, same-Δ floor +0.0477 → net -0.0019; best selected 3-layer net +0.0102 |
| Llama-3.1-8B | Llama | raw full-window +0.0150, same-Δ floor +0.0171 → net -0.0021; best selected 3-layer net +0.0336 |
| Qwen3-1.7B | Qwen | — null (below the capability/scale threshold) |
| Mistral-7B · Mistral-Nemo-12B | Mistral | — null (both scales; sink/magnitude-dominated) |
| Gemma-3-4B | Gemma | — null |

📌 **Key correction: random-û was too weak a floor.** Same-Delta benign cancellation is the right negative control for this candidate. It preserves the net residual update and raw cancellation magnitude, so it catches "A and M cancel because the block is norm-bounding/refining" rather than "A and M carry an epistemic conflict." The remaining selected-window residuals are small and post-hoc; they are not enough to promote to sealed nested-OOB without a fresh pre-registered operating point.

> ⚠️ These are **screening** intervals (split + training variance), not inferential CIs. "Positive" means *clears the screen → eligible to promote to a sealed run*, not *validated*. Two models also exposed control leaks (one warm random-direction control; one broken shuffled-labels control on a reasoning-distill), flagged for the sealed pass.

### Benign Cancellation Baseline

The central identifiability guard is: **hold `Δh = a + m` fixed and test whether the hidden split still carries signal**. `scripts/benign_cancellation_baseline.py` adds a pure NumPy same-`Δh` harness that constructs paired decompositions with byte-identical net residual updates and matched route norms; only the real split places disagreement on the epistemic direction. Current synthetic result: real epistemic split Δ=+0.4833, benign same-`Δh` Δ=-0.0063, net +0.4896.

The historical schema-v2 Qwen/Llama dumps only supported the random-û floor: Qwen2.5 net +0.1048, Qwen3-8B net +0.0987, Llama-3.2-3B net +0.0517, Llama-3.1-8B full-window net +0.0198. Schema-v3 reruns in `experiments/residual-friction/2026-06-06/run-07/` show that those estimates were anti-conservative: full-window same-Δ nets are Qwen2.5 +0.0076, Qwen3-8B -0.0280, Llama-3.2-3B -0.0019, Llama-3.1-8B -0.0021.

Schema v3 dumps now persist the sufficient same-`Δh` projections directly as `Xbenign`: raw interference/veto magnitude is matched, `Δh` is held fixed, and the hidden disagreement channel is rotated off the consequential direction. That lets the offline baseline report `real friction - same-Δh benign` before sealed nested-OOB promotion, without storing full `a`/`m` vectors.

### Residual-Norm Budget Test

`scripts/analyze_residual_budget.py` tests the more prosaic explanation: attention writes a route, the MLP counterweights it, and the block keeps the residual update inside a norm budget. It builds budget features from `||a||`, `||m||`, `||a+m||`, path balance, trim, and gain ratios, then asks whether friction still adds OOF AUROC after those budget features or after the stricter same-`Δh` benign floor.

Run-07 says: budget features explain most of Qwen2.5 and Llama-3.2, partly explain Qwen3, and are redundant once raw friction is present on Llama-3.1. The decisive check is stricter: **friction adds essentially nothing after same-`Δh` benign** across the promoted Qwen/Llama candidates: Qwen2.5 +0.0134, Qwen3 -0.0036, Llama-3.2 -0.0010, Llama-3.1 -0.0043. That supports the residual-budget / benign-cancellation interpretation over a clean Knowledge Veto.

### Run the Internal Knowledge Veto pilot

```bash
# 1) Score a model: capture a/m at the commit locus, dump per-sample features.
.venv/bin/python scripts/pilot_residual_friction.py \
  --models mlx-community/Qwen3-8B-4bit \
  --feature-dump-dir experiments/residual-friction/<DATE>/run-NN/features

# 2) Offline (no MLX): per-layer / sliding-window profile from the dumped features.
.venv/bin/python scripts/analyze_friction_layer_profile.py \
  experiments/residual-friction/<DATE>/run-NN/features/*.npz

# 3) Offline (no MLX): same-Delta synthetic guard + random-u floor from dumps.
.venv/bin/python scripts/benign_cancellation_baseline.py \
  experiments/residual-friction/<DATE>/run-NN/features/*.npz

# 4) Offline (no MLX): residual-norm budget diagnostic.
.venv/bin/python scripts/analyze_residual_budget.py \
  experiments/residual-friction/<DATE>/run-NN/features/*.npz
```

The pilot self-protects on unfamiliar architectures: an allowlist gate → a sub-layer-component gate → a per-layer `a + m` reconstruction check → a length-spanning native-logits parity check (≤ 5e-3) → a finite-fraction check. Anything that does not decompose cleanly as `h + a + m` **aborts loudly** rather than emitting wrong features. The `.npz` dumps are the durable handoff: all downstream statistics (profiles, paired/null-centered tests, the sealed calibrator) run offline without re-loading any model.

**Internal Knowledge Veto (v5) artifacts & code**
- `scripts/pilot_residual_friction.py` — the IKV screen (a/m capture, locked primary endpoint, controls, feature dump).
- `scripts/analyze_friction_layer_profile.py` — offline per-layer / sliding-window profiler over a feature dump.
- `scripts/benign_cancellation_baseline.py` — same-`Δh` synthetic guard + random-û floor report for existing dumps.
- `scripts/analyze_residual_budget.py` — offline norm-budget diagnostic over schema-v2/v3 dumps.
- `scripts/friction_residualizer.py` — veto primitives (interference, directed veto) + the conditional-normal residualizer for the future correctness-labelled variant.
- `scripts/test_residual_friction.py` — identity/algebra unit suite (incl. the byte-identical-raw-friction-separated-only-by-direction contract).
- `model_adapters.py` — sub-layer capture helpers (`layer_supports_sublayer_capture`, pre-norm `a`/`m` split with a verify path).
- `experiments/residual-friction/` — run logs, feature `.npz` dumps, layer profiles.

---

## 🧭 The line so far — v1 → v6 (PRI lineage → Internal Knowledge Veto)

Each step listens for the same commitment signature in a different place; the trend is toward signals that need **less of the output head** and **more of the model's own dynamics**. Versions v1–v4 were developed under the name **PRI**; v5 is **Internal Knowledge Veto**.

| Version | What it reads | Needs `W_u`? | Status |
|---|---|:---:|---|
| **v1** (PRI) | Token surprise coupled to a representation-space cosine rupture (multiplicative). | no | superseded by v2 |
| **v2** (PRI) | A **magnitude** of the commit-time update under a Fisher-pullback geometry. | partial | superseded by v3 |
| **v3** (PRI) | A **direction** observable at the output head — *where* the update points relative to the most decisive commit axes, independent of how far it moved. Validated across a model panel and hardened into a production calibration library. | yes | sealed; **internals & sealed parameters are not in this repo** (live in the pre-registration / paper) |
| **v4 — ACE** | Reads the **attention landscape itself** (a *pre-generation belief readout*), `W_u`-free. Spine of the current paper. | no | sealed / paper |
| **v5 — Internal Knowledge Veto** | **Residual-stream sub-layer friction** (the knowledge layer's veto of the attention route), `W_u`-free and orthogonal to the v3 sum. | no | corrected: mostly benign cancellation / residual budget |
| **v6 — Projection Veto** | **Readout-space conflict** between attention and MLP writes after projection onto frozen `W_u` contrast directions. | yes | first Qwen2.5 screen: raw +, same-Delta net null |

🔒 **On v3:** the precise direction metric, the sealed gate parameters, the geometry correction, and the per-model confirmatory numbers are deliberately **kept out of this public branch**. They belong to the frozen pre-registration and the paper. What matters for the v3→v5 story is only the shape: v3 established that a *direction*-based commit signal beats a *magnitude*-based one and is deployable via per-(model, distribution) calibration — and that motivated pushing the readout off the output head entirely (v4, and now Internal Knowledge Veto).

🏗️ **Production library (model-agnostic).** `pri_calibrator.py` fits a per-(model, deployment-distribution) profile against a labeled `.jsonl` and persists a versioned, selection-bias-corrected (nested out-of-bag bootstrap) calibration profile; `pri_detector.py` is the byte-reproducible scoring side with pipeline-hash drift checks. Any promoted v6 projection-veto signal must go through this same machinery for its real confidence intervals.

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

## 🗺️ Repo map (Projection-Veto-forward)

- **`docs/v6_projection_veto.md`** — v6 draft spec: projection-space conflict, `W_u` contrast-bank choice, same-`Δh` controls.
- **`scripts/pilot_residual_friction.py`** — the IKV / v5 veto screen (see above).
- **`scripts/analyze_friction_layer_profile.py`** — offline layer profiler.
- **`scripts/friction_residualizer.py`**, **`scripts/test_residual_friction.py`** — veto primitives + tests.
- **`pri_calibrator.py`**, **`pri_detector.py`** — production calibration + detection library (model-agnostic; the sealed-run path for any promoted signal).
- **`model_adapters.py`** — forward / hidden-state / vocab alignment across the model suite (tied-embed vs lm_head, post-embed scaling, sliding-window masks, bf16 casting) + the sub-layer capture helpers.
- **`pri_runtime.py`** — shared tracing runtime (prefix/commit hidden-state capture, single-pass forward).
- **`pri_v2_mlx_pipeline.py`**, **`pri_metrics.py`** — legacy v1/v2 metric pipeline, retained for reproduction.
- **`synthetic_logic_loader.py`**, **`synthetic_trace.py`**, **`hidden_state_collector.py`**, **`attention_contribution.py`** — MLX instrumentation stack.
- **`config.py`** — model registry, gate thresholds, numeric constants.
- **`experiments/`** — artifacts, `<slug>/<YYYY-MM-DD>/run-NN/`. `.gitignore` skips large binaries, tracks manifests + analysis JSON.

---

## 🛡️ Pre-registration discipline

This work operates on pre-registration: confirmatory tests have their parameters frozen *before* data generation, and every result carries an explicit `[OPEN] / [PASS] / [FALSIFIED] / [EXCLUDED]` status. Two standing rules:

- **No silent gate overrides.** Any behavioral-gate failure during a launch must file an Amendments entry before any threshold/skip flag is applied — this protects the pre-reg from the "just lower the bar" pattern.
- **Audit the operating point before falsifying.** Sweep the unpinned-parameter neighborhood (layer window, rank, residualizer, …) before writing any `[FALSIFIED]` tag. Localized nulls are common — IKV's Llama-3.1-8B "non-replication" was exactly one of these (a full-window mean hiding a late-layer veto signal).

Screening results (like the IKV pilot) are explicitly **not** confirmatory: they gate promotion to a sealed nested-OOB run, which is where real confidence intervals come from.

---

## 🛠️ Tools

### Commitment Tracker — `scripts/commitment_tracker.py`

Dogfoods the commitment lens on personal infrastructure: scans session transcripts for future-tense declarations, stores them in a local SQLite DB, and surfaces overdue commitments. Regex-based extraction (deterministic, zero-cost), schedulable via cron. A meta-tool — not part of the detector.

```bash
python scripts/commitment_tracker.py scan --days 7
python scripts/commitment_tracker.py check --report
python scripts/commitment_tracker.py status
```

---

## Notes

- The MLX dequantization path matches the `mlx` API on recent Apple Silicon wheels; `to_numpy` handles bf16 via an `mx.float32` cast (Gemma-3 / Qwen3 activations are bf16).
- This repository intentionally excludes the earlier semantic-uncertainty / `hbar_s` logic. The line is geometric/dynamical, not information-theoretic.

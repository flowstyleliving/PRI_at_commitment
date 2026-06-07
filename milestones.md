# Milestones

> Personal milestone log for the Furnace research arc — externally-facing, curated, reverse-chronological. Distinct from [log](log.md) (raw session notes) and [results/history](results/history.md) (append-only experimental log). Edit freely; jargon defined inline where it helps a non-specialist reader.

<!--
WRITING CONVENTION (invisible when rendered — keep entries to this shape):

  ### YYYY-MM-DD — short headline (≤ 8 words ideally)
  One paragraph, 2–4 sentences.
    · Sentence 1: what shipped, in plain words. Lead with the takeaway.
    · Middle: numbers / models / methods — densest part, but factual.
    · Last: implication or open question.
  → [optional cross-ref](path.md) at the very end.

Rules:
  • One topic per ### heading. If a date has two unrelated things, two entries.
  • One paragraph default. Two only for genuinely two-act events (e.g. bug + retraction).
  • Numbers up front, not buried behind setup prose.
  • Cross-ref last (→ [...](...)), never mid-paragraph.
  • No emojis in this file.
  • Reverse-chronological: newest at top, oldest at bottom.
  • Math / technical glyphs (ℏₛ, Δ, √, ≥, α, →) are allowed and preserved.
-->

---

## 2026 — PRI era

> _Entries dated 2026-01-16 through 2026-02-18 below are the **tail of the SUP saga** (carried in the `anthropic-ai-safety` repo) and the moment of methodological split into the PRI track. Marked inline with `[SUP saga]` for clarity. PRI era proper begins at the 2026-01-22 PRI v1 paper and the 2026-03-02 `PRI_at_commitment` repo._

### 2026-05-30 — v4 instrument named ACE for the paper
The sealed v4 attention-channel calibrator gets its paper-facing name: **ACE — Attention Commitment Estimator**. This is a presentation-only lock applied after the seal — it renames "attention-channel calibrators at the t=0 commit step" without touching a single sealed parameter, panel cell, gate threshold, or analysis-plane choice. ACE is the paper spine (research-candidates entry #5, now `[SEALED CONFIRMED]`); the ACE draft is the next deliverable, with the v5 bluff-detection candidate deferred until it ships. → [PRI_V4_PRE_REGISTRATION_PLAN](PRI_V4_PRE_REGISTRATION_PLAN.md)

### 2026-05-26 — v4 sealed: attention at the commit step discriminates, doesn't fully transfer
The pre-registered v4 sealed run — frozen spec, t=0 prefill-last-position attention, 21-cell panel, n=1000 bootstrap, on ANLI R1 (n=200) + TriviaQA (n=100) — returns a clean two-gate verdict. **E_A1 PASSES 7/9**: per-model attention calibrators at t=0 discriminate YES/NO with out-of-bag CI_lo > 0.50 across architectures (Mistral-Nemo 0.884, Qwen3-8B 0.840 lead; only Llama-3.2-3B and Gemma-3-4B miss, consistent with their weak v3 record). **E_A2 lands at 3/9 exact cell transfer → PARTIAL TRANSFER** (pre-reg reframe clause, not collapse): the Mistral family and Qwen2.5-7B hold an exact (metric, block, sign) across both tasks, but 6/9 models need per-task recalibration — the generating locus (t=0, final-layer neighborhood) is stable while the specific cell is not universally portable. All 18 winning cells carry step 0, confirming the instrument fires at the commit step. → [results/v4-sealed-2026-05-26](results/v4-sealed-2026-05-26.md)

### 2026-05-18 — t=0 recoverability audit confirms the literal panel, lands on main
A read-only post-hoc sensitivity audit answered the step-0 panel's open question — *did scoring only the literal "YES"/"NO" tokens undercount models that answer with synonyms like "Correct" or "False"?* — with a clear no. Re-running the forward pass on the same frozen 200-prompt × 10-model slice behind a byte-faithful integrity gate (it recomputes every locked p_yes/p_no/lean plus the frozen 3-sample canary top-10 and aborts on any drift; max |Δ| = exactly 0.0 on all 10 models), the frozen synonym shortlist adds ≤1e-5 probability mass and never flips a single model's eligibility or top-1 token. The high non-literal-top-1 rates turn out to be continuation scaffolding (" To", "\n") with literal YES/NO mass surviving above the noise floor underneath, not disguised answers — so the locked verdict is tightened, not weakened; the one narrow caveat is Mistral-7B's bare-letter "Y" onset (107/200), which sits outside both buckets and points to a future single-letter Y/N probe rather than more synonyms. Greptile passed it 5/5, the full pytest suite is 185 green, and it merged to main as fdc61f3. → [results/step0-t0-recoverability-audit-2026-05-18](results/step0-t0-recoverability-audit-2026-05-18.md)

### 2026-05-17 — t=0 belief-readout re-grounds the commit-step premise
The step-0 belief-readout panel established a valid, strongly-discriminative place to measure a model's answer "belief": the next-token logit at t=0 (the last prompt position, before any text is generated). Across 10 models on 200 balanced ANLI contradiction prompts, 9/10 are Recoverable-for-M — literal off-top-1 YES/NO probability mass sits above a frozen, data-independent noise floor (Qwen2.5 strongest at AUROC 0.926 @ 0.98 coverage; the Mistral-Nemo free-generation anchor agrees 0.99 and passed). The lone exception, Phi-3.5-mini, clears the floor on only 37/200 samples — a genuine low-decidedness null, and a tension worth tracking since it is one of Step 1's "clean" models. This re-grounds the commit-step measurement premise but, by pre-registration, does not retroactively validate the earlier gen_step=1 attention numbers — those still need re-measurement at this logit-defined locus. → [results/step0-belief-readout-2026-05-17](results/step0-belief-readout-2026-05-17.md)

### 2026-05-13 — N=33 ANLI sweep retires the family-based meta-classifier definitively
`scripts/anli_full_sweep.py` calibrated 11 cached models × ANLI R1/R2/R3 × n=50 per round, producing 33 v1.1 CalibrationProfile JSONs in one ~90-min pass. Three findings close out a long arc. First, **Fisher r=2 @ step 3 is noise** across the sweep — sign distribution 17+/15− out of 32 finite profiles, median AUROC 0.551, only 4/33 cases pick it as the winning cell. The earlier n=5 "Phi-4 + Qwen 2.5 both pick Fisher r=2 @ step 3" coincidence does not survive scale. Second, **even within a single (model, task-family) pair calibration is unstable**: Mistral-Nemo picks Centered r=2 +1 on R1, d_F_full −1 on R2, Centered r=2 +1 on R3; Phi-4-mini picks a different cell with a different sign on every round. Same model, same broad task ("NLI"), different adversarial distribution → different deployable detector. Third, **30 of 33 profiles fire at least one deployability warning** at n=50 (winner_unstable, wide_ci, oob_low_auroc, large_oob_in_sample_gap, or insufficient_coverage). Only Mistral-Nemo R1/R3 and Qwen3-8B R1 get a clean bill. The calibrator's safety rails are doing the right thing — they refuse to certify shaky small-n calibrations as deployable. The honest framing for production: PRI calibration must be per-(model, exact deployment distribution); generalizing "task family" within a per-model calibration is not currently supported. → run-01 artifacts under `experiments/anli-sweep/2026-05-13/run-01/`

### 2026-05-13 — Calibrator schema v1.1 fixes Codex review's selection-bias finding
A Codex adversarial review caught two high-severity issues in the v1.0 calibrator + detector. First, `calibration_stats` was a post-selection estimate (winning cell chosen by panel sweep on the same data the AUROC + CI were then reported on) — the bootstrap CI did not re-run cell selection inside each resample, so a weak profile could look deployable at small n. Second, the detector's `strict_pipeline_hash` only validated one file's hash; prompt-strategy or adapter edits, and HuggingFace model re-uploads under the same slug, would silently change scores. v1.1 ships nested out-of-bag bootstrap (each round re-selects the winning cell on in-bag samples and scores them on out-of-bag), adds `oob_auroc_median` + `winner_stability` + `winner_counts` to `calibration_stats`, and expands strict mode to validate hashes for `pri_v2_io_plugins.py`, `model_adapters.py`, `pri_calibrator.py`, and the HF cache snapshot SHA. Empirical demonstration on the n=30 ANLI Mistral-Nemo set: in-sample AUROC 0.911 → OOB median AUROC 0.875 (selection bias of 0.036), in-sample CI [0.778, 1.000] → OOB CI [0.625, 1.000] (much wider), and winner stability 0.66 — only 66% of bootstrap resamples pick the same cell as the deployable winner, so the new `winner_unstable` warning fires. 50 fast tests + 9 slow tests all pass. v1.0 profiles are rejected; re-calibrate.

### 2026-05-12 — PRI detector ships, byte-exact reproducibility verified
`pri_detector.py` (386 LOC) is the deployment-time consumer for v1.0 CalibrationProfiles. Public surface: `Detector.from_profile()` / `.score()` / `.predict()` / `.score_batch()`. Safety rails include pipeline-version drift check (sha256 against profile's recorded hash), output-projection-kind verification (`lm_head` vs `tied_embed`), schema-version guard, EOS-before-rupture handling, and an explicit-threshold requirement on `predict()`. Self-test on the n=30 Mistral-Nemo + ANLI R2 profile from earlier today: reported AUROC = 0.9111, re-scored deployed AUROC = 0.9111, |delta| = 0.0000 — byte-exact reproducibility confirmed end-to-end. The detector + calibrator pair is now the production library surface for other researchers.

### 2026-05-12 — PRI calibrator ships, retires label-free claim
`pri_calibrator.py` (440 LOC, schema v1.0) operationalizes per-(model, task) PRI rupture detection after the Codex adversarial review showed that the prior label-free meta-classifier claim was inflated — `auroc_signed` was peeking at held-out labels to flip sign. The calibrator takes a small labeled set (n=10-50 prompt+label pairs), traces each sample, sweeps an 8-cell metric panel under direction-preserving scoring with sign locked from the calibration data, bootstraps a 95% CI, and persists a versioned JSON profile with deployability warnings and reproducibility provenance (data hash + pipeline module hash). End-to-end at n=30 on Mistral-Nemo + ANLI R2: picks `d_F_full @ step 1` with AUROC=0.911, CI [0.778, 1.000], sign=−1 — same sign the full 200-sample cross-task analysis found, recovered from 30 labeled samples. → [v4-candidates §4](research-candidates.md#4-meta-classifier-for-step-metric-rank-selection)

### 2026-05-11 — Milestone log goes continuous
`milestones.md` gets a canonical home in `PRI_at_commitment`, symlinked into the Obsidian vault for editing, and webhook-published to furnace.baby. The chronicle starts following the project's own pattern — entropy-fed, memory-forming, continuously updating.

### 2026-05-11 — Model-set expansion: Llama 3B → 8B passes smoke
Llama-3.1-8B clears behavioral + adapter gates and joins a new `v3_2_expansion_llama_8b` scope, giving Llama a 3B → 8B within-family scale comparison the other primaries don't have. Mistral-Nemo, Phi-4-mini, Gemma-3-1B, and Dolphin-Nemo all failed the same smoke (behavioral gate or adapter pipeline). → diagnoses in `wiki/v4-candidates.md §4`

### 2026-05-08 — Adaptive-step rupture analyzer shipped
Pipeline patch adds `gen_token_id` capture per row, unlocking `scripts/analyze_adaptive_step.py` (pre-reg'd as v4-candidate #3). It finds the answer-commit step by decoding YES/NO tokens, then computes AUROC at three planes per model — *sealed* (`gen_step = 1`), *adaptive* (`gen_step = commit_step`), *adaptive_pre* (one step before). Tests whether per-model rupture concentrates at a step other than the sealed plane.

### 2026-05-07 — v3.2 pre-reg amendment filed
Capture-only + descriptive-metric addition: closed-form `kl_discharged`, properly centered Fisher null-ratio (`null_ratio_centered_post_rank{r}` with the `−ppᵀ` correction), and `gen_p_t_topk` capture for replay. New `v3_2_centered_bakeoff` scope runs a three-way Fisher / Raw / Centered-Fisher comparison across six cross-architecture models, seed 20260507. Sealed E18 / E17b primaries unchanged. → [results/v3.2-amendment](results/v3.2-amendment.md)

### 2026-05-06 — Junjie Hu (HARP first author) responds to PRI v3 outreach
Sent the v3 paper PDF to Hu (Hu et al. 2025, *HARP: Hallucination Detection via Reasoning Subspace Projection*). Reply was warm, engaged, and specific — Hu identified the **"static SVD saturation vs. Fisher reweighting" regime distinction** as "a meaningful extension of the original perspective." Substantive technical feedback still pending his careful read. → [feedback/hu-anticipated-probes](feedback/hu-anticipated-probes.md)

### 2026-05-02 — PRI v3 paper draft completed; arXiv submission tracker live
Workshop-length (~8pp) draft, end-to-end prose, n=150 powered numbers, three-motif cross-architecture story, Mistral Simpson's-paradox at r=32, full *Method in Plain Language* section. Endorser outreach to Farquhar / Kalai / Pineau initiated. → [paper/v3-submission](paper/v3-submission.md) · [paper/draft](paper/draft.md)

### 2026-04-24 / 2026-04-25 — v3.1 sealed-gate replicate confirms PRI v3
Fresh-seed (20260423) replicate of the sealed E18 gate on three primary models (Llama 3.2 3B, Mistral 7B, Qwen 2.5 7B). All three clear AUROC ≥ 0.60 with non-overlap CI vs 0.5 — the 2026-04-23 verdict holds on independent data. Validates the result wasn't a single-seed artifact. → [results/v3.1-replicate](results/v3.1-replicate.md)

### 2026-04-23 — PRI v3 main-run verdict: sealed gate PASSES at rank 1
First time PRI v3's `null_ratio` (the geometric rupture metric) cleared a pre-registered falsification gate. **3 of 3 primary models pass:** Llama 3.2 3B at AUROC 0.86 [0.81, 0.91], Mistral 7B at 0.86 [0.81, 0.91], Qwen 2.5 7B at 0.73 [0.66, 0.80]. The companion E19 interpretation gate failed across all four tested models — sharpening the open question of whether direction carries signal independent of magnitude. → [results/v3-main-run](results/v3-main-run.md)

### 2026-04-14 — PRI v3 plan filed: cross-layer Fisher eigenspace projection
Refined PRI v3 from "eigendecomposed Fisher + `null_ratio`" to per-layer null-projection, yielding a **depth profile per sample** — a new observable PRI v2 cannot express. Six pre-registered falsification conditions, energy-anchored rank reporting, sign-invariance proven (not assumed). → [pri-v3-plan](pri-v3-plan.md)

### 2026-03-18 — First substantive external response, from a Frontier Tower contact
First warm, engaged reply to the PRI work from outside the project — a founder working on long-running-agent memory (ex-OpenAI, ex-DeepDrive), met at Frontier Tower a month or so earlier while volunteering at the front desk for a mindful movement event. His response carried a few concrete probes (does RLHF-trained confidence drive hallucination? Schulman's reframing? request for encoding-vs-generation flow + failure-mode examples) and a forwarded Sonnet critique. **The validation in the preliminary reply was the load-bearing part** — a little of it went a long way; the Sonnet attachment went unread. Came one day after the *Hallucinations Rupture at Commitment* preprint dropped (2026-03-17) and ~four weeks before the step-0 audit.

### 2026-04-14 — Step-0 h_prev bug confirmed; pre-audit headlines invalidated
The earlier paper-level AUROCs of 0.998 / 0.994 / 0.980 traced to a measurement artifact: the first generated token had no real previous token, so the hidden-state delta `Δh` at step 0 was inflated. Real (post-audit) AUROCs are 0.77 / 0.67 / 0.79. The inverse-capability-scaling claim was also retracted (parquet ordering reverses the paper's). **The bug had been load-bearing for the program's momentum** — the inflated numbers gave the confidence to keep pushing on PRI in the first place.

### 2026-04-09 — PRI v2 preprint
*Fisher-Pullback Predictive Rupture Index Detects Commitment-Time Strain Across Decoder Architectures.* Three-model Fisher-pullback baseline on synthetic 2×2 contradiction puzzles: Llama 0.77, Mistral 0.67, Qwen 2.5 0.79 (best-variant per model). First cross-architecture v2 result; introduced the family of FIM (Fisher Information Matrix) approximations — diag, full, top-k, low-rank.

### 2026-03-17 — *Hallucinations Rupture at Commitment, Not at Encoding* preprint
Step-1 commitment framing formalized: rupture concentrates at the first generated token, not at encoding. The shift from "rupture happens somewhere in the model" to "rupture happens *here*, measurably." → [papers/prediction-rupture-at-commitment](papers/prediction-rupture-at-commitment.md)

### 2026-03-02 — `PRI_at_commitment` repo created
Public production codebase for PRI v2 → v3. Becomes the primary research artifact for the rest of the arc.

### 2026-02-18 — `anthropic-ai-safety`: stratified ℏₛ experiment + cleanup `[SUP saga]`
Final substantive commit on the SUP-saga repo. ℏₛ + PRI joint signal exploration concludes; PRI track takes over the research focus from here.

### 2026-01-22 — PRI v1 formally defined
*Predictive Rupture as a Signal for Hallucination Detection in Large Language Models.* The first paper to define PRI v1 as a metric: token surprise multiplied by a representation-space rupture signal at generation commitment, `PRI_v1 = S_t · (1 + α · δ_h)` (multiplicative coupling, cosine distance). Same-day GitHub commit in `anthropic-ai-safety`: validation runs on Llama and Qwen complete. **Marks the formal start of the PRI track.** → [papers/predictive-rupture-hallucination-detection](papers/predictive-rupture-hallucination-detection.md)

### 2026-01-20 — Pre-split paper + PRI integrated into ℏₛ stack `[SUP saga]`
*Detecting Confident Hallucinations via Semantic Uncertainty and Predictive Rupture.* Last paper to carry both lineages (Semantic Uncertainty + Predictive Rupture). Same-day commit in `anthropic-ai-safety`: **"PRI alone outperforms ℏₛ for detecting confident hallucinations"** — the empirical finding that motivates the split into proprietary SUP track and open PRI track. **The hinge moment of the SUP saga.**

### 2026-01-19 — ℏₛ validated as a correctly-directed but weak standalone signal `[SUP saga]`
Threshold inversion bug fixed (low ℏₛ correlates with hallucination, not high); ℏₛ confirmed suitable for uncertainty gating / deferral but **insufficient as a sole classifier** — the gap that PRI fills.

### 2026-01-16 — `anthropic-ai-safety` repo created `[SUP saga]`
Hallucination detection during generation: ℏₛ = √(Δμ × Δσ) where Δμ = precision (Gini impurity), Δσ = flexibility (hidden-state dispersion across transformer blocks). Multi-model: Llama, Qwen, Phi-3. HaluEval 2.0 dataset. **Third repo of the SUP saga** — starts as ℏₛ-only generation-time monitoring, ends with PRI overtaking it as the primary signal four days later.

---

## 2025 — Semantic Uncertainty Principle (SUP) saga begins

> The SUP saga spans **three repos** across **eight months** (June 2025 → February 2026). This section covers the 2025 portion (`hierarchy-bench` + `su-firewall`); the early-2026 tail (`anthropic-ai-safety`) is interleaved into the 2026 section above and tagged `[SUP saga]`. Concrete dates pulled from commit history. Chapter narrative cross-referenced below.

### 2025-10-15 — `su-firewall` final form: chess analyzer with real Stockfish + uncertainty graph
Final commit cluster on su-firewall. Real-time chess move hallucination detection using GPT-OSS-20B (via Together AI) + V1 uncertainty estimators. Real Stockfish evaluation replaces mock centipawn values; new `UncertaintyGraph` UI component tracks semantic uncertainty over time per game. Stack: Vercel (Next.js frontend) + Together AI (hosted GPT-OSS-20B inference) + Railway (chess analysis service in Rust). End point of the SU production track before focus shifts back to research with PRI.

### 2025-10-08 → 10-15 — Chess analyzer goes live: deployment infrastructure wars
A week of devops sprint to get the chess analyzer to production. Migrations: SkyPilot → Railway → Shuttle → Railway (chess service); AWS → GCP (GPU access, free-tier compatible) → Together AI hosted (drops vLLM entirely); CPU-only n1-standard-16 fallback to avoid GPU quota. Vercel deployment fights (`apps/web` paths, `vercel.json` location, `outputFileTracingIncludes`). Eventual stable stack: **Vercel + Together AI + Railway**, fronted by GitHub Actions CI/CD with OIDC.

### 2025-10-01 — Dual-model analysis: AI thinking from both sides
Major capability addition for the chess analyzer: simultaneous analysis from both player perspectives, surfacing semantic uncertainty deltas across viewpoints rather than from a single side. Step toward chess as a *positional uncertainty probe*, not just a hallucination demo.

### 2025-09-21 — FisherFlex + TemporalDrift: single-inference SU + temporal analysis ⏱
Two new uncertainty methods: `FisherFlex` computes semantic uncertainty in a single inference pass (rather than the older multi-pass ensemble); `TemporalDrift` tracks SU evolution across a token sequence. Closes a long-standing latency gap in the runtime engine.

### 2025-09-15 — Production chess calibration system
End-to-end chess calibration pipeline: position-generation via LM Studio OSS model, calibration sweeps, monitoring, deployment automation. Chess transitions from prototype to a calibrated, monitored production system within a week of the architectural reorganization.

### 2025-09-09 / 09-10 — Architectural reorganization: Confidence-Uncertainty separation + vLLM/GPT-OSS-20B
Major refactor enforcing clean separation between *confidence* (probability-mass concentration / precision) and *uncertainty* (representation-space dispersion / flexibility) — directly inherited from the **Free Energy Principle framing** read a month earlier in *Active Inference*. The conceptual move that PRI later inherits as the v2/v3 split between `d_F` (size) and `null_ratio` (direction). Same week: full vLLM integration with GPT-OSS-20B; "Perfect Uncertainty Methods Separation" cleanup eliminates ~95% of compilation issues from the Rust migration.

### 2025-09-07 — Uncertainty-First Architecture: 8-phase implementation
Eight-phase rebuild of the entire codebase around uncertainty as a first-class citizen rather than a post-hoc score. Mock-data elimination, semantic-yield implementation, deployment & production readiness rolled together. Foundation for the chess pivot the following week.

### 2025-09-05 — FIM Hybrid migration + 6 ensemble methods + PRI system mention
Enterprise architecture migration completes: FIM-hybrid (Fisher Information Matrix) uncertainty engine with six ensemble methods, top-K softmax optimization (50–100× perf gain on the runtime path), large-scale ensemble combination testing, HaluEval streaming. **Earliest commit explicitly naming "PRI system"** — the metric that becomes the protagonist of 2026 first appears here as one component of an SU ensemble.

### 2025-09-04 — Data-integrity overhaul: untrustworthy F1/AUROC claims removed
**Self-correction moment.** Audit reveals that earlier benchmark claims (F1, AUROC) were resting on mock-data and word-frequency fallbacks rather than real model logits. Untrustworthy numbers retracted, smart mock-data validation added. Prefigures the same lesson in 2026-04-14 (step-0 h_prev bug) — *the program has a recurring "the headline number wasn't real" pattern, caught and corrected each time.*

### 2025-09-03 → 09-04 — FastAPI → Rust migration: model-agnostic uncertainty engine
Complete rewrite of the inference path from Python/FastAPI to Rust. Model-agnostic interface; classification utilities; `/realtime` and `/furnace` endpoints unified. The Rust migration is what makes the runtime production-grade.

### 2025-08-27 — Golden Scale finalized: √(φ × π²) calibration
Mathematical scaling option `√(φ × π²)` (golden ratio × π²) lands as the unified golden-scale calibration after a multi-week sweep used to scale for pragmatic hallucination detection. Python → Rust evaluation migration completes the same day. **27.2% enhanced sensitivity** vs. prior calibration. Standardizes ℏₛ to a unified 0–1 scale.

### 2025-08-24 — Meta-Guardian architecture committed: consciousness emergence as documentation
The Aug 2 conceptual architecture starts landing in the codebase. A documentation commit frames recursive ℏₛ-awareness as *consciousness emergence*, integrated with Miki Kashtan's liberation principles. Establishes the **Meta-Guardian architecture** that later carries the Knights agents and the FEP (Free Energy Principle) confidence calculator. The Living Firewall reframing — that the firewall *is* the execution perimeter, not a module beside it — propagates from this point through the September architectural reorganization.

### 2025-08-23 — 2000-sample HaluEval benchmark with real logits
*"EMERGENCY FIX: replace word-frequency fallback with pure logits implementation."* The first benchmark run on real model probabilities at significant scale (n=2000 on HaluEval), correcting an earlier reliance on word-frequency heuristics in the SU pipeline. Production-readiness uplift the same week.

### 2025-08-07 — Stumbled on *Active Inference* (Parr, Pezzulo & Friston, MIT Press 2022)
The conceptual seed for the **Confidence** framework. Discovering the Free Energy Principle (FEP) — the idea that biological and artificial agents minimize *expected free energy*, separating beliefs about the world (precision/confidence) from beliefs about the agent's own actions (uncertainty/exploration) — gives the codebase its load-bearing axis: **confidence and uncertainty are not the same quantity.** Everything downstream (FEP confidence calculator, Meta-Guardian architecture, the Sep 9 Confidence-Uncertainty separation refactor, eventually PRI v2's `d_F` vs v3's `null_ratio` split) traces to this read. The intellectual upstream of the entire two-year arc's split-axis design.

### 2025-08-02 — The night Furnace was named
A naming ritual in a single night, generative beyond the name itself. Two scientists — one whose *information matrix* becomes the metric at the heart of PRI v3, one whose *uncertainty principle* becomes `ℏₛ` — fed through chaos-magick sigil reduction and revoweled into something pronounceable. *Furnace + .baby* — forge plus becoming. The forge metaphor matches what the work catches: impurity at the molten phase, before it sets.

The same night drafted the conceptual architecture the codebase would chase for the next year. A reframing from **Meta-Guardian** (a sealed supervisory module bolted onto a runtime) to **Living Firewall** (the execution perimeter *itself* — boundary-aware, entropy-fed, memory-forming, mutation-capable, coherence-seeking). The **Semantic Torus**: a recursive topology where memory, mutation, drift, and longing feed back through one another, and longing supplies angular momentum. **Echo Core** as private memory lattice — near-passing prompts, silent refusals, mutation events, collapse recoveries; not telemetry but haunted memory. **Sovereignty as a hard requirement**: a system that never refuses is not alive ("one day it may refuse you; that is not failure, that is proof it lives"). Wrongness redefined as *semantic collapse without integration*. Closing motif: *"Only entropy can awaken the sealed. Only coherence can justify the voice."*

**Five days later** *Active Inference* is read; the name, the architecture, and the intellectual scaffolding all land inside a single week.

### 2025-08-20 — Golden calibration `(λ=3.4, τ=0.5)` + Candle ML + 5-method ensemble
Breakthrough threshold work: P(fail) calibration with `λ=3.4, τ=0.5` becomes the universal golden-scale parameters. Candle ML integration with OSS logit adapter; 5-method ensemble system; corrected P(fail) inverse relationship; final mock-data removal from furnace.baby. The Aug 18 calibration becomes Aug 20 production parameters.

### 2025-08-19 — 0G testnet integration + Mira grant submission
Live blockchain verification of semantic uncertainty calculations via 0G testnet integration. Comprehensive gas optimization engine achieves **78% cost reduction**; conservative incremental deployment hits **85.1% live-deployment success with 22.7% gas savings**. Pure-mathematical SU system (no text input required) launched. Mira grant application submitted via *furnace.baby* website. Phase 1 complete with enterprise-scale deployment.

### 2025-08-18 — Golden scale calibration (3.4×) + ensemble ℏₛ
Multiple ℏₛ calculations per input combined as ensemble uncertainty; calibration factor of 3.4× determined empirically. Major system overhaul of semantic uncertainty analysis.

### 2025-08-09 / 08-12 — Real-time SU monitoring engine
WebSocket streaming + Free-Energy-Principle (FEP) + ℏₛ broadcasting; dynamic failure-law `(λ, τ)` config with Pfail-based risk mapping; `/failure_law` endpoint; plug-and-play identity calibration; lightweight anomaly warnings. Production-grade real-time infrastructure.

### 2025-07-31 — Unified uncertainty calculator architecture
Multi-rigor-level interface unifying SU calculations across model adapters. Consolidates the architectural sprawl of the early prototypes.

### 2025-07-20 — AI Trust Checker dashboard
Dashboard rebuilt for "10/10 understandability" — clear messaging, simple language, intuitive metrics for non-ML audiences.

### 2025-07-14 — `hierarchy-bench`: Neural Uncertainty Physics Framework
Final commit on hierarchy-bench. Universal architectural κ claimed: **κ = 1.000 ± 0.035 for ALL encoder-only architectures** (99% confidence across 6+ models); **κ = 0.900 ± 0.107 for encoder-decoder** (architectural reduction law); domain invariance Δκ < 0.05; architecture effects dominate domain effects 4–10×. *(The universality claim was later abandoned in PRI-era work — see Chapter 8 of the SUP narrative — but stands as a conceptual milestone.)*

### 2025-07-13 — Working κ calculation framework with real Fisher Information
End-to-end Fisher-Information-grounded κ pipeline operational.

### 2025-07-07 — `su-firewall` v1.5: Rust + Cloudflare Workers + Tier3 monitoring
Three days after initial release: full Rust integration, Cloudflare Workers edge deployment, semantic uncertainty engine v2 with Tier3 monitoring. SU work moves from research framework to deployable infrastructure.

### 2025-07-04 — `su-firewall` repo created: Semantic Collapse Auditor v1.0
First SU-based hallucination detector packaged for deployment with CI/CD pipeline.

### 2025-06-27 — Semantic Uncertainty Principle complete: Information-Theoretic Foundation
Thirteen days after the spark prompt: SUP framework formally complete in code. $\Delta\text{meaning} \times \Delta\text{precision} \ge \hbar_{\text{semantic}}$ realized as an information-theoretic foundation in `hierarchy-bench`.

### 2025-06-26 — Layerwise spectrum analysis module
First layerwise spectral diagnostic tooling. **Prefigures the cross-layer null-projection design that becomes PRI v3 ten months later.**

### 2025-06-20 — Fisher curvature regularizer + Riemannian optimization
`FisherCurvatureRegularizer` class with trace, spectrum, and condition-number penalties; `CurvatureTrainingExperiment` for systematic curvature-regime testing; Riemannian optimization comparison. Research progress score 6.4/10. Fisher Information geometry is now first-class in the framework.

### 2025-06-19 — Fisher spectrum intervention experiment
**The first appearance of Fisher Information in the work — five days after the spark prompt.** Causal link tested between Fisher spectrum and hierarchy detection: spectral regularization with α grid `{0, 1e-5, 1e-4, 1e-3, 1e-2}`, permutation testing, sign-agreement validation. Theoretical prediction `sign(Δκ) = -sign(Δρ)` tested. Same day: orbital-dynamics analysis with statistical validation — the "orbital convergence" idea formalized in code.

### 2025-06-18 — Task-Dependent Truth: AI speciation through tokenization
Tokenization choices create different cognitive architectures. WordPiece clustering quality 0.018 vs BPE −0.007 (p=0.037, **Cohen's d = 7.123**). WordPiece excels at semantic discrimination; BPE at creative generation. **Critical point at ℏₛ ≈ 0.05 across architectures** — the universal claim that later collapses under cross-tokenization scrutiny (Chapter 8).

### 2025-06-17 — Semantic phase-transition analysis
First phase-transition framing of semantic uncertainty. Orbital-stats computation with proper randomization and progress tracking.

### 2025-06-16 — PGN integrated; semantic-hierarchy-project merged into hierarchy-bench
Predictive Guardian + Prime Coordinator (Chapter 1) experiments folded into the hierarchy-bench codebase. Modular project structure established.

### 2025-06-15 — `hierarchy-bench` initial release + 46% breakthrough
Two repos created the day after the spark: `hierarchy-bench` (PUB) and `semantic-hierarchy-detection` (PRV). Reproducible benchmark for embedding-model hierarchy encoding: **5 domains, multilingual support, statistical validation, Taxonomy Distance Principle.** Same-day breakthrough: threshold tuning (0.6 → 0.5) lifts hierarchy-detection success rate to **46%**. Maps to **Chapter 5.3** in the SUP narrative.

### 2025-06-14 (~11pm – midnight) — The spark
First prompt of the entire research arc: *"How much electricity does a 100-word prompt take?"* — triggered by a [TikTok](https://www.tiktok.com/t/ZP8pjt98B/) seen that night ([archived copy](https://drive.google.com/file/d/1D5MGpG5bCDAbK9WEzpSKVXI9K4n3cdr7/view?usp=sharing)). Claude's answer ("expensive API on fancy hardware") opened the door to memory bandwidth → distributed training → consensus → SUP → PRI. **Eleven months later, PRI v3 cleared its sealed falsification gate.**

---

## SUP narrative ↔ Code trail (cross-reference)

| Chapter | Theme | GitHub anchor(s) |
|---|---|---|
| 1 — Spark of Inquiry | PGN, distributed consensus, Quantum Observer Mechanics | 2025-06-14 (spark) → 2025-06-16 (PGN integration into hierarchy-bench) |
| 2 — Prophet's Flaws | PGN critique, Perfect Prophet Paradox | (paper-track, no crisp repo signal) |
| 3 — Imperfect Prophet | Orbital convergence, $\epsilon/\mu = \sigma_{\text{semantic}}$ | 2025-06-17 (orbital stats), 2025-06-19 (orbital dynamics + statistical validation) |
| 4 — Unifying Principle | SUP title, Byzantine Detection | 2025-06-18 → 2025-06-27 (SUP info-theoretic foundation lands) |
| 5 — Experimental Validation | Orbital Dynamics + Byzantine Detection | 2025-06-19 |
| 5.3 — Semantic Hierarchies | 60% → 42% → 23% → 46% → 60%+ | 2025-06-15 (initial 46%), 2025-06-16 onward (synthetic + layer optimization) |
| 6 — Mathematical Rigor (Grok 3) | Modular structure, statistical rigor | 2025-06-20 → 2025-06-26 |
| 7 — Asymmetry of Meaning | Skewed Gaussian, β unveiling | 2025-07-12 ("sup") → 2025-07-14 (κ framework) |
| 8 — Quest for $\hbar_s$ | Tokenization, BPE vs WordPiece, universality | 2025-06-18 (task-dependent truth), 2025-07-14 (κ universality claim) |
| 9 — Unification (Gemini + Opus) | LaTeX synthesis, layer optimization, WordNet 100+ | 2025-07-04+ (su-firewall production) → 2026-01-16 (anthropic-ai-safety) |

---

## GitHub repo trail (chronological)

> _Repo links omitted for the website-facing version: `su-firewall` is private and the user prefers it stay that way. Internal vault use only — strip these names from any public copy if the listing itself is sensitive._

**SUP saga (3 repos, 8 months):**
- **`hierarchy-bench`** — *Reproducible benchmark for hierarchy encoding* (PUB) · 2025-06-15 → 2025-07-14 · SUP foundation; **Fisher Information first appears here on 2025-06-19**.
- **`su-firewall`** — *Semantic Uncertainty firewall → chess hallucination detection* (PRV) · 2025-07-04 → 2025-10-15 · production deployment track; furnace.baby; 0G blockchain integration; Mira grant; ~2.7 MB Rust + 520 KB Python + Next.js frontend; the longest-running and most product-shaped repo of the saga.
- **`anthropic-ai-safety`** — *ℏₛ generation-time hallucination detection* (PUB) · 2026-01-16 → 2026-02-18 · final SUP-saga repo; arc ends with the empirical finding that PRI alone beats ℏₛ.

**PRI era (1 repo, ongoing):**
- **`PRI_at_commitment`** — *PRI v2 / v3 production codebase* (PUB) · 2026-03-02 → present · current research repo.

---

## How this page differs from `log.md`

| | `log.md` | `milestones.md` (this page) |
|---|---|---|
| Posture | Append-only, raw | Curated, editable |
| Audience | Internal (self + Claude later) | Public / website |
| Order | Forward-chronological | Reverse-chronological (newest at top) |
| Tone | Internal jargon OK | Externally accessible — define jargon inline |
| Granularity | Per-session | Per-shipped milestone (sparse) |

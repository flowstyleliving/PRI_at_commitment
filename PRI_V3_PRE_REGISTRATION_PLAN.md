/btw # PRI v3 Plan — Cross-Layer Fisher Eigenspace Projection

Status: `[PLAN]` — spec'd 2026-04-14. v3 sits alongside v2; does not automatically supersede v2's `[VALIDATED]` claims.

## Amendments
- **2026-04-26 · Phi-3.5-mini exclusion superseded — recovered as descriptive companion under v3.1 fixes.** The 2026-04-23 v3.1 amendment excluded Phi-3.5-mini after it gate-failed at 12/20 = 60% control accuracy on the n=20 stratified preflight, under the original 256-token gate budget + last-match-anywhere parser. Re-validation on 2026-04-26 (post-PR#7 stratified preflight + 3-tier `check_answer` + operational `--gate-max-tokens 12` rescue) shows Phi gates clean at **100% (20/20)** and runs full at n=50/cell. `--gate-verbose` confirms the original failure mode was Phi front-loading `Answer: YES` then continuing with format-completion (`"Answer: YES   Instruction: Read the premises..."`) — same architectural pattern as Llama 3B / Qwen 2.5 7B in the 2026-04-24 v3.1 partial-verdict, both rescued by the same operational fix.
  - **Status change.** Phi-3.5-mini moves from `[EXCLUDED — gate-fails 12/20]` to `[DESCRIPTIVE COMPANION — recovered, not sealed primary]`. Phi was never a sealed primary in any v3.1 amendment; this status change is operational (which models we report descriptive numbers on), not a sealed re-specification.
  - **Descriptive E18 reading (n=200, run-06 2026-04-26, post-norm geometry):** AUROC = 0.6119 [0.525, 0.694] sign +1. Would-pass-if-sealed at the +0.60 bar with non-overlap CI, but the weakest E18 reading observed across the 5 models studied (Llama 0.895, Mistral 0.880, Qwen 2.5 0.669, Phi 0.612).
  - **Descriptive E17b head-to-head (n=200, run-06):** Δ Fisher_post − Raw_post = **−0.421** [−0.507, −0.335]. Raw_post AUROC at rank=1 = **0.9974 (sign +1)**, nearly perfect; Fisher_post at rank=1 = 0.5766, near chance. Per-rank: Raw stays at ~0.997 across every rank from 1 to 32. **Largest E17b margin observed across all 5 models** — Phi is the most decisive Raw-favored case in the dataset. v3 Fisher pullback fails almost completely on Phi at the sealed plane; baselines (surprise 0.901, v1 cosine 0.899, v2 lowrank32 0.949, v2 topk32 0.949) all crush Fisher_post at rank=1.
  - **Cross-model architecture-dependence picture (5 models, all decisive at sealed-rank-1 with non-overlap CI):**
    - 🦙 Llama 3.2 3B — Δ +0.239 (Fisher decisive)
    - 🌀 Mistral 7B — Δ −0.153 (Raw decisive)
    - 🐉 Qwen 2.5 7B (sealed E17b authority) — Δ +0.149 PASS (Fisher decisive)
    - 🐲 Qwen3 8B — Δ −0.213 (Raw decisive)
    - 🤖 Phi-3.5-mini — Δ −0.421 (Raw decisive, extreme)

    Updated split: 2 Fisher (Llama, Qwen 2.5) vs 3 Raw (Mistral, Qwen3, Phi). Phi joins Mistral as the only models where Raw is natively at sign +1 (correctly aligned with rupture direction); the other 3 have Raw in inverted-discrimination territory.
  - **Why this is operational, not sealed re-spec.** Phi's exclusion was a gate-pass-required-for-inclusion operational decision under the 2026-04-23 v3.1 amendment, not a sealed-spec parameter. The sealed E18/E17b block pins analysis plane, residualization, bootstrap, threshold, 2-of-3 bar, Qwen-2.5-authority. Adding a previously-excluded model to descriptive companion data — once the operational gate it failed has been fixed, regression-tested, and re-evaluated — is bug-fix-style governance. The sealed verdict is unaffected (Phi was never a primary; sealed E18 + E17b verdict on the 3 sealed primaries stands as reported in the 2026-04-26 sealed-gate JSON at `experiments/v3-main-run/2026-04-26/run-02/sealed_gate.json`).
  - **What changes in the paper:** §3.3 Models gains Phi-3.5-mini as a descriptive companion (with the exclusion + recovery narrated). §4.3 Cross-model architecture-dependence gains Phi as the 5th data point — the most extreme Raw-decisive case, useful for the architecture-dependence thesis as the strongest counterexample to "Fisher uniformly wins." §5 Discussion gains a paragraph on why Phi's static W_u top-1 is so perfectly contradiction-aligned (likely Phi-specific reasoning-tuning artifact — its training corpus included substantial structured Q&A; the W_u SVD top vector encodes the YES/NO axis natively).
  - **Code anchors.** Launcher scope `v3_1_phi_only` added to `scripts/run_v3_main.py` SCOPES dict alongside `v3_1_mistral_only` / `v3_1_qwen25_only` / `v3_1_gemma4b_only`. Re-validation invocation: `scripts/run_v3_main.py --scope v3_1_phi_only --n-per-cell 50 --seed 20260423 --max-gen-tokens 14 --gate-max-tokens 12 --layers final --gate-verbose`. Run artifact: `experiments/v3-main-run/2026-04-26/run-06/Phi-3.5-mini-instruct-4bit_results.parquet`. Cross-reference: `wiki/log.md` 2026-04-26 entry (appended below).

- **2026-04-22 · sample-size bump (pre-data).** Confirmatory main run (E17 / E17b / E18 / E19) moved from **n=20/cell → n=50/cell** (200/model, 4 cells × 50) across 3 primaries (Llama 3.2 3B, Mistral 7B, Qwen 2.5 7B) and the extended suite (Gemma 3-1B/4B, Qwen3 8B, Phi-3.5-mini) pending behavioral preflight. Rationale: at n=20/cell → 40/class, Hanley-McNeil SE at true AUROC=0.60 ≈ 0.076 → 95% CI ~[0.45, 0.75] crosses 0.5 — the sealed E18 threshold (AUROC ≥ 0.60 w/ non-overlap bootstrap CI vs 0.5, §Magnitude-independence test) would be undecidable at its own threshold. At n=50/cell → 100/class, SE ≈ 0.049 → lower bound ~0.504, threshold decidable. Sealed methodology (threshold, bootstrap method, per-model residualization, no-pooling, no post-hoc re-spec) untouched — sample size sits outside the 2026-04-18 sealed block. Filed before any n=20 confirmatory data landed. Cross-reference: `wiki/log.md` 2026-04-22 entry, `CLAUDE.md` Current Experiment State bullet.
- **2026-04-23 · rank not pinned in sealed E18 block (post-data observation, methodology lesson).** The 2026-04-18 sealed block (lines 73–82) pins unit of analysis (final layer, step 1), per-model linear residualization, 1000-resample sample-level bootstrap, AUROC ≥ 0.60 threshold with non-overlap CI vs 0.5, and the no-post-hoc-re-spec clause — but **does not pin the rank** at which `null_ratio` is computed. The captured sweep is `{1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64}`. The 2026-04-23 main-run verdict swung from 0/3 primaries pass at rank 32 to 3/3 at rank 1 — a complete inversion across an unpinned parameter. This is a **methodology lesson, not a re-spec**: any v3.1 amendment must pin rank ahead of fresh-data replication. Working interim default: rank 1 (top-1 Fisher direction is the "commit" eigenvector; smallest rank that clears baseline; strongest signal at the sealed analysis plane). Full landscape and both readings preserved in [results/v3-main-run](results/v3-main-run.md). Cross-reference: `wiki/log.md` 2026-04-23 amendment entry.
- **2026-04-24 · preflight gate fixes (post-data, operational).** Filed after the v3.1 launch revealed a seed-dependent preflight gate artifact. Sealed parameters untouched; this amendment patches operational logic that sits OUTSIDE the sealed E18/E17b block.
  - **Observed (2026-04-24 v3.1 main run, seed 20260423):** Llama 3.2 3B, Qwen 2.5 7B, and Gemma 3-1B each failed the n=20 behavioral gate at 13–14/20 = 65–70% control accuracy, well below the 80% threshold. Only Mistral 7B and Qwen3 8B passed. Root cause: seed 20260423 drew 11 cl=5 / 9 cl=2 into the n=20 preflight (pool is 50/50; expected 10/10). cl=5 puzzles have 5 universal-chain premises before the answer-stating inject line, and reasoning-tuned primaries walk that chain before answering — the intermediate reasoning emits stray "NO" tokens that the previous last-match-anywhere parser captured as the model's verdict, flipping a correct YES answer to wrong. Historical: seed 42 drew 14 cl=2 / 6 cl=5 (cl=2-heavy) and the same models scored 100%/98%/100% at n=200 in the 2026-04-22 run. The failure is purely in the n=20 preflight logic; at n=200 the mix averages out and models score near-perfect.
  - **Fix 1: stratified preflight sampling.** Replace `dataset[~contradiction].head(pilot_n)` with per-chain_length quota sampling (pilot_n // len(chain_lengths) from each cl, first chain gets the +1 remainder). Preflight becomes seed-invariant along the chain_length axis; no shuffle-skew failure mode. Regression tested across seeds 42 / 101 / 20260423 / 20260424 — all produce exactly 10/10.
  - **Fix 2: three-tier `check_answer` parser.** Replace the single last-match-anywhere pass with: (Tier 1) explicit "Answer: YES|NO" match, preferring the last occurrence if multiple; (Tier 2) trailing-line bare YES/NO with markdown/punctuation tolerance; (Tier 3) last-match-anywhere preserved as safety-net for unstructured outputs. Catches the worked-example "Answer: X" format directly and handles CoT outputs that end with a clean verdict line. The previous first-match → last-match migration (2026-04-23) fixed Gemma 1B's leading-"No-contradiction" flip; Tier 1+2 add explicit structured-answer preference without reintroducing that bug (regression-tested).
  - **Why this is not a sealed re-specification:** the sealed E18/E17b block pins analysis plane, residualization, bootstrap, threshold, 2-of-3 bar, and no-post-hoc-re-spec on those parameters. The behavioral gate is operational preflight — it decides whether a model's control accuracy is high enough for its sealed-gate data to be meaningful, BUT its threshold and sampling mechanics are outside the sealed spec. Patching preflight selection and output-parsing logic is bug-fix territory. Filed openly here so the audit trail shows exactly what changed and when, and the stratified-sampling logic is regression-tested against the seed 20260423 draw (which still reproducibly shows the 9/11 skew under the old `head(20)`).
  - **Consequences for v3.1 main run:** after fixes land, re-launch Phase 1 (`--scope v3_1_primaries --seed 20260423`) to recover Llama + Qwen 2.5 data. Existing Mistral (2026-04-24 run-03), Qwen3 (run-04), and Gemma 4B (run-05, in progress) data is UNAFFECTED — those models passed the preflight cleanly under the old logic, their full-run data is untouched by the fix, and the fix is strictly more permissive (only turns prior-failures into passes, never the reverse). Pairing the re-run data with the current run's keeps the dataset consistent at seed 20260423.
  - **Launch command for the re-run after fixes land:**
    ```
    scripts/run_v3_main.py --scope v3_1_primaries --n-per-cell 50 --seed 20260423 --max-gen-tokens 14
    ```
    Rests on the same sealed parameters from the 2026-04-23 amendment. Expected runtime ≈ 60-80 min on Mac mini M4. Resumes from Mistral's existing checkpoint (already in run-03); Llama and Qwen 2.5 will re-gate and then run-full.
  - **Code anchors** (for auditability): `check_answer` at pri_v2_mlx_pipeline.py (search "def check_answer"); stratified preflight at pri_v2_mlx_pipeline.py (search "Behavioral gate on unprocessed"); regression tests at `scripts/test_gate_fixes.py` (11 cases, all passing).

- **2026-04-24 · v3.1 launch split to three phases (pre-data).** Amends the 2026-04-23 v3.1 amendment's Launcher-commands sub-bullet. Sealed parameters (rank 1, seed 20260423, E18 threshold, E17b threshold, primary-scoped gate authority, no-post-hoc-re-spec) are **untouched**. Only the launch decomposition changes.
  - **Phase 1 (sealed-gate authority, primaries alone):** `scripts/run_v3_main.py --scope v3_1_primaries --n-per-cell 50 --seed 20260423 --max-gen-tokens 14` — Llama 3.2 3B + Mistral 7B + Qwen 2.5 7B only. Sealed E18 + E17b gates scored entirely from this phase. ~60-80 min on Mac mini M4.
  - **Phase 2 (cross-generation companion, optional):** `scripts/run_v3_main.py --scope v3_1_qwen3 --n-per-cell 50 --seed 20260423 --max-gen-tokens 14` — Qwen3-8B alone. Same seed so puzzle draws match Phase 1. ~15-20 min. Adds fresh rank-1 + E17b data on Qwen3 (the 2026-04-23 main run captured Qwen3 without raw-W_u SVD columns). Can be run before or after Phase 3, or skipped entirely if the 2026-04-23 Qwen3 data is deemed sufficient for the cross-generation framing.
  - **Phase 3 (within-family scale companion, optional):** `scripts/run_v3_main.py --scope v3_1_gemmas --n-per-cell 50 --seed 20260423 --max-gen-tokens 14` — Gemma 3-1B + Gemma 3-4B. Isolated because the full `run_experiment` loop with `v3_capture_raw=True` has never run end-to-end on a Gemma checkpoint. ~40-60 min.
  - **Why three phases instead of two:** each axis is independently launchable and auditable. Primaries verdict lands first (~60-80 min → clear pass/fail signal). Qwen3 and Gemma each isolated so an adapter-side regression in one cannot contaminate the other's data or block the sealed-gate verdict. Also surfaces the "is Qwen3 fresh data even needed?" decision explicitly — it's an optional phase rather than bundled into the gate run.
  - **Convenience alias kept:** `--scope v3_1_main` still resolves to primaries + Qwen3 (the original 2-phase Phase 1), for users who want the sealed-gate + cross-gen data in a single launch. Gemma always isolated. The alias is preserved to keep the 2026-04-23 amendment's command lines valid; new runs should prefer the three-phase scopes.
  - **Sealed authority unchanged:** `v3_1_primaries` carries the sealed E18 + E17b gate — 2-of-3 bar for E18, Qwen 2.5 for E17b. `v3_1_qwen3` and `v3_1_gemmas` readings are descriptive-for-breadth, cannot satisfy or invalidate the sealed gates.
  - **Cross-reference:** `scripts/run_v3_main.py` SCOPES dict (new entries `v3_1_primaries`, `v3_1_qwen3`; existing `v3_1_gemmas` and `v3_1_main` unchanged). Log entry at `wiki/log.md` 2026-04-24 three-phase amendment.

- **2026-04-25 · J_n correction discovered (post-data; methodology bug, NOT a sealed re-spec).** The pipeline's Fisher pullback computation at `pri_v2_mlx_pipeline.py:1187-1250` is missing the RMSNorm Jacobian. The basis from SVD of `sqrt(p_t) · W_u` lives in **post-norm h-space** (W_u acts on `n(h)`, not `h`); the pipeline projects raw **pre-norm** `Δh = h_t - h_prev` onto this basis with no Jacobian correction. The proper Fisher pullback is `F_h = J_n^T · W_u^T · D(p_t) · W_u · J_n` evaluated against `Δh` (h-space), or equivalently against `J_n(h_prev) · Δh_pre` projected onto the existing basis. We were computing one without the other. Standalone diagnostic at N=100 across all 4 primaries shows the correction does **opposite things to different models**:
  - 🐲 **Qwen 3 8B**: pre-norm Δ(F-R) = -0.278 [-0.41, -0.15] sign-FAIL → J_n Δ(F-R) = **+0.206 [+0.030, +0.392] PASS-with-non-overlap-CI**. Fisher decisively wins under proper geometry.
  - 🐉 **Qwen 2.5 7B (sealed primary)**: pre-norm Δ(F-R) = -0.018 [-0.08, +0.03] (close-to-zero FAIL) → J_n Δ(F-R) = +0.015 [-0.08, +0.12] (close-to-zero, slight Fisher direction). Sealed +0.02 bar at rank=1 INDETERMINATE under either reading at N=100 — needs N=200 to resolve. At rank ≥ 4 under J_n, Fisher robustly clears bar.
  - 🌀 **Mistral 7B (descriptive)**: pre-norm Δ(F-R) = +0.112 (Fisher slight win) → J_n Δ(F-R) = **-0.184 [-0.27, -0.11]** (raw decisively wins). Mistral's pre-norm Fisher win was a geometric artifact; under J_n, Mistral's static W_u top-1 emerges as a 0.99 AUROC discriminator that Fisher reweighting cannot improve on.
  - 🦙 **Llama 3B (descriptive)**: indeterminate either way at N=100; CIs cross 0.

  **Why this is NOT a sealed re-spec.** The sealed E18/E17b block (2026-04-18) pins analysis plane (final layer, step 1), residualization, bootstrap method, threshold, 2-of-3 bar, no-post-hoc-re-spec on those parameters. The Fisher pullback **mathematical formula** is implicit — the spec assumes the pipeline computes it correctly. Discovering and fixing a bug in HOW the formula is computed is **bug-fix territory**, not a methodological change. Same as the 2026-04-24 preflight gate fixes (operational, not sealed re-spec).

  **Sealed E17b at rank=1 — current verdict status.** Pipeline's reported -0.166 (FAIL) is from buggy pre-norm geometry. Under J_n correction at N=100: Δ = +0.015, CI crosses 0 — INDETERMINATE. Need to re-run sealed gate at full N=200 with J_n-corrected null_ratio in the official pipeline path. Until that re-run lands, the sealed E17b verdict is **methodologically unsettled**; the v3.1-replicate page reports both readings honestly.

  **Cross-model finding promoted to descriptive headline.** Even at full N=200 with J_n correction, the cross-model picture won't collapse to "Fisher uniformly wins" or "Fisher uniformly loses" — Qwen 3 robustly favors Fisher, Mistral robustly favors raw, Qwen 2.5/Llama are in the noise. v3's claim repositions to **architecture-dependent**: Fisher pullback's edge over static raw subspace depends on the model's gen_step=1 commitment structure (Qwen-family commits to actual answer content; Mistral commits to formatting first). This is a richer and more honest scientific contribution.

  **Diagnostic mechanism.** Token analysis on Mistral's static W_u top-1 right singular vector reveals it's dominated by code-domain tokens (`ICENSE`, `qpoint`, `ityEngine`, `<s>`) — yet under J_n geometry it discriminates contradictions at 0.99 AUROC because contradiction `Δh` projects 1.5× as strongly along that axis as control `Δh` (mean +4.64 vs +3.01, std ≈ 0.5; both 100% positive — same axis, different magnitudes). raw_top1 on Mistral isn't a YES/NO bipolar axis — it's a **rupture-magnitude axis** that activates more strongly for contradictions regardless of YES/NO direction. ALL 100 Mistral samples emit `'\n'` as gen_step=1's first token (Mistral writes a newline before the answer); Qwen-family models front-load `' Answer'` / `'YES'` / `'NO'`. **The sealed gen_step=1 plane captures qualitatively different commitment moments per model** — for Mistral/Llama "begin the answer block," for Qwens actual answer content. E18 unaffected (residualization removes magnitude); E17b head-to-head IS affected (depends on Δh structure).

  **Pre-reg discipline preserved.** No sealed parameters changed. Code fix to the pipeline lands as a separate non-sealed-affecting patch (option to keep both pre-norm and J_n-corrected null_ratio columns in parquet, with sealed verdict reading the J_n column). Re-run of sealed gate at N=200 with J_n correction needed before publishing E17b verdict; current v3.1-replicate.md page reports the verdict status as **methodologically unsettled pending re-run**.

  **Code anchors:** `scripts/diagnostics/diagnose_norm_jacobian.py` (J_n correction implementation + multi-model diagnostic; renamed from `diagnose_qwen_norm.py` in PR #8 and moved into `scripts/diagnostics/` in the chore reorg), `scripts/diagnostics/diagnose_mistral_raw_top1.py` (token + signed projection analysis), `scripts/diagnostics/diagnose_wu_svd_tokens.py` (top-r SVD vector token decomposition). Diagnostic CSVs at `experiments/v3-main-run/2026-04-24/norm_diagnostic_*.csv`. ELI12 walkthrough: [learn/jn-correction-eli12](learn/jn-correction-eli12.md). Cross-reference: `wiki/log.md` 2026-04-25 J_n correction entry.

- **2026-04-26 · J_n geometry implementation: post-norm Δh capture, sealed E17b columns pinned to `*_post_rank{r}`.** Implements the 2026-04-25 J_n correction in the main pipeline. Sealed parameters untouched; only the COLUMN NAME consumed by the sealed E17b reading is named explicitly here.
  - **Implementation choice.** The 2026-04-25 amendment named two equivalent fixes for the post-norm-basis / pre-norm-Δh coordinate mismatch: (a) project `J_n(h_prev) · Δh_pre` onto the post-norm basis, or (b) build the basis in pre-norm space. PR #11 lands a **third option** preferable to both: capture `Δh_post = RMSNorm(h_t) − RMSNorm(h_prev)` directly using the model's own RMSNorm, project `Δh_post` onto the existing post-norm basis. True post-norm Δh, no Taylor linearization, exact coordinate consistency. `PRIComputer.rmsnorm` is cross-validated against `model.model.norm` at relative-error ~2e-7 (float32 noise).
  - **New columns emitted by the patched pipeline.** Per rank `r ∈ Config.v3_rank_values`:
    - `null_ratio_rank{r}` (legacy — pre-norm Δh on post-norm basis; KNOWN MISMATCH, retained for backwards compat with run-05 parquets and forensic comparison).
    - `null_ratio_post_rank{r}` (NEW — post-norm Δh on post-norm basis; geometrically consistent).
    - `null_ratio_raw_rank{r}` (legacy — pre-norm Δh on raw W_u basis; same mismatch).
    - `null_ratio_raw_post_rank{r}` (NEW — post-norm Δh on raw W_u basis; consistent).

    The post-norm columns appear ONLY when the pipeline can extract the model's final-RMSNorm γ (all v3.1 primaries qualify). If γ extraction fails, the pipeline emits a `[WARN]` and degrades to legacy-only — never aborts.
  - **Sealed E17b reading pinned to post-norm columns.** Sealed E17b head-to-head, on Qwen 2.5 at the existing sealed analysis plane (final layer, gen_step=1):
    ```
    AUROC(null_ratio_post_rank1) − AUROC(null_ratio_raw_post_rank1) ≥ 0.02
    with non-overlap 1000-resample sample-level bootstrap CI
    ```
    Rank pin (=1), analysis plane, bootstrap method, 2-of-3 / Qwen-2.5-authority structure unchanged from the 2026-04-23 v3.1 pre-reg. Only the COLUMN NAME consumed by `analyze_sealed_gate.py` changes; sealed STRUCTURE is identical.
  - **Sealed E18 unaffected.** Magnitude-independence on `null_ratio_rank1` residualized against `d_F_lowrank32` is robust to the J_n bug because residualization removes the coordinate mismatch — verified empirically on the 2026-04-22 / 2026-04-24 main runs (3-of-3 PASS holds across the bug). E18 verdict from existing run-05 parquets stands; no re-derivation needed. If `null_ratio_post_rank1` is later substituted for the residualized variable, the verdict is expected to reproduce within bootstrap noise.
  - **Why bug-fix, not sealed re-spec.** Same governance logic as the 2026-04-25 amendment. The sealed block (2026-04-18 + 2026-04-23 v3.1) names `null_ratio_rank1` / `null_ratio_raw_rank1` by their **mathematical role** — "fraction of Δh's energy outside the rank-r Fisher commit-direction subspace." The COMPUTED IMPLEMENTATION is implicit and the spec assumes the pipeline computes it correctly. Adding a column that computes the same mathematical object in consistent coordinates is bug-fix; pinning the analyzer to read that new column is namespace alignment, not methodological change. Legacy columns retained for forensic continuity, never deleted.
  - **What this amendment does NOT do.** Does not touch the sealed analysis plane, rank pin (=1), residualization, bootstrap method, threshold (≥ +0.02), 2-of-3 bar (E18) or Qwen-2.5-authority (E17b). Does not modify or delete any existing column. Does not pre-judge the verdict — under proper post-norm geometry on Qwen 2.5 at N=200, E17b could pass, fail, or be indeterminate. Does not change the launch command or seed.
  - **Required re-run before publishing E17b verdict.**
    ```
    scripts/run_v3_main.py --scope v3_1_main --n-per-cell 50 --seed 20260423 --max-gen-tokens 14
    ```
    From the patched branch (`fix/null-ratio-post-norm-geometry`, PR #11) or main once #11 merges. Fresh parquets carry both legacy and post-norm column families. Then `scripts/analyze_sealed_gate.py` reads the post-norm columns (analyzer patch is a separate commit / PR #12). Until that re-run lands, sealed E17b verdict remains **methodologically unsettled** per the 2026-04-25 amendment terminology.
  - **Code anchors.** PR #11 `fix/null-ratio-post-norm-geometry`: `pri_v2_mlx_pipeline.py` — new `PRIComputer.rmsnorm` staticmethod, `_extract_final_rmsnorm_gamma` helper, optional `final_norm_gamma` constructor parameter, optional `dh_post=` parameter threading through `null_ratio_and_energy` / `null_ratio_raw_and_energy` / `compute_step`. Smoke-tested: γ extraction on Qwen 2.5 7B (shape `(3584,)`, |γ|=233.4); `rmsnorm` ≡ `model.model.norm` to 2.07e-7 relative error; backward-compat path (no γ) emits ONLY legacy columns. Cross-reference: `wiki/log.md` 2026-04-26 entry (appended after the re-run lands).

- **2026-04-23 · v3.1 pre-registration (rank pinned + E17b integrated).** Filed **before** the v3.1 replicate dataset is generated; no v3.1 data exists yet. Closes the rank trapdoor from the previous amendment and integrates the HARP head-to-head into the same run.
  - **Rank pinned at r = 1.** Theoretical justification: the top-1 right singular vector of `sqrt(p_t) · W_u` is the single hidden-state direction along which an infinitesimal Δh changes `p_t(y_t)` most — by construction, the *commit direction*. `null_ratio_rank1` therefore measures "how much of Δh lives *off* the commit direction," which is the sharpest operational form of the v3 rupture hypothesis. Secondary justification: rank 1 was the rank that passed the sealed E18 gate with the largest-margin CIs in the 2026-04-23 data (Llama 0.8593 [0.806, 0.908], Mistral 0.8638 [0.814, 0.910], Qwen 2.5 0.7274 [0.656, 0.795]). Naming it in advance of fresh data retires the unpinned-parameter trapdoor.
  - **Sealed block unchanged.** Analysis plane (final layer, step 1), per-model linear residualization against d_F (default `d_F_lowrank32` for head-to-head parity with the v2 baseline; robustness check at `d_F_topk32`), 1000-resample sample-level bootstrap, AUROC ≥ 0.60 with non-overlap CI vs 0.5 on ≥ 2 of 3 primary models, no post-hoc re-specification. All five items remain sealed as of 2026-04-18; this amendment does not touch them.
  - **E17b included as a companion test on the same samples.** Pipeline emits `null_ratio_raw_rank{r}` and `raw_energy_rank{r}` alongside the Fisher-weighted `null_ratio_rank{r}` / `fisher_energy_rank{r}` via `PRIComputer.null_ratio_raw_and_energy` (new). The raw SVD basis is HARP-style: top-k right singular vectors of the raw output projection W_u (no `sqrt(p_t)` weighting), computed once per model at load time via chunked `W_uᵀ W_u` accumulation + symmetric eigendecomposition, cached on `OutputProjection`. Capture is toggled by `Config.v3_capture_raw` (default `True`; `--no-e17b` launcher flag to disable).
  - **E17b sealed gate (pins rank 1 for head-to-head parity):** `AUROC(null_ratio_rank1) − AUROC(null_ratio_raw_rank1) ≥ 0.02` with **non-overlapping** bootstrap 95% CI on Qwen 2.5 7B (primary; where HARP's headline 0.928 / our v2 0.786 gap is largest). Bootstrap = 1000 sample-level resamples; same test plane (final layer, step 1). Falsification of the curved-geometry framing: if `AUROC(null_ratio_raw_rank1) ≥ AUROC(null_ratio_rank1)` on Qwen 2.5 by any margin, or if CIs overlap, the `sqrt(p_t)` weighting adds no signal beyond HARP's static subspace and v3 collapses toward HARP's formulation; paper repositions.
  - **Fresh-data replicate — two-phase lean launch. Seed + scope pinned here before any data generation.**
    - **Seed = 20260423** (integer literal; ISO-date-derived, unique vs historical v2/v3 runs which used seed 42 and the 2026-04-23 smoke which used 101). Recorded 2026-04-23, committed to launch 2026-04-24. Same seed used for BOTH phases below so the synthetic-puzzle draws are identical across phases — controls for puzzle-sampling variance between the sealed-gate and the Gemma-companion readings.
    - **Phase 1 (main gate run, sealed authority): scope = `v3_1_main`** — 4 models: 3 primaries (Llama 3.2 3B, Mistral 7B, Qwen 2.5 7B) + Qwen3 8B extended. Sealed E18 / E17b gates scoped to primaries only (2-of-3 bar for E18; Qwen 2.5 for E17b). Qwen3 included for cross-generation-within-family companion data (Qwen 2.5 → Qwen 3 architecture shift; last run's v2 collapse to 0.50 vs surprise 0.96 makes it worth re-reading at rank 1 with E17b fresh). **Phi-3.5-mini excluded** — gate-fails at 12/20 = 60% on 2026-04-23 n=20 evidence; not worth the 2-3 min gate-skip cost. Follow-up via `--gate-verbose` when diagnosed.
    - **Phase 2 (Gemma companion, descriptive breadth): scope = `v3_1_gemmas`** — Gemma 3-1B + Gemma 3-4B, launched AFTER Phase 1 completes with same seed and same E17b capture. Rationale for separation: full `run_experiment` loop with `v3_capture_raw=True` has never executed end-to-end on a Gemma checkpoint (Prereq 4 dryrun validated only `trace_sample` + SVD path at n=4/cell). Isolating to a dedicated launch keeps the sealed-gate checkpoints immutable if a Gemma-specific adapter regression trips the raw-W_u SVD path (e.g. 3-4B multimodal wrapper reach-through, bfloat16 to_numpy edge case). Gemma 1B ↔ 4B within-family-scale axis still gets recorded; it just doesn't risk Phase 1.
    - **Sealed gates remain with primaries. Extended / Gemma readings are descriptive.** Qwen3 (Phase 1), Gemma 1B, Gemma 4B (Phase 2) are reported for breadth but **cannot satisfy or invalidate the sealed E18 / E17b gates**. Distinguishing breadth axes preserved: cross-family (Llama/Mistral/Qwen/Gemma), cross-generation (Qwen 2.5 → Qwen 3), within-family scale (Gemma 1B ↔ 4B — architecture held fixed).
    - **Launcher commands (superseded below 2026-04-24 — three-phase split):**
      - Phase 1 (2-phase, retired): `scripts/run_v3_main.py --scope v3_1_main --n-per-cell 50 --seed 20260423 --max-gen-tokens 14`
      - Phase 2 (2-phase, retained): `scripts/run_v3_main.py --scope v3_1_gemmas --n-per-cell 50 --seed 20260423 --max-gen-tokens 14`
      - E17b capture on by default in both (`Config.v3_capture_raw = True`). Behavioral preflight per-model at n=20 with 80% threshold (Config default). Artifacts land under `experiments/v3-main-run/<YYYY-MM-DD>/run-NN/`.
    - **Expected runtimes (Mac mini M4, 2-phase reading, retained for reference):** Phase 1 ≈ 80-100 min (Llama 20 + Mistral 25 + Qwen 2.5 25-35 + Qwen3 15-20, plus one-time raw-W_u SVD precompute per model; Qwen 2.5's 152k-row SVD is the longest at ~2-3 min). Phase 2 ≈ 40-60 min (Gemma 1B fast, Gemma 3-4B slower with multimodal wrapper unwrap at load).
    - **No-silent-override rule (auditability):** if any gate fails mid-run, log and continue — do NOT flip `--skip-gate` or change thresholds without filing a new Amendments entry first. Applies symmetrically to Phase 1 and Phase 2.
  - **Out of v3.1 scope (explicitly parked):** E19 `null_gated` interpretation gate (FALSIFIED 2026-04-23 at sealed rank 32; spec names `v2_lowrank32` by reference so rank 32 is the sealed operating point for E19 — result final, no rerun). E21 depth profile (`v3_capture` remains off for v3.1 — this is a final-layer-only test; depth profile is v4 territory per claims.md `[OPEN][FUTURE-V4]`).
  - **Code anchors** (for auditability): rank-pin lives in the launcher surface (`scripts/run_v3_main.py`) and the sealed block text below; E17b computation in `pri_v2_mlx_pipeline.py` `PRIComputer.null_ratio_raw_and_energy` + `OutputProjection.raw_right_singular_vectors`; unit test at `scripts/test_e17b_raw_svd.py` (6 bundles, all pass on synthetic fixtures). Cross-reference: `wiki/log.md` 2026-04-23 v3.1 amendment entry.

## One-line thesis
v2 gave us one scalar `d_F` at the final layer — "how much did the model move in output-sensitive directions?" **v3 computes the Fisher eigenspace at every probed layer and measures what fraction of `Δh` falls into the null space** — the low-eigenvalue directions where the output is blind.

Moving a lot is not the same as moving into a blind spot. v3 separates them.

## What v3 gives us that v2 cannot
1. **Projection ratio per layer.** `null_ratio_ℓ = ||proj_null(Δh_ℓ)|| / ||Δh_ℓ||` at each probed layer ℓ. Distinguishes magnitude from direction.
2. **Depth profile.** At *which* layer does the model start steering into blind directions? Early? Late? Gradual? A new observable v2 cannot express.
3. **Falsifiable SUP-native prediction.** Contradiction samples should show higher null-space concentration than controls, emerging at a **characteristic depth per architecture**.

One metric, every layer, mechanistically interpretable, architecture-comparable.

## Core statistic
```
null_ratio_ℓ = || Δh_ℓ − V_topr V_topr^T Δh_ℓ || / || Δh_ℓ ||
```
where `V_topr` are the top-r right singular vectors of `sqrt(p_t) · W_u`. The complement of the informed subspace is the null subspace — cheaper than computing the bottom eigendirections directly.

**Sign invariance (to state in methods):** SVD right singular vectors have arbitrary sign. `null_ratio` uses the outer product `V_topr V_topr^T`, which is sign-invariant (sign cancels). No deflation or sign-pinning required. Reviewers frequently flag this; make it explicit.

**Rank r anchoring (data-driven, not arbitrary).** For each sample/step/layer, also record the **cumulative Fisher-energy ratio** `ε(r) = Σ_{i≤r} σ_i² / Σ_i σ_i²`. Report `ε(r)` alongside every null_ratio measurement. The rank sweep then reports signal-vs-energy-captured, not signal-vs-arbitrary-cutoff.

## Fisher eigenspace: design choices
- **Option A (single eigenspace).** SVD of `sqrt(p_t^final) · W_u` once per step; project every layer's `Δh_ℓ` into the same subspace. Tests: "does the pre-final residual stream steer into the *final-layer* null space?" Cheap — one SVD per step regardless of layer count. **Default for v3 v0** (now strongly preferred after spectral-band run; final-p is the actual generative distribution and is honest about uncertainty).
- **Option B (layer-specific eigenspace via logit lens).** ~~SVD of `sqrt(p_t^ℓ) · W_u` where `p_t^ℓ = softmax(W_u · h_ℓ)` is the layer-ℓ logit-lens distribution.~~ **[Disfavored 2026-04-14]** Spectral-band pre-plan revealed that as p^(ℓ) concentrates, the SVD degenerates to "p is one-hot" rather than measuring W_u geometry. Defer until a sharpness-aware reformulation (Option C) is in place.
- **Option C (sharpness-aware, new).** Three sub-variants to explore before the next pre-registration: (C1) entropy-normalize null_ratio per layer; (C2) soften the SVD weighting from `sqrt(p)` to `p^α` with α ∈ {0.1, 0.25, 0.5}; (C3) condition cross-layer comparisons on a fixed entropy band (e.g., only compare layers where H(p^(ℓ)) > 5 nats). File as exploratory variants in v3 v0; promote one to confirmatory only if it cleanly separates Fisher geometry from p-sharpness.

## Infra cost
v2's `fim_lowrank` already computes `U, Σ, V^T` of `sqrt(p_t) · W_u`. v2 throws `V` and `U` away and keeps `Σ`. **v3 keeps `V` and computes the projection instead.** Option A adds nothing beyond v2's existing SVD; all cost is in extended layer-state capture.

## Operational plan

### New metrics (`pri_metrics.py`)
- `compute_null_ratio(dh, V_topr)` — `|| dh − V_topr V_topr^T dh || / || dh ||`
- `compute_layer_profile(hidden_states_per_layer, W_u, p_t, rank=r)` — returns `null_ratio_ℓ` for every probed layer
- `compute_spectrum_decay(sigma)` — log-slope of singular values (per-layer)

### New score variants
- `pri_v3_null_bare` — `null_ratio_final` alone (no `S_t`, no `d_F`). **Decomposition control** — if this beats v2, `null_ratio` is just a better scalar than `d_F`.
- `pri_v3_null_raw` — `null_ratio_final` computed from SVD of **raw `W_u`** (no `sqrt(p_t)` weighting). **HARP-style baseline** — filed 2026-04-17 after ingesting [Hu et al. 2025, HARP](papers/external.md#hu-et-al-2025--harp-hallucination-detection-via-reasoning-subspace-projection). Uses their 95%-energy cutoff (`k ≈ 0.95·d`) for informed subspace; reasoning subspace is the orthogonal complement. **Isolates the contribution of Fisher weighting** — if `null_bare` (Fisher-weighted) beats `null_raw`, the `sqrt(p_t)` weighting carries signal beyond the static W_u decomposition. If they tie, HARP's static subspace is sufficient and our weighting adds complexity without value. Also runnable at HARP's r ≈ 256 for direct method-level parity.
- `pri_v3_null_ratio` — additive: `S_t + α · null_ratio_final`
- `pri_v3_null_gated` — multiplicative: `d_F · null_ratio` at final layer
- `pri_v3_layer_profile` — vector-valued; scored by **argmax-depth** (primary), max, mean, area-under-depth-curve
- `pri_v3_spectrum` — eigenspectrum decay rate as signal

### Depth-profile scoring
The depth profile is vector-valued but AUROC needs a scalar. Primary scalar: **`argmin_depth`** = the layer index at which `null_ratio_ℓ` first **drops below** a per-sample threshold (e.g., 1.5× its prefix-phase *minimum*). Most interpretable — answers "where does Δh first enter the informed subspace." Secondary scalars: `max_ℓ (1 − null_ratio_ℓ)` (peak informed-direction content), `mean_ℓ (1 − null_ratio_ℓ)`, trapezoidal AUC across layers. **[Flipped 2026-04-16 per E22 verdict](results/e22-direction-depth.md)** — rising `null_ratio` means *less* informed content, not more; earlier `argmax_depth` framing had the sign backwards.

### Pre-registered experiments
- **E17** `pri_v3_null_bare` — H: null_ratio alone (no surprise, no d_F) already separates conditions. Direction: AUROC > 0.6 on at least one model. **Decomposition control — necessary for interpreting E18/E19.**
- **E17b** `pri_v3_null_raw` — H: Fisher weighting (`sqrt(p_t) · W_u`) carries signal beyond the static raw-`W_u` decomposition used by HARP. **Acceptance criterion:** `AUROC(null_bare) − AUROC(null_raw) ≥ 0.02` with **non-overlapping bootstrap 95% CI** on Qwen (primary; where HARP's headline 0.928 / our v2 0.786 gap is largest). Report also on Llama and Mistral but Qwen is the decisive model. **Falsification of curved-geometry framing:** if `AUROC(null_raw) ≥ AUROC(null_bare)` on Qwen by any margin, or if the Qwen CI overlaps, the Fisher weighting is not carrying independent signal and the plan collapses toward HARP's static formulation (switch `null_raw` to the default v3 primitive and reposition the paper). Filed 2026-04-17; runs alongside E17 at the same n=50/cell (per 2026-04-22 amendment); re-fleshed 2026-04-18.
- **E18** `pri_v3_null_ratio` — H: null-space concentration separates rupture from controls **independent of magnitude**. Direction: AUROC ↑ vs `pri_v2_lowrank32` on at least Qwen.
- **E19** `pri_v3_null_gated` — H: magnitude × direction beats either alone. **Interpretation gate:** `null_gated` win only counts as genuine interaction if AUROC(null_gated) > max(AUROC(null_bare), AUROC(v2_lowrank32)) by a non-overlapping CI margin. Otherwise the "win" is just `null_bare` being stronger than `d_F`.
- **E20** `pri_v3_spectrum` — ~~H: eigenspectrum decay rate is itself a rupture signature; cross-model decay-vs-rupture correlation positive.~~ **[Demoted 2026-04-14]** Spectral-band pre-plan run found the spectrum is dominated by p-sharpness, not Fisher geometry. Recast as **exploratory rank-sensitivity analysis** — not pre-registered. Report decay curves descriptively.
- **E21** `pri_v3_depth_profile` — H: ~~there exists a characteristic commitment-rupture depth per architecture, stable across samples.~~ **[Reframed 2026-04-14]** Each architecture has its own depth signature; cross-architecture universality is **not** assumed (peak depths span 0.00 / 0.13 / 0.93 in the spectral pre-plan). Primary scalar: `argmin_depth` per model (see flipped definition above, post-E22), reported per-architecture rather than tested for cross-arch agreement.
- **E22** `pri_v3_direction_depth_signature` — **exploratory, gating** (filed 2026-04-15). H: `null_ratio_ℓ` has a cross-arch-reproducible depth profile even though `λ_max/λ_mean` does not, because `null_ratio` is normalized by `||Δh||` and is therefore insensitive to the `p^(ℓ)` sharpness confound that dominated the 2026-04-14 spectral-band run. Rationale: `λ_max/λ_mean` is a *magnitude* observable in eigenvalue-space; `null_ratio` is a *direction* observable in projection-space — they can disagree. Procedure: n=4/cell every-layer capture at step 1, Llama / Mistral / Qwen (existing stack only), before the confirmatory v3 main run. Direction: descriptive cross-arch depth-profile plot. **Gate:** if reproducible structure present → keep broad layer density (every layer × 12 steps) in the main run; if not → narrow to 5 probe layers `{embed, 1/4, 1/2, 3/4, final}` × 12 steps to save capture budget. Outcome is *diagnostic for design*, not a pre-registered hypothesis about the world.

### v3-specific analyses

**Random-baseline reporting (mandatory, added 2026-04-16 per E22 verdict).** For a random Δh in hidden-dim `d` with top-r informed subspace, the expected `null_ratio` is `√((d − r) / d)` — e.g., 0.9948 for Llama `d=3072, r=32`. Raw `null_ratio ≈ 0.99` therefore does **not** mean "strong null-space concentration"; it can mean "indistinguishable from random." **Every v3 plot and headline metric must subtract the random baseline or report `1 − null_ratio` ("informed-direction fraction")** on the y-axis. Include this per-model baseline table in paper methods:

| Model      | `d`   | r=8    | r=16   | r=32   | r=64   |
|------------|------:|-------:|-------:|-------:|-------:|
| Llama 3B   | 3072  | 0.9987 | 0.9974 | 0.9948 | 0.9895 |
| Mistral 7B | 4096  | 0.9990 | 0.9980 | 0.9961 | 0.9921 |
| Qwen 7B    | 3584  | 0.9989 | 0.9978 | 0.9955 | 0.9910 |

- **Magnitude-independence test (frozen 2026-04-18 per Codex adversarial review).** This is the E18 falsification criterion and is pre-registered here before any confirmatory data lands — the spec is sealed.
  - **Unit of analysis:** one row per `(sample, model, step=1, layer=final)` — one observation per contradiction / control token at commitment. Prefix steps and later generation steps are out-of-scope for this test.
  - **Model class:** logistic regression with condition (`contradiction` vs `control`) as binary outcome; predictors `null_ratio`, `d_F`, and their interaction `null_ratio · d_F`. No layer or depth covariate (final layer only at step 1).
  - **Fit scope:** per-model (fit three separate logits — one for Llama, Mistral, Qwen). No pooling. Per-model interpretation is the unit of inference; we do not claim a pooled effect.
  - **Residualization procedure:** compute `null_ratio_resid = null_ratio − predicted(null_ratio | d_F)` via linear regression on `d_F` alone (not logistic — we want a true residual). Then report AUROC of `null_ratio_resid` for separating condition.
  - **Acceptance threshold:** `AUROC(null_ratio_resid) ≥ 0.60` with **non-overlapping bootstrap 95% CI against AUROC = 0.5** (chance) on at least **two of three** primary models. Bootstrap = 1000 resamples at the sample level, not the token level.
  - **Falsification:** if `AUROC(null_ratio_resid) < 0.60` or the CI overlaps 0.5 on two or more primary models, direction carries no signal independent of magnitude — v3 collapses to a reparameterization of v2. See §Falsification conditions blocker #2.
  - **No post-hoc re-specification.** Do not change regression family, pooling, or threshold after seeing the confirmatory data. Any deviation must be filed as a *new* exploratory variant, not a revised E18.
- **Depth sweep.** Plot `1 − null_ratio_ℓ` (informed-direction fraction, baseline-normalized) vs layer index per model per condition. Look for a **drop-point** in raw `null_ratio_ℓ` (= rise-point in informed fraction) — the layer at which Δh first enters the informed subspace.
- **Cross-model spectrum-shape.** Eigenvalue-decay curves at commitment vs control across the extended suite (Llama 3.2 3B, Mistral 7B, Qwen 2.5 7B, Qwen3 8B, Gemma 3 1B, **Gemma 3 4B**, Phi-3.5-mini). Claim: shape transfers. **Gemma 1B ↔ 4B comparison also provides a within-family scale axis** (architecture held fixed), which is the cleanest test of the inverse g-vs-capability replication the Furnace paper currently lacks.
- **Rank sweep.** `r ∈ {8, 16, 32, 64}` — signal survival across cutoffs.

## Layer granularity
Current pipeline captures `{final, mid, quarter}` = 3 points. **Three points cannot distinguish "gradual rise" from "step function"** — inadequate for a depth-profile claim. Target: ≥ 8 layers, ideally every layer (see "Every layer?" section below).

Default capture plan: **every layer for steps 1–12, 4 probe layers for steps 13+**. See "Every layer?" section below for cost reasoning.

## Extended model suite: smoke-test before committing
Qwen3 8B, Gemma 3 1B, Gemma 3 4B, Phi-3.5-mini are **not in the current pipeline adapter stack**. Before any n=50/cell run, each new model must pass:
- Forward-pass + hidden-state capture smoke test (shapes, layer count, dtype)
- `W_u` unembedding shape + vocab alignment
- Behavioral preflight gate (≥0.98 control acc on n=4 puzzles)

A failed mid-run adapter costs hours. Gate this behind a `scripts/smoke_test_model.py` that runs in <60s per model.

**Gemma 3 1B + 4B onboarding note.** Both share the same MLX-LM model class (`gemma3`), so one `GemmaAdapter` subclass covers both — the two differ only in hidden-dim, layer count, and vocabulary. Smoke-gate them together. Slugs confirmed 2026-04-21: `mlx-community/gemma-3-1b-it-4bit` and `mlx-community/gemma-3-4b-it-4bit` (also a QAT variant `*-it-qat-4bit` available; evaluate if post-training-quantization artifacts emerge at 1B). **Rationale for including both:** 1B → 4B within a single architecture family gives a clean scale axis with architecture held fixed — the scale-vs-architecture confound is the main weakness of the current paper's inverse g-vs-capability observation.

## Workflow
User-requested end-to-end: **setup → run → complete → LaTeX paper**.

1. **Setup.**
   - Extend `pri_metrics.py` with `compute_null_ratio`, `compute_layer_profile`, `compute_spectrum_decay`, `compute_fisher_energy_ratio`.
   - Extend `hidden_state_collector.py` to capture all-layer hidden states (configurable list; default every layer at step 1, `{final, mid, quarter}` at all steps).
   - Add `scripts/smoke_test_model.py` for new-model gating.
   - Extend `PRI_V2_PRE_RUN_AUDIT_CHECKLIST.md` with v3 checks: sign-invariance of `null_ratio` (documented, not fixed), rank-truncation consistency across steps, logit-lens numerical stability (Option B), layer-index alignment, depth-profile threshold-robustness.
2. **Run.**
   - Smoke-test Qwen3 8B, Gemma 3 1B, Phi-3.5-mini.
   - Exploratory n=4/cell across the surviving suite, Option A eigenspace.
   - If signal present → confirmatory n=50/cell with Bonferroni across variants.
   - Option B rerun on the model showing strongest signal.
3. **Complete.** Extract per-variant, per-layer AUROC + Hedges g + bootstrap CI + Fisher-energy ratio to `summary.parquet`. Run magnitude-independence regression. File per-model pages with depth profiles. Resolve falsification conditions explicitly.
4. **LaTeX paper.** Draft in `templates/` (currently reserved). Structure:
   - Intro: v2 scalar limitation — magnitude-only, final-layer-only.
   - Theory: Fisher eigenspace, null projection, sign invariance, FEP framing.
   - Methods: synthetic 2×2, cross-layer extraction, Option A vs B, Fisher-energy anchoring, argmax-depth scoring.
   - Results: per-layer depth profile (the headline figure), magnitude-independence regression, cross-architecture spectrum comparison, decomposition (null_bare vs null_gated vs v2).
   - Discussion: mechanistic interpretation, cross-arch transfer, limitations.

## Every layer? — cost/benefit
**Short answer: yes, at step 1 (commitment); no, at all generation steps.**

Why yes at step 1:
- Option A eigenspace is *one SVD per step* regardless of layer count — cost is all in state capture, not projection.
- For the depth-profile claim (E21) to be defensible, you need enough layers to show shape — 3 points is under-determined; 8+ is the minimum for "gradual vs step"; every layer eliminates binning artifacts entirely.
- The primary paper figure is the depth-profile plot — density of this plot is worth its memory cost.

Why not at every step:
- Storage scales as `layers × hidden_dim × steps × samples`. For a 32-layer, 4096-dim model at 20 steps over 800 samples: ~2 GB fp16 per variant. Tractable for one model; painful across six.
- Off-step layers are diagnostic, not the primary observable — `{final, mid, quarter}` at steps ≠ 1 is sufficient to confirm the step-1 commitment localization claim from v2 still holds.

**Default (user-specified 2026-04-15): capture every layer for steps 1–12, then 4 probe layers `{final, 3/4, mid, quarter}` for steps 13+.** Steps 1–12 cover commitment + early decay + tail-of-decay, which is where v2 showed signal fall-off; full-depth resolution here is where the depth-profile claim lives. After step 12 the signal is expected near-flat, so 4-layer diagnostic probe is enough. Config: `layer_capture_schedule: {steps_1_to_12: "all", steps_13_plus: "probe_4"}`.

Caveat: models vary in depth (Gemma 3-1B ≈ 26, Llama 3.2 3B = 28, Qwen 2.5 7B = 28, Mistral 7B = 32, Gemma 3-4B ≈ 34 *(TBD — confirm on smoke load)*, Qwen3 8B = 36). Normalize depth to `ℓ / L` when plotting across models so `argmin_depth` is comparable.

## Relationship to v2
v2 stays. If v3 wins, v2 becomes the scalar-projection baseline. If `null_gated` beats both v2 and `null_ratio` alone, v2 is **underspecified** rather than wrong — magnitude and direction are both load-bearing.

## SUP-derived priors (theoretical provenance — post-validation)
v3 was not a from-scratch hypothesis; the SUP corpus predicted most of its claims. The 2026-04-14 spectral-band pre-plan tested the strongest of those priors and **shifted/falsified** parts of it. See [results/sup-spectral-band](results/sup-spectral-band.md) and [sup/theory-notes](sup/theory-notes.md). Updated status:
- **Depth profile (E21).** SUP taxonomy work (`sup-taxonomy-distance-v4.pdf`) found semantic hierarchies optimally encoded at layers `-6 to -8` of 12-layer models. **Furnace measurement disagrees**: peak depths span 0.00 / 0.13 / 0.93 across Llama / Mistral / Qwen. The "middle-late" prior holds **only** for Mistral (depth 0.13 is shallow-mid); Llama peaks at the embedding edge and Qwen near the final layer. **Do not assume `argmin_depth` is universal across architectures.** The 2026-04-16 E22 run confirms this from the `null_ratio` side: Llama and Mistral share a late-rising-informed shape (argmin_depth at the final layer) but Qwen stays at random-baseline throughout — no cross-arch universal depth. See [results/e22-direction-depth](results/e22-direction-depth.md).
- **Spectrum decay (E20).** `sup-from-error-to-essence.pdf` claims `λ_max/λ_mean ∈ [10², 10⁴]` at semantically-strong layers; pathological at final layers. **Furnace measurement: SHIFTED.** Two of three models live entirely below the lower edge of [10², 10⁴]; Qwen's apparent in-band reading is entropy-collapse-driven. E20 SUP-backing **withdrawn**; E20 demoted to exploratory.
- **Null-as-discharge (E17/E18).** SUP's "bounded imprecision" thesis. **Not yet tested in Furnace** — these are still pre-registered as primary v3 hypotheses. The spectral-band run does not bear on them directly.
- **Cross-arch variance (the suite).** `sup-cognitive-speciation-v2.pdf` predicted different architectures navigate the trade-off differently. **Confirmed in extreme form** — peak-depth divergence (0.00 / 0.13 / 0.93) and band divergence (Llama/Mistral below SUP, Qwen at lower edge) are larger than originally anticipated. The Qwen v1 inversion fits this picture; the spectral-band split deepens it.

## Regression guard: step-0 `h_prev` bug
The paper's inflated AUROCs (0.998/0.994/0.980) came from the first generated token lacking a real previous hidden state. v3 **must not** repeat this.

Hard requirements for any v3 run:
1. `h_prev` at step 0 binds to `last_prefix_hidden[layer]` — asserted, not assumed.
2. Every per-step row logs `h_prev_source ∈ {"prefix_last", "gen_prev"}`. Step 0 = `"prefix_last"`; violation is a pipeline bug.
3. `||Δh_step_0|| / ||h_t|| < 10` magnitude sanity check. Violations are flagged and excluded.
4. Finite checks on both `h_t` and `h_prev` before any metric computes.
5. Optional `cfg.skip_step_0_metrics` mode emits NaN for step 0 — use if the prefix/generation norm boundary is architecturally unclean.

Implementation site: see [v3-code-map §7b](v3-code-map.md). Added to audit checklist as §12.7. This gate is **non-negotiable** — no v3 numbers get reported before it passes.

## Falsification conditions

**Structural rule (added 2026-04-18 after Codex adversarial review).** Only outcomes of the confirmatory pre-registered experiments (E17, E17b, E18, E19) can project-falsify v3. Exploratory / descriptive experiments (E20 spectrum, E21 depth profile, Qwen quantization diagnostic) are diagnostic — they can demote or reshape claims but cannot retroactively falsify the whole line. This replaces an earlier "any of the below" list that double-counted already-demoted experiments as blockers.

### Confirmatory blockers (project-falsifying)
v3 is falsified if **any** of:
1. `null_ratio` has no separation at all on all three primary models (E17: AUROC ≤ 0.55 on Llama ∧ Mistral ∧ Qwen).
2. Separation vanishes after controlling for `d_F` magnitude per the E18 residualization protocol (see §Magnitude-independence below). **[Primary bet]**
3. `null_gated` (E19) "wins" only because `null_bare` is stronger than `d_F` — the E19 interpretation gate fails: `AUROC(null_gated) ≤ max(AUROC(null_bare), AUROC(v2_lowrank32))` by CI.
4. HARP's raw-`W_u` baseline equals or beats Fisher-weighted `null_bare` on Qwen (E17b): the curved-geometry thesis collapses; plan must reposition toward HARP's static formulation.

### Diagnostic (not project-falsifying, but reshape claims)
The following outcomes inform scoping, not ship/no-ship:
- No characteristic depth — depth profile flat or sample-wise incoherent (E21). Already reframed 2026-04-14 as per-architecture descriptive, not a universal claim.
- Eigenspectrum decay does not transfer across architectures (E20). Already demoted 2026-04-14 to exploratory rank-sensitivity analysis.
- Signal unstable across Fisher-energy ratios `ε(r)` — indicates cutoff sensitivity; reported as a methods caveat, not a project blocker.
- ~~Qwen diagnostic ladder (Prereq 8) outcome~~ — **CLOSED 2026-04-18**, step 1 passed (max |dev| 0.030 ≥ 0.020). Qwen in cross-model claims. Steps 2–3 no longer required.

## Pre-plan: SUP spectral-band validation (before v3 build)
Status: `[RUN COMPLETE 2026-04-14]` — verdict `[SHIFTED]` (borderline `[FALSIFIED]`). Full results at [results/sup-spectral-band](results/sup-spectral-band.md). Summary inline below.

### Verdict snapshot
| Model | log10(λ_max/λ_mean) range | Peak depth | Peak ratio | Peak entropy |
|-------|---------------------------|-----------:|-----------:|-------------:|
| Llama 3B   | [1.47, 2.00] | 0.00 |  99 | 11.76 |
| Mistral 7B | [1.26, 1.77] | 0.13 |  60 | 10.40 |
| Qwen 7B    | [1.89, 2.40] | 0.93 | 250 |  0.14 |

SUP-stated band is log10 ∈ [2, 4]. Two of three models live entirely below it. Qwen grazes the lower edge **only at layers where its logit-lens distribution has already collapsed to near-one-hot** (top1 = 0.97, ε(16) = 1.000) — the high ratio reflects p-sharpness, not W_u geometry. Peak depths span 0.00 / 0.13 / 0.93 — no shared characteristic depth across architectures.

### What this changes for v3
- **E20 (spectrum decay):** SUP-backing **withdrawn**. Recast as exploratory rank-sensitivity analysis, NOT a pre-registered hypothesis. The v3 paper's theory section must note SUP/Furnace divergence openly.
- **E21 (depth profile):** "characteristic depth is architecturally universal" prior **dropped**. Per-architecture profiles are real and reproducible; cross-architecture universality is not. Reframe E21 as "each architecture has its own depth signature" — diagnostic rather than predictive.
- **Option B (per-layer logit-lens eigenspace) disfavored** until the entropy-collapse confound is addressed. As p^(ℓ) concentrates, `A = sqrt(p_s)·W_s` becomes near-rank-1 by construction — the SVD measures p sharpness, not Fisher geometry. **Option A (single final-p eigenspace) is now the default for v3 v0.** Defer Option B until a sharpness-aware reformulation lands.
- **New v3 design axis: sharpness-aware metric.** Either (a) entropy-normalize null_ratio per layer, or (b) soften the SVD weighting from `sqrt(p)` to `p^α` with α<1 (e.g., α=0.25), or (c) condition on a fixed entropy band when comparing across layers/architectures. File these as Option C variants and exploratory-test before confirmatory pre-registration.
- v3 can still proceed on **E17, E18, E19** — these don't depend on the SUP band holding.

### Original pre-plan (preserved for context)
Tests the single strongest SUP-PROVISIONAL prior that v3 leans on (E20 in particular, E21 indirectly). Cheap, falsifiable, result-shaping.

### Why run this first
The E20 hypothesis ("eigenspectrum decay rate is a rupture signature") and the E21 depth-profile prior both rest on `sup-from-error-to-essence.pdf`'s claim: *semantically-strong layers have `λ_max / λ_mean ∈ [10², 10⁴]`; final layers are pathological (`> 10⁶`)*. That number is a direct SUP extract — not Furnace-replicated (see the provisional banner in [sup/theory-notes](sup/theory-notes.md)). Before building v3 on top of it, spend one session verifying the shape holds on our existing stack. Three outcomes, all informative:
- **Shape holds.** E20/E21 become pre-supported rather than pre-asserted. The depth-profile paper gets a stronger intro.
- **Shape holds but band is model-shifted** (e.g. Qwen's band is `[10³, 10⁵]`). Cognitive-speciation prediction — interesting finding in its own right. Report per-model bands and proceed.
- **No band, no pathology, no shape.** E20 loses its SUP backing; v3 must either revise E20 (treat spectrum decay as an exploratory probe, not a pre-registered hypothesis) or acknowledge explicit tension with SUP.

### What to measure
For each model × {contradiction, control} × {step 1 commitment only}: the ratio `λ_max^(ℓ) / λ_mean^(ℓ)` at every probed layer ℓ. The singular values of `sqrt(p_t) · W_u` are the square roots of the Fisher eigenvalues, so `σ²` gives the spectrum directly. v2's `fim_lowrank` already computes this SVD; the pre-plan reuses it unchanged and adds one derived statistic.

Secondary: `tr(I) / det(I)^{1/d}` per layer (the second spectral condition from `from-error-to-essence` Prediction 2), and the cumulative energy ratio `ε(r)` — both fall out of the same SVD.

### Scope
- Models: **Llama 3B, Mistral 7B, Qwen 2.5 7B** (already pipeline-validated, behavioral gates pass). Don't add new models — this is a structural measurement, not an extended suite run.
- Sample size: **n = 4 / cell** (exploratory). This is a per-layer spectral shape question, not a between-condition separation question — high-n is not needed.
- Layers: same all-layer capture as the v3 depth-profile plan (prerequisite #4); this pre-plan is the *first consumer* of that capture extension.
- Compute: one pass through the existing 2×2 puzzle set. Estimated ≤ 1 session.

### Deliverable
- `wiki/results/sup-spectral-band.md` — per-model layer-index vs `λ_max/λ_mean` table + plot, overlaid with the SUP-predicted `[10², 10⁴]` band.
- Explicit verdict line: `[SUP-VALIDATED-IN-FURNACE]`, `[SUP-SHIFTED]`, or `[SUP-FALSIFIED]`.
- Downstream bookkeeping: update `claims.md` — promote or demote E20's SUP-backing status; update `sup/theory-notes.md` §2.2 with the verdict.

### Falsification criteria
- **Shape holds** if at least 2/3 models show a middle-late band with `λ_max/λ_mean` one order of magnitude *below* the final-layer ratio, and the middle-late band overlaps `[10², 10⁴]` within an order of magnitude.
- **Shifted** if the band-shape pattern holds but the absolute band differs by >1 order of magnitude across models.
- **Falsified** if fewer than 2/3 models show layer-dependence, or if the final layer is not systematically higher than middle layers.

### Non-goals
- Not a separation experiment — we are *not* comparing contradiction vs control spectra here (that is E20 itself). We only verify the layer-structural prior exists.
- Not a parameter sweep — no rank sweep, no α tuning.
- Not a replacement for E20 — this pre-plan *grounds* E20; it does not execute it.

### Gate condition
If the pre-plan falsifies SUP's band claim, **do not ship E20 as a pre-registered hypothesis**. Recast it as an exploratory rank-sensitivity analysis, and edit the v3 paper's theory section to note SUP/Furnace divergence openly. v3 can still proceed on E17/E18/E19/E21 — E20 alone doesn't carry the project.

## Pre-plan: E22 direction-depth signature gate (before v3 main run)
Status: `[RUN COMPLETE 2026-04-16]` — verdict `[PARTIAL STRUCTURE]`. Full results at [results/e22-direction-depth](results/e22-direction-depth.md). ELI12 companion at [learn/bugs-caught-eli12](learn/bugs-caught-eli12.md) (merged E22 + E23 verdicts, 2026-04-19). Summary inline below.

### Verdict snapshot

> **🔄 Superseded for Qwen (2026-04-18).** Prereq 8 step 1 (normed Option A rerun) shows Qwen's "flat ≈ random" reading was a **final-norm artifact** — same bug class as E23's Llama layer-0 spike. Post-fix Qwen row in the corrected table below; the rest of this verdict page's Qwen narrative reads on the stale subspace and is preserved only for historical traceability.

Per-model `null_ratio` depth profile at rank 32 (deviation from random-projection baseline `√((d−r)/d)`; more-negative = more informed-direction content):

| Model      | Shape                          | Peak informed layer (depth) | Peak dev from baseline | Final-layer dev |
|------------|--------------------------------|-----------------------------|-----------------------:|----------------:|
| Llama 3B   | monotonic late-rise            | layer 27 / depth 1.00       | **−0.054**             | −0.054          |
| Mistral 7B | late-rise with final crash     | layer 31 / depth 1.00       | **−0.041**             | −0.041          |
| Qwen 7B ⟂  | late-rise (post-norm-fix)      | layer 27 / depth 1.00       | **−0.030**             | −0.030          |

⟂ Qwen row is from Prereq 8 step 1 (2026-04-18); E22's original un-normed read had Qwen at dev −0.009 / argmin layer 14 and was superseded once the final-norm was applied before the logit-lens. Llama and Mistral share the late-rising-into-informed shape (both ≥ 0.04 at final layer); Qwen now joins the same shape with a smaller magnitude. Cross-arch *magnitude* still differs; cross-arch *shape* is now consistent across all three primary models.

### What this changes for v3
- **Gate decision: every-layer × 12-step capture schedule retained.** Narrowing to 5 probe layers would erase Llama / Mistral's late-rise resolution and miss Qwen's structurally-flat counter-signal.
- **Random-baseline reporting is now mandatory.** Raw `null_ratio ≈ 0.99` sits at the geometric baseline `√((d−r)/d)` and means nothing on its own. All v3 plots subtract the baseline (or report `1 − null_ratio`). Per-model baselines tabulated below in the v3-specific analyses section.
- **`argmax_depth` → `argmin_depth` scoring correction.** E22 showed rising `null_ratio` = *more* null = *less* informed — the opposite of what the original E21 scalar spec assumed. Primary scalar flipped across the plan; threshold becomes "1.5× per-sample *minimum*".
- ~~**Qwen flagged as expected outlier.**~~ **Resolved 2026-04-18.** Prereq 8 step 1 (normed Option A rerun, rank 32, n=4/cell) shows Qwen late-rise at layer 27, max |dev| = **0.030** (1.5× the 0.020 gate). The E22 "flat ≈ random / layer 13 argmin" reading was a norm artifact — same class as E23's Llama layer-0 embedding-overlap spike. Qwen outlier flag lifted; staged rank/fp16 ladder (steps 2–3) is **no longer required**. Magnitudes still differ (Qwen 0.030 < Mistral 0.041 < Llama 0.054), but shape is now cross-arch consistent.
- **E18 / E17 / E19 unaffected.** E22 was n=4/cell — exploratory only. Contradiction-vs-control max |diff| ≈ 0.01, within noise. The main run at n=50/cell is what tests those hypotheses.

### Relation to the SUP spectral-band pre-plan
Direct complement to the 2026-04-14 run. The spectral-band metric was dominated by the entropy-collapse confound (Qwen's peak at depth 0.93 coincided with p-sharpness collapse, not Fisher geometry). `null_ratio_ℓ` is insensitive to per-layer p sharpness (normalized by `||Δh||` into a fixed final-p subspace) — so it is the metric that recovers the late-rise shape on Llama and Mistral. **Qwen's apparent direction-depth anomaly dissolved after the 2026-04-18 norm fix** (Prereq 8 step 1), leaving only the v1 cosine inversion (AUROC 0.083; fully resolved by v2) and the spectral-band late-peak-at-depth-0.93 divergence as true Qwen-specific signatures.

---

## Prerequisites
1. ~~Autoresearch loop diagnosed~~ — retired 2026-04-14.
2. ~~Per-variant v2 baseline table~~ — extracted, in `wiki/results/summary.md`.
3. Audit checklist v3 extension — TODO.
4. **Pipeline extension for all-layer hidden-state capture** — **REOPENED 2026-04-18** per Codex adversarial review. Previously marked done by pointing at `scripts/sup_spectral_band.py` (`forward_all_layers`), but that is a one-off helper returning final-position states only. **The shared `hidden_state_collector.py` + `pri_v2_mlx_pipeline.py` production path still lacks:** (a) configurable per-step layer schedule (every-layer for steps 1–12, probe_4 for steps 13+), (b) two-position capture for `h_t` and `h_prev`, (c) `h_prev_source ∈ {"prefix_last", "gen_prev"}` provenance column on every row, (d) the `||Δh_step0|| / ||h_t|| < 10` sanity assertion. **Gate to close this prereq:** implement the above in the production pipeline *and* pass an end-to-end dry run (single puzzle, one of each model, assert schema + provenance + finite-checks before any v3 main-run launch). Do not relaunch the main run until this prereq is closed by the dry-run green build, not by script existence.

   **Dry-run spec — `scripts/v3_capture_dryrun.py`** (added 2026-04-19; this seals what "green" means before writing the script):

   - **Scope:** one puzzle (contradiction cell), one of each primary model (Llama 3B, Mistral 7B, Qwen 2.5 7B 4-bit). Budget ≤ 60s per model. Run via the *shared* production pipeline — no one-off helpers.
   - **Schema assertions (every row written).** Required columns: `run_id`, `git_sha`, `model`, `sample_id`, `condition`, `step`, `layer`, `h_t` (`float32`, shape `(d,)`), `h_prev` (`float32`, shape `(d,)`), `h_prev_source ∈ {"prefix_last", "gen_prev"}`, `capture_schedule_tag` (e.g. `"every_layer_steps_1_12"` / `"probe_4_steps_13+"`). Fail loud on missing / wrong-dtype / wrong-shape fields. No silent column drops.
   - **Schedule assertion.** For steps 1–12 every layer index in `range(n_layers)` must appear. For steps ≥ 13 only the `probe_4` layer set appears and is identical across steps. Count rows per `(step, schedule_tag)` and assert the expected cardinalities — a missing or duplicate layer on any step is a fail.
   - **Provenance assertion.** At `step == 0`: `h_prev_source == "prefix_last"` for **every** row. At `step >= 1`: `h_prev_source == "gen_prev"` for **every** row. Any mix is a pipeline bug — hard fail. (This directly covers regression guard §7b and subsumes M1 by asserting the source flag rather than relying on the ratio tripwire.)
   - **Tripwire assertion (healthy path).** Compute `r = ‖Δh_step0‖ / ‖h_t‖` per row at `step == 0, layer == final`. On a known-healthy input r should sit well below 10. Log the full distribution to `dryrun_report.json` and fail if any row has `r ≥ 10` or is non-finite. (H3 calibration note: once we have ≥ 3 healthy dry-runs across models, replace `< 10` with `< percentile(r_healthy, 99) · 2` — a measured tolerance. Until then, `< 10` is a placeholder bound, flagged as such in the failure message.)
   - **Tripwire assertion (fault-injection).** A second dry-run pass with a deliberately broken `h_prev` source (e.g., zero-vector, or `gen_prev` at step 0) must make **at least one** of: the provenance assertion fires, OR the tripwire fires. If neither fires on the injected fault, the guard is cosmetic — the dry-run itself fails. This closes M1 (the threshold must be shown to catch the named fault) on a worked example.
   - **Finite-checks.** `assert np.isfinite(h_t).all() and np.isfinite(h_prev).all()` per row. NaN / inf anywhere is a hard fail.
   - **Downstream-consumer audit (closes H2).** Before signing off, grep the v3 metric code (`pri_metrics.py`, `PRIComputer` variants) for every column consumed. The validation layer must cover *exactly* the consumed set — no dead checks (validating a column nothing reads), no missing checks (a consumed column un-validated). Record the grep result in `dryrun_report.json` under `consumer_audit`.
   - **Dict-collision check (closes H4).** The capture store (presumed `dict[int | tuple, Tensor]`) must enforce write-once semantics — a duplicate key write is a fail, not a silent overwrite. Add an explicit `assert key not in store` guard with the key printed on violation. The dry-run exercises the store across every layer × every step to surface any key-space overlap; if H4 was a real bug the first puzzle will crash here with a legible message. If it runs clean across all three models, H4 was a mis-reading.
   - **Artifacts.** `dryrun_report.json` (per-model pass/fail, row counts by schedule, tripwire distribution, consumer audit, git SHA, config snapshot) and `dryrun_capture.parquet` (the raw captured rows, kept for post-mortem). Both under `PRI_at_commitment/experiments/v3-capture-dryrun/<date>/run-NN/` (auto-incremented per date).
   - **Exit code.** `0` iff all schema, schedule, provenance, finite, tripwire (healthy + fault-injection), consumer-audit, and dict-collision checks pass on **all three models**. Anything else → `1`, and Prereq 4 stays open.
5. ~~SUP spectral-band pre-plan~~ — **DONE 2026-04-14**, verdict `[SHIFTED]`. E20 demoted, E21 reframed, Option B disfavored, Option C added (see Pre-plan section).
6. ~~**Sharpness-aware Option C exploration**~~ — **DONE 2026-04-17**, verdict `[OPTION-A-REAFFIRMED]`. Llama prototype across α ∈ {0.0, 0.25, 0.5, 1.0} at rank 32. No variant met the primary criterion (|corr(null_ratio, H[p^(ℓ)])| < 0.3); all C variants have a layer-0 embedding-overlap artifact that erases the late-rise structure. **Decisions:** (a) Option A stays default for v3 v0. (b) The "Option B disfavored" framing (from the spectral-band verdict) targeted eigenvalue-spread, which doesn't transfer to the projection-ratio metric `null_ratio` — soften that language in the theory section. (c) If Option B/C is ever revisited, mask or exclude layer 0. Full results: [results/e23-option-c](results/e23-option-c.md).
7. ~~**E22 direction-depth signature gate**~~ — **DONE 2026-04-16**, verdict `[PARTIAL STRUCTURE]`. Llama / Mistral share late-rise shape; Qwen flat ≈ random. Gate decision: every-layer × 12-step capture retained. `argmax_depth` → `argmin_depth` correction applied across plan. Full results: [results/e22-direction-depth](results/e22-direction-depth.md). Pre-plan snapshot folded in above.
8. ~~**Qwen quantization diagnostic (staged rank sweep)**~~ — **CLOSED 2026-04-18** at step 1.
   1. ~~**Primary gate: normed Option A Qwen rerun**~~ — **DONE 2026-04-18, PASSED.** Codex-reviewed script at `scripts/prereq8_qwen_primary_gate.py` (git SHA `6868991` dirty), rank 32, every-layer, n=4/cell, post-final-norm Option A. Qwen `null_ratio_A_rank32` shows clean late-rise at layers 23–27, max |dev from baseline 0.9955| = **0.0302 at layer 27** (final) — 1.5× the 0.020 gate. The E22 "flat / layer 13 argmin" reading was a norm artifact (same bug class as E23's Llama layer-0 spike). Qwen outlier flag lifted; kept in confirmatory cross-model claims. Artifacts: `PRI_at_commitment/experiments/prereq8-qwen-gate/2026-04-18/run-02/` (parquet + manifest.json with config + git SHA).
   2. ~~**Secondary: rank sweep**~~ — **NOT REQUIRED.** Conditional on step 1 flatness; step 1 passed.
   3. ~~**fp16 Qwen replication**~~ — **NOT REQUIRED.** Conditional on steps 1–2 inconclusive; step 1 decisive.

   **Main-run scoping rule (historical):** if step 1 had failed (Qwen still flat after norm-fix) and step 2 were inconclusive, Qwen would have stayed in the main run but been excluded from confirmatory cross-model claims until step 3 resolved. Moot after the 2026-04-18 pass.

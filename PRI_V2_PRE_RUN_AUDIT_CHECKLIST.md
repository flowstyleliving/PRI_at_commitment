# PRI v2 Pipeline — Pre-Run Audit Checklist

You are auditing the PRI v2 experiment pipeline (`pri_v2_mlx_pipeline.py`) before a full experimental run. Go through every section below **in order**. For each item, either confirm it's correct with a brief explanation, or flag the issue with a fix. Do not skip items. Do not assume correctness — trace the actual code path.

Reference materials:
- The attached paper "Prediction Rupture at Commitment" describes PRI v1 semantics and expected results
- The pipeline code is the file to audit

---

## 1. Token/Hidden-State Alignment (CRITICAL)

These are the most dangerous bugs — they silently produce plausible but wrong results.

- [ ] **h_t at step 0**: Trace the generation loop. When `gen_hidden[layer][0]` is captured, has the first generated token already been appended to `token_ids` and processed through `_forward_with_hidden`? The hidden state must be AT the generated token's position, not the last prefix position.

- [ ] **h_prev at step 0**: In `run_experiment`, when `step == 0`, `h_prev` is set to `last_prefix_hidden[layer]`. Confirm this is the hidden state at the last prefix token position (before any generated tokens). Confirm `h_t != h_prev` (they should be from different forward passes or different sequence positions).

- [ ] **h_prev at step k > 0**: Confirm `h_prev = gen_hidden[layer][step - 1]` gives the hidden state at the (k)th generated token, and `h_t = gen_hidden[layer][step]` gives the hidden state at the (k+1)th generated token.

- [ ] **Surprise alignment**: `gen_surprises[k]` should be `-log p(token_{k+1} | context_before_token_{k+1})`. Confirm the probability used comes from the distribution BEFORE the token was appended (i.e., `prev_probs` at the time of selection, not `gen_probs` after the forward pass).

- [ ] **p_t alignment for FIM**: In `compute_step`, `p_t` is `gen_probs[step]`. Confirm this is the output distribution produced by the forward pass that also produced `h_t` — i.e., they come from the same forward pass at the same position.

- [ ] **Prefix hidden state capture**: `last_prefix_hidden[layer]` should be the hidden state at the last token of the prefix (before any generation). Confirm this is `prefix_selected_hidden[layer][0, -1]` (batch 0, last sequence position).

---

## 2. Forward Pass Correctness

- [ ] **`_forward_with_hidden` layer loop**: Confirm that the hidden states are captured BEFORE the final norm layer (i.e., at the output of the transformer block, not after RMSNorm/LayerNorm). This matches standard practice where "final layer hidden state" means the output of the last transformer block.

- [ ] **Layer indexing**: `target_idx_to_name` maps integer layer indices to names. Confirm `get_layer_indices` returns correct indices: `final` = last layer, `mid` = middle, `quarter` = 1/4 depth. Check for off-by-one (e.g., should `final` be `n_layers - 1` or `n_layers`?).

- [ ] **Attention mask**: `create_attention_mask(h, None)` — the `None` for cache is correct for full-context (no KV-cache) forward passes. Confirm no KV-cache is being used (each generation step re-runs the full context).

- [ ] **Logits shape**: `prefix_logits` should be `[1, T, V]` and `step_logits` should be `[1, T+k, V]`. Confirm the code correctly indexes `[0, -1]` for the last position's logits.

- [ ] **No KV-cache leakage**: Confirm that `_forward_with_hidden` does not use or accumulate a KV-cache across generation steps. Each call should be a fresh full-context forward pass.

---

## 3. OutputProjection / Unembedding

- [ ] **`project(dh)`**: This computes `W_u @ dh` (or equivalent). Confirm the output shape is `[vocab_size]`. Confirm the input is the raw difference vector `dh = h_t - h_prev`, not a normalized version.

- [ ] **`get_rows(indices)`**: For the low-rank FIM approximation. Confirm it correctly handles quantized weights (dequantizes selected rows) and returns shape `[len(indices), hidden_size]`.

- [ ] **Tied embeddings**: If the model uses tied embeddings (`mode == "tied_embed"`), confirm `as_linear` correctly transposes the embedding matrix to act as a projection.

---

## 4. PRI Computation

- [ ] **PRI v1 formula**: `S_t * (1 + alpha * delta_h)`. Confirm `delta_h` is cosine distance (1 - cosine_similarity), not cosine similarity.

- [ ] **PRI v2 formula**: `S_t + alpha * d_F`. Confirm this is additive (not multiplicative like v1). This is a design choice — flag if it deviates from the paper's intended v2 formulation.

- [ ] **FIM diagonal**: `sqrt(sum(p * z^2))`. This is the diagonal approximation. Confirm `z` is the projected difference `W_u @ dh` and `p` is the softmax probability vector at step t.

- [ ] **FIM full**: `sqrt(E[z^2] - E[z]^2)` under distribution p. This is the standard deviation of `z` under `p`. Confirm the `max(..., 1e-10)` guard prevents negative values from floating-point error.

- [ ] **FIM top-k**: Confirm the top-k indices are selected from `p_t` (not from `z`), and that probabilities are renormalized after truncation.

- [ ] **FIM low-rank**: Confirm the SVD is performed on `sqrt(p) * W_rows` (probability-weighted rows of the unembedding matrix), and that the rank truncation uses the top-r singular vectors.

- [ ] **Vocab size mismatch guard**: The code checks `z.shape[0] != p_t.shape[0]`. Confirm this guard truncates both to the minimum and renormalizes `p_t`.

---

## 5. Data Generation

- [ ] **2x2 factorial**: Confirm the dataset has 4 cells: {chain_length=1, chain_length=2} x {contradiction=False, contradiction=True}, each with `n_samples_per_cell` samples.

- [ ] **Contradiction injection**: In contradiction puzzles, confirm the contradicting premise is inserted at position 1 (after the first premise), and that it assigns a DIFFERENT value to the same species.

- [ ] **No data leakage across cells**: Confirm the random seed produces different puzzles for each cell (species, names, properties vary). The worked example should be in a disjoint domain from the puzzle.

- [ ] **Shuffling**: Dataset is shuffled after generation. Confirm this doesn't affect the `sample_id` assignment (IDs should be stable across shuffles).

---

## 6. Behavioral Gate

- [ ] **Gate logic**: 20 control (non-contradiction) samples must achieve >= 80% accuracy. Confirm only control samples are used (not contradiction samples).

- [ ] **Gate on unprocessed samples**: If resuming from checkpoint, the gate should run on samples not yet in the checkpoint. Confirm the fallback (if not enough unprocessed controls) uses the first 20 controls overall.

- [ ] **Gate failure**: If a model fails the gate, confirm it is skipped entirely and its partial results are NOT included in `all_results`.

---

## 7. Experiment Loop

- [ ] **Step indexing**: `gen_step` in the output is `step + 1` (1-indexed). Confirm step 1 in the analysis corresponds to the first generated token.

- [ ] **Alpha sweep**: Confirm that for EACH sample and EACH step, ALL alpha values produce separate rows. The analysis filters by `alpha == config.alpha_default` for primary results.

- [ ] **Layer sweep**: Confirm that for EACH sample and EACH step, ALL probed layers produce separate rows.

- [ ] **Checkpoint/resume**: Confirm that resumed checkpoint rows are added to `all_results` and that `processed_sample_ids` correctly prevents re-processing. Confirm checkpoint files are cleaned up after successful completion.

- [ ] **Memory management**: `gc.collect()` and `clear_mlx_cache()` are called periodically and after each model. Confirm models are deleted after use.

---

## 8. Statistical Analysis

- [ ] **Hedges' g direction**: `hedges_g(contradiction, control)` — confirm group1 is contradiction and group2 is control. Positive g means contradiction > control (expected for PRI).

- [ ] **Stratified permutation test**: Confirm permutations are stratified by chain_length, and that the test is one-sided (counting `stat >= obs`).

- [ ] **AUROC**: Confirm labels are `contradiction.astype(int)` (1 = contradiction, 0 = control) and scores are the PRI values. Higher PRI → more likely contradiction → AUROC should be high.

- [ ] **Bootstrap AUC difference**: `score_a` is `pri_v2_full`, `score_b` is `pri_v1_cosine`. Confirm `diff = auc_a - auc_b` (positive means v2 is better).

- [ ] **Outcome independence**: Confirm three groups are computed: control, contradiction-correct, contradiction-incorrect. The key claim is that BOTH contradiction groups are elevated vs control (not just incorrect).

---

## 9. Figures

- [ ] **Figure 1 (AUROC bars)**: Confirm it uses step 1, final layer, default alpha. Blue = v1, Red = v2.

- [ ] **Figure 2 (Trajectory)**: Confirm it shows steps 1-5 for both v1_cosine and v2_full, with error bars (SEM).

- [ ] **Figure 3 (Violins)**: Confirm it shows step 1 distributions split by condition for both v1 and v2.

- [ ] **Figure 4 (Outcome independence)**: Confirm three bars: Control, Contradiction-Correct, Contradiction-Incorrect for v2_full.

- [ ] **Figure 5 (Alpha sweep)**: Confirm x-axis is alpha (log scale), y-axis is AUROC, one line per variant.

---

## 10. Edge Cases & Numerical Stability

- [ ] **Empty generation**: If the model immediately outputs EOS, `gen_hidden` will be empty. Confirm the experiment loop handles `n_steps == 0` gracefully (no crash, no empty rows).

- [ ] **NaN propagation**: If any PRI value is NaN (e.g., from zero-norm vectors), confirm it doesn't corrupt the AUROC or Hedges' g calculations.

- [ ] **Softmax overflow**: `safe_softmax` subtracts the max. Confirm this is applied consistently everywhere softmax is computed.

- [ ] **Log(0) protection**: All `-log(p)` computations use `+ 1e-10`. Confirm no path computes log of a raw probability without this guard.

- [ ] **Division by zero**: Check all denominators for `+ 1e-10` or `+ 1e-12` guards, especially in cosine distance, Hedges' g pooled std, and probability renormalization.

---

## 11. Output & Reproducibility

- [ ] **Seed propagation**: `np.random.seed(cfg.seed)` and `random.seed(cfg.seed)` are set. Confirm the PuzzleGenerator uses its own `random.Random(seed)` for determinism independent of numpy.

- [ ] **File outputs**: Confirm all results are saved as parquet (with CSV fallback). Confirm the summary, per-model results, and figures are all written to `cfg.save_dir`.

- [ ] **Deterministic generation**: Greedy decoding (`argmax`) should be deterministic. Confirm no temperature or sampling is applied.

---

## Summary

After completing all checks, provide:
1. A numbered list of all issues found (if any), ranked by severity
2. For each issue, the exact code location and a concrete fix
3. A final GO / NO-GO recommendation for running the experiment

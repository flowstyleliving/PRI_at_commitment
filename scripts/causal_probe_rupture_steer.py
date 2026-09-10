#!/usr/bin/env python3
"""Causal probe: patch the Fisher rupture direction at commit step.

Intervention: h_commit_post → h_commit_post + alpha * v_top
where v_top  = top-1 right singular vector of sqrt(p_commit) · W_u
      p_commit = first-token probability distribution (from prefix phase)

Records whether the committed YES/NO token changes under various alpha magnitudes.

Steps from plan:
  4.2 pilot  (--mode pilot  --n 5)  — zero-alpha unit test + alpha sweep on 5 samples
  4.3 main   (--mode main   --n 40) — 20 control (label=0) + 20 contradiction (label=1)

Usage
-----
# 4.2 pilot (5 samples, alpha sweep, unit test)
python scripts/causal_probe_rupture_steer.py \\
    --mode pilot \\
    --data experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl \\
    --out experiments/causal-probe/2026-05-25/pilot.json

# 4.3 main (20 control + 20 contradiction)
python scripts/causal_probe_rupture_steer.py \\
    --mode main \\
    --data experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl \\
    --out experiments/causal-probe/2026-05-25/main.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pri_runtime as pipeline
from pri_runtime import (
    PRIComputer,
    OutputProjection,
    _extract_final_rmsnorm_gamma,
    find_layers,
    get_layer_indices,
    trace_sample,
)
from pri_v2_io_plugins import get_prompt_strategy

MODEL_SLUG = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

# Alpha sweep magnitudes for the steer direction.
ALPHA_PILOT = [-100.0, -50.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
ALPHA_MAIN  = [-100.0, -50.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

SUPPORT_SIZE = 256  # top-k tokens for Fisher SVD (matches null_ratio_and_energy)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_v_top(
    p_commit: np.ndarray,
    output_projection: OutputProjection,
    support: int = SUPPORT_SIZE,
) -> Optional[np.ndarray]:
    """Top-1 right singular vector of sqrt(p_s) · W_s (post-norm h-space).

    This is the direction that maximally changes the output distribution
    (in the uncentered Fisher sense). Same computation as null_ratio_and_energy
    at rank=1, but we return Vt[0] instead of the null-ratio scalar.
    """
    support_eff = int(min(max(support, 16), p_commit.shape[0]))
    idx = np.argpartition(-p_commit, kth=support_eff - 1)[:support_eff]
    p_s = p_commit[idx]
    W_s = output_projection.get_rows(idx)
    if W_s is None or W_s.ndim != 2:
        return None
    A = (np.sqrt(p_s + 1e-10)[:, None]) * W_s.astype(np.float64)
    try:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return Vt[0].astype(np.float32)  # shape (d,), in post-norm h-space


def patch_committed_token(
    h_commit_post: np.ndarray,
    v_top: np.ndarray,
    alpha: float,
    output_projection: OutputProjection,
) -> Tuple[int, float]:
    """Return (argmax_token, top_logit) for the patched hidden state."""
    h_patched = (h_commit_post + alpha * v_top).astype(np.float32)
    logits = output_projection.project(h_patched)
    top_tok = int(np.argmax(logits))
    return top_tok, float(logits[top_tok])


def probe_sample(
    model: Any,
    tokenizer: Any,
    projection: OutputProjection,
    layer_indices: Dict[str, int],
    gamma: np.ndarray,
    prompt_strategy: Any,
    prompt: str,
    label: int,
    alphas: List[float],
    yes_no_ids: Optional[Tuple[List[int], List[int]]] = None,
) -> Dict[str, Any]:
    """Run one sample through the probe.

    Returns a dict with the original committed token, v_top l2-norm,
    and per-alpha results (patched token, whether it flipped).
    """
    wrapped = prompt_strategy(prompt, tokenizer)
    trace = trace_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=wrapped,
        layer_indices=layer_indices,
        output_projection=projection,
        max_new_tokens=1,
    )

    # h_commit_pre: pre-final-norm hidden state at last prefix position.
    h_commit_pre = trace["last_prefix_hidden"]["final"]
    # p_commit: probability distribution for the first generated (committed) token.
    p_commit = trace["prefix_probs"][-1]
    # Original committed token (chosen from prefix-phase logits).
    orig_token = int(trace["gen_token_ids"][0]) if trace["gen_token_ids"] else int(np.argmax(p_commit))
    orig_text = tokenizer.decode([orig_token])

    h_commit_post = PRIComputer.rmsnorm(h_commit_pre, gamma)

    # Zero-alpha unit test: alpha=0 patch must reproduce original committed token.
    if 0.0 in alphas:
        logits_zero = projection.project(h_commit_post)
        zero_tok = int(np.argmax(logits_zero))
        unit_test_pass = (zero_tok == orig_token)
    else:
        unit_test_pass = None

    v_top = compute_v_top(p_commit, projection)
    if v_top is None:
        return {
            "label": label,
            "orig_token": orig_token,
            "orig_text": orig_text,
            "v_top_available": False,
            "unit_test_pass": unit_test_pass,
            "alpha_results": [],
        }

    v_top_l2 = float(np.linalg.norm(v_top))

    # Compute null_ratio_post_rank1 for this commit step (link to v3 metric).
    # Requires h_prev = last prefix hidden at gen_step=0 (prefix[-2] position).
    # Use last_prefix_hidden as h_t (step=1 semantics: rupture from prefix end to gen step 1).
    # We don't have h_prev_pre cleanly here, so emit the norm of the direction as a proxy.
    # delta_logit_orig = how much v_top affects the committed token logit (signed).
    logits_orig = projection.project(h_commit_post)
    delta_logit_from_v_top = float(projection.project(v_top)[orig_token])
    logit_gap = float(logits_orig[orig_token] - np.sort(logits_orig)[::-1][1])

    # Classify orig_token as YES/NO/other.
    if yes_no_ids is not None:
        yes_ids, no_ids = yes_no_ids
        if orig_token in yes_ids:
            orig_answer = "YES"
        elif orig_token in no_ids:
            orig_answer = "NO"
        else:
            orig_answer = "OTHER"
    else:
        orig_answer = "UNKNOWN"

    alpha_results = []
    min_flip_alpha = None
    for alpha in alphas:
        patched_tok, patched_top_logit = patch_committed_token(
            h_commit_post, v_top, alpha, projection
        )
        patched_text = tokenizer.decode([patched_tok])
        flipped = (patched_tok != orig_token)
        if flipped and min_flip_alpha is None:
            min_flip_alpha = alpha

        if yes_no_ids is not None:
            yes_ids, no_ids = yes_no_ids
            if patched_tok in yes_ids:
                patched_answer = "YES"
            elif patched_tok in no_ids:
                patched_answer = "NO"
            else:
                patched_answer = "OTHER"
        else:
            patched_answer = "UNKNOWN"

        alpha_results.append({
            "alpha": alpha,
            "patched_token": patched_tok,
            "patched_text": patched_text,
            "patched_answer": patched_answer,
            "flipped": flipped,
        })

    return {
        "label": label,
        "orig_token": orig_token,
        "orig_text": orig_text,
        "orig_answer": orig_answer,
        "v_top_available": True,
        "v_top_l2": v_top_l2,
        "logit_gap": logit_gap,
        "delta_logit_from_v_top": delta_logit_from_v_top,
        "min_flip_alpha": min_flip_alpha,
        "unit_test_pass": unit_test_pass,
        "alpha_results": alpha_results,
    }


def identify_yes_no_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    """Return (yes_ids, no_ids) — token IDs that decode to YES or NO variants.

    Includes single-char 'Y'/'N' which Mistral-family models often generate as
    the first token of a YES/NO answer (before the full word is tokenized).
    """
    yes_variants = ["YES", " YES", "Yes", " Yes", "Y", " Y"]
    no_variants = ["NO", " NO", "No", " No", "N", " N"]
    yes_ids = []
    no_ids = []
    for v in yes_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
    for v in no_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    return list(set(yes_ids)), list(set(no_ids))


def load_data(path: pathlib.Path, mode: str, n_per_label: int, seed: int = 42) -> List[Dict]:
    """Load and filter samples.

    pilot: first n_per_label samples regardless of label (for quick inspection)
    main:  n_per_label label=0 (control) + n_per_label label=1 (contradiction)
    """
    rows = [json.loads(l) for l in open(path)]
    rng = np.random.default_rng(seed)
    if mode == "pilot":
        idx = list(range(min(n_per_label, len(rows))))
        return [rows[i] for i in idx]
    else:  # main
        label0 = [r for r in rows if r["label"] == 0]
        label1 = [r for r in rows if r["label"] == 1]
        rng.shuffle(label0)
        rng.shuffle(label1)
        return label0[:n_per_label] + label1[:n_per_label]


def summarize_flips(results: List[Dict], alphas: List[float]) -> Dict:
    """Per-label, per-alpha flip counts and rates."""
    label0 = [r for r in results if r["label"] == 0 and r["v_top_available"]]
    label1 = [r for r in results if r["label"] == 1 and r["v_top_available"]]

    summary: Dict[str, Any] = {"n_label0": len(label0), "n_label1": len(label1), "by_alpha": []}

    alpha_to_idx = {a: i for i, a in enumerate(alphas)}

    for alpha in alphas:
        ai = alpha_to_idx.get(alpha)
        if ai is None:
            continue

        def _flips(group):
            return sum(1 for r in group if r["alpha_results"][ai]["flipped"])

        def _answer_yes(group):
            return sum(1 for r in group if r["alpha_results"][ai]["patched_answer"] == "YES")

        def _answer_no(group):
            return sum(1 for r in group if r["alpha_results"][ai]["patched_answer"] == "NO")

        n0 = len(label0)
        n1 = len(label1)
        f0 = _flips(label0)
        f1 = _flips(label1)
        summary["by_alpha"].append({
            "alpha": alpha,
            "label0_flip_n": f0,
            "label0_flip_rate": round(f0 / max(n0, 1), 3),
            "label1_flip_n": f1,
            "label1_flip_rate": round(f1 / max(n1, 1), 3),
            "label0_yes_n": _answer_yes(label0),
            "label0_no_n": _answer_no(label0),
            "label1_yes_n": _answer_yes(label1),
            "label1_no_n": _answer_no(label1),
        })

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=5,
                        help="n_per_label (pilot: total n; main: n per label class)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = pathlib.Path(args.data)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alphas = ALPHA_PILOT if args.mode == "pilot" else ALPHA_MAIN

    print(f"[causal-probe] mode={args.mode}  data={data_path}  out={out_path}")
    print(f"[causal-probe] model={MODEL_SLUG}")
    print(f"[causal-probe] alpha_sweep={alphas}")

    data_hash = sha256_file(data_path)
    print(f"[causal-probe] data_hash={data_hash}")

    # Load data
    samples = load_data(data_path, args.mode, args.n, args.seed)
    print(f"[causal-probe] samples={len(samples)} (mode={args.mode})")

    # Load model
    import mlx.core as mx
    model, tokenizer, projection, layer_indices = pipeline.load_model(MODEL_SLUG)
    gamma = _extract_final_rmsnorm_gamma(model)
    if gamma is None:
        raise RuntimeError("Could not extract final RMSNorm gamma from model.")
    prompt_strategy = get_prompt_strategy(MODEL_SLUG)

    yes_ids, no_ids = identify_yes_no_ids(tokenizer)
    print(f"[causal-probe] YES token IDs: {yes_ids}  ({[tokenizer.decode([i]) for i in yes_ids]})")
    print(f"[causal-probe] NO  token IDs: {no_ids}   ({[tokenizer.decode([i]) for i in no_ids]})")

    # 4.2 zero-alpha unit test — run first sample only
    print("\n[causal-probe] === Unit test: zero-alpha must reproduce original committed token ===")
    unit_result = probe_sample(
        model, tokenizer, projection, layer_indices, gamma,
        prompt_strategy,
        samples[0]["prompt"], samples[0]["label"],
        alphas=[0.0],
        yes_no_ids=(yes_ids, no_ids),
    )
    unit_pass = unit_result.get("unit_test_pass")
    print(f"  orig_token={unit_result['orig_token']} ('{unit_result['orig_text']}')  "
          f"unit_test_pass={unit_pass}")
    if not unit_pass:
        print("  *** UNIT TEST FAILED — zero-alpha patch does not reproduce original token ***")
        print("  *** Aborting: patch geometry is wrong. ***")
        sys.exit(1)
    print("  Unit test PASSED.")

    # Run all samples
    print(f"\n[causal-probe] Running {len(samples)} samples...")
    results = []
    t0 = time.time()
    for i, sample in enumerate(samples):
        t1 = time.time()
        r = probe_sample(
            model, tokenizer, projection, layer_indices, gamma,
            prompt_strategy,
            sample["prompt"], sample["label"],
            alphas=alphas,
            yes_no_ids=(yes_ids, no_ids),
        )
        results.append(r)
        elapsed = time.time() - t1
        orig_ans = r.get("orig_answer", "?")
        gap = r.get("logit_gap", float("nan"))
        mfa = r.get("min_flip_alpha")
        mfa_str = f"{mfa:+.0f}" if mfa is not None else "none"
        print(f"  [{i+1}/{len(samples)}] label={sample['label']}  "
              f"orig='{r['orig_text'].strip()}' ({orig_ans})  "
              f"gap={gap:.2f}  min_flip_alpha={mfa_str}  "
              f"elapsed={elapsed:.1f}s")
        # Print flip summary for pilot
        if args.mode == "pilot":
            for ar in r["alpha_results"]:
                flag = "  *** FLIP ***" if ar["flipped"] else ""
                print(f"    alpha={ar['alpha']:+6.1f}  tok='{ar['patched_text'].strip()}'  "
                      f"({ar['patched_answer']}){flag}")

    total_time = time.time() - t0
    print(f"\n[causal-probe] Done: {len(results)} samples in {total_time:.1f}s")

    # Summarize flips (main mode is most informative)
    summary = summarize_flips(results, alphas)
    print("\n[causal-probe] === Flip summary ===")
    print(f"  n_label0 (control/entailment): {summary['n_label0']}")
    print(f"  n_label1 (contradiction):       {summary['n_label1']}")
    print(f"  {'alpha':>8}  {'L0 flips':>10}  {'L0 rate':>8}  {'L1 flips':>10}  {'L1 rate':>8}")
    for row in summary["by_alpha"]:
        print(f"  {row['alpha']:>8.1f}  {row['label0_flip_n']:>10}  "
              f"{row['label0_flip_rate']:>8.3f}  {row['label1_flip_n']:>10}  "
              f"{row['label1_flip_rate']:>8.3f}")

    # Write output
    out = {
        "schema_version": "causal_probe_v1",
        "model": MODEL_SLUG,
        "mode": args.mode,
        "data": str(data_path),
        "data_hash_sha256": data_hash,
        "n_samples": len(results),
        "alphas": alphas,
        "support_size": SUPPORT_SIZE,
        "yes_ids": yes_ids,
        "no_ids": no_ids,
        "results": results,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[causal-probe] Output: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pilot — candidate #9 residual-stream sub-layer friction (DRAFT, raw-friction).

Hypothesis (wiki/research-candidates.md #9): within a block the residual update
is Δh = a + m (a = attention write post-W_o pre-add; m = MLP write pre-add). The
commit tell may live in the *friction* between a and m — which is NOT recoverable
from the sum Δh. This pilot measures RAW friction at the t=0 commit locus (last
prefix-token position) per block and asks the pre-registered question:

  Does friction add discrimination ON TOP OF v3 null_ratio (and beyond mere
  route size / magnitude)?  (the decisive bar)

This is the RAW-FRICTION pilot. The "isolation baseline" residualizer
(InterferenceResidualizer) is intentionally NOT used here — it needs a real
grounded/correctness label to define "normal refinement", which this belief-
labelled data does not provide (see LABEL SEMANTICS). It belongs to the future
correctness-labelled variant.

PRIMARY (locked) endpoint, stated before the run:
  delta_friction_over_null_route — the incremental cross-fit OOF AUROC of
  {null_ratio + route-size + friction} over {null_ratio + route-size}, per
  pre-specified model. SCREEN rule (go/no-go, NOT inference): the PRIMARY
  repeated-CV split-sensitivity interval clears 0 on a pre-specified model AND
  both controls sit at ~0. The inferential verdict comes only from a subsequent
  sealed calibrator nested-OOB run. All other deltas / the marginal table are
  DESCRIPTIVE only (no decision weight; not multiplicity-corrected).

Controls (must hold): shuffled labels -> Δ~0; random-û -> Δ~0 (incremental, at
both null and null+route baselines); route-size-only (delta_route_over_null)
shows how much of any win is mere magnitude — friction must add BEYOND it.

LIMITATION — magnitude/routing control is partial: Xroute holds {mean‖a‖,
mean‖m‖, mean hidden-norm, prefix-len} but NOT a BOS/position-sink term (that
needs gaze capture, not wired here). So a friction win survives norm/length
confounds but is not yet proven free of sink dynamics — flagged for the sealed run.

LABEL SEMANTICS — READ THIS:
  The ANLI label is the BELIEF class (0 = entailment/YES, 1 = contradiction/NO),
  the SAME target v3/ACE are scored against in this project — NOT a correctness/
  hallucination label. This pilot tests "does friction discriminate the commit
  class beyond v3", the project's operationalization. Do NOT phrase any result as
  a "hallucination tell" — that needs the correctness-labelled variant.

INTERVAL CAVEAT: the reported interval is a repeated-stratified-CV SPLIT-
SENSITIVITY interval (split + retraining variance), which the earlier fixed-split
bootstrap ignored. It is NOT an inferential CI — it does not add data-resampling
variance. Treat it as a screening band; the eventual sealed run uses the
calibrator's nested-OOB bootstrap for the real interval.

NOT YET RUN — gated on a codex adversarial review (user instruction 2026-06-06).

Usage (after review clears):
  .venv/bin/python scripts/pilot_residual_friction.py \
      --data experiments/v4-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import pri_calibrator as cal
import pri_runtime as rt
from model_adapters import (
    build_attention_masks,
    layer_supports_sublayer_capture,
    pick_layer_mask,
    post_embed_scale,
)
# Friction primitives from the hardened module (NOT the residualizer).
from friction_residualizer import (
    directed_veto as f_directed_veto,
    interference as f_interference,
)

import mlx.core as mx

SLUG_MAP: Dict[str, str] = {
    "Mistral-7B": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "Qwen2.5-7B": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Gemma-3-4B": "mlx-community/gemma-3-4b-it-4bit",
}
# Family allowlist: only standard Llama-arch pre-norm blocks decompose into a+m.
# (Codex 2026-06-06: attr-check alone is insufficient; gate the pilot explicitly.)
ALLOWED_SLUG_SUBSTR = ("Mistral", "Qwen", "Llama", "gemma-3")

# Friction summary features for the nested-model friction block. Pinned by
# protocol (NOT enforced in code; the sealed run pins them via the calibrator).
PINNED_FRICTION_FEATS = ["mean_interference", "mean_veto", "mean_directed_veto"]
ROUTE_FEATS = ["mean_na", "mean_nm", "mean_hnorm", "prefix_len"]
LAYER_DUMP_FEATS = [
    "interference", "veto", "directed_veto", "rand_directed_veto",
    "cos", "na", "nm", "hnorm",
]


# ── friction primitives ──────────────────────────────────────────────────────
# f_interference + f_directed_veto come from friction_residualizer; cos + veto
# are trivial and defined here (identical to the test_residual_friction contract).

def _cos(a: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a)); nm = float(np.linalg.norm(m))
    return float(a @ m) / (na * nm) if na > eps and nm > eps else 0.0


def _veto(a: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    return -float(a @ m) / na if na > eps else 0.0


def _clip_residual_like(x: mx.array, y: mx.array) -> mx.array:
    """Mirror Gemma 3's clip_residual without importing private MLX-LM symbols."""
    if x.dtype != mx.float16:
        return x + y
    bound = mx.finfo(mx.float16).max
    return mx.clip(x.astype(mx.float32) + y.astype(mx.float32), -bound, bound).astype(mx.float16)


def _layer_supports_candidate9_capture(layer) -> bool:
    if layer_supports_sublayer_capture(layer):
        return True
    return all(
        hasattr(layer, name)
        for name in (
            "self_attn", "mlp", "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
        )
    )


def _candidate9_layer_capture(
    layer,
    h: mx.array,
    mask,
    verify: bool = False,
    verify_atol: float = 1e-3,
    verify_rtol: float = 5e-3,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Return `(out, a, m)` using the layer's actual residual additions.

    Standard Llama/Mistral/Qwen blocks add raw attention and MLP writes:
    `out = h + a + m`. Gemma 3 instead normalizes each sub-layer output before
    the residual add, so its comparable residual writes are the post-norm
    additions, not the raw self-attn/MLP outputs.
    """
    if layer_supports_sublayer_capture(layer) and not hasattr(layer, "pre_feedforward_layernorm"):
        a = layer.self_attn(layer.input_layernorm(h), mask, None)
        h2 = h + a
        m = layer.mlp(layer.post_attention_layernorm(h2))
        out = h2 + m
    elif _layer_supports_candidate9_capture(layer):
        attn_raw = layer.self_attn(layer.input_layernorm(h), mask, None)
        a = layer.post_attention_layernorm(attn_raw)
        h2 = _clip_residual_like(h, a)
        mlp_raw = layer.mlp(layer.pre_feedforward_layernorm(h2))
        m = layer.post_feedforward_layernorm(mlp_raw)
        out = _clip_residual_like(h2, m)
    else:
        raise ValueError(f"layer {type(layer).__name__} does not expose candidate #9 capture components")

    if verify:
        ref = layer(h, mask, None)
        mx.eval(out, ref)
        if not bool(mx.allclose(out, ref, atol=verify_atol, rtol=verify_rtol).all().item()):
            max_abs = float(mx.max(mx.abs(out - ref)).item())
            raise RuntimeError(
                f"candidate9 layer reconstruction mismatch (max|Δ|={max_abs:.3e} "
                f"> atol={verify_atol:.1e}); captured residual writes are invalid for this family."
            )
    return out, a, m


def _native_last_logits(model, token_ids: List[int]) -> np.ndarray:
    """Native full-model logits at the last position (ground-truth forward)."""
    x = mx.array(np.array(token_ids, dtype=np.int32)[None, :])
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    mx.eval(out)
    return rt.to_numpy(out[0, -1]).astype(np.float64)


def _my_last_logits(core, projection, h: mx.array) -> np.ndarray:
    """Final norm + output projection on the manual forward's last hidden,
    mirroring trace_sample's tail — used only for sample-0 native parity."""
    if hasattr(core, "norm"):
        hn = core.norm(h)
    elif hasattr(core, "final_layernorm"):
        hn = core.final_layernorm(h)
    else:
        raise RuntimeError("Could not locate final norm on model core.")
    if projection.mode == "tied_embed":
        logits = projection.layer.as_linear(hn)
    else:
        logits = projection.layer(hn)
    mx.eval(logits)
    return rt.to_numpy(logits[0, -1]).astype(np.float64)


# ── friction forward: capture per-layer a, m at the last prefix position ──────

def _friction_forward(
    model, core, layers, projection, token_ids: List[int],
    verify_layers: bool, verify_native: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Return (a_last, m_last, h_last_norm) per layer at the last prefix position
    (t=0). Two independent guards:
      verify_layers — per-layer assert that manual a+m reconstructs the black-box
        block (the a/m decomposition / model-family check). Expensive; sample 0.
      verify_native — END-TO-END assert that this manual forward's final logits
        match the NATIVE model(x) logits — the only check that catches a wrong
        mask / post-embed / projection path (per-layer verify reuses the same
        mask, so it cannot). Run on a length-SPANNING set of samples so a
        length-dependent mask drift can't hide."""
    x = mx.array(np.array(token_ids, dtype=np.int32)[None, :])
    if hasattr(core, "embed_tokens"):
        h = core.embed_tokens(x)
    elif hasattr(core, "wte"):
        h = core.wte(x)
    else:
        raise RuntimeError("Could not locate token embedding layer on model.")
    h = post_embed_scale(core, h)
    fa_mask, swa_mask = build_attention_masks(core, h)

    a_last: List[np.ndarray] = []
    m_last: List[np.ndarray] = []
    h_norm: List[float] = []
    for layer in layers:
        mask = pick_layer_mask(layer, fa_mask, swa_mask)
        out, a, m = _candidate9_layer_capture(layer, h, mask, verify=verify_layers)
        mx.eval(out, a, m)
        a_last.append(rt.to_numpy(a[0, -1]).astype(np.float64))
        m_last.append(rt.to_numpy(m[0, -1]).astype(np.float64))
        h = out
        h_norm.append(float(np.linalg.norm(rt.to_numpy(h[0, -1]).astype(np.float64))))

    if verify_native:
        mine = _my_last_logits(core, projection, h)
        native = _native_last_logits(model, token_ids)
        rel = float(np.linalg.norm(mine - native) / (np.linalg.norm(native) + 1e-12))
        if not (np.isfinite(rel) and rel <= 5e-3):
            raise RuntimeError(
                f"friction-forward vs native logits mismatch at len={len(token_ids)}: "
                f"rel-L2={rel:.3e} (> 5e-3). The manual mask / post-embed / projection "
                f"path drifts from the native forward — a/m captured off the wrong path."
            )
        print(f"    native-logits parity OK at len={len(token_ids)} (rel-L2={rel:.2e})", flush=True)
    return a_last, m_last, h_norm


# ── per-layer friction → per-sample summaries ────────────────────────────────

def _layer_friction(
    a_last, m_last, h_norm, lo: int, hi: int, rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Per-layer friction over [lo, hi). directed_veto uses û = previous block's
    Δh = a+m (rung-1 neighbour: upstream of this block, NOT independent of a_L
    but not circular-with-this-block's-own-(a+m)); rand_dveto uses a random û."""
    cols = {k: [] for k in ("cos", "interference", "veto", "directed_veto",
                            "rand_directed_veto", "na", "nm", "hnorm")}
    for L in range(lo, hi):
        a, m = a_last[L], m_last[L]
        cols["cos"].append(_cos(a, m))
        cols["interference"].append(f_interference(a, m))
        cols["veto"].append(_veto(a, m))
        cols["na"].append(float(np.linalg.norm(a)))
        cols["nm"].append(float(np.linalg.norm(m)))
        cols["hnorm"].append(float(h_norm[L]))
        if L >= 1:
            u = a_last[L - 1] + m_last[L - 1]  # neighbour-block Δh
            cols["directed_veto"].append(f_directed_veto(a, m, u))
            r = rng.standard_normal(a.shape[0])
            cols["rand_directed_veto"].append(f_directed_veto(a, m, r))
        else:
            cols["directed_veto"].append(np.nan)
            cols["rand_directed_veto"].append(np.nan)
    return {k: np.array(v) for k, v in cols.items()}


def _summarize(lf: Dict[str, np.ndarray]) -> Dict[str, float]:
    def mean(x): return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    def mx_(x):  return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan
    def mn_(x):  return float(np.nanmin(x)) if np.isfinite(x).any() else np.nan
    return {
        "mean_cos": mean(lf["cos"]), "min_cos": mn_(lf["cos"]),
        "mean_interference": mean(lf["interference"]), "max_interference": mx_(lf["interference"]),
        "mean_veto": mean(lf["veto"]), "max_veto": mx_(lf["veto"]),
        "mean_directed_veto": mean(lf["directed_veto"]), "max_directed_veto": mx_(lf["directed_veto"]),
        "mean_rand_directed_veto": mean(lf["rand_directed_veto"]),
        "mean_na": mean(lf["na"]), "mean_nm": mean(lf["nm"]), "mean_hnorm": mean(lf["hnorm"]),
    }


# ── evaluation ───────────────────────────────────────────────────────────────

def _signfree_auroc(scores: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(scores)
    if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
        return float("nan")
    auc = float(roc_auc_score(y[ok], scores[ok]))
    return max(auc, 1.0 - auc)


def _oof_probs(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold P(y=1) from a per-fold standardized logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    oof = np.full(len(y), np.nan)
    finite_row = np.isfinite(X).all(axis=1)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        tr = tr[finite_row[tr]]
        te_ok = te[finite_row[te]]
        if len(np.unique(y[tr])) < 2 or len(tr) < 10 or len(te_ok) == 0:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[tr], y[tr])
        oof[te_ok] = clf.predict_proba(X[te_ok])[:, 1]
    return oof


def _repeated_cv_delta(
    Xa: np.ndarray, Xb: np.ndarray, y: np.ndarray, repeats: int, folds: int, seed: int,
) -> Dict[str, float]:
    """Incremental AUROC of model B over model A via REPEATED stratified CV.
    Each repeat reshuffles folds and refits both models -> captures split +
    training variance (the fixed-split bootstrap ignored both). Returns
    {auc_a, auc_b, delta_median, delta_lo, delta_hi, n_repeats}."""
    from sklearn.metrics import roc_auc_score
    aucs_a, aucs_b, deltas = [], [], []
    for r in range(repeats):
        pa = _oof_probs(Xa, y, folds, seed + r)
        pb = _oof_probs(Xb, y, folds, seed + r)
        ok = np.isfinite(pa) & np.isfinite(pb)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            continue
        ya, aa = y[ok], roc_auc_score(y[ok], pa[ok])
        bb = roc_auc_score(ya, pb[ok])
        aucs_a.append(aa); aucs_b.append(bb); deltas.append(bb - aa)
    if not deltas:
        return {k: float("nan") for k in
                ("auc_a", "auc_b", "delta_median", "delta_lo", "delta_hi")} | {"n_repeats": 0}
    d = np.array(deltas)
    return {
        "auc_a": float(np.mean(aucs_a)), "auc_b": float(np.mean(aucs_b)),
        "delta_median": float(np.median(d)),
        "delta_lo": float(np.percentile(d, 2.5)),
        "delta_hi": float(np.percentile(d, 97.5)),
        "n_repeats": len(deltas),
    }


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def _dump_feature_matrices(
    out_dir: Path,
    slug: str,
    data_path: str,
    prompts: List[str],
    y: np.ndarray,
    layer_lo: int,
    layer_hi: int,
    args,
    *,
    Xnull: np.ndarray,
    Xroute: np.ndarray,
    Xfric: np.ndarray,
    Xrand: np.ndarray,
    summaries: Dict[str, np.ndarray],
    layer_tensor: np.ndarray,
) -> Path:
    """Persist per-sample matrices for offline sealed re-analysis.

    The expensive part of this pilot is the model forward. This dump is the
    handoff point: downstream scripts can run paired/null-centered or OOB
    statistics without touching MLX.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_name(slug)}.residual_friction_features.npz"
    prompt_sha256 = np.array(
        [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts],
        dtype="U64",
    )
    meta = {
        "schema": "residual_friction_features_v2",
        "model_slug": slug,
        "data_path": data_path,
        "n_samples": int(len(y)),
        "label_semantics": "ANLI belief class: 0=entail/YES, 1=contradict/NO",
        "layer_lo_index": int(layer_lo),
        "layer_hi_index": int(layer_hi),
        "layer_lo_fraction": float(args.layer_lo),
        "layer_hi_fraction": float(args.layer_hi),
        "seed": int(args.seed),
        "null_feature_names": ["null_ratio_post_rank1"],
        "route_feature_names": list(ROUTE_FEATS),
        "friction_feature_names": list(PINNED_FRICTION_FEATS),
        "random_feature_names": ["mean_rand_directed_veto"],
        "layer_feature_names": list(LAYER_DUMP_FEATS),
    }
    np.savez_compressed(
        path,
        y=np.asarray(y, dtype=np.int32),
        Xnull=np.asarray(Xnull, dtype=np.float64),
        Xroute=np.asarray(Xroute, dtype=np.float64),
        Xfric=np.asarray(Xfric, dtype=np.float64),
        Xrand=np.asarray(Xrand, dtype=np.float64),
        prompt_sha256=prompt_sha256,
        summary_names=np.array(list(summaries.keys()), dtype=object),
        summary_matrix=np.column_stack([summaries[k] for k in summaries.keys()]).astype(np.float64),
        layer_indices=np.arange(layer_lo, layer_hi, dtype=np.int32),
        layer_feature_names=np.array(LAYER_DUMP_FEATS, dtype=object),
        layer_tensor=np.asarray(layer_tensor, dtype=np.float64),
        metadata=json.dumps(meta, sort_keys=True),
    )
    print(f"    wrote feature dump: {path}", flush=True)
    return path


# ── per-model run ────────────────────────────────────────────────────────────

def run_model(slug: str, prompts: List[str], y: np.ndarray, args, data_path: str = "") -> Dict:
    if not any(s in slug for s in ALLOWED_SLUG_SUBSTR):
        raise SystemExit(
            f"Refusing {slug}: not in the standard-pre-norm allowlist "
            f"{ALLOWED_SLUG_SUBSTR}. a/m decomposition is only validated there."
        )
    print(f"  loading {slug} ...", flush=True)
    state = cal.load_calibration_state(slug)
    lname = state.layer_name
    layers = rt.find_layers(state.model)
    core = state.model.model if hasattr(state.model, "model") else state.model
    n_layers = len(layers)
    if not all(_layer_supports_candidate9_capture(L) for L in layers):
        raise SystemExit(f"{slug}: some layers lack candidate #9 sub-layer components.")

    lo = max(0, min(n_layers - 1, int(round(args.layer_lo * n_layers))))
    hi = max(lo + 1, min(n_layers, int(round(args.layer_hi * n_layers))))
    if not (0 <= lo < hi <= n_layers):
        raise SystemExit(f"bad layer range [{lo},{hi}) for {n_layers} layers")
    print(f"    {n_layers} layers; friction summarized over [{lo}, {hi})", flush=True)

    rng = np.random.default_rng(args.seed)
    summaries: List[Dict[str, float]] = []
    layer_rows: List[np.ndarray] = []
    null_ratio: List[float] = []
    prefix_len: List[int] = []

    # Pre-tokenize to pick a length-SPANNING set for native-logits parity
    # (catches length-dependent mask drift that sample-0-only would miss).
    all_tokens = [rt.encode_text(state.tokenizer, state.prompt_strategy(p, state.tokenizer)) for p in prompts]
    lens = [len(t) for t in all_tokens]
    parity_idx = set(int(j) for j in (0, np.argmin(lens), np.argmax(lens), len(prompts) // 2))
    print(f"    native-parity check on samples {sorted(parity_idx)} "
          f"(prefix lens {sorted(lens[j] for j in parity_idx)})", flush=True)

    for i, prompt in enumerate(prompts):
        token_ids = all_tokens[i]

        # (1) canonical trace → v3 null_ratio at t=0 (apples-to-apples baseline)
        trace = cal._trace_one_prompt(
            state.model, state.tokenizer, state.projection,
            state.layer_indices, prompt, state.prompt_strategy, max_new_tokens=1,
        )
        if list(trace["prefix_token_ids"]) != list(token_ids):
            raise RuntimeError(
                f"token-id mismatch at sample {i}: friction forward and canonical "
                f"trace tokenized differently -> a/m and null_ratio at different "
                f"positions. ({len(token_ids)} vs {len(trace['prefix_token_ids'])})"
            )
        h_t0 = trace["last_prefix_hidden"][lname]
        prefix_seq = trace["prefix_hidden"][lname]
        h_prev0 = prefix_seq[-2] if len(prefix_seq) >= 2 else h_t0
        p_t0 = trace["prefix_probs"][-1]
        gen_surps = trace["gen_surprises"]
        S_commit = float(gen_surps[0]) if len(gen_surps) else 0.0
        r0 = state.pri_computer.compute_step(
            h_t=h_t0, h_prev=h_prev0, p_t=p_t0, S_t=S_commit, alpha=1.0,
            topk_values=[32], lowrank_values=[32], v3_rank_values=[1],
            v3_capture_raw=False, v3_capture_centered=False,
        )
        null_ratio.append(float(r0.get("null_ratio_post_rank1", np.nan)))

        # (2) friction forward → per-layer a, m at t=0 (verify family on sample 0)
        a_last, m_last, h_norm = _friction_forward(
            state.model, core, layers, state.projection, token_ids,
            verify_layers=(i == 0), verify_native=(i in parity_idx))
        lf = _layer_friction(a_last, m_last, h_norm, lo, hi, rng)
        summaries.append(_summarize(lf))
        layer_rows.append(np.column_stack([lf[k] for k in LAYER_DUMP_FEATS]))
        prefix_len.append(len(token_ids))

        if (i + 1) % 25 == 0 or i + 1 == len(prompts):
            print(f"    [{i+1}/{len(prompts)}]", flush=True)

    null_ratio = np.array(null_ratio)
    keys = list(summaries[0].keys())
    S = {k: np.array([s[k] for s in summaries]) for k in keys}
    S["prefix_len"] = np.array(prefix_len, dtype=np.float64)
    layer_tensor = np.stack(layer_rows, axis=0)

    # fail loud on non-finite decisive inputs (don't silently return a "nan result")
    def _check_finite(name, arr):
        frac = float(np.isfinite(arr).all(axis=1).mean()) if arr.ndim == 2 else float(np.isfinite(arr).mean())
        if frac < args.min_finite_frac:
            raise RuntimeError(f"{name}: only {frac:.1%} finite (< {args.min_finite_frac:.0%}); aborting.")

    Xnull = null_ratio[:, None]
    Xfric = np.column_stack([S[k] for k in PINNED_FRICTION_FEATS])
    # Richer magnitude/routing control: norms + residual norm + prefix length.
    Xroute = np.column_stack([S[k] for k in ROUTE_FEATS])
    Xrand = S["mean_rand_directed_veto"][:, None]
    for nm_, X in [("null_ratio", Xnull), ("friction", Xfric), ("route", Xroute)]:
        _check_finite(nm_, X)

    if args.feature_dump_dir:
        _dump_feature_matrices(
            Path(args.feature_dump_dir), slug, data_path, prompts, y, lo, hi, args,
            Xnull=Xnull, Xroute=Xroute, Xfric=Xfric, Xrand=Xrand, summaries=S,
            layer_tensor=layer_tensor,
        )

    marg = {k: _signfree_auroc(S[k], y) for k in keys}
    marg["null_ratio"] = _signfree_auroc(null_ratio, y)

    rp, fo, sd = args.cv_repeats, args.folds, args.seed
    Xnr = np.column_stack([Xnull, Xroute])
    inc = {
        # PRIMARY (locked) endpoint:
        "delta_friction_over_null_route": _repeated_cv_delta(Xnr, np.column_stack([Xnull, Xroute, Xfric]), y, rp, fo, sd),
        # descriptive secondaries:
        "delta_friction_over_null": _repeated_cv_delta(Xnull, np.column_stack([Xnull, Xfric]), y, rp, fo, sd),
        # route-size gain over null — shows how much of any friction win is just magnitude:
        "delta_route_over_null": _repeated_cv_delta(Xnull, Xnr, y, rp, fo, sd),
        # negative controls (incremental, same machinery), at both baselines:
        "delta_rand_over_null": _repeated_cv_delta(Xnull, np.column_stack([Xnull, Xrand]), y, rp, fo, sd),
        "delta_rand_over_null_route": _repeated_cv_delta(Xnr, np.column_stack([Xnr, Xrand]), y, rp, fo, sd),
    }
    yk = y.copy()
    np.random.default_rng(sd + 7).shuffle(yk)
    inc["delta_shuffled_labels"] = _repeated_cv_delta(
        Xnr, np.column_stack([Xnull, Xroute, Xfric]), yk, rp, fo, sd)

    return {"marg": marg, "inc": inc, "n": len(y)}


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "  nan"


def _print_model(name: str, res: Dict) -> None:
    print(f"\n{'='*66}\n  {name}   (n={res['n']})\n{'='*66}")
    print("  marginal sign-free AUROC (DESCRIPTIVE only — sign-free inflates):")
    for k in ["null_ratio", "mean_interference", "max_interference", "mean_veto",
              "max_veto", "mean_cos", "min_cos", "mean_directed_veto",
              "max_directed_veto", "mean_na", "mean_nm", "mean_hnorm"]:
        print(f"    {k:<24s} {_fmt(res['marg'].get(k))}")
    inc = res["inc"]

    def line(key, lab, primary=False):
        d = inc[key]
        tag = " ✅>0" if np.isfinite(d["delta_lo"]) and d["delta_lo"] > 0 else ""
        star = " ★PRIMARY" if primary else ""
        print(f"    {lab:<30s} aucA={_fmt(d['auc_a'])} aucB={_fmt(d['auc_b'])}  "
              f"Δ={_fmt(d['delta_median'])} [{_fmt(d['delta_lo'])},{_fmt(d['delta_hi'])}]"
              f" (r={d['n_repeats']}){tag}{star}")

    print("\n  incremental cross-fit OOF AUROC — repeated-CV SPLIT-SENSITIVITY interval")
    print("  (split+training variance only; NOT an inferential CI — screening, not proof):")
    line("delta_friction_over_null_route", "friction | null+route", primary=True)
    line("delta_friction_over_null", "friction | null  (descr.)")
    line("delta_route_over_null", "route-size | null  (descr.)")
    line("delta_rand_over_null", "random-û | null  (ctrl→0)")
    line("delta_rand_over_null_route", "random-û | null+route  (ctrl→0)")
    line("delta_shuffled_labels", "shuffled-labels  (ctrl→0)")
    print("\n  Screen (go/no-go, not inference): PRIMARY ★ interval clears 0 AND both")
    print("  controls ~0  →  promote to a sealed calibrator nested-OOB run for the real CI.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="experiments/v4-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl")
    ap.add_argument("--models", nargs="+", default=list(SLUG_MAP.keys()))
    ap.add_argument("--layer-lo", type=float, default=0.25, help="layer-range start fraction (pinned)")
    ap.add_argument("--layer-hi", type=float, default=0.75, help="layer-range end fraction (pinned)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=200, help="repeated-CV repeats for the CI")
    ap.add_argument("--min-finite-frac", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument(
        "--feature-dump-dir",
        default="",
        help="optional directory for per-sample .npz feature matrices; enables offline sealed re-analysis without rerunning MLX",
    )
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    prompts = [r["prompt"] for r in rows]
    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    print(f"Loaded {len(prompts)} samples (pos={int((y==1).sum())}, neg={int((y==0).sum())})")
    print(f"Data: {args.data}")
    print("NOTE: label = belief class (0=entail/YES, 1=contradict/NO), not correctness.\n")

    for name in args.models:
        slug = SLUG_MAP.get(name, name)
        print(f"\n>>> {name}", flush=True)
        res = run_model(slug, prompts, y, args, data_path=args.data)
        _print_model(name, res)
        import gc
        gc.collect()
        try:
            rt.clear_mlx_cache()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pilot — v6 Projection Veto (readout-space attention/MLP conflict).

v5 residual friction mostly measured benign cancellation / residual norm budget.
v6 asks the same a/m question only after projecting onto answer-relevant readout
directions:

    p_c(x) = <u_c, x>
    projection_veto = -p_c(a) * p_c(m)

Primary contrast in this pilot is a frozen YES-vs-NO token-bucket direction from
W_u. This is W_u-dependent by design: we are testing whether attention and MLP
fight over the steering wheel, not whether they fight somewhere in residual
space.

SCREEN ONLY: repeated-CV split-sensitivity intervals are not inferential CIs.
The sealed path remains pri_calibrator.py nested-OOB if a v6 statistic survives
same-Delta and projection-budget controls.
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
from pilot_residual_friction import (  # reuse the hardened a/m forward path
    ALLOWED_SLUG_SUBSTR,
    ROUTE_FEATS,
    _candidate9_layer_capture,  # noqa: F401  (import asserts path remains importable)
    _friction_forward,
    _layer_supports_candidate9_capture,
    _native_last_logits,  # noqa: F401
    _repeated_cv_delta,
    _signfree_auroc,
)


SLUG_MAP: Dict[str, str] = {
    "Qwen2.5-7B": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Qwen3-8B": "mlx-community/Qwen3-8B-4bit",
    "Llama-3.2-3B": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "Llama-3.1-8B": "mlx-community/Llama-3.1-8B-Instruct-4bit",
}

PINNED_PROJ_FEATS = [
    "mean_proj_veto",
    "mean_proj_trim",
    "mean_proj_amplify",
]
PROJ_BUDGET_FEATS = [
    "mean_abs_pa",
    "mean_abs_pm",
    "mean_abs_pnet",
    "mean_proj_path",
    "mean_proj_trim",
    "mean_proj_amplify",
]
LAYER_DUMP_FEATS = [
    "pa",
    "pm",
    "pnet",
    "proj_veto",
    "same_delta_proj_veto",
    "proj_path",
    "proj_trim",
    "proj_amplify",
    "na",
    "nm",
    "delta_norm",
    "hnorm",
]


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


def _token_bucket(tokenizer, variants: List[str]) -> Tuple[np.ndarray, List[str]]:
    ids: List[int] = []
    decoded: List[str] = []
    seen = set()
    for text in variants:
        enc = rt.encode_text(tokenizer, text)
        if not enc:
            continue
        # Keep the token that actually carries the answer string. Prefix
        # variants such as "\nYES" may tokenize as ["\n", "YES"]; including
        # the standalone newline in both YES and NO buckets corrupts the
        # contrast. The final token is the answer-bearing piece for the variants
        # we enumerate here.
        tok = int(enc[-1])
        dec = rt.decode_ids(tokenizer, [tok])
        if not dec.strip():
            continue
        if tok in seen:
            continue
        seen.add(tok)
        ids.append(tok)
        decoded.append(dec)
    if not ids:
        raise RuntimeError(f"empty token bucket for variants={variants}")
    return np.asarray(ids, dtype=np.int32), decoded


def _yes_no_contrast(projection, tokenizer) -> Tuple[np.ndarray, Dict[str, object]]:
    # Include bare and whitespace-prefixed forms; chat tokenizers vary on whether
    # "YES" is one token or split into leading-space + token.
    yes_ids, yes_dec = _token_bucket(tokenizer, ["YES", " Yes", " yes", "\nYES", "\nYes"])
    no_ids, no_dec = _token_bucket(tokenizer, ["NO", " No", " no", "\nNO", "\nNo"])
    yes_rows = projection.get_rows(yes_ids)
    no_rows = projection.get_rows(no_ids)
    if yes_rows is None or no_rows is None:
        raise RuntimeError("OutputProjection.get_rows failed for YES/NO buckets")
    # Positive direction = NO minus YES, matching ANLI label 1 = contradiction/NO.
    u = no_rows.mean(axis=0) - yes_rows.mean(axis=0)
    nu = float(np.linalg.norm(u))
    if not np.isfinite(nu) or nu <= 1e-12:
        raise RuntimeError("degenerate YES-vs-NO projection direction")
    meta = {
        "contrast": "NO_bucket_minus_YES_bucket",
        "yes_token_ids": yes_ids.astype(int).tolist(),
        "no_token_ids": no_ids.astype(int).tolist(),
        "yes_decoded": yes_dec,
        "no_decoded": no_dec,
        "direction_norm": nu,
    }
    return (u / nu).astype(np.float64), meta


def _layer_projection_features(
    a_last: List[np.ndarray],
    m_last: List[np.ndarray],
    h_norm: List[float],
    lo: int,
    hi: int,
    uhat: np.ndarray,
) -> Dict[str, np.ndarray]:
    cols = {k: [] for k in LAYER_DUMP_FEATS}
    for L in range(lo, hi):
        a = np.asarray(a_last[L], dtype=np.float64)
        m = np.asarray(m_last[L], dtype=np.float64)
        d = a + m
        na = float(np.linalg.norm(a))
        nm = float(np.linalg.norm(m))
        nd = float(np.linalg.norm(d))
        pa = float(a @ uhat)
        pm = float(m @ uhat)
        pnet = pa + pm
        path = abs(pa) + abs(pm)
        trim = path - abs(pnet)
        veto = -pa * pm
        same = -0.25 * pnet * pnet
        amplify = path / (na + nm + 1e-12)
        cols["pa"].append(pa)
        cols["pm"].append(pm)
        cols["pnet"].append(pnet)
        cols["proj_veto"].append(veto)
        cols["same_delta_proj_veto"].append(same)
        cols["proj_path"].append(path)
        cols["proj_trim"].append(trim)
        cols["proj_amplify"].append(amplify)
        cols["na"].append(na)
        cols["nm"].append(nm)
        cols["delta_norm"].append(nd)
        cols["hnorm"].append(float(h_norm[L]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _summarize(pf: Dict[str, np.ndarray]) -> Dict[str, float]:
    def mean(x): return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
    def mx(x): return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan
    return {
        "mean_pa": mean(pf["pa"]),
        "mean_pm": mean(pf["pm"]),
        "mean_pnet": mean(pf["pnet"]),
        "mean_abs_pa": mean(np.abs(pf["pa"])),
        "mean_abs_pm": mean(np.abs(pf["pm"])),
        "mean_abs_pnet": mean(np.abs(pf["pnet"])),
        "mean_proj_veto": mean(pf["proj_veto"]),
        "max_proj_veto": mx(pf["proj_veto"]),
        "mean_same_delta_proj_veto": mean(pf["same_delta_proj_veto"]),
        "mean_proj_path": mean(pf["proj_path"]),
        "mean_proj_trim": mean(pf["proj_trim"]),
        "mean_proj_amplify": mean(pf["proj_amplify"]),
        "mean_na": mean(pf["na"]),
        "mean_nm": mean(pf["nm"]),
        "mean_hnorm": mean(pf["hnorm"]),
        "mean_delta_norm": mean(pf["delta_norm"]),
    }


def _check_finite(name: str, X: np.ndarray, min_frac: float) -> None:
    frac = float(np.isfinite(X).all(axis=1).mean()) if X.ndim == 2 else float(np.isfinite(X).mean())
    if frac < min_frac:
        raise RuntimeError(f"{name}: only {frac:.1%} finite (< {min_frac:.0%}); aborting.")


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
    Xproj: np.ndarray,
    Xsame: np.ndarray,
    Xbudget: np.ndarray,
    summaries: Dict[str, np.ndarray],
    layer_tensor: np.ndarray,
    contrast_meta: Dict[str, object],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_safe_name(slug)}.projection_veto_features.npz"
    prompt_sha256 = np.array([hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts], dtype="U64")
    meta = {
        "schema": "projection_veto_features_v1",
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
        "projection_feature_names": list(PINNED_PROJ_FEATS),
        "same_delta_feature_names": ["mean_proj_trim", "mean_proj_amplify", "mean_same_delta_proj_veto"],
        "projection_budget_feature_names": list(PROJ_BUDGET_FEATS),
        "layer_feature_names": list(LAYER_DUMP_FEATS),
        "contrast": contrast_meta,
    }
    np.savez_compressed(
        path,
        y=np.asarray(y, dtype=np.int32),
        Xnull=np.asarray(Xnull, dtype=np.float64),
        Xroute=np.asarray(Xroute, dtype=np.float64),
        Xproj=np.asarray(Xproj, dtype=np.float64),
        Xsame=np.asarray(Xsame, dtype=np.float64),
        Xbudget=np.asarray(Xbudget, dtype=np.float64),
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


def run_model(slug: str, prompts: List[str], y: np.ndarray, args, data_path: str = "") -> Dict:
    if not any(s in slug for s in ALLOWED_SLUG_SUBSTR):
        raise SystemExit(f"Refusing {slug}: not in allowlist {ALLOWED_SLUG_SUBSTR}")
    print(f"  loading {slug} ...", flush=True)
    state = cal.load_calibration_state(slug)
    lname = state.layer_name
    layers = rt.find_layers(state.model)
    core = state.model.model if hasattr(state.model, "model") else state.model
    n_layers = len(layers)
    if not all(_layer_supports_candidate9_capture(L) for L in layers):
        raise SystemExit(f"{slug}: some layers lack v6 a/m capture components.")

    lo = max(0, min(n_layers - 1, int(round(args.layer_lo * n_layers))))
    hi = max(lo + 1, min(n_layers, int(round(args.layer_hi * n_layers))))
    print(f"    {n_layers} layers; projection veto summarized over [{lo}, {hi})", flush=True)

    uhat, contrast_meta = _yes_no_contrast(state.projection, state.tokenizer)
    print(f"    contrast {contrast_meta['contrast']} norm={contrast_meta['direction_norm']:.3f}", flush=True)
    print(f"      YES ids={contrast_meta['yes_token_ids']} decoded={contrast_meta['yes_decoded']}", flush=True)
    print(f"      NO  ids={contrast_meta['no_token_ids']} decoded={contrast_meta['no_decoded']}", flush=True)

    summaries: List[Dict[str, float]] = []
    layer_rows: List[np.ndarray] = []
    null_ratio: List[float] = []
    prefix_len: List[int] = []

    all_tokens = [rt.encode_text(state.tokenizer, state.prompt_strategy(p, state.tokenizer)) for p in prompts]
    lens = [len(t) for t in all_tokens]
    parity_idx = set(int(j) for j in (0, np.argmin(lens), np.argmax(lens), len(prompts) // 2))
    print(f"    native-parity check on samples {sorted(parity_idx)} "
          f"(prefix lens {sorted(lens[j] for j in parity_idx)})", flush=True)

    for i, prompt in enumerate(prompts):
        token_ids = all_tokens[i]
        trace = cal._trace_one_prompt(
            state.model, state.tokenizer, state.projection,
            state.layer_indices, prompt, state.prompt_strategy, max_new_tokens=1,
        )
        if list(trace["prefix_token_ids"]) != list(token_ids):
            raise RuntimeError(f"token-id mismatch at sample {i}")
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

        a_last, m_last, h_norm = _friction_forward(
            state.model, core, layers, state.projection, token_ids,
            verify_layers=(i == 0), verify_native=(i in parity_idx))
        pf = _layer_projection_features(a_last, m_last, h_norm, lo, hi, uhat)
        summaries.append(_summarize(pf))
        layer_rows.append(np.column_stack([pf[k] for k in LAYER_DUMP_FEATS]))
        prefix_len.append(len(token_ids))

        if (i + 1) % 25 == 0 or i + 1 == len(prompts):
            print(f"    [{i+1}/{len(prompts)}]", flush=True)

    null_ratio_arr = np.asarray(null_ratio, dtype=np.float64)
    keys = list(summaries[0].keys())
    S = {k: np.asarray([s[k] for s in summaries], dtype=np.float64) for k in keys}
    S["prefix_len"] = np.asarray(prefix_len, dtype=np.float64)
    layer_tensor = np.stack(layer_rows, axis=0)

    Xnull = null_ratio_arr[:, None]
    Xroute = np.column_stack([S[k] for k in ROUTE_FEATS])
    Xproj = np.column_stack([S[k] for k in PINNED_PROJ_FEATS])
    Xsame = np.column_stack([S["mean_proj_trim"], S["mean_proj_amplify"], S["mean_same_delta_proj_veto"]])
    Xbudget = np.column_stack([S[k] for k in PROJ_BUDGET_FEATS])
    for nm, X in [("null", Xnull), ("route", Xroute), ("projection", Xproj),
                  ("same_delta", Xsame), ("projection_budget", Xbudget)]:
        _check_finite(nm, X, args.min_finite_frac)

    if args.feature_dump_dir:
        _dump_feature_matrices(
            Path(args.feature_dump_dir), slug, data_path, prompts, y, lo, hi, args,
            Xnull=Xnull, Xroute=Xroute, Xproj=Xproj, Xsame=Xsame, Xbudget=Xbudget,
            summaries=S, layer_tensor=layer_tensor, contrast_meta=contrast_meta,
        )

    base = np.column_stack([Xnull, Xroute])
    base_budget = np.column_stack([base, Xbudget])
    rp, fo, sd = args.cv_repeats, args.folds, args.seed
    inc = {
        "delta_proj_over_null_route": _repeated_cv_delta(base, np.column_stack([base, Xproj]), y, rp, fo, sd),
        "delta_same_over_null_route": _repeated_cv_delta(base, np.column_stack([base, Xsame]), y, rp, fo, sd),
        "delta_budget_over_null_route": _repeated_cv_delta(base, base_budget, y, rp, fo, sd),
        "delta_proj_over_null_route_budget": _repeated_cv_delta(base_budget, np.column_stack([base_budget, Xproj]), y, rp, fo, sd),
        "delta_proj_over_null_route_same": _repeated_cv_delta(np.column_stack([base, Xsame]), np.column_stack([base, Xsame, Xproj]), y, rp, fo, sd),
    }
    yk = y.copy()
    np.random.default_rng(sd + 7).shuffle(yk)
    inc["delta_shuffled_labels"] = _repeated_cv_delta(base, np.column_stack([base, Xproj]), yk, rp, fo, sd)
    marg = {k: _signfree_auroc(S[k], y) for k in keys}
    marg["null_ratio"] = _signfree_auroc(null_ratio_arr, y)
    return {"marg": marg, "inc": inc, "n": len(y), "contrast": contrast_meta}


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else "  nan"


def _print_model(name: str, res: Dict) -> None:
    print(f"\n{'='*72}\n  {name}   (n={res['n']})\n{'='*72}")
    print("  marginal sign-free AUROC (descriptive):")
    for k in [
        "null_ratio", "mean_proj_veto", "max_proj_veto", "mean_same_delta_proj_veto",
        "mean_proj_trim", "mean_proj_amplify", "mean_abs_pa", "mean_abs_pm",
        "mean_abs_pnet", "mean_na", "mean_nm", "mean_delta_norm",
    ]:
        print(f"    {k:<30s} {_fmt(res['marg'].get(k))}")

    def line(key, label):
        d = res["inc"][key]
        tag = " ✅>0" if np.isfinite(d["delta_lo"]) and d["delta_lo"] > 0 else ""
        print(f"    {label:<38s} aucA={_fmt(d['auc_a'])} aucB={_fmt(d['auc_b'])}  "
              f"Δ={_fmt(d['delta_median'])} [{_fmt(d['delta_lo'])},{_fmt(d['delta_hi'])}]"
              f" (r={d['n_repeats']}){tag}")

    print("\n  incremental cross-fit OOF AUROC — repeated-CV split-sensitivity:")
    line("delta_proj_over_null_route", "projection-veto | null+route")
    line("delta_same_over_null_route", "same-Δ projection | null+route")
    line("delta_budget_over_null_route", "projection-budget | null+route")
    line("delta_proj_over_null_route_budget", "projection-veto | null+route+budget")
    line("delta_proj_over_null_route_same", "projection-veto | null+route+same-Δ")
    line("delta_shuffled_labels", "shuffled-labels")
    real = res["inc"]["delta_proj_over_null_route"]["delta_median"]
    same = res["inc"]["delta_same_over_null_route"]["delta_median"]
    print(f"\n  same-Δ net projection-veto Δ = {real - same:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="experiments/v4-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl")
    ap.add_argument("--models", nargs="+", default=["Qwen2.5-7B"])
    ap.add_argument("--layer-lo", type=float, default=0.25)
    ap.add_argument("--layer-hi", type=float, default=0.75)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=200)
    ap.add_argument("--min-finite-frac", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--feature-dump-dir", default="")
    ap.add_argument("--limit", type=int, default=0, help="debug: use first N samples only")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    prompts = [r["prompt"] for r in rows]
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
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
    raise SystemExit(main())

#!/usr/bin/env python3
"""Reconstruct the per-layer friction profile from a pilot `.npz` feature dump.

Candidate #9 (residual-stream sub-layer friction). The pilot
(`pilot_residual_friction.py`) writes per-sample matrices to a `.npz` when run
with `--feature-dump-dir`; this script consumes that dump OFFLINE (no MLX) and
emits the `layer_profile.txt` diagnostic: which individual block / which 3-layer
window carries the incremental friction signal over the global null+route
baseline.

Provenance: run-03/04/05 (2026-06-06) shipped a `layer_profile.txt` produced by
an ephemeral, un-committed Codex script. This is the committed reconstruction —
it reuses the pilot's own `_repeated_cv_delta` / `_signfree_auroc` so the numbers
match byte-for-byte (validated against run-03's Qwen2.5 profile). Closes that
reproducibility gap and makes profiles regenerable for every model from its dump.

Per-layer columns:
  raw_delta lo hi  — incremental split-sensitivity Δ (and 95% band) of THIS
                     layer's friction triplet {interference, veto, directed_veto}
                     added to the GLOBAL [null_ratio + route] baseline.
  rand_delta       — same, but adding this layer's random-û directed_veto.
  net              — raw_delta − rand_delta (random-control-subtracted).
  veto/interf/dveto_auc — marginal sign-free AUROC of each single-layer metric.

SCREEN-ONLY: these are repeated-CV split-sensitivity intervals, NOT inferential
CIs (same caveat as the pilot panel). The real CI is the sealed calibrator's
nested-OOB bootstrap.

Usage:
  .venv/bin/python scripts/analyze_friction_layer_profile.py \
      experiments/residual-friction/2026-06-06/run-06/features/*.npz
  # writes layer_profile.txt beside each dump's run dir (or --out PATH for one).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from pilot_residual_friction import _repeated_cv_delta, _signfree_auroc  # noqa: E402

# layer_tensor feature column order (pilot LAYER_DUMP_FEATS):
F_INTERF, F_VETO, F_DVETO, F_RANDDVETO = 0, 1, 2, 3
TRIPLET = [F_INTERF, F_VETO, F_DVETO]


def _profile(npz_path: Path, seed: int, repeats: int, folds: int) -> str:
    d = np.load(npz_path, allow_pickle=True)
    y = d["y"]
    Xnull = np.asarray(d["Xnull"], dtype=np.float64)
    Xroute = np.asarray(d["Xroute"], dtype=np.float64)
    LT = np.asarray(d["layer_tensor"], dtype=np.float64)  # (n, L, 8)
    layer_idx = [int(i) for i in d["layer_indices"]]
    feat_names = [str(f) for f in d["layer_feature_names"]]
    schema = "residual_friction_features_v2"
    base = np.column_stack([Xnull, Xroute])

    lines: List[str] = []
    lines.append(f"dump_schema {schema} tensor {LT.shape} features {feat_names}")

    # baseline [null+route] OOF AUROC (auc_a is identical across the per-layer
    # calls; grab it from the first one below and back-fill the header).
    header_idx = len(lines)
    lines.append("")  # placeholder for baseline_auc_approx

    lines.append("")
    lines.append("Per-layer friction triplet over global null+route; net subtracts same-layer random-u increment")
    lines.append("layer raw_delta lo hi rand_delta net veto_auc interf_auc dveto_auc")
    base_auc = float("nan")
    nL = LT.shape[1]
    for j in range(nL):
        Xtrip = np.column_stack([base, LT[:, j, TRIPLET]])
        Xrand = np.column_stack([base, LT[:, j, F_RANDDVETO]])
        raw = _repeated_cv_delta(base, Xtrip, y, repeats, folds, seed)
        rnd = _repeated_cv_delta(base, Xrand, y, repeats, folds, seed)
        if not np.isfinite(base_auc):
            base_auc = raw["auc_a"]
        net = raw["delta_median"] - rnd["delta_median"]
        va = _signfree_auroc(LT[:, j, F_VETO], y)
        ia = _signfree_auroc(LT[:, j, F_INTERF], y)
        da = _signfree_auroc(LT[:, j, F_DVETO], y)
        lines.append(
            f"{layer_idx[j]:02d} {raw['delta_median']:+.4f} {raw['delta_lo']:+.4f} "
            f"{raw['delta_hi']:+.4f} {rnd['delta_median']:+.4f} {net:+.4f} "
            f"{va:.4f} {ia:.4f} {da:.4f}"
        )

    lines.append("")
    lines.append("3-layer sliding windows; friction block flattened, random block matched by one rand-u column per layer")
    lines.append("window raw_delta lo hi rand_delta net")
    for j in range(nL - 2):
        sl = slice(j, j + 3)
        Xtrip = np.column_stack([base, LT[:, sl, TRIPLET].reshape(len(y), -1)])
        Xrand = np.column_stack([base, LT[:, sl, F_RANDDVETO]])
        raw = _repeated_cv_delta(base, Xtrip, y, repeats, folds, seed)
        rnd = _repeated_cv_delta(base, Xrand, y, repeats, folds, seed)
        net = raw["delta_median"] - rnd["delta_median"]
        lines.append(
            f"{layer_idx[j]:02d}-{layer_idx[j+2]:02d} {raw['delta_median']:+.4f} "
            f"{raw['delta_lo']:+.4f} {raw['delta_hi']:+.4f} {rnd['delta_median']:+.4f} {net:+.4f}"
        )

    lines[header_idx] = f"baseline_auc_approx {base_auc:.3f}"
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", help="pilot feature-dump .npz file(s)")
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--cv-repeats", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="", help="write to this path (single input only); "
                    "default writes layer_profile.txt in each dump's parent run dir")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing files")
    args = ap.parse_args()

    for p in args.npz:
        path = Path(p)
        text = _profile(path, args.seed, args.cv_repeats, args.folds)
        if args.stdout:
            print(f"# === {path} ===")
            print(text)
        else:
            out = Path(args.out) if args.out else path.parent.parent / "layer_profile.txt"
            out.write_text(text)
            print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

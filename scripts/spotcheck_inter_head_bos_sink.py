#!/usr/bin/env python3
"""Small BOS-sink spot-check for inter-head JS-radius.

Replays a small slice of samples, captures the target layer's gen_step=1
attention row, and compares raw AUROC for:
  1. full JS-radius over all key positions
  2. BOS-stripped JS-radius after dropping key position 0 and renormalizing

Intended as a cheap falsification pass before a full panel rerun.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPO_ROOT))

import pri_v2_io_plugins as io_plugins
import pri_v2_mlx_pipeline as pipeline
from pri_calibrator import _load_calibration_jsonl
from scripts.diagnose_inter_head_disagreement import (
    _find_layers,
    _js_radius,
    _js_radius_no_bos,
    _raw_auroc,
    _target_layer_map,
    attention_capture,
)


def main() -> int:
    p = argparse.ArgumentParser(description="BOS-sink spot-check for inter-head JS-radius")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--layer", choices=("final", "mid", "last_minus_1"), default="final")
    args = p.parse_args()

    cfg = pipeline.Config()
    cfg.layers_to_probe = ["final"]
    cfg.v3_capture = False
    model, tokenizer, projection, layer_indices = pipeline.load_model(args.model, cfg)
    layers = _find_layers(model)
    target = {args.layer: _target_layer_map(len(layers))[args.layer]}
    prompts, labels, _ = _load_calibration_jsonl(args.data)
    prompt_strategy = io_plugins.get_prompt_strategy(args.model)

    ys: List[int] = []
    full_scores: List[float] = []
    no_bos_scores: List[float] = []
    for i, prompt in enumerate(prompts[: args.limit]):
        wrapped = prompt_strategy(prompt, tokenizer)
        with attention_capture(layers, target) as captures:
            pipeline.trace_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=wrapped,
                layer_indices=layer_indices,
                output_projection=projection,
                max_new_tokens=1,
                v3_capture=False,
            )
        caps = captures[args.layer]
        if len(caps) < 2:
            continue
        w = caps[1]
        ys.append(int(labels[i]))
        full_scores.append(_js_radius(w))
        no_bos_scores.append(_js_radius_no_bos(w))

    y = np.array(ys, dtype=np.float64)
    full = np.array(full_scores, dtype=np.float64)
    no_bos = np.array(no_bos_scores, dtype=np.float64)
    full_a, full_s = _raw_auroc(y, full)
    nobos_a, nobos_s = _raw_auroc(y, no_bos)
    delta = no_bos - full
    delta_mean = float(np.nanmean(delta)) if delta.size else float("nan")

    print(f"model={args.model}")
    print(f"layer={args.layer}  usable={len(ys)}/{min(args.limit, len(prompts))}")
    print(f"full_js_raw_auroc={full_a:.4f}  dir={full_s}")
    print(f"no_bos_js_raw_auroc={nobos_a:.4f}  dir={nobos_s}")
    print(f"mean(no_bos - full)={delta_mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

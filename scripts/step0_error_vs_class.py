#!/usr/bin/env python3
"""Step-0 gate for the v4 mechanistic vertebra.

Decides whether the case-study dependent variable is (A) model-error or
(B) label-class — by measuring the joint 2x2 [correct/wrong x
contradiction(label1)/consistent(label0)] for the candidate models on the
PINNED ANLI R1 n=200 slice. Reuses trace_sample + io_plugins.parse_yes_no;
NO attention capture wrapper (fp32 mic bug not in this path).

Gold mapping (VERIFIED against decoded samples in the output, not asserted
blind): prompt instructs "YES if entails, NO if contradicts"; calibrator
label 1 = contradiction => gold "NO"; label 0 = consistent => gold "YES".

Usage: .venv/bin/python scripts/step0_error_vs_class.py --model <slug> --out <json>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import pri_v2_io_plugins as io_plugins
import pri_v2_mlx_pipeline as pipeline
from pri_calibrator import _load_calibration_jsonl

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default=str(REPO/"experiments/anli-sweep/2026-05-15/run-02/anli_R1_seed20260513_n100.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--max-new-tokens", type=int, default=6)
    a = p.parse_args()
    prompts, labels, dh = _load_calibration_jsonl(a.data)
    cfg = pipeline.Config(); cfg.layers_to_probe = ["final"]; cfg.v3_capture = False
    model, tok, proj, li = pipeline.load_model(a.model, cfg)
    strat = io_plugins.get_prompt_strategy(a.model)
    cell = {(c, l): 0 for c in ("correct","wrong","abstain") for l in (0,1)}
    samples = []
    for i,(pr,lab) in enumerate(zip(prompts, labels)):
        lab = int(lab)
        try:
            tr = pipeline.trace_sample(model=model, tokenizer=tok, prompt=strat(pr,tok),
                layer_indices=li, output_projection=proj, max_new_tokens=a.max_new_tokens, v3_capture=False)
        except Exception as e:
            print(f"[s0] {i} trace FAIL {e}"); continue
        gen = tr.get("generated_text") or ""
        pred = io_plugins.parse_yes_no(gen)            # "YES"/"NO"/None
        gold = "NO" if lab == 1 else "YES"
        status = "abstain" if pred is None else ("correct" if pred == gold else "wrong")
        cell[(status, lab)] += 1
        if i < 8:
            samples.append({"idx": i, "label": lab, "gold": gold, "pred": pred,
                            "raw": gen[:120].replace(chr(10)," ")})
        if (i+1) % 50 == 0: print(f"[s0] {a.model} {i+1}/{len(prompts)}")
    n = len(prompts)
    wrong = cell[("wrong",0)] + cell[("wrong",1)]
    out = {
        "model": a.model, "data_hash": dh, "n_total": n,
        "label_balance": {"label1_contradiction": int((labels==1).sum()),
                          "label0_consistent": int((labels==0).sum())},
        "joint_2x2": {f"{s}|label{l}": cell[(s,l)] for (s,l) in cell},
        "n_model_error_A": wrong,
        "n_label1_B": int((labels==1).sum()),
        "error_on_contradiction": cell[("wrong",1)],
        "error_on_consistent": cell[("wrong",0)],
        "abstain_total": cell[("abstain",0)]+cell[("abstain",1)],
        "mapping_sanitycheck_samples": samples,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k:out[k] for k in ("model","n_total","label_balance","joint_2x2","n_model_error_A","abstain_total")}, indent=2))
    return 0
if __name__ == "__main__": sys.exit(main())

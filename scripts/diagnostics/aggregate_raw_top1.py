#!/usr/bin/env python3
"""Aggregate per-model raw_top1 diagnostic JSON sidecars into the §5.1
cross-model W_u top-1 token table.

Reads {model_short}_top1_summary.json from --in-dir and emits:
  - Markdown table (stdout) ready to drop into paper §5.1 / wiki
  - per-model condensed records as JSON

Columns in the table:
  - model
  - V (vocab)
  - top-1 σ
  - σ-gap (σ_1 / σ_2 — bigger ⇒ more singular-value isolation)
  - gen_step=1 modal token (and frac of samples)
  - V_raw[0] character (pos / neg side semantic class — manual annotation)
  - YES/NO/Answer projection magnitudes (positive vs negative)
  - ctrl_mean ± std on V_raw[0] (signed)
  - contr_mean ± std on V_raw[0] (signed)
  - Δ (contr − ctrl) and direction (rupture-magnitude axis ↑ or content axis ⇄)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


MODELS = [
    "Llama-3.2-3B-Instruct-4bit",
    "Mistral-7B-Instruct-v0.3-4bit",
    "Qwen2.5-7B-Instruct-4bit",
    "Qwen3-8B-4bit",
    "Phi-3.5-mini-instruct-4bit",
    "gemma-3-4b-it-4bit",
]

EMOJI = {
    "Llama-3.2-3B-Instruct-4bit": "🦙",
    "Mistral-7B-Instruct-v0.3-4bit": "🌀",
    "Qwen2.5-7B-Instruct-4bit": "🐉",
    "Qwen3-8B-4bit": "🐲",
    "Phi-3.5-mini-instruct-4bit": "🪼",
    "gemma-3-4b-it-4bit": "🌸",
}


def load_summary(in_dir: Path, short: str) -> dict:
    p = in_dir / f"{short}_top1_summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fmt_token_list(records: List[dict], k: int = 5) -> str:
    if not records:
        return "—"
    parts = []
    for r in records[:k]:
        d = r.get("decoded", "?")
        d = d.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        parts.append(f"`{d}` ({r['proj']:+.2f})")
    return ", ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-md", default=None,
                    help="Optional: write Markdown table to this file")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)

    summaries: Dict[str, dict] = {}
    for short in MODELS:
        s = load_summary(in_dir, short)
        if s:
            summaries[short] = s

    if not summaries:
        print(f"No summaries found in {in_dir}", file=sys.stderr)
        return 1

    lines: List[str] = []
    lines.append("## Cross-model W_u top-1 right-singular-vector character\n")
    lines.append(
        "Per-model: σ-spectrum top-1, σ-gap to top-2, top-5 positive and top-5 "
        "negative tokens projecting onto V_raw[0], targeted projections of "
        "answer-relevant tokens, modal gen_step=1 commit token, and per-sample "
        "signed Δh_jn · V_raw[0] for ctrl vs contr (N=100 stratified, "
        "seed=20260423).\n"
    )

    # Spectrum + commit token table
    lines.append("### Spectrum + commit token\n")
    lines.append("| Model | V | σ_1 | σ_1/σ_2 | modal gen_step=1 | frac |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for short in MODELS:
        s = summaries.get(short)
        if not s:
            lines.append(f"| {EMOJI.get(short,'')} {short} | _PENDING_ |  |  |  |  |")
            continue
        em = EMOJI.get(short, "")
        sigma1 = s["top1_sigma"]
        sigma2 = s["top8_sigma"][1] if len(s["top8_sigma"]) > 1 else 1.0
        gap = sigma1 / max(sigma2, 1e-9)
        ntc = s.get("next_token_counts", {})
        if ntc:
            modal_tok, modal_n = max(ntc.items(), key=lambda kv: kv[1])
            total = sum(ntc.values())
            frac = modal_n / total
            modal_disp = modal_tok.replace("\n", "\\n").replace("\r", "\\r")
        else:
            modal_disp = "?"
            frac = 0.0
        lines.append(
            f"| {em} {short} | {s['vocab_size']:,} | {sigma1:.2f} | "
            f"{gap:.2f}× | `{modal_disp}` | {frac:.0%} |"
        )

    # Top tokens
    lines.append("\n### V_raw[0] top-5 tokens (positive / negative)\n")
    lines.append("| Model | top-5 positive | top-5 negative |")
    lines.append("|---|---|---|")
    for short in MODELS:
        s = summaries.get(short)
        if not s:
            lines.append(f"| {EMOJI.get(short,'')} {short} | _PENDING_ | _PENDING_ |")
            continue
        em = EMOJI.get(short, "")
        pos_str = fmt_token_list(s.get("top_pos_tokens", []), 5)
        neg_str = fmt_token_list(s.get("top_neg_tokens", []), 5)
        lines.append(f"| {em} {short} | {pos_str} | {neg_str} |")

    # Targeted projections
    lines.append("\n### Targeted projections onto V_raw[0]\n")
    lines.append(
        "Compares signed projections of `' YES'` / `' NO'` / `' Answer'` / "
        "`'\\n'` (newline). If `' YES'` and `' NO'` project with *opposite* "
        "signs and *similar magnitudes*, V_raw[0] is a content (YES/NO) axis. "
        "If both project with the *same* sign, V_raw[0] is something else "
        "(rupture-magnitude axis or unrelated).\n"
    )
    lines.append("| Model | ` YES` | ` NO` | ` Answer` | `\\n` | axis interpretation |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for short in MODELS:
        s = summaries.get(short)
        if not s:
            lines.append(f"| {EMOJI.get(short,'')} {short} | _PENDING_ |  |  |  |  |")
            continue
        em = EMOJI.get(short, "")
        targeted = s.get("targeted_tokens", {})
        def first_match(tag: str) -> str:
            for k, v in targeted.items():
                if k.startswith(tag):
                    return f"{v['top1']:+.3f}"
            return "—"
        yes = first_match("' YES'")
        no = first_match("' NO'")
        ans = first_match("' Answer'")
        nl = first_match("'\\n'")
        # Heuristic axis interpretation
        try:
            yes_v = float(yes) if yes != "—" else None
            no_v = float(no) if no != "—" else None
            if yes_v is not None and no_v is not None:
                if yes_v * no_v < 0 and abs(abs(yes_v) - abs(no_v)) < 0.5 * max(abs(yes_v), abs(no_v)):
                    axis = "content (YES/NO bipolar)"
                elif yes_v * no_v > 0:
                    axis = "non-content (same-sign)"
                else:
                    axis = "ambiguous"
            else:
                axis = "?"
        except Exception:
            axis = "?"
        lines.append(f"| {em} {short} | {yes} | {no} | {ans} | {nl} | {axis} |")

    # Per-sample Δh_jn · V_raw[0]
    lines.append("\n### Per-sample signed Δh_jn · V_raw[0] (N=100, ctrl vs contr)\n")
    lines.append("| Model | ctrl mean ± std | contr mean ± std | Δ (contr − ctrl) | ctrl frac>0 | contr frac>0 | axis usage |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|---|")
    for short in MODELS:
        s = summaries.get(short)
        if not s:
            lines.append(f"| {EMOJI.get(short,'')} {short} | _PENDING_ |  |  |  |  |  |")
            continue
        em = EMOJI.get(short, "")
        d = s.get("distributions", {}).get("signed_proj_raw_top1", {})
        if not d:
            lines.append(f"| {em} {short} | _N/A_ |  |  |  |  |  |")
            continue
        cf = d["ctrl_pos_frac"]
        kf = d["contr_pos_frac"]
        # Axis usage interpretation
        if cf > 0.95 and kf > 0.95:
            usage = "monotone same-sign (rupture-magnitude axis)"
        elif cf < 0.05 and kf < 0.05:
            usage = "monotone same-sign (rupture-magnitude axis, native −)"
        elif abs(cf - 0.5) < 0.2 and abs(kf - 0.5) < 0.2:
            usage = "bipolar (content axis)"
        else:
            usage = "mixed"
        lines.append(
            f"| {em} {short} | {d['ctrl_mean']:+.3f} ± {d['ctrl_std']:.3f} | "
            f"{d['contr_mean']:+.3f} ± {d['contr_std']:.3f} | "
            f"{d['delta_mean']:+.3f} | {cf:.0%} | {kf:.0%} | {usage} |"
        )

    out_text = "\n".join(lines) + "\n"
    print(out_text)
    if args.out_md:
        Path(args.out_md).write_text(out_text)
        print(f"\n→ wrote {args.out_md}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

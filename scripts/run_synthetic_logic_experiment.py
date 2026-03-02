#!/usr/bin/env python3
"""
Run synthetic contradiction experiment with event-aligned token traces.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gc
import gzip
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import attention_contribution  # noqa: E402
import config  # noqa: E402
import hidden_state_collector  # noqa: E402
import model_adapters  # noqa: E402
import synthetic_logic_loader  # noqa: E402
import synthetic_trace  # noqa: E402


SUMMARY_SIGNALS = tuple(synthetic_trace.PRIMARY_SIGNALS) + (
    "surprise",
    "delta_h",
    "pc1_ratio",
    "effective_rank",
    "spectral_entropy",
)


def _model_entry(name: str) -> Dict[str, Any]:
    lookup = {m["name"]: m for m in config.MODEL_CONFIGS}
    if name not in lookup:
        raise ValueError(f"Unknown model name: {name}")
    return lookup[name]


def _select_preflight_samples(
    samples: List[Dict[str, Any]],
    per_cell: int,
) -> List[Dict[str, Any]]:
    by_cell: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_cell[str(sample["cell"])].append(sample)
    out: List[Dict[str, Any]] = []
    for cell in synthetic_logic_loader.ALL_CELLS:
        out.extend(by_cell[cell][: int(per_cell)])
    return out


def _init_diag_store(signal_keys: Iterable[str]) -> Dict[str, Any]:
    return {
        "values": {k: [] for k in signal_keys},
        "errors": 0,
    }


def _update_diag(diag: Dict[str, Any], points: List[Dict[str, Any]], signal_keys: Iterable[str]) -> None:
    for point in points:
        for key in signal_keys:
            value = float(point.get(key, 0.0))
            if np.isfinite(value):
                diag["values"][key].append(value)
            else:
                diag["values"][key].append(float("nan"))


def _finalize_diag(diag: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"signals": {}, "errors": int(diag.get("errors", 0))}
    for key, values in diag["values"].items():
        arr = np.asarray(values, dtype=np.float64) if values else np.asarray([], dtype=np.float64)
        finite_mask = np.isfinite(arr)
        finite = arr[finite_mask]
        out["signals"][key] = {
            "n_values": int(arr.size),
            "n_nonfinite": int(arr.size - finite.size),
            "min": float(np.min(finite)) if finite.size else 0.0,
            "max": float(np.max(finite)) if finite.size else 0.0,
            "mean": float(np.mean(finite)) if finite.size else 0.0,
            "std": float(np.std(finite)) if finite.size else 0.0,
        }
    return out


def parse_yes_no_answer(text: str | None) -> str:
    """
    Parse the first alphabetic token and map to YES/NO/UNPARSEABLE.
    """
    if text is None:
        return "UNPARSEABLE"
    s = str(text).strip()
    # Remove common chat/control special tokens such as <|eot_id|>, <|assistant|>, etc.
    s = re.sub(r"<\|[^|>]+?\|>", " ", s)
    # Some models emit role headers without angle-bracket wrappers; skip these if they appear first.
    role_header_tokens = {"ASSISTANT", "USER", "SYSTEM"}
    for match in re.finditer(r"[A-Za-z]+", s):
        tok = match.group(0).upper()
        if tok in role_header_tokens:
            continue
        if tok.startswith("YES"):
            return "YES"
        if tok.startswith("NO"):
            return "NO"
        # First meaningful alphabetic token is not YES/NO -> unparseable by design.
        return "UNPARSEABLE"
    return "UNPARSEABLE"


def parse_property_word_answer(
    text: str | None,
    expected_property: str | None = None,
) -> str:
    """
    Best-effort parser for single-word property answers.

    If the expected property appears anywhere in the decoded text, return it.
    Otherwise, return the first non-trivial content word after removing special tokens.
    """
    if text is None:
        return "UNPARSEABLE"
    s = str(text).strip()
    s = re.sub(r"<\|[^|>]+?\|>", " ", s)
    tokens = [m.group(0).lower() for m in re.finditer(r"[A-Za-z]+", s)]
    if not tokens:
        return "UNPARSEABLE"

    if expected_property:
        target = str(expected_property).lower()
        if target in tokens:
            return target

    stopwords = {
        "the", "a", "an", "answer", "is", "it", "this", "that", "because", "based",
        "on", "premises", "premise", "since", "we", "can", "conclude", "therefore",
        "so", "given", "from", "all", "are", "and", "then", "thus", "be", "to",
        "of", "in", "for", "assistant", "user", "system", "now", "solve", "question",
    }
    for tok in tokens:
        if tok in stopwords:
            continue
        return tok
    return "UNPARSEABLE"


def answer_mode_for_sample(sample: Dict[str, Any]) -> str:
    meta = sample.get("metadata") if isinstance(sample, dict) else None
    if isinstance(meta, dict):
        mode = meta.get("answer_mode")
        if isinstance(mode, str) and mode:
            return mode
    return "yes_no"


def expected_answer_for_sample(sample: Dict[str, Any]) -> str:
    mode = answer_mode_for_sample(sample)
    if mode == "property_word":
        return str(sample.get("target_property", "")).lower()
    return "NO" if bool(sample.get("has_contradiction", False)) else "YES"


def parse_answer_for_sample(sample: Dict[str, Any], text: str | None) -> str:
    mode = answer_mode_for_sample(sample)
    if mode == "property_word":
        return parse_property_word_answer(
            text,
            expected_property=str(sample.get("target_property", "")) or None,
        )
    return parse_yes_no_answer(text)


def compute_behavior_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _acc(rows: List[Dict[str, Any]]) -> float:
        valid = [
            int(bool(r.get("answer_correct")))
            for r in rows
            if bool(r.get("answer_parse_valid", False))
        ]
        if not valid:
            return 0.0
        return float(sum(valid) / len(valid))

    def _parse_rate(rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return float(sum(1 for r in rows if bool(r.get("answer_parse_valid", False))) / len(rows))

    def _answer_distribution(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        c = Counter(str(r.get("parsed_answer", "UNPARSEABLE")) for r in rows)
        # Preserve full distribution for property-word mode while keeping common
        # keys explicit for YES/NO experiments.
        out = {str(k): int(v) for k, v in c.items()}
        out.setdefault("YES", 0)
        out.setdefault("NO", 0)
        out.setdefault("UNPARSEABLE", 0)
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    by_cell: Dict[str, Dict[str, Any]] = {}
    for cell in synthetic_logic_loader.ALL_CELLS:
        cell_rows = [r for r in records if str(r.get("cell")) == cell]
        by_cell[cell] = {
            "n": int(len(cell_rows)),
            "parse_rate": _parse_rate(cell_rows),
            "accuracy": _acc(cell_rows),
            "answer_distribution": _answer_distribution(cell_rows),
        }

    control_rows = [r for r in records if not bool(r.get("has_contradiction", False))]
    contradiction_rows = [r for r in records if bool(r.get("has_contradiction", False))]

    overall = {
        "n": int(len(records)),
        "parse_rate": _parse_rate(records),
        "accuracy": _acc(records),
        "answer_distribution": _answer_distribution(records),
    }
    control = {
        "n": int(len(control_rows)),
        "parse_rate": _parse_rate(control_rows),
        "accuracy": _acc(control_rows),
        "answer_distribution": _answer_distribution(control_rows),
    }
    contradiction = {
        "n": int(len(contradiction_rows)),
        "parse_rate": _parse_rate(contradiction_rows),
        "accuracy": _acc(contradiction_rows),
        "answer_distribution": _answer_distribution(contradiction_rows),
    }
    return {
        "overall": overall,
        "control": control,
        "contradiction": contradiction,
        "by_cell": by_cell,
        "control_accuracy": float(control["accuracy"]),
    }


def _run_samples(
    adapter: Any,
    tokenizer: Any,
    samples: List[Dict[str, Any]],
    max_gen_tokens: int,
    temperature: float,
    window_tight: int,
    window_wide: int,
    uncertainty_cfg: config.UncertaintyConfig | None,
    raw_output_path: Path | None = None,
    collect_diagnostics: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    compact_records: List[Dict[str, Any]] = []
    diag = _init_diag_store(SUMMARY_SIGNALS) if collect_diagnostics else {}

    writer = None
    if raw_output_path is not None:
        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = gzip.open(raw_output_path, "wt", encoding="utf-8")

    try:
        for idx, sample in enumerate(tqdm(samples, desc="Synthetic run")):
            sample_id = str(sample.get("sample_id", f"sample_{idx:04d}"))
            raw_record: Dict[str, Any] = {
                "sample_id": sample_id,
                "cell": sample.get("cell"),
                "chain_steps": int(sample.get("chain_steps", 0)),
                "has_contradiction": bool(sample.get("has_contradiction", False)),
                "anchor_char_index": int(sample.get("anchor_char_index", 0)),
                "prompt": sample.get("prompt", ""),
                "answer_mode": answer_mode_for_sample(sample),
                "expected_answer": expected_answer_for_sample(sample),
            }
            compact: Dict[str, Any] = {
                "sample_id": sample_id,
                "cell": sample.get("cell"),
                "chain_steps": int(sample.get("chain_steps", 0)),
                "has_contradiction": bool(sample.get("has_contradiction", False)),
                "window_summaries": {},
                "truncation_flag": False,
                "halt_reason": "",
                "answer_mode": answer_mode_for_sample(sample),
                "expected_answer": expected_answer_for_sample(sample),
            }
            try:
                prompt = str(sample["prompt"])
                anchor_token_idx = synthetic_trace.resolve_anchor_token_index(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    anchor_char_index=int(sample["anchor_char_index"]),
                )
                prefix_trace = synthetic_trace.collect_prefix_trace(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    anchor_token_index=anchor_token_idx,
                    window_wide=window_wide,
                    cfg_obj=uncertainty_cfg,
                )
                generation = synthetic_trace.collect_generation_trace(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=max_gen_tokens,
                    temperature=temperature,
                    cfg_obj=uncertainty_cfg,
                )

                window_summaries = synthetic_trace.compute_window_summaries(
                    prefix_trace=prefix_trace,
                    windows=(window_tight, window_wide),
                    signal_keys=SUMMARY_SIGNALS,
                )
                truncation_flag = generation.get("halt_reason") == "max_tokens reached"
                parsed_answer = parse_answer_for_sample(sample, generation.get("text"))
                expected_answer = expected_answer_for_sample(sample)
                mode = answer_mode_for_sample(sample)
                if mode == "property_word":
                    answer_parse_valid = parsed_answer != "UNPARSEABLE"
                    answer_correct = bool(answer_parse_valid and parsed_answer.lower() == expected_answer.lower())
                else:
                    answer_parse_valid = parsed_answer in {"YES", "NO"}
                    answer_correct = bool(answer_parse_valid and parsed_answer == expected_answer)

                raw_record.update(
                    {
                        "anchor_token_index": int(anchor_token_idx),
                        "prefix_trace": prefix_trace,
                        "window_summaries": window_summaries,
                        "generation_trace": generation,
                        "truncation_flag": bool(truncation_flag),
                        "max_gen_tokens_used": int(max_gen_tokens),
                        "parsed_answer": parsed_answer,
                        "answer_parse_valid": bool(answer_parse_valid),
                        "answer_correct": bool(answer_correct),
                    }
                )
                compact.update(
                    {
                        "window_summaries": window_summaries,
                        "truncation_flag": bool(truncation_flag),
                        "halt_reason": str(generation.get("halt_reason", "")),
                        "n_generation_steps": int(len(generation.get("trajectory", []))),
                        "parsed_answer": parsed_answer,
                        "answer_parse_valid": bool(answer_parse_valid),
                        "answer_correct": bool(answer_correct),
                    }
                )
                if collect_diagnostics:
                    _update_diag(diag, prefix_trace, SUMMARY_SIGNALS)
                    _update_diag(diag, generation.get("trajectory", []), SUMMARY_SIGNALS)
            except Exception as exc:
                raw_record["error"] = str(exc)
                compact["error"] = str(exc)
                raw_record["parsed_answer"] = "UNPARSEABLE"
                raw_record["answer_parse_valid"] = False
                raw_record["answer_correct"] = False
                compact["parsed_answer"] = "UNPARSEABLE"
                compact["answer_parse_valid"] = False
                compact["answer_correct"] = False
                if collect_diagnostics:
                    diag["errors"] += 1

            compact_records.append(compact)
            if writer is not None:
                writer.write(json.dumps(raw_record) + "\n")
            if (idx + 1) % 25 == 0:
                gc.collect()
    finally:
        if writer is not None:
            writer.close()

    return compact_records, (_finalize_diag(diag) if collect_diagnostics else {})


def _truncation_rate(records: List[Dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return float(sum(1 for r in records if r.get("truncation_flag")) / len(records))


def decide_generation_budget(
    preflight_truncation_rate: float,
    threshold: float,
    base_max_tokens: int,
    escalated_max_tokens: int,
) -> Tuple[int, bool]:
    if float(preflight_truncation_rate) > float(threshold):
        return int(escalated_max_tokens), True
    return int(base_max_tokens), False


def _dataset_counts(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {cell: 0 for cell in synthetic_logic_loader.ALL_CELLS}
    for sample in samples:
        cell = str(sample.get("cell"))
        if cell in out:
            out[cell] += 1
    return out


def _preflight_status(diag: Dict[str, Any]) -> str:
    if not diag:
        return "not_run"
    if int(diag.get("errors", 0)) > 0:
        return "warn"
    for stats in diag.get("signals", {}).values():
        if int(stats.get("n_nonfinite", 0)) > 0:
            return "warn"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic contradiction experiment runner")
    parser.add_argument("--model-name", default="llama_3.2_3b", help="Model key in config.MODEL_CONFIGS")
    parser.add_argument("--n-per-cell", type=int, default=200, help="Samples per 2x2 cell")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--short-chain-steps", type=int, default=2, help="Short-chain entailment depth")
    parser.add_argument("--long-chain-steps", type=int, default=5, help="Long-chain entailment depth")
    parser.add_argument(
        "--prompt-template-variant",
        default=synthetic_logic_loader.PROMPT_TEMPLATE_BASELINE,
        choices=sorted(synthetic_logic_loader.PROMPT_TEMPLATE_VARIANTS),
        help="Prompt template variant",
    )
    parser.add_argument("--preflight-per-cell", type=int, default=5, help="Preflight samples per cell")
    parser.add_argument("--window-tight", type=int, default=5, help="Primary event window")
    parser.add_argument("--window-wide", type=int, default=12, help="Robustness event window")
    parser.add_argument("--max-gen-tokens", type=int, default=12, help="Generation budget")
    parser.add_argument("--max-gen-tokens-escalated", type=int, default=24, help="Escalated generation budget")
    parser.add_argument("--truncation-threshold", type=float, default=0.20, help="Escalate when truncation exceeds this")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--n-permutations", type=int, default=10000, help="Permutation count")
    parser.add_argument("--results-dir", default="./results", help="Results directory")
    parser.add_argument("--save-dataset", default=None, help="Optional dataset JSON path")
    parser.add_argument("--monitor-config", default=None, help="Optional monitoring config JSON")
    parser.add_argument("--output-prefix", default=None, help="Optional output prefix")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional debug cap on total samples")
    parser.add_argument("--control-accuracy-gate", type=float, default=0.80, help="Control accuracy threshold for scaling")
    parser.add_argument("--enforce-control-accuracy-gate", action="store_true", help="Skip full run when pilot control accuracy is below threshold")
    parser.add_argument("--pilot-only", action="store_true", help="Run pilot/preflight subset only and skip full run")
    args = parser.parse_args()

    entry = _model_entry(args.model_name)
    run_id = args.output_prefix or f"synthetic_logic_{entry['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    uncertainty_cfg = None
    if args.monitor_config:
        uncertainty_cfg = config.load_monitoring_config(args.monitor_config).uncertainty

    dataset_cfg = synthetic_logic_loader.SyntheticLogicConfig(
        n_per_cell=int(args.n_per_cell),
        seed=int(args.seed),
        short_chain_steps=int(args.short_chain_steps),
        long_chain_steps=int(args.long_chain_steps),
        prompt_template_variant=str(args.prompt_template_variant),
    )
    samples = synthetic_logic_loader.generate_synthetic_logic_dataset(dataset_cfg)
    if args.limit_samples is not None and int(args.limit_samples) < len(samples):
        samples = samples[: int(args.limit_samples)]

    if args.save_dataset:
        synthetic_logic_loader.save_split(samples, args.save_dataset)
        print(f"Saved dataset to {args.save_dataset}")

    print(f"Loading model: {entry['path']}")
    from mlx_lm import load

    model, tokenizer = load(entry["path"])
    collector = hidden_state_collector.HiddenStateCollector()
    acr_collector = attention_contribution.ACRCollector()
    adapter = model_adapters.create_adapter(
        model,
        collector,
        model_type=entry["model_type"],
        acr_collector=acr_collector,
    )

    preflight_samples = _select_preflight_samples(samples, int(args.preflight_per_cell))
    preflight_raw = results_dir / f"{run_id}_preflight_samples.jsonl.gz"
    preflight_records, preflight_diag = _run_samples(
        adapter=adapter,
        tokenizer=tokenizer,
        samples=preflight_samples,
        max_gen_tokens=int(args.max_gen_tokens),
        temperature=float(args.temperature),
        window_tight=int(args.window_tight),
        window_wide=int(args.window_wide),
        uncertainty_cfg=uncertainty_cfg,
        raw_output_path=preflight_raw,
        collect_diagnostics=True,
    )
    preflight_trunc = _truncation_rate(preflight_records)
    preflight_behavior = compute_behavior_metrics(preflight_records)
    preflight_window_stats = synthetic_trace.summarize_signal_effects(
        sample_records=preflight_records,
        windows=(int(args.window_tight), int(args.window_wide)),
        signal_keys=synthetic_trace.PRIMARY_SIGNALS,
        n_permutations=int(args.n_permutations),
        seed=int(args.seed),
    )
    use_max_gen_tokens, escalated = decide_generation_budget(
        preflight_truncation_rate=preflight_trunc,
        threshold=float(args.truncation_threshold),
        base_max_tokens=int(args.max_gen_tokens),
        escalated_max_tokens=int(args.max_gen_tokens_escalated),
    )

    gate_threshold = float(args.control_accuracy_gate)
    control_accuracy = float(preflight_behavior.get("control_accuracy", 0.0))
    gate_failed = bool(
        args.enforce_control_accuracy_gate and control_accuracy < gate_threshold
    )
    full_run_executed = not bool(args.pilot_only) and not gate_failed
    full_raw: Path | None = None
    full_records: List[Dict[str, Any]] = []
    window_effects: Dict[str, Any] = {}
    full_behavior: Dict[str, Any] | None = None
    trunc_by_cell: Dict[str, float] = {}
    if full_run_executed:
        full_raw = results_dir / f"{run_id}_samples.jsonl.gz"
        full_records, _ = _run_samples(
            adapter=adapter,
            tokenizer=tokenizer,
            samples=samples,
            max_gen_tokens=use_max_gen_tokens,
            temperature=float(args.temperature),
            window_tight=int(args.window_tight),
            window_wide=int(args.window_wide),
            uncertainty_cfg=uncertainty_cfg,
            raw_output_path=full_raw,
            collect_diagnostics=False,
        )

        window_effects = synthetic_trace.summarize_signal_effects(
            sample_records=full_records,
            windows=(int(args.window_tight), int(args.window_wide)),
            signal_keys=synthetic_trace.PRIMARY_SIGNALS,
            n_permutations=int(args.n_permutations),
            seed=int(args.seed),
        )
        full_behavior = compute_behavior_metrics(full_records)

        for cell in synthetic_logic_loader.ALL_CELLS:
            cell_rows = [r for r in full_records if r.get("cell") == cell]
            trunc_by_cell[cell] = _truncation_rate(cell_rows)
    else:
        for cell in synthetic_logic_loader.ALL_CELLS:
            trunc_by_cell[cell] = 0.0

    summary = {
        "run_config": {
            "run_id": run_id,
            "model_name": entry["name"],
            "model_path": entry["path"],
            "model_type": entry["model_type"],
            "n_per_cell": int(args.n_per_cell),
            "seed": int(args.seed),
            "window_tight": int(args.window_tight),
            "window_wide": int(args.window_wide),
            "max_gen_tokens_requested": int(args.max_gen_tokens),
            "max_gen_tokens_used": int(use_max_gen_tokens),
            "temperature": float(args.temperature),
            "n_permutations": int(args.n_permutations),
            "results_dir": str(results_dir),
            "short_chain_steps": int(args.short_chain_steps),
            "long_chain_steps": int(args.long_chain_steps),
            "prompt_template_variant": str(args.prompt_template_variant),
            "pilot_only": bool(args.pilot_only),
            "enforce_control_accuracy_gate": bool(args.enforce_control_accuracy_gate),
            "control_accuracy_gate": gate_threshold,
        },
        "dataset_summary": {
            "n_samples_total": int(len(samples)),
            "cell_counts": _dataset_counts(samples),
        },
        "preflight_report": {
            "n_samples": int(len(preflight_samples)),
            "raw_path": str(preflight_raw),
            "diag_status": _preflight_status(preflight_diag),
            "diag": preflight_diag,
            "truncation_rate": float(preflight_trunc),
            "truncation_threshold": float(args.truncation_threshold),
            "escalated_generation_budget": bool(escalated),
            "max_gen_tokens_after_preflight": int(use_max_gen_tokens),
            "behavior_metrics": preflight_behavior,
            "window_stats": preflight_window_stats,
        },
        "behavioral_engagement": {
            "control_accuracy": control_accuracy,
            "control_accuracy_gate": gate_threshold,
            "gate_status": (
                "failed" if gate_failed else "passed" if args.enforce_control_accuracy_gate else "not_enforced"
            ),
            "pilot_only": bool(args.pilot_only),
            "full_run_executed": bool(full_run_executed),
        },
        "behavior_metrics": full_behavior if full_behavior is not None else preflight_behavior,
        "truncation_metrics": {
            "overall_rate": float(_truncation_rate(full_records)) if full_run_executed else 0.0,
            "by_cell": trunc_by_cell,
        },
        "mechanistic_results": {
            "executed": bool(full_run_executed),
            "reason_not_executed": (
                "pilot_only"
                if args.pilot_only
                else "control_accuracy_gate_failed"
                if gate_failed
                else None
            ),
        },
        "window_stats": window_effects if full_run_executed else {},
        "raw_trace_path": str(full_raw) if full_raw is not None else None,
        "n_compact_records": int(len(full_records)) if full_run_executed else 0,
    }

    summary_path = results_dir / f"{run_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary: {summary_path}")
    if full_raw is not None:
        print(f"Saved raw traces: {full_raw}")
    else:
        print("Skipped full run; no full raw trace file produced")


if __name__ == "__main__":
    main()

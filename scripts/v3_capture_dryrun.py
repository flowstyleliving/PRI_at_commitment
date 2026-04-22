#!/usr/bin/env python3
"""
Prereq 4 gate — end-to-end dry-run of the v3 capture schedule on the shared
production pipeline (`pri_v2_mlx_pipeline.trace_sample` with `v3_capture=True`).

Sealed spec: v3 plan §Prerequisites.4 (eight assertion bundles).

Scope (intentionally tiny — schema / provenance / tripwire / parquet / audit):
  * One contradiction-cell synthetic-logic puzzle (seed 42).
  * Three models: Llama 3B, Mistral 7B, Qwen 2.5 7B (all 4-bit MLX).
  * Eight assertion bundles per model:
      B1 schema           — per-row dtype/shape/column check against sealed spec.
      B2 schedule         — every-layer at steps 1–12, probe_4 identical at 13+.
      B3 provenance       — step-0 prefix_last / step≥1 gen_prev (closes §7b).
      B4 finite           — np.isfinite(h_t) and np.isfinite(h_prev).
      B5 tripwire_healthy — ||Δh_step0||/||h_t|| < 10 at final layer.
      B6 tripwire_fault   — mutate h_prev_source_log[0]→'gen_prev'; guard fires
                            (closes M1 on a worked example).
      B7 dict_collision   — H4 write-once guard exercises every layer × step
                            without duplicate-key crash.
      B8 consumer_audit   — grep PRIComputer.compute_step vs parquet schema.

Exit 0 = all eight bundles green on all three models; v3 main run may launch.
Exit non-zero = shared pipeline still unfit; do NOT launch the main run.

Artifacts (auto-incremented run-NN per date):
  experiments/v3-capture-dryrun/<YYYY-MM-DD>/run-NN/
    dryrun_capture.parquet   union of per-(step,layer) rows across all 3 models
    dryrun_report.json       per-model bundle results + config + git SHA
    {model}_dryrun.json      per-model assertion log (diagnostic detail)
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from mlx_lm import load as mlx_load

from pri_v2_mlx_pipeline import (
    OutputProjection,
    find_layers,
    get_layer_indices,
    trace_sample,
)
from synthetic_logic_loader import (
    SyntheticLogicConfig,
    generate_synthetic_logic_dataset,
)
from scripts._paths import experiment_run_dir


# ---- Config ----

MODELS = [
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
]
SEED = 42
# Bumped 4 → 14 so step_idx ∈ {12, 13} land in the probe_4 regime, giving the
# "probe_4 identical across steps" schedule assertion two steps to compare.
MAX_NEW_TOKENS = 14
ALL_FOR_FIRST_N_STEPS = 12
H_PREV_SANITY_MAX_RATIO = 10.0
PROBE_LAYERS = ["final", "mid", "quarter"]  # paper-path layers to keep populated
PROBE_FALLBACK = ["final", "three_quarters", "mid", "quarter"]

# capture_schedule_tag values written onto each parquet row. step_idx is 0-indexed
# in code; tag labels use the 1-indexed step numbering from the sealed spec.
TAG_EVERY_LAYER = "every_layer_steps_1_12"
TAG_PROBE_4 = "probe_4_steps_13+"

PARQUET_COLUMNS = [
    "run_id", "git_sha", "model", "sample_id", "condition",
    "step", "layer", "h_t", "h_prev", "h_prev_source", "capture_schedule_tag",
]
PARQUET_SCHEMA = pa.schema([
    ("run_id", pa.string()),
    ("git_sha", pa.string()),
    ("model", pa.string()),
    ("sample_id", pa.string()),
    ("condition", pa.string()),
    ("step", pa.int32()),
    ("layer", pa.string()),
    ("h_t", pa.list_(pa.float32())),
    ("h_prev", pa.list_(pa.float32())),
    ("h_prev_source", pa.string()),
    ("capture_schedule_tag", pa.string()),
])

EXPERIMENT_SLUG = "v3-capture-dryrun"

# Consumer under audit (H2). The grep target is PRIComputer.compute_step, whose
# per-row inputs (h_t, h_prev, p_t, S_t) define the "consumed set" for the audit.
CONSUMER_FILE = _REPO_ROOT / "pri_v2_mlx_pipeline.py"
CONSUMER_SYMBOL = "PRIComputer.compute_step"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "-dirty"
            return sha
    except Exception:
        pass
    return "unknown"


def _assert(cond: bool, msg: str, errors: List[str]) -> None:
    if not cond:
        errors.append(msg)


def model_slug(name: str) -> str:
    return name.split("/")[-1].replace(".", "_").lower()


def schedule_tag(step_idx: int) -> str:
    """Map a 0-indexed gen step to its spec tag.

    step_idx < 12 → every-layer window (spec step 1..12).
    step_idx ≥ 12 → probe_4 window (spec step 13+).
    """
    return TAG_EVERY_LAYER if step_idx < ALL_FOR_FIRST_N_STEPS else TAG_PROBE_4


def provenance_errors(h_prev_source_log: List[str]) -> List[str]:
    """Pure provenance check — used both on the healthy log and the fault-
    injected copy. Step 0 must be 'prefix_last'; all later steps 'gen_prev'.
    """
    errs: List[str] = []
    for si, src in enumerate(h_prev_source_log):
        want = "prefix_last" if si == 0 else "gen_prev"
        if src != want:
            errs.append(
                f"provenance: step {si} h_prev_source={src!r}, expected {want!r}"
            )
    return errs


def fault_injection_probe(h_prev_source_log: List[str]) -> Dict[str, Any]:
    """M1 guard — mutate the healthy log and re-run the provenance check.

    Mutation: h_prev_source_log[0] ← 'gen_prev' (the exact fault §7b exists to
    catch — step-0 inflation from treating the previous *generated* token as
    h_prev instead of the last prefix hidden state).

    The spec explicitly forbids zero-vector injection here: the pipeline raises
    on ratio ≥ 10 at pri_v2_mlx_pipeline.py:819, which would mask the injection
    and make the guard look cosmetic even when it is not.
    """
    mutated = copy.deepcopy(list(h_prev_source_log))
    if not mutated:
        return {
            "mutation": "noop_empty_log",
            "healthy_errors": [],
            "mutated_errors": [],
            "fired": False,
            "passed": False,
            "reason": "empty h_prev_source_log — nothing to mutate",
        }
    before = mutated[0]
    mutated[0] = "gen_prev"
    healthy = provenance_errors(list(h_prev_source_log))
    mutated_errs = provenance_errors(mutated)
    fired = len(mutated_errs) > len(healthy)
    return {
        "mutation": f"h_prev_source_log[0] {before!r} -> 'gen_prev'",
        "healthy_errors": healthy,
        "mutated_errors": mutated_errs,
        "fired": bool(fired),
        # The check passes iff the healthy log is clean AND the mutation fires.
        "passed": bool(len(healthy) == 0 and fired),
    }


def consumer_audit() -> Dict[str, Any]:
    """H2 audit — grep PRIComputer.compute_step for the fields it actually
    consumes, cross-check against the parquet schema.

    Deliberately no-fuzz: we parse the explicit signature + method body, not
    string heuristics, so a rename in the pipeline surfaces as a mismatch.
    """
    src = CONSUMER_FILE.read_text()
    method_match = re.search(
        r"def compute_step\(\s*self,\s*(.*?)\)\s*->\s*Dict\[str,\s*float\]:",
        src, re.DOTALL,
    )
    sig_fields: List[str] = []
    if method_match:
        for raw in method_match.group(1).split(","):
            name = raw.strip().split(":")[0].strip()
            if name and name != "self":
                sig_fields.append(name)

    # Per-row vs per-step split — needed because the parquet schema is per
    # (step, layer), but compute_step also consumes per-step scalars (p_t, S_t).
    per_row_consumed = [f for f in sig_fields if f in {"h_t", "h_prev"}]
    per_step_consumed = [f for f in sig_fields if f in {"p_t", "S_t"}]
    config_consumed = [
        f for f in sig_fields
        if f in {"alpha", "topk_values", "lowrank_values", "v3_rank_values"}
    ]

    # Schema-validated per-row fields (parquet asserts dtype/shape on these).
    schema_validated_per_row = ["h_t", "h_prev"]

    dead_checks = sorted(set(schema_validated_per_row) - set(per_row_consumed))
    missing_checks = sorted(set(per_row_consumed) - set(schema_validated_per_row))

    return {
        "consumer": CONSUMER_SYMBOL,
        "consumer_file": str(CONSUMER_FILE.relative_to(_REPO_ROOT)),
        "signature_fields": sig_fields,
        "consumed_per_row": per_row_consumed,
        "consumed_per_step": per_step_consumed,
        "consumed_config": config_consumed,
        "parquet_schema_columns": PARQUET_COLUMNS,
        "schema_validated_per_row_fields": schema_validated_per_row,
        "dead_schema_checks": dead_checks,
        "missing_schema_checks_for_consumed": missing_checks,
        # Per-step fields (p_t, S_t) are validated at trace level rather than in
        # the per-row parquet schema; not a missing check.
        "per_step_validation_notes": (
            "p_t via trace['gen_probs'][si] finite check; "
            "S_t via trace['gen_surprises'][si] finite check"
        ),
        "passed": not dead_checks and not missing_checks,
    }


def build_parquet_rows(
    *,
    run_id: str,
    git_sha: str,
    model_name: str,
    sample_id: str,
    condition: str,
    trace: Dict[str, Any],
    hidden_dim: int,
    errors: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Flatten the v3 capture trace into parquet rows + assert schema per row.

    Returns (rows, counts_by_tag). Rows conform to PARQUET_SCHEMA; any per-row
    violation is appended to `errors`.
    """
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {TAG_EVERY_LAYER: 0, TAG_PROBE_4: 0}

    captures_by_step = trace["gen_captures_by_step"]
    log = trace["h_prev_source_log"]
    for si, step_captures in enumerate(captures_by_step):
        tag = schedule_tag(si)
        source = log[si] if si < len(log) else "<missing>"
        for lname, payload in step_captures.items():
            h_t = payload["h_t"]
            h_prev = payload["h_prev_causal"]

            # Per-row schema assertions (B1) — the sealed spec says fail loud
            # on missing / wrong-dtype / wrong-shape fields.
            if not isinstance(h_t, np.ndarray) or not isinstance(h_prev, np.ndarray):
                errors.append(
                    f"schema: step {si} layer '{lname}': h_t/h_prev not ndarray"
                )
                continue
            if h_t.shape != (hidden_dim,) or h_prev.shape != (hidden_dim,):
                errors.append(
                    f"schema: step {si} layer '{lname}': "
                    f"h_t.shape={h_t.shape} h_prev.shape={h_prev.shape} "
                    f"expected ({hidden_dim},)"
                )
                continue
            h_t32 = h_t.astype(np.float32, copy=False)
            h_prev32 = h_prev.astype(np.float32, copy=False)
            if h_t32.dtype != np.float32 or h_prev32.dtype != np.float32:
                errors.append(
                    f"schema: step {si} layer '{lname}': dtype "
                    f"h_t={h_t32.dtype} h_prev={h_prev32.dtype} expected float32"
                )
                continue
            if source not in ("prefix_last", "gen_prev"):
                errors.append(
                    f"schema: step {si} layer '{lname}': h_prev_source={source!r} "
                    f"not in {{'prefix_last','gen_prev'}}"
                )
                continue
            if tag not in (TAG_EVERY_LAYER, TAG_PROBE_4):
                errors.append(
                    f"schema: step {si} layer '{lname}': unknown tag {tag!r}"
                )
                continue

            rows.append({
                "run_id": run_id,
                "git_sha": git_sha,
                "model": model_name,
                "sample_id": sample_id,
                "condition": condition,
                "step": int(si),
                "layer": str(lname),
                "h_t": h_t32.tolist(),
                "h_prev": h_prev32.tolist(),
                "h_prev_source": source,
                "capture_schedule_tag": tag,
            })
            counts[tag] = counts.get(tag, 0) + 1
    return rows, counts


def check_model(
    model_name: str,
    prompt: str,
    sample_id: str,
    condition: str,
    run_id: str,
    git_sha: str,
) -> Dict[str, Any]:
    print(f"\n[{model_name}]")
    t0 = time.time()
    model, tokenizer = mlx_load(model_name)
    projection = OutputProjection(model)
    n_layers = len(find_layers(model))
    hidden_dim = projection.hidden_size
    print(
        f"  loaded: layers={n_layers} hidden={hidden_dim} "
        f"vocab={projection.vocab_size}  ({time.time()-t0:.1f}s)"
    )

    layer_indices = get_layer_indices(n_layers, PROBE_LAYERS)
    t1 = time.time()
    trace = trace_sample(
        model, tokenizer, prompt,
        layer_indices, projection,
        max_new_tokens=MAX_NEW_TOKENS,
        v3_capture=True,
        v3_all_for_first_n_steps=ALL_FOR_FIRST_N_STEPS,
        v3_probe_fallback=PROBE_FALLBACK,
        h_prev_sanity_max_ratio=H_PREV_SANITY_MAX_RATIO,
    )
    dt = time.time() - t1
    print(f"  traced: {len(trace['gen_token_ids'])} gen tokens in {dt:.1f}s")

    # Bundle error buckets. A per-bundle list means we can report fine-grained
    # pass/fail instead of a single flat errors blob.
    b_schema: List[str] = []
    b_schedule: List[str] = []
    b_provenance: List[str] = []
    b_finite: List[str] = []
    b_tripwire_healthy: List[str] = []
    b_tripwire_fault: List[str] = []
    b_dict_collision: List[str] = []  # Populated only if the pipeline raised.

    n_steps = len(trace["gen_captures_by_step"])

    # Structural sanity (any of these failing means the trace is malformed; they
    # roll up under the schema bundle because the rest of the bundles depend on
    # a well-formed trace).
    _assert(trace["v3_capture"] is True,
            "v3_capture flag missing on trace", b_schema)
    _assert(trace["n_layers"] == n_layers,
            f"trace.n_layers={trace['n_layers']} != actual {n_layers}", b_schema)
    _assert(n_steps >= 1, "no generation steps captured (EOS immediately?)", b_schema)
    _assert(len(trace["gen_layer_indices_by_step"]) == n_steps,
            "gen_layer_indices_by_step length != gen_captures_by_step length",
            b_schema)
    _assert(len(trace["h_prev_source_log"]) == n_steps,
            "h_prev_source_log length != n_steps", b_schema)

    # B2 schedule — every-layer window + probe_4 identical-across-steps.
    probe_key_sets: List[frozenset] = []
    for si in range(n_steps):
        names = trace["gen_layer_indices_by_step"][si]
        captures = trace["gen_captures_by_step"][si]
        _assert(set(captures.keys()) == set(names.keys()),
                f"step {si}: captures keys mismatch schedule keys", b_schedule)

        if si < ALL_FOR_FIRST_N_STEPS:
            covered = {idx for idx in names.values()}
            missing = set(range(n_layers)) - covered
            if missing:
                b_schedule.append(
                    f"step {si} (tag {TAG_EVERY_LAYER}): missing layer indices "
                    f"{sorted(missing)[:8]}{'…' if len(missing) > 8 else ''}"
                )
        else:
            probe_names = frozenset(names.keys())
            probe_key_sets.append(probe_names)
            missing = set(PROBE_FALLBACK) - probe_names
            if missing:
                b_schedule.append(
                    f"step {si} (tag {TAG_PROBE_4}): probe_fallback names "
                    f"missing {sorted(missing)}"
                )
    # Every probe_4 step must carry the same layer-name set.
    if len(set(probe_key_sets)) > 1:
        b_schedule.append(
            f"{TAG_PROBE_4}: layer sets differ across steps "
            f"({[sorted(s) for s in set(probe_key_sets)][:2]}...)"
        )

    # B4 finite-checks on captured vectors (independent of parquet row build).
    for si in range(n_steps):
        for lname, payload in trace["gen_captures_by_step"][si].items():
            for key in ("h_t", "h_prev_causal"):
                if key not in payload:
                    b_schema.append(
                        f"step {si} layer '{lname}': missing {key}"
                    )
                    continue
                vec = payload[key]
                if not isinstance(vec, np.ndarray) or vec.shape != (hidden_dim,):
                    # Schema-level — caught again in build_parquet_rows.
                    continue
                if not np.all(np.isfinite(vec)):
                    b_finite.append(
                        f"step {si} layer '{lname}' {key}: non-finite values"
                    )

    # B3 provenance (healthy log).
    b_provenance.extend(provenance_errors(list(trace["h_prev_source_log"])))

    # B5 tripwire_healthy — step-0 ||Δh||/||h_t|| must be finite and < max.
    sanity = trace.get("step0_sanity") or {}
    dh_over_ht = sanity.get("dh_over_ht")
    if dh_over_ht is None:
        b_tripwire_healthy.append("step0_sanity missing dh_over_ht")
    else:
        if not np.isfinite(dh_over_ht):
            b_tripwire_healthy.append("step0_sanity.dh_over_ht not finite")
        if dh_over_ht >= H_PREV_SANITY_MAX_RATIO:
            b_tripwire_healthy.append(
                f"step0_sanity.dh_over_ht={dh_over_ht:.4f} ≥ max "
                f"{H_PREV_SANITY_MAX_RATIO} "
                f"(placeholder bound; to be re-set from measured percentile)"
            )
    if "causal_matches_prefix_last" in sanity and (
        sanity["causal_matches_prefix_last"] != 1.0
    ):
        b_tripwire_healthy.append(
            f"step0 causal T-1 != prefix_last "
            f"(fingerprint {sanity['causal_matches_prefix_last']})"
        )

    # B6 tripwire_fault — mutate the log and re-run the provenance check.
    fault = fault_injection_probe(list(trace["h_prev_source_log"]))
    if not fault["passed"]:
        b_tripwire_fault.append(
            "provenance guard failed to fire on h_prev_source_log[0]='gen_prev'"
        )

    # Backward-compat: paper-path gen_hidden populated for every probe name
    # (rolls up under schema because the main run's paper-path columns depend
    # on these).
    for lname in PROBE_LAYERS:
        _assert(lname in trace["gen_hidden"],
                f"paper-path gen_hidden['{lname}'] missing", b_schema)
        _assert(len(trace["gen_hidden"].get(lname, [])) == n_steps,
                f"paper-path gen_hidden['{lname}'] length != n_steps", b_schema)

    # Build parquet rows (also validates B1 schema per row).
    rows, counts_by_tag = build_parquet_rows(
        run_id=run_id,
        git_sha=git_sha,
        model_name=model_name,
        sample_id=sample_id,
        condition=condition,
        trace=trace,
        hidden_dim=hidden_dim,
        errors=b_schema,
    )

    # B7 dict_collision — the H4 guard lives in pri_v2_mlx_pipeline.py
    # (trace_sample, step_captures write-once check) and raises RuntimeError
    # on duplicate-key write. If we reached here, it did not fire across
    # every (step × layer) write of this run. Pass.
    # (An H4 violation would have surfaced as an exception caught upstream.)

    bundle_results = {
        "schema": {"passed": not b_schema, "errors": b_schema},
        "schedule": {"passed": not b_schedule, "errors": b_schedule},
        "provenance": {"passed": not b_provenance, "errors": b_provenance},
        "finite": {"passed": not b_finite, "errors": b_finite},
        "tripwire_healthy": {
            "passed": not b_tripwire_healthy, "errors": b_tripwire_healthy,
        },
        "tripwire_fault_injection": {
            "passed": not b_tripwire_fault,
            "errors": b_tripwire_fault,
            "detail": fault,
        },
        "dict_collision": {"passed": not b_dict_collision, "errors": b_dict_collision},
    }
    all_errors = (
        b_schema + b_schedule + b_provenance + b_finite
        + b_tripwire_healthy + b_tripwire_fault + b_dict_collision
    )
    passed = len(all_errors) == 0

    # Per-row tripwire distribution: reconstruct ||Δh||/||h_t|| for every
    # (step 0, layer) row and log the percentiles; fail if any row ≥ max.
    r_samples: List[float] = []
    if n_steps >= 1:
        for lname, payload in trace["gen_captures_by_step"][0].items():
            h_t = payload.get("h_t")
            h_p = payload.get("h_prev_causal")
            if not isinstance(h_t, np.ndarray) or not isinstance(h_p, np.ndarray):
                continue
            ht_norm = float(np.linalg.norm(h_t)) + 1e-30
            r = float(np.linalg.norm(h_t - h_p)) / ht_norm
            if not np.isfinite(r) or r >= H_PREV_SANITY_MAX_RATIO:
                b_tripwire_healthy.append(
                    f"per-row tripwire: step 0 layer '{lname}' r={r:.4f} "
                    f"(finite={np.isfinite(r)}) ≥ max {H_PREV_SANITY_MAX_RATIO}"
                )
            r_samples.append(r)
    tripwire_dist = {
        "count": len(r_samples),
        "min": float(np.min(r_samples)) if r_samples else None,
        "p50": float(np.percentile(r_samples, 50)) if r_samples else None,
        "p99": float(np.percentile(r_samples, 99)) if r_samples else None,
        "max": float(np.max(r_samples)) if r_samples else None,
        "step0_final_layer_r": (
            float(sanity["dh_over_ht"])
            if sanity.get("dh_over_ht") is not None else None
        ),
    }
    # Re-sync the bucket after the per-row pass in case it appended.
    bundle_results["tripwire_healthy"] = {
        "passed": not b_tripwire_healthy, "errors": b_tripwire_healthy,
    }
    all_errors = (
        b_schema + b_schedule + b_provenance + b_finite
        + b_tripwire_healthy + b_tripwire_fault + b_dict_collision
    )
    passed = len(all_errors) == 0

    record = {
        "model": model_name,
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "n_gen_steps": int(n_steps),
        "h_prev_source_log": list(trace["h_prev_source_log"]),
        "step0_sanity": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in sanity.items()
        },
        "generated_text": trace["generated_text"][:120],
        "row_counts_by_schedule": counts_by_tag,
        "tripwire_distribution": tripwire_dist,
        "bundle_results": bundle_results,
        "passed": passed,
        "errors": all_errors,
        "rows": rows,
    }
    if not passed:
        print(f"  FAIL ({len(all_errors)} error(s)):")
        for e in all_errors[:20]:
            print(f"    - {e}")
        if len(all_errors) > 20:
            print(f"    ... +{len(all_errors) - 20} more")
    else:
        print(
            f"  OK — step-0 dh/ht={sanity.get('dh_over_ht', float('nan')):.4f} "
            f"(layer={sanity.get('layer_name')}) "
            f"causal==prefix_last={bool(sanity.get('causal_matches_prefix_last', 0.0))} "
            f"rows={sum(counts_by_tag.values())} "
            f"fault_fired={fault['fired']}"
        )
    return record


def main() -> int:
    # Auto-incremented run-NN per date. No --run-id / --force: each invocation
    # gets a fresh run directory; nothing can be overwritten.
    argparse.ArgumentParser(description="Prereq 4 v3-capture dry-run").parse_args()

    out_dir = experiment_run_dir(EXPERIMENT_SLUG)
    run_id = out_dir.name
    git_sha = _git_sha()

    print("=" * 72)
    print("Prereq 4 — v3 capture dry-run (eight assertion bundles)")
    print(f"  models: {len(MODELS)}")
    print(f"  max_new_tokens={MAX_NEW_TOKENS} "
          f"(steps 1–12 every-layer / 13+ probe_4)")
    print(f"  all_for_first_n_steps={ALL_FOR_FIRST_N_STEPS}, fallback={PROBE_FALLBACK}")
    print(f"  h_prev_sanity_max_ratio={H_PREV_SANITY_MAX_RATIO}")
    print(f"  output: {out_dir}")
    print(f"  git_sha: {git_sha}")
    print("=" * 72)

    cfg = SyntheticLogicConfig(n_per_cell=1, seed=SEED)
    samples = generate_synthetic_logic_dataset(cfg)
    # Sealed spec: "one puzzle (contradiction cell)". Pick the first sample with
    # has_contradiction=True rather than relying on shuffle ordering.
    contradiction_samples = [s for s in samples if s.get("has_contradiction")]
    chosen = contradiction_samples[0] if contradiction_samples else samples[0]
    prompt = chosen["prompt"]
    sample_id = chosen["sample_id"]
    condition = chosen["cell"]
    print(
        f"\nUsing contradiction puzzle (sample_id={sample_id}, cell={condition})"
    )

    # Global B8 — consumer audit is run once against the pipeline source. Its
    # verdict is a pipeline-wide fact, not a per-model one.
    audit = consumer_audit()

    records: List[Dict[str, Any]] = []
    for model_name in MODELS:
        try:
            rec = check_model(
                model_name, prompt,
                sample_id=sample_id,
                condition=condition,
                run_id=run_id,
                git_sha=git_sha,
            )
        except RuntimeError as exc:
            # H4 step_captures write-once guard was converted from `assert`
            # to `raise RuntimeError` (pri_v2_mlx_pipeline.py) so the check
            # survives `python -O`. Catch RuntimeError here, inspect the
            # message for the H4 marker, and attribute to B7 dict_collision.
            msg = f"{exc}"
            is_h4 = "H4 dict-collision" in msg
            rec = {
                "model": model_name,
                "passed": False,
                "errors": [f"RuntimeError: {msg}"],
                "bundle_results": {
                    "dict_collision": {
                        "passed": False,
                        "errors": [msg] if is_h4 else [],
                    },
                },
                "traceback": traceback.format_exc(),
                "rows": [],
                "row_counts_by_schedule": {},
                "tripwire_distribution": {},
            }
            print(f"  ASSERTION: {msg}")
        except Exception as exc:
            rec = {
                "model": model_name,
                "passed": False,
                "errors": [f"exception: {exc.__class__.__name__}: {exc}"],
                "traceback": traceback.format_exc(),
                "rows": [],
                "row_counts_by_schedule": {},
                "tripwire_distribution": {},
            }
            print(f"  EXCEPTION: {exc}")
        records.append(rec)
        # Per-model diagnostic dump (minus the bulky rows) — useful for quick
        # post-mortem without having to open the parquet.
        dump = {k: v for k, v in rec.items() if k != "rows"}
        (out_dir / f"{model_slug(model_name)}_dryrun.json").write_text(
            json.dumps(dump, indent=2, default=str) + "\n"
        )

    # Aggregate parquet rows across all models (per sealed spec: one
    # dryrun_capture.parquet under the run dir).
    all_rows: List[Dict[str, Any]] = []
    for r in records:
        all_rows.extend(r.get("rows", []))
    parquet_path = out_dir / "dryrun_capture.parquet"
    if all_rows:
        try:
            table = pa.Table.from_pylist(all_rows, schema=PARQUET_SCHEMA)
            pq.write_table(table, parquet_path)
            print(f"\nparquet: {parquet_path} ({len(all_rows)} rows)")
        except Exception as exc:
            print(f"\nparquet write FAILED: {exc}")
    else:
        print("\nparquet: skipped (no rows captured)")

    all_passed = (
        all(r.get("passed", False) for r in records)
        and audit["passed"]
    )

    report = {
        "run_id": run_id,
        "utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(_REPO_ROOT)),
        "git_sha": git_sha,
        "config": {
            "models": MODELS,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "all_for_first_n_steps": ALL_FOR_FIRST_N_STEPS,
            "probe_fallback": PROBE_FALLBACK,
            "probe_layers_paper_path": PROBE_LAYERS,
            "h_prev_sanity_max_ratio": H_PREV_SANITY_MAX_RATIO,
            "sample_id": sample_id,
            "condition": condition,
        },
        "parquet": {
            "path": str(parquet_path.relative_to(out_dir)),
            "schema": {f.name: str(f.type) for f in PARQUET_SCHEMA},
            "total_rows": len(all_rows),
        },
        "consumer_audit": audit,
        "all_passed": all_passed,
        "per_model": [
            {
                "model": r["model"],
                "passed": r.get("passed", False),
                "n_errors": len(r.get("errors", [])),
                "row_counts_by_schedule": r.get("row_counts_by_schedule", {}),
                "tripwire_distribution": r.get("tripwire_distribution", {}),
                "bundle_results": {
                    name: {
                        "passed": body.get("passed", False),
                        "n_errors": len(body.get("errors", [])),
                    }
                    for name, body in (
                        r.get("bundle_results", {}) or {}
                    ).items()
                },
            }
            for r in records
        ],
    }
    (out_dir / "dryrun_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )

    print("\n" + "=" * 72)
    if all_passed:
        print("VERDICT: DRY-RUN GREEN — Prereq 4 closed. v3 main run may launch.")
    else:
        n_fail = sum(1 for r in records if not r.get("passed", False))
        audit_note = "" if audit["passed"] else " [consumer_audit also failed]"
        print(
            f"VERDICT: FAIL — {n_fail}/{len(records)} model(s) failed"
            f"{audit_note}."
        )
        print("         DO NOT launch v3 main run until the pipeline clears.")
    print(f"report:   {out_dir / 'dryrun_report.json'}")
    print(f"parquet:  {parquet_path}")
    print("=" * 72)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

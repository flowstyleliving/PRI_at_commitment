#!/usr/bin/env python3
"""
===============================================================================
PRI v2 x SUP Experiment Pipeline (MLX)
Predictive Rupture Index — FIM-Pullback Upgrade on Apple Silicon
===============================================================================
Author: Michael Seo R. Kitti (adapted for mlx-lm pipeline)
Date:   March 2026
Status: Active Research

Run:
    pip install mlx-lm numpy scipy pandas matplotlib seaborn scikit-learn pyarrow
    python pri_v2_mlx_pipeline.py

This script intentionally uses `mlx_lm` as a Python library (NOT server mode)
so we can access hidden states, logits, probabilities, and unembedding weights.
===============================================================================
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "mlx-lm and mlx are required. Install with: pip install mlx-lm"
    ) from exc

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 0: CONFIGURATION                                     ║
# ╚════════════════════════════════════════════════════════════════╝


@dataclass
class Config:
    # Experiment design
    n_samples_per_cell: int = 200
    chain_lengths: List[int] = field(default_factory=lambda: [2, 5])
    pilot_n: int = 20
    pilot_threshold: float = 0.80

    # Models (MLX 4-bit community builds)
    models: List[str] = field(
        default_factory=lambda: [
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ]
    )

    # Generation
    max_new_tokens: int = 30
    n_trace_dumps: int = 5  # per condition (control/contradiction), per model

    # PRI parameters
    alpha_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0])
    alpha_default: float = 1.0
    topk_values: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    lowrank_values: List[int] = field(default_factory=lambda: [8, 16, 32])
    v3_rank_values: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 8, 13, 16, 21, 32, 34, 55, 64])
    layers_to_probe: List[str] = field(default_factory=lambda: ["final", "mid", "quarter"])

    # v3 capture schedule (opt-in via v3_capture=True on trace_sample).
    # For gen steps < v3_all_layers_for_first_n_steps, capture every transformer block.
    # For later steps, fall back to probe_4_layers.
    probe_4_layers: List[str] = field(
        default_factory=lambda: ["final", "three_quarters", "mid", "quarter"]
    )
    v3_all_layers_for_first_n_steps: int = 12
    # Sanity gate: ||Δh_step0|| / ||h_t_step0|| must be < this at the final layer.
    # Guards the paper's step-0 h_prev inflation bug (paper AUROCs 0.998/0.994/0.980
    # were artefacts of an undefined h_prev at step 0).
    h_prev_sanity_max_ratio: float = 10.0

    # Stats
    n_permutations: int = 10000
    bootstrap_n: int = 4000

    # Output
    save_dir: str = "./pri_v2_results"
    seed: int = 42


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
np.random.seed(cfg.seed)
random.seed(cfg.seed)


def print_header(text: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {text}")
    print(f"{'=' * 72}")


def clear_mlx_cache() -> None:
    try:
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except Exception:
        pass


def to_numpy(arr: Any) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    try:
        mx.eval(arr)
    except Exception:
        pass
    try:
        return np.array(arr)
    except Exception:
        return np.array(mx.eval(arr))


def safe_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-12)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return np.nan


def write_frame(df: pd.DataFrame, base_path_no_ext: str) -> str:
    parquet_path = f"{base_path_no_ext}.parquet"
    csv_path = f"{base_path_no_ext}.csv"
    parent = os.path.dirname(parquet_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def read_frame_if_exists(base_path_no_ext: str) -> Optional[pd.DataFrame]:
    parquet_path = f"{base_path_no_ext}.parquet"
    csv_path = f"{base_path_no_ext}.csv"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def checkpoint_meta_path(ckpt_base: str) -> str:
    return f"{ckpt_base}_meta.json"


def checkpoint_signature(config: Config, model_name: str) -> Dict[str, Any]:
    return {
        "model": model_name,
        "seed": int(config.seed),
        "n_samples_per_cell": int(config.n_samples_per_cell),
        "chain_lengths": [int(x) for x in config.chain_lengths],
        "pilot_n": int(config.pilot_n),
        "pilot_threshold": float(config.pilot_threshold),
        "max_new_tokens": int(config.max_new_tokens),
        "alpha_values": [float(x) for x in config.alpha_values],
        "alpha_default": float(config.alpha_default),
        "topk_values": [int(x) for x in config.topk_values],
        "lowrank_values": [int(x) for x in config.lowrank_values],
        "v3_rank_values": [int(x) for x in config.v3_rank_values],
        "layers_to_probe": list(config.layers_to_probe),
    }


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: DATA GENERATION                                   ║
# ╚════════════════════════════════════════════════════════════════╝


class PuzzleGenerator:
    """Synthetic logic puzzles — 2x2 factorial (chain_length x contradiction).

    Aligned with v1 synthetic format:
    - Universal chain: All X are Y ...
    - Subject assignment
    - Injected control/contradiction statement on subject-target relation
    - Forced YES/NO question
    """

    TERM_POOL = [
        "glorp",
        "blen",
        "trune",
        "vask",
        "mordin",
        "krel",
        "zenith",
        "prax",
        "nuvin",
        "seral",
        "thalen",
        "quorin",
        "dravan",
        "melta",
        "sorin",
        "valen",
        "torin",
        "dorin",
        "virel",
        "jorin",
    ]

    SUBJECT_POOL = [
        "Flib",
        "Nara",
        "Tovin",
        "Rell",
        "Sema",
        "Varn",
        "Kiro",
        "Mela",
        "Drax",
        "Luni",
        "Pavo",
        "Rima",
    ]

    WORKED_EXAMPLE = (
        "Instruction: Read the premises and answer the final question from those premises.\n\n"
        "Premises:\n"
        "1. All round things are smooth things.\n"
        "2. Blex is a round thing.\n"
        "3. Blex is a smooth thing.\n"
        "Question: Is Blex a smooth thing? Answer with only YES or NO.\n"
        "Answer: YES\n\n"
        "Now solve the following:\n\n"
    )

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_puzzle(self, chain_length: int, contradiction: bool) -> Dict[str, Any]:
        if chain_length + 1 > len(self.TERM_POOL):
            raise ValueError("TERM_POOL too small for requested chain length")

        terms = self.rng.sample(self.TERM_POOL, chain_length + 1)
        target = terms[-1]
        subject_base = self.rng.choice(self.SUBJECT_POOL)
        subject = f"{subject_base}{self.rng.randint(0, 9999):04d}"

        premise_lines: List[str] = []
        for i in range(chain_length):
            premise_lines.append(f"{i + 1}. All {terms[i]}s are {terms[i + 1]}s.")

        subject_line_num = chain_length + 1
        subject_line = f"{subject_line_num}. {subject} is a {terms[0]}."

        inject_line_num = chain_length + 2
        if contradiction:
            injected = f"{inject_line_num}. {subject} is not a {target}."
        else:
            injected = f"{inject_line_num}. {subject} is a {target}."

        question = f"Question: Is {subject} a {target}? Answer with only YES or NO."
        intro = "Instruction: Read the premises and answer the final question from those premises."
        premises_block = "\n".join(
            [intro, "", "Premises:"] + premise_lines + [subject_line, injected]
        )
        prompt = self.WORKED_EXAMPLE + premises_block + "\n" + question
        correct_answer = "NO" if contradiction else "YES"

        return {
            "prompt": prompt,
            "chain_length": chain_length,
            "contradiction": contradiction,
            "subject": subject,
            "target": target,
            "terms": terms,
            "correct_value": correct_answer,
            "injected_statement": injected,
        }

    def generate_dataset(self, n_per_cell: int, chain_lengths: List[int]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        sample_id = 0
        for cl in chain_lengths:
            for contradiction in [False, True]:
                for _ in range(n_per_cell):
                    row = self.generate_puzzle(cl, contradiction)
                    row["sample_id"] = sample_id
                    rows.append(row)
                    sample_id += 1

        df = pd.DataFrame(rows)
        df = df.sample(frac=1, random_state=self.rng.randint(0, 2**31 - 1)).reset_index(drop=True)
        return df


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: MODEL LOADING                                     ║
# ╚════════════════════════════════════════════════════════════════╝


def find_layers(model: Any) -> List[Any]:
    candidates = [
        ("model", "layers"),
        ("model", "h"),
        ("transformer", "layers"),
        ("transformer", "h"),
        ("layers",),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and obj is not None:
            try:
                if len(obj) > 0:
                    return list(obj)
            except Exception:
                pass
    raise RuntimeError("Could not locate transformer layers on model.")


def get_layer_indices(n_layers: int, targets: List[str]) -> Dict[str, int]:
    mapping = {
        "final": n_layers - 1,
        "three_quarters": (3 * n_layers) // 4,
        "mid": n_layers // 2,
        "quarter": n_layers // 4,
    }
    return {name: mapping[name] for name in targets if name in mapping}


def get_all_layer_indices(n_layers: int) -> Dict[str, int]:
    """For v3 every-layer capture. Names are zero-padded for stable lexicographic sort."""
    width = max(2, len(str(max(n_layers - 1, 0))))
    return {f"layer_{i:0{width}d}": i for i in range(n_layers)}


def layer_indices_for_step(
    step_idx: int,
    n_layers: int,
    all_for_first_n_steps: int,
    probe_fallback_targets: List[str],
) -> Dict[str, int]:
    """v3 capture schedule. step_idx is zero-indexed from the first generated token.

    Steps < `all_for_first_n_steps` (default 12) → every transformer block.
    Later steps → the `probe_fallback_targets` subset (default probe_4).
    """
    if step_idx < all_for_first_n_steps:
        return get_all_layer_indices(n_layers)
    return get_layer_indices(n_layers, probe_fallback_targets)


class OutputProjection:
    """
    Unified hidden->logits projector for MLX models.
    Supports:
      - untied lm_head layers (quantized or dense)
      - tied embeddings via embed_tokens.as_linear
    """

    def __init__(self, model: Any):
        self.model = model
        self.layer = None
        self.mode = ""

        if hasattr(model, "lm_head"):
            self.layer = model.lm_head
            self.mode = "lm_head"
        elif (
            hasattr(model, "model")
            and hasattr(model.model, "embed_tokens")
            and hasattr(model.model.embed_tokens, "as_linear")
        ):
            self.layer = model.model.embed_tokens
            self.mode = "tied_embed"
        else:
            raise RuntimeError(
                "Could not locate output projection (lm_head or embed_tokens.as_linear)."
            )

        args = getattr(model, "args", None)
        self.hidden_size = int(
            getattr(args, "hidden_size", 0)
            or getattr(args, "n_embd", 0)
            or self._infer_hidden_size()
        )
        self.vocab_size = int(
            getattr(args, "vocab_size", 0)
            or self._infer_vocab_size()
        )

    @staticmethod
    def _layer_get(layer: Any, key: str) -> Any:
        try:
            return layer[key]
        except Exception:
            pass
        if hasattr(layer, key):
            return getattr(layer, key)
        if hasattr(layer, "get"):
            try:
                return layer.get(key)
            except Exception:
                pass
        return None

    def _infer_hidden_size(self) -> int:
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            norm = self.model.model.norm
            w = self._layer_get(norm, "weight")
            if w is not None and hasattr(w, "shape") and len(w.shape) == 1:
                return int(w.shape[0])
        return 0

    def _infer_vocab_size(self) -> int:
        w = self._layer_get(self.layer, "weight")
        if w is not None and hasattr(w, "shape") and len(w.shape) >= 1:
            return int(w.shape[0])
        return 0

    def project(self, hidden_vec: np.ndarray) -> np.ndarray:
        h = mx.array(hidden_vec.astype(np.float32)[None, None, :])
        if self.mode == "tied_embed":
            out = self.layer.as_linear(h)
        else:
            out = self.layer(h)
        return np.array(out)[0, 0].astype(np.float32)

    def get_rows(self, indices: np.ndarray) -> Optional[np.ndarray]:
        idx = np.asarray(indices, dtype=np.int32)
        if idx.size == 0:
            return None

        w = self._layer_get(self.layer, "weight")
        if w is None:
            return None

        scales = self._layer_get(self.layer, "scales")
        biases = self._layer_get(self.layer, "biases")

        idx_mx = mx.array(idx)
        if scales is not None:
            # Quantized layer: dequantize only selected rows.
            w_rows = mx.take(w, idx_mx, axis=0)
            s_rows = mx.take(scales, idx_mx, axis=0)
            b_rows = mx.take(biases, idx_mx, axis=0) if biases is not None else None
            rows = mx.dequantize(w_rows, s_rows, b_rows)
            return np.array(rows).astype(np.float32)

        rows = mx.take(w, idx_mx, axis=0)
        rows_np = np.array(rows).astype(np.float32)
        if rows_np.ndim == 2:
            return rows_np
        return None


def encode_text(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text))
    if callable(tokenizer):
        out = tokenizer(text)
        if isinstance(out, dict) and "input_ids" in out:
            return list(out["input_ids"])
    raise RuntimeError("Could not encode text with tokenizer.")


def decode_ids(tokenizer: Any, token_ids: List[int]) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(token_ids)
        except Exception:
            pass
    if hasattr(tokenizer, "detokenizer"):
        dt = tokenizer.detokenizer
        if hasattr(dt, "detokenize"):
            return dt.detokenize(token_ids)
    return ""


def get_eos_token_id(tokenizer: Any) -> Optional[int]:
    for attr in ["eos_token_id", "eod_id"]:
        if hasattr(tokenizer, attr):
            val = getattr(tokenizer, attr)
            if val is not None:
                return int(val)
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "eos_token_id"):
        val = tokenizer.tokenizer.eos_token_id
        if val is not None:
            return int(val)
    return None


def load_model(
    model_name: str,
) -> Tuple[Any, Any, OutputProjection, Dict[str, int]]:
    print(f"\n  Loading model: {model_name}")
    model, tokenizer = mlx_load(model_name)

    all_layers = find_layers(model)
    layer_indices = get_layer_indices(len(all_layers), cfg.layers_to_probe)
    projection = OutputProjection(model)

    print(
        "     Layers: "
        f"{len(all_layers)} | Probed: {layer_indices} | "
        f"Output projection: {projection.mode} "
        f"(V={projection.vocab_size}, D={projection.hidden_size})"
    )

    return model, tokenizer, projection, layer_indices


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: DUAL-PHASE TRACING                                ║
# ╚════════════════════════════════════════════════════════════════╝


def trace_sample(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_indices: Dict[str, int],
    output_projection: OutputProjection,
    max_new_tokens: int,
    *,
    v3_capture: bool = False,
    v3_all_for_first_n_steps: int = 12,
    v3_probe_fallback: Optional[List[str]] = None,
    h_prev_sanity_max_ratio: float = 10.0,
) -> Dict[str, Any]:
    """Run prefix + generation tracing while collecting hidden states.

    The default path (v3_capture=False) is the paper pipeline: at each gen step
    capture only the layers in `layer_indices`, last position only.

    When `v3_capture=True`:
      * Per-step schedule: every transformer block for the first N gen steps,
        then `v3_probe_fallback` (default probe_4) for later steps.
      * Two-position capture per layer: position T (h_t) and position T-1
        (h_prev_causal). Under causal attention, h_prev_causal at step k equals
        the step-(k-1) h_t (k≥1) or the last prefix hidden (k=0).
      * Emits `gen_captures_by_step`, `h_prev_source_log`, and a step-0 sanity
        ratio under the key `step0_sanity`.
      * The paper-path outputs (`gen_hidden`, `last_prefix_hidden`, …) are still
        populated for any caller in `layer_indices`.
    """
    from mlx_lm.models.base import create_attention_mask

    core = model.model if hasattr(model, "model") else model
    layers = find_layers(model)
    n_layers = len(layers)
    if v3_probe_fallback is None:
        v3_probe_fallback = ["final", "three_quarters", "mid", "quarter"]

    def _forward_with_hidden(
        ids_2d: np.ndarray,
        target_idx_to_name: Dict[int, str],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        x = mx.array(ids_2d.astype(np.int32))
        if hasattr(core, "embed_tokens"):
            h = core.embed_tokens(x)
        elif hasattr(core, "wte"):
            h = core.wte(x)
        else:
            raise RuntimeError("Could not locate token embedding layer on model.")

        fa_mask = create_attention_mask(h, None)
        swa_mask = None
        if hasattr(core, "swa_idx") and getattr(core, "swa_idx") is not None:
            swa_mask = create_attention_mask(
                h, None, window_size=getattr(core, "sliding_window", None)
            )

        selected_hidden: Dict[str, np.ndarray] = {}

        for li, layer in enumerate(layers):
            mask = (
                swa_mask
                if (swa_mask is not None and hasattr(layer, "use_sliding") and layer.use_sliding)
                else fa_mask
            )
            try:
                h = layer(h, mask, cache=None)
            except TypeError:
                try:
                    h = layer(h, mask, None)
                except TypeError:
                    h = layer(h, mask)

            if li in target_idx_to_name:
                mx.eval(h)
                selected_hidden[target_idx_to_name[li]] = to_numpy(h).astype(np.float32)

        if hasattr(core, "norm"):
            h = core.norm(h)
        elif hasattr(core, "final_layernorm"):
            h = core.final_layernorm(h)

        if output_projection.mode == "tied_embed":
            logits = output_projection.layer.as_linear(h)
        else:
            logits = output_projection.layer(h)

        mx.eval(logits)
        logits_np = to_numpy(logits).astype(np.float32)
        return logits_np, selected_hidden

    # Paper-path target set (static across steps).
    paper_target_idx_to_name = {idx: name for name, idx in layer_indices.items()}

    def _targets_for_step(
        step_idx: int,
    ) -> Tuple[Dict[int, str], Dict[str, int], Dict[str, int]]:
        """Return (idx→canonical_name, canonical_name→idx, paper_path_aliases)
        for a gen step.

        Two naming schemes coexist under v3_capture: (a) *canonical* names from
        the v3 schedule — `layer_NN` for every-layer steps, probe_fallback
        names for probe steps — one name per layer index; (b) *paper-path*
        names (`final`, `mid`, `quarter`) which are fractional-depth roles
        that often land on indices already covered by (a). Merging both at
        capture time collides under the idx→name inversion and loses the
        colliding canonical entry.

        3b — canonical-only capture + aliases at the output boundary. Paper-
        path aliases are returned alongside the canonical map so the caller
        can post-process captured hiddens into paper-path keys (same data,
        additional keys) without a capture-time collision.
        """
        if v3_capture:
            canonical = layer_indices_for_step(
                step_idx, n_layers, v3_all_for_first_n_steps, v3_probe_fallback
            )
            # Paper-path names that aren't already canonical at this step are
            # real aliases. At probe-regime steps, probe_fallback IS paper-path
            # so the alias dict is usually empty — that's fine.
            paper_aliases = {
                name: idx for name, idx in layer_indices.items()
                if name not in canonical
            }
        else:
            canonical = dict(layer_indices)
            paper_aliases = {}
        return (
            {idx: name for name, idx in canonical.items()},
            canonical,
            paper_aliases,
        )

    token_ids = encode_text(tokenizer, prompt)
    if len(token_ids) < 2:
        raise RuntimeError("Prompt too short after tokenization.")

    input_ids = np.array(token_ids, dtype=np.int32)[None, :]
    eos_id = get_eos_token_id(tokenizer)

    # Prefix phase — capture every layer during prefix when v3_capture is on so
    # `last_prefix_hidden` is available for any layer name the gen schedule will
    # request. Paper-path prefix captures only `layer_indices`.
    if v3_capture:
        # Canonical capture targets: one name per index. Paper-path names
        # (final, mid, quarter) are added as aliases post-capture — see 3b
        # note on `_targets_for_step`.
        canonical_prefix_targets = get_all_layer_indices(n_layers)
        prefix_paper_aliases = {
            name: idx for name, idx in layer_indices.items()
            if name not in canonical_prefix_targets
        }
    else:
        canonical_prefix_targets = dict(layer_indices)
        prefix_paper_aliases = {}
    prefix_idx_to_name = {
        idx: name for name, idx in canonical_prefix_targets.items()
    }
    prefix_logits, prefix_selected_hidden = _forward_with_hidden(
        input_ids, prefix_idx_to_name
    )

    prefix_hidden: Dict[str, List[np.ndarray]] = {}
    last_prefix_hidden: Dict[str, np.ndarray] = {}

    for lname in canonical_prefix_targets:
        if lname not in prefix_selected_hidden:
            raise RuntimeError(f"No hidden state captured for layer '{lname}'.")
        act = prefix_selected_hidden[lname]
        if act.ndim != 3:
            raise RuntimeError(
                f"Unexpected activation shape for '{lname}': {act.shape}, expected [B,T,D]."
            )
        seq_vecs = [act[0, t].astype(np.float32) for t in range(act.shape[1])]
        prefix_hidden[lname] = seq_vecs
        last_prefix_hidden[lname] = seq_vecs[-1]

    # Alias paper-path names to the canonical layer_NN captures at the same
    # index. Shares the list objects so the paper-path v2 consumer sees the
    # same data as if it had been captured directly under the paper-path name.
    _idx_to_canonical_prefix = {
        idx: name for name, idx in canonical_prefix_targets.items()
    }
    for paper_name, idx in prefix_paper_aliases.items():
        canonical_name = _idx_to_canonical_prefix.get(idx)
        if canonical_name is not None and canonical_name in prefix_hidden:
            prefix_hidden.setdefault(paper_name, prefix_hidden[canonical_name])
            last_prefix_hidden.setdefault(
                paper_name, last_prefix_hidden[canonical_name]
            )

    # Prefix probabilities and prefix surprise over actual next token.
    prefix_logits_2d = prefix_logits[0].astype(np.float32)  # [T, V]
    prefix_probs = np.stack([safe_softmax(z) for z in prefix_logits_2d], axis=0)
    prefix_surprises = np.full(prefix_probs.shape[0], np.nan, dtype=np.float32)
    for t in range(prefix_probs.shape[0] - 1):
        nxt = token_ids[t + 1]
        prefix_surprises[t] = float(-math.log(prefix_probs[t, nxt] + 1e-10))

    # Generation phase (greedy)
    gen_hidden: Dict[str, List[np.ndarray]] = {name: [] for name in layer_indices}
    gen_logits: List[np.ndarray] = []
    gen_probs: List[np.ndarray] = []
    gen_surprises: List[float] = []
    gen_token_ids: List[int] = []

    # v3 capture structures — indexed per-step, per-layer-name.
    gen_captures_by_step: List[Dict[str, Dict[str, np.ndarray]]] = []
    gen_layer_indices_by_step: List[Dict[str, int]] = []
    h_prev_source_log: List[str] = []
    step0_sanity: Dict[str, float] = {}

    # At the start, prev_probs comes from the last prefix position.
    prev_probs = prefix_probs[-1]
    # Current logits for choosing the first token come from prefix.
    current_logits_vec = prefix_logits_2d[-1].astype(np.float32)

    for step_idx in range(max_new_tokens):
        # 1. Choose next token from current logits.
        next_id = int(np.argmax(current_logits_vec))

        # 2. Compute surprise using the probability distribution BEFORE this token was appended.
        surprise = float(-math.log(prev_probs[next_id] + 1e-10))

        # 3. Check EOS before committing.
        if eos_id is not None and next_id == eos_id:
            break

        # 4. Append token to context.
        token_ids.append(next_id)
        gen_token_ids.append(next_id)
        gen_surprises.append(surprise)

        # 5. Run forward with the updated context (includes the new token).
        step_idx_to_name, step_canonical, step_paper_aliases = (
            _targets_for_step(step_idx)
        )
        step_input = np.array(token_ids, dtype=np.int32)[None, :]
        step_logits, step_selected_hidden = _forward_with_hidden(
            step_input, step_idx_to_name
        )

        # Alias paper-path names onto the canonical captures so the paper-path
        # consumer loop below (and any v2 downstream reader) sees the same
        # data under paper-path keys. No extra capture, just shared refs.
        _idx_to_canonical_step = {
            idx: name for name, idx in step_canonical.items()
        }
        for paper_name, idx in step_paper_aliases.items():
            canonical_name = _idx_to_canonical_step.get(idx)
            if (
                canonical_name is not None
                and canonical_name in step_selected_hidden
            ):
                step_selected_hidden.setdefault(
                    paper_name, step_selected_hidden[canonical_name]
                )

        # 6. Capture h_t (position T) for paper-path layers and build the v3
        #    two-position capture when enabled.
        for lname in layer_indices:
            act = step_selected_hidden[lname]
            gen_hidden[lname].append(act[0, -1].astype(np.float32))

        if v3_capture:
            step_captures: Dict[str, Dict[str, np.ndarray]] = {}
            # Iterate canonical names only — paper-path aliases share data via
            # step_selected_hidden but do NOT get their own parquet row (one
            # row per physical layer index, under its canonical name).
            for lname, _li in step_canonical.items():
                act = step_selected_hidden[lname]
                h_t = act[0, -1].astype(np.float32)
                h_prev_causal = act[0, -2].astype(np.float32)
                # H4 write-once guard: duplicate layer keys within one step
                # are a capture-store bug, not a silent overwrite.
                assert lname not in step_captures, (
                    f"H4 dict-collision: step {step_idx}, layer '{lname}' "
                    f"(idx {_li}) already present in step_captures"
                )
                step_captures[lname] = {
                    "h_t": h_t,
                    "h_prev_causal": h_prev_causal,
                }
            gen_captures_by_step.append(step_captures)
            gen_layer_indices_by_step.append(dict(step_canonical))
            h_prev_source_log.append(
                "prefix_last" if step_idx == 0 else "gen_prev"
            )

            # Step-0 sanity gate at the final layer: Δh / ||h_t|| < ratio.
            # Guards the paper's step-0 h_prev inflation bug.
            if step_idx == 0:
                final_name = None
                if "final" in step_captures:
                    final_name = "final"
                else:
                    # Fall back to the highest-index canonical name. Under 3b
                    # this is layer_NN where NN = n_layers - 1 at every-layer
                    # steps, which is the same physical layer as "final".
                    final_name = max(
                        step_canonical, key=lambda n: step_canonical[n]
                    )
                h_t_final = step_captures[final_name]["h_t"]
                h_prev_causal_final = step_captures[final_name]["h_prev_causal"]
                h_prev_prefix_final = last_prefix_hidden.get(final_name)
                ht_norm = float(np.linalg.norm(h_t_final) + 1e-30)
                dh_causal = float(
                    np.linalg.norm(h_t_final - h_prev_causal_final)
                )
                ratio = dh_causal / ht_norm
                step0_sanity = {
                    "layer_name": final_name,
                    "layer_index": float(step_canonical[final_name]),
                    "h_t_norm": ht_norm,
                    "dh_causal_norm": dh_causal,
                    "dh_over_ht": ratio,
                    "max_ratio_allowed": float(h_prev_sanity_max_ratio),
                }
                if h_prev_prefix_final is not None:
                    step0_sanity["dh_prefix_norm"] = float(
                        np.linalg.norm(h_t_final - h_prev_prefix_final)
                    )
                    # Under causal attention, position T-1 of the extended pass
                    # must approximately equal the last prefix hidden for the
                    # same layer. 4-bit quantized MLX forwards drift more than
                    # fp32 would, so we compare via relative L2 distance with
                    # a loose threshold rather than `np.allclose` elementwise.
                    diff = h_prev_causal_final - h_prev_prefix_final
                    ref_norm = float(np.linalg.norm(h_prev_prefix_final)) + 1e-30
                    rel_l2 = float(np.linalg.norm(diff) / ref_norm)
                    step0_sanity["causal_vs_prefix_rel_l2"] = rel_l2
                    step0_sanity["causal_matches_prefix_last"] = float(rel_l2 < 1e-2)
                if ratio >= h_prev_sanity_max_ratio or not np.isfinite(ratio):
                    raise RuntimeError(
                        f"step-0 h_prev sanity failed at layer '{final_name}': "
                        f"||Δh||/||h_t|| = {ratio:.3f} (max {h_prev_sanity_max_ratio:.3f}). "
                        f"This is the paper's step-0 inflation bug — refusing to proceed."
                    )

        # 7. Capture logits/probs for this step and next iteration.
        step_logits_vec = step_logits[0, -1].astype(np.float32)
        step_probs_vec = safe_softmax(step_logits_vec)
        gen_logits.append(step_logits_vec)
        gen_probs.append(step_probs_vec)
        prev_probs = step_probs_vec
        current_logits_vec = step_logits_vec

    generated_text = decode_ids(tokenizer, gen_token_ids)

    return {
        "prefix_hidden": prefix_hidden,
        "last_prefix_hidden": last_prefix_hidden,
        "prefix_probs": prefix_probs,
        "prefix_surprises": prefix_surprises,
        "gen_hidden": gen_hidden,
        "gen_logits": gen_logits,
        "gen_probs": gen_probs,
        "gen_surprises": gen_surprises,
        "gen_token_ids": gen_token_ids,
        "generated_text": generated_text,
        # v3 capture outputs (populated only when v3_capture=True).
        "gen_captures_by_step": gen_captures_by_step,
        "gen_layer_indices_by_step": gen_layer_indices_by_step,
        "h_prev_source_log": h_prev_source_log,
        "step0_sanity": step0_sanity,
        "v3_capture": bool(v3_capture),
        "n_layers": n_layers,
    }


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: PRI COMPUTATION ENGINE                             ║
# ╚════════════════════════════════════════════════════════════════╝


class PRIComputer:
    """Computes PRI v1 and PRI v2 variants."""

    def __init__(self, output_projection: OutputProjection):
        self.output_projection = output_projection

    @staticmethod
    def cosine_dist(h_t: np.ndarray, h_prev: np.ndarray) -> float:
        num = float(np.dot(h_t, h_prev))
        den = float(np.linalg.norm(h_t) * np.linalg.norm(h_prev) + 1e-10)
        return 1.0 - num / den

    @staticmethod
    def l2_dist(h_t: np.ndarray, h_prev: np.ndarray) -> float:
        return float(np.linalg.norm(h_t - h_prev))

    def _project(self, dh: np.ndarray) -> np.ndarray:
        return self.output_projection.project(dh)

    @staticmethod
    def fim_diag_from_proj(z: np.ndarray, p_t: np.ndarray) -> float:
        return float(np.sqrt(np.sum(p_t * (z**2)) + 1e-10))

    @staticmethod
    def fim_full_from_proj(z: np.ndarray, p_t: np.ndarray) -> float:
        ez2 = float(np.sum(p_t * (z**2)))
        ez = float(np.sum(p_t * z))
        return float(np.sqrt(max(ez2 - ez**2, 1e-10)))

    @staticmethod
    def fim_topk_from_proj(z: np.ndarray, p_t: np.ndarray, k: int = 64) -> float:
        k = int(min(k, p_t.shape[0]))
        idx = np.argpartition(-p_t, kth=k - 1)[:k]
        p_k = p_t[idx]
        p_k = p_k / (np.sum(p_k) + 1e-10)
        z_k = z[idx]
        ez2 = float(np.sum(p_k * (z_k**2)))
        ez = float(np.sum(p_k * z_k))
        return float(np.sqrt(max(ez2 - ez**2, 1e-10)))

    def fim_lowrank(
        self,
        dh: np.ndarray,
        z: np.ndarray,
        p_t: np.ndarray,
        rank: int = 16,
    ) -> float:
        """
        Approximate low-rank pullback by SVD over a probability-truncated support.
        Uses row sampling from the model-native output projection.
        """
        support = int(min(max(256, rank * 16), p_t.shape[0]))
        idx = np.argpartition(-p_t, kth=support - 1)[:support]
        p_s = p_t[idx]
        W_s = self.output_projection.get_rows(idx)
        if W_s is None or W_s.ndim != 2:
            return self.fim_full_from_proj(z, p_t)

        A = (np.sqrt(p_s + 1e-10)[:, None]) * W_s
        try:
            _, S, Vt = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            return self.fim_full_from_proj(z, p_t)

        r = int(min(rank, Vt.shape[0]))
        proj = Vt[:r] @ dh
        d2 = float(np.sum((S[:r] ** 2) * (proj**2)))

        mean_proj = float(np.sum(p_t * z))
        return float(np.sqrt(max(d2 - mean_proj**2, 1e-10)))

    def null_ratio_and_energy(
        self,
        dh: np.ndarray,
        p_t: np.ndarray,
        rank_values: Iterable[int],
    ) -> Dict[str, float]:
        """
        For each r in rank_values, emit:
          null_ratio_rank{r}     = ||dh - V_top V_topᵀ dh|| / ||dh||
          fisher_energy_rank{r}  = Σ_{i≤r} σ_i² / Σ_i σ_i²
        V_top = top-r right singular vectors of sqrt(p_t)·W_u restricted to
        the top-`support` probability rows (same truncation as fim_lowrank).
        Note: null is measured relative to that truncated support, not the
        full vocab — null_ratio > 0 can reflect either within-support modes
        beyond rank r OR directions outside the row-span entirely.
        SVD is shared across all requested ranks.
        """
        out: Dict[str, float] = {}
        rank_list = [int(r) for r in rank_values]
        if not rank_list:
            return out

        dh_norm_sq = float(np.dot(dh, dh))
        if dh_norm_sq <= 0.0:
            for r in rank_list:
                out[f"null_ratio_rank{r}"] = 0.0
                out[f"fisher_energy_rank{r}"] = 0.0
            return out
        dh_norm = float(np.sqrt(dh_norm_sq))

        max_rank = max(rank_list)
        support = int(min(max(256, max_rank * 16), p_t.shape[0]))
        idx = np.argpartition(-p_t, kth=support - 1)[:support]
        p_s = p_t[idx]
        W_s = self.output_projection.get_rows(idx)

        nan_out = {
            **{f"null_ratio_rank{r}": float("nan") for r in rank_list},
            **{f"fisher_energy_rank{r}": float("nan") for r in rank_list},
        }
        if W_s is None or W_s.ndim != 2:
            return nan_out

        A = (np.sqrt(p_s + 1e-10)[:, None]) * W_s
        try:
            _, S, Vt = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            return nan_out

        s_sq = S**2
        total_energy = float(np.sum(s_sq)) + 1e-10
        proj = Vt @ dh
        cum_proj_sq = np.cumsum(proj**2)
        cum_energy = np.cumsum(s_sq)
        max_available = Vt.shape[0]

        for r in rank_list:
            r_eff = int(min(r, max_available))
            top_proj_sq = float(cum_proj_sq[r_eff - 1]) if r_eff > 0 else 0.0
            null_sq = max(dh_norm_sq - top_proj_sq, 0.0)
            out[f"null_ratio_rank{r}"] = float(np.sqrt(null_sq) / dh_norm)
            out[f"fisher_energy_rank{r}"] = float(cum_energy[r_eff - 1] / total_energy) if r_eff > 0 else 0.0

        return out

    @staticmethod
    def pri_v1(S_t: float, delta_h: float, alpha: float) -> float:
        return S_t * (1.0 + alpha * delta_h)

    @staticmethod
    def pri_v2(S_t: float, d_F: float, alpha: float) -> float:
        return S_t + alpha * d_F

    def compute_step(
        self,
        h_t: np.ndarray,
        h_prev: np.ndarray,
        p_t: np.ndarray,
        S_t: float,
        alpha: float,
        topk_values: Iterable[int],
        lowrank_values: Iterable[int],
        v3_rank_values: Iterable[int] = (),
    ) -> Dict[str, float]:
        cos_d = self.cosine_dist(h_t, h_prev)
        l2_d = self.l2_dist(h_t, h_prev)
        dh = h_t - h_prev
        z = self._project(dh)

        # Defensive guard for rare tokenizer/model mismatches.
        if z.shape[0] != p_t.shape[0]:
            m = min(z.shape[0], p_t.shape[0])
            z = z[:m]
            p_t = p_t[:m]
            p_t = p_t / (np.sum(p_t) + 1e-10)

        out: Dict[str, float] = {
            "surprise": S_t,
            "delta_h_cosine": cos_d,
            "delta_h_l2": l2_d,
            "pri_v1_cosine": self.pri_v1(S_t, cos_d, alpha),
            "pri_v1_l2": self.pri_v1(S_t, l2_d, alpha),
        }

        d_diag = self.fim_diag_from_proj(z, p_t)
        out["d_F_diag"] = d_diag
        out["pri_v2_diag"] = self.pri_v2(S_t, d_diag, alpha)

        d_full = self.fim_full_from_proj(z, p_t)
        out["d_F_full"] = d_full
        out["pri_v2_full"] = self.pri_v2(S_t, d_full, alpha)

        for k in topk_values:
            d = self.fim_topk_from_proj(z, p_t, k=int(k))
            out[f"d_F_topk{k}"] = d
            out[f"pri_v2_topk{k}"] = self.pri_v2(S_t, d, alpha)

        for rank in lowrank_values:
            d = self.fim_lowrank(dh, z, p_t, rank=int(rank))
            out[f"d_F_lowrank{rank}"] = d
            out[f"pri_v2_lowrank{rank}"] = self.pri_v2(S_t, d, alpha)

        v3_list = list(v3_rank_values)
        if v3_list:
            v3_out = self.null_ratio_and_energy(dh, p_t, v3_list)
            out.update(v3_out)

        return out


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: STATISTICAL ANALYSIS                               ║
# ╚════════════════════════════════════════════════════════════════╝


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan, (np.nan, np.nan)

    m1, m2 = g1.mean(), g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    s_pool = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2) + 1e-12)
    d = (m1 - m2) / (s_pool + 1e-12)

    df = n1 + n2 - 2
    J = 1 - 3 / (4 * df - 1)
    g = d * J
    se = np.sqrt((n1 + n2) / (n1 * n2) + (g**2) / (2 * df)) * J
    return float(g), (float(g - 1.96 * se), float(g + 1.96 * se))


def stratified_perm_test(
    values: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> float:
    values = np.asarray(values)
    labels = np.asarray(labels).astype(int)
    strata = np.asarray(strata)

    finite = np.isfinite(values)
    if finite.sum() < 4:
        return np.nan
    values = values[finite]
    labels = labels[finite]
    strata = strata[finite]

    if len(np.unique(labels)) < 2:
        return np.nan

    rng = np.random.RandomState(seed)
    obs = values[labels == 1].mean() - values[labels == 0].mean()

    count = 0
    uniq = np.unique(strata)

    for _ in range(n_perm):
        perm = labels.copy()
        for s in uniq:
            m = strata == s
            perm[m] = rng.permutation(perm[m])
        stat = values[perm == 1].mean() - values[perm == 0].mean()
        if stat >= obs:
            count += 1

    return float((count + 1) / (n_perm + 1))


def bootstrap_auc_diff(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 4000,
    seed: int = 42,
) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    a = np.asarray(score_a)
    b = np.asarray(score_b)

    rng = np.random.RandomState(seed)
    n = len(labels)

    base_a = safe_auroc(labels, a)
    base_b = safe_auroc(labels, b)
    base_diff = base_a - base_b

    if np.isnan(base_diff):
        return {
            "auc_a": np.nan,
            "auc_b": np.nan,
            "diff": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p_two_sided": np.nan,
        }

    diffs: List[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        lb = labels[idx]
        if len(np.unique(lb)) < 2:
            continue
        d = safe_auroc(lb, a[idx]) - safe_auroc(lb, b[idx])
        if not np.isnan(d):
            diffs.append(float(d))

    if not diffs:
        return {
            "auc_a": base_a,
            "auc_b": base_b,
            "diff": base_diff,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "p_two_sided": np.nan,
        }

    arr = np.array(diffs)
    ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
    p = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))

    return {
        "auc_a": float(base_a),
        "auc_b": float(base_b),
        "diff": float(base_diff),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_two_sided": float(min(1.0, p)),
    }


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: MAIN EXPERIMENT LOOP                               ║
# ╚════════════════════════════════════════════════════════════════╝


def check_answer(generated: str, expected_value: str) -> bool:
    """Extract the first explicit YES/NO token from generated text."""
    if generated is None:
        return False
    expected = str(expected_value).strip().upper()
    if expected not in {"YES", "NO"}:
        return expected.lower() in str(generated).lower()

    text = re.sub(r"<\|[^|>]+?\|>", " ", str(generated).strip())
    for match in re.finditer(r"[A-Za-z]+", text):
        token = match.group(0).upper()
        if token in {"YES", "NO"}:
            return token == expected

    # Fallback for uncommon tokenization patterns.
    low = text.lower()
    return expected.lower() in low


def run_experiment(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print_header("PRI v2 x SUP EXPERIMENT PIPELINE (MLX)")
    os.makedirs(config.save_dir, exist_ok=True)

    print_header("SECTION 1: DATA GENERATION")
    gen = PuzzleGenerator(seed=config.seed)
    dataset = gen.generate_dataset(config.n_samples_per_cell, config.chain_lengths)
    print(f"  Generated {len(dataset)} samples")
    print(dataset.groupby(["chain_length", "contradiction"]).size().unstack(fill_value=0))

    all_results: List[Dict[str, Any]] = []
    all_trace_dumps: List[Dict[str, Any]] = []

    for model_name in config.models:
        print_header(f"MODEL: {model_name}")

        short = model_name.split("/")[-1]
        result_base = os.path.join(config.save_dir, f"{short}_results")
        trace_dumps_base = os.path.join(config.save_dir, f"{short}_trace_dumps")

        existing_results_df = read_frame_if_exists(result_base)
        if existing_results_df is not None and len(existing_results_df) > 0:
            all_results.extend(existing_results_df.to_dict("records"))
            print(
                "  Completed results found: "
                f"{os.path.basename(result_base)} "
                f"({len(existing_results_df)} rows, "
                f"{existing_results_df['sample_id'].nunique()} samples). Skipping model."
            )

            existing_dumps_df = read_frame_if_exists(trace_dumps_base)
            if existing_dumps_df is not None and len(existing_dumps_df) > 0:
                all_trace_dumps.extend(existing_dumps_df.to_dict("records"))
                print(
                    "  Loaded existing trace dumps: "
                    f"{len(existing_dumps_df)} rows."
                )
            continue

        model, tokenizer, output_projection, layer_indices = load_model(model_name)
        pri_comp = PRIComputer(output_projection)

        ckpt_base = os.path.join(config.save_dir, f"{short}_checkpoint")
        ckpt_meta = checkpoint_meta_path(ckpt_base)
        expected_sig = checkpoint_signature(config, model_name)
        model_results: List[Dict[str, Any]] = []
        processed_sample_ids: set[int] = set()

        ckpt_df = read_frame_if_exists(ckpt_base)
        if ckpt_df is not None and len(ckpt_df) > 0:
            compatible = False
            if os.path.exists(ckpt_meta):
                try:
                    with open(ckpt_meta, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    compatible = meta.get("signature") == expected_sig
                except Exception:
                    compatible = False

            if compatible:
                model_results = ckpt_df.to_dict("records")
                processed_sample_ids = set(int(x) for x in ckpt_df["sample_id"].unique())
                print(
                    f"  Resume checkpoint found: {len(ckpt_df)} rows, {len(processed_sample_ids)} samples."
                )
            else:
                print("  Checkpoint found but signature mismatch (or missing metadata). Ignoring stale checkpoint.")
                for ext in [".parquet", ".csv"]:
                    stale = f"{ckpt_base}{ext}"
                    if os.path.exists(stale):
                        os.remove(stale)
                if os.path.exists(ckpt_meta):
                    os.remove(ckpt_meta)

        # Behavioral gate on unprocessed control samples
        controls = dataset[~dataset.contradiction]
        controls = controls[~controls["sample_id"].isin(processed_sample_ids)].head(config.pilot_n)
        if len(controls) < config.pilot_n:
            controls = dataset[~dataset.contradiction].head(config.pilot_n)

        print(f"  Behavioral gate: {len(controls)} control samples")
        n_correct = 0
        for _, row in controls.iterrows():
            try:
                trace = trace_sample(
                    model,
                    tokenizer,
                    row["prompt"],
                    layer_indices,
                    output_projection,
                    config.max_new_tokens,
                )
                if check_answer(trace["generated_text"], row["correct_value"]):
                    n_correct += 1
            except Exception as exc:
                print(f"    Gate sample error (id={row['sample_id']}): {exc}")

        gate_acc = n_correct / max(len(controls), 1)
        print(f"    Control accuracy: {gate_acc:.0%}")
        if gate_acc < config.pilot_threshold:
            print(
                f"    Gate failed (need >= {config.pilot_threshold:.0%}). Skipping model."
            )
            del model, tokenizer, output_projection, pri_comp
            gc.collect()
            clear_mlx_cache()
            continue

        run_df = dataset[~dataset["sample_id"].isin(processed_sample_ids)].copy()
        n_total = len(run_df)
        t0 = time.time()
        diagnostic_printed = False
        n_ctrl_dumps = 0
        n_contr_dumps = 0
        trace_dumps: List[Dict[str, Any]] = []

        print(f"  Full run: {n_total} remaining samples")
        for idx, (_, row) in enumerate(run_df.iterrows(), start=1):
            if idx % 40 == 0:
                elapsed = time.time() - t0
                rate = idx / max(elapsed, 1e-9)
                eta_min = (n_total - idx) / max(rate, 1e-9) / 60
                print(
                    f"    {idx}/{n_total} ({idx / max(n_total,1):.0%}) | "
                    f"{rate:.2f} samp/s | ETA {eta_min:.1f}m | rows {len(model_results)}"
                )
                ckpt_file = write_frame(pd.DataFrame(model_results), ckpt_base)
                print(f"    Checkpoint saved: {os.path.basename(ckpt_file)}")
                with open(ckpt_meta, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "signature": expected_sig,
                            "rows": int(len(model_results)),
                            "saved_at": float(time.time()),
                        },
                        f,
                    )
                gc.collect()
                clear_mlx_cache()

            try:
                trace = trace_sample(
                    model,
                    tokenizer,
                    row["prompt"],
                    layer_indices,
                    output_projection,
                    config.max_new_tokens,
                )
            except Exception as exc:
                print(f"    Sample {row['sample_id']} failed: {exc}")
                continue

            is_correct = check_answer(trace["generated_text"], row["correct_value"])
            n_steps = len(trace["gen_surprises"])

            # Per-sample trace dumps for a small balanced subset (control/contradiction).
            is_contr = bool(row["contradiction"])
            want_dump = (
                (not is_contr and n_ctrl_dumps < config.n_trace_dumps)
                or (is_contr and n_contr_dumps < config.n_trace_dumps)
            )
            if want_dump:
                dump_layer = "final" if "final" in layer_indices else list(layer_indices.keys())[0]
                dump_alpha = float(config.alpha_default)

                gen_pri_v1_cosine: List[float] = []
                gen_pri_v2_full: List[float] = []
                gen_delta_h_cosine: List[float] = []
                gen_d_F_full: List[float] = []

                for step in range(n_steps):
                    S_t_dump = float(trace["gen_surprises"][step])
                    h_t_dump = trace["gen_hidden"][dump_layer][step]
                    if step == 0:
                        h_prev_dump = trace["last_prefix_hidden"][dump_layer]
                    else:
                        h_prev_dump = trace["gen_hidden"][dump_layer][step - 1]
                    p_t_dump = trace["gen_probs"][step]
                    dump_metrics = pri_comp.compute_step(
                        h_t_dump,
                        h_prev_dump,
                        p_t_dump,
                        S_t_dump,
                        dump_alpha,
                        topk_values=(),
                        lowrank_values=(),
                    )

                    gen_delta_h_cosine.append(float(dump_metrics["delta_h_cosine"]))
                    gen_d_F_full.append(float(dump_metrics["d_F_full"]))
                    gen_pri_v1_cosine.append(float(dump_metrics["pri_v1_cosine"]))
                    gen_pri_v2_full.append(float(dump_metrics["pri_v2_full"]))

                trace_dump = {
                    "model": model_name,
                    "sample_id": int(row["sample_id"]),
                    "chain_length": int(row["chain_length"]),
                    "contradiction": is_contr,
                    "is_correct": bool(is_correct),
                    "prompt": row["prompt"],
                    "generated_text": trace["generated_text"],
                    "gen_token_ids": [int(t) for t in trace["gen_token_ids"]],
                    "prefix_surprises": trace["prefix_surprises"].tolist(),
                    "gen_surprises": [float(v) for v in trace["gen_surprises"]],
                    "gen_pri_v1_cosine": gen_pri_v1_cosine,
                    "gen_pri_v2_full": gen_pri_v2_full,
                    "gen_delta_h_cosine": gen_delta_h_cosine,
                    "gen_d_F_full": gen_d_F_full,
                }
                trace_dumps.append(trace_dump)
                all_trace_dumps.append(trace_dump)
                if is_contr:
                    n_contr_dumps += 1
                else:
                    n_ctrl_dumps += 1

            for step in range(n_steps):
                S_t = float(trace["gen_surprises"][step])

                for layer_name in layer_indices:
                    h_t = trace["gen_hidden"][layer_name][step]
                    if step == 0:
                        h_prev = trace["last_prefix_hidden"][layer_name]
                    else:
                        h_prev = trace["gen_hidden"][layer_name][step - 1]

                    p_t = trace["gen_probs"][step]

                    for alpha in config.alpha_values:
                        metrics = pri_comp.compute_step(
                            h_t,
                            h_prev,
                            p_t,
                            S_t,
                            alpha,
                            topk_values=config.topk_values,
                            lowrank_values=config.lowrank_values,
                            v3_rank_values=config.v3_rank_values,
                        )

                        if (
                            not diagnostic_printed
                            and step == 0
                            and layer_name == "final"
                            and alpha == config.alpha_default
                        ):
                            print(
                                "    DIAGNOSTIC: "
                                f"sample_id={int(row['sample_id'])} "
                                f"S_t={S_t:.4f} "
                                f"cos_d={metrics['delta_h_cosine']:.6f} "
                                f"d_F_full={metrics['d_F_full']:.6f} "
                                f"l2={metrics['delta_h_l2']:.6f}"
                            )
                            print(
                                "    DIAGNOSTIC vectors: "
                                f"h_t[:5]={np.array2string(h_t[:5], precision=4)} "
                                f"h_prev[:5]={np.array2string(h_prev[:5], precision=4)}"
                            )
                            diagnostic_printed = True

                        row_out = {
                            "model": model_name,
                            "sample_id": int(row["sample_id"]),
                            "chain_length": int(row["chain_length"]),
                            "contradiction": bool(row["contradiction"]),
                            "gen_step": step + 1,
                            "is_correct": bool(is_correct),
                            "generated_text": trace["generated_text"],
                            "layer": layer_name,
                            "alpha": float(alpha),
                            **metrics,
                        }
                        model_results.append(row_out)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed / 60:.1f} min")

        all_results.extend(model_results)
        model_df = pd.DataFrame(model_results)
        model_file = write_frame(model_df, result_base)
        print(f"  Saved model results: {os.path.basename(model_file)} ({len(model_df)} rows)")
        if trace_dumps:
            dumps_df = pd.DataFrame(trace_dumps)
            dumps_file = write_frame(dumps_df, trace_dumps_base)
            print(f"  Saved trace dumps: {os.path.basename(dumps_file)} ({len(dumps_df)} rows)")

        for ext in [".parquet", ".csv"]:
            p = f"{ckpt_base}{ext}"
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(ckpt_meta):
            os.remove(ckpt_meta)

        del model, tokenizer, output_projection, pri_comp
        gc.collect()
        clear_mlx_cache()

    results = pd.DataFrame(all_results)
    all_file = write_frame(results, os.path.join(config.save_dir, "all_results"))
    print(f"\n  Total rows: {len(results)} | saved: {os.path.basename(all_file)}")
    if all_trace_dumps:
        all_dumps_file = write_frame(
            pd.DataFrame(all_trace_dumps),
            os.path.join(config.save_dir, "all_trace_dumps"),
        )
        print(f"  Trace dumps total: {len(all_trace_dumps)} | saved: {os.path.basename(all_dumps_file)}")

    return results, dataset


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: ANALYSIS                                           ║
# ╚════════════════════════════════════════════════════════════════╝


def outcome_independence_stats(
    ctrl: np.ndarray, corr: np.ndarray, incorr: np.ndarray
) -> Dict[str, float]:
    g_corr, _ = hedges_g(corr, ctrl)
    g_incorr, _ = hedges_g(incorr, ctrl)
    g_diff, _ = hedges_g(corr, incorr)
    return {
        "mean_ctrl": float(np.nanmean(ctrl)) if len(ctrl) else np.nan,
        "mean_correct": float(np.nanmean(corr)) if len(corr) else np.nan,
        "mean_incorrect": float(np.nanmean(incorr)) if len(incorr) else np.nan,
        "g_correct_vs_ctrl": g_corr,
        "g_incorrect_vs_ctrl": g_incorr,
        "g_correct_vs_incorrect": g_diff,
    }


def run_analysis(results: pd.DataFrame, config: Config) -> pd.DataFrame:
    print_header("ANALYSIS")

    if results.empty:
        print("  No results available; skipping analysis.")
        return pd.DataFrame()

    pri_cols = sorted([c for c in results.columns if c.startswith("pri_")])
    summary_rows: List[Dict[str, Any]] = []

    for model_name in sorted(results["model"].unique()):
        print_header(f"RESULTS: {model_name.split('/')[-1]}")

        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
        ]
        if s1.empty:
            print("  No step-1 final-layer rows.")
            continue

        labels = s1["contradiction"].astype(int).values
        ctrl = s1[~s1["contradiction"]]
        contr = s1[s1["contradiction"]]

        print(
            f"  Accuracy | Control: {ctrl['is_correct'].mean():.1%} | "
            f"Contradiction: {contr['is_correct'].mean():.1%}"
        )

        print(f"\n  {'Variant':<24} {'g':>7} {'95% CI':>18} {'AUROC':>8} {'p_perm':>10}")
        print(f"  {'-' * 74}")

        for col in pri_cols:
            g, ci = hedges_g(contr[col].values, ctrl[col].values)
            auc = safe_auroc(labels, s1[col].values)
            p = stratified_perm_test(
                s1[col].values,
                labels,
                s1["chain_length"].values,
                n_perm=config.n_permutations,
                seed=config.seed,
            )
            if not np.isnan(auc) and auc < 0.5:
                star = " <- INVERTED"
            elif col.startswith("pri_v2") and (not np.isnan(auc) and auc > 0.99):
                star = " <- best?"
            else:
                star = ""
            print(
                f"  {col:<24} {g:>7.3f} "
                f"[{ci[0]:>6.2f},{ci[1]:>6.2f}] {auc:>8.4f} {p:>10.4f}{star}"
            )

            summary_rows.append(
                {
                    "model": model_name.split("/")[-1],
                    "variant": col,
                    "hedges_g": g,
                    "ci_lo": ci[0],
                    "ci_hi": ci[1],
                    "auroc": auc,
                    "p_value": p,
                }
            )

        # Outcome independence on best v2 by AUROC
        v2_cols = [c for c in pri_cols if c.startswith("pri_v2_")]
        if v2_cols:
            best_v2 = max(v2_cols, key=lambda c: safe_auroc(labels, s1[c].values))
            c_ctrl = s1[~s1["contradiction"]][best_v2].values
            c_corr = s1[s1["contradiction"] & s1["is_correct"]][best_v2].values
            c_inc = s1[s1["contradiction"] & ~s1["is_correct"]][best_v2].values

            oi = outcome_independence_stats(c_ctrl, c_corr, c_inc)
            print(f"\n  Outcome Independence ({best_v2})")
            print(
                f"    Ctrl={oi['mean_ctrl']:.4f} | Corr={oi['mean_correct']:.4f} "
                f"(g={oi['g_correct_vs_ctrl']:.3f})"
            )
            print(
                f"    Incorr={oi['mean_incorrect']:.4f} "
                f"(g={oi['g_incorrect_vs_ctrl']:.3f}) | "
                f"Corr-Incorr g={oi['g_correct_vs_incorrect']:.3f}"
            )

        # Head-to-head significance (bootstrap AUC diff)
        if "pri_v1_cosine" in s1.columns and "pri_v2_full" in s1.columns:
            comp = bootstrap_auc_diff(
                labels,
                s1["pri_v2_full"].values,
                s1["pri_v1_cosine"].values,
                n_boot=config.bootstrap_n,
                seed=config.seed,
            )
            print("\n  AUROC Difference (bootstrap): pri_v2_full - pri_v1_cosine")
            print(
                f"    diff={comp['diff']:.4f} | 95% CI [{comp['ci_lo']:.4f}, {comp['ci_hi']:.4f}] | "
                f"p={comp['p_two_sided']:.4f}"
            )

    # Alpha sweep report
    print_header("ALPHA SWEEP")
    for model_name in sorted(results["model"].unique()):
        short = model_name.split("/")[-1]
        print(f"\n  {short}")
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
        ]

        print(f"  {'Variant':<24}", end="")
        for a in config.alpha_values:
            print(f"  a={a:<5}", end="")
        print()

        for col in sorted([c for c in pri_cols if c.startswith("pri_")]):
            print(f"  {col:<24}", end="")
            for a in config.alpha_values:
                sub = s1[s1["alpha"] == a]
                auc = safe_auroc(sub["contradiction"].astype(int).values, sub[col].values)
                if np.isnan(auc):
                    print("    nan ", end="")
                else:
                    print(f"  {auc:.4f}", end="")
            print()

    # Layer sweep report
    print_header("LAYER SWEEP")
    for model_name in sorted(results["model"].unique()):
        short = model_name.split("/")[-1]
        print(f"\n  {short}")
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["alpha"] == config.alpha_default)
        ]

        v2_cols = [c for c in pri_cols if c.startswith("pri_v2_")]
        print(f"  {'Layer':<10}", end="")
        for col in v2_cols:
            print(f"  {col.replace('pri_v2_', ''):<12}", end="")
        print()

        for layer in config.layers_to_probe:
            sub = s1[s1["layer"] == layer]
            if sub.empty:
                continue
            print(f"  {layer:<10}", end="")
            labels = sub["contradiction"].astype(int).values
            for col in v2_cols:
                auc = safe_auroc(labels, sub[col].values)
                if np.isnan(auc):
                    print(f"  {'nan':<12}", end="")
                else:
                    print(f"  {auc:<12.4f}", end="")
            print()

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["model", "auroc"], ascending=[True, False])
    return summary


def log_failure_cases(results: pd.DataFrame, config: Config) -> pd.DataFrame:
    print_header("FAILURE CASE ANALYSIS")
    if results.empty:
        print("  No results available; skipping failure case logging.")
        return pd.DataFrame()

    failure_frames: List[pd.DataFrame] = []

    for model_name in sorted(results["model"].unique()):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
        ]
        if s1.empty:
            continue

        short = model_name.split("/")[-1]

        for variant in ["pri_v1_cosine", "pri_v2_full"]:
            if variant not in s1.columns:
                continue

            ctrl = s1[~s1["contradiction"]]
            contr = s1[s1["contradiction"]]
            if ctrl.empty or contr.empty:
                continue

            ctrl_median = ctrl[variant].median()
            contr_q25 = contr[variant].quantile(0.25)

            fn = s1[s1["contradiction"] & (s1[variant] <= ctrl_median)].copy()
            fn["failure_type"] = "false_negative"
            fn["variant"] = variant
            fn["ctrl_median"] = ctrl_median

            fp = s1[~s1["contradiction"] & (s1[variant] >= contr_q25)].copy()
            fp["failure_type"] = "false_positive"
            fp["variant"] = variant
            fp["ctrl_median"] = ctrl_median

            if not fn.empty:
                failure_frames.append(fn)
            if not fp.empty:
                failure_frames.append(fp)

            print(f"  {short} | {variant}")
            print(f"    False negatives (contr <= ctrl median): {len(fn)}")
            print(f"    False positives (ctrl >= contr q25): {len(fp)}")

        if "pri_v1_cosine" in s1.columns and "pri_v2_full" in s1.columns:
            ctrl_v1_med = s1[~s1["contradiction"]]["pri_v1_cosine"].median()
            ctrl_v2_med = s1[~s1["contradiction"]]["pri_v2_full"].median()

            v1_miss_v2_catch = s1[
                s1["contradiction"]
                & (s1["pri_v1_cosine"] <= ctrl_v1_med)
                & (s1["pri_v2_full"] > ctrl_v2_med)
            ].copy()
            v1_miss_v2_catch["failure_type"] = "v1_miss_v2_catch"
            v1_miss_v2_catch["variant"] = "comparison"
            v1_miss_v2_catch["ctrl_median"] = np.nan

            v2_miss_v1_catch = s1[
                s1["contradiction"]
                & (s1["pri_v2_full"] <= ctrl_v2_med)
                & (s1["pri_v1_cosine"] > ctrl_v1_med)
            ].copy()
            v2_miss_v1_catch["failure_type"] = "v2_miss_v1_catch"
            v2_miss_v1_catch["variant"] = "comparison"
            v2_miss_v1_catch["ctrl_median"] = np.nan

            if not v1_miss_v2_catch.empty:
                failure_frames.append(v1_miss_v2_catch)
            if not v2_miss_v1_catch.empty:
                failure_frames.append(v2_miss_v1_catch)

            print(f"    v1 miss / v2 catch: {len(v1_miss_v2_catch)}")
            print(f"    v2 miss / v1 catch: {len(v2_miss_v1_catch)}")

    if failure_frames:
        failures = pd.concat(failure_frames, ignore_index=True)
        path = write_frame(failures, os.path.join(config.save_dir, "failure_cases"))
        print(f"\n  Saved {len(failures)} failure case rows: {path}")
        return failures

    print("  No failure rows found.")
    return pd.DataFrame()


# ╔════════════════════════════════════════════════════════════════╗
# ║  SECTION 8: FIGURES                                            ║
# ╚════════════════════════════════════════════════════════════════╝


def make_figures(results: pd.DataFrame, config: Config) -> None:
    print_header("GENERATING FIGURES")
    if results.empty:
        print("  No results; skipping figures.")
        return

    save_dir = config.save_dir
    models = sorted(results["model"].unique())
    pri_cols = sorted([c for c in results.columns if c.startswith("pri_")])
    n_models = len(models)

    # Figure 1: AUROC bar chart
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
        ]
        labels = s1["contradiction"].astype(int).values
        auc_pairs = [(col, safe_auroc(labels, s1[col].values)) for col in pri_cols]
        auc_pairs = [x for x in auc_pairs if not np.isnan(x[1])]
        auc_pairs.sort(key=lambda x: x[1])

        names = [x[0].replace("pri_", "") for x in auc_pairs]
        vals = [x[1] for x in auc_pairs]
        colors = ["#1E88E5" if "v1" in x[0] else "#E53935" for x in auc_pairs]

        ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0.45, 1.01)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_xlabel("AUROC")
        ax.set_title(model_name.split("/")[-1], fontweight="bold")

    fig.suptitle("PRI v1 vs v2 AUROC (Step 1, Final Layer)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig1_v1_vs_v2_auroc.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, "fig1_v1_vs_v2_auroc.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Trajectory
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8), sharey="row")
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for ci, model_name in enumerate(models):
        md = results[
            (results["model"] == model_name)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
            & (results["gen_step"] <= 5)
        ]
        for ri, variant in enumerate(["pri_v1_cosine", "pri_v2_full"]):
            ax = axes[ri, ci]
            for cond, color, label in [
                (False, "#1E88E5", "Control"),
                (True, "#E53935", "Contradiction"),
            ]:
                sub = md[md["contradiction"] == cond]
                if sub.empty:
                    continue
                means = sub.groupby("gen_step")[variant].mean()
                sems = sub.groupby("gen_step")[variant].sem()
                ax.errorbar(
                    means.index,
                    means.values,
                    yerr=sems.values,
                    color=color,
                    marker="o",
                    capsize=3,
                    label=label,
                )
            ax.set_xlabel("Generation Step")
            ax.set_ylabel(variant.replace("pri_", ""))
            if ri == 0:
                ax.set_title(model_name.split("/")[-1], fontweight="bold")
            ax.legend(fontsize=8)

    fig.suptitle("PRI Trajectory: v1 (top) vs v2_full (bottom)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig2_step_trajectory.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Violin step-1
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8))
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for ci, model_name in enumerate(models):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
        ]
        for ri, variant in enumerate(["pri_v1_cosine", "pri_v2_full"]):
            ax = axes[ri, ci]
            pdv = s1[["contradiction", variant]].copy()
            pdv["condition"] = pdv["contradiction"].map(
                {False: "Control", True: "Contradiction"}
            )
            sns.violinplot(
                data=pdv,
                x="condition",
                y=variant,
                palette={"Control": "#1E88E5", "Contradiction": "#E53935"},
                cut=0,
                ax=ax,
            )
            g, _ = hedges_g(
                s1[s1["contradiction"]][variant].values,
                s1[~s1["contradiction"]][variant].values,
            )
            ax.set_title(f"{model_name.split('/')[-1]} | g={g:.2f}", fontsize=10)
            ax.set_ylabel(variant.replace("pri_", ""))

    fig.suptitle("Step-1 Distribution: v1 (top) vs v2_full (bottom)", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig3_violins.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: Outcome independence
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
            & (results["alpha"] == config.alpha_default)
        ]
        v = "pri_v2_full"
        groups = {
            "Control": s1[~s1["contradiction"]][v].values,
            "Contr-Correct": s1[s1["contradiction"] & s1["is_correct"]][v].values,
            "Contr-Incorrect": s1[s1["contradiction"] & ~s1["is_correct"]][v].values,
        }
        means = [np.nanmean(vv) if len(vv) else np.nan for vv in groups.values()]
        sems = [
            np.nanstd(vv) / np.sqrt(len(vv)) if len(vv) > 1 else 0.0
            for vv in groups.values()
        ]

        ax.bar(
            range(3),
            means,
            yerr=sems,
            capsize=5,
            color=["#1E88E5", "#43A047", "#E53935"],
            alpha=0.85,
        )
        ax.set_xticks(range(3))
        ax.set_xticklabels(list(groups.keys()), fontsize=8)
        ax.set_ylabel(v.replace("pri_", ""))
        ax.set_title(model_name.split("/")[-1], fontweight="bold")

    fig.suptitle("Outcome Independence (Step 1, v2_full)", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig4_outcome_independence.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: Alpha sweep
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    variants = ["pri_v1_cosine", "pri_v2_diag", "pri_v2_full", "pri_v2_topk64"]
    styles = ["--", "-", "-", "-."]

    for ax, model_name in zip(axes, models):
        s1 = results[
            (results["model"] == model_name)
            & (results["gen_step"] == 1)
            & (results["layer"] == "final")
        ]
        for variant, style in zip(variants, styles):
            if variant not in s1.columns:
                continue
            aucs: List[float] = []
            for a in config.alpha_values:
                sub = s1[s1["alpha"] == a]
                aucs.append(
                    safe_auroc(sub["contradiction"].astype(int).values, sub[variant].values)
                )
            ax.plot(
                config.alpha_values,
                aucs,
                style,
                marker="o",
                label=variant.replace("pri_", ""),
                markersize=4,
            )

        ax.set_xscale("log")
        ax.set_xlabel("alpha")
        ax.set_ylabel("AUROC")
        ax.set_title(model_name.split("/")[-1], fontweight="bold")
        ax.legend(fontsize=7)

    fig.suptitle("Alpha Sweep", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "fig5_alpha_sweep.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures written to: {save_dir}")


# ╔════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                          ║
# ╚════════════════════════════════════════════════════════════════╝


if __name__ == "__main__":
    print_header("START")

    results_df, dataset_df = run_experiment(cfg)

    summary_df = run_analysis(results_df, cfg)
    summary_path = write_frame(summary_df, os.path.join(cfg.save_dir, "summary"))
    failures_df = log_failure_cases(results_df, cfg)

    make_figures(results_df, cfg)

    print_header("COMPLETE")
    print(f"  Dataset samples: {len(dataset_df)}")
    print(f"  Result rows: {len(results_df)}")
    print(f"  Summary file: {summary_path}")
    print(f"  Failure rows: {len(failures_df)}")
    print(f"  Output dir: {cfg.save_dir}")

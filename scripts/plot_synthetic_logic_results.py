#!/usr/bin/env python3
"""
Plot event-aligned synthetic contradiction results from summary + raw JSONL.gz outputs.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PRIMARY_SIGNALS = ("pri", "delta_sigma_jsd", "acr_mid_mean")


def _load_summary(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_raw(path: str | Path, max_samples: int | None = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= int(max_samples):
                break
    return records


def _mean_by_offset(
    records: List[Dict[str, Any]],
    signal: str,
    window: int,
    has_contradiction: bool,
) -> np.ndarray:
    offsets = np.arange(-window, window + 1, 1, dtype=int)
    values: List[List[float]] = [[] for _ in offsets]
    for row in records:
        if bool(row.get("has_contradiction", False)) != bool(has_contradiction):
            continue
        for point in row.get("prefix_trace", []):
            rel = int(point.get("relative_offset", 0))
            if rel < -window or rel > window:
                continue
            idx = rel + window
            values[idx].append(float(point.get(signal, 0.0)))
    out = np.zeros_like(offsets, dtype=float)
    for i, bucket in enumerate(values):
        out[i] = float(np.mean(bucket)) if bucket else 0.0
    return out


def _plot_event_aligned(
    records: List[Dict[str, Any]],
    window: int,
    output_path: Path,
) -> None:
    offsets = np.arange(-window, window + 1, 1, dtype=int)
    fig, axes = plt.subplots(1, len(PRIMARY_SIGNALS), figsize=(14, 4), dpi=200)
    axes = np.atleast_1d(axes)
    for idx, signal in enumerate(PRIMARY_SIGNALS):
        ax = axes[idx]
        y_ctrl = _mean_by_offset(records, signal, window, has_contradiction=False)
        y_contra = _mean_by_offset(records, signal, window, has_contradiction=True)
        ax.plot(offsets, y_ctrl, linewidth=2, color="#1f77b4", label="Control")
        ax.plot(offsets, y_contra, linewidth=2, color="#d62728", label="Contradiction")
        ax.axvline(0, linestyle="--", linewidth=1, color="black", alpha=0.6)
        ax.set_title(signal)
        ax.set_xlabel("Relative token offset from anchor")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle(f"Event-aligned mean trajectories (window ±{window})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _collect_abs_peak_offsets(
    records: List[Dict[str, Any]],
    signal: str,
    window_key: str,
    has_contradiction: bool,
) -> List[float]:
    out: List[float] = []
    for row in records:
        if bool(row.get("has_contradiction", False)) != bool(has_contradiction):
            continue
        stats = row.get("window_summaries", {}).get(window_key, {}).get(signal)
        if not stats:
            continue
        out.append(float(stats.get("abs_peak_offset", 0.0)))
    return out


def _plot_peak_offsets(
    records: List[Dict[str, Any]],
    window: int,
    output_path: Path,
) -> None:
    window_key = f"w{window}"
    fig, axes = plt.subplots(1, len(PRIMARY_SIGNALS), figsize=(14, 4), dpi=200)
    axes = np.atleast_1d(axes)
    for idx, signal in enumerate(PRIMARY_SIGNALS):
        ax = axes[idx]
        ctrl = _collect_abs_peak_offsets(records, signal, window_key, has_contradiction=False)
        contra = _collect_abs_peak_offsets(records, signal, window_key, has_contradiction=True)
        bins = np.arange(0, window + 2, 1) - 0.5
        ax.hist(ctrl, bins=bins, alpha=0.6, color="#1f77b4", label="Control")
        ax.hist(contra, bins=bins, alpha=0.6, color="#d62728", label="Contradiction")
        ax.set_title(f"{signal} abs peak offset")
        ax.set_xlabel("|peak offset|")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle(f"Absolute peak-offset distributions (window ±{window})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot synthetic contradiction results")
    parser.add_argument("--summary", required=True, help="Path to *_summary.json")
    parser.add_argument("--raw", default=None, help="Optional raw JSONL.gz path override")
    parser.add_argument("--output-dir", default="./figures", help="Figure output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for plotting speed")
    args = parser.parse_args()

    summary = _load_summary(args.summary)
    raw_path = args.raw or summary.get("raw_trace_path")
    if not raw_path:
        raise ValueError("No raw trace path provided in summary or --raw")

    records = _load_raw(raw_path, max_samples=args.max_samples)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    w_tight = int(summary.get("run_config", {}).get("window_tight", 5))
    w_wide = int(summary.get("run_config", {}).get("window_wide", 12))

    outputs = {
        "event_aligned_w_tight": outdir / f"{Path(args.summary).stem}_event_aligned_w{w_tight}.png",
        "event_aligned_w_wide": outdir / f"{Path(args.summary).stem}_event_aligned_w{w_wide}.png",
        "peak_offsets_w_tight": outdir / f"{Path(args.summary).stem}_peak_offsets_w{w_tight}.png",
        "peak_offsets_w_wide": outdir / f"{Path(args.summary).stem}_peak_offsets_w{w_wide}.png",
    }

    _plot_event_aligned(records, w_tight, outputs["event_aligned_w_tight"])
    _plot_event_aligned(records, w_wide, outputs["event_aligned_w_wide"])
    _plot_peak_offsets(records, w_tight, outputs["peak_offsets_w_tight"])
    _plot_peak_offsets(records, w_wide, outputs["peak_offsets_w_wide"])

    figure_paths = {k: str(v) for k, v in outputs.items()}
    print(json.dumps({"figure_paths": figure_paths}, indent=2))


if __name__ == "__main__":
    main()

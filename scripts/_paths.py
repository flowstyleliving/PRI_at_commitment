"""Shared path helpers for experiment scripts.

Artifacts live under <repo>/experiments/<slug>/<YYYY-MM-DD>/run-NN/.
NN is auto-incremented per day: the first run is run-01, the next run-02, etc.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"


def experiment_run_dir(slug: str, run_date: str | None = None) -> Path:
    """Return a freshly-numbered run directory for an experiment.

    Creates <repo>/experiments/<slug>/<YYYY-MM-DD>/run-NN/. NN is the next
    two-digit number after the highest existing run-NN in that date directory.
    """
    day = run_date or date.today().isoformat()
    date_dir = EXPERIMENTS_ROOT / slug / day
    date_dir.mkdir(parents=True, exist_ok=True)

    nums = []
    for entry in date_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("run-"):
            suffix = entry.name[len("run-"):]
            if suffix.isdigit():
                nums.append(int(suffix))
    next_num = (max(nums) + 1) if nums else 1
    run_dir = date_dir / f"run-{next_num:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

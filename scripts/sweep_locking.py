#!/usr/bin/env python3
"""Shared lock helpers for long-running sweep orchestrators."""
from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator

LOCK_FILENAME = ".sweep.lock.json"


def _lock_metadata() -> Dict[str, object]:
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at_iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
    }


def _format_existing_lock(lock_path: Path) -> str:
    try:
        meta = json.loads(lock_path.read_text())
    except Exception:
        return f"another sweep already holds {lock_path} (lockfile unreadable)"
    return (
        f"another sweep already holds {lock_path}: "
        f"pid={meta.get('pid')} host={meta.get('hostname')} "
        f"started_at={meta.get('started_at_iso')} cwd={meta.get('cwd')}"
    )


@contextmanager
def hold_out_dir_lock(out_dir: Path) -> Iterator[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / LOCK_FILENAME
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RuntimeError(_format_existing_lock(lock_path)) from exc

    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_lock_metadata(), f, indent=2, sort_keys=True)
            f.write("\n")
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def claim_next_run_dir(base_dir: Path, *, prefix: str = "run-", width: int = 2) -> Path:
    """Atomically create the next numbered run directory under `base_dir`."""
    base_dir.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = base_dir / f"{prefix}{idx:0{width}d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            idx += 1

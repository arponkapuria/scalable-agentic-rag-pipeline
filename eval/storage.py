"""
Per-question JSON result files, one per (stage, question_id) — the whole
resumability mechanism for every eval run. A run checks whether the
key(s) it's about to compute already exist in the question's file before
making any call; a crash loses at most the one call in flight, never a
whole question, and a fresh process just continues where the last one
left off. This is what makes an eval spanning several days (hostage to
Groq's daily token cap) actually SAFE to re-run: `python -m eval.cli
generate` after Groq's daily reset only recomputes what's missing.

Also used generically to load eval/dataset/questions.json (any JSON file,
not just a per-question result) — same load() function, no special-casing
needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def question_path(stage: str, results_dir: Path, qid: str) -> Path:
    d = results_dir / stage
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{qid}.json"


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write-to-temp-then-rename — os.replace is atomic on POSIX, so a
    crash mid-write never leaves a half-written / corrupted result file
    for the next run to trip over."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def update(stage: str, results_dir: Path, qid: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Load, merge in `updates`, atomic-rewrite. The merge (not overwrite)
    is what lets, e.g., run_judge.py fill in `faithfulness` on a file
    run_generation.py already wrote `shipped` into, without either step
    needing to know about the other's keys."""
    path = question_path(stage, results_dir, qid)
    current = load(path)
    current.update(updates)
    save_atomic(path, current)
    return current


def has_keys(stage: str, results_dir: Path, qid: str, keys: list[str]) -> bool:
    data = load(question_path(stage, results_dir, qid))
    return all(k in data and data[k] is not None for k in keys)
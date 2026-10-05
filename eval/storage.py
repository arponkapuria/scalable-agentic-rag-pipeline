"""Per-question JSON result storage that lets every eval stage resume after a crash."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def question_path(stage: str, results_dir: Path, qid: str) -> Path:
    """Returns the result file path for a stage and question, creating the stage folder if needed.

    Args:
        stage: Stage name such as "retrieval", "generation" or "judge".
        results_dir: Root results directory.
        qid: Question id.
    """
    d = results_dir / stage
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{qid}.json"


def load(path: Path) -> dict[str, Any]:
    """Reads a JSON file, returning an empty dict if it doesn't exist."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_atomic(path: Path, data: dict[str, Any]) -> None:
    """Writes JSON to a temp file then renames it, so a crash never leaves a corrupt file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def update(stage: str, results_dir: Path, qid: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Merges new keys into a question's result file and returns the merged data.

    Merging lets different stages add their own keys to the same file without overwriting each other.
    """
    path = question_path(stage, results_dir, qid)
    current = load(path)
    current.update(updates)
    save_atomic(path, current)
    return current


def has_keys(stage: str, results_dir: Path, qid: str, keys: list[str]) -> bool:
    """Returns True if every key is present and non-null, used to skip finished work."""
    data = load(question_path(stage, results_dir, qid))
    return all(k in data and data[k] is not None for k in keys)
"""LIBERO-Plus task and language helpers.

The official Cosmos checkpoint contains embeddings for the 40 original LIBERO
instructions. Most LIBERO-Plus visual perturbations append implementation
suffixes to the task language. Those suffixes are not instructions and must not
trigger an on-demand 11B T5 load. Genuine language perturbations are preserved.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

LANGUAGE_CATEGORY = "Language Instructions"
_SPACE = re.compile(r"\s+")


def normalize_instruction(text: str) -> str:
    return _SPACE.sub(" ", text.strip().lower())


def load_embedding_keys(path: str) -> list[str]:
    with open(path, "rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"T5 cache must contain a dict, got {type(data)!r}")
    return list(data)


def resolve_instruction(
    task_language: str,
    category: str,
    embedding_keys: list[str],
) -> tuple[str, str]:
    """Return (policy instruction, resolution method).

    Visual/physics perturbation suffixes are mapped to the longest original
    instruction prefix. Language perturbations may only use their exact
    paraphrase embedding; falling back to the original would invalidate the
    language robustness experiment.
    """

    normalized = normalize_instruction(task_language)
    by_normalized = {normalize_instruction(key): key for key in embedding_keys}
    if normalized in by_normalized:
        return by_normalized[normalized], "exact"
    if category == LANGUAGE_CATEGORY:
        raise KeyError(
            "Missing T5 embedding for a genuine language perturbation: "
            f"{task_language!r}. Precompute this paraphrase; do not substitute "
            "the original instruction."
        )
    candidates = [
        (norm, original)
        for norm, original in by_normalized.items()
        if normalized.startswith(norm + " ")
    ]
    if not candidates:
        raise KeyError(
            f"Cannot map LIBERO-Plus instruction {task_language!r} to any "
            "checkpoint-cached original instruction."
        )
    _, original = max(candidates, key=lambda pair: len(pair[0]))
    return original, "canonical_prefix"


def load_task_list(path: str, task_names: list[str] | None = None) -> list[dict[str, Any]]:
    with open(path) as stream:
        payload = json.load(stream)
    tasks = payload["tasks"] if isinstance(payload, dict) else payload
    if task_names:
        wanted = set(task_names)
        tasks = [task for task in tasks if task["name"] in wanted]
        missing = wanted - {task["name"] for task in tasks}
        if missing:
            raise KeyError(f"task_names absent from task list: {sorted(missing)}")
    return tasks


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def task_list_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


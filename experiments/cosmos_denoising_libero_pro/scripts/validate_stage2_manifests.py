#!/usr/bin/env python3
"""Fail-fast validation for all data-driven stage-2 manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
MANIFESTS = EXPERIMENT / "manifests"
OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
AUDIT = (
    OUTPUT
    / "assets/cosmos_libero_pro_t5_embeddings_stage2.audit.json"
)
FILES = {
    "stage2_original_seed_repeats.jsonl": 320,
    "stage2_hard_subset.jsonl": 5824,
    "stage2_language_all_paraphrases.jsonl": 480,
    "stage2_disagreement_video_replay.jsonl": 833,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    embeddings = audit["embeddings"]
    errors: list[str] = []
    rows: list[dict] = []
    counts = {}
    for name, expected in FILES.items():
        selected = load_jsonl(MANIFESTS / name)
        counts[name] = len(selected)
        rows.extend(selected)
        if len(selected) != expected:
            errors.append(f"{name}: {len(selected)} rows != {expected}")
    duplicates = [
        key
        for key, count in Counter(row["episode_key"] for row in rows).items()
        if count != 1
    ]
    if duplicates:
        errors.append(f"duplicate episode keys: {duplicates[:3]}")

    checked_files: set[tuple[str, str]] = set()
    init_lengths: dict[str, int | None] = {}
    for row in rows:
        if row.get("stage") != "stage2":
            errors.append(f"stage invariant failed: {row['episode_key']}")
        is_video_replay = "video_replay" in str(row["config_id"])
        allowed_steps = {1, 2, 3, 4, 5, 6} if is_video_replay else {1, 3, 5, 6}
        if row["denoising_steps"] not in allowed_steps:
            errors.append(f"unselected denoising step: {row['episode_key']}")
        if (
            row["action_horizon"] != 16
            or row["decode_future"]
            or not row["deterministic"]
            or row["randomize_seed"]
        ):
            errors.append(f"runtime invariant failed: {row['episode_key']}")
        instruction = row.get("instruction")
        if not isinstance(instruction, str):
            errors.append(f"non-string instruction: {row['episode_key']}")
            continue
        metadata = embeddings.get(instruction)
        if metadata is None:
            errors.append(f"T5 instruction missing: {instruction!r}")
        elif (
            row.get("t5_embedding_sha256") != metadata["tensor_sha256"]
            or metadata["shape"] != [1, 512, 1024]
        ):
            errors.append(f"T5 tensor mismatch: {row['episode_key']}")
        for path_field, hash_field in (
            ("bddl_path", "bddl_sha256"),
            ("init_path", "init_sha256"),
        ):
            path = Path(row[path_field])
            identity = (str(path), row[hash_field])
            if identity in checked_files:
                continue
            checked_files.add(identity)
            if not path.is_file() or sha256(path) != row[hash_field]:
                errors.append(f"{path_field} missing/hash mismatch: {path}")
        init_path = row["init_path"]
        if init_path not in init_lengths:
            try:
                states = torch.load(init_path, map_location="cpu", weights_only=False)
                init_lengths[init_path] = len(states)
                if len(states) < 1:
                    errors.append(f"empty init file: {init_path}")
            except Exception as error:
                init_lengths[init_path] = None
                errors.append(f"init unreadable {init_path}: {error}")
        length = init_lengths[init_path]
        if length is not None and int(row["init_state_index"]) >= length:
            errors.append(f"init index out of range: {row['episode_key']}")

    result = {
        "status": "failed" if errors else "passed",
        "episodes": len(rows),
        "unique_episode_keys": len({row["episode_key"] for row in rows}),
        "repeat_selected_steps": sorted(
            {
                int(row["denoising_steps"])
                for row in rows
                if "video_replay" not in str(row["config_id"])
            }
        ),
        "video_replay_steps": sorted(
            {
                int(row["denoising_steps"])
                for row in rows
                if "video_replay" in str(row["config_id"])
            }
        ),
        "unique_instructions": len({row["instruction"] for row in rows}),
        "t5_exact_coverage": bool(audit["exact_coverage"]),
        "counts": counts,
        "errors": errors,
    }
    target = EXPERIMENT / "summaries/stage2_manifest_validation.json"
    target.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

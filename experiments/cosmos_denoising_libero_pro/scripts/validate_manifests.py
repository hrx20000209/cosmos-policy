#!/usr/bin/env python3
"""Fail-fast validation of manifests, hashes, T5 coverage and invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT / "experiments/cosmos_denoising_libero_pro"
OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-t5", action="store_true")
    args = parser.parse_args()
    original = load(EXPERIMENT / "manifests/original_full.jsonl")
    pro = load(EXPERIMENT / "manifests/libero_pro_full.jsonl")
    supported_pro = load(EXPERIMENT / "manifests/libero_pro_full_supported.jsonl")
    rows = original + pro
    errors = []
    if len(original) != 240:
        errors.append(f"original rows {len(original)} != 240")
    if len(pro) != 1200:
        errors.append(f"PRO rows {len(pro)} != 1200")
    if len(supported_pro) != 1170:
        errors.append(f"supported PRO rows {len(supported_pro)} != 1170")
    keys = Counter(row["episode_key"] for row in rows)
    duplicates = [key for key, count in keys.items() if count != 1]
    if duplicates:
        errors.append(f"duplicate episode keys: {duplicates[:3]}")
    for category in ("language", "object", "position", "task", "environment"):
        selected = [row for row in pro if row["perturbation_category"] == category]
        if len(selected) != 240:
            errors.append(f"{category}: {len(selected)} rows != 240")
    for row in rows:
        bddl = Path(row["bddl_path"])
        init = Path(row["init_path"])
        if not bddl.is_file() or sha256(bddl) != row["bddl_sha256"]:
            errors.append(f"BDDL missing/hash mismatch: {bddl}")
            continue
        if not init.is_file() or sha256(init) != row["init_sha256"]:
            errors.append(f"init missing/hash mismatch: {init}")
            continue
        try:
            states = torch.load(init, weights_only=False)
            if int(row["init_state_index"]) >= len(states):
                errors.append(f"init index out of range: {init}")
        except Exception as error:
            errors.append(f"init unreadable {init}: {error}")
        if row["denoising_steps"] not in {1, 2, 3, 4, 5, 6}:
            errors.append(f"forbidden step: {row['denoising_steps']}")
        if row["action_horizon"] != 16 or row["decode_future"]:
            errors.append(f"runtime invariant failed: {row['episode_key']}")
    if not args.skip_t5:
        cache_path = OUTPUT / "assets/cosmos_libero_pro_t5_embeddings.pkl"
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
        required = {row["instruction"] for row in rows}
        missing = sorted(required - set(cache))
        extra = sorted(set(cache) - required)
        if missing:
            errors.append(f"T5 missing exact instructions: {missing[:3]}")
        if extra:
            errors.append(f"T5 has {len(extra)} non-manifest keys")
        for command in required & set(cache):
            if tuple(cache[command].shape) != (1, 512, 1024):
                errors.append(f"T5 shape mismatch: {command!r}")
        t5_manifest = json.loads(
            (EXPERIMENT / "manifests/libero_pro_t5_cache_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        tensor_hashes = {
            command: metadata["tensor_sha256"]
            for command, metadata in t5_manifest["instructions"].items()
        }
        for row in rows:
            if row.get("t5_embedding_sha256") != tensor_hashes.get(row["instruction"]):
                errors.append(f"T5 manifest hash mismatch: {row['episode_key']}")
    result = {
        "status": "failed" if errors else "passed",
        "rows": len(rows),
        "unique_keys": len(keys),
        "original": len(original),
        "libero_pro": len(pro),
        "libero_pro_supported": len(supported_pro),
        "unique_instructions": len({row["instruction"] for row in rows}),
        "unchanged_official_environment_variants": sum(
            row["perturbation_category"] == "environment" and not row["variant_applied"]
            for row in pro
        )
        // 6,
        "errors": errors,
    }
    target = EXPERIMENT / "summaries/manifest_validation.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

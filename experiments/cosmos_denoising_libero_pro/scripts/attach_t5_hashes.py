#!/usr/bin/env python3
"""Attach exact T5 tensor hashes to every formal manifest row."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
MANIFESTS = PROJECT / "experiments/cosmos_denoising_libero_pro/manifests"
AUDIT = Path(
    "/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/"
    "cosmos_libero_pro_t5_embeddings.audit.json"
)


def update_jsonl(path: Path, hashes: dict[str, str]) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["t5_embedding_sha256"] = hashes[row["instruction"]]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def update_json(path: Path, hashes: dict[str, str]) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["t5_embedding_sha256"] = hashes[row["instruction"]]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    hashes = {
        instruction: metadata["tensor_sha256"]
        for instruction, metadata in audit["embeddings"].items()
    }
    for name in (
        "original_full.jsonl",
        "libero_pro_full.jsonl",
        "full_sweep.jsonl",
        "libero_pro_full_supported.jsonl",
        "formal_full_supported.jsonl",
        "correctness_original.jsonl",
        "correctness_libero_pro.jsonl",
        "correctness_all.jsonl",
    ):
        update_jsonl(MANIFESTS / name, hashes)
    for name in (
        "libero_40tasks_seed195.json",
        "libero_pro_single_perturbation_seed195.json",
    ):
        update_json(MANIFESTS / name, hashes)
    cache_manifest = {
        "cache_path": audit["output"],
        "cache_sha256": audit["output_sha256"],
        "instruction_count": audit["instruction_count"],
        "exact_coverage": audit["exact_coverage"],
        "instructions": audit["embeddings"],
    }
    (MANIFESTS / "libero_pro_t5_cache_manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"updated": 8, "instruction_count": len(hashes)}, indent=2))


if __name__ == "__main__":
    main()

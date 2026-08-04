#!/usr/bin/env python3
"""Attach exact tensor hashes from the independently expanded stage-2 T5 cache."""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
MANIFESTS = EXPERIMENT / "manifests"
AUDIT = Path(
    "/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/"
    "cosmos_libero_pro_t5_embeddings_stage2.audit.json"
)
NAMES = (
    "stage2_original_seed_repeats.jsonl",
    "stage2_hard_subset.jsonl",
    "stage2_language_all_paraphrases.jsonl",
    "stage2_disagreement_video_replay.jsonl",
    "stage2_t5_union.jsonl",
)


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    hashes = {
        instruction: metadata["tensor_sha256"]
        for instruction, metadata in audit["embeddings"].items()
    }
    counts = {}
    for name in NAMES:
        path = MANIFESTS / name
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        ]
        for row in rows:
            row["t5_embedding_sha256"] = hashes[row["instruction"]]
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        counts[name] = len(rows)
    result = {
        "cache_path": audit["output"],
        "cache_sha256": audit["output_sha256"],
        "instruction_count": audit["instruction_count"],
        "exact_coverage": audit["exact_coverage"],
        "manifests": counts,
    }
    (EXPERIMENT / "summaries/stage2_t5_attachment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

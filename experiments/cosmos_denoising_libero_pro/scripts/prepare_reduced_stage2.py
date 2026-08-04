#!/usr/bin/env python3
"""Create a compact, deterministic stage-2 subset after the broad first round."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
MANIFESTS = EXPERIMENT / "manifests"
SUMMARIES = EXPERIMENT / "summaries"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_stage2 import load_jsonl, write_jsonl

DIRECT_REASONS = {
    "5_success_1_failure",
    "5_success_3_failure",
    "6_success_5_failure",
    "3_success_5_failure",
}
REPRESENTATIVES_PER_CATEGORY = 3
INIT_INDICES = {0, 1, 2}


def stable_order(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    cases = pd.read_csv(SUMMARIES / "paired_disagreement_cases.csv")
    cases["variant_key"] = (
        cases["domain"].astype(str)
        + "|"
        + cases["perturbation_category"].astype(str)
        + "|"
        + cases["task_uid"].astype(str)
    )
    direct = cases[cases["reason"].isin(DIRECT_REASONS)]
    selected_keys = set(direct["variant_key"])
    representative_rows = []
    matched = cases[
        (cases["reason"] == "matched_original_success_pro_failure")
        & (cases["domain"] == "libero_pro")
        & ~cases["variant_key"].isin(selected_keys)
    ]
    for category, group in matched.groupby("perturbation_category"):
        unique = group.drop_duplicates("variant_key").copy()
        unique["_order"] = unique["variant_key"].map(stable_order)
        chosen = unique.sort_values("_order").head(REPRESENTATIVES_PER_CATEGORY)
        selected_keys.update(chosen["variant_key"])
        representative_rows.extend(chosen.to_dict("records"))

    full_hard = load_jsonl(MANIFESTS / "stage2_hard_subset.jsonl")
    reduced_hard = [
        row
        for row in full_hard
        if (
            f"{row['domain']}|{row['perturbation_category']}|{row['task_uid']}"
            in selected_keys
            and int(row["init_state_index"]) in INIT_INDICES
        )
    ]
    write_jsonl(MANIFESTS / "stage2_hard_subset_reduced.jsonl", reduced_hard)

    # Keep all direct denoising disagreements.  For distribution-shift-only
    # cases, retain at most two paired cases per category from selected variants.
    selected_cases = [row for row in direct.to_dict("records")]
    representative_keys = {
        row["variant_key"] for row in representative_rows
    }
    representative_cases = cases[
        (cases["variant_key"].isin(representative_keys))
        & (cases["reason"] == "matched_original_success_pro_failure")
    ].copy()
    representative_cases["_order"] = representative_cases.apply(
        lambda row: stable_order(
            f"{row['variant_key']}|{row['failure_episode_key']}|"
            f"{row['success_episode_key']}"
        ),
        axis=1,
    )
    for _, group in representative_cases.groupby("perturbation_category"):
        selected_cases.extend(group.sort_values("_order").head(2).to_dict("records"))
    replay_first_round_keys = {
        str(value)
        for row in selected_cases
        for value in (row.get("failure_episode_key"), row.get("success_episode_key"))
        if pd.notna(value)
    }
    full_replay = load_jsonl(MANIFESTS / "stage2_disagreement_video_replay.jsonl")
    reduced_replay = [
        row
        for row in full_replay
        if str(row.get("first_round_episode_key")) in replay_first_round_keys
    ]
    write_jsonl(
        MANIFESTS / "stage2_disagreement_video_replay_reduced.jsonl",
        reduced_replay,
    )

    by_category = {}
    for key in selected_keys:
        _, category, _ = key.split("|", 2)
        by_category[category] = by_category.get(category, 0) + 1
    result = {
        "policy": (
            "保留全部 denoising 直接分歧变体；其余 original-success/PRO-fail "
            "按 perturbation 固定哈希最多选 3 个代表。"
        ),
        "selected_steps": [1, 3, 5, 6],
        "seeds": [195, 196, 197],
        "init_state_indices": [0, 1, 2],
        "selected_hard_variants": len(selected_keys),
        "selected_hard_variants_by_category": by_category,
        "hard_new_episode_manifest_rows": len(reduced_hard),
        "reduced_video_replay_rows": len(reduced_replay),
        "original_large_hard_manifest_rows": len(full_hard),
        "original_large_video_manifest_rows": len(full_replay),
        "already_recorded_large_sweep_rows_are_preserved": True,
    }
    (SUMMARIES / "stage2_reduced_plan.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a small task-disjoint manifest for shadow validity labels.

The collection executes only the frozen R2 route.  F1/P1/PV0 candidates are
computed post-control solely as offline labels.  This manifest intentionally
starts at 12 tasks x 2 initial states before any larger collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("reports/server_deep_validation/manifests/full_scale_40_task.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tasks-per-split", type=int, default=4)
    parser.add_argument("--inits-per-task", type=int, default=2)
    args = parser.parse_args()
    if args.tasks_per_split < 1 or args.inits_per_task < 1:
        raise ValueError("tasks and initial states must be positive")

    source_rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line]
    chosen: list[dict] = []
    split_tasks: dict[str, list[str]] = {}
    for split in ("discovery", "validation", "heldout"):
        task_ids = sorted({str(row["task_uid"]) for row in source_rows if row["split"] == split})
        selected = task_ids[: args.tasks_per_split]
        if len(selected) != args.tasks_per_split:
            raise RuntimeError(f"not enough {split} tasks in {args.source}")
        split_tasks[split] = selected
        for task_uid in selected:
            options = sorted(
                (row for row in source_rows if row["split"] == split and row["task_uid"] == task_uid),
                key=lambda row: int(row["init_state_index"]),
            )
            if len(options) < args.inits_per_task:
                raise RuntimeError(f"{task_uid} has too few initial states")
            for original in options[: args.inits_per_task]:
                row = dict(original)
                row["collection"] = "execution_validity_shadow_r2"
                row["episode_key"] = hashlib.sha256(
                    (
                        "execution_validity_shadow_r2|"
                        f"{row['split']}|{row['task_uid']}|{row['init_state_index']}|{row['seed']}"
                    ).encode()
                ).hexdigest()
                chosen.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen), encoding="utf-8"
    )
    write_json(
        args.summary,
        {
            "schema_version": 1,
            "collection": "EXECUTION_VALIDITY_DATASET_small_shadow_r2",
            "source_manifest": str(args.source),
            "route_executed": "pv0_r2",
            "shadow_labels": ["F1", "P1", "PV0"],
            "shadow_labels_runtime_policy_input": False,
            "tasks_per_split": args.tasks_per_split,
            "inits_per_task": args.inits_per_task,
            "episodes": len(chosen),
            "task_disjoint_split_tasks": split_tasks,
            "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "expansion_rule": "Expand only after this task-disjoint 12-task pilot passes contracts and feature analysis.",
        },
    )
    print(json.dumps({"episodes": len(chosen), "split_tasks": split_tasks}, ensure_ascii=False))


if __name__ == "__main__":
    main()

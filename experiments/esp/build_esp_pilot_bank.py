#!/usr/bin/env python3
"""Freeze a small task-disjoint ESP pilot from the existing paired state bank.

This is deliberately a pilot (12 tasks x 4 states), not the protocol's final
300-state evaluation.  It exists to validate all causal intervention paths
before allocating multi-GPU capacity to the formal run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json, read_jsonl


def stable_order(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=Path, default=Path("reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/esp/ESP_PILOT_STATE_BANK.jsonl"))
    parser.add_argument("--task-split", type=Path, default=Path("reports/esp/TASK_SPLIT.json"))
    parser.add_argument("--states-per-task", type=int, default=4)
    args = parser.parse_args()
    if args.states_per_task < 1:
        raise ValueError("states-per-task must be positive")

    rows = read_jsonl(args.state_index)
    by_split_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_split_task[str(row["split"])][str(row["task_uid"])].append(row)
    desired = {"discovery": 4, "validation": 4, "heldout": 4}
    selected_tasks: dict[str, list[str]] = {}
    for split, count in desired.items():
        tasks = sorted(by_split_task[split], key=stable_order)
        if len(tasks) < count:
            raise RuntimeError(f"{split} has only {len(tasks)} tasks")
        selected_tasks[split] = tasks[:count]

    selected: list[dict[str, Any]] = []
    for split, tasks in selected_tasks.items():
        for task in tasks:
            candidates = sorted(by_split_task[split][task], key=lambda row: int(row["global_index"]))
            stride = max(1, len(candidates) // args.states_per_task)
            picks = [candidates[min(i * stride, len(candidates) - 1)] for i in range(args.states_per_task)]
            for row in picks:
                selected.append(dict(row))

    by_split_task_selected: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        by_split_task_selected[row["split"]][row["task_uid"]].append(row)
    for row in selected:
        tasks = selected_tasks[row["split"]]
        position = tasks.index(row["task_uid"])
        other_task = tasks[(position + 1) % len(tasks)]
        peer = by_split_task_selected[row["split"]][other_task][int(row["global_index"]) % args.states_per_task]
        row["shuffle_state_key"] = peer["state_key"]
        row["shuffle_task_uid"] = peer["task_uid"]
        if row["shuffle_task_uid"] == row["task_uid"]:
            raise RuntimeError("shuffle task is not task-disjoint")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    atomic_write_json(args.task_split, {
        "schema_version": 1,
        "purpose": "ESP mechanism pilot; fixed before pilot execution",
        "status": "PILOT_NOT_FORMAL_300_STATE_EVALUATION",
        "source_state_bank": str(args.state_index),
        "states_per_task": args.states_per_task,
        "splits": selected_tasks,
        "counts": {split: len(tasks) for split, tasks in selected_tasks.items()},
        "total_tasks": sum(len(tasks) for tasks in selected_tasks.values()),
        "total_states": len(selected),
        "shuffle": "same split, different selected task; condition is a strong-intervention sanity control only",
    })
    print(json.dumps({"tasks": sum(map(len, selected_tasks.values())), "states": len(selected), "output": str(args.output)}))


if __name__ == "__main__":
    main()

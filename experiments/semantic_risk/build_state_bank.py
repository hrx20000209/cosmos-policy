#!/usr/bin/env python3
"""Freeze the task-disjoint 16-task semantic-risk state bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json, read_jsonl


def ordered(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl"))
    parser.add_argument("--bank", type=Path, default=Path("reports/semantic_risk/SEMANTIC_RISK_STATE_BANK.jsonl"))
    parser.add_argument("--split", type=Path, default=Path("reports/semantic_risk/TASK_SPLIT.json"))
    parser.add_argument("--states-per-task", type=int, default=32)
    args = parser.parse_args()
    task_counts = {"discovery": 6, "validation": 5, "heldout": 5}
    rows = read_jsonl(args.source)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["split"])][str(row["task_uid"])].append(row)
    chosen: dict[str, list[str]] = {}
    selected: list[dict[str, Any]] = []
    for split, n_tasks in task_counts.items():
        candidates = sorted(grouped[split], key=ordered)
        if len(candidates) < n_tasks:
            raise RuntimeError(f"only {len(candidates)} tasks available for {split}")
        chosen[split] = candidates[:n_tasks]
        for task in chosen[split]:
            states = sorted(grouped[split][task], key=lambda row: int(row["global_index"]))
            if len(states) < args.states_per_task:
                raise RuntimeError(f"{task} has {len(states)} states, needs {args.states_per_task}")
            # Uniform time coverage, unique indices, deterministic before inference.
            indices = [round(i * (len(states) - 1) / (args.states_per_task - 1)) for i in range(args.states_per_task)]
            if len(set(indices)) != args.states_per_task:
                raise RuntimeError("state sampling produced duplicate indices")
            selected.extend(dict(states[index]) for index in indices)
    args.bank.parent.mkdir(parents=True, exist_ok=True)
    args.bank.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    atomic_write_json(args.split, {
        "schema_version": 1,
        "purpose": "Semantic-risk E4/E5/E6 fixed task-disjoint state bank",
        "source_state_bank": str(args.source),
        "task_counts": task_counts,
        "states_per_task": args.states_per_task,
        "total_tasks": sum(task_counts.values()),
        "total_states": len(selected),
        "splits": chosen,
        "heldout_policy": "Heldout task IDs are fixed before feature/model selection and are not used for tuning.",
    })
    print(json.dumps({"tasks": sum(task_counts.values()), "states": len(selected), "bank": str(args.bank)}))


if __name__ == "__main__":
    main()

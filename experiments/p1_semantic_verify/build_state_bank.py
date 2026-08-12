#!/usr/bin/env python3
"""Build the E12 paired-state bank for one E11 split.

Selection is deterministic and independent of any route outcome: round-robin
over the stored episodes of a task, taking within each episode the control
steps that are maximally spread, so a task's 16 states are not a stack of
adjacent requests from one rollout.  Every selected row is already H=16
aligned by ``build_state_index``; ``load_request`` re-asserts it at load time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json, read_jsonl

STATE_INDEX = Path("reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl")
TASK_SPLIT = Path("reports/p1_semantic_verify/TASK_SPLIT_E12.json")


def spread(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Take ``count`` items with maximal spacing along control_step."""

    ordered = sorted(items, key=lambda row: int(row["control_step"]))
    if count >= len(ordered):
        return ordered
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)] if count > 1 else [0]
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for position in positions:
        while position in used:
            position = (position + 1) % len(ordered)
        used.add(position)
        picked.append(ordered[position])
    return sorted(picked, key=lambda row: int(row["control_step"]))


def select_for_task(rows: list[dict[str, Any]], per_task: int) -> list[dict[str, Any]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_key"]].append(row)
    episodes = sorted(by_episode, key=lambda key: hashlib.sha256(f"E12:{key}".encode()).hexdigest())
    quota = {key: per_task // len(episodes) for key in episodes}
    for index in range(per_task - sum(quota.values())):
        quota[episodes[index % len(episodes)]] += 1
    selected: list[dict[str, Any]] = []
    shortfall = 0
    for key in episodes:
        take = spread(by_episode[key], quota[key])
        shortfall += quota[key] - len(take)
        selected.extend(take)
    # Redistribute any shortfall (short episodes) onto episodes that still have room.
    while shortfall > 0:
        progressed = False
        for key in episodes:
            remaining = [row for row in by_episode[key] if row["state_key"] not in {s["state_key"] for s in selected}]
            if not remaining:
                continue
            selected.append(spread(remaining, 1)[0])
            shortfall -= 1
            progressed = True
            if shortfall == 0:
                break
        if not progressed:
            break
    return sorted(selected, key=lambda row: (row["episode_key"], int(row["control_step"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("discovery", "validation", "heldout"))
    parser.add_argument("--per-task", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = json.loads(TASK_SPLIT.read_text(encoding="utf-8"))["splits"][args.split]
    source = read_jsonl(STATE_INDEX)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        if row["task_uid"] in tasks:
            by_task[row["task_uid"]].append(row)

    entries: list[dict[str, Any]] = []
    per_task_audit: dict[str, Any] = {}
    for task in tasks:
        rows = by_task.get(task, [])
        if not rows:
            raise RuntimeError(f"no aligned states available for {task}")
        picked = select_for_task(rows, args.per_task)
        steps_by_episode: dict[str, list[int]] = defaultdict(list)
        for row in picked:
            steps_by_episode[row["episode_key"]].append(int(row["control_step"]))
        gaps = [
            b - a
            for steps in steps_by_episode.values()
            for a, b in zip(sorted(steps), sorted(steps)[1:])
        ]
        per_task_audit[task] = {
            "available": len(rows),
            "selected": len(picked),
            "episodes_used": len(steps_by_episode),
            "min_within_episode_step_gap": min(gaps) if gaps else None,
            "adjacent_request_pairs": sum(gap == 16 for gap in gaps),
        }
        for row in picked:
            entries.append({**row, "e12_split": args.split})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(args.output)

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "split": args.split,
        "tasks": len(tasks),
        "states": len(entries),
        "per_task_target": args.per_task,
        "temporal_gap_actions": 16,
        "selection_rule": "round-robin over sha256-ordered episodes; maximal control-step spread within episode",
        "state_index": str(STATE_INDEX),
        "per_task": per_task_audit,
        "bank": str(args.output),
    }
    atomic_write_json(args.output.with_name(f"{args.output.stem}_audit.json"), audit)
    print(json.dumps({"split": args.split, "states": len(entries), "tasks": len(tasks)}))


if __name__ == "__main__":
    main()

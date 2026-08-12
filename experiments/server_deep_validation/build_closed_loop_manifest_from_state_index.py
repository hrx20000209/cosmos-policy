#!/usr/bin/env python3
"""Recover the 200 unique closed-loop scenarios from the frozen state index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: dict[str, dict] = {}
    stable = ("task_uid", "task_name", "suite", "split", "instruction", "init_state_index", "seed", "bddl_path", "init_path", "libero_repo", "libero_config_path")
    for line in args.state_index.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        key = str(raw["episode_key"])
        candidate = {name: raw[name] for name in stable}
        candidate["episode_key"] = key
        candidate["max_steps"] = MAX_STEPS[candidate["suite"]]
        candidate["denoising_steps"] = 1
        if key in rows and any(rows[key][name] != candidate[name] for name in candidate):
            raise RuntimeError(f"inconsistent state rows for {key}")
        rows[key] = candidate
    ordered = sorted(rows.values(), key=lambda row: (row["split"], row["task_uid"], row["init_state_index"]))
    if len(ordered) != 200 or len({row["task_uid"] for row in ordered}) != 40:
        raise RuntimeError(f"expected 200 scenarios / 40 tasks, got {len(ordered)} / {len({row['task_uid'] for row in ordered})}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")
    print(json.dumps({"status": "PASS", "episodes": len(ordered), "tasks": len({row['task_uid'] for row in ordered}), "output": str(args.output)}))


if __name__ == "__main__":
    main()

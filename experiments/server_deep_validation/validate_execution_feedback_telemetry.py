#!/usr/bin/env python3
"""Validate runtime-only execution feedback emitted by the PV0 episode runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FORBIDDEN = {"object", "contact", "reward", "success", "predicate", "subgoal", "task_stage"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.episode.read_text(encoding="utf-8"))
    if raw.get("status") != "PASS":
        raise SystemExit("episode is not PASS")
    contract = raw.get("execution_feedback_contract", {})
    if contract.get("runtime_observables_only") is not True:
        raise SystemExit("missing runtime-only contract")
    feedback = raw.get("execution_feedback", [])
    if not feedback:
        raise SystemExit("no execution feedback samples")
    for index, sample in enumerate(feedback):
        blob = json.dumps(sample).lower()
        if any(token in blob for token in FORBIDDEN):
            raise SystemExit(f"forbidden telemetry field at sample {index}")
        for point in (sample.get("before"), sample.get("after")):
            if set(point or {}) != {"eef_pos", "eef_quat", "gripper_qpos"}:
                raise SystemExit(f"unexpected proprio schema at sample {index}")
            if np.asarray(point["eef_pos"]).shape != (3,) or np.asarray(point["eef_quat"]).shape != (4,) or np.asarray(point["gripper_qpos"]).shape != (2,):
                raise SystemExit(f"bad proprio shape at sample {index}")
        if np.asarray(sample.get("planned_action")).shape != (7,) or np.asarray(sample.get("executed_action")).shape != (7,):
            raise SystemExit(f"bad action shape at sample {index}")
        if sample.get("arm_command_changed_by_interruption"):
            if not np.allclose(sample["planned_action"][6], sample["executed_action"][6]):
                raise SystemExit(f"interruption changed gripper at sample {index}")
    print(json.dumps({"status": "PASS", "samples": len(feedback), "episode": str(args.episode)}))


if __name__ == "__main__":
    main()

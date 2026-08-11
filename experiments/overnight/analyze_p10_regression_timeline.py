#!/usr/bin/env python3
"""Extract a non-interventional action timeline for the known P3 regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TARGET = "open_the_top_drawer_and_put_the_bowl_inside"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.shard:
        rows.extend(json.loads(path.read_text())["records"])
    selected = {
        row["configuration"]: row
        for row in rows
        if row["task_name"] == TARGET and int(row["init_state_index"]) == 0
        and row["configuration"] in {"fresh", "predict_correct"}
    }
    if set(selected) != {"fresh", "predict_correct"}:
        raise RuntimeError(f"missing paired regression rows: {selected.keys()}")
    fresh = np.load(selected["fresh"]["action_chunks_path"])
    corrected = np.load(selected["predict_correct"]["action_chunks_path"])
    n = min(len(fresh), len(corrected))
    timeline = []
    for index in range(n):
        delta = fresh[index] - corrected[index]
        timeline.append({
            "request_index": index,
            "nominal_control_step": index * 16,
            "fresh_chunk_mean_step_l2": float(np.mean(np.linalg.norm(fresh[index], axis=-1))),
            "corrected_chunk_mean_step_l2": float(np.mean(np.linalg.norm(corrected[index], axis=-1))),
            "fresh_vs_corrected_chunk_mean_step_l2": float(np.mean(np.linalg.norm(delta, axis=-1))),
            "fresh_vs_corrected_chunk_first_step_l2": float(np.linalg.norm(delta[0])),
            "gripper_disagreement": float(np.mean((fresh[index, :, -1] >= 0) != (corrected[index, :, -1] >= 0))),
        })
    output = {
        "schema_version": "p10_regression_timeline_v1",
        "task_name": TARGET,
        "init_state_index": 0,
        "fresh": {k: selected["fresh"].get(k) for k in ("success", "episode_steps", "inference_count", "termination_reason", "action_chunks_path")},
        "predict_correct": {k: selected["predict_correct"].get(k) for k in ("success", "episode_steps", "inference_count", "termination_reason", "action_chunks_path")},
        "timeline": timeline,
        "caveat": "Action chunks are aligned by request index; after the first action divergence, observations are not identical. This is diagnosis only and does not alter the method.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "requests": n}))


if __name__ == "__main__":
    main()

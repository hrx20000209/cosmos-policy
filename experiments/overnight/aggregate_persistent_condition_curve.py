#!/usr/bin/env python3
"""Normalize the existing fixed-step persistent-condition curve.

The source artifact contains a predicted baseline and persistent fresh-condition
arrival points.  This script only relabels those points as PPPP/PPPF/PPFF/PFFF/
FFFF for the system-level report; it does not run or modify inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    labels = {0: "FFFF", 1: "PFFF", 2: "PPFF", 3: "PPPF"}
    rows: list[dict[str, Any]] = []
    raw_path = Path(source.get("raw_output", ""))
    raw_states: list[dict[str, Any]] = []
    if raw_path.exists():
        raw_states = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    for state in raw_states:
        for correction in state.get("corrections", []):
            if int(state.get("denoising_steps", 0)) != 4:
                continue
            if "arrival_forward_index" not in correction:
                continue
            arrival = int(correction["arrival_forward_index"])
            rows.append({
                "state_index": state.get("state_index"),
                "split": state.get("split"),
                "task_suite": state.get("task_suite"),
                "task_id": state.get("task_id"),
                "arrival_forward_index": arrival,
                "schedule": "P" * arrival + "F" * (4 - arrival),
                "action_recovery": correction.get("action_recovery"),
                "future_recovery": correction.get("future_recovery"),
                "latency_ms": correction.get("latency_ms"),
            })
    # Some versions of the compact artifact do not expose per-state d4 lists;
    # preserve the already computed aggregate in that case.
    aggregate: dict[str, Any] = {}
    for arrival in range(4):
        key = f"d4_arrival{arrival}_all"
        if key in source.get("summary", {}):
            item = source["summary"][key]
            aggregate[labels[arrival]] = {
                "n": item.get("n"),
                "median_action_recovery": item.get("median_action_recovery"),
                "median_future_recovery": item.get("median_future_recovery"),
                "median_latency_ms": item.get("median_latency_ms"),
                "arrival_forward_index": arrival,
            }
    aggregate["PPPP"] = {
        "n": int(source.get("states", 0)) if isinstance(source.get("states", 0), int) else len(source.get("states", [])),
        "median_action_recovery": 0.0,
        "median_future_recovery": 0.0,
        "median_latency_ms": None,
        "arrival_forward_index": None,
        "interpretation": "predicted-only baseline; no fresh condition assimilation",
    }
    output = {
        "schema_version": "persistent_condition_curve_v1",
        "experiment": "fixed_four_forward_feedback_assimilation_curve",
        "source": str(args.input),
        "checkpoint": source.get("checkpoint"),
        "checkpoint_sha256": source.get("checkpoint_sha256"),
        "value_used": source.get("value_used", False),
        "labels": ["PPPP", "PPPF", "PPFF", "PFFF", "FFFF"],
        "aggregate": aggregate,
        "per_state_rows": rows,
        "caveat": "PPPP is the predicted-only reference; FFFF is the fully fresh reference.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

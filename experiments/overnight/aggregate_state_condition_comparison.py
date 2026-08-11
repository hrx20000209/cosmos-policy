#!/usr/bin/env python3
"""Join the same-state condition curve and oracle state-repair artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("task_suite"), row.get("task_id"), row.get("episode_index"), row.get("control_step"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair2d", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repairs = [json.loads(line) for line in args.repair2d.read_text().splitlines() if line.strip()]
    persistent = [json.loads(line) for line in args.persistent.read_text().splitlines() if line.strip()]
    persistent_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in persistent:
        if int(row.get("denoising_steps", 0)) == 4:
            persistent_by_key[key(row)] = row
    records: list[dict[str, Any]] = []
    for repair in repairs:
        match = persistent_by_key.get(key(repair))
        if match is None:
            continue
        hidden = repair.get("hidden_repairs", [])
        diffusion = repair.get("diffusion_repairs", [])
        def find(items: list[dict[str, Any]], **wanted: Any) -> dict[str, Any] | None:
            return next((item for item in items if all(item.get(k) == v for k, v in wanted.items())), None)
        records.append({
            "state_key": key(repair),
            "split": repair.get("split"),
            "task_suite": repair.get("task_suite"),
            "task_id": repair.get("task_id"),
            "control_step": repair.get("control_step"),
            "baseline_predicted_to_fresh_action": repair.get("baseline_predicted_to_fresh"),
            "baseline_predicted_to_fresh_future": repair.get("baseline_future_predicted_to_fresh"),
            "persistent_condition": {
                "PPPF": next((c for c in match.get("corrections", []) if int(c.get("arrival_forward_index", -1)) == 3), None),
                "PPFF": next((c for c in match.get("corrections", []) if int(c.get("arrival_forward_index", -1)) == 2), None),
                "PFFF": next((c for c in match.get("corrections", []) if int(c.get("arrival_forward_index", -1)) == 1), None),
                "FFFF": next((c for c in match.get("corrections", []) if int(c.get("arrival_forward_index", -1)) == 0), None),
            },
            "oracle_state_repair": {
                "hidden_stage4_block8_current_visual": find(hidden, stage=4, block=8, group="current_visual"),
                "hidden_stage4_block8_all_dynamic_nonvalue": find(hidden, stage=4, block=8, group="all_dynamic_nonvalue"),
                "hidden_stage4_block16_current_visual": find(hidden, stage=4, block=16, group="current_visual"),
                "diffusion_stage4_future_visual": find(diffusion, stage=4, group="future_visual"),
                "diffusion_stage4_action": find(diffusion, stage=4, group="action"),
                "diffusion_stage4_future_plus_action": find(diffusion, stage=4, group="future_plus_action"),
            },
        })
    output = {
        "schema_version": "state_vs_condition_comparison_v1",
        "experiment": "same_state_condition_switch_vs_oracle_state_repair",
        "repair2d_source": str(args.repair2d),
        "persistent_source": str(args.persistent),
        "matched_states": len(records),
        "records": records,
        "unavailable_condition": "x_s_state_patch was not present in the existing repair2d artifact and was not inferred from x0 patch results",
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "matched_states": len(records)}))


if __name__ == "__main__":
    main()

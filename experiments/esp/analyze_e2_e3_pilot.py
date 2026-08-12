#!/usr/bin/env python3
"""Independent, preregistered summary for ESP causal/probe pilot shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json


def task_balanced_spearman(frame: pd.DataFrame, x: str, y: str) -> tuple[float | None, dict[str, float | None]]:
    values: dict[str, float | None] = {}
    for task, group in frame.groupby("task_uid"):
        if group[x].nunique() < 2 or group[y].nunique() < 2:
            values[str(task)] = None
        else:
            values[str(task)] = float(spearmanr(group[x], group[y]).statistic)
    valid = [value for value in values.values() if value is not None and np.isfinite(value)]
    return (float(np.mean(valid)) if valid else None), values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, default=Path("reports/esp/shards/pilot_offline"))
    parser.add_argument("--e2-output", type=Path, default=Path("reports/esp/E2_PILOT_RESULT.json"))
    parser.add_argument("--e3-output", type=Path, default=Path("reports/esp/E3_PILOT_RESULT.json"))
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in sorted(args.shard_dir.glob("SHARD_*.json"))]
    if not payloads or any(payload.get("status") != "PASS" for payload in payloads):
        raise RuntimeError("all pilot shards must be PASS before analysis")
    states = [row for payload in payloads for row in payload["completed"]]
    if len({row["state_id"] for row in states}) != len(states):
        raise RuntimeError("duplicate state IDs across shards")
    e2_rows, e3_rows = [], []
    for state in states:
        for camera in state["camera"]:
            for name, intervention in camera["e2"].items():
                e2_rows.append({
                    "state_id": state["state_id"], "task_uid": state["task_uid"], "split": state["split"],
                    "camera_name": camera["camera_name"], "intervention": name,
                    "mean_step_l2": intervention["action_distance_to_f1"]["pair"]["mean_step_l2"],
                    "first_action_l2": intervention["action_distance_to_f1"]["pair"]["first_action_l2"],
                    "scope_ok": intervention["scope"]["only_target_slot_changed"],
                })
            for k, probe in camera["e3"]["probes"].items():
                e3_rows.append({
                    "state_id": state["state_id"], "task_uid": state["task_uid"], "split": state["split"],
                    "camera_name": camera["camera_name"], "k": int(k),
                    "delta_hidden_primary_rms": probe["delta_hidden_primary_rms"],
                    "early_equivalence_rms": probe["e0_full_forward_same_block_rms"],
                    "target_imag": camera["e2"]["imagined"]["action_distance_to_f1"]["pair"]["mean_step_l2"],
                    "scope_ok": camera["e3"]["scope"]["only_target_slot_changed"],
                })
    e2 = pd.DataFrame(e2_rows)
    e3 = pd.DataFrame(e3_rows)
    Path("artifacts/esp").mkdir(parents=True, exist_ok=True)
    e2.to_parquet("artifacts/esp/e2_causal_camera_pilot.parquet", index=False)
    e3.to_parquet("artifacts/esp/e3_probe_pilot.parquet", index=False)
    primary = e2[e2.intervention == "imagined"].copy()
    top = primary.sort_values(["state_id", "mean_step_l2", "camera_name"], ascending=[True, False, True]).groupby("state_id").first().reset_index()
    top_by_task = {task: Counter(group.camera_name).most_common(1)[0][0] for task, group in top.groupby("task_uid")}
    switching = {
        task: float(1.0 - group.camera_name.value_counts(normalize=True).max())
        for task, group in top.groupby("task_uid")
    }
    e2_summary = {
        "status": "PILOT_ONLY_NOT_FORMAL_GATE", "states": len(states), "tasks": int(primary.task_uid.nunique()),
        "rows": len(e2), "all_scope_checks_pass": bool(e2.scope_ok.all()),
        "per_camera_intervention_mean_step_l2": {
            f"{camera}:{intervention}": float(group.mean_step_l2.mean())
            for (camera, intervention), group in e2.groupby(["camera_name", "intervention"])
        },
        "within_task_switching_fraction": switching, "mean_within_task_switching_fraction": float(np.mean(list(switching.values()))),
        "task_modal_camera": top_by_task,
        "decision": "E2_PILOT_SIGNAL_ONLY; protocol requires >=300 states for formal E2 gate",
    }
    depth = {}
    for k, group in e3.groupby("k"):
        summary, per_task = task_balanced_spearman(group, "delta_hidden_primary_rms", "target_imag")
        depth[str(k)] = {"task_balanced_spearman": summary, "per_task": per_task,
                         "pair_spearman": float(spearmanr(group.delta_hidden_primary_rms, group.target_imag).statistic),
                         "mean_delta": float(group.delta_hidden_primary_rms.mean())}
    e3_summary = {
        "status": "PILOT_ONLY_NOT_FORMAL_GATE", "states": len(states), "tasks": int(e3.task_uid.nunique()), "rows": len(e3),
        "all_scope_checks_pass": bool(e3.scope_ok.all()), "max_early_equivalence_rms": float(e3.early_equivalence_rms.max()),
        "depth": depth,
        "decision": "E3_PILOT_SIGNAL_ONLY; no heldout selection or GO/NO-GO claim from 48 states",
    }
    args.e2_output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.e2_output, e2_summary)
    atomic_write_json(args.e3_output, e3_summary)
    print(json.dumps({"states": len(states), "e2": str(args.e2_output), "e3": str(args.e3_output)}))


if __name__ == "__main__":
    main()

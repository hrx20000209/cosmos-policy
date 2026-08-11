#!/usr/bin/env python3
"""Aggregate the compute-matched persistent-condition ablation shards.

This is deliberately an analysis-only script.  It does not alter model
outputs, introduce a scheduler, or use privileged simulator state.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


CONDITIONS = ("fresh_1", "predicted_1", "predicted_predicted", "predicted_fresh", "fresh_fresh")


def percentile(values: list[float], q: float) -> float | None:
    values = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def load_records(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        manifests.append({k: payload.get(k) for k in (
            "schema_version", "checkpoint", "checkpoint_sha256", "checkpoint_policy",
            "value_used", "privileged_state_runtime_input", "adaptive_scheduler_used",
            "threshold_used", "finetuning_used", "state_source", "shard_index", "shard_count",
        )})
        for local_index, row in enumerate(payload["records"]):
            # Older shards intentionally left global_state_index unset; retain
            # the shard/local provenance so that duplicate checks are robust.
            row = dict(row)
            row["_source_shard"] = payload.get("shard_index")
            row["_source_record_index"] = local_index
            records.append(row)
    records.sort(key=lambda row: (row.get("_source_shard", -1), row.get("_source_record_index", -1)))
    return records, manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records, manifests = load_records([Path(p) for p in args.input])
    if not records:
        raise RuntimeError("No records found")
    indices = [(r.get("_source_shard"), r.get("_source_record_index")) for r in records]
    if len(set(indices)) != len(indices):
        raise RuntimeError("Duplicate source record in ablation shards")

    condition_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    task_metrics: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    transition_metrics: dict[str, list[float]] = defaultdict(list)
    pairwise: dict[str, list[float]] = defaultdict(list)
    for row in records:
        # task_id is local to each suite split; include suite so the
        # discovery/validation/heldout task-disjointness is preserved.
        task = f"{row.get('task_suite', 'unknown')}::{row.get('task_id', 'unknown')}"
        conditions = row["conditions"]
        for condition in CONDITIONS:
            item = conditions[condition]
            for metric_name, field in (
                ("action_error_to_fresh_fresh", "action_error_to_fresh_fresh"),
                ("future_error_to_fresh_fresh", "future_error_to_fresh_fresh"),
                ("gripper_disagreement", "gripper_disagreement_to_fresh_fresh"),
            ):
                value = float(item[field])
                condition_metrics[condition][metric_name].append(value)
                task_metrics[task][condition][metric_name].append(value)
            metrics = item.get("metrics", {})
            for metric_name, field in (
                ("wall_latency_ms", "wall_latency_ms"),
                ("gpu_work_ms", "model_generate_inclusive_ms"),
            ):
                value = float(metrics[field])
                condition_metrics[condition][metric_name].append(value)
                task_metrics[task][condition][metric_name].append(value)

        pp = conditions["predicted_predicted"]
        pf = conditions["predicted_fresh"]
        transition_metrics["action_error_reduction_pp_to_pf"].append(
            float(pp["action_error_to_fresh_fresh"]) - float(pf["action_error_to_fresh_fresh"])
        )
        transition_metrics["future_error_reduction_pp_to_pf"].append(
            float(pp["future_error_to_fresh_fresh"]) - float(pf["future_error_to_fresh_fresh"])
        )
        transition_metrics["action_recovery_pf"].append(
            1.0 - float(pf["action_error_to_fresh_fresh"]) / max(float(pp["action_error_to_fresh_fresh"]), 1e-8)
        )
        transition_metrics["future_recovery_pf"].append(
            1.0 - float(pf["future_error_to_fresh_fresh"]) / max(float(pp["future_error_to_fresh_fresh"]), 1e-8)
        )
        transition_metrics["gripper_disagreement_reduction_pp_to_pf"].append(
            1.0 - float(pf["gripper_disagreement_to_fresh_fresh"])
            / max(float(pp["gripper_disagreement_to_fresh_fresh"]), 1e-8)
        )
        pairwise["predicted_1_vs_predicted_predicted_action_delta"].append(
            float(conditions["predicted_1"]["action_error_to_fresh_fresh"])
            - float(pp["action_error_to_fresh_fresh"])
        )
        pairwise["predicted_predicted_vs_predicted_fresh_action_delta"].append(
            float(pp["action_error_to_fresh_fresh"])
            - float(pf["action_error_to_fresh_fresh"])
        )

    per_task: dict[str, Any] = {}
    for task, by_condition in sorted(task_metrics.items()):
        per_task[task] = {
            condition: {metric: stats(values) for metric, values in metrics.items()}
            for condition, metrics in sorted(by_condition.items())
        }

    summary = {
        condition: {metric: stats(values) for metric, values in metrics.items()}
        for condition, metrics in condition_metrics.items()
    }
    output = {
        "schema_version": "feedback_assimilation_ablation_aggregate_v1",
        "experiment": "compute_matched_feedback_assimilation_ablation",
        "records": len(records),
        "unique_tasks": len(per_task),
        "task_ids": sorted(per_task),
        "conditions": list(CONDITIONS),
        "reference": "fresh_fresh",
        "summary": summary,
        "transition_summary": {key: stats(values) for key, values in transition_metrics.items()},
        "pairwise_summary": {key: stats(values) for key, values in pairwise.items()},
        "per_task": per_task,
        "provenance": {
            "shards": manifests,
            "value_used": False,
            "privileged_state_runtime_input": False,
            "adaptive_scheduler_used": False,
            "threshold_used": False,
            "finetuning_used": False,
            "interpretation": (
                "Predicted→Fresh and Predicted→Predicted both use two denoiser forwards; "
                "the difference is the persistent current visual condition on the second forward."
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "records": len(records), "unique_tasks": len(per_task)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

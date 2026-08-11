#!/usr/bin/env python3
"""Merge P3 shards and compute paired closed-loop statistics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def transition(rows: dict[tuple[str, int, int], dict[str, dict]]) -> dict[str, int | float]:
    counts = Counter()
    for pair in rows.values():
        fresh = bool(pair["fresh"]["success"])
        other = bool(pair["other"]["success"])
        if fresh and other:
            counts["fresh_success_other_success"] += 1
        elif fresh and not other:
            counts["fresh_success_other_failure"] += 1
        elif not fresh and other:
            counts["fresh_failure_other_success"] += 1
        else:
            counts["both_failure"] += 1
    fresh_successes = counts["fresh_success_other_success"] + counts["fresh_success_other_failure"]
    return {
        **{key: counts[key] for key in (
            "fresh_success_other_success",
            "fresh_success_other_failure",
            "fresh_failure_other_success",
            "both_failure",
        )},
        "n": len(rows),
        "fresh_successes": fresh_successes,
        "regression_rate_on_fresh_success": (
            counts["fresh_success_other_failure"] / fresh_successes
            if fresh_successes else 0.0
        ),
        "success_gain_rate_on_fresh_failure": (
            counts["fresh_failure_other_success"]
            / (counts["fresh_failure_other_success"] + counts["both_failure"])
            if counts["fresh_failure_other_success"] + counts["both_failure"] else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.shard]
    records = [record for payload in payloads for record in payload["records"]]
    records.sort(key=lambda record: (
        record["task_name"], record["init_state_index"], record["seed"], record["configuration"]
    ))

    configs = sorted({record["configuration"] for record in records})
    by_config = {config: [record for record in records if record["configuration"] == config] for config in configs}
    outcomes = {}
    for config, config_records in by_config.items():
        by_task = defaultdict(lambda: {"episodes": 0, "successes": 0})
        for record in config_records:
            by_task[record["task_name"]]["episodes"] += 1
            by_task[record["task_name"]]["successes"] += int(record["success"])
        outcomes[config] = {
            "episodes": len(config_records),
            "successes": sum(bool(record["success"]) for record in config_records),
            "success_rate": sum(bool(record["success"]) for record in config_records) / len(config_records),
            "termination_reasons": dict(Counter(record["termination_reason"] for record in config_records)),
            "by_task": dict(sorted(by_task.items())),
        }

    pair_index: dict[tuple[str, int, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        key = (record["task_name"], record["init_state_index"], record["seed"])
        pair_index[key][record["configuration"]] = record
    expected_configs = {"fresh", "predicted_reuse", "predict_correct"}
    complete_pairs = {
        key: value for key, value in pair_index.items() if expected_configs.issubset(value)
    }
    fresh_vs_simple = {
        key: {"fresh": value["fresh"], "other": value["predicted_reuse"]}
        for key, value in complete_pairs.items()
    }
    fresh_vs_pc = {
        key: {"fresh": value["fresh"], "other": value["predict_correct"]}
        for key, value in complete_pairs.items()
    }
    simple_vs_pc = {
        key: {"fresh": value["predicted_reuse"], "other": value["predict_correct"]}
        for key, value in complete_pairs.items()
    }

    paired_by_task = {}
    for task in sorted({key[0] for key in complete_pairs}):
        task_pairs = {key: value for key, value in complete_pairs.items() if key[0] == task}
        paired_by_task[task] = {
            "episodes": len(task_pairs),
            "successes": {
                config: sum(bool(value[config]["success"]) for value in task_pairs.values())
                for config in sorted(expected_configs)
            },
            "fresh_vs_predicted_reuse": transition({
                key: {"fresh": value["fresh"], "other": value["predicted_reuse"]}
                for key, value in task_pairs.items()
            }),
            "fresh_vs_predict_correct": transition({
                key: {"fresh": value["fresh"], "other": value["predict_correct"]}
                for key, value in task_pairs.items()
            }),
        }

    raw_rows = []
    raw_paths = []
    for payload in payloads:
        raw_path = payload.get("raw_trace_path")
        if raw_path:
            raw_paths.append(raw_path)
            path = Path(raw_path)
            if path.exists():
                with path.open() as handle:
                    raw_rows.extend(json.loads(line) for line in handle if line.strip())

    trace_latency = {}
    for config in configs:
        rows = [row for row in raw_rows if row.get("configuration") == config]
        trace_latency[config] = {
            "request_count": len(rows),
            "total_policy_request_latency_ms": distribution([
                float(row["total_policy_request_latency_ms"]) for row in rows
            ]),
            "dit_denoising_latency_ms": distribution([
                float(row["dit_denoising_latency_ms"]) for row in rows
            ]),
            "vae_encoding_latency_ms": distribution([
                float(row["vae_encoding_latency_ms"]) for row in rows
            ]),
        }

    base = payloads[0]
    output = {
        "schema_version": "p3-aggregate-v1",
        "experiment": "predict_correct_p3_closed_loop_aggregate",
        "source_shards": [str(path) for path in args.shard],
        "checkpoint": base["checkpoint"],
        "checkpoint_sha256": base["checkpoint_sha256"],
        "checkpoint_policy": base["checkpoint_policy"],
        "configurations": configs,
        "task_count": len({record["task_name"] for record in records}),
        "episode_count": len(records),
        "complete_paired_episode_count": len(complete_pairs),
        "initial_states_per_task": 3,
        "denoising_steps": base["denoising_steps"],
        "action_horizon": base["action_horizon"],
        "value_used": base["value_used"],
        "privileged_state_runtime_input": base["privileged_state_runtime_input"],
        "adaptive_scheduler_used": base["adaptive_scheduler_used"],
        "threshold_used": base["threshold_used"],
        "finetuning_used": base["finetuning_used"],
        "outcomes": outcomes,
        "paired_transitions": {
            "fresh_vs_predicted_reuse": transition(fresh_vs_simple),
            "fresh_vs_predict_correct": transition(fresh_vs_pc),
            "predicted_reuse_vs_predict_correct": transition(simple_vs_pc),
        },
        "paired_by_task": paired_by_task,
        "trace_latency": trace_latency,
        "raw_trace_paths": raw_paths,
        "record_count_by_shard": [len(payload["records"]) for payload in payloads],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "episodes": len(records),
        "complete_pairs": len(complete_pairs),
        "outcomes": outcomes,
        "paired_transitions": output["paired_transitions"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

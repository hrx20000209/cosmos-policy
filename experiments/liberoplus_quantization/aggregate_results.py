#!/usr/bin/env python3
"""Aggregate completed JSONL episodes without inventing missing results."""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

EXP = Path(__file__).resolve().parent
RAW = EXP / "raw"
PROFILES = EXP / "profiles"
OUT = EXP / "summaries"

EPISODE_COLUMNS = [
    "experiment_name",
    "config_hash",
    "task_suite",
    "task_id",
    "task_name",
    "perturbation_category",
    "difficulty",
    "seed",
    "episode_index",
    "initial_state_index",
    "initial_state_sha256",
    "success",
    "termination_reason",
    "environment_steps",
    "policy_calls",
    "episode_total_ms",
    "policy_inference_total_ms",
    "environment_total_ms",
    "replanning_overhead_ratio",
    "peak_gpu_memory_mb",
    "peak_process_memory_mb",
    "stale_action_ratio",
    "action_discontinuity_l1_mean",
    "action_discontinuity_l2_mean",
    "observation_staleness_steps_mean",
    "observation_staleness_ms_mean",
]


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    probability = successes / total
    denominator = 1 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1 - probability) / total + z**2 / (4 * total**2))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def read_jsonl(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in glob.glob(pattern):
        with open(name) as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL {name}:{line_number}: {error}") from error
    return rows


def metadata_by_hash() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for name in PROFILES.glob("run_*.json"):
        payload = json.loads(name.read_text())
        metadata[payload["config_hash"]] = payload
    return metadata


def nested_stat(row: pd.Series, field: str, stat: str) -> float:
    value = row.get(field)
    if isinstance(value, dict):
        return value.get(stat, math.nan)
    return math.nan


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(str(RAW / "episodes_*.jsonl"))
    episodes = pd.DataFrame(rows)
    if episodes.empty:
        episodes = pd.DataFrame(columns=EPISODE_COLUMNS)
    else:
        episodes = episodes[episodes["record_status"] == "completed"].copy()
        episodes = episodes.drop_duplicates(
            subset=["config_hash", "task_suite", "task_id", "seed", "episode_index"],
            keep="last",
        )
        for source, prefix in [
            ("policy_latency_ms", "policy_latency"),
            ("environment_step_latency_ms", "environment_step_latency"),
        ]:
            for stat in ["mean", "std", "p50", "p90", "p95"]:
                episodes[f"{prefix}_{stat}_ms"] = episodes.apply(
                    lambda row, source=source, stat=stat: nested_stat(row, source, stat),
                    axis=1,
                )

    metadata = metadata_by_hash()
    for field, getter in [
        ("precision", lambda item: item["config"]["precision"]),
        ("quantization_mode", lambda item: item["config"]["quantization_mode"]),
        ("quantization_scope", lambda item: item["config"]["quantization_scope"]),
        ("num_open_loop_steps", lambda item: item["config"]["num_open_loop_steps"]),
        ("denoising_steps", lambda item: item["config"]["denoising_steps"]),
        ("dynamic_replan", lambda item: item["config"]["dynamic_replan"]["enabled"]),
        (
            "real_quantization",
            lambda item: item["quantization"].get("real_quantization", False),
        ),
        (
            "hardware_accelerated",
            lambda item: item["quantization"].get("hardware_accelerated", False),
        ),
    ]:
        if not episodes.empty:
            episodes[field] = episodes["config_hash"].map(
                lambda config_hash, getter=getter: (
                    getter(metadata[config_hash]) if config_hash in metadata else None
                )
            )
    episodes.to_csv(OUT / "episodes.csv", index=False)

    fairness_rows: list[dict[str, Any]] = []
    if not episodes.empty:
        fairness_keys = ["task_suite", "task_id", "seed", "episode_index"]
        for key, group in episodes.groupby(fairness_keys, dropna=False):
            hashes = group["initial_state_sha256"].nunique()
            fairness_rows.append(
                {
                    **dict(zip(fairness_keys, key)),
                    "configurations": group["config_hash"].nunique(),
                    "unique_initial_state_hashes": hashes,
                    "fair": hashes == 1,
                }
            )
    fairness = pd.DataFrame(
        fairness_rows,
        columns=[
            "task_suite",
            "task_id",
            "seed",
            "episode_index",
            "configurations",
            "unique_initial_state_hashes",
            "fair",
        ],
    )
    fairness.to_csv(OUT / "fairness_audit.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    if not episodes.empty:
        for (config_hash, experiment_name), group in episodes.groupby(
            ["config_hash", "experiment_name"]
        ):
            successes = int(group["success"].sum())
            total = len(group)
            low, high = wilson(successes, total)
            first = group.iloc[0]
            summary_rows.append(
                {
                    "config_hash": config_hash,
                    "experiment_name": experiment_name,
                    "precision": first.get("precision"),
                    "quantization_mode": first.get("quantization_mode"),
                    "quantization_scope": first.get("quantization_scope"),
                    "real_quantization": first.get("real_quantization"),
                    "hardware_accelerated": first.get("hardware_accelerated"),
                    "num_open_loop_steps": first.get("num_open_loop_steps"),
                    "denoising_steps": first.get("denoising_steps"),
                    "dynamic_replan": first.get("dynamic_replan"),
                    "completed_episodes": total,
                    "successful_episodes": successes,
                    "success_rate": successes / total,
                    "success_ci95_low": low,
                    "success_ci95_high": high,
                    "policy_latency_mean_ms": group["policy_latency_mean_ms"].mean(),
                    "policy_latency_median_ms": group["policy_latency_p50_ms"].median(),
                    "policy_latency_p90_ms": group["policy_latency_p90_ms"].mean(),
                    "policy_latency_p95_ms": group["policy_latency_p95_ms"].mean(),
                    "environment_step_latency_mean_ms": group[
                        "environment_step_latency_mean_ms"
                    ].mean(),
                    "episode_latency_mean_ms": group["episode_total_ms"].mean(),
                    "episode_latency_median_ms": group["episode_total_ms"].median(),
                    "peak_gpu_memory_mb": group["peak_gpu_memory_mb"].max(),
                    "peak_process_memory_mb": group["peak_process_memory_mb"].max(),
                    "model_calls_per_episode": group["policy_calls"].mean(),
                    "valid_executed_actions_per_chunk": group[
                        "valid_executed_actions_per_chunk"
                    ].mean(),
                    "stale_action_ratio": group["stale_action_ratio"].mean(),
                    "action_discontinuity_l1_mean": group[
                        "action_discontinuity_l1_mean"
                    ].mean(),
                    "action_discontinuity_l2_mean": group[
                        "action_discontinuity_l2_mean"
                    ].mean(),
                    "observation_staleness_steps_mean": group[
                        "observation_staleness_steps_mean"
                    ].mean(),
                    "observation_staleness_ms_mean": group[
                        "observation_staleness_ms_mean"
                    ].mean(),
                    "replanning_overhead_ratio": group[
                        "replanning_overhead_ratio"
                    ].mean(),
                    "average_power_watts": group["average_power_watts"].mean(),
                    "energy_joules_estimated": group["energy_joules_estimated"].mean(),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "overall_summary.csv", index=False)

    if episodes.empty:
        failures = pd.DataFrame(
            columns=["experiment_name", "termination_reason", "count"]
        )
    else:
        failures = (
            episodes.groupby(["experiment_name", "termination_reason"])
            .size()
            .rename("count")
            .reset_index()
        )
    failures.to_csv(OUT / "failure_summary.csv", index=False)

    inventory = {
        "completed_libero_plus_episodes": int(len(episodes)),
        "completed_configurations": int(
            episodes["config_hash"].nunique() if not episodes.empty else 0
        ),
        "fairness_violations": int(
            (~fairness["fair"]).sum() if not fairness.empty else 0
        ),
        "original_libero_baseline": json.loads(
            (OUT / "baseline_sanity_original_libero10.json").read_text()
        ),
        "microbenchmark": json.loads(
            (OUT / "latency_microbench_combined.json").read_text()
        ),
        "missing_result_policy": (
            "Missing configurations remain absent/NaN; no values are imputed."
        ),
    }
    (OUT / "result_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_ci(values: list[float], seed: int = 0, samples: int = 10_000) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(samples, len(array)), replace=True), axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    traces = read_jsonl(run_dir / "inference_trace.jsonl")
    summary_path = run_dir / "summary.json"
    run_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    warmup_requests = int(run_summary.get("latency", {}).get("warmup_requests_excluded", 0))
    measured_traces = traces[warmup_requests:]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["task_name"]].append(episode)
    per_task = {}
    for task, rows in sorted(grouped.items()):
        values = [float(row["success"]) for row in rows]
        times = [float(row["episode_wall_clock_time_s"]) for row in rows]
        per_task[task] = {
            "episodes": len(rows),
            "success_mean": float(np.mean(values)),
            "success_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "success_bootstrap_95_ci": bootstrap_ci(values),
            "success_wilson_95_ci": wilson_ci(sum(bool(value) for value in values), len(values)),
            "episode_time_mean_s": float(np.mean(times)),
            "episode_time_std_s": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "episode_time_bootstrap_95_ci": bootstrap_ci(times),
        }
    all_success = [float(row["success"]) for row in episodes]
    all_episode_times = [float(row["episode_wall_clock_time_s"]) for row in episodes]
    latencies = [float(row["total_policy_request_latency_ms"]) for row in measured_traces]
    dit_latencies = [float(row["dit_denoising_latency_ms"]) for row in measured_traces]
    decode_latencies = [float(row["future_state_decode_latency_ms"]) for row in measured_traces]
    selected_steps = sorted({int(row["selected_denoising_steps"]) for row in traces})
    forward_counts = [int(row["denoiser_forward_count"]) for row in measured_traces]
    return {
        "run_dir": str(run_dir),
        "per_task": per_task,
        "macro_success_rate": float(np.mean([row["success_mean"] for row in per_task.values()])) if per_task else None,
        "aggregate_success_rate": float(np.mean(all_success)) if all_success else None,
        "aggregate_success_bootstrap_95_ci": bootstrap_ci(all_success),
        "aggregate_success_wilson_95_ci": wilson_ci(
            sum(bool(value) for value in all_success),
            len(all_success),
        ),
        "episode_time_mean_s": float(np.mean(all_episode_times)) if all_episode_times else None,
        "episode_time_std_s": (
            float(np.std(all_episode_times, ddof=1)) if len(all_episode_times) > 1 else 0.0
        ),
        "episode_time_bootstrap_95_ci": bootstrap_ci(all_episode_times),
        "warmup_requests_excluded": warmup_requests,
        "measured_requests": len(measured_traces),
        "policy_latency_mean_ms": float(np.mean(latencies)) if latencies else None,
        "policy_latency_std_ms": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
        "policy_latency_bootstrap_95_ci": bootstrap_ci(latencies),
        "dit_latency_mean_ms": float(np.mean(dit_latencies)) if dit_latencies else None,
        "future_decode_latency_mean_ms": float(np.mean(decode_latencies)) if decode_latencies else None,
        "denoising_steps": selected_steps[0] if len(selected_steps) == 1 else selected_steps,
        "denoiser_forward_count": int(sum(forward_counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/summary.json"))
    args = parser.parse_args()
    output = [summarize_run(path) for path in args.run_dirs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

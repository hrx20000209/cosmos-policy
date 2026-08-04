#!/usr/bin/env python3
"""Create immediate per-phase health metrics and an intermediate diagnostic plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
EXPERIMENT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    manifest = load_jsonl(args.manifest)
    expected = {row["episode_key"] for row in manifest}
    latest = {}
    requests = []
    for path in OUTPUT.glob("raw/*/episodes.shard*.jsonl"):
        for row in load_jsonl(path):
            if row.get("episode_key") in expected:
                latest[row["episode_key"]] = row
    for path in OUTPUT.glob("raw/*/requests.shard*.jsonl"):
        requests.extend(row for row in load_jsonl(path) if row.get("episode_key") in expected)
    requests = [
        row
        for row in requests
        if row.get("episode_key") in latest
        and row.get("run_id") == latest[row["episode_key"]].get("run_id")
    ]

    request_seen: dict[str, int] = {}
    measured = []
    for row in sorted(requests, key=lambda value: (value["run_id"], value["inference_start_ns"])):
        index = request_seen.get(row["run_id"], 0)
        request_seen[row["run_id"]] = index + 1
        physical_gpu = row.get("physical_gpu_id")
        clean_timing_gpu = physical_gpu is None or int(physical_gpu) >= 4
        if index >= 10 and clean_timing_gpu:
            measured.append(row)
    by_step = {}
    for step in range(1, 7):
        episode_subset = [row for row in latest.values() if int(row["denoising_steps"]) == step]
        request_subset = [
            row for row in measured if int(row["selected_denoising_steps"]) == step
        ]
        invalid = [
            row
            for row in episode_subset
            if not row.get("environment_valid", False)
            or str(row.get("termination_reason", "")).startswith(
                ("validation_error:", "fatal:", "error:")
            )
        ]
        valid = [row for row in episode_subset if row not in invalid]
        by_step[str(step)] = {
            "episodes": len(episode_subset),
            "valid_episodes": len(valid),
            "successes": sum(bool(row["success"]) for row in valid),
            "success_rate": (
                sum(bool(row["success"]) for row in valid) / len(valid) if valid else None
            ),
            "invalid_episodes": len(invalid),
            "mean_policy_latency_ms": (
                float(np.mean([row["total_policy_request_latency_ms"] for row in request_subset]))
                if request_subset
                else None
            ),
            "mean_dit_latency_ms": (
                float(np.mean([row["dit_denoising_latency_ms"] for row in request_subset]))
                if request_subset
                else None
            ),
            "mean_forward_count": (
                float(np.mean([row["denoiser_forward_count"] for row in request_subset]))
                if request_subset
                else None
            ),
            "forward_mismatches": sum(
                int(row["selected_denoising_steps"]) != int(row["denoiser_forward_count"])
                for row in request_subset
            ),
        }
    result = {
        "phase": args.phase,
        "expected_episodes": len(expected),
        "completed_episodes": len(latest),
        "request_records": len(requests),
        "warmup_requests_excluded": len(requests) - len(measured),
        "by_step": by_step,
    }
    summary_path = EXPERIMENT / "summaries" / f"phase_{args.phase}_interim.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    steps = np.arange(1, 7)
    success = [by_step[str(step)]["success_rate"] for step in steps]
    latency = [by_step[str(step)]["mean_policy_latency_ms"] for step in steps]
    dit = [by_step[str(step)]["mean_dit_latency_ms"] for step in steps]
    forwards = [
        by_step[str(step)]["mean_forward_count"]
        if by_step[str(step)]["mean_forward_count"] is not None
        else np.nan
        for step in steps
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(steps, success, marker="o")
    axes[0, 0].set(title="Success rate", ylim=(0, 1.02))
    axes[0, 1].plot(steps, latency, marker="o")
    axes[0, 1].set(title="Policy latency", ylabel="ms")
    axes[1, 0].plot(steps, dit, marker="o")
    axes[1, 0].set(title="DiT latency", ylabel="ms")
    axes[1, 1].plot(steps, steps, "--", color="black", label="expected")
    axes[1, 1].scatter(steps, forwards, label="observed")
    axes[1, 1].set(title="Forward-count audit")
    for ax in axes.flat:
        ax.set(xlabel="Denoising steps", xticks=steps)
        ax.grid(alpha=0.25)
    plot = EXPERIMENT / "plots" / f"interim_{args.phase}"
    fig.savefig(plot.with_suffix(".png"), dpi=300)
    fig.savefig(plot.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

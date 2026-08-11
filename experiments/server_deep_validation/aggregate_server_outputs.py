"""Aggregate Queue-A collection and ablation artifacts without Cosmos value."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def collection_summary(root: Path, manifest: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_key = {row["episode_key"]: row for row in rows}
    files = list(root.glob("group*/episode_*.pt")) + list(root.glob("episode_*.pt"))
    episodes = []
    for path in files:
        try:
            value = torch.load(path, weights_only=False)
            episodes.append(value)
        except Exception as error:
            episodes.append({"load_error": f"{type(error).__name__}:{error}", "path": str(path)})
    split_requests: dict[str, int] = defaultdict(int)
    split_episodes: dict[str, int] = defaultdict(int)
    task_requests: dict[str, int] = defaultdict(int)
    latencies = []
    for episode in episodes:
        key = episode.get("episode_key")
        row = by_key.get(key, episode)
        split = row.get("split", episode.get("split", "unknown"))
        split_episodes[split] += 1
        split_requests[split] += len(episode.get("requests", []))
        task_requests[row.get("task_uid", "unknown")] += len(episode.get("requests", []))
        latencies.extend(float(request.get("policy_latency_ms", 0.0)) for request in episode.get("requests", []))
    return {
        "episodes": len(episodes),
        "episodes_with_load_error": sum("load_error" in episode for episode in episodes),
        "requests": int(sum(split_requests.values())),
        "split_episodes": dict(split_episodes),
        "split_requests": dict(split_requests),
        "task_count": len(task_requests),
        "task_requests": dict(task_requests),
        "fresh_collection_latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95), "count": len(latencies)},
        "value_used": False,
    }


def flatten_metric(values: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    result = []
    for value in values:
        current: Any = value
        for key in path:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, (int, float)) and np.isfinite(current):
            result.append(float(current))
    return result


def summarize_metric(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p05": percentile(values, 5),
        "p95": percentile(values, 95),
    }


def ablation_summary(root: Path) -> dict[str, Any]:
    paths = list(root.glob("group*/state_*.pt")) + list(root.glob("state_*.pt"))
    states = []
    for path in paths:
        try:
            states.append(torch.load(path, weights_only=False))
        except Exception:
            continue
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        by_split[state.get("split", "unknown")].append(state)
    condition_names = ("F1", "P1", "PP", "PF", "FF")
    metrics: dict[str, Any] = {}
    for split, split_states in sorted(by_split.items()):
        metrics[split] = {
            "states": len(split_states),
            "tasks": len({state.get("task_uid") for state in split_states}),
            "action_to_F1": {
                condition: {
                    key: summarize_metric(flatten_metric([state.get("metrics", {}) for state in split_states], ("action_to_F1", condition, key)))
                    for key in ("l2", "cosine", "norm_ratio", "first_action_l2", "gripper_abs_error", "gripper_sign_match")
                }
                for condition in condition_names
            },
            "future_to_F1": {
                condition: {
                    key: summarize_metric(flatten_metric([state.get("metrics", {}) for state in split_states], ("future_to_F1", condition, key)))
                    for key in ("l2", "cosine", "norm_ratio")
                }
                for condition in condition_names
            },
            "innovation": {
                field: summarize_metric(flatten_metric([state.get("metrics", {}) for state in split_states], ("action_innovation", field)))
                for field in ("target_norm", "solver_norm", "feedback_norm", "solver_target_cosine", "feedback_target_cosine", "feedback_target_projection", "feedback_target_sign_agreement", "solver_target_projection", "did_to_f1_l2", "did_less_error_than_pf", "did_less_error_than_pp")
            },
        }
    return {
        "states": len(states),
        "splits": {split: len(values) for split, values in sorted(by_split.items())},
        "metrics": metrics,
        "value_used": False,
        "privileged_runtime_state_used": "only for exact offline state restore, never passed to policy",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "experiment": "server_inflight_wam_deep_validation",
        "collection": collection_summary(args.collection_root, args.manifest),
    }
    if args.ablation_root is not None and args.ablation_root.exists():
        result["ablation"] = ablation_summary(args.ablation_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

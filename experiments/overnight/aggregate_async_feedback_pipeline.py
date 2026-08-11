#!/usr/bin/env python3
"""Merge one-mode cross-request pipeline runs without re-running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def merge(paths: list[Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text()) for path in paths]
    hashes = {payload["checkpoint_sha256"] for payload in payloads}
    if len(hashes) != 1:
        raise RuntimeError(f"checkpoint mismatch: {hashes}")
    episodes = [episode for payload in payloads for episode in payload.get("episodes", [])]
    modes = [payload["modes"][0] for payload in payloads]
    summary: dict[str, Any] = {}
    for mode in modes:
        subset = [episode for episode in episodes if episode["mode"] == mode]
        def med(field: str) -> float | None:
            values = [episode["metrics"][field]["p50"] for episode in subset if episode["metrics"][field]["p50"] is not None]
            return float(np.nanmedian(values)) if values else None
        def med95(field: str) -> float | None:
            values = [episode["metrics"][field]["p95"] for episode in subset if episode["metrics"][field]["p95"] is not None]
            return float(np.nanmedian(values)) if values else None
        starvation = [episode["metrics"]["buffer_starvation_rate"] for episode in subset if episode["metrics"]["buffer_starvation_rate"] is not None]
        summary[mode] = {
            "episodes": len(subset),
            "successes": sum(bool(episode["success"]) for episode in subset),
            "success_rate": float(np.mean([bool(episode["success"]) for episode in subset])) if subset else None,
            "commit_latency_p50_ms": med("commit_latency_ms"),
            "commit_latency_p95_ms": med95("commit_latency_ms"),
            "inference_latency_p50_ms": med("inference_latency_ms"),
            "inference_latency_p95_ms": med95("inference_latency_ms"),
            "execution_overlap_p50_ms": med("execution_overlap_ms"),
            "execution_overlap_p95_ms": med95("execution_overlap_ms"),
            "commit_wait_after_execution_p50_ms": med("commit_wait_after_execution_ms"),
            "commit_wait_after_execution_p95_ms": med95("commit_wait_after_execution_ms"),
            "buffer_starvation_rate_mean": float(np.mean(starvation)) if starvation else None,
        }
    return {
        "schema_version": "async-feedback-assimilation-pipeline-aggregate-v1",
        "experiment": "cross_request_action_commit_pipeline",
        "checkpoint": payloads[0]["checkpoint"],
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "checkpoint_policy": payloads[0]["checkpoint_policy"],
        "modes": modes,
        "task_indices": payloads[0]["task_indices"],
        "init_states": payloads[0]["init_states"],
        "action_step_duration_ms": payloads[0]["action_step_duration_ms"],
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
        "timing_model": "simulator execution dwell; not a physical SO101 timing claim",
        "summary": summary,
        "episodes": episodes,
        "source_files": [str(path) for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge([Path(path) for path in args.input])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "episodes": len(result["episodes"]), "modes": result["modes"]}))


if __name__ == "__main__":
    main()

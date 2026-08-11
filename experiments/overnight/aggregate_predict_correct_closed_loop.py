"""Aggregate Predict-Correct shards against exact paired Phase 2A fresh rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def key(row: dict) -> tuple[str, int, int]:
    base_label = row.get("base_label")
    if base_label is None:
        suffix = f"_init{int(row['init_state_index'])}"
        label = str(row["label"])
        if not label.endswith(suffix):
            raise ValueError(f"cannot derive base label from {label!r}")
        base_label = label[: -len(suffix)]
    return (str(base_label), int(row["init_state_index"]), int(row["seed"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--phase2a", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.shard]
    phase2a = json.loads(args.phase2a.read_text(encoding="utf-8"))
    hashes = {shard["checkpoint_sha256"] for shard in shards} | {phase2a["checkpoint_sha256"]}
    if len(hashes) != 1:
        raise ValueError(f"checkpoint mismatch: {hashes}")

    candidate_rows = [row for shard in shards for row in shard["result"]["episodes"]]
    candidate = {key(row): row for row in candidate_rows}
    if len(candidate) != len(candidate_rows):
        raise ValueError("duplicate candidate keys")
    fresh_rows = [
        row
        for row in phase2a["records"]
        if row["configuration"] == "fresh" and int(row["init_state_index"]) in {0, 1} and key(row) in candidate
    ]
    fresh = {key(row): row for row in fresh_rows}
    if fresh.keys() != candidate.keys():
        raise ValueError(f"paired key mismatch fresh-only={fresh.keys()-candidate.keys()} candidate-only={candidate.keys()-fresh.keys()}")

    pairs = []
    for item_key in sorted(candidate):
        fresh_success = bool(fresh[item_key]["success"])
        candidate_success = bool(candidate[item_key]["success"])
        pairs.append(
            {
                "base_label": item_key[0],
                "init_state_index": item_key[1],
                "seed": item_key[2],
                "fresh_success": fresh_success,
                "candidate_success": candidate_success,
                "transition": f"{int(fresh_success)}->{int(candidate_success)}",
                "fresh_steps": int(fresh[item_key]["episode_steps"]),
                "candidate_steps": int(candidate[item_key]["episode_steps"]),
            }
        )

    transitions = {
        label: sum(pair["transition"] == label for pair in pairs)
        for label in ("1->1", "1->0", "0->1", "0->0")
    }
    traces = [trace for shard in shards for trace in shard["result"]["traces"]]
    corrected = [trace for trace in traces if trace.get("extra", {}).get("visual_input_mode") == "predict_correct"]
    bootstrap = [trace for trace in traces if trace.get("extra", {}).get("visual_input_mode") == "fresh"]

    def stage_values(rows: list[dict], name: str) -> list[float]:
        return [float(row.get("extra", {}).get("non_overlapping_stage_ms", {}).get(name, 0.0)) for row in rows]

    latency = {
        "corrected_requests": {
            "total_ms": stats([float(row["total_policy_request_latency_ms"]) for row in corrected]),
            "camera_preprocessing_ms": stats(stage_values(corrected, "camera_preprocessing_ms")),
            "vae_prefix_ms": stats(stage_values(corrected, "vae_encoding_ms")),
            "two_denoiser_forwards_ms": stats(stage_values(corrected, "dit_denoising_ms")),
            "denoiser_forward_1_ms": stats(
                [float(row["per_denoising_step_latency_ms"][0]) for row in corrected]
            ),
            "denoiser_forward_2_ms": stats(
                [float(row["per_denoising_step_latency_ms"][1]) for row in corrected]
            ),
        },
        "fresh_bootstrap_requests": {
            "n": len(bootstrap),
            "total_ms": stats([float(row["total_policy_request_latency_ms"]) for row in bootstrap]),
        },
        "phase2a_fresh_reference": phase2a.get("latency"),
    }
    per_task = {}
    for task in sorted({pair["base_label"] for pair in pairs}):
        group = [pair for pair in pairs if pair["base_label"] == task]
        per_task[task] = {
            "fresh_successes": sum(pair["fresh_success"] for pair in group),
            "candidate_successes": sum(pair["candidate_success"] for pair in group),
            "episodes": len(group),
        }

    payload = {
        "schema_version": 1,
        "experiment": "persistent_predict_correct_closed_loop_aggregate",
        "checkpoint_sha256": hashes.pop(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "episodes": len(pairs),
        "tasks": len(per_task),
        "initial_states_per_task": 2,
        "fresh_successes": sum(pair["fresh_success"] for pair in pairs),
        "candidate_successes": sum(pair["candidate_success"] for pair in pairs),
        "transitions": transitions,
        "pairs": pairs,
        "per_task": per_task,
        "latency": latency,
        "protocol": {
            "bootstrap": "one-step full fresh",
            "subsequent": "two forwards: predicted condition then persistent fresh visual condition",
            "fresh_prefix_pixel_frames": 13,
            "adaptive_scheduler": False,
            "threshold": False,
            "value_used": False,
            "privileged_runtime_input": False,
            "finetuning": False,
            "fresh_reference_reused": "Phase 2A exact task/init/seed/checkpoint rows"
        },
        "shards": [str(path) for path in args.shard],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("episodes", "fresh_successes", "candidate_successes", "transitions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

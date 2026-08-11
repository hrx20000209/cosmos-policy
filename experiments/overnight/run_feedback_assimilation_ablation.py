#!/usr/bin/env python3
"""Compute-matched feedback-assimilation ablation on fixed real WAM states.

The five conditions are:

  fresh_1, predicted_1, predicted_predicted, predicted_fresh, fresh_fresh

The two-forward conditions share the same seed and solver semantics.  The
predicted_fresh condition starts with the predicted visual condition and
switches the persistent visual condition to the current fresh prefix at the
second denoiser forward.  This is a diagnostic ablation; it uses no value,
threshold, scheduler, privileged state, or learned component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero_harness import (  # noqa: E402
    RealLiberoEnvironment,
    configure_repository_paths,
    extract_observation,
)
from experiments.progressive_wam.run_p1_trajectory_dump import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_STATS,
    DEFAULT_T5,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
)
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


VISUAL_SLOTS = (2, 3)
FUTURE_SLOTS = (5, 6, 7)
TRAJECTORY_FILES = (
    "/data/rxhuang/wam_overnight/trajectories/mechanism-discovery-d1-diagnostics1248/checkpoints.pt",
    "/data/rxhuang/wam_overnight/trajectories/mechanism-validation-d1-diagnostics1248/checkpoints.pt",
    "/data/rxhuang/wam_overnight/trajectories/mechanism-heldout-d1-diagnostics1248/checkpoints.pt",
)


def obs_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def action_array(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(-1, 7)


def latent_array(result: dict[str, Any]) -> np.ndarray:
    return result["generated_latent"].detach().float().cpu().numpy()


def action_discrepancy(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_action = action_array(left)
    right_action = action_array(right)
    return float(np.mean(np.linalg.norm(left_action - right_action, axis=-1)))


def future_discrepancy(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_latent = latent_array(left)
    right_latent = latent_array(right)
    return float(np.mean(np.abs(left_latent[:, :, FUTURE_SLOTS] - right_latent[:, :, FUTURE_SLOTS])))


def gripper_disagreement(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_gripper = action_array(left)[:, -1] >= 0.0
    right_gripper = action_array(right)[:, -1] >= 0.0
    return float(np.mean(left_gripper != right_gripper))


def load_states() -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for trajectory_path in TRAJECTORY_FILES:
        episodes = torch.load(trajectory_path, weights_only=False)
        for episode in episodes:
            requests = episode["requests"]
            for request_index in range(1, len(requests)):
                previous = requests[request_index - 1]
                current = requests[request_index]
                if "diagnostic_schedules" not in previous:
                    continue
                if 1 not in previous["diagnostic_schedules"] and "1" not in previous["diagnostic_schedules"]:
                    continue
                states.append(
                    {
                        "trajectory_path": trajectory_path,
                        "task_suite": episode["task_suite"],
                        "task_id": int(episode["task_id"]),
                        "task_description": episode["task_description"],
                        "episode_index": int(episode["episode_index"]),
                        "episode_seed": int(episode["seed"]),
                        "control_step": int(current["control_step"]),
                        "request_index": request_index,
                        "seed": int(episode["seed"]),
                        "sim_state": np.asarray(current["sim_state"], dtype=np.float64),
                        "previous_request": previous,
                    }
                )
    states.sort(key=lambda row: (
        row["task_suite"], row["task_id"], row["episode_index"], row["request_index"]
    ))
    return states


def build_predicted_latent(fresh_result: dict[str, Any], previous_request: dict[str, Any]) -> torch.Tensor:
    predicted_latent = fresh_result["orig_clean_latent_frames"].detach().clone()
    schedule = previous_request["diagnostic_schedules"]
    schedule = schedule[1] if 1 in schedule else schedule["1"]
    previous_future = torch.as_tensor(
        schedule["future_latents"][-1],
        device=predicted_latent.device,
        dtype=predicted_latent.dtype,
    )
    predicted_latent[:, :, 2] = previous_future[:, 1]
    predicted_latent[:, :, 3] = previous_future[:, 2]
    return predicted_latent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--libero-repo", default="/home/rxhuang/Projects/LIBERO")
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("invalid shard index")

    checkpoint = Path(args.checkpoint).resolve()
    if "so101" in str(checkpoint).lower() or "finet" in str(checkpoint).lower():
        raise ValueError(f"refusing finetuned/SO101 checkpoint: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != KNOWN_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected checkpoint SHA256 {digest}")

    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(REPO_ROOT)}})
    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)
    model.eval()
    model.inference_condition_transform = None

    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    all_states = load_states()
    states = [state for index, state in enumerate(all_states) if index % args.shard_count == args.shard_index]
    output_path = Path(args.output)
    raw_path = Path(args.raw_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or raw_path.exists():
        raise FileExistsError("refusing to overwrite existing ablation output")

    env_cache: dict[tuple[str, int], RealLiberoEnvironment] = {}
    rows: list[dict[str, Any]] = []

    def call(
        observation: Any,
        state: dict[str, Any],
        *,
        steps: int,
        previous_latent: torch.Tensor | None = None,
        transform: Any = None,
    ) -> tuple[dict[str, Any], dict[str, float], float]:
        metrics: dict[str, float] = {}
        cfg._inference_metrics_sink = metrics
        model.sampler.step_timing_events = []
        model.inference_condition_transform = transform
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            result = get_action(
                cfg,
                model,
                dataset_stats,
                obs_dict(observation),
                state["task_description"],
                seed=state["seed"],
                randomize_seed=False,
                num_denoising_steps_action=steps,
                generate_future_state_and_value_in_parallel=False,
                decode_future_state=False,
                skip_vae_encoding=previous_latent is not None,
                previous_generated_latent=previous_latent,
                skip_camera_preprocessing=previous_latent is not None,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        finally:
            model.inference_condition_transform = None
            model.sampler.step_timing_events = None
        wall_ms = (time.perf_counter() - start) * 1000.0
        if torch.cuda.is_available():
            metrics["peak_memory_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
            metrics["peak_memory_reserved_bytes"] = float(torch.cuda.max_memory_reserved())
        metrics["wall_latency_ms"] = wall_ms
        return result, metrics, wall_ms

    raw_handle = raw_path.open("w", encoding="utf-8")
    try:
        for local_index, state in enumerate(states):
            key = (state["task_suite"], state["task_id"])
            if key not in env_cache:
                env_cache[key] = RealLiberoEnvironment(key[0], key[1], args.resolution)
                env_cache[key].reset(state["episode_index"])
            env = env_cache[key]
            restore(env, state["sim_state"])
            raw = env.env.regenerate_obs_from_state(state["sim_state"])
            observation = extract_observation(raw, flip_vertical=True)

            fresh_1, fresh_1_metrics, _ = call(observation, state, steps=1)
            predicted_latent = build_predicted_latent(fresh_1, state["previous_request"])
            predicted_1, predicted_1_metrics, _ = call(
                observation, state, steps=1, previous_latent=predicted_latent
            )
            predicted_predicted, pp_metrics, _ = call(
                observation, state, steps=2, previous_latent=predicted_latent
            )

            fresh_visual = fresh_1["orig_clean_latent_frames"][:, :, VISUAL_SLOTS].detach().clone()

            def persistent_fresh(*, denoiser_forward_index: int, condition: Any) -> Any:
                if denoiser_forward_index >= 1:
                    condition.gt_frames[:, :, VISUAL_SLOTS] = fresh_visual.to(
                        device=condition.gt_frames.device,
                        dtype=condition.gt_frames.dtype,
                    )
                return condition

            predicted_fresh, pf_metrics, _ = call(
                observation,
                state,
                steps=2,
                previous_latent=predicted_latent,
                transform=persistent_fresh,
            )
            fresh_fresh, ff_metrics, _ = call(observation, state, steps=2)

            ff_action = action_array(fresh_fresh)
            pp_action = action_array(predicted_predicted)
            pf_action = action_array(predicted_fresh)
            pp_action_error = action_discrepancy(predicted_predicted, fresh_fresh)
            pf_action_error = action_discrepancy(predicted_fresh, fresh_fresh)
            pp_future_error = future_discrepancy(predicted_predicted, fresh_fresh)
            pf_future_error = future_discrepancy(predicted_fresh, fresh_fresh)
            row = {
                "global_state_index": all_states.index(state) if False else None,
                "local_state_index": local_index,
                "task_suite": state["task_suite"],
                "task_id": state["task_id"],
                "task_description": state["task_description"],
                "episode_index": state["episode_index"],
                "episode_seed": state["episode_seed"],
                "control_step": state["control_step"],
                "request_index": state["request_index"],
                "seed": state["seed"],
                "conditions": {
                    "fresh_1": {"action_error_to_fresh_fresh": action_discrepancy(fresh_1, fresh_fresh), "future_error_to_fresh_fresh": future_discrepancy(fresh_1, fresh_fresh), "gripper_disagreement_to_fresh_fresh": gripper_disagreement(fresh_1, fresh_fresh), "metrics": fresh_1_metrics},
                    "predicted_1": {"action_error_to_fresh_fresh": action_discrepancy(predicted_1, fresh_fresh), "future_error_to_fresh_fresh": future_discrepancy(predicted_1, fresh_fresh), "gripper_disagreement_to_fresh_fresh": gripper_disagreement(predicted_1, fresh_fresh), "metrics": predicted_1_metrics},
                    "predicted_predicted": {"action_error_to_fresh_fresh": pp_action_error, "future_error_to_fresh_fresh": pp_future_error, "gripper_disagreement_to_fresh_fresh": gripper_disagreement(predicted_predicted, fresh_fresh), "metrics": pp_metrics},
                    "predicted_fresh": {"action_error_to_fresh_fresh": pf_action_error, "future_error_to_fresh_fresh": pf_future_error, "gripper_disagreement_to_fresh_fresh": gripper_disagreement(predicted_fresh, fresh_fresh), "metrics": pf_metrics},
                    "fresh_fresh": {"action_error_to_fresh_fresh": 0.0, "future_error_to_fresh_fresh": 0.0, "gripper_disagreement_to_fresh_fresh": 0.0, "metrics": ff_metrics},
                },
                "recovery": {
                    "action_recovery_predicted_to_fresh": 1.0 - pf_action_error / max(pp_action_error, 1e-8),
                    "future_recovery_predicted_to_fresh": 1.0 - pf_future_error / max(pp_future_error, 1e-8),
                    "gripper_disagreement_reduction": 1.0 - gripper_disagreement(predicted_fresh, fresh_fresh) / max(gripper_disagreement(predicted_predicted, fresh_fresh), 1e-8),
                },
                "metadata": {
                    "value_used": False,
                    "privileged_state_runtime_input": False,
                    "adaptive_scheduler_used": False,
                    "threshold_used": False,
                    "finetuning_used": False,
                    "checkpoint_sha256": digest,
                    "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
                    "solver_semantics": "fixed native Cosmos sampler; one forward for 1-step, two forwards for 2-step",
                },
            }
            rows.append(row)
            raw_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            raw_handle.flush()
            if (local_index + 1) % 10 == 0 or local_index + 1 == len(states):
                print(json.dumps({"shard": args.shard_index, "completed": local_index + 1, "total": len(states)}), flush=True)
    finally:
        raw_handle.close()
        for env in env_cache.values():
            env.close()

    summary: dict[str, Any] = {}
    for condition in ("fresh_1", "predicted_1", "predicted_predicted", "predicted_fresh", "fresh_fresh"):
        values = [row["conditions"][condition] for row in rows]
        summary[condition] = {
            "n": len(values),
            "action_error_to_fresh_fresh_mean": float(np.mean([v["action_error_to_fresh_fresh"] for v in values])),
            "action_error_to_fresh_fresh_median": float(np.median([v["action_error_to_fresh_fresh"] for v in values])),
            "future_error_to_fresh_fresh_mean": float(np.mean([v["future_error_to_fresh_fresh"] for v in values])),
            "future_error_to_fresh_fresh_median": float(np.median([v["future_error_to_fresh_fresh"] for v in values])),
            "gripper_disagreement_mean": float(np.mean([v["gripper_disagreement_to_fresh_fresh"] for v in values])),
            "wall_latency_ms": {key: float(np.percentile([v["metrics"]["wall_latency_ms"] for v in values], percentile)) for key, percentile in (("p50", 50), ("p95", 95), ("p99", 99))},
            "gpu_work_ms": {key: float(np.percentile([v["metrics"].get("model_generate_inclusive_ms", v["metrics"]["wall_latency_ms"]) for v in values], percentile)) for key, percentile in (("p50", 50), ("p95", 95), ("p99", 99))},
        }
    recovery = [row["recovery"] for row in rows]
    summary["predicted_fresh_recovery"] = {
        "action_mean": float(np.mean([r["action_recovery_predicted_to_fresh"] for r in recovery])),
        "action_median": float(np.median([r["action_recovery_predicted_to_fresh"] for r in recovery])),
        "future_mean": float(np.mean([r["future_recovery_predicted_to_fresh"] for r in recovery])),
        "future_median": float(np.median([r["future_recovery_predicted_to_fresh"] for r in recovery])),
    }
    output = {
        "schema_version": "feedback-assimilation-ablation-v1",
        "experiment": "compute_matched_feedback_assimilation_ablation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
        "state_source": TRAJECTORY_FILES,
        "total_state_count": len(all_states),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "state_count": len(rows),
        "conditions": ["fresh_1", "predicted_1", "predicted_predicted", "predicted_fresh", "fresh_fresh"],
        "summary": summary,
        "records": rows,
        "raw_output": str(raw_path),
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "states": len(rows), "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

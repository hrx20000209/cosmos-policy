"""Test asynchronous persistent visual-condition correction during denoising.

The speculative run starts from the previous request's imagined visual state.
At a selected denoiser call, oracle fresh visual latents become available and
replace the persistent current-visual condition for that and all later calls.
This is a mechanism diagnostic: it uses no Cosmos value and does not train or
implement a scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
)
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


VISUAL_SLOTS = (2, 3)
FUTURE_SLOTS = (5, 6, 7)


def obs_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def action_array(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)


def action_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(np.mean(np.linalg.norm(action_array(left) - action_array(right), axis=-1)))


def latent_l1(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(
        torch.mean(
            torch.abs(left["generated_latent"][:, :, FUTURE_SLOTS] - right["generated_latent"][:, :, FUTURE_SLOTS])
        ).item()
    )


def load_selected(split_dirs: list[list[str]], states_per_split: int) -> list[tuple[str, dict, dict, dict]]:
    selected = []
    for split, directory_string in split_dirs:
        directory = Path(directory_string)
        source = directory / "checkpoints.pt"
        if not source.exists():
            source = directory / "checkpoints.partial.pt"
        episodes = torch.load(source, weights_only=False)
        candidates = []
        seen_tasks = set()
        for episode in episodes:
            key = (episode["task_suite"], int(episode["task_id"]))
            if key in seen_tasks or len(episode["requests"]) < 2:
                continue
            seen_tasks.add(key)
            candidates.append((split, episode, episode["requests"][0], episode["requests"][1]))
        positions = np.linspace(0, len(candidates) - 1, min(states_per_split, len(candidates))).round().astype(int)
        selected.extend(candidates[int(position)] for position in positions)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", action="append", nargs=2, metavar=("SPLIT", "DIR"), required=True)
    parser.add_argument("--states-per-split", type=int, default=3)
    parser.add_argument("--schedules", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json")
    parser.add_argument("--t5-embeddings", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--libero-repo", default="/home/rxhuang/Projects/LIBERO")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if "so101" in str(checkpoint).lower() or "finet" in str(checkpoint).lower():
        raise ValueError(f"refusing finetuned/SO101 checkpoint: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != KNOWN_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected checkpoint SHA256 {digest}")
    schedules = sorted(set(args.schedules))
    if not schedules or schedules[0] < 1:
        raise ValueError("all schedules must be positive")

    output = Path(args.output)
    raw_output = Path(args.raw_output) if args.raw_output else output.with_suffix(".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.exists():
        raise FileExistsError(raw_output)

    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(REPO_ROOT)}})
    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)
    model.eval()
    model.inference_condition_transform = None

    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    def call(
        observation: Any,
        episode: dict,
        request: dict,
        *,
        steps: int,
        latent=None,
        integrated_prefix_correction: bool = False,
    ) -> tuple[dict, float]:
        started = time.perf_counter()
        result = get_action(
            cfg,
            model,
            dataset_stats,
            obs_dict(observation),
            episode["task_description"],
            seed=int(request["seed"]),
            randomize_seed=False,
            num_denoising_steps_action=steps,
            generate_future_state_and_value_in_parallel=True,
            decode_future_state=False,
            skip_vae_encoding=latent is not None,
            previous_generated_latent=latent,
            skip_camera_preprocessing=latent is not None and not integrated_prefix_correction,
            persistent_visual_correction_prefix_frames=(13 if integrated_prefix_correction else None),
            persistent_visual_correction_arrival=1,
        )
        return result, 1000.0 * (time.perf_counter() - started)

    selected = load_selected(args.split_dir, args.states_per_split)
    rows = []
    env_cache: dict[tuple[str, int], RealLiberoEnvironment] = {}
    raw_handle = raw_output.open("w", encoding="utf-8")
    try:
        for state_index, (split, episode, previous_request, request) in enumerate(selected):
            key = (episode["task_suite"], int(episode["task_id"]))
            if key not in env_cache:
                env_cache[key] = RealLiberoEnvironment(key[0], key[1], 256)
                env_cache[key].reset(int(episode["episode_index"]))
            env = env_cache[key]
            sim_state = np.asarray(request["sim_state"], dtype=np.float64)
            restore(env, sim_state)
            raw = env.env.regenerate_obs_from_state(sim_state)
            observation = extract_observation(raw, flip_vertical=True)

            native_fresh, native_ms = call(observation, episode, request, steps=1)
            fresh_condition = native_fresh["orig_clean_latent_frames"].detach().clone()
            predicted_condition = fresh_condition.clone()
            previous_schedule = previous_request["diagnostic_schedules"]
            deployment_key = 1 if 1 in previous_schedule else "1"
            previous_future = torch.as_tensor(
                previous_schedule[deployment_key]["future_latents"][-1],
                device=predicted_condition.device,
                dtype=predicted_condition.dtype,
            )
            predicted_condition[:, :, 2] = previous_future[:, 1]
            predicted_condition[:, :, 3] = previous_future[:, 2]

            for steps in schedules:
                if steps == 1:
                    fresh, fresh_ms = native_fresh, native_ms
                else:
                    fresh, fresh_ms = call(observation, episode, request, steps=steps, latent=fresh_condition)
                predicted, predicted_ms = call(observation, episode, request, steps=steps, latent=predicted_condition)
                baseline_action = action_distance(predicted, fresh)
                baseline_future = latent_l1(predicted, fresh)
                corrections = []

                for arrival in range(steps):
                    fresh_visual = fresh_condition[:, :, VISUAL_SLOTS].detach().clone()

                    def transform(*, denoiser_forward_index: int, condition: Any) -> Any:
                        if denoiser_forward_index >= arrival:
                            condition.gt_frames[:, :, VISUAL_SLOTS] = fresh_visual.to(
                                device=condition.gt_frames.device,
                                dtype=condition.gt_frames.dtype,
                            )
                        return condition

                    model.inference_condition_transform = transform
                    try:
                        corrected, corrected_ms = call(
                            observation, episode, request, steps=steps, latent=predicted_condition
                        )
                    finally:
                        model.inference_condition_transform = None
                    corrected_action = action_distance(corrected, fresh)
                    corrected_future = latent_l1(corrected, fresh)
                    corrections.append(
                        {
                            "arrival_forward_index": arrival,
                            "remaining_forward_fraction": (steps - arrival) / steps,
                            "action_distance_to_fresh": corrected_action,
                            "future_l1_to_fresh": corrected_future,
                            "action_recovery": 1.0 - corrected_action / max(baseline_action, 1e-8),
                            "future_recovery": 1.0 - corrected_future / max(baseline_future, 1e-8),
                            "latency_ms": corrected_ms,
                        }
                    )

                integrated = None
                if steps == 2:
                    integrated_result, integrated_ms = call(
                        observation,
                        episode,
                        request,
                        steps=steps,
                        latent=predicted_condition,
                        integrated_prefix_correction=True,
                    )
                    integrated_action = action_distance(integrated_result, fresh)
                    integrated_future = latent_l1(integrated_result, fresh)
                    integrated = {
                        "prefix_pixel_frames": 13,
                        "arrival_forward_index": 1,
                        "action_distance_to_fresh": integrated_action,
                        "future_l1_to_fresh": integrated_future,
                        "action_recovery": 1.0 - integrated_action / max(baseline_action, 1e-8),
                        "future_recovery": 1.0 - integrated_future / max(baseline_future, 1e-8),
                        "latency_ms": integrated_ms,
                    }

                row = {
                    "state_index": state_index,
                    "split": split,
                    "task_suite": key[0],
                    "task_id": key[1],
                    "episode_index": int(episode["episode_index"]),
                    "control_step": int(request["control_step"]),
                    "denoising_steps": steps,
                    "native_fresh_one_step_latency_ms": native_ms,
                    "fresh_latency_ms": fresh_ms,
                    "predicted_latency_ms": predicted_ms,
                    "baseline_action_distance": baseline_action,
                    "baseline_future_l1": baseline_future,
                    "corrections": corrections,
                    "integrated_prefix_correction": integrated,
                }
                rows.append(row)
                raw_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                raw_handle.flush()
            print(f"[persistent-condition] {state_index + 1}/{len(selected)} {split} {key}", flush=True)
    finally:
        model.inference_condition_transform = None
        raw_handle.close()
        for env in env_cache.values():
            env.close()

    summary = {}
    for steps in schedules:
        for arrival in range(steps):
            for split in sorted({row["split"] for row in rows} | {"all"}):
                values = [
                    correction
                    for row in rows
                    if row["denoising_steps"] == steps and (split == "all" or row["split"] == split)
                    for correction in row["corrections"]
                    if correction["arrival_forward_index"] == arrival
                ]
                if not values:
                    continue
                summary[f"d{steps}_arrival{arrival}_{split}"] = {
                    "n": len(values),
                    "median_action_recovery": float(np.median([value["action_recovery"] for value in values])),
                    "median_future_recovery": float(np.median([value["future_recovery"] for value in values])),
                    "median_latency_ms": float(np.median([value["latency_ms"] for value in values])),
                }

    result = {
        "experiment": "persistent_visual_condition_correction",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "states": len(selected),
        "schedules": schedules,
        "deployment_baseline_steps": 1,
        "diagnostic_only": True,
        "oracle_fresh_visual_latent_required": True,
        "value_used": False,
        "summary": summary,
        "integrated_prefix_summary": {
            split: {
                "n": len(values),
                "median_action_recovery": float(np.median([value["action_recovery"] for value in values])),
                "median_future_recovery": float(np.median([value["future_recovery"] for value in values])),
                "median_latency_ms": float(np.median([value["latency_ms"] for value in values])),
            }
            for split in sorted({row["split"] for row in rows} | {"all"})
            if (
                values := [
                    row["integrated_prefix_correction"]
                    for row in rows
                    if row["denoising_steps"] == 2
                    and row["integrated_prefix_correction"] is not None
                    and (split == "all" or row["split"] == split)
                ]
            )
        },
        "native_fresh_one_step_latency": {
            "n": len(selected),
            "median_ms": float(
                np.median(
                    [
                        row["native_fresh_one_step_latency_ms"]
                        for row in rows
                        if row["denoising_steps"] == schedules[0]
                    ]
                )
            ),
            "mean_ms": float(
                np.mean(
                    [
                        row["native_fresh_one_step_latency_ms"]
                        for row in rows
                        if row["denoising_steps"] == schedules[0]
                    ]
                )
            ),
        },
        "raw_output": str(raw_output),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

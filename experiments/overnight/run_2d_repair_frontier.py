"""Oracle denoise-stage × DiT-block repair frontier on restored LIBERO states.

Fresh hidden activations and fresh x0 trajectories are oracle information.
This script asks where a correction *could* enter a pretrained joint WAM; it
does not implement a deployable scheduler and never reads Cosmos value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.runtime.model_probe import FullHiddenCapture  # noqa: E402
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
)
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


BLOCKS = (8, 16, 24)
STAGES = (0, 1, 3)
HIDDEN_GROUPS = {
    "current_visual": (2, 3),
    "future_visual": (6, 7),
    "action": (4,),
    "all_dynamic_nonvalue": (1, 2, 3, 4, 5, 6, 7),
}
DIFFUSION_GROUPS = {
    "future_visual": (6, 7),
    "future_proprio": (5,),
    "action": (4,),
    "future_plus_action": (4, 5, 6, 7),
    "all_nonvalue": (0, 1, 2, 3, 4, 5, 6, 7),
}
TOTAL_BLOCKS = 28


def obs_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=-1)))


def action_array(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)


def latent_l1(left: dict[str, Any], right: dict[str, Any], slots: tuple[int, ...]) -> float:
    left_latent = left["generated_latent"][:, :, slots]
    right_latent = right["generated_latent"][:, :, slots]
    return float(torch.mean(torch.abs(left_latent - right_latent)).item())


def load_selected(split_dirs: list[list[str]], states_per_split: int) -> list[tuple[str, dict, dict, dict]]:
    selected = []
    for split, directory_string in split_dirs:
        directory = Path(directory_string)
        source = directory / "checkpoints.pt"
        if not source.exists():
            source = directory / "checkpoints.partial.pt"
        episodes = torch.load(source, weights_only=False)
        per_split = []
        seen_tasks = set()
        for episode in episodes:
            key = (episode["task_suite"], int(episode["task_id"]))
            if key in seen_tasks or len(episode["requests"]) < 2:
                continue
            seen_tasks.add(key)
            per_split.append((split, episode, episode["requests"][0], episode["requests"][1]))
            if len(per_split) >= states_per_split:
                break
        selected.extend(per_split)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", action="append", nargs=2, metavar=("SPLIT", "DIR"), required=True)
    parser.add_argument("--states-per-split", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
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
    if any(stage >= args.steps for stage in STAGES):
        raise ValueError(f"stages {STAGES} require at least {max(STAGES) + 1} denoiser forwards")

    output = Path(args.output)
    raw_output = Path(args.raw_output) if args.raw_output else output.with_suffix(".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.exists():
        raise FileExistsError(raw_output)

    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(REPO_ROOT)}})
    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)
    model.eval()
    sampler = model.sampler

    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    def call(observation: Any, episode: dict, request: dict, latent: torch.Tensor | None = None) -> dict:
        return get_action(
            cfg,
            model,
            dataset_stats,
            obs_dict(observation),
            episode["task_description"],
            seed=int(request["seed"]),
            randomize_seed=False,
            num_denoising_steps_action=args.steps,
            generate_future_state_and_value_in_parallel=True,
            decode_future_state=False,
            skip_vae_encoding=latent is not None,
            previous_generated_latent=latent,
            skip_camera_preprocessing=latent is not None,
        )

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

            fresh_hidden: list[dict[int, torch.Tensor]] = []
            fresh_x0: list[torch.Tensor] = []
            model.intermediate_feature_ids = list(BLOCKS)
            model.intermediate_feature_reducer = FullHiddenCapture()

            def capture_fresh(**kwargs: Any) -> None:
                features = list(model.last_intermediate_features or ())
                if len(features) != len(BLOCKS):
                    raise RuntimeError(f"fresh stage has {len(features)} hidden tensors")
                fresh_hidden.append(dict(zip(BLOCKS, features, strict=True)))
                fresh_x0.append(kwargs["x0_pred_B_StateShape"].detach().clone())

            sampler.checkpoint_hook = capture_fresh
            fresh = call(observation, episode, request)
            sampler.checkpoint_hook = None
            model.intermediate_feature_ids = None
            model.intermediate_feature_reducer = None
            if len(fresh_hidden) != args.steps:
                raise RuntimeError(f"captured {len(fresh_hidden)} fresh stages, expected {args.steps}")

            predicted_condition = fresh["orig_clean_latent_frames"].detach().clone()
            previous_schedule = previous_request["diagnostic_schedules"]
            deployment_key = 1 if 1 in previous_schedule else "1"
            previous_future = torch.as_tensor(
                previous_schedule[deployment_key]["future_latents"][-1],
                device=predicted_condition.device,
                dtype=predicted_condition.dtype,
            )
            predicted_condition[:, :, 2] = previous_future[:, 1]
            predicted_condition[:, :, 3] = previous_future[:, 2]
            predicted = call(observation, episode, request, predicted_condition)
            fresh_action = action_array(fresh)
            predicted_action = action_array(predicted)
            baseline_distance = distance(predicted_action, fresh_action)
            baseline_future_distance = latent_l1(predicted, fresh, (5, 6, 7))

            hidden_repairs = []
            for stage in STAGES:
                for block in BLOCKS:
                    for group_name, slots in HIDDEN_GROUPS.items():
                        def pre_hook(*, denoiser_forward_index: int, **_: Any) -> None:
                            if denoiser_forward_index == stage:
                                model.net.activation_patch_request = {
                                    "fresh_hidden_by_block": {block: fresh_hidden[stage][block]},
                                    "slot_groups": {group_name: slots},
                                }
                            else:
                                model.net.activation_patch_request = None

                        def transform(*, denoiser_forward_index: int, predicted_clean: torch.Tensor, **_: Any) -> torch.Tensor:
                            if denoiser_forward_index != stage:
                                return predicted_clean
                            patched = model.last_activation_patch_latents
                            if not patched or block not in patched or group_name not in patched[block]:
                                raise RuntimeError(f"missing hidden repair stage={stage} block={block} group={group_name}")
                            return patched[block][group_name]

                        sampler.pre_denoise_hook = pre_hook
                        sampler.x0_transform = transform
                        try:
                            repaired = call(observation, episode, request, predicted_condition)
                        finally:
                            sampler.pre_denoise_hook = None
                            sampler.x0_transform = None
                            model.net.activation_patch_request = None
                        repaired_distance = distance(action_array(repaired), fresh_action)
                        hidden_repairs.append(
                            {
                                "stage": stage + 1,
                                "block": block,
                                "group": group_name,
                                "distance_to_fresh": repaired_distance,
                                "future_distance_to_fresh": latent_l1(repaired, fresh, (5, 6, 7)),
                                "future_change_from_predicted": latent_l1(repaired, predicted, (5, 6, 7)),
                                "recovery": 1.0 - repaired_distance / max(baseline_distance, 1e-8),
                                "ideal_remaining_dit_fraction": (
                                    (TOTAL_BLOCKS - block - 1) + (args.steps - stage - 1) * TOTAL_BLOCKS
                                ) / (args.steps * TOTAL_BLOCKS),
                            }
                        )

            diffusion_repairs = []
            for stage in STAGES:
                for group_name, slots in DIFFUSION_GROUPS.items():
                    def diffusion_transform(
                        *, denoiser_forward_index: int, predicted_clean: torch.Tensor, **_: Any
                    ) -> torch.Tensor:
                        if denoiser_forward_index != stage:
                            return predicted_clean
                        corrected = predicted_clean.clone()
                        corrected[:, :, slots] = fresh_x0[stage][:, :, slots].to(
                            device=corrected.device, dtype=corrected.dtype
                        )
                        return corrected

                    sampler.x0_transform = diffusion_transform
                    try:
                        repaired = call(observation, episode, request, predicted_condition)
                    finally:
                        sampler.x0_transform = None
                    repaired_distance = distance(action_array(repaired), fresh_action)
                    diffusion_repairs.append(
                        {
                            "stage": stage + 1,
                            "group": group_name,
                            "distance_to_fresh": repaired_distance,
                            "future_distance_to_fresh": latent_l1(repaired, fresh, (5, 6, 7)),
                            "future_change_from_predicted": latent_l1(repaired, predicted, (5, 6, 7)),
                            "recovery": 1.0 - repaired_distance / max(baseline_distance, 1e-8),
                            "ideal_remaining_dit_fraction": (
                                (args.steps - stage - 1) * TOTAL_BLOCKS / (args.steps * TOTAL_BLOCKS)
                            ),
                        }
                    )

            row = {
                "state_index": state_index,
                "split": split,
                "task_suite": key[0],
                "task_id": key[1],
                "episode_index": int(episode["episode_index"]),
                "control_step": int(request["control_step"]),
                "baseline_predicted_to_fresh": baseline_distance,
                "baseline_future_predicted_to_fresh": baseline_future_distance,
                "hidden_repairs": hidden_repairs,
                "diffusion_repairs": diffusion_repairs,
            }
            rows.append(row)
            raw_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            raw_handle.flush()
            print(
                f"[repair2d] {state_index + 1}/{len(selected)} {split} {key} baseline={baseline_distance:.6f}",
                flush=True,
            )
            del fresh_hidden, fresh_x0
    finally:
        raw_handle.close()
        sampler.checkpoint_hook = None
        sampler.pre_denoise_hook = None
        sampler.x0_transform = None
        model.net.activation_patch_request = None
        for env in env_cache.values():
            env.close()

    hidden_summary = {}
    for stage in STAGES:
        for block in BLOCKS:
            for group_name in HIDDEN_GROUPS:
                values = [
                    repair["recovery"]
                    for row in rows
                    for repair in row["hidden_repairs"]
                    if repair["stage"] == stage + 1 and repair["block"] == block and repair["group"] == group_name
                ]
                hidden_summary[f"s{stage + 1}_b{block}_{group_name}"] = {
                    "n": len(values),
                    "median_recovery": float(np.median(values)),
                    "mean_recovery": float(np.mean(values)),
                }
    diffusion_summary = {}
    for stage in STAGES:
        for group_name in DIFFUSION_GROUPS:
            values = [
                repair["recovery"]
                for row in rows
                for repair in row["diffusion_repairs"]
                if repair["stage"] == stage + 1 and repair["group"] == group_name
            ]
            diffusion_summary[f"s{stage + 1}_{group_name}"] = {
                "n": len(values),
                "median_recovery": float(np.median(values)),
                "mean_recovery": float(np.mean(values)),
            }
    result = {
        "experiment": "oracle_2d_repair_frontier",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "states": len(rows),
        "denoising_steps": args.steps,
        "deployment_baseline_steps": 1,
        "diagnostic_only": True,
        "oracle_fresh_prefix_required": True,
        "value_used": False,
        "hidden_summary": hidden_summary,
        "diffusion_summary": diffusion_summary,
        "raw_output": str(raw_output),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Collect the denoising-step × DiT-block surface on saved LIBERO states.

The executed dataset is produced by the one-step policy.  This script restores
those exact simulator states and performs 1/2/4/8-step diagnostics with the
same observation and seed.  It records only pooled per-slot block features;
the value slot and full attention maps are never read.
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

from cosmos_policy.runtime.model_probe import SpatialTokenReducer, TokenProbeConfig  # noqa: E402
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation  # noqa: E402
from experiments.progressive_wam.cosmos_hook import CosmosCheckpointCapture  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
)
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


BLOCK_IDS = (0, 4, 8, 12, 16, 20, 24, 27)
SLOT_IDS = (1, 2, 3, 4, 5, 6, 7)
SLOT_NAMES = (
    "current_proprio",
    "current_wrist",
    "current_primary",
    "action",
    "future_proprio",
    "future_wrist",
    "future_primary",
)


def obs_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def mean_step_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=-1)))


def select_states(episodes: list[dict[str, Any]], states_per_task: int) -> list[tuple[dict, dict, dict, dict]]:
    by_task: dict[tuple[str, int], list[tuple[dict, dict, dict, dict]]] = {}
    for episode in episodes:
        requests = episode["requests"]
        # A paired state needs the previous request's imagined future and the
        # next request's realized observation.  Exclude episode boundaries.
        for index in range(1, len(requests) - 1):
            key = (episode["task_suite"], int(episode["task_id"]))
            by_task.setdefault(key, []).append(
                (episode, requests[index - 1], requests[index], requests[index + 1])
            )
    selected = []
    for key in sorted(by_task):
        rows = by_task[key]
        positions = np.linspace(0, len(rows) - 1, min(states_per_task, len(rows))).round().astype(int)
        selected.extend(rows[int(position)] for position in positions)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True, choices=("discovery", "validation", "heldout"))
    parser.add_argument("--states-per-task", type=int, default=8)
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
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

    trajectory_dir = Path(args.trajectory_dir)
    source = trajectory_dir / "checkpoints.pt"
    if not source.exists():
        source = trajectory_dir / "checkpoints.partial.pt"
    episodes = torch.load(source, weights_only=False)
    selected = select_states(episodes, args.states_per_task)

    output_dir = Path(args.output_dir)
    state_dir = output_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "states.jsonl"
    if summary_path.exists():
        raise FileExistsError(summary_path)

    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(REPO_ROOT)}})
    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)
    model.eval()
    model.intermediate_feature_ids = list(BLOCK_IDS)
    model.intermediate_feature_reducer = SpatialTokenReducer(TokenProbeConfig(slot_indices=SLOT_IDS))
    capture = CosmosCheckpointCapture(
        model,
        cfg,
        dataset_stats,
        capture_future_latent=True,
        capture_value=False,
        capture_compact_hidden=True,
    )
    capture.install()

    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    env_cache: dict[tuple[str, int], RealLiberoEnvironment] = {}
    summaries = []
    try:
        with summary_path.open("w", encoding="utf-8") as summary_file:
            for state_index, (episode, previous_request, request, next_request) in enumerate(selected):
                key = (episode["task_suite"], int(episode["task_id"]))
                if key not in env_cache:
                    env_cache[key] = RealLiberoEnvironment(key[0], key[1], 256)
                    env_cache[key].reset(int(episode["episode_index"]))
                env = env_cache[key]
                restore(env, np.asarray(request["sim_state"], dtype=np.float64))
                raw = env.env.regenerate_obs_from_state(np.asarray(request["sim_state"], dtype=np.float64))
                observation = extract_observation(raw, flip_vertical=True)
                schedules: dict[int, dict[str, Any]] = {}
                predicted_schedules: dict[int, dict[str, Any]] = {}
                final_clean = None
                predicted_condition = None
                for num_steps in args.steps:
                    request_id = f"surface:{args.split}:{state_index}:k{num_steps}"
                    with capture.request(request_id, 0.0) as checkpoints:
                        result = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            obs_dict(observation),
                            episode["task_description"],
                            seed=int(request["seed"]),
                            randomize_seed=False,
                            num_denoising_steps_action=int(num_steps),
                            generate_future_state_and_value_in_parallel=True,
                            decode_future_state=False,
                        )
                    if len(checkpoints) != num_steps:
                        raise RuntimeError(f"k={num_steps}: captured {len(checkpoints)} checkpoints")
                    capture.assert_indices_match(result["latent_indices"])
                    actions = np.stack(
                        [capture.unnormalize_action(cp.predicted_clean_action.numpy()) for cp in checkpoints]
                    )
                    futures = np.stack([cp.predicted_future_latent.numpy() for cp in checkpoints])
                    hidden = np.stack([cp.compact_hidden.numpy() for cp in checkpoints])
                    official = np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)
                    schedules[int(num_steps)] = {
                        "actions": actions.astype(np.float32),
                        "future_latents": futures.astype(np.float16),
                        "compact_hidden": hidden.astype(np.float16),
                        "sigmas": np.asarray([cp.sigma for cp in checkpoints], dtype=np.float64),
                        "official_final": official,
                        "final_agreement_max_abs": float(np.max(np.abs(actions[-1] - official))),
                    }
                    if predicted_condition is None:
                        predicted_condition = result["orig_clean_latent_frames"].detach().clone()
                        previous_schedule = previous_request["diagnostic_schedules"]
                        deployment_key = 1 if 1 in previous_schedule else "1"
                        previous_future = torch.as_tensor(
                            previous_schedule[deployment_key]["future_latents"][-1],
                            device=predicted_condition.device,
                            dtype=predicted_condition.dtype,
                        )
                        # Previous future wrist/primary become the speculative
                        # current wrist/primary.  Current proprio remains real
                        # and is injected by get_action below.
                        predicted_condition[:, :, 2] = previous_future[:, 1]
                        predicted_condition[:, :, 3] = previous_future[:, 2]

                    predicted_request_id = f"surface:{args.split}:{state_index}:predicted:k{num_steps}"
                    with capture.request(predicted_request_id, 0.0) as predicted_checkpoints:
                        predicted_result = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            obs_dict(observation),
                            episode["task_description"],
                            seed=int(request["seed"]),
                            randomize_seed=False,
                            num_denoising_steps_action=int(num_steps),
                            generate_future_state_and_value_in_parallel=True,
                            decode_future_state=False,
                            skip_vae_encoding=True,
                            previous_generated_latent=predicted_condition,
                            skip_camera_preprocessing=True,
                        )
                    if len(predicted_checkpoints) != num_steps:
                        raise RuntimeError(
                            f"predicted k={num_steps}: captured {len(predicted_checkpoints)} checkpoints"
                        )
                    predicted_actions = np.stack(
                        [capture.unnormalize_action(cp.predicted_clean_action.numpy()) for cp in predicted_checkpoints]
                    )
                    predicted_futures = np.stack(
                        [cp.predicted_future_latent.numpy() for cp in predicted_checkpoints]
                    )
                    predicted_hidden = np.stack([cp.compact_hidden.numpy() for cp in predicted_checkpoints])
                    predicted_official = np.asarray(predicted_result["actions"], dtype=np.float32).reshape(16, 7)
                    predicted_schedules[int(num_steps)] = {
                        "actions": predicted_actions.astype(np.float32),
                        "future_latents": predicted_futures.astype(np.float16),
                        "compact_hidden": predicted_hidden.astype(np.float16),
                        "sigmas": np.asarray([cp.sigma for cp in predicted_checkpoints], dtype=np.float64),
                        "official_final": predicted_official,
                        "final_agreement_max_abs": float(
                            np.max(np.abs(predicted_actions[-1] - predicted_official))
                        ),
                    }
                    if num_steps == max(args.steps):
                        final_clean = result["orig_clean_latent_frames"].detach().cpu().to(torch.float16)

                # One-step modality ablations isolate wrist, primary, and
                # proprio effects without multiplying the expensive 8-step
                # diagnostic.  They are causal input interventions, not
                # runtime policies.
                input_variants: dict[str, dict[str, Any]] = {}
                variant_conditions = {
                    "predicted_wrist_fresh_primary": {2: 2},
                    "fresh_wrist_predicted_primary": {3: 3},
                }
                fresh_condition = result["orig_clean_latent_frames"].detach().clone()
                for variant_name, copied_slots in variant_conditions.items():
                    variant_condition = fresh_condition.clone()
                    for slot in copied_slots:
                        variant_condition[:, :, slot] = predicted_condition[:, :, slot]
                    with capture.request(f"surface:{args.split}:{state_index}:{variant_name}", 0.0) as variant_cp:
                        variant_result = get_action(
                            cfg,
                            model,
                            dataset_stats,
                            obs_dict(observation),
                            episode["task_description"],
                            seed=int(request["seed"]),
                            randomize_seed=False,
                            num_denoising_steps_action=1,
                            generate_future_state_and_value_in_parallel=True,
                            decode_future_state=False,
                            skip_vae_encoding=True,
                            previous_generated_latent=variant_condition,
                            skip_camera_preprocessing=True,
                        )
                    input_variants[variant_name] = {
                        "action": np.asarray(variant_result["actions"], dtype=np.float32).reshape(16, 7),
                        "compact_hidden": variant_cp[0].compact_hidden.numpy().astype(np.float16),
                    }

                perturbed_obs = obs_dict(observation)
                perturbed_obs["proprio"] = np.asarray(perturbed_obs["proprio"], dtype=np.float32).copy()
                perturbed_obs["proprio"][:6] += 0.02
                with capture.request(f"surface:{args.split}:{state_index}:perturbed_proprio", 0.0) as proprio_cp:
                    proprio_result = get_action(
                        cfg,
                        model,
                        dataset_stats,
                        perturbed_obs,
                        episode["task_description"],
                        seed=int(request["seed"]),
                        randomize_seed=False,
                        num_denoising_steps_action=1,
                        generate_future_state_and_value_in_parallel=True,
                        decode_future_state=False,
                    )
                input_variants["fresh_visual_perturbed_proprio"] = {
                    "action": np.asarray(proprio_result["actions"], dtype=np.float32).reshape(16, 7),
                    "compact_hidden": proprio_cp[0].compact_hidden.numpy().astype(np.float16),
                    "proprio_delta": np.asarray(perturbed_obs["proprio"] - observation.proprio, dtype=np.float32),
                }

                restore(env, np.asarray(next_request["sim_state"], dtype=np.float64))
                next_raw = env.env.regenerate_obs_from_state(np.asarray(next_request["sim_state"], dtype=np.float64))
                next_observation = extract_observation(next_raw, flip_vertical=True)
                with capture.request(f"surface:{args.split}:{state_index}:target", 0.0):
                    target = get_action(
                        cfg,
                        model,
                        dataset_stats,
                        obs_dict(next_observation),
                        episode["task_description"],
                        seed=int(request["seed"]),
                        randomize_seed=False,
                        num_denoising_steps_action=1,
                        generate_future_state_and_value_in_parallel=True,
                        decode_future_state=False,
                    )
                target_visual = target["orig_clean_latent_frames"][0, :, [2, 3]].detach().cpu().to(torch.float16)

                raw_record = {
                    "schedules": schedules,
                    "predicted_visual_schedules": predicted_schedules,
                    "input_variants": input_variants,
                    "target_visual_latent": target_visual,
                    "target_proprio": np.asarray(next_observation.proprio, dtype=np.float32),
                    "current_clean_latent": final_clean,
                }
                state_path = state_dir / f"state_{state_index:04d}.pt"
                torch.save(raw_record, state_path)

                max_steps = max(args.steps)
                reference_action = schedules[max_steps]["actions"][-1]
                reference_future = schedules[max_steps]["future_latents"][-1]
                stage_rows = {}
                for num_steps, schedule in schedules.items():
                    stage_rows[str(num_steps)] = []
                    for stage in range(num_steps):
                        visual = schedule["future_latents"][stage, :, 1:3]
                        stage_rows[str(num_steps)].append(
                            {
                                "stage": stage + 1,
                                "sigma": float(schedule["sigmas"][stage]),
                                "action_l2_to_k8_final": mean_step_l2(schedule["actions"][stage], reference_action),
                                "future_l1_to_k8_final": float(np.mean(np.abs(schedule["future_latents"][stage] - reference_future))),
                                "visual_l1_to_target": float(np.mean(np.abs(visual - target_visual.numpy()))),
                            }
                        )
                stage_label = None
                labels = episode.get("stage_labels", [])
                if labels and int(request["control_step"]) < len(labels):
                    stage_label = labels[int(request["control_step"])]
                summary = {
                    "state_index": state_index,
                    "split": args.split,
                    "task_suite": key[0],
                    "task_id": key[1],
                    "task": episode["task_description"],
                    "episode_index": int(episode["episode_index"]),
                    "request_id": request["request_id"],
                    "control_step": int(request["control_step"]),
                    "offline_stage": stage_label,
                    "raw_state": str(state_path),
                    "stages": stage_rows,
                    "fresh_vs_predicted_action_l2": mean_step_l2(
                        schedules[max_steps]["actions"][-1],
                        predicted_schedules[max_steps]["actions"][-1],
                    ),
                    "fresh_vs_predicted_visual_condition_l1": float(
                        torch.mean(torch.abs(fresh_condition[:, :, 2:4] - predicted_condition[:, :, 2:4])).item()
                    ),
                }
                summaries.append(summary)
                summary_file.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n")
                summary_file.flush()
                print(f"[surface] {args.split} {state_index + 1}/{len(selected)} {key}", flush=True)
    finally:
        capture.uninstall()
        for env in env_cache.values():
            env.close()

    manifest = {
        "schema_version": 1,
        "experiment": "denoise_block_surface",
        "split": args.split,
        "source": str(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "denoising_steps": list(args.steps),
        "deployment_baseline_steps": 1,
        "block_ids": list(BLOCK_IDS),
        "slot_ids": list(SLOT_IDS),
        "slot_names": list(SLOT_NAMES),
        "value_used": False,
        "full_attention_saved": False,
        "states": len(summaries),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

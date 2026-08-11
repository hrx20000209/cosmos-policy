"""Coarse 4×4 latent-space visual patch map for next-action recovery."""

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

from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
)
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


CAMERAS = {"wrist": 2, "primary": 3}


def obs_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def action_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=-1)))


def select(split_dirs: list[list[str]], states_per_split: int) -> list[tuple[str, dict, dict, dict]]:
    selected = []
    for split, directory_string in split_dirs:
        directory = Path(directory_string)
        source = directory / "checkpoints.pt"
        if not source.exists():
            source = directory / "checkpoints.partial.pt"
        episodes = torch.load(source, weights_only=False)
        seen = set()
        for episode in episodes:
            key = (episode["task_suite"], int(episode["task_id"]))
            if key in seen or len(episode["requests"]) < 2:
                continue
            seen.add(key)
            selected.append((split, episode, episode["requests"][0], episode["requests"][1]))
            if len([row for row in selected if row[0] == split]) >= states_per_split:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", action="append", nargs=2, metavar=("SPLIT", "DIR"), required=True)
    parser.add_argument("--states-per-split", type=int, default=3)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--output", required=True)
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

    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(REPO_ROOT)}})
    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)
    model.eval()
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    def call(observation: Any, episode: dict, request: dict, latent=None) -> dict:
        return get_action(
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
            skip_vae_encoding=latent is not None,
            previous_generated_latent=latent,
            skip_camera_preprocessing=latent is not None,
        )

    selected = select(args.split_dir, args.states_per_split)
    rows = []
    env_cache: dict[tuple[str, int], RealLiberoEnvironment] = {}
    try:
        for state_index, (split, episode, previous, request) in enumerate(selected):
            key = (episode["task_suite"], int(episode["task_id"]))
            if key not in env_cache:
                env_cache[key] = RealLiberoEnvironment(key[0], key[1], 256)
                env_cache[key].reset(int(episode["episode_index"]))
            env = env_cache[key]
            sim_state = np.asarray(request["sim_state"], dtype=np.float64)
            restore(env, sim_state)
            raw = env.env.regenerate_obs_from_state(sim_state)
            observation = extract_observation(raw, flip_vertical=True)
            fresh = call(observation, episode, request)
            fresh_condition = fresh["orig_clean_latent_frames"].detach().clone()
            previous_schedule = previous["diagnostic_schedules"]
            deployment_key = 1 if 1 in previous_schedule else "1"
            previous_future = torch.as_tensor(
                previous_schedule[deployment_key]["future_latents"][-1],
                device=fresh_condition.device,
                dtype=fresh_condition.dtype,
            )
            predicted_condition = fresh_condition.clone()
            predicted_condition[:, :, 2] = previous_future[:, 1]
            predicted_condition[:, :, 3] = previous_future[:, 2]
            predicted = call(observation, episode, request, predicted_condition)
            fresh_action = np.asarray(fresh["actions"], dtype=np.float32)
            predicted_action = np.asarray(predicted["actions"], dtype=np.float32)
            denominator = action_distance(predicted_action, fresh_action)
            height, width = fresh_condition.shape[-2:]
            y_edges = np.linspace(0, height, args.grid + 1).round().astype(int)
            x_edges = np.linspace(0, width, args.grid + 1).round().astype(int)
            patches = []
            for camera, slot in CAMERAS.items():
                total_camera_energy = float(
                    torch.sum(torch.abs(fresh_condition[:, :, slot] - predicted_condition[:, :, slot])).item()
                )
                for gy in range(args.grid):
                    for gx in range(args.grid):
                        ys = slice(int(y_edges[gy]), int(y_edges[gy + 1]))
                        xs = slice(int(x_edges[gx]), int(x_edges[gx + 1]))
                        patched = predicted_condition.clone()
                        patched[:, :, slot, ys, xs] = fresh_condition[:, :, slot, ys, xs]
                        result = call(observation, episode, request, patched)
                        action = np.asarray(result["actions"], dtype=np.float32)
                        remaining = action_distance(action, fresh_action)
                        patch_energy = float(
                            torch.sum(
                                torch.abs(
                                    fresh_condition[:, :, slot, ys, xs]
                                    - predicted_condition[:, :, slot, ys, xs]
                                )
                            ).item()
                        )
                        patches.append(
                            {
                                "camera": camera,
                                "gy": gy,
                                "gx": gx,
                                "action_recovery": 1.0 - remaining / max(denominator, 1e-8),
                                "remaining_action_l2": remaining,
                                "patch_energy_fraction": patch_energy / max(total_camera_energy, 1e-8),
                            }
                        )
            row = {
                "state_index": state_index,
                "split": split,
                "task_suite": key[0],
                "task_id": key[1],
                "baseline_action_l2": denominator,
                "patches": patches,
            }
            rows.append(row)
            print(f"[spatial] {state_index + 1}/{len(selected)} {split} {key}", flush=True)
    finally:
        for env in env_cache.values():
            env.close()

    summaries = {}
    for split in sorted({row["split"] for row in rows}):
        for camera in CAMERAS:
            cells = {}
            for gy in range(args.grid):
                for gx in range(args.grid):
                    values = [
                        patch["action_recovery"]
                        for row in rows
                        if row["split"] == split
                        for patch in row["patches"]
                        if patch["camera"] == camera and patch["gy"] == gy and patch["gx"] == gx
                    ]
                    cells[f"{gy},{gx}"] = float(np.median(values))
            positive = np.asarray(list(cells.values()))
            summaries[f"{split}_{camera}"] = {
                "median_recovery_by_cell": cells,
                "top_cell": max(cells, key=cells.get),
                "top_cell_median_recovery": float(np.max(positive)),
                "positive_cell_fraction": float(np.mean(positive > 0)),
            }
    result = {
        "experiment": "coarse_4x4_spatial_control_relevant_visual_map",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "states": len(rows),
        "grid": args.grid,
        "summaries": summaries,
        "rows": rows,
        "value_used": False,
        "temporal_guardrail": (
            "LIBERO's 33-frame VAE axis is a modality/slot assembly, not chronological sensing history; "
            "therefore no fictitious recent-k/old-frame temporal map is reported."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

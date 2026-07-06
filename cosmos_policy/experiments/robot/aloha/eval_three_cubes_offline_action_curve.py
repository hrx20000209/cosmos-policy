"""Run one checkpoint query and align every predicted SO101 dimension with ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cosmos_policy.datasets.lerobot_so101_dataset import LeRobotSO101Dataset
from cosmos_policy.experiments.robot.aloha.deploy import PolicyServer
from cosmos_policy.experiments.robot.aloha.deploy_so101_three_cubes import SO101DeployConfig
from cosmos_policy.experiments.robot.aloha.so101_schema import load_schema, print_schema_summary
from cosmos_policy.experiments.robot.cosmos_utils import get_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default="/data/rxhuang/three_cubes_1")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-denoising-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=195)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema()
    dataset = LeRobotSO101Dataset(
        data_dir=args.dataset_root,
        is_train=True,
        val_episodes=0,
        chunk_size=30,
        use_image_aug=False,
    )
    matches = np.flatnonzero((dataset.episode_indices == args.episode) & (dataset.frame_indices == args.timestep))
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely find episode={args.episode}, timestep={args.timestep}")
    global_index = int(matches[0])
    task = dataset.tasks[int(dataset.task_indices[global_index])]
    observation = {
        "primary_image": dataset.video_store.read("front", global_index),
        "left_wrist_image": dataset.video_store.read("right", global_index),
        "right_wrist_image": dataset.video_store.read("wrist", global_index),
        "proprio": dataset.states[global_index].copy(),
    }

    cfg = SO101DeployConfig(
        ckpt_path=args.checkpoint,
        seed=args.seed,
        num_denoising_steps_action=args.num_denoising_steps,
        dataset_stats_path=str(Path(__file__).with_name("three_cubes_so101_dataset_statistics.json")),
        t5_text_embeddings_path=str(Path(args.dataset_root) / "t5_embeddings.pkl"),
    )
    server = PolicyServer(cfg)
    result = get_action(
        cfg,
        server.model,
        server.dataset_stats,
        observation,
        task,
        seed=args.seed,
        randomize_seed=False,
        num_denoising_steps_action=args.num_denoising_steps,
        generate_future_state_and_value_in_parallel=False,
    )
    predicted = np.asarray(result["actions"], np.float32)
    if predicted.ndim == 3:
        predicted = predicted[0]
    ground_truth = dataset.absolute_action_chunk(global_index)
    if predicted.shape != ground_truth.shape:
        raise ValueError(f"Prediction/GT shape mismatch: {predicted.shape} vs {ground_truth.shape}")
    error = predicted - ground_truth
    mse_by_dim = np.mean(error**2, axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"action_curve_episode_{args.episode}_t_{args.timestep}"
    np.savez(
        args.output_dir / f"{stem}.npz",
        predicted_actions=predicted,
        ground_truth_actions=ground_truth,
        absolute_error=np.abs(error),
        squared_error=error**2,
        joint_order=np.asarray(schema["joint_order"]),
    )
    metrics = {
        "episode": args.episode,
        "timestep": args.timestep,
        "checkpoint": args.checkpoint,
        "joint_order": schema["joint_order"],
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "mse_by_dimension": dict(zip(schema["joint_order"], mse_by_dim.tolist(), strict=True)),
        "predicted_range": [float(predicted.min()), float(predicted.max())],
        "ground_truth_range": [float(ground_truth.min()), float(ground_truth.max())],
        "predicted_range_by_dimension": {
            name: [float(predicted[:, dim].min()), float(predicted[:, dim].max())]
            for dim, name in enumerate(schema["joint_order"])
        },
        "ground_truth_range_by_dimension": {
            name: [float(ground_truth[:, dim].min()), float(ground_truth[:, dim].max())]
            for dim, name in enumerate(schema["joint_order"])
        },
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    x = np.arange(len(predicted))
    for dim, axis in enumerate(axes.flat):
        axis.plot(x, ground_truth[:, dim], label="ground truth", linewidth=2)
        axis.plot(x, predicted[:, dim], label="predicted", linewidth=1.6)
        axis.set_title(f"{dim}: {schema['joint_order'][dim]} | MSE={mse_by_dim[dim]:.4f}")
        axis.set_ylabel("degree" if dim < 5 else "gripper % range")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("action step")
    axes[-1, 1].set_xlabel("action step")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{stem}.png", dpi=160)
    plt.close(fig)
    print_schema_summary(schema)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

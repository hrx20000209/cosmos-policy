"""离线生成 SO101 action chunk，并与物理尺度 ground truth 逐关节对齐。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from cosmos_policy.datasets.so101_lerobot_dataset import SO101LeRobotCosmosDataset
from cosmos_policy.experiments.robot.aloha.deploy import DeployConfig
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)


@dataclass
class SO101OfflineEvalConfig(DeployConfig):
    """只用于离线推理；不会创建 server，也不会发送机器人命令。"""

    suite: str = "aloha"  # 三相机 11-slot 布局与 ALOHA 推理布局相同。
    config: str = "cosmos_predict2_2b_480p_so101_lerobot"
    action_dim: int = 6
    chunk_size: int = 50
    num_wrist_images: int = 2
    num_third_person_images: int = 1
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    trained_with_image_aug: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t5_text_embeddings_path", required=True)
    parser.add_argument("--dataset_stats_path", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--num_denoising_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--no_test_time_center_crop", action="store_true")
    return parser.parse_args()


def _to_uint8(video: torch.Tensor, temporal_index: int) -> np.ndarray:
    image = video[:, temporal_index].detach().cpu().permute(1, 2, 0)
    if image.dtype == torch.uint8:
        return image.numpy()

    image = image.float()
    vmin = float(image.min())
    vmax = float(image.max())
    if vmin >= -1.5 and vmax <= 1.5 and vmin < 0:
        image = (image + 1.0) * 127.5
    elif vmin >= 0.0 and vmax <= 1.5:
        image = image * 255.0
    return image.round().clamp(0, 255).to(torch.uint8).numpy()


def _save_input_images(sample_dir: Path, observation: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, float]] = {}
    for key, filename in (
        ("primary_image", "input_primary.png"),
        ("left_wrist_image", "input_left_wrist.png"),
        ("right_wrist_image", "input_right_wrist.png"),
    ):
        image = observation[key]
        Image.fromarray(image).save(sample_dir / filename)
        records[key] = {
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": float(image.mean()),
            "std": float(image.std()),
        }
        if image.max() == image.min() or image.std() < 1.0:
            raise RuntimeError(f"{key} 输入图疑似全黑/全白/常量，停止评估：{records[key]}")
    return records


def _select_indices(dataset: SO101LeRobotCosmosDataset, count: int) -> list[int]:
    if count < 1:
        raise ValueError("num_samples 必须大于 0")
    episodes = np.asarray(dataset.dataset.hf_dataset["episode_index"], dtype=np.int64)
    unique_episodes = np.unique(episodes)
    selected_episodes = unique_episodes[np.linspace(0, len(unique_episodes) - 1, count, dtype=int)]
    fractions = (0.2, 0.4, 0.6, 0.8)
    result = []
    for sample_index, episode in enumerate(selected_episodes):
        candidates = np.flatnonzero(episodes == episode)
        local = min(int(fractions[sample_index % len(fractions)] * len(candidates)), len(candidates) - 1)
        result.append(int(candidates[local]))
    return result


def _direction_agreement(predicted: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_delta = np.diff(predicted, axis=1)
    gt_delta = np.diff(ground_truth, axis=1)
    moving = np.abs(gt_delta) > 1e-3
    agreement = np.zeros(ground_truth.shape[-1], dtype=np.float64)
    counts = moving.sum(axis=(0, 1))
    for dim in range(ground_truth.shape[-1]):
        if counts[dim]:
            agreement[dim] = np.mean(
                np.sign(pred_delta[..., dim][moving[..., dim]])
                == np.sign(gt_delta[..., dim][moving[..., dim]])
            )
        else:
            agreement[dim] = np.nan
    return agreement, counts


def _metric_dict(names: list[str], values: np.ndarray) -> dict[str, float | None]:
    return {
        name: (float(value) if np.isfinite(value) else None)
        for name, value in zip(names, values, strict=True)
    }


def _save_plot(
    path: Path,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    names: list[str],
    episode: int,
    frame: int,
) -> None:
    error = np.abs(predicted - ground_truth)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    steps = np.arange(len(predicted))
    for dim, axis in enumerate(axes.flat):
        axis.plot(steps, ground_truth[:, dim], label="ground truth", linewidth=2)
        axis.plot(steps, predicted[:, dim], label="predicted", linewidth=1.6)
        axis.plot(steps, error[:, dim], label="absolute error", linewidth=1.2, alpha=0.8)
        axis.set_title(f"{dim}: {names[dim]} | MAE={error[:, dim].mean():.3f}")
        axis.grid(alpha=0.25)
        axis.set_ylabel("degree" if dim < 5 else "gripper range")
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("action step")
    axes[-1, 1].set_xlabel("action step")
    fig.suptitle(f"episode={episode}, frame={frame}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.chunk_size != 50:
        raise ValueError("当前 SO101 Cosmos schema 固定为 50-step action chunk")
    dataset = SO101LeRobotCosmosDataset(
        repo_id=args.repo_id,
        root=args.root,
        chunk_size=args.chunk_size,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        dataset_stats_path=args.dataset_stats_path,
        use_image_aug=False,
        use_stronger_image_aug=False,
    )
    names = list(dataset.action_names or ())
    expected = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    if names != expected or dataset.action_dim != 6 or dataset.action_mode != "absolute":
        raise RuntimeError(
            f"SO101 action schema 不确定，停止离线推理：names={names}, "
            f"dim={dataset.action_dim}, mode={dataset.action_mode}"
        )

    cfg = SO101OfflineEvalConfig(
        ckpt_path=args.checkpoint,
        chunk_size=args.chunk_size,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        num_denoising_steps_action=args.num_denoising_steps,
        seed=args.seed,
        trained_with_image_aug=not args.no_test_time_center_crop,
    )
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    print(f"离线评估 action key='action', dim=6, mode='absolute', joint order={names}")
    print(f"action range min={stats['actions_min'].tolist()}, max={stats['actions_max'].tolist()}")
    print(f"gripper range=[{stats['actions_min'][5]}, {stats['actions_max'][5]}]")
    model, cosmos_config = get_model(cfg)
    if int(cosmos_config.dataloader_train.dataset.chunk_size) != args.chunk_size:
        raise RuntimeError("checkpoint config 的 chunk_size 与评估参数不一致")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_records = []
    for ordinal, index in enumerate(_select_indices(dataset, args.num_samples)):
        sample = dataset[index]
        observation = {
            "primary_image": _to_uint8(sample["video"], 13),
            "left_wrist_image": _to_uint8(sample["video"], 5),
            "right_wrist_image": _to_uint8(sample["video"], 9),
            "proprio": sample["physical_proprio"].numpy(),
        }
        episode = int(sample["episode_index"])
        frame = int(sample["frame_index"])
        stem = f"sample_{ordinal:02d}_episode_{episode}_frame_{frame}"
        input_image_stats = _save_input_images(args.output_dir / stem, observation)
        result = get_action(
            cfg,
            model,
            stats,
            observation,
            sample["command"],
            seed=args.seed + ordinal,
            randomize_seed=False,
            num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
        predicted = np.asarray(result["actions"], dtype=np.float32)
        target = sample["physical_actions"].numpy().astype(np.float32)
        if predicted.shape != target.shape or target.shape != (args.chunk_size, 6):
            raise RuntimeError(f"prediction/GT shape 不一致：{predicted.shape} vs {target.shape}")
        predictions.append(predicted)
        targets.append(target)
        _save_plot(args.output_dir / f"{stem}.png", predicted, target, names, episode, frame)
        np.savez_compressed(
            args.output_dir / f"{stem}.npz",
            predicted_actions=predicted,
            ground_truth_actions=target,
            absolute_error=np.abs(predicted - target),
            joint_order=np.asarray(names),
        )
        sample_records.append(
            {
                "dataset_index": index,
                "episode": episode,
                "frame": frame,
                "mae": float(np.mean(np.abs(predicted - target))),
                "first_5_step_mae": float(np.mean(np.abs(predicted[:5] - target[:5]))),
                "input_image_stats": input_image_stats,
            }
        )
        print(
            f"[{ordinal + 1}/{args.num_samples}] episode={episode}, "
            f"frame={frame}, MAE={sample_records[-1]['mae']:.4f}"
        )

    predicted_all = np.stack(predictions)
    target_all = np.stack(targets)
    error = predicted_all - target_all
    mae = np.mean(np.abs(error), axis=(0, 1))
    rmse = np.sqrt(np.mean(error**2, axis=(0, 1)))
    pred_std = np.std(predicted_all, axis=(0, 1))
    gt_std = np.std(target_all, axis=(0, 1))
    scale_ratio = pred_std / np.maximum(gt_std, 1e-6)
    pred_temporal_std = np.mean(np.std(predicted_all, axis=1), axis=0)
    gt_temporal_std = np.mean(np.std(target_all, axis=1), axis=0)
    temporal_scale_ratio = pred_temporal_std / np.maximum(gt_temporal_std, 1e-6)
    per_chunk_temporal_ratio = np.std(predicted_all, axis=1) / np.maximum(
        np.std(target_all, axis=1), 1e-6
    )
    collapsed_chunk_fraction = np.mean(per_chunk_temporal_ratio < 0.2, axis=0)
    direction, direction_counts = _direction_agreement(predicted_all, target_all)
    constant_joints = [names[i] for i in range(6) if temporal_scale_ratio[i] < 0.2]
    reverse_gripper = bool(direction_counts[5] >= 10 and direction[5] < 0.4)
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": args.num_samples,
        "chunk_size": args.chunk_size,
        "action_key": "action",
        "action_mode": "absolute",
        "joint_order": names,
        "action_range_min": stats["actions_min"].tolist(),
        "action_range_max": stats["actions_max"].tolist(),
        "per_joint_mae": _metric_dict(names, mae),
        "per_joint_rmse": _metric_dict(names, rmse),
        "per_joint_pred_std": _metric_dict(names, pred_std),
        "per_joint_gt_std": _metric_dict(names, gt_std),
        "per_joint_scale_ratio": _metric_dict(names, scale_ratio),
        "per_joint_mean_temporal_pred_std": _metric_dict(names, pred_temporal_std),
        "per_joint_mean_temporal_gt_std": _metric_dict(names, gt_temporal_std),
        "per_joint_temporal_scale_ratio": _metric_dict(names, temporal_scale_ratio),
        "per_joint_collapsed_chunk_fraction": _metric_dict(names, collapsed_chunk_fraction),
        "per_joint_direction_agreement": _metric_dict(names, direction),
        "per_joint_direction_samples": dict(zip(names, direction_counts.tolist(), strict=True)),
        "gripper_mae": float(mae[5]),
        "gripper_direction_agreement": float(direction[5]) if np.isfinite(direction[5]) else None,
        "first_5_step_mae": float(np.mean(np.abs(error[:, :5]))),
        "full_50_step_mae": float(np.mean(np.abs(error))),
        "obvious_reverse_gripper": reverse_gripper,
        "constant_predicted_joints": constant_joints,
        "possible_mean_collapse": bool(np.median(temporal_scale_ratio) < 0.35),
        "possible_temporal_action_collapse": bool(np.median(temporal_scale_ratio) < 0.35),
        "possible_scale_too_small": [names[i] for i in range(6) if scale_ratio[i] < 0.5],
        "possible_scale_too_large": [names[i] for i in range(6) if scale_ratio[i] > 2.0],
        "samples": sample_records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "per_joint_metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "joint",
                "mae",
                "rmse",
                "pred_std",
                "gt_std",
                "global_scale_ratio",
                "temporal_pred_std",
                "temporal_gt_std",
                "temporal_scale_ratio",
                "collapsed_chunk_fraction",
                "direction_agreement",
            ]
        )
        for index, name in enumerate(names):
            writer.writerow(
                [
                    name,
                    mae[index],
                    rmse[index],
                    pred_std[index],
                    gt_std[index],
                    scale_ratio[index],
                    pred_temporal_std[index],
                    gt_temporal_std[index],
                    temporal_scale_ratio[index],
                    collapsed_chunk_fraction[index],
                    direction[index],
                ]
            )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

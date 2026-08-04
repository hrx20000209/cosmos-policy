"""按固定 query stride 拼接 action chunk，评估一个完整 SO101 episode。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cosmos_policy.datasets.so101_lerobot_dataset import SO101LeRobotCosmosDataset
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.scripts.eval_so101_action_curves import SO101OfflineEvalConfig, _to_uint8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t5_text_embeddings_path", required=True)
    parser.add_argument("--dataset_stats_path", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--query_stride", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--num_denoising_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.query_stride <= args.chunk_size:
        raise ValueError("query_stride 必须在 [1, chunk_size] 内")
    dataset = SO101LeRobotCosmosDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=[args.episode],
        chunk_size=args.chunk_size,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        dataset_stats_path=args.dataset_stats_path,
        use_image_aug=False,
        use_stronger_image_aug=False,
    )
    names = list(dataset.action_names or ())
    if names != [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]:
        raise RuntimeError(f"action schema 不确定，停止评估：{names}")
    episode_ids = np.asarray(dataset.dataset.hf_dataset["episode_index"], dtype=np.int64)
    episode_indices = np.flatnonzero(episode_ids == args.episode)
    if not len(episode_indices):
        raise ValueError(f"找不到 episode={args.episode}")

    cfg = SO101OfflineEvalConfig(
        ckpt_path=args.checkpoint,
        chunk_size=args.chunk_size,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        num_denoising_steps_action=args.num_denoising_steps,
        seed=args.seed,
        trained_with_image_aug=True,
    )
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    print(f"离线评估 action key='action', dim=6, mode='absolute', joint order={names}")
    print(f"action range min={stats['actions_min'].tolist()}, max={stats['actions_max'].tolist()}")
    print(f"gripper range=[{stats['actions_min'][5]}, {stats['actions_max'][5]}]")
    model, cosmos_config = get_model(cfg)
    if int(cosmos_config.dataloader_train.dataset.chunk_size) != args.chunk_size:
        raise RuntimeError("checkpoint config 的 chunk_size 与评估参数不一致")

    predicted_segments = []
    target_segments = []
    predicted_chunks = []
    target_chunks = []
    segment_lengths = []
    query_frames = []
    for query_number, local_start in enumerate(range(0, len(episode_indices), args.query_stride)):
        index = int(episode_indices[local_start])
        sample = dataset[index]
        observation = {
            "primary_image": _to_uint8(sample["video"], 13),
            "left_wrist_image": _to_uint8(sample["video"], 5),
            "right_wrist_image": _to_uint8(sample["video"], 9),
            "proprio": sample["physical_proprio"].numpy(),
        }
        result = get_action(
            cfg,
            model,
            stats,
            observation,
            sample["command"],
            seed=args.seed + query_number,
            randomize_seed=False,
            num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
        predicted_chunk = np.asarray(result["actions"], dtype=np.float32)
        target_chunk = sample["physical_actions"].numpy().astype(np.float32)
        take = min(args.query_stride, len(episode_indices) - local_start)
        predicted_segments.append(predicted_chunk[:take])
        target_segments.append(target_chunk[:take])
        predicted_chunks.append(predicted_chunk)
        target_chunks.append(target_chunk)
        segment_lengths.append(take)
        query_frames.append(int(sample["frame_index"]))
        print(
            f"query {query_number + 1}: frame={query_frames[-1]}, take={take}, "
            f"chunk MAE={np.mean(np.abs(predicted_chunk[:take] - target_chunk[:take])):.4f}"
        )

    predicted = np.concatenate(predicted_segments)
    target = np.concatenate(target_segments)
    if predicted.shape != target.shape or predicted.shape != (len(episode_indices), 6):
        raise RuntimeError(f"完整 episode 拼接 shape 错误：{predicted.shape} vs {target.shape}")
    error = predicted - target
    predicted_chunks_array = np.stack(predicted_chunks)
    target_chunks_array = np.stack(target_chunks)
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(error**2, axis=0))
    pred_delta = np.diff(predicted, axis=0)
    target_delta = np.diff(target, axis=0)
    moving = np.abs(target_delta) > 1e-3
    direction = []
    for dim in range(6):
        direction.append(
            float(
                np.mean(
                    np.sign(pred_delta[:, dim][moving[:, dim]])
                    == np.sign(target_delta[:, dim][moving[:, dim]])
                )
            )
            if np.any(moving[:, dim])
            else None
        )

    first5_by_query = np.mean(np.abs(predicted_chunks_array[:, :5] - target_chunks_array[:, :5]), axis=(1, 2))
    pred_std = np.std(predicted, axis=0)
    target_std = np.std(target, axis=0)
    scale_ratio = pred_std / np.maximum(target_std, 1e-6)
    chunk_pred_temporal_std = np.std(predicted_chunks_array, axis=1)
    chunk_target_temporal_std = np.std(target_chunks_array, axis=1)
    # 先跨 query 聚合 std 再做比值，避免静止 GT chunk 的近零分母把均值放大到数万倍。
    mean_temporal_ratio = np.mean(chunk_pred_temporal_std, axis=0) / np.maximum(
        np.mean(chunk_target_temporal_std, axis=0), 1e-6
    )
    meaningful_motion = chunk_target_temporal_std > np.maximum(target_std * 1e-3, 1e-3)
    collapsed_chunk_fraction = np.asarray(
        [
            np.mean(
                chunk_pred_temporal_std[:, dim][meaningful_motion[:, dim]]
                < 0.2 * chunk_target_temporal_std[:, dim][meaningful_motion[:, dim]]
            )
            if np.any(meaningful_motion[:, dim])
            else 0.0
            for dim in range(6)
        ]
    )
    constant_output_joints = [names[index] for index in range(6) if mean_temporal_ratio[index] < 0.2]
    constant_output_flag = bool(constant_output_joints)
    mean_collapse_risk = bool(np.median(mean_temporal_ratio) < 0.35)

    boundaries = np.cumsum(segment_lengths)[:-1]
    if len(boundaries):
        pred_jumps = np.abs(predicted[boundaries] - predicted[boundaries - 1])
        target_jumps = np.abs(target[boundaries] - target[boundaries - 1])
    else:
        pred_jumps = np.zeros((0, 6), dtype=np.float32)
        target_jumps = np.zeros((0, 6), dtype=np.float32)
    action_range = np.maximum(stats["actions_max"] - stats["actions_min"], 1e-6)
    normalized_pred_jumps = pred_jumps / action_range
    normalized_target_jumps = target_jumps / action_range
    jump_score = float(np.mean(pred_jumps)) if pred_jumps.size else 0.0
    normalized_jump_score = float(np.mean(normalized_pred_jumps)) if pred_jumps.size else 0.0
    gt_jump_score = float(np.mean(target_jumps)) if target_jumps.size else 0.0
    normalized_gt_jump_score = float(np.mean(normalized_target_jumps)) if target_jumps.size else 0.0
    jump_excess_ratio = jump_score / max(gt_jump_score, 1e-6)
    max_normalized_jump = float(np.max(normalized_pred_jumps)) if pred_jumps.size else 0.0
    obvious_discontinuity = bool(normalized_jump_score > 0.1 or max_normalized_jump > 0.3)
    gripper_reverse = bool(direction[5] is not None and direction[5] < 0.4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:03d}_stride_{args.query_stride}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        predicted_actions=predicted,
        ground_truth_actions=target,
        absolute_error=np.abs(error),
        predicted_action_chunks=predicted_chunks_array,
        ground_truth_action_chunks=target_chunks_array,
        query_frames=np.asarray(query_frames),
        segment_lengths=np.asarray(segment_lengths),
        joint_order=np.asarray(names),
    )
    summary = {
        "checkpoint": args.checkpoint,
        "action_key": "action",
        "action_dim": 6,
        "action_mode": "absolute",
        "action_range_min": stats["actions_min"].tolist(),
        "action_range_max": stats["actions_max"].tolist(),
        "gripper_range": [float(stats["actions_min"][5]), float(stats["actions_max"][5])],
        "episode": args.episode,
        "episode_length": len(episode_indices),
        "query_stride": args.query_stride,
        "num_queries": len(query_frames),
        "num_denoising_steps": args.num_denoising_steps,
        "joint_order": names,
        "per_joint_mae": dict(zip(names, mae.tolist(), strict=True)),
        "per_joint_rmse": dict(zip(names, rmse.tolist(), strict=True)),
        "per_joint_direction_agreement": dict(zip(names, direction, strict=True)),
        "per_joint_scale_ratio": dict(zip(names, scale_ratio.tolist(), strict=True)),
        "per_joint_mean_chunk_temporal_scale_ratio": dict(zip(names, mean_temporal_ratio.tolist(), strict=True)),
        "per_joint_collapsed_chunk_fraction": dict(zip(names, collapsed_chunk_fraction.tolist(), strict=True)),
        "full_episode_mae": float(np.mean(np.abs(error))),
        "stitched_trajectory_mae": float(np.mean(np.abs(error))),
        "first_query_first_5_step_mae": float(first5_by_query[0]),
        "all_query_first_5_step_mae": float(np.mean(first5_by_query)),
        "action_jump_score": jump_score,
        "normalized_action_jump_score": normalized_jump_score,
        "ground_truth_action_jump_score": gt_jump_score,
        "normalized_ground_truth_action_jump_score": normalized_gt_jump_score,
        "action_jump_excess_ratio": jump_excess_ratio,
        "max_normalized_action_jump": max_normalized_jump,
        "obvious_action_discontinuity": obvious_discontinuity,
        "gripper_mae": float(mae[5]),
        "gripper_direction_agreement": direction[5],
        "obvious_reverse_gripper": gripper_reverse,
        "constant_output_flag": constant_output_flag,
        "constant_output_joints": constant_output_joints,
        "mean_collapse_risk": mean_collapse_risk,
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True)
    steps = np.arange(len(predicted))
    for dim, axis in enumerate(axes.flat):
        axis.plot(steps, target[:, dim], label="ground truth", linewidth=1.8)
        axis.plot(steps, predicted[:, dim], label="predicted (stitched)", linewidth=1.2)
        axis.plot(steps, np.abs(error[:, dim]), label="absolute error", linewidth=0.9, alpha=0.75)
        for query_frame in query_frames:
            axis.axvline(query_frame, color="gray", linewidth=0.3, alpha=0.18)
        axis.set_title(f"{dim}: {names[dim]} | MAE={mae[dim]:.3f}, RMSE={rmse[dim]:.3f}")
        axis.set_ylabel("degree" if dim < 5 else "gripper range")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("episode frame")
    axes[-1, 1].set_xlabel("episode frame")
    fig.suptitle(
        f"SO101 full episode {args.episode} | checkpoint={Path(args.checkpoint).name} | "
        f"query stride={args.query_stride}, denoise={args.num_denoising_steps}"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

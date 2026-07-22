"""按小 query stride 密集重新查询模型，模拟闭环重规划，并用时间集成（temporal
ensembling，指数衰减加权平均重叠 chunk 预测）拼接一个完整 SO101 episode 的 action 曲线。

与 eval_so101_full_episode_action_curve.py 的区别：后者 query_stride 通常等于
chunk_size，chunk 之间完全不重叠，直接硬切换，因此预测轨迹呈阶梯状；本脚本
query_stride 远小于 chunk_size，让相邻 chunk 大量重叠，同一帧会被多个 chunk
预测覆盖，再按 ACT 论文式的指数衰减权重（越新查询出的预测权重越高）做加权平均，
更贴近真实部署时"边走边重新规划"的闭环控制。
"""

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
    parser.add_argument("--query_stride", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--num_denoising_steps", type=int, default=5)
    parser.add_argument("--ensemble_decay", type=float, default=0.15, help="指数衰减系数 m：weight = exp(-m * age)")
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
    episode_length = len(episode_indices)

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
    print(f"离线评估（时间集成）action key='action', dim=6, mode='absolute', joint order={names}")
    model, cosmos_config = get_model(cfg)
    if int(cosmos_config.dataloader_train.dataset.chunk_size) != args.chunk_size:
        raise RuntimeError("checkpoint config 的 chunk_size 与评估参数不一致")

    # 每帧累加加权预测和权重和，最后一次性归一化，避免为每帧都存一份完整候选列表。
    weighted_sum = np.zeros((episode_length, 6), dtype=np.float64)
    weight_total = np.zeros((episode_length, 1), dtype=np.float64)
    target = np.zeros((episode_length, 6), dtype=np.float32)
    target_filled = np.zeros(episode_length, dtype=bool)
    query_frames = []

    for query_number, local_start in enumerate(range(0, episode_length, args.query_stride)):
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
        take = min(args.chunk_size, episode_length - local_start)
        query_frames.append(int(sample["frame_index"]))

        # age=0 是本次查询里最新鲜（刚预测出）的那一步，随着 chunk 内位置增加，
        # 这一步的动作是"更早之前规划出来的、更旧的"，权重按指数衰减降低。
        ages = np.arange(take)
        weights = np.exp(-args.ensemble_decay * ages)
        for i in range(take):
            t = local_start + i
            weighted_sum[t] += weights[i] * predicted_chunk[i]
            weight_total[t, 0] += weights[i]
            if not target_filled[t]:
                target[t] = target_chunk[i]
                target_filled[t] = True
        print(f"query {query_number + 1}: frame={query_frames[-1]}, take={take}")

    if not np.all(target_filled):
        raise RuntimeError("存在未被任何 query 覆盖的帧，请检查 query_stride/chunk_size 组合")
    predicted = (weighted_sum / np.maximum(weight_total, 1e-8)).astype(np.float32)

    error = predicted - target
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
                    np.sign(pred_delta[:, dim][moving[:, dim]]) == np.sign(target_delta[:, dim][moving[:, dim]])
                )
            )
            if np.any(moving[:, dim])
            else None
        )
    pred_std = np.std(predicted, axis=0)
    target_std = np.std(target, axis=0)
    scale_ratio = pred_std / np.maximum(target_std, 1e-6)

    pred_jump = np.abs(pred_delta)
    target_jump = np.abs(target_delta)
    action_range = np.maximum(stats["actions_max"] - stats["actions_min"], 1e-6)
    normalized_pred_jump = pred_jump / action_range
    normalized_target_jump = target_jump / action_range
    jump_score = float(np.mean(pred_jump))
    normalized_jump_score = float(np.mean(normalized_pred_jump))
    gt_jump_score = float(np.mean(target_jump))
    normalized_gt_jump_score = float(np.mean(normalized_target_jump))
    jump_excess_ratio = jump_score / max(gt_jump_score, 1e-6)
    max_normalized_jump = float(np.max(normalized_pred_jump))
    gripper_reverse = bool(direction[5] is not None and direction[5] < 0.4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{args.episode:03d}_ensembled_stride_{args.query_stride}_decay_{args.ensemble_decay}"
    np.savez_compressed(
        args.output_dir / f"{stem}.npz",
        predicted_actions=predicted,
        ground_truth_actions=target,
        absolute_error=np.abs(error),
        query_frames=np.asarray(query_frames),
        joint_order=np.asarray(names),
    )
    summary = {
        "checkpoint": args.checkpoint,
        "mode": "temporal_ensembling",
        "episode": args.episode,
        "episode_length": episode_length,
        "query_stride": args.query_stride,
        "chunk_size": args.chunk_size,
        "num_queries": len(query_frames),
        "num_denoising_steps": args.num_denoising_steps,
        "ensemble_decay": args.ensemble_decay,
        "joint_order": names,
        "per_joint_mae": dict(zip(names, mae.tolist(), strict=True)),
        "per_joint_rmse": dict(zip(names, rmse.tolist(), strict=True)),
        "per_joint_direction_agreement": dict(zip(names, direction, strict=True)),
        "per_joint_scale_ratio": dict(zip(names, scale_ratio.tolist(), strict=True)),
        "full_episode_mae": float(np.mean(np.abs(error))),
        "action_jump_score": jump_score,
        "normalized_action_jump_score": normalized_jump_score,
        "ground_truth_action_jump_score": gt_jump_score,
        "normalized_ground_truth_action_jump_score": normalized_gt_jump_score,
        "action_jump_excess_ratio": jump_excess_ratio,
        "max_normalized_action_jump": max_normalized_jump,
        "gripper_mae": float(mae[5]),
        "gripper_direction_agreement": direction[5],
        "obvious_reverse_gripper": gripper_reverse,
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(3, 2, figsize=(18, 13), sharex=True)
    steps = np.arange(episode_length)
    for dim, axis in enumerate(axes.flat):
        axis.plot(steps, target[:, dim], label="ground truth", linewidth=1.8)
        axis.plot(steps, predicted[:, dim], label="predicted (temporal-ensembled)", linewidth=1.2)
        axis.plot(steps, np.abs(error[:, dim]), label="absolute error", linewidth=0.9, alpha=0.75)
        axis.set_title(f"{dim}: {names[dim]} | MAE={mae[dim]:.3f}, RMSE={rmse[dim]:.3f}")
        axis.set_ylabel("degree" if dim < 5 else "gripper range")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("episode frame")
    axes[-1, 1].set_xlabel("episode frame")
    fig.suptitle(
        f"SO101 full episode {args.episode} (temporal ensembling) | checkpoint={Path(args.checkpoint).name} | "
        f"query stride={args.query_stride}, chunk={args.chunk_size}, decay={args.ensemble_decay}, "
        f"denoise={args.num_denoising_steps}"
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

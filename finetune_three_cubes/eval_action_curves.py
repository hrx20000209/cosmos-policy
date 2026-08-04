"""阶段 3.2：离线动作对比图（GT vs 生成的 action chunk），单个 checkpoint。

- 用固定的 (episode, start_frame) 对（constants 里硬编码），固定随机种子 → 固定初始噪声，
  保证不同 checkpoint 之间可比（差异只来自模型进步，不来自采样抖动）。
- 每个 action 维度一个 subplot，横轴=chunk 内时间步，GT 实线 / 预测虚线。
- 画在**反归一化后的物理量纲**上（关节角 deg / gripper %）。
- 标题写清 checkpoint step、episode id、起始帧。
- 逐维 MAE/RMSE + 全维平均，写入 <out>/step_XXXXXX/metrics.json。

复用仓库推理链路 get_model / get_action（与 eval_so101_action_curves.py 相同）。
动作是关节角（joint-space），所以不给末端位姿 mm 误差。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constants import (
    ACTION_NAMES,
    ACTION_UNITS,
    COLOR_GT,
    COLOR_PRED,
    FIXED_COMPARISON_EPISODES,
    FIXED_COMPARISON_START_FRAMES,
    REPO_ID,
    STATS_PATH,
    T5_EMBEDDINGS_PATH,
    DATA_ROOT,
)

from cosmos_policy.datasets.so101_lerobot_dataset import SO101LeRobotCosmosDataset
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.scripts.eval_so101_action_curves import SO101OfflineEvalConfig, _to_uint8


def _step_of(ckpt: str) -> int:
    m = re.search(r"iter_(\d+)", ckpt)
    return int(m.group(1)) if m else -1


def _find_index(dataset: SO101LeRobotCosmosDataset, episode: int, frame: int) -> int:
    eps = np.asarray(dataset.dataset.hf_dataset["episode_index"], dtype=np.int64)
    frs = np.asarray(dataset.dataset.hf_dataset["frame_index"], dtype=np.int64)
    hits = np.flatnonzero((eps == episode) & (frs == frame))
    if len(hits) == 0:
        # 退回到该 episode 内最接近的帧
        cand = np.flatnonzero(eps == episode)
        if len(cand) == 0:
            raise ValueError(f"数据集中找不到 episode {episode}")
        nearest = cand[np.argmin(np.abs(frs[cand] - frame))]
        return int(nearest)
    return int(hits[0])


def _plot(out_png: Path, pred: np.ndarray, gt: np.ndarray, step: int, episode: int, frame: int) -> dict:
    steps = np.arange(len(pred))
    err = np.abs(pred - gt)
    mae = err.mean(axis=0)
    rmse = np.sqrt(((pred - gt) ** 2).mean(axis=0))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for d, ax in enumerate(axes.flat):
        ax.plot(steps, gt[:, d], color=COLOR_GT, lw=2.0, ls="-", label="ground truth")
        ax.plot(steps, pred[:, d], color=COLOR_PRED, lw=1.8, ls="--", label="predicted")
        ax.set_title(f"{ACTION_NAMES[d]} [{ACTION_UNITS[d]}]  MAE={mae[d]:.2f} RMSE={rmse[d]:.2f}", fontsize=10)
        ax.grid(alpha=0.25)
        if d == 0:
            ax.legend(fontsize=9, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("chunk step")
    fig.suptitle(
        f"Predicted vs GT action  |  step={step}  episode={episode}  start_frame={frame}  "
        f"|  overall MAE={mae.mean():.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {
        "per_joint_mae": {ACTION_NAMES[d]: float(mae[d]) for d in range(6)},
        "per_joint_rmse": {ACTION_NAMES[d]: float(rmse[d]) for d in range(6)},
        "overall_mae": float(mae.mean()),
        "overall_rmse": float(rmse.mean()),
        "gripper_mae": float(mae[5]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default=DATA_ROOT)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--t5_text_embeddings_path", default=T5_EMBEDDINGS_PATH)
    p.add_argument("--dataset_stats_path", default=STATS_PATH)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--num_denoising_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--chunk_size", type=int, default=50)
    args = p.parse_args()

    step = _step_of(args.checkpoint)
    step_dir = args.output_dir / f"step_{step:09d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    dataset = SO101LeRobotCosmosDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=FIXED_COMPARISON_EPISODES,
        chunk_size=args.chunk_size,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        dataset_stats_path=args.dataset_stats_path,
        use_image_aug=False,
        use_stronger_image_aug=False,
    )

    cfg = SO101OfflineEvalConfig(
        ckpt_path=args.checkpoint,
        chunk_size=args.chunk_size,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        num_denoising_steps_action=args.num_denoising_steps,
        seed=args.seed,
    )
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(cfg)

    pairs = list(zip(FIXED_COMPARISON_EPISODES, FIXED_COMPARISON_START_FRAMES))
    all_metrics = []
    for ordinal, (episode, frame) in enumerate(pairs):
        index = _find_index(dataset, episode, frame)
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
            seed=args.seed + ordinal,  # 固定 → 固定初始噪声，跨 checkpoint 可比
            randomize_seed=False,
            num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
        pred = np.asarray(result["actions"], dtype=np.float32)          # 已反归一化的物理量
        gt = sample["physical_actions"].numpy().astype(np.float32)
        real_frame = int(sample["frame_index"])
        stem = f"ep{episode}_frame{real_frame}"
        m = _plot(step_dir / f"{stem}.png", pred, gt, step, episode, real_frame)
        m.update({"episode": episode, "start_frame": real_frame, "dataset_index": index})
        all_metrics.append(m)
        np.savez_compressed(
            step_dir / f"{stem}.npz",
            predicted=pred, ground_truth=gt, joint_order=np.asarray(ACTION_NAMES),
        )
        print(f"[step {step}] ep{episode} frame{real_frame}: overall MAE={m['overall_mae']:.3f} gripper MAE={m['gripper_mae']:.3f}")

    overall = float(np.mean([m["overall_mae"] for m in all_metrics]))
    summary = {"step": step, "checkpoint": args.checkpoint, "overall_mae": overall, "samples": all_metrics}
    (step_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[step {step}] mean overall MAE over {len(pairs)} samples = {overall:.4f}")
    print(f"saved: {step_dir}")


if __name__ == "__main__":
    main()

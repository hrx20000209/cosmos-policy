"""阶段 1.4 数据验证：从真实 DataLoader 取一个 batch，肉眼 + 数值双重确认数据正确。

产物：
  - <out>/data_validation_frames.png  : 一个样本的 6 张关键帧（current/future × primary/wristL/wristR）
  - <out>/data_validation_actions.png : 每个 action 维度一个 subplot，反归一化到物理量纲的 GT 曲线
  - stdout                            : 每个 tensor 的 shape/dtype/range、NaN/Inf 检查、denorm 一致性

只使用固定的对比 val episode（constants.FIXED_COMPARISON_EPISODES），与后续动作对比图同源。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import (
    ACTION_NAMES,
    ACTION_UNITS,
    CHUNK_SIZE,
    FIXED_COMPARISON_EPISODES,
    OKABE_ITO,
    REPO_ID,
    STATS_PATH,
    T5_EMBEDDINGS_PATH,
    DATA_ROOT,
)
from norm_utils import denormalize_actions, load_stats

from cosmos_policy.datasets.so101_lerobot_dataset import SO101LeRobotCosmosDataset

# 41-frame duplicated video 中各语义帧的时间索引（slot: 1blank+4*10 duplicate）。
FRAME_INDEX = {
    "current primary": 13,
    "current wrist-L": 5,
    "current wrist-R": 9,
    "future primary": 33,
    "future wrist-L": 25,
    "future wrist-R": 29,
}


def _frame_uint8(video: torch.Tensor, t: int) -> np.ndarray:
    img = video[:, t].permute(1, 2, 0).cpu().numpy()
    if img.dtype != np.uint8:
        img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return img


def _describe(name: str, x: torch.Tensor) -> None:
    if not torch.is_tensor(x):
        print(f"  {name:32s} (非 tensor) = {x}")
        return
    xf = x.float()
    n_nan = int(torch.isnan(xf).sum())
    n_inf = int(torch.isinf(xf).sum())
    flag = "  <== NaN/Inf!" if (n_nan or n_inf) else ""
    print(
        f"  {name:32s} shape={tuple(x.shape)!s:22s} dtype={str(x.dtype):14s} "
        f"min={xf.min():+9.3f} max={xf.max():+9.3f} mean={xf.mean():+9.3f} "
        f"nan={n_nan} inf={n_inf}{flag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DATA_ROOT)
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--t5_text_embeddings_path", default=T5_EMBEDDINGS_PATH)
    parser.add_argument("--dataset_stats_path", default=STATS_PATH)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--video_backend", default="pyav")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.dataset_stats_path)

    # 只用固定对比 val episodes，clean 设置（无增强），与训练 val dataset 语义一致。
    dataset = SO101LeRobotCosmosDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=FIXED_COMPARISON_EPISODES,
        chunk_size=args.chunk_size,
        final_image_size=224,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        dataset_stats_path=args.dataset_stats_path,
        normalize_actions=True,
        normalize_proprio=True,
        action_mode="absolute",
        use_proprio=True,
        use_image_aug=False,
        use_stronger_image_aug=False,
        num_duplicates_per_image=4,
        return_value_function_returns=False,
        video_backend=args.video_backend,
    )
    print(f"\ndataset length (固定对比 episodes {FIXED_COMPARISON_EPISODES}) = {len(dataset)}")

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    batch = next(iter(loader))

    print("\n=== batch tensor 概览（经过 DataLoader 默认 collate） ===")
    for key in sorted(batch.keys()):
        _describe(key, batch[key])

    # denorm 一致性：denorm(batch['actions']) 应约等于 batch['physical_actions']
    norm_actions = batch["actions"].cpu().numpy()
    phys_actions = batch["physical_actions"].cpu().numpy()
    recon = np.stack([denormalize_actions(norm_actions[i], stats) for i in range(norm_actions.shape[0])])
    max_err = float(np.max(np.abs(recon - phys_actions)))
    print(f"\ndenorm(actions) vs physical_actions 最大绝对误差 = {max_err:.6f}  ({'PASS' if max_err < 1e-2 else 'FAIL'})")

    print("\n=== batch 内每个样本的 episode / frame / 归一化范围 ===")
    for i in range(batch["actions"].shape[0]):
        ep = int(batch["episode_index"][i])
        fr = int(batch["frame_index"][i])
        nmin = norm_actions[i].min(0)
        nmax = norm_actions[i].max(0)
        print(
            f"  sample {i}: episode={ep} frame={fr} "
            f"norm_action min={np.round(nmin, 2).tolist()} max={np.round(nmax, 2).tolist()}"
        )
        assert -1.05 <= nmin.min() and nmax.max() <= 1.05, "归一化动作超出 [-1,1] 太多"

    # --- 图 1：关键帧网格（取 batch 里第 0 个样本） ---
    sample_video = batch["video"][0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (name, t) in zip(axes.flat, FRAME_INDEX.items()):
        ax.imshow(_frame_uint8(sample_video, t))
        ax.set_title(name, fontsize=11)
        ax.axis("off")
    ep0, fr0 = int(batch["episode_index"][0]), int(batch["frame_index"][0])
    fig.suptitle(f"three_cubes_1 sample frames — episode {ep0}, frame {fr0}", fontsize=13)
    fig.tight_layout()
    frames_path = args.output_dir / "data_validation_frames.png"
    fig.savefig(frames_path, dpi=130)
    plt.close(fig)

    # --- 图 2：每个 action 维度一个 subplot，反归一化物理量纲，叠加 batch 里各样本 ---
    steps = np.arange(args.chunk_size)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for d, ax in enumerate(axes.flat):
        for i in range(min(batch["actions"].shape[0], len(OKABE_ITO))):
            ep = int(batch["episode_index"][i])
            fr = int(batch["frame_index"][i])
            ax.plot(steps, recon[i, :, d], color=OKABE_ITO[i], lw=1.6, label=f"ep{ep} f{fr}")
        ax.set_title(f"{ACTION_NAMES[d]}  [{ACTION_UNITS[d]}]", fontsize=11)
        ax.axhline(float(stats["actions_min"][d]), color="0.7", ls=":", lw=0.8)
        ax.axhline(float(stats["actions_max"][d]), color="0.7", ls=":", lw=0.8)
        ax.grid(alpha=0.25)
        if d == 0:
            ax.legend(fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("chunk step")
    fig.suptitle("Ground-truth action chunk (denormalized to physical units)", fontsize=13)
    fig.tight_layout()
    actions_path = args.output_dir / "data_validation_actions.png"
    fig.savefig(actions_path, dpi=130)
    plt.close(fig)

    print(f"\n已保存: {frames_path}")
    print(f"已保存: {actions_path}")


if __name__ == "__main__":
    main()

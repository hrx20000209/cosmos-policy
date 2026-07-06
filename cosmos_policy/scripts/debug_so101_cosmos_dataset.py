"""调试 SO101LeRobotCosmosDataset 的时间、维度、归一化和 11-slot 图像布局。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cosmos_policy.datasets.so101_lerobot_dataset import LATENT_INDICES, SO101LeRobotCosmosDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--t5_text_embeddings_path", required=True)
    parser.add_argument("--dataset_stats_path", required=True)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--camera_map_json")
    parser.add_argument("--output", type=Path, default=Path("so101_dataset_debug.png"))
    parser.add_argument("--video_backend", default="pyav")
    return parser.parse_args()


def _image_at(video, temporal_index: int) -> np.ndarray:
    image = video[:, temporal_index].permute(1, 2, 0).cpu().numpy()
    if image.dtype != np.uint8:
        image = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return image


def main() -> None:
    args = parse_args()
    camera_map = json.loads(args.camera_map_json) if args.camera_map_json else None
    dataset = SO101LeRobotCosmosDataset(
        repo_id=args.repo_id,
        root=args.root,
        chunk_size=args.chunk_size,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        dataset_stats_path=args.dataset_stats_path,
        camera_map=camera_map,
        use_image_aug=False,
        use_stronger_image_aug=False,
        video_backend=args.video_backend,
    )
    episode_ids = np.asarray(dataset.dataset.hf_dataset["episode_index"], dtype=np.int64)
    first_episode = int(episode_ids[0])
    episode_local_indices = np.flatnonzero(episode_ids == first_episode)
    indices = [
        int(episode_local_indices[0]),
        int(episode_local_indices[len(episode_local_indices) // 2]),
        int(episode_local_indices[-1]),
    ]
    labels = ("episode beginning", "episode middle", "near episode end")
    print(
        f"dataset length={len(dataset)}, fps={dataset.fps}, camera keys={dataset.camera_map}, "
        f"action dim={dataset.action_dim}, proprio dim={dataset.proprio_dim}"
    )
    stats = dataset.dataset_stats
    action_names = stats.get("action_names") or [f"action[{index}]" for index in range(dataset.action_dim)]
    print(f"action key={dataset.action_key!r}, action mode={dataset.action_mode!r}")
    print(f"joint order={action_names}")
    print(f"dataset action range min={stats['actions_min'].tolist()}")
    print(f"dataset action range max={stats['actions_max'].tolist()}")
    gripper_indices = [index for index, name in enumerate(action_names) if "gripper" in name.lower()]
    if len(gripper_indices) != 1:
        raise RuntimeError(f"无法唯一确定 gripper 维度：action_names={action_names}")
    gripper_index = gripper_indices[0]
    print(
        f"gripper index={gripper_index}, name={action_names[gripper_index]!r}, "
        f"dataset range=[{stats['actions_min'][gripper_index]}, {stats['actions_max'][gripper_index]}]"
    )

    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    temporal_indices = (13, 5, 9, 33, 25, 29)
    image_names = ("current primary", "current wrist L", "current wrist R", "future primary", "future wrist L", "future wrist R")
    for row, (label, index) in enumerate(zip(labels, indices, strict=True)):
        sample = dataset[index]
        print(f"\n[{label}] local_index={index}, episode={sample['episode_index']}, frame={sample['frame_index']}")
        print(f"video={tuple(sample['video'].shape)}, action={tuple(sample['actions'].shape)}")
        print(
            f"proprio={tuple(sample['proprio'].shape)}, future_proprio={tuple(sample['future_proprio'].shape)}, "
            f"command={sample['command']!r}"
        )
        print(f"latent indices={LATENT_INDICES}")
        physical = sample["physical_actions"].numpy()
        normalized = sample["actions"].numpy()
        print(
            f"action physical min/max={physical.min(axis=0).tolist()} / {physical.max(axis=0).tolist()}\n"
            f"action normalized min/max={normalized.min(axis=0).tolist()} / {normalized.max(axis=0).tolist()}"
        )
        for col, (time_index, name) in enumerate(zip(temporal_indices, image_names, strict=True)):
            axes[row, col].imshow(_image_at(sample["video"], time_index))
            axes[row, col].set_title(f"{label}\n{name}")
            axes[row, col].axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(f"\n已保存 dataset 可视化: {args.output}")


if __name__ == "__main__":
    main()

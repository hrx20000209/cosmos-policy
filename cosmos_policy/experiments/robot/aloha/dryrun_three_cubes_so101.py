"""Decode one real batch and validate the SO101/Cosmos tensor contract."""

import argparse
import json

import numpy as np
from torch.utils.data import DataLoader

from cosmos_policy.datasets.lerobot_so101_dataset import LeRobotSO101Dataset
from cosmos_policy.experiments.robot.aloha.so101_schema import load_schema, print_schema_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/data/rxhuang/three_cubes_1")
    parser.add_argument("--overfit-episodes", type=int, default=1)
    parser.add_argument("--t5-embeddings-path", default="")
    args = parser.parse_args()
    dataset = LeRobotSO101Dataset(
        data_dir=args.dataset_root,
        overfit_num_episodes=args.overfit_episodes,
        use_image_aug=False,
        t5_text_embeddings_path=args.t5_embeddings_path,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    expected = {"video": (1, 3, 41, 224, 224), "actions": (1, 30, 6), "proprio": (1, 6)}
    actual = {key: tuple(batch[key].shape) for key in expected}
    if actual != expected:
        raise ValueError(f"Cosmos batch contract mismatch: expected={expected}, actual={actual}")
    absolute = batch["absolute_actions"].numpy()
    normalized = batch["actions"].numpy()
    print_schema_summary(load_schema())
    print(
        json.dumps(
            {
                "batch_shapes": actual,
                "absolute_action_range": [float(absolute.min()), float(absolute.max())],
                "normalized_action_range": [float(normalized.min()), float(normalized.max())],
                "action_latent_idx": batch["action_latent_idx"].tolist(),
                "current_camera_latent_indices": [
                    batch["current_wrist_image_latent_idx"].item(),
                    batch["current_wrist_image2_latent_idx"].item(),
                    batch["current_image_latent_idx"].item(),
                ],
            },
            indent=2,
        )
    )
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized action")
    print("DRYRUN PASSED: metadata, video decode, normalization, batch, and latent layout are aligned.")


if __name__ == "__main__":
    main()

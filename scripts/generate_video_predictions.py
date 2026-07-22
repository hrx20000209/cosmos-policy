#!/usr/bin/env python3
"""Run Cosmos world-model inference and create decoded future-frame videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cosmos_policy.datasets.so101_lerobot_dataset import SO101LeRobotCosmosDataset
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.scripts.eval_so101_action_curves import SO101OfflineEvalConfig, _to_uint8


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("no frames to write")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.putText(result, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t5-text-embeddings-path", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--episode", type=int, default=50)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--query-stride", type=int, default=60)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--num-denoising-steps", type=int, default=5)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = SO101LeRobotCosmosDataset(
        repo_id=args.repo_id, root=args.root, chunk_size=args.chunk_size,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        dataset_stats_path=args.dataset_stats_path,
        use_image_aug=False, use_stronger_image_aug=False,
    )
    episode_ids = np.asarray(dataset.dataset.hf_dataset["episode_index"], dtype=np.int64)
    indices = np.flatnonzero(episode_ids == args.episode)
    if not len(indices):
        raise ValueError(f"episode {args.episode} not found")

    cfg = SO101OfflineEvalConfig(
        ckpt_path=args.checkpoint, chunk_size=args.chunk_size,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        num_denoising_steps_action=args.num_denoising_steps, seed=args.seed,
        trained_with_image_aug=True,
    )
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = get_model(cfg)
    if int(cosmos_config.dataloader_train.dataset.chunk_size) != args.chunk_size:
        raise RuntimeError("checkpoint and requested chunk sizes differ")

    # Dataset temporal locations for unique slots 7/8/9 after 4x repetition.
    camera_spec = {
        "wrist_left": ("future_wrist_image", 25),
        "wrist_right": ("future_wrist_image2", 29),
        "primary": ("future_image", 33),
    }
    predicted: dict[str, list[np.ndarray]] = {name: [] for name in camera_spec}
    targets: dict[str, list[np.ndarray]] = {name: [] for name in camera_spec}
    query_indices = indices[:: args.query_stride][: args.num_queries]
    for ordinal, index in enumerate(query_indices):
        sample = dataset[int(index)]
        observation = {
            "primary_image": _to_uint8(sample["video"], 13),
            "left_wrist_image": _to_uint8(sample["video"], 5),
            "right_wrist_image": _to_uint8(sample["video"], 9),
            "proprio": sample["physical_proprio"].numpy(),
        }
        result = get_action(
            cfg, model, stats, observation, sample["command"], seed=args.seed + ordinal,
            randomize_seed=False, num_denoising_steps_action=args.num_denoising_steps,
            generate_future_state_and_value_in_parallel=True,
        )
        images = result.get("future_image_predictions", {})
        for camera, (prediction_key, gt_index) in camera_spec.items():
            if prediction_key not in images:
                raise KeyError(f"model omitted {prediction_key}; keys={list(images)}")
            predicted[camera].append(np.asarray(images[prediction_key], dtype=np.uint8))
            targets[camera].append(_to_uint8(sample["video"], gt_index))
        print(f"query={ordinal + 1}/{len(query_indices)} frame={int(sample['frame_index'])}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_comparisons: list[np.ndarray] = []
    for camera in camera_spec:
        pred_frames, gt_frames = predicted[camera], targets[camera]
        labeled_pred = [_label(frame, f"prediction step={args.step} ep={args.episode} {camera}") for frame in pred_frames]
        comparisons = [
            np.hstack((_label(gt, f"ground truth {camera}"), pred))
            for gt, pred in zip(gt_frames, labeled_pred, strict=True)
        ]
        _write_video(args.output_dir / f"step_{args.step}_prediction_{camera}.mp4", labeled_pred, args.fps)
        _write_video(args.output_dir / f"step_{args.step}_comparison_{camera}.mp4", comparisons, args.fps)
        all_comparisons.extend(comparisons[:2])

    # Primary-camera aliases requested by the task plus a compact contact sheet.
    _write_video(args.output_dir / f"step_{args.step}_prediction.mp4", predicted["primary"], args.fps)
    primary_comparison = [np.hstack(pair) for pair in zip(targets["primary"], predicted["primary"], strict=True)]
    _write_video(args.output_dir / f"step_{args.step}_comparison.mp4", primary_comparison, args.fps)
    thumb = [cv2.resize(frame, (640, 224)) for frame in all_comparisons]
    sheet = np.vstack(thumb)
    cv2.imwrite(str(args.output_dir / f"step_{args.step}_frames.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()

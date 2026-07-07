"""沿训练 denoise 路径评估 SO101 action latent 的 teacher-forced 重建。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from cosmos_policy._src.imaginaire.utils import misc
from cosmos_policy.datasets.so101_lerobot_dataset import SO101LeRobotCosmosDataset
from cosmos_policy.experiments.robot.cosmos_utils import (
    extract_action_chunk_from_latent_sequence,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    unnormalize_actions,
)
from cosmos_policy.scripts.eval_so101_action_curves import (
    SO101OfflineEvalConfig,
    _metric_dict,
    _save_plot,
    _select_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t5_text_embeddings_path", required=True)
    parser.add_argument("--dataset_stats_path", required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--low_sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _model_kwargs(batch: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "action_chunk": batch["actions"],
        "action_indices": batch["action_latent_idx"],
        "proprio": batch["proprio"],
        "current_proprio_indices": batch["current_proprio_latent_idx"],
        "future_proprio": batch["future_proprio"],
        "future_proprio_indices": batch["future_proprio_latent_idx"],
        "future_wrist_image_indices": batch["future_wrist_image_latent_idx"],
        "future_wrist_image2_indices": batch.get("future_wrist_image2_latent_idx"),
        "future_image_indices": batch["future_image_latent_idx"],
        "future_image2_indices": batch.get("future_image2_latent_idx"),
        "rollout_data_mask": batch["rollout_data_mask"],
        "world_model_sample_mask": batch["world_model_sample_mask"],
        "value_function_sample_mask": batch["value_function_sample_mask"],
        "value_function_return": batch["value_function_return"],
        "value_indices": batch["value_latent_idx"],
    }


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    sample: dict[str, object],
    sigma_mode: str,
    low_sigma: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch = misc.to(default_collate([sample]), device="cuda")
    _, x0, condition = model.get_data_and_condition(batch)
    if sigma_mode == "random_training_sigma":
        sigma, epsilon = model.draw_training_sigma_and_epsilon(x0.size(), condition)
    else:
        sigma = torch.full((x0.shape[0], 1), low_sigma, device=x0.device, dtype=torch.float32)
        epsilon = torch.randn_like(x0)
    # 训练由 mixed-precision/FSDP 负责 dtype 对齐；离线直接调用训练路径时需显式复现 bf16 autocast。
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, _, _, _ = model.compute_loss_with_epsilon_and_sigma(
            x0,
            condition,
            epsilon,
            sigma,
            **_model_kwargs(batch),
        )
    prediction = extract_action_chunk_from_latent_sequence(
        output["model_pred"].x0,
        action_shape=(batch["actions"].shape[1], batch["actions"].shape[2]),
        action_indices=batch["action_latent_idx"],
    )
    return prediction.float().cpu().numpy(), float(sigma.flatten()[0])


def main() -> None:
    args = parse_args()
    if not 0 < args.low_sigma <= 0.1:
        raise ValueError("low_sigma 必须在 (0, 0.1]，避免把高噪声误称为 low-sigma")
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
            f"SO101 action schema 不确定，停止 teacher-forced eval：names={names}, "
            f"dim={dataset.action_dim}, mode={dataset.action_mode}"
        )

    cfg = SO101OfflineEvalConfig(
        ckpt_path=args.checkpoint,
        chunk_size=args.chunk_size,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_text_embeddings_path,
        seed=args.seed,
    )
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = get_model(cfg)
    if int(cosmos_config.dataloader_train.dataset.chunk_size) != args.chunk_size:
        raise RuntimeError("checkpoint config 的 chunk_size 与 teacher-forced 参数不一致")
    print(f"action key='action', dim=6, mode='absolute', joint order={names}")
    print(f"action range min={stats['actions_min'].tolist()}, max={stats['actions_max'].tolist()}")
    print(f"gripper range=[{stats['actions_min'][5]}, {stats['actions_max'][5]}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices = _select_indices(dataset, args.num_samples)
    summaries: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "action_key": "action",
        "action_dim": 6,
        "action_mode": "absolute",
        "joint_order": names,
        "action_range_min": stats["actions_min"].tolist(),
        "action_range_max": stats["actions_max"].tolist(),
        "low_sigma": args.low_sigma,
    }
    for mode in ("low_sigma", "random_training_sigma"):
        predicted_all = []
        target_all = []
        sigma_values = []
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for ordinal, index in enumerate(indices):
            sample = dataset[index]
            predicted_normalized, sigma = _predict(
                model,
                sample,
                mode,
                args.low_sigma,
                args.seed + ordinal,
            )
            predicted = unnormalize_actions(predicted_normalized, stats)[0]
            target = sample["physical_actions"].numpy().astype(np.float32)
            predicted_all.append(predicted)
            target_all.append(target)
            sigma_values.append(sigma)
            episode = int(sample["episode_index"])
            frame = int(sample["frame_index"])
            stem = f"sample_{ordinal:02d}_episode_{episode}_frame_{frame}"
            _save_plot(mode_dir / f"{stem}.png", predicted, target, names, episode, frame)
            np.savez_compressed(
                mode_dir / f"{stem}.npz",
                predicted_actions=predicted,
                ground_truth_actions=target,
                predicted_normalized_actions=predicted_normalized[0],
                sigma=sigma,
                joint_order=np.asarray(names),
            )
        predicted_array = np.stack(predicted_all)
        target_array = np.stack(target_all)
        error = predicted_array - target_array
        mae = np.mean(np.abs(error), axis=(0, 1))
        rmse = np.sqrt(np.mean(error**2, axis=(0, 1)))
        result = {
            "num_samples": len(indices),
            "sigma_values": sigma_values,
            "mae": float(np.mean(np.abs(error))),
            "per_joint_mae": _metric_dict(names, mae),
            "per_joint_rmse": _metric_dict(names, rmse),
        }
        summaries[mode] = result
        (mode_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{mode}: physical MAE={result['mae']:.6f}")
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()

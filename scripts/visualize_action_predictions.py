#!/usr/bin/env python3
"""Create standard chunk/full-episode action diagnostics from Cosmos inference NPZ.

The input may use either the generic keys (predicted/ground_truth/action_names)
or the native Cosmos offline-evaluation keys
(predicted_actions/ground_truth_actions/joint_order).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _array(data: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    raise KeyError(f"none of {keys} present; available={data.files}")


def _lag(pred: np.ndarray, target: np.ndarray) -> int | None:
    """Return cross-correlation peak lag (prediction relative to GT)."""
    if np.std(pred) == 0 or np.std(target) == 0:
        return None
    p = pred - pred.mean()
    t = target - target.mean()
    return int(np.argmax(np.correlate(p, t, mode="full")) - (len(t) - 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--chunk-index", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.predictions)
    pred = _array(data, "predicted", "predicted_actions").astype(np.float32)
    gt = _array(data, "ground_truth", "ground_truth_actions").astype(np.float32)
    names = _array(data, "action_names", "joint_order").tolist() if any(
        key in data for key in ("action_names", "joint_order")
    ) else [f"action_{index}" for index in range(gt.shape[1])]
    if pred.shape != gt.shape or gt.ndim != 2:
        raise ValueError(f"expected matching [time,action_dim], got {pred.shape} and {gt.shape}")

    error = pred - gt
    pred_delta, gt_delta = np.diff(pred, axis=0), np.diff(gt, axis=0)
    moving = np.abs(gt_delta) > 1e-3
    metrics: dict[str, object] = {
        "global_step": args.step,
        "diagnostic_split": "training-set diagnostic (not validation/test)",
        "space": "original_physical_scale",
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_pearson": None,
        "per_dimension": {},
    }
    correlations: list[float] = []
    for dim, name in enumerate(names):
        corr = float(np.corrcoef(pred[:, dim], gt[:, dim])[0, 1]) if (
            np.std(pred[:, dim]) and np.std(gt[:, dim])
        ) else None
        if corr is not None and np.isfinite(corr):
            correlations.append(corr)
        direction = float(np.mean(np.sign(pred_delta[:, dim][moving[:, dim]]) == np.sign(gt_delta[:, dim][moving[:, dim]]))) if np.any(moving[:, dim]) else None
        metrics["per_dimension"][str(name)] = {
            "mse": float(np.mean(error[:, dim] ** 2)),
            "mae": float(np.mean(np.abs(error[:, dim]))),
            "rmse": float(np.sqrt(np.mean(error[:, dim] ** 2))),
            "pearson": corr,
            "direction_accuracy": direction,
            "cross_correlation_peak_lag_frames": _lag(pred[:, dim], gt[:, dim]),
            "predicted_std": float(np.std(pred[:, dim])),
            "ground_truth_std": float(np.std(gt[:, dim])),
        }
    metrics["mean_pearson"] = float(np.mean(correlations)) if correlations else None

    if "predicted_action_chunks" in data and "ground_truth_action_chunks" in data:
        pred_chunks = np.asarray(data["predicted_action_chunks"])
        gt_chunks = np.asarray(data["ground_truth_action_chunks"])
        horizon_mae = np.mean(np.abs(pred_chunks - gt_chunks), axis=(0, 2))
        metrics["chunk_horizon_mae"] = horizon_mae.tolist()
        metrics["chunk_first_5_mae"] = float(np.mean(horizon_mae[:5]))
        metrics["chunk_remaining_mae"] = float(np.mean(horizon_mae[5:])) if len(horizon_mae) > 5 else None
        index = min(max(args.chunk_index, 0), len(pred_chunks) - 1)
        chunk_pred, chunk_gt = pred_chunks[index], gt_chunks[index]
    else:
        chunk_pred, chunk_gt = pred[: min(30, len(pred))], gt[: min(30, len(gt))]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"step_{args.step}"
    np.savez_compressed(
        f"{prefix}_predictions.npz", predicted=pred, ground_truth=gt,
        absolute_error=np.abs(error), action_names=np.asarray(names),
    )
    Path(f"{prefix}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    def draw(path: Path, predicted: np.ndarray, target: np.ndarray, title: str) -> None:
        fig, axes = plt.subplots(len(names), 1, figsize=(14, max(8, 2.25 * len(names))), sharex=True, squeeze=False)
        x = np.arange(len(target))
        for dim, name in enumerate(names):
            ax = axes[dim, 0]
            ax.plot(x, target[:, dim], label="ground truth", linewidth=1.5)
            ax.plot(x, predicted[:, dim], label="prediction", linewidth=1.1)
            ax.fill_between(x, target[:, dim], predicted[:, dim], alpha=0.12, label="absolute error")
            ax.set_ylabel(str(name)); ax.grid(alpha=0.2)
        axes[0, 0].legend(ncol=3); axes[-1, 0].set_xlabel("frame / horizon step")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

    draw(Path(f"{prefix}_episode_comparison.png"), pred, gt, f"Training-set episode diagnostic | step {args.step}")
    draw(Path(f"{prefix}_chunk_comparison.png"), chunk_pred, chunk_gt, f"Training-set action chunk diagnostic | step {args.step}")


if __name__ == "__main__":
    main()

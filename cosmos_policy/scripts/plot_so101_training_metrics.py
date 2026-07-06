"""绘制 SO101 joint/action-only 训练指标及平滑趋势。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint_metrics", type=Path, required=True)
    parser.add_argument("--action_only_metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth_points", type=int, default=10)
    return parser.parse_args()


def _load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "train/loss" in record:
            records.append(record)
    if not records:
        raise ValueError(f"metrics 中没有训练记录：{path}")
    return records


def _plot(axis, records: list[dict], key: str, label: str, smooth: int) -> None:
    x = np.asarray([record["iteration"] for record in records])
    y = np.asarray([record[key] for record in records], dtype=np.float64)
    axis.plot(x, y, alpha=0.18, linewidth=0.7)
    if len(y) >= smooth:
        smoothed = np.convolve(y, np.ones(smooth) / smooth, mode="valid")
        axis.plot(x[smooth - 1 :], smoothed, label=label, linewidth=2)
    else:
        axis.plot(x, y, label=label, linewidth=2)


def main() -> None:
    args = parse_args()
    joint = _load(args.joint_metrics)
    action_only = _load(args.action_only_metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    _plot(axes[0, 0], joint, "train/loss", "joint total loss", args.smooth_points)
    _plot(axes[0, 0], action_only, "train/loss", "action-only total loss", args.smooth_points)
    axes[0, 0].set_title("Total loss (different masks; compare trends only)")

    _plot(
        axes[0, 1],
        joint,
        "train/demo_sample_action_l1_loss",
        "joint action L1",
        args.smooth_points,
    )
    _plot(
        axes[0, 1],
        action_only,
        "train/demo_sample_action_l1_loss",
        "action-only action L1",
        args.smooth_points,
    )
    axes[0, 1].set_title("Action L1 (directly comparable)")

    for key, label in (
        ("train/demo_sample_future_proprio_l1_loss", "future proprio L1"),
        ("train/demo_sample_future_wrist_image_l1_loss", "future wrist L1"),
        ("train/demo_sample_future_image_l1_loss", "future primary L1"),
    ):
        _plot(axes[1, 0], joint, key, label, args.smooth_points)
    axes[1, 0].set_title("Joint future-state losses")

    joint_action = np.asarray([row["train/demo_sample_action_l1_loss"] for row in joint])
    action_action = np.asarray([row["train/demo_sample_action_l1_loss"] for row in action_only])
    window = min(20, len(joint_action), len(action_action))
    labels = ["joint\nfirst", "joint\nlast", "action-only\nfirst", "action-only\nlast"]
    values = [
        joint_action[:window].mean(),
        joint_action[-window:].mean(),
        action_action[:window].mean(),
        action_action[-window:].mean(),
    ]
    axes[1, 1].bar(labels, values)
    axes[1, 1].set_title(f"Action L1: first/last {window} logged points")
    for index, value in enumerate(values):
        axes[1, 1].text(index, value, f"{value:.4f}", ha="center", va="bottom")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(loc="best") if axis.lines else None
        axis.set_xlabel("iteration")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"已保存：{args.output}")
    print(
        f"joint action L1: first={values[0]:.6f}, last={values[1]:.6f}; "
        f"action-only action L1: first={values[2]:.6f}, last={values[3]:.6f}"
    )


if __name__ == "__main__":
    main()

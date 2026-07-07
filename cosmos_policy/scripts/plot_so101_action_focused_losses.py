"""绘制多组 SO101 action-focused 实验的 total loss 与 action L1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=/path/to/metrics.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth_points", type=int, default=20)
    return parser.parse_args()


def _load(spec: str) -> tuple[str, list[dict[str, float]]]:
    label, path = spec.split("=", 1)
    rows = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if "train/loss" in row and "train/demo_sample_action_l1_loss" in row:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"{label} 没有可画的训练记录：{path}")
    return label, rows


def _plot(axis: plt.Axes, rows: list[dict[str, float]], key: str, label: str, smooth: int) -> None:
    steps = np.asarray([row["iteration"] for row in rows])
    values = np.asarray([row[key] for row in rows])
    axis.plot(steps, values, alpha=0.12, linewidth=0.5)
    window = min(smooth, len(values))
    averaged = np.convolve(values, np.ones(window) / window, mode="valid")
    axis.plot(steps[window - 1 :], averaged, linewidth=2, label=label)


def main() -> None:
    args = parse_args()
    runs = [_load(spec) for spec in args.run]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for label, rows in runs:
        _plot(axes[0], rows, "train/loss", label, args.smooth_points)
        _plot(axes[1], rows, "train/demo_sample_action_l1_loss", label, args.smooth_points)
    axes[0].set_title("Training total loss (loss masks/scales differ)")
    axes[1].set_title("Training action latent L1")
    for axis in axes:
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

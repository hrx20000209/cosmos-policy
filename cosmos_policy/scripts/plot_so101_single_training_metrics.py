"""Plot one SO101 Cosmos Policy training run from local metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SO101 train/val metrics for one run")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_step", type=int)
    parser.add_argument("--smooth_points", type=int, default=10)
    return parser.parse_args()


def _load(path: Path, max_step: int | None) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record.get("iteration", record.get("step", -1)))
        if max_step is not None and step > max_step:
            continue
        record["_step"] = step
        records.append(record)
    if not records:
        raise ValueError(f"No metrics found in {path}")
    return records


def _series(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [record for record in records if key in record and record["_step"] >= 0]
    if not rows:
        return np.asarray([]), np.asarray([])
    return (
        np.asarray([record["_step"] for record in rows], dtype=np.int64),
        np.asarray([record[key] for record in rows], dtype=np.float64),
    )


def _plot(axis, records: list[dict], key: str, label: str, smooth_points: int) -> bool:
    x, y = _series(records, key)
    if len(x) == 0:
        return False
    axis.plot(x, y, alpha=0.2, linewidth=0.8)
    if len(y) >= smooth_points:
        smoothed = np.convolve(y, np.ones(smooth_points) / smooth_points, mode="valid")
        axis.plot(x[smooth_points - 1 :], smoothed, linewidth=2.0, label=label)
    else:
        axis.plot(x, y, linewidth=1.6, label=label)
    return True


def main() -> None:
    args = parse_args()
    records = _load(args.metrics, args.max_step)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False)
    plotted = []
    plotted.append(_plot(axes[0, 0], records, "train/loss", "train/loss", args.smooth_points))
    plotted.append(_plot(axes[0, 0], records, "val/loss", "val/loss", args.smooth_points))
    axes[0, 0].set_title("Total loss")

    for key in (
        "train/demo_sample_action_l1_loss",
        "train/demo_sample_action_mse_loss",
        "val/demo_sample_action_l1_loss",
        "val/demo_sample_action_mse_loss",
    ):
        plotted.append(_plot(axes[0, 1], records, key, key, args.smooth_points))
    axes[0, 1].set_title("Action losses")

    for key in (
        "train/demo_sample_future_proprio_l1_loss",
        "train/demo_sample_future_wrist_image_l1_loss",
        "train/demo_sample_future_image_l1_loss",
        "val/demo_sample_future_proprio_l1_loss",
    ):
        plotted.append(_plot(axes[1, 0], records, key, key, args.smooth_points))
    axes[1, 0].set_title("Future-state losses")

    for key in ("train/lr", "train/grad_norm", "lr", "grad_norm"):
        plotted.append(_plot(axes[1, 1], records, key, key, args.smooth_points))
    axes[1, 1].set_title("LR / grad norm")

    if not any(plotted):
        raise ValueError(f"No known plot keys found in {args.metrics}")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("optimizer step")
        if axis.lines:
            axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()

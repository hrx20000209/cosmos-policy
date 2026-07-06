"""Plot the run-local Cosmos train/validation component losses."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.run_dir / "metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [
        "loss",
        "demo_sample_action_mse_loss",
        "demo_sample_action_l1_loss",
        "demo_sample_future_proprio_mse_loss",
        "demo_sample_future_wrist_image_mse_loss",
        "demo_sample_future_image_mse_loss",
        "demo_sample_value_mse_loss",
    ]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for metric, axis in zip(keys, axes.flat, strict=False):
        for split in ("train", "val"):
            prefix = f"{split}/"
            points = [(row["iteration"], row[prefix + metric]) for row in records if prefix + metric in row]
            if points:
                axis.plot([x for x, _ in points], [y for _, y in points], label=split)
        axis.set_title(metric)
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    axes.flat[-1].axis("off")
    fig.tight_layout()
    output = args.run_dir / "loss_curves.png"
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()

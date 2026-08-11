"""Create compact figures for the overnight feedback-repair report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    repair = json.loads(args.repair.read_text(encoding="utf-8"))
    persistent = json.loads(args.persistent.read_text(encoding="utf-8"))

    stages = (1, 2, 4)
    blocks = (8, 16, 24)
    groups = ("current_visual", "future_visual", "action", "all_dynamic_nonvalue")
    figure, axes = plt.subplots(1, len(groups), figsize=(14, 3.5), constrained_layout=True)
    for axis, group in zip(axes, groups, strict=True):
        matrix = np.asarray(
            [
                [repair["hidden_summary"][f"s{stage}_b{block}_{group}"]["median_recovery"] for block in blocks]
                for stage in stages
            ]
        )
        image = axis.imshow(matrix, vmin=-0.2, vmax=1.0, cmap="RdYlGn", aspect="auto")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{100 * matrix[row, column]:.1f}", ha="center", va="center", fontsize=8)
        axis.set_title(group.replace("_", " "))
        axis.set_xticks(range(len(blocks)), blocks)
        axis.set_yticks(range(len(stages)), stages)
        axis.set_xlabel("DiT block")
        axis.set_ylabel("denoiser forward")
    figure.colorbar(image, ax=axes, label="median fresh-action recovery", shrink=0.8)
    figure.savefig(args.output_dir / "oracle_2d_hidden_repair.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    split_styles = {"discovery": "o-", "validation": "s-", "heldout": "^-", "all": "D--"}
    for steps in (2, 4):
        for split, style in split_styles.items():
            arrivals = range(steps)
            recoveries = [
                persistent["summary"][f"d{steps}_arrival{arrival}_{split}"]["median_action_recovery"]
                for arrival in arrivals
            ]
            remaining = [(steps - arrival) / steps for arrival in arrivals]
            axis.plot(
                remaining,
                recoveries,
                style,
                label=f"{steps}-step {split}",
                alpha=1.0 if split == "all" else 0.55,
            )
    axis.axhline(0.9, color="black", linestyle=":", linewidth=1, label="90% recovery")
    axis.set_xlim(0.2, 1.05)
    axis.set_ylim(0.82, 1.01)
    axis.set_xlabel("fraction of denoiser forwards receiving fresh condition")
    axis.set_ylabel("median fresh-action recovery")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.savefig(args.output_dir / "persistent_condition_recovery.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()

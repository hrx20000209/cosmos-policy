#!/usr/bin/env python3
"""Audit normalization and temporal alignment after the episode-50 overfit test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = json.loads(args.stats.read_text())
    rows: list[dict[str, object]] = []
    for parquet_path in sorted((args.root / "data").rglob("*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=["episode_index", "frame_index", "timestamp", "action", "observation.state"],
        )
        for row in table.to_pylist():
            if int(row["episode_index"]) == args.episode:
                rows.append(row)
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows:
        raise RuntimeError(f"episode {args.episode} not found")

    actions = np.asarray([row["action"] for row in rows], dtype=np.float64)
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float64)
    action_min = np.asarray(stats["actions_min"], dtype=np.float64)
    action_max = np.asarray(stats["actions_max"], dtype=np.float64)
    normalized = 2.0 * (actions - action_min) / np.maximum(action_max - action_min, 1e-6) - 1.0
    camera_keys = sorted(
        key
        for key, value in json.loads((args.root / "meta" / "info.json").read_text())["features"].items()
        if value.get("dtype") in {"video", "image"}
    )

    starts = [0, len(rows) // 2, max(0, len(rows) - args.chunk_size - 1)]
    windows = []
    for start in starts:
        action_rows = rows[start : start + 5]
        future = rows[min(start + args.chunk_size, len(rows) - 1)]
        windows.append(
            {
                "observation_frame_index": int(rows[start]["frame_index"]),
                "observation_timestamp": float(rows[start]["timestamp"]),
                "camera_keys_at_observation": camera_keys,
                "first_five_actions": [
                    {
                        "frame_index": int(row["frame_index"]),
                        "timestamp": float(row["timestamp"]),
                        "action": [float(x) for x in row["action"]],
                    }
                    for row in action_rows
                ],
                "future_frame_index": int(future["frame_index"]),
                "future_timestamp": float(future["timestamp"]),
                "expected_future_offset_seconds": args.chunk_size / 30.0,
            }
        )

    result = {
        "episode": args.episode,
        "frames": len(rows),
        "fps": 30,
        "action_physical_mean": actions.mean(axis=0).tolist(),
        "action_physical_std": actions.std(axis=0).tolist(),
        "action_normalized_mean": normalized.mean(axis=0).tolist(),
        "action_normalized_std": normalized.std(axis=0).tolist(),
        "action_normalized_global_mean": float(normalized.mean()),
        "action_normalized_global_std": float(normalized.std()),
        "action_normalized_min": normalized.min(axis=0).tolist(),
        "action_normalized_max": normalized.max(axis=0).tolist(),
        "state_action_same_frame_mae": np.mean(np.abs(states - actions), axis=0).tolist(),
        "alignment_windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.metrics and args.plot:
        metrics = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
        steps = [item["step"] for item in metrics]
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(steps, [item["mae"] for item in metrics], marker="o")
        axes[0].axhline(metrics[0]["mae"] / 10.0, color="red", linestyle="--", label="10x reduction threshold")
        axes[0].set_ylabel("MAE (physical scale)")
        axes[0].legend()
        axes[0].grid(alpha=0.25)
        axes[1].plot(steps, [item["mean_pearson"] for item in metrics], marker="o")
        axes[1].axhline(0.9, color="red", linestyle="--", label="pass threshold")
        axes[1].set_xlabel("optimizer step")
        axes[1].set_ylabel("mean Pearson")
        axes[1].legend()
        axes[1].grid(alpha=0.25)
        fig.suptitle(f"Episode {args.episode} overfit — fixed 5-step open-loop diagnostic")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=160)
        plt.close(fig)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

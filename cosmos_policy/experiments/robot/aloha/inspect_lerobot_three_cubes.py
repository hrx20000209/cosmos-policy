"""Inspect and hard-validate a LeRobot v3 SO101 dataset without loading videos."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from cosmos_policy.experiments.robot.aloha.so101_schema import (
    load_schema,
    print_schema_summary,
    validate_dataset_metadata,
)


def stats(array: np.ndarray) -> dict[str, list[float]]:
    return {name: getattr(array, name)(axis=0).tolist() for name in ("min", "max", "mean", "std")}


def inspect_dataset(root: Path) -> dict:
    with (root / "meta/info.json").open() as f:
        info = json.load(f)
    schema = load_schema()
    validate_dataset_metadata(info, schema)

    parquet_files = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")
    table = pq.read_table(parquet_files)
    actions = np.asarray(table[schema["action_key"]].to_pylist(), dtype=np.float32)
    states = np.asarray(table[schema["state_key"]].to_pylist(), dtype=np.float32)
    episodes = np.asarray(table["episode_index"])
    episode_ids, episode_lengths = np.unique(episodes, return_counts=True)
    action_stats, state_stats = stats(actions), stats(states)

    print_schema_summary(schema, action_stats)
    print("\n=== LEROBOT DATASET ===")
    print(json.dumps(info, indent=2))
    print(f"state stats      : {json.dumps(state_stats)}")
    print(f"gripper action   : min={actions[:, -1].min():.6f} max={actions[:, -1].max():.6f}")
    print(f"episodes         : {dict(zip(episode_ids.tolist(), episode_lengths.tolist(), strict=True))}")
    for index in range(min(5, len(actions))):
        print(f"sample {index}: observation.state={states[index].shape}, action={actions[index].shape}")

    report = {
        "dataset_root": str(root),
        "features": info["features"],
        "camera_keys": schema["camera_keys"],
        "state_key": schema["state_key"],
        "action_key": schema["action_key"],
        "action_type": schema["action_type"],
        "joint_order": schema["joint_order"],
        "fps": info["fps"],
        "episode_lengths": dict(zip(episode_ids.tolist(), episode_lengths.tolist(), strict=True)),
        "action_stats": action_stats,
        "state_stats": state_stats,
        "gripper_range": [float(actions[:, -1].min()), float(actions[:, -1].max())],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = inspect_dataset(args.dataset_root)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()

"""导出 LeRobot SO101 数据集的 Cosmos Policy normalization stats。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cosmos_policy.datasets.so101_lerobot_dataset import _require_lerobot, compute_so101_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 SO101 action/proprio min-max stats")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--state_key", default="observation.state")
    parser.add_argument("--action_key", default="action")
    parser.add_argument("--action_mode", choices=("absolute", "delta", "delta_to_absolute"), default="absolute")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        episodes=args.episodes,
        download_videos=False,
    )
    stats = compute_so101_statistics(dataset, args.state_key, args.action_key, args.action_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(f"已保存 SO101 stats: {args.output}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

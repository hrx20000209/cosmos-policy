"""为 LeRobot metadata 中的全部 task 生成 Cosmos T5 embeddings。"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from cosmos_policy.datasets.so101_lerobot_dataset import _require_lerobot
from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 SO101 LeRobot task T5 embeddings")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, download_videos=False)
    tasks = sorted(str(task) for task in dataset.meta.tasks.index.tolist())
    if not tasks:
        raise ValueError("LeRobot metadata 中没有 task，无法生成 T5 embeddings")
    embeddings = generate_t5_embeddings(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(embeddings, file)
    print(f"已保存 {len(embeddings)} 个 SO101 task embeddings: {args.output}")


if __name__ == "__main__":
    main()

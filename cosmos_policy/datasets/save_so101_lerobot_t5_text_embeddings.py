"""为 LeRobot metadata 中的全部 task 生成 Cosmos T5 embeddings。"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from cosmos_policy.datasets.so101_lerobot_dataset import _require_lerobot
import torch

from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 SO101 LeRobot task T5 embeddings")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="google-t5/t5-11b")
    parser.add_argument(
        "--tokenizer-name",
        help="Tokenizer path when the encoder checkpoint and tokenizer are stored separately",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root, download_videos=False)
    tasks = sorted(str(task) for task in dataset.meta.tasks.index.tolist())
    if not tasks:
        raise ValueError("LeRobot metadata 中没有 task，无法生成 T5 embeddings")
    encoder = CosmosT5TextEncoder(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        device=args.device,
        local_files_only=Path(args.model_name).exists(),
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    )
    embeddings = {
        task: encoder.encode_prompts(task).to(dtype=torch.bfloat16).cpu()
        for task in tasks
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(embeddings, file)
    print(f"已保存 {len(embeddings)} 个 SO101 task embeddings: {args.output}")


if __name__ == "__main__":
    main()

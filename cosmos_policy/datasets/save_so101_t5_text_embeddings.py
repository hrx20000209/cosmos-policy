"""Generate the single task embedding used by the Three Cubes SO101 dataset."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings, save_embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/rxhuang/three_cubes_1")
    args = parser.parse_args()
    root = Path(args.data_dir)
    commands = pq.read_table(root / "meta/tasks.parquet")["task"].to_pylist()
    save_embeddings(generate_t5_embeddings(commands), str(root))


if __name__ == "__main__":
    main()

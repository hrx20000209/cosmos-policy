#!/usr/bin/env python3
"""Split a mixed manifest into one file per independent config."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[row["config_id"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for config_id, rows in sorted(groups.items()):
        path = args.output_dir / f"{config_id}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        index[config_id] = {"path": str(path), "episodes": len(rows)}
    (args.output_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()

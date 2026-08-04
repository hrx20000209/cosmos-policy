#!/usr/bin/env python3
"""Parallel helper for official PRO init generation.

It writes only category-disjoint init files. The authoritative prepare_assets
process subsequently re-hashes every file and writes the single audit JSON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()
    root = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
    audit = json.loads((root / "assets/libero_pro_asset_audit.json").read_text(encoding="utf-8"))
    rows = [row for row in audit if row["category"] == args.category]
    if len(rows) != 40:
        raise SystemExit(f"{args.category}: expected 40 rows, got {len(rows)}")
    script = Path(__file__).with_name("prepare_assets.py")
    spec = importlib.util.spec_from_file_location("prepare_assets_for_init_shard", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load prepare_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_init_states(
        Path("/data/rxhuang/repos/LIBERO-PRO"), root, rows, args.seed, overwrite=False
    )


if __name__ == "__main__":
    main()

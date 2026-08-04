#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero_harness import (
    apply_overrides,
    configure_repository_paths,
    load_yaml,
    run_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Cosmos/LingBot-VA LIBERO evaluation harness")
    parser.add_argument("--model", choices=["cosmos", "lingbot_va"], required=True)
    parser.add_argument("--task_suite", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--set", action="append", default=[], dest="overrides", help="dotted.key=value")
    parser.add_argument("--mock", action="store_true", help="infrastructure smoke test; never a model result")
    args = parser.parse_args()
    config = apply_overrides(load_yaml(args.config), args.overrides)
    config["model"]["name"] = args.model
    if args.task_suite:
        config["evaluation"]["task_suite"] = args.task_suite
    configure_repository_paths(config)
    output_dir = args.output_dir or config.get("output_dir", "outputs/wam_libero")
    run_dir = run_experiment(args.model, config, output_dir, args.mock)
    print(run_dir)


if __name__ == "__main__":
    main()

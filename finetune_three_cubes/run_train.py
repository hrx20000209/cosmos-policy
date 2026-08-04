"""three_cubes 全参微调训练入口（不修改上游）。

流程：把本项目的实验 + callbacks 注册进 Hydra ConfigStore，然后复用 cosmos_policy
自带的 load_config / launch。等价于 `cosmos_policy.scripts.train`，只是多注册了我们的
experiment=so101_three_cubes_full_ft。

用法（多卡 FSDP）：
  PYTHONPATH=finetune_three_cubes:.:~/Projects/lerobot/src \
  IMAGINAIRE_OUTPUT_ROOT=/data/rxhuang/cosmos_three_cubes_runs \
  torchrun --nproc_per_node=4 -m finetune_three_cubes.run_train \
    --config=cosmos_policy/config/config.py -- \
    experiment=so101_three_cubes_full_ft
"""

from __future__ import annotations

import argparse
import os
import traceback

from loguru import logger as logging

from cosmos_policy._src.imaginaire.config import load_config, pretty_print_overrides
from cosmos_policy._src.imaginaire.lazy_config import LazyConfig
from cosmos_policy._src.imaginaire.serialization import to_yaml
from cosmos_policy.scripts.train import launch

from finetune_three_cubes.config.experiment_full_ft import register as register_experiments


def main() -> None:
    # 关键：在 load_config 之前把我们的 experiment 注册进 ConfigStore。
    register_experiments()

    parser = argparse.ArgumentParser(description="three_cubes full fine-tuning")
    parser.add_argument("--config", help="Path to the config file", required=False)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--dryrun", action="store_true", help="只解析并保存 config，不训练")
    args = parser.parse_args()

    if args.dryrun:
        os.environ["COSMOS_POLICY_DRYRUN"] = "1"
    config = load_config(args.config, args.opts, enable_one_logger=True)

    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        try:
            to_yaml(config, f"{config.job.path_local}/config.yaml")
        except Exception:
            logging.error(f"to_yaml failed: {traceback.format_exc()}")
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        launch(config, args)


if __name__ == "__main__":
    main()

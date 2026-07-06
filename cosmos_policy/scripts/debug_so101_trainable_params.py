"""实例化 SO101 模型并验证 full/partial DiT optimizer 参数集合。"""

from __future__ import annotations

import argparse

from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init, model_init
from cosmos_policy._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cosmos_policy/config/config.py")
    parser.add_argument("--experiment", default="cosmos_predict2_2b_480p_so101_lerobot")
    parser.add_argument(
        "--finetune_mode",
        choices=("full_dit", "partial_dit_last_n", "all", "action_only_head_if_exists"),
        default="partial_dit_last_n",
    )
    parser.add_argument("--train_last_n_dit_blocks", type=int, default=8)
    args = parser.parse_args()
    opts = [
        "--",
        f"experiment={args.experiment}",
        f"model.config.finetune_mode={args.finetune_mode}",
        f"model.config.train_last_n_dit_blocks={args.train_last_n_dit_blocks}",
    ]
    config = load_config(args.config, opts, enable_one_logger=True)
    with distributed_init():
        distributed.init()
    config.validate()
    config.freeze()
    with model_init():
        if str(config.checkpoint.load_path).endswith(".pt"):
            model = create_model_from_consolidated_checkpoint_with_fsdp(config)
        else:
            model = instantiate(config.model)
    optimizer, _ = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for _, parameter in model.named_parameters() if parameter.requires_grad}
    print(f"微调模式: {model.finetune_report.mode}")
    print(f"总参数: {model.finetune_report.total_params:,}")
    print(f"可训练参数: {model.finetune_report.trainable_params:,}")
    print("主要解冻模块:")
    for name in model.finetune_report.trainable_module_names:
        print(f"  - {name}")
    print(f"可训练 parameter tensors: {len(trainable)}")
    print(f"optimizer parameter tensors: {len(optimizer_ids)}（集合验证通过）")


if __name__ == "__main__":
    main()

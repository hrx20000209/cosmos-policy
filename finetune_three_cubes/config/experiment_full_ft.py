"""three_cubes_1 全参数微调实验配置（阶段 2.1）。

继承已确认的基线 `cosmos_predict2_2b_480p_so101_lerobot`
(full_dit + joint_action_future_state)，只覆盖需要改的部分：
  - 输出 / job 名
  - 加入确定性验证 loss + checkpoint 保留 callback
  - 把关键超参显式摆出来（全部走配置，脚本里不硬编码）

所有超参都在 TUNABLES 区块，改这里即可，无需动脚本。
通过 finetune_three_cubes/run_train.py 注册进 Hydra ConfigStore，不修改上游文件。
"""

from __future__ import annotations

import os

from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel

from finetune_three_cubes.callbacks import (
    CheckpointRetentionCallback,
    DeterministicValLossCallback,
)
from finetune_three_cubes.constants import (
    REPO_ID,
    STATS_PATH,
    T5_EMBEDDINGS_PATH,
    DATA_ROOT,
)
from finetune_three_cubes.single_batch_dataset import SingleBatchDataset

from cosmos_policy.datasets.so101_lerobot_dataset import SO101LeRobotCosmosDataset

_SO101_CAMERA_MAP = {
    "primary": "observation.images.front",
    "wrist_left": "observation.images.right",
    "wrist_right": "observation.images.wrist",
}

# 用户指示：不划分验证集，全部 100 个 episode 都用来训练。
_ALL_EPISODES = list(range(100))
_all_episodes_train_dataset = L(SO101LeRobotCosmosDataset)(
    repo_id=REPO_ID,
    root=DATA_ROOT,
    episodes=_ALL_EPISODES,
    chunk_size=50,
    final_image_size=224,
    t5_text_embeddings_path=T5_EMBEDDINGS_PATH,
    dataset_stats_path=STATS_PATH,
    camera_map=_SO101_CAMERA_MAP,
    state_key="observation.state",
    action_key="action",
    normalize_actions=True,
    normalize_proprio=True,
    action_mode="absolute",
    use_proprio=True,
    use_image_aug=True,
    use_stronger_image_aug=True,
    num_duplicates_per_image=4,
    return_value_function_returns=False,
)

# ============================ TUNABLES ============================
BASE_EXPERIMENT = "cosmos_predict2_2b_480p_so101_lerobot"

LR = 1e-5
MAX_ITER = 3000               # ~2 epoch（有效 batch 64 时约 2 epoch）；先跑再决定是否续
WARMUP_STEPS = 200            # 与 MAX_ITER 成比例
GRAD_ACCUM_ITER = 16          # 单卡 batch=1 × grad_accum × num_gpus = 有效 batch
LOGGING_ITER = 10
VALIDATION_ITER = 500         # 确定性 val + 存 checkpoint 的节奏
SAVE_ITER = 500
KEEP_LAST_N_CKPT = 5
# 必须等于 torchrun 的 --nproc_per_node（world_size）；分片全参优化器状态。
# 由 train.sh 通过 FSDP_SHARD_SIZE 环境变量与 NPROC 保持一致。
FSDP_SHARD_SIZE = int(os.environ.get("FSDP_SHARD_SIZE", "4"))
ACTION_LOSS_MULTIPLIER = 1    # 现有整数上采样机制；>1 时放大 action slot 的 loss 权重

# 确定性验证 loss（固定样本 + 固定 sigma 网格 + 固定噪声）
DET_VAL_CAPTURE_BATCHES = 8
DET_VAL_SIGMA_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
DET_VAL_NOISE_SEED = 1234
# =================================================================


def _make(name: str, group: str, overrides: dict) -> LazyDict:
    base = dict(
        defaults=[
            f"/experiment/{BASE_EXPERIMENT}",
            "_self_",
        ],
        optimizer=dict(lr=LR),
        scheduler=dict(
            # 与 MAX_ITER 对齐：warmup 后在一个 cycle 内衰减，之后保持低 lr。
            cycle_lengths=[MAX_ITER, 100000000000000],
            warm_up_steps=[WARMUP_STEPS, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        trainer=dict(
            # 用户指示：不划分验证集（且 Cosmos 内置 validation_step 对 SO101 会 KeyError('guidance')），
            # 因此关闭内置 validation，用全部数据训练。确定性 loss 由下面的 callback 按节奏自行计算。
            run_validation=False,
            run_validation_on_start=False,
            logging_iter=LOGGING_ITER,
            max_iter=MAX_ITER,
            grad_accum_iter=GRAD_ACCUM_ITER,
            callbacks=dict(
                deterministic_val=L(DeterministicValLossCallback)(
                    capture_num_batches=DET_VAL_CAPTURE_BATCHES,
                    det_val_every=VALIDATION_ITER,
                    sigma_grid=list(DET_VAL_SIGMA_GRID),
                    noise_seed=DET_VAL_NOISE_SEED,
                ),
                ckpt_retention=L(CheckpointRetentionCallback)(
                    keep_last_n=KEEP_LAST_N_CKPT,
                ),
            ),
        ),
        checkpoint=dict(save_iter=SAVE_ITER),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_all_episodes_train_dataset,
            batch_size=1,
            drop_last=True,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                finetune_mode="full_dit",
                so101_loss_mode="joint_action_future_state",
                action_loss_multiplier=ACTION_LOSS_MULTIPLIER,
                fsdp_shard_size=FSDP_SHARD_SIZE,
            )
        ),
        # wandb 模式由 config.job.wandb_mode 决定（不是 WANDB_MODE 环境变量），显式设离线。
        job=dict(group=group, name=name, wandb_mode="offline"),
    )
    # 深合并 overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return LazyDict(base)


so101_three_cubes_full_ft = _make(
    name="so101_three_cubes_full_ft",
    group="three_cubes_full_ft",
    overrides={},
)


# ---- 阶段 4.2：端到端 smoke test（极小配置，跑通全链路 + 断点续训） ----
so101_three_cubes_smoke = _make(
    name="so101_three_cubes_smoke",
    group="three_cubes_full_ft",
    overrides=dict(
        trainer=dict(
            max_iter=40,
            logging_iter=5,
            grad_accum_iter=2,   # smoke：小 accum 加速，只为跑通链路
            run_validation=False,
            callbacks=dict(
                deterministic_val=L(DeterministicValLossCallback)(
                    capture_num_batches=2,
                    det_val_every=20,
                    sigma_grid=[1.0, 4.0, 16.0],
                    noise_seed=DET_VAL_NOISE_SEED,
                ),
                ckpt_retention=L(CheckpointRetentionCallback)(keep_last_n=3),
            ),
        ),
        checkpoint=dict(save_iter=20),
    ),
)


# ---- 阶段 4.1：单-batch 过拟合（video + action loss 都必须 → 0） ----
_overfit_dataset = L(SingleBatchDataset)(
    num_fixed=2,
    repo_id=REPO_ID,
    root=DATA_ROOT,
    episodes=[0],
    chunk_size=50,
    final_image_size=224,
    t5_text_embeddings_path=T5_EMBEDDINGS_PATH,
    dataset_stats_path=STATS_PATH,
    camera_map=_SO101_CAMERA_MAP,
    state_key="observation.state",
    action_key="action",
    normalize_actions=True,
    normalize_proprio=True,
    action_mode="absolute",
    use_proprio=True,
    use_image_aug=False,           # 过拟合：关掉数据增强
    use_stronger_image_aug=False,
    num_duplicates_per_image=4,
    return_value_function_returns=False,
)
so101_three_cubes_overfit = _make(
    name="so101_three_cubes_overfit",
    group="three_cubes_full_ft",
    overrides=dict(
        optimizer=dict(lr=1e-4),   # 过拟合用大一点的 lr 更快
        trainer=dict(
            max_iter=300,
            logging_iter=5,
            grad_accum_iter=1,     # 每步一次 optimizer step，曲线更干净
            run_validation=False,
        ),
        checkpoint=dict(save_iter=300),
        dataloader_train=L(DataLoader)(
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            dataset=_overfit_dataset,
            batch_size=1,
            drop_last=True,
        ),
    ),
)


def register() -> None:
    """把本模块的实验注册进 Hydra ConfigStore（供 experiment=<name> 选择）。"""
    cs = ConfigStore.instance()
    for item in [so101_three_cubes_full_ft, so101_three_cubes_smoke, so101_three_cubes_overfit]:
        cs.store(
            group="experiment",
            package="_global_",
            name=item["job"]["name"],
            node=item,
        )

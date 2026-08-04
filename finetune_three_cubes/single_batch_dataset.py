"""单-batch 过拟合用数据集包装（阶段 4.1）。

只暴露固定的 K 个底层样本（默认某个 episode 的前 K 帧），__len__=K。
配合 batch_size 使每个训练 step 反复看到完全相同的一批样本，用来验证
video loss 和 action loss 都能被拉到接近 0——否则说明 action 分支梯度没通/
mask/shift/归一化有问题。

包装（组合）SO101LeRobotCosmosDataset，不改其行为；关闭增强由 use_image_aug=False 控制。
"""

from __future__ import annotations

from torch.utils.data import Dataset

from cosmos_policy.datasets.so101_lerobot_dataset import SO101LeRobotCosmosDataset


class SingleBatchDataset(Dataset):
    def __init__(self, num_fixed: int = 2, fixed_indices: list[int] | None = None, **base_kwargs):
        self.base = SO101LeRobotCosmosDataset(**base_kwargs)
        if fixed_indices is not None:
            self.indices = [int(i) for i in fixed_indices]
        else:
            n = min(int(num_fixed), len(self.base))
            self.indices = list(range(n))
        if not self.indices:
            raise ValueError("SingleBatchDataset 需要至少一个固定样本")
        print(f"SingleBatchDataset: 固定 {len(self.indices)} 个样本用于过拟合，底层 index={self.indices}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i % len(self.indices)]]

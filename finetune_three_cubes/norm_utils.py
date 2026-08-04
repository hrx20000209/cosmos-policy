"""动作 / proprio 的归一化与反归一化工具。

归一化方案：**min/max 线性映射到 [-1, 1]**，统计量从 train split (episode 0..94)
计算并保存在 STATS_PATH。正向公式与 dataset loader 中的 `minmax_normalize` 完全一致：

    scale = max(upper - lower, 1e-6)
    x_norm = 2 * (x_phys - lower) / scale - 1

反归一化是其精确逆：

    x_phys = (x_norm + 1) / 2 * scale + lower

推理 / 画对比图时必须用同一份 STATS_PATH，绝不能换成 ALOHA 或别的 stats。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-6


def load_stats(path: str | Path) -> dict[str, Any]:
    """读取 SO101 stats JSON，把数值统计量转成 float32 ndarray。"""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 stats 文件: {path}")
    raw = json.loads(path.read_text())
    out: dict[str, Any] = dict(raw)
    for key, value in raw.items():
        if key.endswith(("_min", "_max", "_mean", "_std")):
            out[key] = np.asarray(value, dtype=np.float32)
    return out


def _scale(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.maximum(upper - lower, _EPS)


def minmax_normalize(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """物理量 -> [-1, 1]。与 so101_lerobot_dataset.minmax_normalize 一致。"""
    return (2.0 * (x - lower) / _scale(lower, upper) - 1.0).astype(np.float32)


def minmax_denormalize(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """[-1, 1] -> 物理量（正向的精确逆）。"""
    return ((x + 1.0) / 2.0 * _scale(lower, upper) + lower).astype(np.float32)


def normalize_actions(x_phys: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return minmax_normalize(x_phys, stats["actions_min"], stats["actions_max"])


def denormalize_actions(x_norm: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return minmax_denormalize(x_norm, stats["actions_min"], stats["actions_max"])


def normalize_proprio(x_phys: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return minmax_normalize(x_phys, stats["proprio_min"], stats["proprio_max"])


def denormalize_proprio(x_norm: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return minmax_denormalize(x_norm, stats["proprio_min"], stats["proprio_max"])


def roundtrip_max_abs_error(stats: dict[str, Any], n: int = 4096, seed: int = 0) -> dict[str, float]:
    """在物理范围内随机采样，检查 denorm(norm(x)) ≈ x。返回逐类最大绝对误差。"""
    rng = np.random.default_rng(seed)
    result: dict[str, float] = {}
    for name, lo_key, hi_key, norm_fn, denorm_fn in (
        ("actions", "actions_min", "actions_max", normalize_actions, denormalize_actions),
        ("proprio", "proprio_min", "proprio_max", normalize_proprio, denormalize_proprio),
    ):
        lo, hi = stats[lo_key], stats[hi_key]
        x = rng.uniform(lo, hi, size=(n, lo.shape[0])).astype(np.float32)
        recon = denorm_fn(norm_fn(x, stats), stats)
        result[name] = float(np.max(np.abs(recon - x)))
    return result


if __name__ == "__main__":
    import argparse

    from constants import STATS_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", default=STATS_PATH)
    args = parser.parse_args()

    stats = load_stats(args.stats_path)
    print(f"stats: {args.stats_path}")
    print(f"  action_mode = {stats.get('action_mode')}")
    print(f"  action_names = {stats.get('action_names')}")
    print(f"  actions_min = {stats['actions_min'].tolist()}")
    print(f"  actions_max = {stats['actions_max'].tolist()}")
    errs = roundtrip_max_abs_error(stats)
    print(f"roundtrip denorm(norm(x)) 最大绝对误差: {errs}")
    tol = 1e-3
    assert errs["actions"] < tol and errs["proprio"] < tol, f"roundtrip 误差过大: {errs}"
    print(f"PASS: 所有 roundtrip 误差 < {tol}")

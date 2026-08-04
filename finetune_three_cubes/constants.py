"""three_cubes_1 全参数微调管线的固定常量（split / 专用可视化 episode / 路径 / 调色板）。

所有下游脚本从这里读取，避免 split 和 episode id 在多个文件里各写一份而漂移。
"""

from __future__ import annotations

import os

# --- 数据集路径（与 cosmos_policy_experiment_configs.py 的 SO101_LEROBOT_* 环境变量保持一致） ---
DATA_ROOT = os.environ.get("SO101_LEROBOT_ROOT", "/data/rxhuang/three_cubes_1")
REPO_ID = os.environ.get("SO101_LEROBOT_REPO_ID", "local/three_cubes_1")
T5_EMBEDDINGS_PATH = os.environ.get(
    "SO101_LEROBOT_T5", os.path.join(DATA_ROOT, "so101_t5_embeddings.pkl")
)
STATS_PATH = os.environ.get(
    "SO101_LEROBOT_STATS",
    os.path.join(DATA_ROOT, "so101_dataset_statistics_train_episodes_000_094.json"),
)

# --- train / val 按 episode 划分（不按帧，防止同 episode 相邻帧泄漏） ---
# 与实验配置 cosmos_predict2_2b_480p_so101_lerobot 完全一致：train=0..94, val=95..99。
TOTAL_EPISODES = 100
TRAIN_EPISODES = list(range(0, 95))
VAL_EPISODES = list(range(95, 100))

# --- 阶段 1.2：固定挑选的“动作对比图专用” val episode（硬编码，跨 checkpoint 可比） ---
# 从 VAL_EPISODES 中均匀选 3 条；后续 action 对比图 / GIF 只用这几条。
FIXED_COMPARISON_EPISODES = [95, 97, 99]

# 每条对比 episode 里固定的起始帧（观测时刻），用于生成 action chunk 并与 GT 对齐。
# 每条 episode 长度约 507~538 帧，chunk_size=50，所以起始帧留足 chunk 余量。
FIXED_COMPARISON_START_FRAMES = [100, 250, 400]

# --- 动作语义（来自 meta/info.json 与 stats，绝对关节角） ---
ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
# 前 5 维单位为标定后的度数(deg)，gripper 为标定后的百分比(0..100, 越大越开)。
ACTION_UNITS = ["deg", "deg", "deg", "deg", "deg", "%"]
ACTION_DIM = 6
CHUNK_SIZE = 50
FPS = 30

# --- Okabe–Ito 色盲安全定性调色板（用于多关节同轴叠加时按维度上色） ---
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
]
# GT / 预测 用颜色 + 线型双重编码（solid=GT, dashed=pred）。
COLOR_GT = "#000000"
COLOR_PRED = "#D55E00"

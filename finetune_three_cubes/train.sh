#!/usr/bin/env bash
# three_cubes 全参微调启动脚本（多卡 FSDP）。
# 同一条命令既能全新启动，也能断点续训（trainer 自动读 latest_checkpoint.txt）。
#
# 用法:
#   bash finetune_three_cubes/train.sh                       # 用默认 GPU/实验
#   GPUS=5,6,7 NPROC=3 bash finetune_three_cubes/train.sh    # 自定义 GPU
#   EXTRA="trainer.max_iter=200" bash finetune_three_cubes/train.sh  # 追加 hydra override
set -euo pipefail

REPO=/home/rxhuang/Projects/cosmos-policy
cd "$REPO"

# ---- 可配置环境 ----
GPUS="${GPUS:-0,5,6,7}"
NPROC="${NPROC:-4}"
EXPERIMENT="${EXPERIMENT:-so101_three_cubes_full_ft}"
# FSDP 分片数必须等于 world_size(NPROC)；默认自动保持一致。
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-$NPROC}"
MASTER_PORT="${MASTER_PORT:-12355}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-/data/rxhuang/cosmos_three_cubes_runs}"
export SO101_LEROBOT_ROOT="${SO101_LEROBOT_ROOT:-/data/rxhuang/three_cubes_1}"
export SO101_LEROBOT_REPO_ID="${SO101_LEROBOT_REPO_ID:-local/three_cubes_1}"
# wandb 默认离线 + 本地 metrics.jsonl（阶段 2.5）
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$IMAGINAIRE_OUTPUT_ROOT/wandb}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export CUDA_VISIBLE_DEVICES="$GPUS"
# sitecustomize(typing.Self shim) + 本项目包 + lerobot fork
export PYTHONPATH="$REPO/finetune_three_cubes:$REPO:/home/rxhuang/Projects/lerobot/src:${PYTHONPATH:-}"

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT" "$WANDB_DIR"

echo "[train.sh] GPUS=$CUDA_VISIBLE_DEVICES NPROC=$NPROC EXPERIMENT=$EXPERIMENT"
echo "[train.sh] OUTPUT_ROOT=$IMAGINAIRE_OUTPUT_ROOT WANDB_MODE=$WANDB_MODE"

# 用 venv 的 torch.distributed.run（等价 torchrun），避免依赖 PATH 上的 torchrun。
PY="${PYTHON_BIN:-$REPO/.venv/bin/python}"
exec "$PY" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  -m finetune_three_cubes.run_train \
  --config=cosmos_policy/config/config.py -- \
  experiment="$EXPERIMENT" \
  ${EXTRA:-}

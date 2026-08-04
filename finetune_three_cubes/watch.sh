#!/usr/bin/env bash
# 阶段 3.2/3.1/3.3 离线 watcher：检测到新 checkpoint 就自动
#   1) 画动作对比图（固定 episode/frame/种子，反归一化）
#   2) 重绘 loss 曲线（train + 确定性 val）
#   3) 更新演化 GIF/PDF
# 完全不占训练 GPU（默认用单独一张 EVAL_GPU），也不拖慢训练。
#
# 用法:
#   RUN_DIR=/data/rxhuang/cosmos_three_cubes_runs/cosmos_policy/three_cubes_full_ft/so101_three_cubes_full_ft \
#   EVAL_GPU=0 bash finetune_three_cubes/watch.sh
set -uo pipefail

REPO=/home/rxhuang/Projects/cosmos-policy
cd "$REPO"

RUN_DIR="${RUN_DIR:?必须设置 RUN_DIR（训练 run 目录，含 checkpoints/ 和 metrics.jsonl）}"
EVAL_GPU="${EVAL_GPU:-0}"
INTERVAL="${INTERVAL:-60}"
NUM_DENOISING="${NUM_DENOISING:-10}"
SEED="${SEED:-1234}"
EVAL_OUT="${EVAL_OUT:-$RUN_DIR/action_curves}"

export CUDA_VISIBLE_DEVICES="$EVAL_GPU"
export PYTHONPATH="$REPO/finetune_three_cubes:$REPO:/home/rxhuang/Projects/lerobot/src:${PYTHONPATH:-}"
export SO101_LEROBOT_ROOT="${SO101_LEROBOT_ROOT:-/data/rxhuang/three_cubes_1}"
PY="$REPO/.venv/bin/python"
CKPT_DIR="$RUN_DIR/checkpoints"
PROCESSED="$EVAL_OUT/.processed.txt"
mkdir -p "$EVAL_OUT"; touch "$PROCESSED"

echo "[watch] RUN_DIR=$RUN_DIR EVAL_GPU=$EVAL_GPU INTERVAL=${INTERVAL}s"
while true; do
  # 每轮都重绘 loss 曲线（便宜，无需 GPU 权重）
  "$PY" finetune_three_cubes/plot_losses.py --run_dir "$RUN_DIR" >/dev/null 2>&1 || true

  if [ -d "$CKPT_DIR" ]; then
    for ckpt in $(ls -d "$CKPT_DIR"/iter_* 2>/dev/null | sort); do
      name=$(basename "$ckpt")
      # 只处理目录型 DCP checkpoint，且未处理过的
      if [ -d "$ckpt" ] && ! grep -qxF "$name" "$PROCESSED"; then
        echo "[watch] 新 checkpoint: $name -> 生成动作对比图"
        if "$PY" finetune_three_cubes/eval_action_curves.py \
            --checkpoint "$ckpt" --output_dir "$EVAL_OUT" \
            --num_denoising_steps "$NUM_DENOISING" --seed "$SEED"; then
          echo "$name" >> "$PROCESSED"
          "$PY" finetune_three_cubes/make_gif.py --eval_dir "$EVAL_OUT" >/dev/null 2>&1 || true
        else
          echo "[watch] $name 评估失败，稍后重试"
        fi
      fi
    done
  fi
  sleep "$INTERVAL"
done

#!/usr/bin/env bash
# Phase 2 baseline sanity check on ORIGINAL LIBERO (stock verified run_libero_eval.py).
# Usage: run_baseline_eval.sh <gpu> <num_trials_per_task> <run_id_note> [task_suite]
set -euo pipefail
GPU=${1:-0}
NTRIAL=${2:-1}
NOTE=${3:-smoke_bf16}
SUITE=${4:-libero_10}

cd /home/rxhuang/Projects/cosmos-policy
source experiments/liberoplus_quantization/env.sh
export CUDA_VISIBLE_DEVICES=$GPU
export MUJOCO_EGL_DEVICE_ID=$GPU
CKPT=/data/rxhuang/models/cosmos-policy-libero-2b

.venv/bin/python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
  --config cosmos_predict2_2b_480p_libero__inference_only \
  --ckpt_path $CKPT/Cosmos-Policy-LIBERO-Predict2-2B.pt \
  --config_file cosmos_policy/config/config.py \
  --use_wrist_image True \
  --use_proprio True \
  --normalize_proprio True \
  --unnormalize_actions True \
  --dataset_stats_path $CKPT/libero_dataset_statistics.json \
  --t5_text_embeddings_path $CKPT/libero_t5_embeddings.pkl \
  --trained_with_image_aug True \
  --chunk_size 16 \
  --num_open_loop_steps 16 \
  --task_suite_name $SUITE \
  --num_trials_per_task $NTRIAL \
  --local_log_dir experiments/liberoplus_quantization/logs/ \
  --randomize_seed False \
  --seed 195 \
  --deterministic True \
  --use_variance_scale False \
  --use_jpeg_compression True \
  --flip_images True \
  --num_denoising_steps_action 5 \
  --num_denoising_steps_future_state 1 \
  --num_denoising_steps_value 1 \
  --ar_future_prediction False \
  --ar_value_prediction False \
  --available_gpus "0" \
  --run_id_note $NOTE

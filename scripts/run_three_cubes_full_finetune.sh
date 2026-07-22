#!/usr/bin/env bash
set -euo pipefail

# Reproducible launcher for the local Three_Cubes_1 full-parameter run.
REPO_DIR=/root/autodl-tmp/cosmos-policy
OUTPUT_DIR=${COSMOS_THREE_CUBES_OUTPUT:-$REPO_DIR/outputs/cosmos_three_cubes_full_finetune}
DATA_ROOT=$OUTPUT_DIR/data/hrx2000/Three_Cubes_1
MAX_STEPS=${1:-10000}
SAVE_STEPS=${2:-500}
NPROC=${NPROC_PER_NODE:-1}

export SO101_LEROBOT_ROOT=$DATA_ROOT
export SO101_LEROBOT_REPO_ID=hrx2000/Three_Cubes_1
export SO101_LEROBOT_T5=$DATA_ROOT/so101_t5_embeddings.pkl
export SO101_LEROBOT_STATS=$DATA_ROOT/so101_dataset_statistics.json
export COSMOS_BASE_CHECKPOINT=$OUTPUT_DIR/pretrained/model-480p-16fps.pt
export COSMOS_TOKENIZER_CHECKPOINT=$OUTPUT_DIR/pretrained/cosmos_official/tokenizer/tokenizer.pth
export IMAGINAIRE_OUTPUT_ROOT=$OUTPUT_DIR/runs
export COSMOS_ENABLE_WANDB=${COSMOS_ENABLE_WANDB:-0}

cd "$REPO_DIR"
torchrun --nproc_per_node="$NPROC" -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_so101_lerobot \
  optimizer=adamw \
  model.config.finetune_mode=all \
  model.config.use_lora=false \
  model.config.lambda_video=1.0 \
  model.config.lambda_action=1.0 \
  dataloader_train.dataset.episodes=null \
  trainer.run_validation=false \
  trainer.max_iter="$MAX_STEPS" \
  checkpoint.save_iter="$SAVE_STEPS"

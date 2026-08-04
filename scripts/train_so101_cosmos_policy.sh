#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Train SO101 Cosmos Policy from nvidia/Cosmos-Policy-ALOHA-Predict2-2B.

Environment overrides:
  GPUS                 CUDA device list for training (default: 0,1,2,5)
  NPROC                torchrun processes (default: number of GPUS)
  STEPS                optimizer steps (default: 20000)
  SAVE_ITER            checkpoint interval (default: 500)
  VAL_ITER             validation interval (default: 500)
  RUN_VALIDATION       use Cosmos built-in validation (default: false)
  GRAD_ACCUM           gradient accumulation (default: 16)
  LR                   optimizer LR (default: 1e-5)
  RUN_NAME             job name (default: timestamped full_dit run)
  OUTPUT_ROOT          Imaginaire output root
  DATA_ROOT            LeRobot dataset root (default: /data/rxhuang/three_cubes_1)
  REPO_ID              LeRobot repo id (default: local/three_cubes_1)
  FORCE_STATS=1        recompute train-only stats
  FORCE_T5=1           recompute task T5 embeddings
  DRYRUN=1             write resolved config only

Examples:
  STEPS=20 bash scripts/train_so101_cosmos_policy.sh
  GPUS=0,1,2,5 STEPS=20000 RUN_NAME=so101_full_dit_20k bash scripts/train_so101_cosmos_policy.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

COSMOS_ROOT=${COSMOS_ROOT:-/home/rxhuang/Projects/cosmos-policy}
LEROBOT_ROOT=${LEROBOT_ROOT:-/home/rxhuang/Projects/lerobot}
PYTHON=${PYTHON:-${COSMOS_ROOT}/.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-/data/rxhuang/three_cubes_1}
REPO_ID=${REPO_ID:-local/three_cubes_1}
T5_PATH=${T5_PATH:-${DATA_ROOT}/so101_t5_embeddings.pkl}
STATS_PATH=${STATS_PATH:-${DATA_ROOT}/so101_dataset_statistics_train_episodes_000_094.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${COSMOS_ROOT}/output_train/three_cubes}
RUN_NAME=${RUN_NAME:-so101_cosmos_policy_full_dit_$(date +%Y%m%d_%H%M%S)}
GPUS=${GPUS:-0,1,2,5}
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NPROC=${NPROC:-${#GPU_ARRAY[@]}}
FSDP_SHARD_SIZE=${FSDP_SHARD_SIZE:-${NPROC}}
MASTER_PORT=${MASTER_PORT:-12341}
STEPS=${STEPS:-20000}
SAVE_ITER=${SAVE_ITER:-500}
VAL_ITER=${VAL_ITER:-500}
RUN_VALIDATION=${RUN_VALIDATION:-false}
GRAD_ACCUM=${GRAD_ACCUM:-16}
LR=${LR:-1e-5}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-2}
WANDB_MODE=${WANDB_MODE:-disabled}

export PYTHONPATH="${COSMOS_ROOT}:${LEROBOT_ROOT}/src:${PYTHONPATH:-}"
export SO101_LEROBOT_ROOT="${DATA_ROOT}"
export SO101_LEROBOT_REPO_ID="${REPO_ID}"
export SO101_LEROBOT_T5="${T5_PATH}"
export SO101_LEROBOT_STATS="${STATS_PATH}"
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_ROOT}"
cd "${COSMOS_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found or not executable: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${T5_PATH}" || "${FORCE_T5:-0}" == "1" ]]; then
  "${PYTHON}" -m cosmos_policy.datasets.save_so101_lerobot_t5_text_embeddings \
    --repo_id "${REPO_ID}" \
    --root "${DATA_ROOT}" \
    --output "${T5_PATH}"
fi

if [[ ! -f "${STATS_PATH}" || "${FORCE_STATS:-0}" == "1" ]]; then
  "${PYTHON}" -m cosmos_policy.datasets.export_so101_lerobot_stats \
    --repo_id "${REPO_ID}" \
    --root "${DATA_ROOT}" \
    --episodes $(seq 0 94) \
    --output "${STATS_PATH}"
fi

RUN_DIR="${OUTPUT_ROOT}/cosmos_policy/so101_lerobot/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
echo "Run dir: ${RUN_DIR}"
echo "Training GPUs: ${GPUS} (nproc=${NPROC}, fsdp_shard_size=${FSDP_SHARD_SIZE})"
echo "Stats: ${STATS_PATH}"
echo "T5: ${T5_PATH}"

OPTS=(
  "--"
  "experiment=cosmos_predict2_2b_480p_so101_lerobot"
  "job.project=cosmos_policy"
  "job.group=so101_lerobot"
  "job.name=${RUN_NAME}"
  "job.wandb_mode=${WANDB_MODE}"
  "trainer.max_iter=${STEPS}"
  "trainer.grad_accum_iter=${GRAD_ACCUM}"
  "trainer.logging_iter=10"
  "trainer.run_validation=${RUN_VALIDATION}"
  "trainer.validation_iter=${VAL_ITER}"
  "trainer.max_val_iter=32"
  "checkpoint.save_iter=${SAVE_ITER}"
  "optimizer.lr=${LR}"
  "model.config.finetune_mode=full_dit"
  "model.config.fsdp_shard_size=${FSDP_SHARD_SIZE}"
  "dataloader_train.batch_size=${BATCH_SIZE}"
  "dataloader_train.num_workers=${NUM_WORKERS}"
  "dataloader_val.batch_size=1"
  "dataloader_val.num_workers=${VAL_NUM_WORKERS}"
)

if [[ "${DRYRUN:-0}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    -m cosmos_policy.scripts.train --dryrun --config=cosmos_policy/config/config.py "${OPTS[@]}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON}" -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
  -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py "${OPTS[@]}"

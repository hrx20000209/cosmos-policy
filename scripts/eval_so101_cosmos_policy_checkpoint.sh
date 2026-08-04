#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Evaluate one SO101 Cosmos Policy checkpoint.

Required:
  CHECKPOINT=/path/to/checkpoints/iter_000000500

Common overrides:
  RUN_DIR              Training run directory containing metrics.jsonl
  STEP                 Step label, inferred from checkpoint name when omitted
  EVAL_GPU             CUDA device for eval (default: 5)
  DATA_ROOT            Dataset root (default: /data/rxhuang/three_cubes_1)
  REPO_ID              LeRobot repo id (default: local/three_cubes_1)
  NUM_SAMPLES          Teacher-forced validation starts (default: 8)
  EPISODE              Teacher-forced validation episode (default: 95)
  FULL_EPISODE=1       Also run full-episode open-loop on episodes 95 and 97

Example:
  CHECKPOINT=/path/to/checkpoints/iter_000000500 RUN_DIR=/path/to/run \
    bash scripts/eval_so101_cosmos_policy_checkpoint.sh
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
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT=/path/to/checkpoint}
RUN_DIR=${RUN_DIR:-$(dirname "$(dirname "${CHECKPOINT}")")}
STEP=${STEP:-$(basename "${CHECKPOINT}" | sed -E 's/[^0-9]*([0-9]+)/\1/')}
STEP_PADDED=$(printf "%09d" "${STEP#0}")
EVAL_GPU=${EVAL_GPU:-5}
NUM_SAMPLES=${NUM_SAMPLES:-8}
EPISODE=${EPISODE:-95}
FULL_EPISODE=${FULL_EPISODE:-0}
QUERY_STRIDE=${QUERY_STRIDE:-8}
NUM_DENOISING_STEPS=${NUM_DENOISING_STEPS:-10}

export PYTHONPATH="${COSMOS_ROOT}:${LEROBOT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "${COSMOS_ROOT}"

LOSS_DIR="${RUN_DIR}/loss_curves"
CURVE_DIR="${RUN_DIR}/eval_curves/step_${STEP_PADDED}"
mkdir -p "${LOSS_DIR}" "${CURVE_DIR}"

if [[ -f "${RUN_DIR}/metrics.jsonl" ]]; then
  "${PYTHON}" -m cosmos_policy.scripts.plot_so101_single_training_metrics \
    --metrics "${RUN_DIR}/metrics.jsonl" \
    --output "${LOSS_DIR}/step_${STEP_PADDED}.png" \
    --max_step "${STEP#0}"
fi

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" -m cosmos_policy.scripts.eval_so101_action_curves \
  --repo_id "${REPO_ID}" \
  --root "${DATA_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --t5_text_embeddings_path "${T5_PATH}" \
  --dataset_stats_path "${STATS_PATH}" \
  --episodes "${EPISODE}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_denoising_steps "${NUM_DENOISING_STEPS}" \
  --output_dir "${CURVE_DIR}"

if [[ "${FULL_EPISODE}" == "1" ]]; then
  for ep in 95 97; do
    CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON}" -m cosmos_policy.scripts.eval_so101_full_episode_action_curve \
      --repo_id "${REPO_ID}" \
      --root "${DATA_ROOT}" \
      --checkpoint "${CHECKPOINT}" \
      --t5_text_embeddings_path "${T5_PATH}" \
      --dataset_stats_path "${STATS_PATH}" \
      --episode "${ep}" \
      --query_stride "${QUERY_STRIDE}" \
      --num_denoising_steps "${NUM_DENOISING_STEPS}" \
      --output_dir "${RUN_DIR}/full_episode_curves/step_${STEP_PADDED}/episode_${ep}"
  done
fi

echo "Loss plot: ${LOSS_DIR}/step_${STEP_PADDED}.png"
echo "Action eval: ${CURVE_DIR}"

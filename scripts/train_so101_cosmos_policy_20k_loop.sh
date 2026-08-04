#!/usr/bin/env bash
set -euo pipefail

COSMOS_ROOT=${COSMOS_ROOT:-/home/rxhuang/Projects/cosmos-policy}
RUN_NAME=${RUN_NAME:-so101_full_dit_20k_$(date +%Y%m%d_%H%M%S)}
TOTAL_STEPS=${TOTAL_STEPS:-20000}
CHUNK_STEPS=${CHUNK_STEPS:-500}
SAVE_ITER=${SAVE_ITER:-500}
EVAL_GPU=${EVAL_GPU:-5}
EPISODE=${EPISODE:-95}
NUM_SAMPLES=${NUM_SAMPLES:-8}
FULL_EPISODE_EVERY=${FULL_EPISODE_EVERY:-2000}
OUTPUT_ROOT=${OUTPUT_ROOT:-${COSMOS_ROOT}/output_train/three_cubes}
RUN_DIR="${OUTPUT_ROOT}/cosmos_policy/so101_lerobot/${RUN_NAME}"

cd "${COSMOS_ROOT}"

next_step="${CHUNK_STEPS}"
if [[ -f "${RUN_DIR}/checkpoints/latest_checkpoint.txt" ]]; then
  latest="$(tr -d '[:space:]' < "${RUN_DIR}/checkpoints/latest_checkpoint.txt")"
  latest_step="$(basename "${latest}" | sed -E 's/[^0-9]*([0-9]+)/\1/')"
  if [[ -n "${latest_step}" ]]; then
    next_step=$((10#${latest_step} + CHUNK_STEPS))
  fi
fi

while (( next_step <= TOTAL_STEPS )); do
  echo "=== Training ${RUN_NAME} to step ${next_step}/${TOTAL_STEPS} ==="
  RUN_NAME="${RUN_NAME}" \
    STEPS="${next_step}" \
    SAVE_ITER="${SAVE_ITER}" \
    RUN_VALIDATION=false \
    bash scripts/train_so101_cosmos_policy.sh

  ckpt_step="$(printf "%09d" "${next_step}")"
  checkpoint="${RUN_DIR}/checkpoints/iter_${ckpt_step}"
  if [[ ! -d "${checkpoint}" ]]; then
    echo "Expected checkpoint missing: ${checkpoint}" >&2
    exit 1
  fi

  full_episode=0
  if (( next_step % FULL_EPISODE_EVERY == 0 )); then
    full_episode=1
  fi

  echo "=== Evaluating ${checkpoint} ==="
  CHECKPOINT="${checkpoint}" \
    RUN_DIR="${RUN_DIR}" \
    STEP="${next_step}" \
    EVAL_GPU="${EVAL_GPU}" \
    EPISODE="${EPISODE}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    FULL_EPISODE="${full_episode}" \
    bash scripts/eval_so101_cosmos_policy_checkpoint.sh

  next_step=$((next_step + CHUNK_STEPS))
done

echo "Completed ${TOTAL_STEPS} steps for ${RUN_NAME}"

#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Watch a SO101 Cosmos Policy run directory and evaluate new checkpoints.

Required:
  RUN_DIR=/path/to/output_train/.../cosmos_policy/so101_lerobot/RUN_NAME

Common overrides:
  EVAL_GPU=5            GPU used for checkpoint evaluation
  INTERVAL=60           Poll interval in seconds
  FULL_EVERY=2000       Run full-episode curves every N steps
  MIN_STEP=1            Ignore checkpoints below this step

Example:
  RUN_DIR=/home/rxhuang/Projects/cosmos-policy/output_train/three_cubes/cosmos_policy/so101_lerobot/so101_full_dit_20k \
    EVAL_GPU=5 bash scripts/watch_so101_cosmos_policy_checkpoints.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

RUN_DIR=${RUN_DIR:?Set RUN_DIR=/path/to/training/run}
COSMOS_ROOT=${COSMOS_ROOT:-/home/rxhuang/Projects/cosmos-policy}
INTERVAL=${INTERVAL:-60}
FULL_EVERY=${FULL_EVERY:-2000}
MIN_STEP=${MIN_STEP:-1}
STATE_DIR="${RUN_DIR}/.eval_state"
mkdir -p "${STATE_DIR}"

echo "Watching ${RUN_DIR}/checkpoints"
while true; do
  shopt -s nullglob
  for checkpoint in "${RUN_DIR}"/checkpoints/iter_*; do
    [[ -d "${checkpoint}" ]] || continue
    base=$(basename "${checkpoint}")
    step=$(sed -E 's/[^0-9]*([0-9]+)/\1/' <<< "${base}")
    step_num=$((10#${step}))
    if (( step_num < MIN_STEP )); then
      continue
    fi
    done_file="${STATE_DIR}/${base}.done"
    if [[ -f "${done_file}" ]]; then
      continue
    fi
    full_episode=0
    if (( step_num % FULL_EVERY == 0 )); then
      full_episode=1
    fi
    echo "Evaluating ${checkpoint} (step=${step_num}, full_episode=${full_episode})"
    CHECKPOINT="${checkpoint}" RUN_DIR="${RUN_DIR}" STEP="${step_num}" FULL_EPISODE="${full_episode}" \
      bash "${COSMOS_ROOT}/scripts/eval_so101_cosmos_policy_checkpoint.sh"
    date -Is > "${done_file}"
  done
  sleep "${INTERVAL}"
done

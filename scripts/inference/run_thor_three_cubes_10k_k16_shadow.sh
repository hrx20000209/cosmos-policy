#!/usr/bin/env bash
# Start the 10K/K=16 Cosmos server on Thor in no-actuation shadow mode.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/hrx/Projects/cosmos-policy}
LEROBOT_SRC=${LEROBOT_SRC:-/home/hrx/Projects/lerobot/src}
MODEL_ROOT=${MODEL_ROOT:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy}
PYTHON_BIN=${PYTHON_BIN:-/home/hrx/miniconda3/envs/cosmos/bin/python}
PORT=${PORT:-8082}
# Keep the full K=16 horizon so that overlapping replans can be merged by
# LeRobot's client-side queue aggregation.  Override only for a narrow
# protocol smoke test (for example ACTIONS_PER_CHUNK=1).
ACTIONS_PER_CHUNK=${ACTIONS_PER_CHUNK:-16}

for path in \
  "$MODEL_ROOT/model/.metadata" \
  "$MODEL_ROOT/configs/eval_config.py" \
  "$MODEL_ROOT/processed_data/dataset_statistics.json" \
  "$MODEL_ROOT/processed_data/t5_text_embeddings.pkl"; do
  [[ -f "$path" ]] || { echo "Missing runtime artifact: $path" >&2; exit 2; }
done

export PYTHONPATH="$MODEL_ROOT:$REPO_DIR:$LEROBOT_SRC${PYTHONPATH:+:$PYTHONPATH}"
export THREE_CUBES_OUTPUT_ROOT="$MODEL_ROOT"
# Dataset construction is lazy during inference, but keep this explicit for
# config provenance rather than silently using a training-server path.
export THREE_CUBES_DATASET_ROOT="${THREE_CUBES_DATASET_ROOT:-/home/hrx/Projects/datasets/three_cubes_1}"
# Thor already has the Hugging Face snapshot layout.  Use it strictly offline
# and do not inherit the stale localhost proxy from an interactive shell.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

exec "$PYTHON_BIN" -m cosmos_policy.experiments.robot.so101_async_deploy_three_cubes_k16 \
  --host=127.0.0.1 \
  --port="$PORT" \
  --fps=2 \
  --inference_latency=3.0 \
  --obs_queue_timeout=3.0 \
  --ckpt_path="$MODEL_ROOT/model" \
  --cosmos_config=cosmos_predict2_2b_three_cubes_full_ft \
  --config_file=configs/eval_config.py \
  --dataset_stats_path="$MODEL_ROOT/processed_data/dataset_statistics.json" \
  --t5_text_embeddings_path="$MODEL_ROOT/processed_data/t5_text_embeddings.pkl" \
  --primary_camera_key=front \
  --left_wrist_camera_key=right \
  --right_wrist_camera_key=wrist \
  --num_denoising_steps_action=10 \
  --actions_per_chunk="$ACTIONS_PER_CHUNK" \
  --max_delta_from_observation=4.0 \
  --max_gripper_delta_from_observation=4.0 \
  --max_step_delta=2.0 \
  --max_gripper_step_delta=2.0 \
  --dry_run_zero_actions=true

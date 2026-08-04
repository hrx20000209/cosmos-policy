#!/usr/bin/env bash
# Run one unified YAML configuration, sharded over the listed GPUs.
# Usage: run_pilot.sh [config_name_or_path] [comma_separated_gpus] [limit]
set -euo pipefail

CONFIG=${1:-baseline_bf16.yaml}
GPUS=${2:-0}
LIMIT=${3:-0}

cd /home/rxhuang/Projects/cosmos-policy
source experiments/liberoplus_quantization/env.sh

args=(
  .venv/bin/python
  experiments/liberoplus_quantization/scripts/sweep_runner.py
  --configs "$CONFIG"
  --gpus "$GPUS"
)
if [[ "$LIMIT" -gt 0 ]]; then
  args+=(--limit "$LIMIT")
fi
"${args[@]}"

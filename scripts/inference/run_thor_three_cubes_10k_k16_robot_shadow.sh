#!/usr/bin/env bash
# Supervised hardware-in-the-loop validation for the K=16 three-cubes model.
#
# This launches LeRobot's real SO-101 client against the *zero-action* shadow
# server on 127.0.0.1:8082.  It opens the cameras and /dev/ttyACM1 and records
# the complete async queue/timeline, but the server echoes the current proprio
# rather than a model action.  It is deliberately not a nonzero-actuation
# launcher.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/hrx/Projects/cosmos-policy}
CLIENT=${CLIENT:-$REPO_DIR/scripts/inference/run_cosmos_three_cubes_client.sh}

[[ -x "$CLIENT" ]] || { echo "Missing executable client: $CLIENT" >&2; exit 2; }

export SERVER_ADDRESS=${SERVER_ADDRESS:-127.0.0.1:8082}
export ROBOT_PORT=${ROBOT_PORT:-/dev/ttyACM0}
export ROBOT_ID=${ROBOT_ID:-follower_arm}
# Fixed camera-slot ordering used by the three-cubes checkpoint on Thor.
export FRONT_CAMERA=${FRONT_CAMERA:-4}
export RIGHT_CAMERA=${RIGHT_CAMERA:-2}
export WRIST_CAMERA=${WRIST_CAMERA:-0}
export ACTIONS_PER_CHUNK=${ACTIONS_PER_CHUNK:-16}
export CHUNK_SIZE_THRESHOLD=${CHUNK_SIZE_THRESHOLD:-0.70}
# Overlapping timesteps are blended rather than replaced:
#   0.7 * already queued action + 0.3 * newest replan.
export AGGREGATE_FN_NAME=${AGGREGATE_FN_NAME:-conservative}
export FPS=${FPS:-2}
export MAX_RELATIVE_TARGET=${MAX_RELATIVE_TARGET:-2.0}
# The wrist camera /dev/video0 falls back to YUYV; front=/dev/video4 and
# right=/dev/video2 use the shared MJPG setting. This combination was verified
# concurrently at 640x480@30.
export WRIST_FOURCC=${WRIST_FOURCC:-YUYV}
export DISPLAY_DATA=${DISPLAY_DATA:-false}
export RECORD_TIMELINE=${RECORD_TIMELINE:-true}
export TIMELINE_SAVE_IMAGES=${TIMELINE_SAVE_IMAGES:-key}
export TIMELINE_LOG_DIR=${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline/cosmos_three_cubes_10k_k16_shadow}

exec "$CLIENT"

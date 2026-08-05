#!/usr/bin/env bash
set -euo pipefail

# Robot-side client for the Cosmos Policy three_cubes_1 deployment on a real SO-101.
#
# This is the stock lerobot.async_inference.robot_client -- the Cosmos side lives entirely
# in run_cosmos_three_cubes_server.sh. Camera indices, robot port and task string are
# copied from run_lingbot_va_three_cubes.sh, which drives the same physical setup.
#
# Start the server FIRST and wait for "Loaded Cosmos SO101 checkpoint" before running this.
#
# SAFETY: MAX_RELATIVE_TARGET is set by default here. It is LeRobot's client-side cap on
# per-step joint motion and is independent of the server-side clamps, so it still protects
# you if the server config is wrong. Keep it.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

# Must match PORT in run_cosmos_three_cubes_server.sh (8081, since 8080 is usually the
# LingBot-VA server on this host).
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8081}"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

# front=2 / right=0 / wrist=4 on this host. A swapped camera order silently degrades the
# policy -- the three views feed fixed slots. Verify with:
#   python -m lerobot.scripts.lerobot_find_cameras opencv
FRONT_CAMERA="${FRONT_CAMERA:-2}"
RIGHT_CAMERA="${RIGHT_CAMERA:-0}"
WRIST_CAMERA="${WRIST_CAMERA:-4}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

# USB BANDWIDTH -- this is why the motor bus throws "Incorrect status packet".
# The arm's serial adapter (/dev/ttyACM1 = USB 1-4.2.3) sits on the SAME USB 2.0 hub as all
# three cameras (1-4.2.1 / 1-4.2.2 / 1-4.2.4). UVC reserves isochronous bandwidth up front,
# and the wrist camera only supports YUYV (no MJPG), so at 640x480x30 it alone claims
# 640*480*2*30 = 18.4 MB/s of a ~40 MB/s practical bus. The CDC-ACM serial link is left
# without slots and sync_read('Present_Position') fails.
#
# We only sample one frame per replan (~0.7 Hz), so streaming faster buys nothing.
#
# What each camera actually supports (v4l2-ctl --list-formats-ext):
#   front /dev/video2, right /dev/video0 : MJPG, 30 fps ONLY -- no other rate is offered,
#       and LeRobot's _validate_fps() hard-fails if the driver reports a different rate.
#       They stay at 30; being compressed they are not the bandwidth problem anyway.
#   wrist /dev/video4 : YUYV only (no MJPG), but it does offer 30/25/20/15/10/5 fps.
#       This is the camera to slow down -- uncompressed 640x480 costs 614 KB per frame,
#       so 30 fps = 18.4 MB/s and 10 fps = 6.1 MB/s, freeing ~12 MB/s on the shared bus.
#
# The durable fix is physical: move the arm's USB cable to a hub with no cameras on it
# (e.g. the 1-4.1 hub) -- verify with:  udevadm info -q property -n /dev/ttyACM1 | grep DEVPATH
# Both back at their native 30. WRIST_CAMERA_FPS was briefly set to 10 on a USB-bandwidth
# hypothesis that later measurement DISPROVED (the servo bus stayed clean at 0 failures with
# all three cameras streaming at 30, and under full CPU load), and running the wrist camera
# off its native rate correlated with "exceeded maximum consecutive read failures" crashes.
CAMERA_FPS="${CAMERA_FPS:-30}"
WRIST_CAMERA_FPS="${WRIST_CAMERA_FPS:-30}"
FOURCC="${FOURCC:-MJPG}"
# video4 exposes YUYV only -- MJPG is not an option for the wrist camera.
WRIST_FOURCC="${WRIST_FOURCC:-YUYV}"

# Exact training instruction for Three_Cubes_1 (meta/tasks.parquet). The server looks the
# T5 embedding up by this exact string; any edit changes the conditioning.
TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"

# The Cosmos server returns chunk_size=30 actions and truncates to its own
# actions_per_chunk. Keep this in sync with the server value.
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-10}"

# CRITICAL. robot_client._ready_to_send_observation() is
#     queue.qsize() / action_chunk_size <= chunk_size_threshold
# so 0.0 means "only ask for the next chunk once the queue is completely empty" -- the arm
# executes the whole chunk and then stands still for a full inference (~1.2 s at denoise=3,
# ~2.5 s at denoise=10). That is a guaranteed stall every single chunk, and it defeats the
# entire point of async inference.
#
# To actually hide the latency, the actions still queued when the request fires must cover
# the inference time:  threshold x (ACTIONS_PER_CHUNK / FPS)  >=  server latency.
# At denoise=3 (~1.2 s), 30 actions and FPS=15 the chunk covers 2.0 s, so 0.7 x 2.0 = 1.4 s
# of buffer against a 1.2 s inference -- continuous motion with margin.
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.7}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"

# Must be an int (draccus) and must equal the server's FPS. 10 actions at 2 fps covers
# 5.0 s, so CHUNK_SIZE_THRESHOLD=0.7 leaves 3.5 s of queued motion against a ~2.55 s
# inference -- the arm keeps moving instead of stalling. See the FPS note in the server
# script for why 3 is too tight.
FPS="${FPS:-2}"

# Client-side per-step joint cap (degrees). Independent of the server clamps. Only tighten.
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-5.0}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline/cosmos_three_cubes}"
TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-key}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

mkdir -p "${TIMELINE_LOG_DIR}"
cd "${REPO_DIR}"

ROBOT_SAFETY_ARGS=()
if [[ -n "${MAX_RELATIVE_TARGET}" ]]; then
  ROBOT_SAFETY_ARGS+=(--robot.max_relative_target="${MAX_RELATIVE_TARGET}")
fi

# policy_type/pretrained_name_or_path are part of the async handshake the client always
# sends. The Cosmos server ignores the weights path and serves its own checkpoint; the
# path is passed only so the handshake carries a meaningful identifier.
exec "${PYTHON_BIN}" -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  "${ROBOT_SAFETY_ARGS[@]}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${FOURCC}\"}, right: {type: opencv, index_or_path: ${RIGHT_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${FOURCC}\"}, wrist: {type: opencv, index_or_path: ${WRIST_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${WRIST_CAMERA_FPS}, fourcc: \"${WRIST_FOURCC}\"} }" \
  --task="${TASK}" \
  --policy_type="cosmos_policy" \
  --pretrained_name_or_path="/home/hrx/Projects/models/cosmos_policy/iter_000005000" \
  --policy_device="${POLICY_DEVICE}" \
  --client_device="${CLIENT_DEVICE}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME}" \
  --fps="${FPS}" \
  --record_timeline="${RECORD_TIMELINE}" \
  --timeline_log_dir="${TIMELINE_LOG_DIR}" \
  --timeline_save_images="${TIMELINE_SAVE_IMAGES}" \
  --display_data="${DISPLAY_DATA}"

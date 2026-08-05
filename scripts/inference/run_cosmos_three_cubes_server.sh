#!/usr/bin/env bash
set -euo pipefail

# Cosmos Policy async inference server for three_cubes_1 on a real SO-101 (Jetson Thor).
#
# Serves the step-5000 full-finetune checkpoint through LeRobot's async-inference gRPC
# protocol. The robot side stays the stock lerobot.async_inference.robot_client; only the
# policy server is replaced (see run_cosmos_three_cubes_client.sh).
#
# FIRST REAL-ROBOT DEPLOYMENT OF THIS CHECKPOINT.
# ----------------------------------------------
# This checkpoint has only ever been evaluated offline, on the training set, open-loop.
# It has never driven a physical arm. Offline diagnostics showed action-chunk boundary
# jumps up to ~30x the ground-truth step size under naive chunk-splicing, dropping to
# ~2.5x with higher-rate replanning + temporal ensembling. That ensembling is NOT
# implemented in this server -- it simply executes the first ACTIONS_PER_CHUNK steps of
# each fresh chunk. Treat the arm's behavior as completely unknown.
#
# Stage 1 (DRY_RUN=true, the default here) forces every action to the current measured
# proprio, so the arm should not move at all. Use it to prove out gRPC, camera keys,
# the action schema and model load. Only after that passes clean, set DRY_RUN=false,
# with a person next to the arm and a hand on the e-stop.
#
# Port note: 8080 on this host is normally taken by the LingBot-VA policy server, so this
# script defaults to 8081. Keep the client's --server_address in sync.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/cosmos-policy}"
LEROBOT_SRC="${LEROBOT_SRC:-/home/hrx/Projects/lerobot/src}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/cosmos/bin/python}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8081}"

CKPT_PATH="${CKPT_PATH:-/home/hrx/Projects/models/cosmos_policy/iter_000005000}"
# Regenerated on Thor from the LeRobot Three_Cubes_1 metadata with the repo's own
# compute_so101_statistics(action_mode="absolute"). min/max (the only stats the inference
# path reads) match the reference file bit-exactly. The older in-repo
# aloha/three_cubes_so101_dataset_statistics.json is NOT usable here: it predates the
# action_names/state_names/action_dim schema fields and fails the server's startup gate.
DATASET_STATS_PATH="${DATASET_STATS_PATH:-/home/hrx/Projects/models/cosmos_policy/so101_dataset_statistics.json}"
T5_EMBEDDINGS_PATH="${T5_EMBEDDINGS_PATH:-/home/hrx/Projects/models/cosmos_policy/so101_t5_embeddings.pkl}"
COSMOS_CONFIG="${COSMOS_CONFIG:-cosmos_predict2_2b_480p_so101_lerobot}"

# Trained with chunk_size=30 (NOT the 50 the upstream template assumed).
# ACTIONS_PER_CHUNK is how many of those 30 steps are executed before replanning from a
# fresh observation. Lower = more frequent replanning = smaller boundary jumps. Do not
# raise this to paper over jitter.
# 10 replans per 30-step chunk. This is the value the offline action-curve figures were
# generated with; 30 (full open-loop chunk) was tried and measured no better.
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-10}"

# Profiled on this Thor (cosmos_policy/scripts/profile_so101_latency.py):
#   denoise=10 -> 2551 ms/chunk   DiT 1862 ms (73%)  VAE encode 590 ms (23%)
#   denoise=3  -> ~1230 ms        DiT ~560 ms        VAE encode 590 ms
# The DiT term is the only one denoising steps touch; VAE encode is a fixed ~590 ms floor.
#
# DO NOT lower this to buy speed. A 180-frame teacher-forced MAE sweep made denoise=2..10
# look equivalent, but that metric hides what actually matters: in a closed-loop replay the
# commanded displacement per chunk is 7.7 deg mean / 41.8 deg max at denoise=10 versus
# 22.4 deg mean / 52.1 deg max at denoise=3 -- roughly 3x more aggressive -- while tracking
# error is unchanged (66.7 vs 70.8 deg). Fewer denoising steps buy speed by making the
# actions noisier, which on hardware looks like the arm slamming into the table and
# oscillating.
NUM_DENOISING_STEPS_ACTION="${NUM_DENOISING_STEPS_ACTION:-10}"

# MEASURED ON THIS THOR (2026-07-22, synthetic-observation smoke test, bf16, 3 cameras,
# num_denoising_steps_action=10): ~5.3 s for the first chunk (warmup), then ~2.6-2.8 s
# per chunk steady state -- i.e. ~0.37 Hz replanning.
#
# Async inference only overlaps while a chunk covers more wall-clock time than the next
# chunk takes to compute. With ACTIONS_PER_CHUNK=10, staying ahead of a ~2.7 s inference
# needs 10/FPS >= 2.7, i.e. FPS <= ~3.7. At FPS=30 the arm would execute 0.33 s of motion
# and then sit still for ~2.4 s, over and over.
#
# So default to FPS=3 (the same rate the LingBot-VA deployment settled on). Actions are
# absolute joint targets, so the demonstrated path is preserved -- it is just executed
# ~10x slower than the 30 fps training data. For a first real-robot run, slower is good.
# Lowering NUM_DENOISING_STEPS_ACTION would buy speed at the cost of action quality; that
# tradeoff has NOT been characterized for this checkpoint, so it is left alone.
# fps is an int on both sides (draccus rejects 2.5), so the choice is 2 or 3.
# Bounded by the real latency: at denoise=10 a chunk costs ~2.55 s, and the client keeps the
# arm moving only while CHUNK_SIZE_THRESHOLD x (ACTIONS_PER_CHUNK / FPS) >= 2.55 s.
#   FPS=2 -> 0.7 x (10/2) = 3.50 s of queued motion. Comfortable.
#   FPS=3 -> 0.7 x (10/3) = 2.33 s. Short by ~0.2 s; needs CHUNK_SIZE_THRESHOLD=0.85,
#            which then leaves only ~0.1 s of margin.
# 2 is chosen for robustness. This is the honest ceiling of the device: 2.55 s of inference
# per 10 executed steps cannot be made to look like 30 fps motion, so the arm replays the
# demonstrated path roughly 15x slower. Absolute joint targets mean the path is preserved.
FPS="${FPS:-2}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-2.6}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2.0}"

# Camera keys as produced by the LeRobot SO101Follower client. Must match the client's
# --robot.cameras names AND the training data (observation.images.{front,right,wrist}).
PRIMARY_CAMERA_KEY="${PRIMARY_CAMERA_KEY:-front}"
LEFT_WRIST_CAMERA_KEY="${LEFT_WRIST_CAMERA_KEY:-right}"
RIGHT_WRIST_CAMERA_KEY="${RIGHT_WRIST_CAMERA_KEY:-wrist}"

# Safety clamps, in physical units (body joints degrees, gripper 0..100).
# These are deliberately conservative. Given the known chunk-boundary jumps, only ever
# tighten these -- never widen them, and never set them to 0 (which disables the clamp).
# Measured over all 100 episodes, the motion the demonstrations themselves need within one
# 10-step window is p99 19.7 / 46.4 / 44.9 / 22.2 / 31.8 / 18.4 deg for
# pan/lift/elbow/wrist_flex/wrist_roll/gripper. 8.0 is therefore ~6x too small on the two
# joints that do most of the work, and the arm can only crawl. It is left conservative here
# ON PURPOSE -- raise it deliberately, with a person on the e-stop:
#     MAX_DELTA_FROM_OBSERVATION=50 ./run_cosmos_three_cubes_server.sh
MAX_DELTA_FROM_OBSERVATION="${MAX_DELTA_FROM_OBSERVATION:-8.0}"

# Raised from 8.0 on evidence: ground-truth gripper open/close transitions are 10-12 deg
# over 5-6 frames, so an 8 deg cap CANNOT complete a single grasp within a chunk -- the
# gripper would need two or more replans to close, by which time the plan has moved on.
# This clamp bounds the gripper only; per-step speed is still limited by
# MAX_GRIPPER_STEP_DELTA below.
MAX_GRIPPER_DELTA_FROM_OBSERVATION="${MAX_GRIPPER_DELTA_FROM_OBSERVATION:-25.0}"
MAX_STEP_DELTA="${MAX_STEP_DELTA:-4.0}"
MAX_GRIPPER_STEP_DELTA="${MAX_GRIPPER_STEP_DELTA:-5.0}"

# Stage 1 default: no motion. Override explicitly to go live.
DRY_RUN="${DRY_RUN:-true}"

for f in "${CKPT_PATH}" "${DATASET_STATS_PATH}" "${T5_EMBEDDINGS_PATH}"; do
  if [[ ! -e "${f}" ]]; then
    echo "Missing required artifact: ${f}" >&2
    exit 2
  fi
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
  echo "Port ${PORT} is already in use -- another policy server is likely running." >&2
  echo "Pick a free PORT or stop the other server first." >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "true" ]]; then
  echo "*** DRY_RUN=false: the arm WILL move. Stand by the e-stop. ***" >&2
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${LEROBOT_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" -m cosmos_policy.experiments.robot.so101_async_deploy \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --ckpt_path="${CKPT_PATH}" \
  --cosmos_config="${COSMOS_CONFIG}" \
  --dataset_stats_path="${DATASET_STATS_PATH}" \
  --t5_text_embeddings_path="${T5_EMBEDDINGS_PATH}" \
  --primary_camera_key="${PRIMARY_CAMERA_KEY}" \
  --left_wrist_camera_key="${LEFT_WRIST_CAMERA_KEY}" \
  --right_wrist_camera_key="${RIGHT_WRIST_CAMERA_KEY}" \
  --num_denoising_steps_action="${NUM_DENOISING_STEPS_ACTION}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --max_delta_from_observation="${MAX_DELTA_FROM_OBSERVATION}" \
  --max_gripper_delta_from_observation="${MAX_GRIPPER_DELTA_FROM_OBSERVATION}" \
  --max_step_delta="${MAX_STEP_DELTA}" \
  --max_gripper_step_delta="${MAX_GRIPPER_STEP_DELTA}" \
  --dry_run_zero_actions="${DRY_RUN}"

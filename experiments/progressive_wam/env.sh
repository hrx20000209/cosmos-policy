# Environment for the progressive-WAM experiments.
#   source experiments/progressive_wam/env.sh <gpu-index>
#
# LIBERO renders on the CPU through a user-space OSMesa build because this user
# has no /dev/dri access; CUDA inference still runs on the selected GPU. Do not
# extrapolate the CPU-rendering episode wall time to Jetson.
GPU="${1:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_EGL_DEVICE_ID="${GPU}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/rxhuang/Projects/cosmos-policy:${PYTHONPATH}"
echo "[progressive-wam] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_GL=${MUJOCO_GL}"

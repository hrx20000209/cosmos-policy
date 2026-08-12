#!/usr/bin/env bash
# Frozen E12 worker environment. Source with the physical GPU id as $1.
# Explicit EGL is required: the shared helper defaults to OSMesa, which fails
# on this server's PyOpenGL build before policy construction.
set -u
cd /home/rxhuang/Projects/cosmos-policy
source .venv/bin/activate
export PYTHONPATH=.
export HF_HOME=/data/hf_cache
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${1}"
export EVAL_PHYSICAL_GPU="${1}"
export MUJOCO_EGL_DEVICE_ID="${1}"

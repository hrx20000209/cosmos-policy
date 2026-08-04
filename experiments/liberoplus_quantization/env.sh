# Source before any LIBERO run on this host.
# GPU render nodes (/dev/dri) are group video/render — this user is NOT in them,
# so mesa EGL fails with "Permission denied". NVIDIA headless EGL uses /dev/nvidia*
# (world-accessible) instead. Force the NVIDIA EGL vendor library.
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export PYOPENGL_PLATFORM=egl
# per-process: also set MUJOCO_EGL_DEVICE_ID / CUDA_VISIBLE_DEVICES to the target GPU
export HF_HOME=/data/rxhuang/hf_cache
export UV_CACHE_DIR=/data/rxhuang/uv_cache
export CKPT_LOCAL=/data/rxhuang/models/cosmos-policy-libero-2b
export TOKENIZERS_PARALLELISM=false
# Xet finalize step hangs on this host -> use standard robust LFS download.
# HF auth token is read automatically from $HF_HOME/token (chmod 600), not stored here.
export HF_HUB_DISABLE_XET=1
# LIBERO-Plus env_wrapper imports `wand` (ImageMagick). No system libMagickWand
# (no sudo) -> installed IM into a userspace conda prefix. APPEND (not prepend)
# so it never shadows the venv's own libs.
export MAGICK_HOME=/data/rxhuang/envs/imagemagick
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$MAGICK_HOME/lib

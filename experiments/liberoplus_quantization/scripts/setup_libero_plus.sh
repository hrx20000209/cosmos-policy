#!/usr/bin/env bash
# Install LIBERO-Plus as a drop-in replacement for `libero` in the cosmos .venv,
# unzip assets, repoint ~/.libero/config.yaml. Run AFTER original-LIBERO jobs exit.
set -euo pipefail
LP=/data/rxhuang/LIBERO-plus
VENV=/home/rxhuang/Projects/cosmos-policy/.venv
export UV_CACHE_DIR=/data/rxhuang/uv_cache UV_LINK_MODE=copy
cd /home/rxhuang/Projects/cosmos-policy

echo "=== 1. unzip assets into $LP/libero/libero/assets ==="
if [ ! -d "$LP/libero/libero/assets/turbosquid_objects" ]; then
  cd /data/rxhuang/LIBERO-plus-assets
  unzip -q -o assets.zip -d /tmp/lp_assets_extract
  # the zip may contain an 'assets/' top dir; place its contents under libero/libero/assets
  if [ -d /tmp/lp_assets_extract/assets ]; then
    cp -rn /tmp/lp_assets_extract/assets/* "$LP/libero/libero/assets/"
  else
    cp -rn /tmp/lp_assets_extract/* "$LP/libero/libero/assets/"
  fi
  echo "assets placed"; ls "$LP/libero/libero/assets" | head
  cd /home/rxhuang/Projects/cosmos-policy
else
  echo "assets already present"
fi

echo "=== 2. pip install -e LIBERO-plus (drop-in over libero) ==="
uv pip install --python $VENV/bin/python -e "$LP" --no-deps
uv pip install --python $VENV/bin/python scikit-image wand 2>/dev/null || true

echo "=== 3. repoint ~/.libero/config.yaml to LIBERO-plus ==="
cp ~/.libero/config.yaml ~/.libero/config.yaml.orig_libero.bak 2>/dev/null || true
cat > ~/.libero/config.yaml <<YAML
assets: $LP/libero/libero/./assets
bddl_files: $LP/libero/libero/./bddl_files
benchmark_root: $LP/libero/libero
datasets: /data/rxhuang/libero/datasets
init_states: $LP/libero/libero/./init_files
YAML
echo "config.yaml ->"; cat ~/.libero/config.yaml

echo "=== 4. verify drop-in: perturbed task counts ==="
$VENV/bin/python - <<'PY'
import libero, os
print("libero pkg:", libero.__file__)
from libero.libero import benchmark
bd = benchmark.get_benchmark_dict()
for s in ["libero_spatial","libero_object","libero_goal","libero_10"]:
    try:
        n = bd[s]().n_tasks
        print(f"  {s}: n_tasks={n}")
    except Exception as e:
        print(f"  {s}: ERR {e}")
PY
echo "SETUP_LIBERO_PLUS_DONE"

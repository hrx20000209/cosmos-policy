#!/usr/bin/env bash
set -euo pipefail

# Never place a Hugging Face access token in this script or in shell history.
# Example: export HF_TOKEN='...'
: "${HF_TOKEN:?Please export HF_TOKEN before running this script}"
TOK="$HF_TOKEN"
ROOT="${HF_HOME:-/data/rxhuang/hf_cache}/hub/models--nvidia--Cosmos-Predict2-2B-Video2World"
REV=f50c09f5d8ab133a90cac3f4886a6471e9ba3f18
B=$ROOT/blobs
base=https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/resolve/main

dl () { # url blobname expected_size snapshot_path blob_relative_path
  local url=$1 blob=$2 sz=$3 snap=$4
  echo "[$(date +%T)] downloading $blob"
  curl -fsSL -C - -H "Authorization: Bearer $TOK" -o "$B/$blob.incomplete" "$url"
  local got
  got=$(stat -c %s "$B/$blob.incomplete")
  if [ "$got" = "$sz" ]; then
    mv "$B/$blob.incomplete" "$B/$blob"
    mkdir -p "$(dirname "$ROOT/snapshots/$REV/$snap")"
    ln -sf "$5/$blob" "$ROOT/snapshots/$REV/$snap"
    echo "[$(date +%T)] OK $snap ($got bytes)"
  else
    echo "[$(date +%T)] SIZE MISMATCH $blob got=$got want=$sz" >&2
    return 1
  fi
}

mkdir -p "$B"
dl "$base/model-480p-16fps.pt" fbc4f05d948078539cb5d7a8e59b6f40f940e4836b5b8a31dcab03e3e807a6f0 3913017214 model-480p-16fps.pt ../../blobs &
dl "$base/tokenizer/tokenizer.pth" 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 507609880 tokenizer/tokenizer.pth ../../../blobs &
wait
echo "ALL_BASE_DONE"

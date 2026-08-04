#!/usr/bin/env python3
"""Precompute T5-11B text embeddings for LIBERO-Plus task instructions.

The checkpoint ships only 40 original-instruction embeddings. LIBERO-Plus
perturbed tasks (esp. the 'Language Instructions' category) use paraphrased /
suffixed instructions that miss that cache, which otherwise triggers an
on-demand 43 GB T5-11B load inside every rollout shard. This script loads
T5-11B ONCE, encodes all instructions for a given task list, merges them into
the original cache, and writes an augmented pkl that all shards can load cheaply.

Each embedding is (1, 512, 1024) bf16 ~= 1 MB, so scope the task list to what you
will actually evaluate (a few hundred -> a few hundred MB).

Run (needs a free GPU with ~21 GB):
  source env.sh; CUDA_VISIBLE_DEVICES=g MUJOCO_EGL_DEVICE_ID=g \
    .venv/bin/python .../precompute_t5.py --tasklist <json> --out <pkl> \
      [--include-language]   # else skips 'Language Instructions' (canonical prefix used)
"""
import argparse
import json
import os
import pickle
import sys

import torch

T5_LOCAL = "/data/rxhuang/models/t5-11b"
ORIG_CACHE = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"


def collect_instructions(tasklist_path, include_language):
    from libero.libero import benchmark
    tasks = json.load(open(tasklist_path))
    tasks = tasks["tasks"] if isinstance(tasks, dict) else tasks
    bmc = {}
    out = {}   # instruction -> None (dedup)
    for t in tasks:
        if not include_language and t.get("category") == "Language Instructions":
            continue
        s = t["suite"]
        if s not in bmc:
            bmc[s] = benchmark.get_benchmark_dict()[s]()
        lang = bmc[s].get_task(t["id"] - 1).language
        out[lang] = None
    return list(out.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasklist", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-language", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    instrs = collect_instructions(args.tasklist, args.include_language)
    print(f"collected {len(instrs)} unique instructions "
          f"({'incl' if args.include_language else 'excl'} language perturbations)")

    # start from the original 40-entry cache (keep those exact tensors)
    cache = {}
    if os.path.exists(ORIG_CACHE):
        cache = pickle.load(open(ORIG_CACHE, "rb"))
        print(f"loaded {len(cache)} original embeddings")

    todo = [s for s in instrs if s not in cache]
    print(f"{len(todo)} need computing ({len(instrs) - len(todo)} already cached)")
    if not todo:
        pickle.dump(cache, open(args.out, "wb"))
        print(f"nothing to compute; wrote {args.out} with {len(cache)} entries")
        return

    from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading T5-11B from {T5_LOCAL} on {dev} (bf16, ~21 GB)...")
    enc = CosmosT5TextEncoder(model_name=T5_LOCAL, device=dev, local_files_only=True)
    print("T5-11B loaded.")

    n = 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        emb = enc.encode_prompts(chunk, max_length=512)   # (B, 512, 1024)
        emb = emb.to(torch.bfloat16).cpu()
        for j, s in enumerate(chunk):
            cache[s] = emb[j:j + 1].clone()               # (1, 512, 1024)
        n += len(chunk)
        if n % 40 == 0 or n >= len(todo):
            print(f"  encoded {n}/{len(todo)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pickle.dump(cache, open(args.out, "wb"))
    sz = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {len(cache)} embeddings ({sz:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())

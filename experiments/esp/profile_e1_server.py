#!/usr/bin/env python3
"""Clean-GPU server-side F1/P1 cost characterization for ESP E1.

Cosmos exposes one joint VAE encode rather than a per-camera encoding API, so
this profiles end-to-end F1 versus legal P1 reuse and labels their difference
as an *end-to-end visual-refresh delta*, not a falsely isolated VAE kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.esp.run_e2_e3_pilot_shard import load_request, render, run_action
from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS, DEFAULT_T5_EMBEDDINGS, ORIGINAL_CHECKPOINT,
    atomic_write_json, build_model, checkpoint_contract, configure_libero,
    read_jsonl, set_up_cuda,
)


def summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
            "p05": float(np.quantile(array, .05)), "p95": float(np.quantile(array, .95))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=Path, default=Path("reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("reports/esp/E1_RESULT.json"))
    args = parser.parse_args()
    if args.repeats < 50 or args.warmup < 1:
        raise ValueError("protocol requires warmup >=1 and formal repeats >=50")
    entry = read_jsonl(args.state_index)[args.index]
    source, target = load_request(entry)
    checkpoint, gpu = checkpoint_contract(ORIGINAL_CHECKPOINT), set_up_cuda(0.40)
    configure_libero(entry)
    source_obs, target_obs = render(entry, source), render(entry, target)
    previous = torch.from_numpy(np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
    cfg, stats, model = build_model(checkpoint=ORIGINAL_CHECKPOINT, dataset_stats=DEFAULT_DATASET_STATS, t5_embeddings=DEFAULT_T5_EMBEDDINGS)
    f1_cuda: list[float] = []
    p1_cuda: list[float] = []
    f1_wall: list[float] = []
    p1_wall: list[float] = []
    try:
        for _ in range(args.warmup):
            run_action(cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=None)
            run_action(cfg, model, stats, source_obs, entry["instruction"], int(entry["seed"]), previous=previous)
        for _ in range(args.repeats):
            _, _, _, cuda_ms, wall_ms = run_action(cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=None)
            f1_cuda.append(cuda_ms); f1_wall.append(wall_ms)
            _, _, _, cuda_ms, wall_ms = run_action(cfg, model, stats, source_obs, entry["instruction"], int(entry["seed"]), previous=previous)
            p1_cuda.append(cuda_ms); p1_wall.append(wall_ms)
    finally:
        model = None
        torch.cuda.empty_cache()
    f1, p1 = summary(f1_cuda), summary(p1_cuda)
    delta = float(f1["median"] - p1["median"])
    payload = {
        "schema_version": 1, "status": "PASS", **checkpoint, "gpu": gpu, "state_id": entry["state_key"],
        "warmup": args.warmup, "repeats": args.repeats, "denoising_steps": 1, "value_used": False,
        "finetuning_used": False, "server_only": True,
        "camera_compute_separable": False,
        "camera_separability_decision": "NO_GO",
        "architecture_decision": "E1_SELECTIVE_SENSING_ARCH_NO_GO",
        "reason": "both camera streams are concatenated into a joint temporally causal VAE input; no independent camera encode path avoids the other stream's VAE work",
        "f1_cuda_ms": f1, "p1_cuda_ms": p1, "f1_wall_ms": summary(f1_wall), "p1_wall_ms": summary(p1_wall),
        "end_to_end_visual_refresh_delta_cuda_ms_median": delta,
        "end_to_end_visual_refresh_delta_fraction_of_f1": float(delta / f1["median"]),
        "interpretation": "This difference is a conservative end-to-end F1-vs-P1 route delta, not a claimed isolated VAE kernel time.",
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "f1_median": f1["median"], "p1_median": p1["median"]}))


if __name__ == "__main__":
    main()

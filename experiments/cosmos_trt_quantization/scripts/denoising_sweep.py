#!/usr/bin/env python3
"""BF16 eager denoising-step sweep with 100 warmups and 300 measurements."""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"
OLD = ROOT / "experiments/liberoplus_quantization/scripts"
sys.path[:0] = [str(ROOT), str(OLD)]
from quant_microbench import CKPT_DIR, Cfg, get_fixed_observation  # noqa: E402

_magick_wand = "/data/rxhuang/envs/imagemagick/lib/libMagickWand.so"
if Path(_magick_wand).exists():
    ctypes.CDLL(_magick_wand, mode=ctypes.RTLD_GLOBAL)


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def errors(action, baseline):
    a = np.asarray(action, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    diff = a - b
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    jerk = np.diff(a, n=3, axis=0)
    return {
        "action_cosine_vs_bf16_5step": float(np.dot(a.ravel(), b.ravel()) / denom),
        "action_l1_vs_bf16_5step": float(np.mean(np.abs(diff))),
        "action_l2_vs_bf16_5step": float(np.linalg.norm(diff)),
        "action_max_abs_vs_bf16_5step": float(np.max(np.abs(diff))),
        "action_jerk_l2_mean": float(np.linalg.norm(jerk, axis=1).mean()),
        "chunk_boundary_variation_l2": float(np.linalg.norm(a[-1] - a[0])),
        "nan_inf_count": int((~np.isfinite(a)).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=195)
    args = parser.parse_args()

    from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, load_dataset_stats
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size

    cfg = Cfg(num_denoising_steps_action=5)
    dataset_stats = load_dataset_stats(f"{CKPT_DIR}/libero_dataset_statistics.json")
    observation, label = get_fixed_observation(cfg, get_image_resize_size(cfg.model_family))
    cached = pickle.load(open(f"{CKPT_DIR}/libero_t5_embeddings.pkl", "rb"))
    matches = [key for key in cached if label.startswith(key)]
    embedding_key = label if label in cached else max(matches, key=len)
    embedding = cached[embedding_key].detach().float().cpu().numpy()
    model, _ = get_model(cfg)
    model.eval()

    net_events = []

    def pre(_module, _inputs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        net_events.append([event, None])

    def post(_module, _inputs, output):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        net_events[-1][1] = event
        return output

    handles = [model.net.register_forward_pre_hook(pre), model.net.register_forward_hook(post)]

    def one(steps):
        net_events.clear()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        begin.record()
        output = get_action(
            cfg,
            model,
            dataset_stats,
            observation,
            embedding,
            seed=args.seed,
            num_denoising_steps_action=steps,
            generate_future_state_and_value_in_parallel=False,
            batch_size=1,
        )
        end.record()
        torch.cuda.synchronize()
        action = output["actions"] if isinstance(output, dict) else output
        step_ms = [start.elapsed_time(finish) for start, finish in net_events]
        return (
            np.asarray(action),
            begin.elapsed_time(end),
            (time.perf_counter_ns() - wall_start) / 1e6,
            step_ms,
        )

    # Compute the exact reference first; every call is deterministic for this seed.
    baseline, _, _, _ = one(5)
    run_id = time.strftime("%Y%m%dT%H%M%S_bf16_eager")
    raw_path = EXP / "raw/denoising_sweep_microbenchmark.csv"
    summary_path = EXP / "summaries/denoising_sweep_summary.csv"
    raw_exists = raw_path.exists()
    raw_fields = [
        "run_id", "backend", "denoising_steps", "iteration", "policy_cuda_ms",
        "policy_wall_ms", "denoiser_total_ms", "per_step_mean_ms",
        "gpu_allocated_mb", "gpu_reserved_mb",
    ]
    summary_rows = []
    with raw_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        if not raw_exists:
            writer.writeheader()
        for steps in (1, 2, 3, 4, 5):
            for index in range(args.warmup):
                one(steps)
                if (index + 1) % 25 == 0:
                    print(f"steps={steps} warmup={index + 1}/{args.warmup}", flush=True)
            cuda_values, wall_values, denoiser_values = [], [], []
            last_action = None
            for index in range(args.iters):
                last_action, cuda_ms, wall_ms, step_ms = one(steps)
                cuda_values.append(cuda_ms)
                wall_values.append(wall_ms)
                denoiser_values.append(sum(step_ms))
                writer.writerow(
                    {
                        "run_id": run_id,
                        "backend": "bf16_eager",
                        "denoising_steps": steps,
                        "iteration": index,
                        "policy_cuda_ms": round(cuda_ms, 5),
                        "policy_wall_ms": round(wall_ms, 5),
                        "denoiser_total_ms": round(sum(step_ms), 5),
                        "per_step_mean_ms": round(float(np.mean(step_ms)), 5),
                        "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 3),
                        "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 3),
                    }
                )
                handle.flush()
                if (index + 1) % 50 == 0:
                    print(f"steps={steps} measure={index + 1}/{args.iters}", flush=True)
            metric = errors(last_action, baseline)
            summary_rows.append(
                {
                    "run_id": run_id,
                    "backend": "bf16_eager",
                    "denoising_steps": steps,
                    **{f"policy_cuda_ms_{key}": value for key, value in stats(cuda_values).items()},
                    **{f"policy_wall_ms_{key}": value for key, value in stats(wall_values).items()},
                    **{f"denoiser_total_ms_{key}": value for key, value in stats(denoiser_values).items()},
                    **metric,
                }
            )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    for handle in handles:
        handle.remove()
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""100-warmup / 300-iteration full policy benchmark using one reused TRT engine."""

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
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"
OLD = ROOT / "experiments/liberoplus_quantization/scripts"
sys.path[:0] = [str(ROOT), str(OLD), str(EXP)]

from quant_microbench import CKPT_DIR, Cfg, get_fixed_observation  # noqa: E402

_magick_wand = "/data/rxhuang/envs/imagemagick/lib/libMagickWand.so"
if Path(_magick_wand).exists():
    ctypes.CDLL(_magick_wand, mode=ctypes.RTLD_GLOBAL)


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


class EngineNetAdapter(nn.Module):
    """Match MinimalV1LVGDiT kwargs while calling the fixed TRT engine."""

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.step_events = []

    def forward(
        self,
        x_B_C_T_H_W,
        timesteps_B_T,
        crossattn_emb,
        condition_video_input_mask_B_C_T_H_W=None,
        fps=None,
        padding_mask=None,
        **_kwargs,
    ):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = self.engine(
            x_B_C_T_H_W,
            timesteps_B_T,
            crossattn_emb,
            condition_video_input_mask_B_C_T_H_W,
            fps,
            padding_mask,
        )
        end.record()
        self.step_events.append((start, end))
        return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("bf16_trt", "fp8_trt"), required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=195)
    args = parser.parse_args()

    import torch_tensorrt  # noqa: F401
    from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, load_dataset_stats
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size

    cfg = Cfg(num_denoising_steps_action=5)
    dataset_stats = load_dataset_stats(f"{CKPT_DIR}/libero_dataset_statistics.json")
    observation, label = get_fixed_observation(cfg, get_image_resize_size(cfg.model_family))
    with open(f"{CKPT_DIR}/libero_t5_embeddings.pkl", "rb") as handle:
        cached = pickle.load(handle)
    matches = [key for key in cached if label.startswith(key)]
    embedding_key = label if label in cached else max(matches, key=len)
    embedding = cached[embedding_key].detach().float().cpu().numpy()
    model, _ = get_model(cfg)
    model.eval()

    stem = args.backend.split("_")[0]
    ts_path = EXP / f"engines/cosmos_denoiser_{stem}.ts"
    engine = torch.jit.load(str(ts_path)).cuda().eval()
    old_net = model.net
    adapter = EngineNetAdapter(engine).cuda().eval()
    model.net = adapter
    del old_net
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def one():
        adapter.step_events.clear()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        output = get_action(
            cfg,
            model,
            dataset_stats,
            observation,
            embedding,
            seed=args.seed,
            num_denoising_steps_action=5,
            generate_future_state_and_value_in_parallel=False,
            batch_size=1,
        )
        end.record()
        torch.cuda.synchronize()
        action = output["actions"] if isinstance(output, dict) else output
        step_ms = [a.elapsed_time(b) for a, b in adapter.step_events]
        return (
            np.asarray(action),
            start.elapsed_time(end),
            (time.perf_counter_ns() - wall_start) / 1e6,
            step_ms,
        )

    for index in range(args.warmup):
        one()
        if (index + 1) % 25 == 0:
            print(f"warmup {index + 1}/{args.warmup}", flush=True)

    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{args.backend}"
    path = EXP / "raw/policy_microbenchmark_trt.csv"
    exists = path.exists()
    fields = [
        "run_id", "backend", "iteration", "policy_cuda_ms", "policy_wall_ms",
        "step_1_ms", "step_2_ms", "step_3_ms", "step_4_ms", "step_5_ms",
        "gpu_allocated_mb", "gpu_reserved_mb",
    ]
    cuda_values, wall_values, step_values = [], [], []
    last_action = None
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for index in range(args.iters):
            last_action, cuda_ms, wall_ms, steps = one()
            if len(steps) != 5:
                raise RuntimeError(f"Expected 5 reused TRT calls, observed {len(steps)}")
            cuda_values.append(cuda_ms)
            wall_values.append(wall_ms)
            step_values.extend(steps)
            writer.writerow(
                {
                    "run_id": run_id,
                    "backend": args.backend,
                    "iteration": index,
                    "policy_cuda_ms": cuda_ms,
                    "policy_wall_ms": wall_ms,
                    **{f"step_{i + 1}_ms": value for i, value in enumerate(steps)},
                    "gpu_allocated_mb": torch.cuda.memory_allocated() / 2**20,
                    "gpu_reserved_mb": torch.cuda.memory_reserved() / 2**20,
                }
            )
            handle.flush()
            if (index + 1) % 50 == 0:
                print(f"measure {index + 1}/{args.iters}", flush=True)

    reference = np.load(EXP / "raw/torchao_action_bf16.npy").astype(np.float64)
    action = last_action.astype(np.float64)
    diff = action - reference
    denom = np.linalg.norm(action) * np.linalg.norm(reference)
    summary = {
        "run_id": run_id,
        "backend": args.backend,
        "warmup": args.warmup,
        "iterations": args.iters,
        "cuda": stats(cuda_values),
        "wall": stats(wall_values),
        "denoiser_step": stats(step_values),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20,
        "action_cosine": float(np.dot(action.ravel(), reference.ravel()) / denom),
        "action_l1": float(np.mean(np.abs(diff))),
        "action_l2": float(np.linalg.norm(diff)),
        "action_max_abs": float(np.max(np.abs(diff))),
        "nan_inf_count": int((~np.isfinite(action)).sum()),
        "engine_reused_per_policy_call": True,
        "engine_calls_per_policy_call": 5,
    }
    (EXP / f"summaries/policy_microbenchmark_{args.backend}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    np.save(EXP / f"raw/policy_action_{args.backend}.npy", last_action)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

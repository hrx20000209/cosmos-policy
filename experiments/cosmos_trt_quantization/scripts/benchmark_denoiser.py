#!/usr/bin/env python3
"""100-warmup / 1000-iteration single-denoiser CUDA-event benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"
OLD = ROOT / "experiments/liberoplus_quantization/scripts"
sys.path[:0] = [str(ROOT), str(OLD), str(EXP)]
from quant_microbench import Cfg  # noqa: E402
import quant_lib  # noqa: E402
from wrappers.cosmos_denoiser_wrapper import CosmosDenoiserWrapper, fixture_args  # noqa: E402


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("bf16_eager", "bf16_trt", "fp8_trt"), required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    fixture = torch.load(EXP / "calibration/fixed_denoiser_inputs.pt", map_location="cpu", weights_only=True)
    inputs = fixture_args(fixture)

    if args.backend == "bf16_eager":
        from cosmos_policy.experiments.robot.cosmos_utils import get_model

        model, _ = get_model(Cfg())
        model.eval()
        quant_lib.configure_model_precision(model, "bf16")
        module = CosmosDenoiserWrapper(model.net).eval()
        reference = None
    else:
        import torch_tensorrt  # noqa: F401 - registers serialized TRT ops

        stem = "bf16" if args.backend == "bf16_trt" else "fp8"
        ts_path = EXP / f"engines/cosmos_denoiser_{stem}.ts"
        ep_path = EXP / f"engines/cosmos_denoiser_{stem}.ep"
        if ts_path.exists():
            module = torch.jit.load(str(ts_path)).cuda().eval()
        else:
            module = torch.export.load(ep_path).module().cuda().eval()
        reference = None

    def one():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start.record()
        with torch.inference_mode():
            output = module(*inputs)
        end.record()
        torch.cuda.synchronize()
        return output, start.elapsed_time(end), (time.perf_counter_ns() - wall_start) / 1e6

    for index in range(args.warmup):
        reference, _, _ = one()
        if (index + 1) % 25 == 0:
            print(f"warmup {index + 1}/{args.warmup}", flush=True)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{args.backend}"
    path = EXP / "raw/denoiser_microbenchmark.csv"
    exists = path.exists()
    fields = ["run_id", "backend", "iteration", "cuda_ms", "wall_ms", "gpu_allocated_mb", "gpu_reserved_mb"]
    cuda_values, wall_values = [], []
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for index in range(args.iters):
            output, cuda_ms, wall_ms = one()
            cuda_values.append(cuda_ms)
            wall_values.append(wall_ms)
            writer.writerow(
                {
                    "run_id": run_id,
                    "backend": args.backend,
                    "iteration": index,
                    "cuda_ms": round(cuda_ms, 6),
                    "wall_ms": round(wall_ms, 6),
                    "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 3),
                    "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 3),
                }
            )
            handle.flush()
            if (index + 1) % 100 == 0:
                print(f"measure {index + 1}/{args.iters}", flush=True)
    summary = {
        "run_id": run_id,
        "backend": args.backend,
        "warmup": args.warmup,
        "iterations": args.iters,
        "cuda": stats(cuda_values),
        "wall": stats(wall_values),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "finite": bool(torch.isfinite(output.float()).all()),
    }
    (EXP / f"summaries/denoiser_microbenchmark_{args.backend}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

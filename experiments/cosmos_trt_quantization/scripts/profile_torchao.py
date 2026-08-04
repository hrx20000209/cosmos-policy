#!/usr/bin/env python3
"""Reproducible torchao latency and kernel profiling on one fixed policy input.

Run one mode per process so quantized models never coexist on the GPU.  Latency
mode uses 30 warmups and 200 measured full policy calls.  Torch-profiler and
Nsight modes use the same 30-call warmup followed by one traced policy call.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import pickle
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"
OLD = ROOT / "experiments/liberoplus_quantization/scripts"
sys.path[:0] = [str(ROOT), str(OLD)]

from quant_microbench import CKPT_DIR, Cfg, get_fixed_observation  # noqa: E402
import quant_lib  # noqa: E402

_magick_wand = "/data/rxhuang/envs/imagemagick/lib/libMagickWand.so"
if Path(_magick_wand).exists():
    ctypes.CDLL(_magick_wand, mode=ctypes.RTLD_GLOBAL)


MODE_SLUG = {
    "bf16": "bf16",
    "fp8_backbone": "fp8",
    "int8_weight_only": "int8_wo",
    "int8_backbone": "int8_w8a8",
    "int4_weight_only": "int4_wo",
    "fake_int4_backbone": "fake_int4",
}


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def append_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_kernel_trace(trace_path: Path, mode: str, run_id: str) -> list[dict]:
    """Aggregate raw CUDA kernel events emitted by Kineto."""
    payload = json.loads(trace_path.read_text())
    totals: dict[str, list[float]] = defaultdict(list)
    for event in payload.get("traceEvents", []):
        category = str(event.get("cat", "")).lower()
        if event.get("ph") == "X" and ("kernel" in category or category == "gpu_memcpy"):
            totals[str(event.get("name", "unknown"))].append(float(event.get("dur", 0.0)))
    rows = []
    for name, durations_us in sorted(totals.items(), key=lambda item: -sum(item[1])):
        rows.append(
            {
                "run_id": run_id,
                "quant_mode": mode,
                "kernel_name": name,
                "count": len(durations_us),
                "total_cuda_us": round(sum(durations_us), 3),
                "mean_cuda_us": round(sum(durations_us) / len(durations_us), 3),
                "max_cuda_us": round(max(durations_us), 3),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODE_SLUG), required=True)
    parser.add_argument("--phase", choices=("latency", "torchprof", "nsys"), default="latency")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    run_id = args.run_id or f"{time.strftime('%Y%m%dT%H%M%S')}_{MODE_SLUG[args.mode]}_{uuid.uuid4().hex[:6]}"

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size

    cfg = Cfg(num_denoising_steps_action=5)
    dataset_stats = load_dataset_stats(f"{CKPT_DIR}/libero_dataset_statistics.json")
    observation, task_label = get_fixed_observation(cfg, get_image_resize_size(cfg.model_family))
    with open(f"{CKPT_DIR}/libero_t5_embeddings.pkl", "rb") as handle:
        cached_embeddings = pickle.load(handle)
    prefix_matches = [key for key in cached_embeddings if task_label.startswith(key)]
    embedding_key = task_label if task_label in cached_embeddings else (
        max(prefix_matches, key=len) if prefix_matches else None
    )
    if embedding_key is None:
        raise KeyError(f"No cached base instruction matches {task_label!r}")
    task_embedding = cached_embeddings[embedding_key]
    if isinstance(task_embedding, torch.Tensor):
        task_embedding = task_embedding.detach().float().cpu().numpy()
    model, _ = get_model(cfg)
    model.eval()
    quant_lib.configure_model_precision(model, "bf16")
    audit = quant_lib.apply_quantization(model.net, args.mode, group_size=128)
    audit.update(quant_lib.verify_quantized(model.net))
    audit.update({"run_id": run_id, "phase": args.phase, "warmup": args.warmup, "iters": args.iters})
    console_audit = {key: value for key, value in audit.items() if key != "candidate_module_names"}
    print(json.dumps(console_audit, indent=2, default=str), flush=True)

    step_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def net_pre(_module, _inputs):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        step_events.append((start, None))
        if args.phase == "nsys":
            torch.cuda.nvtx.range_push("cosmos_denoiser_step")

    def net_post(_module, _inputs, output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        start, _ = step_events[-1]
        step_events[-1] = (start, end)
        if args.phase == "nsys":
            torch.cuda.nvtx.range_pop()
        return output

    pre_handle = model.net.register_forward_pre_hook(net_pre)
    post_handle = model.net.register_forward_hook(net_post)

    def one_call():
        step_events.clear()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter_ns()
        start_event.record()
        output = get_action(
            cfg,
            model,
            dataset_stats,
            observation,
            task_embedding,
            seed=args.seed,
            num_denoising_steps_action=5,
            generate_future_state_and_value_in_parallel=False,
            batch_size=1,
        )
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        cuda_ms = start_event.elapsed_time(end_event)
        steps = [start.elapsed_time(end) for start, end in step_events]
        action = output["actions"] if isinstance(output, dict) else output
        return np.asarray(action), cuda_ms, wall_ms, steps

    for index in range(args.warmup):
        one_call()
        if (index + 1) % 10 == 0:
            print(f"warmup {index + 1}/{args.warmup}", flush=True)

    slug = MODE_SLUG[args.mode]
    if args.phase == "latency":
        rows = []
        cuda_values, wall_values, all_step_values = [], [], []
        last_action = None
        for index in range(args.iters):
            last_action, cuda_ms, wall_ms, steps = one_call()
            cuda_values.append(cuda_ms)
            wall_values.append(wall_ms)
            all_step_values.extend(steps)
            rows.append(
                {
                    "run_id": run_id,
                    "quant_mode": args.mode,
                    "iteration": index,
                    "policy_cuda_ms": round(cuda_ms, 5),
                    "policy_wall_ms": round(wall_ms, 5),
                    "step_1_ms": round(steps[0], 5),
                    "step_2_ms": round(steps[1], 5),
                    "step_3_ms": round(steps[2], 5),
                    "step_4_ms": round(steps[3], 5),
                    "step_5_ms": round(steps[4], 5),
                    "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 3),
                    "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 3),
                }
            )
            if (index + 1) % 20 == 0:
                print(f"measure {index + 1}/{args.iters}", flush=True)
        path = EXP / f"raw/torchao_latency_{slug}.csv"
        append_csv(path, list(rows[0]), rows)
        np.save(EXP / f"raw/torchao_action_{slug}.npy", last_action)
        audit["policy_cuda_ms"] = percentiles(cuda_values)
        audit["policy_wall_ms"] = percentiles(wall_values)
        audit["denoiser_step_ms"] = percentiles(all_step_values)
        audit["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        audit["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        (EXP / f"summaries/torchao_latency_{slug}.json").write_text(
            json.dumps(audit, indent=2, default=str) + "\n"
        )
        print(json.dumps({key: value for key, value in audit.items() if key != "candidate_module_names"}, indent=2, default=str))
    elif args.phase == "torchprof":
        trace_path = EXP / f"profiles/torchao_{slug}_trace.json"
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            with torch.profiler.record_function("cosmos_policy_5step"):
                one_call()
        profiler.export_chrome_trace(str(trace_path))
        op_rows = []
        for event in profiler.key_averages(group_by_input_shape=True):
            op_rows.append(
                {
                    "run_id": run_id,
                    "quant_mode": args.mode,
                    "operator": event.key,
                    "count": event.count,
                    "self_cpu_us": round(event.self_cpu_time_total, 3),
                    "cpu_total_us": round(event.cpu_time_total, 3),
                    "self_cuda_us": round(getattr(event, "self_device_time_total", 0.0), 3),
                    "cuda_total_us": round(getattr(event, "device_time_total", 0.0), 3),
                    "input_shapes": str(event.input_shapes),
                }
            )
        append_csv(
            EXP / f"profiles/torchao_{slug}_operator_summary.csv",
            list(op_rows[0]),
            op_rows,
        )
        kernel_rows = parse_kernel_trace(trace_path, args.mode, run_id)
        if kernel_rows:
            append_csv(EXP / "profiles/torchao_kernel_summary.csv", list(kernel_rows[0]), kernel_rows)
        print(f"trace={trace_path} kernels={len(kernel_rows)}")
    else:
        # Nsight's NVTX capture trigger is unreliable on this host/driver
        # combination.  CUDA profiler API markers make the capture window
        # explicit and exclude model loading plus warmup.
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push(f"cosmos_policy_5step_{slug}")
        one_call()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
        print("nsys capture payload complete")

    pre_handle.remove()
    post_handle.remove()


if __name__ == "__main__":
    main()

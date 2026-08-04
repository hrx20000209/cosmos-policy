#!/usr/bin/env python3
"""Instrumented, resumable, shardable LIBERO(-Plus) eval driver.

Reuses the VERIFIED run_episode() from run_libero_eval for correctness, but:
  - drives an explicit (suite, task_id) subset from a pilot task list
  - applies a quantization mode to model.net (quant_lib)
  - monkeypatches get_action to log per-policy-call latency (CUDA events) to
    raw/inference_steps.csv
  - writes one row per episode to raw/trials.csv (schema in exp_common)
  - resumable: skips (quant_mode, suite, task_id, seed) already in trials.csv
  - single GPU per process; shard across GPUs by --shard i/N

Run one shard:
  source env.sh; CUDA_VISIBLE_DEVICES=g MUJOCO_EGL_DEVICE_ID=g \
    .venv/bin/python driver_liberoplus.py --quant_mode bf16 --tasklist <json> \
      --shard 0/8 --seed 195 --out_tag pilot
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import exp_common as ec
import quant_lib

CKPT_DIR = "/data/rxhuang/models/cosmos-policy-libero-2b"
CKPT_PT = f"{CKPT_DIR}/Cosmos-Policy-LIBERO-Predict2-2B.pt"


@dataclass(eq=False)
class Cfg:
    # identity / model
    suite: str = "libero"
    model_family: str = "cosmos"
    config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    ckpt_path: str = CKPT_PT
    config_file: str = "cosmos_policy/config/config.py"
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    # inputs
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    # prediction
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    # eval loop
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 1
    env_img_res: int = 256
    initial_states_path: str = "DEFAULT"
    data_collection: bool = False
    deterministic: bool = True
    deterministic_reset: bool = False
    deterministic_reset_seed: Optional[int] = None
    seed: int = 195
    randomize_seed: bool = False
    # best-of-N / parallel (disabled: single-worker)
    num_queries_best_of_n: int = 1
    use_parallel_inference: bool = False
    parallel_timeout: float = 120.0


def load_tasklist(path, suites_filter=None):
    d = json.load(open(path))
    tasks = d["tasks"] if isinstance(d, dict) else d
    if suites_filter:
        tasks = [t for t in tasks if t["suite"] in suites_filter]
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant_mode", default="bf16")
    ap.add_argument("--tasklist", default="experiments/liberoplus_quantization/task_lists/liberoplus_pilot_seed195.json")
    ap.add_argument("--shard", default="0/1", help="i/N")
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--denoise", type=int, default=5)
    ap.add_argument("--num_open_loop_steps", type=int, default=16)
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--out_tag", default="pilot")
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #tasks")
    args = ap.parse_args()
    i, N = (int(x) for x in args.shard.split("/"))

    from cosmos_policy.experiments.robot import libero as _libpkg  # noqa
    import cosmos_policy.experiments.robot.libero.run_libero_eval as R
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, load_dataset_stats, init_t5_text_embeddings_cache)
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size
    from libero.libero import benchmark

    cfg = Cfg(num_denoising_steps_action=args.denoise,
              num_open_loop_steps=args.num_open_loop_steps, chunk_size=args.chunk_size,
              seed=args.seed)

    # --- resume set ---
    trials_path = f"{ec.RAW}/trials_{args.out_tag}.csv"
    infer_path = f"{ec.RAW}/inference_steps_{args.out_tag}.csv"
    done = ec.completed_keys(trials_path, ["quant_mode", "task_suite", "task_id", "seed"])

    tasks = load_tasklist(args.tasklist)
    tasks = [t for k, t in enumerate(tasks) if k % N == i]  # shard by stride
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[shard {i}/{N}] {len(tasks)} tasks, quant={args.quant_mode}, seed={args.seed}")

    # --- model ---
    model, _ = get_model(cfg)
    model.eval()
    audit = quant_lib.apply_quantization(model.net, args.quant_mode, group_size=args.group_size)
    audit.update(quant_lib.verify_quantized(model.net))
    os.makedirs(ec.PROF, exist_ok=True)
    json.dump(audit, open(f"{ec.PROF}/{args.quant_mode}_apply_audit.json", "w"), indent=2, default=str)
    print("quant audit:", {k: audit[k] for k in ("quant_backend", "hardware_accelerated", "coverage_frac", "quantized_modules")})

    dataset_stats = load_dataset_stats(f"{CKPT_DIR}/libero_dataset_statistics.json")
    init_t5_text_embeddings_cache(f"{CKPT_DIR}/libero_t5_embeddings.pkl", worker_id=0)
    resize_size = get_image_resize_size(cfg.model_family)

    # --- latency instrumentation: wrap get_action in R's namespace ---
    _orig_get_action = R.get_action
    ctr = {"suite": "", "task_id": -1, "episode": 0, "call": 0, "infer_ms": 0.0}
    infer_csv = ec.CsvAppender(infer_path, ec.INFER_FIELDS)

    def timed_get_action(*a, **k):
        # Skip the future-image VAE decode (not needed for control; a real robot
        # loop never decodes future frames). Action chunk is unchanged.
        k["generate_future_state_and_value_in_parallel"] = False
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter_ns(); s.record()
        out = _orig_get_action(*a, **k)
        e.record(); torch.cuda.synchronize()
        gpu_ms = s.elapsed_time(e); wall_ms = (time.perf_counter_ns() - t0) / 1e6
        ctr["infer_ms"] += wall_ms
        m = ec.mem_stats_mb()
        infer_csv.write({
            "quant_mode": args.quant_mode, "task_id": f"{ctr['suite']}:{ctr['task_id']}",
            "episode_id": ctr["episode"], "policy_call_index": ctr["call"],
            "chunk_size": cfg.chunk_size, "num_open_loop_steps": cfg.num_open_loop_steps,
            "preprocess_ms": "", "h2d_ms": "", "model_forward_ms": round(gpu_ms, 3),
            "denoising_ms": round(gpu_ms, 3), "action_decode_ms": "", "postprocess_ms": "",
            "total_policy_step_ms": round(wall_ms, 3),
            "amortized_ms_per_action": round(wall_ms / cfg.num_open_loop_steps, 4),
            "gpu_memory_allocated_mb": round(m["allocated"], 1),
            "gpu_memory_reserved_mb": round(m["reserved"], 1),
        })
        ctr["call"] += 1
        return out
    R.get_action = timed_get_action

    trials_csv = ec.CsvAppender(trials_path, ec.TRIALS_FIELDS)
    cfg_hash = ec.config_hash({"quant": args.quant_mode, "denoise": args.denoise,
                               "nols": args.num_open_loop_steps, "chunk": args.chunk_size, "seed": args.seed})
    commit = ec.git_commit()

    # cache benchmark objects per suite
    bm_cache = {}
    for t in tasks:
        suite, tid = t["suite"], t["id"] - 1   # id is 1-indexed in task_classification
        key = (args.quant_mode, suite, str(tid), str(args.seed))
        if key in done:
            continue
        if suite not in bm_cache:
            bm_cache[suite] = benchmark.get_benchmark_dict()[suite]()
        task_suite = bm_cache[suite]
        cfg.task_suite_name = suite
        # env + init state
        task = task_suite.get_task(tid)
        env, task_description = R.get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        init_states = task_suite.get_task_init_states(tid)
        ec.reset_peak_mem()
        ctr.update({"suite": suite, "task_id": tid, "episode": 0, "call": 0, "infer_ms": 0.0})
        ep_t0 = time.perf_counter_ns()
        try:
            success, *_ = R.run_episode(cfg, env, task_description, model, None,
                                        dataset_stats, None, resize_size,
                                        initial_state=init_states[0], log_file=None)
            term = "success" if success else "timeout"
        except Exception as e:
            success = False; term = f"error:{type(e).__name__}"
            print(f"  task {suite}:{tid} ERROR {e}")
        ep_ms = (time.perf_counter_ns() - ep_t0) / 1e6
        try:
            env.close()
        except Exception:
            pass
        mem = ec.mem_stats_mb()
        trials_csv.write({
            "model_family": "cosmos", "checkpoint": "Cosmos-Policy-LIBERO-Predict2-2B",
            "quant_mode": args.quant_mode, "quant_backend": audit["quant_backend"],
            "hardware_accelerated": audit["hardware_accelerated"],
            "task_id": tid, "task_name": t["name"], "task_suite": suite,
            "perturbation_category": t.get("category", ""), "difficulty": t.get("difficulty_level", ""),
            "perturbation_level": t.get("difficulty_level", ""), "seed": args.seed,
            "success": int(bool(success)), "termination_reason": term,
            "environment_steps": ctr["call"] * cfg.num_open_loop_steps, "executed_actions": ctr["call"] * cfg.num_open_loop_steps,
            "policy_calls": ctr["call"],
            "episode_total_ms": round(ep_ms, 1), "inference_total_ms": round(ctr["infer_ms"], 1),
            "environment_total_ms": round(ep_ms - ctr["infer_ms"], 1),
            "peak_memory_mb": round(mem["peak_allocated"], 1),
            "git_commit": commit, "config_hash": cfg_hash,
        })
        print(f"  {suite}:{tid} [{t.get('category','')}/d{t.get('difficulty_level','')}] "
              f"success={int(bool(success))} calls={ctr['call']} ep={ep_ms/1000:.1f}s infer={ctr['infer_ms']/1000:.1f}s")

    trials_csv.close(); infer_csv.close()
    print(f"[shard {i}/{N}] DONE quant={args.quant_mode}")


if __name__ == "__main__":
    main()

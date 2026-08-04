#!/usr/bin/env python3
"""Phase 3b: numerical-consistency + latency microbenchmark per quant mode.

For each mode: reload the model, quantize model.net, then repeatedly run the
verified get_action() path on ONE fixed real LIBERO observation. We wrap the
inner DiT denoising call (model.generate_samples_from_batch) with CUDA-event
timing (the dominant, quantized cost) and also time the full policy step
(wall clock). Actions are compared to the bf16 baseline (MSE / max-abs / cosine).

Outputs:
  profiles/<mode>_quantization_audit.json   (audit + numerics + latency stats)
  raw/microbench_latency.csv                 (per-iter latencies, all modes)

Run: source env.sh; CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
     .venv/bin/python .../quant_microbench.py --modes bf16,fp8_backbone,int8_backbone,int4_weight_only,fake_int4_backbone
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import exp_common as ec
import quant_lib

CKPT_DIR = "/data/rxhuang/models/cosmos-policy-libero-2b"
CKPT_PT = f"{CKPT_DIR}/Cosmos-Policy-LIBERO-Predict2-2B.pt"


@dataclass
class Cfg:
    suite: str = "libero"
    model_family: str = "cosmos"
    config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    ckpt_path: str = CKPT_PT
    config_file: str = "cosmos_policy/config/config.py"
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    trained_with_image_aug: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    deterministic: bool = True


def get_fixed_observation(cfg, resize_size):
    """Reset libero_10 task 0 to its first init state and return one observation."""
    from libero.libero import benchmark
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    task = bm.get_task(0)
    env, desc = get_libero_env(task, "cosmos", resolution=256)
    init_states = bm.get_task_init_states(0)
    env.reset()
    obs = env.set_init_state(init_states[0])
    # step a few no-ops so the arm is mid-motion (representative activations)
    for _ in range(20):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    observation = prepare_observation(obs, resize_size, cfg.flip_images)
    env.close()
    return observation, task.language


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="bf16,fp8_backbone,int8_backbone,int4_weight_only,fake_int4_backbone")
    ap.add_argument("--denoise", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--compile", action="store_true", help="torch.compile(model.net) after quant (torchao intended path)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()
    modes = args.modes.split(",")
    TAG = args.tag or ("compiled" if args.compile else "eager")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, get_action, load_dataset_stats, init_t5_text_embeddings_cache)
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size

    cfg = Cfg(num_denoising_steps_action=args.denoise)
    dataset_stats = load_dataset_stats(f"{CKPT_DIR}/libero_dataset_statistics.json")
    init_t5_text_embeddings_cache(f"{CKPT_DIR}/libero_t5_embeddings.pkl", worker_id=0)
    resize_size = get_image_resize_size(cfg.model_family)
    obs, task_label = get_fixed_observation(cfg, resize_size)
    print(f"fixed obs: primary{obs['primary_image'].shape} wrist{obs['wrist_image'].shape} "
          f"proprio{obs['proprio'].shape} task={task_label!r}")

    lat_csv = ec.CsvAppender(f"{ec.RAW}/microbench_latency_{TAG}.csv",
                             ["quant_mode", "iter", "denoise_ms", "policy_step_ms",
                              "gpu_mem_alloc_mb", "gpu_mem_reserved_mb"])
    baseline_action = None
    summary = {}

    for mode in modes:
        print(f"\n===== mode {mode} =====")
        torch.cuda.empty_cache(); ec.reset_peak_mem()
        model, _ = get_model(cfg)
        model.eval()
        quant_lib.configure_model_precision(
            model, "fp16" if mode in ("fp16", "float16") else "bf16"
        )
        # audit BEFORE timing
        audit = quant_lib.apply_quantization(model.net, mode, group_size=128)
        vq = quant_lib.verify_quantized(model.net)
        audit.update(vq)
        audit["compiled"] = bool(args.compile)
        if args.compile:
            try:
                model.net = torch.compile(model.net)
                print("torch.compile applied to model.net (compiles on first warmup call)")
            except Exception as e:
                print("torch.compile FAILED:", e); audit["compile_error"] = str(e)
        print("audit:", {k: audit[k] for k in ("quant_mode", "quant_backend", "hardware_accelerated",
                                                "n_candidate_linears", "coverage_frac", "quantized_modules")})

        # wrap the DiT denoise call with cuda-event timing
        inner = model.generate_samples_from_batch
        timer = {"ms": 0.0}

        def wrapped(*a, **k):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); out = inner(*a, **k); e.record(); torch.cuda.synchronize()
            timer["ms"] = s.elapsed_time(e)
            return out
        model.generate_samples_from_batch = wrapped

        def one_call():
            timer["ms"] = 0.0
            t0 = time.perf_counter_ns()
            out = get_action(cfg, model, dataset_stats, obs, task_label,
                             seed=args.seed, num_denoising_steps_action=args.denoise,
                             generate_future_state_and_value_in_parallel=False, batch_size=1)
            torch.cuda.synchronize()
            step_ms = (time.perf_counter_ns() - t0) / 1e6
            act = out["actions"] if isinstance(out, dict) else out   # full chunk [16,7]
            return np.asarray(act, dtype=np.float64), timer["ms"], step_ms

        # warmup
        for _ in range(args.warmup):
            act, _, _ = one_call()
        # timed
        den, stp = [], []
        for i in range(args.iters):
            act, d, s = one_call()
            den.append(d); stp.append(s)
            m = ec.mem_stats_mb()
            lat_csv.write({"quant_mode": mode, "iter": i, "denoise_ms": round(d, 4),
                           "policy_step_ms": round(s, 4),
                           "gpu_mem_alloc_mb": round(m["allocated"], 1),
                           "gpu_mem_reserved_mb": round(m["reserved"], 1)})
        mem = ec.mem_stats_mb()

        # numerics vs bf16 baseline (use last action; deterministic seed)
        act_now = act  # from last timed call
        if mode == "bf16" and baseline_action is None:
            baseline_action = act_now.copy()
        num = {}
        if baseline_action is not None:
            diff = act_now - baseline_action
            denom = np.linalg.norm(act_now) * np.linalg.norm(baseline_action)
            num = {
                "action_mse_vs_bf16": float(np.mean(diff ** 2)),
                "action_maxabs_vs_bf16": float(np.max(np.abs(diff))),
                "action_cosine_vs_bf16": float(np.dot(act_now.ravel(), baseline_action.ravel()) / denom) if denom > 0 else None,
            }

        audit.update({
            "denoise_steps": args.denoise,
            "latency_denoise_ms": ec.latency_summary(den),
            "latency_policy_step_ms": ec.latency_summary(stp),
            "peak_mem_allocated_mb": mem["peak_allocated"],
            "peak_mem_reserved_mb": mem["peak_reserved"],
            "numerics": num,
            "git_commit": ec.git_commit(),
        })
        with open(f"{ec.PROF}/{mode}_quantization_audit_{TAG}.json", "w") as f:
            json.dump(audit, f, indent=2, default=str)
        summary[mode] = {
            "hw": audit["hardware_accelerated"],
            "denoise_p50": audit["latency_denoise_ms"]["p50"],
            "step_p50": audit["latency_policy_step_ms"]["p50"],
            "peak_mem_mb": mem["peak_allocated"],
            **num,
        }
        print(f"denoise p50={audit['latency_denoise_ms']['p50']:.1f}ms  "
              f"step p50={audit['latency_policy_step_ms']['p50']:.1f}ms  peakmem={mem['peak_allocated']:.0f}MB  num={num}")

        model.generate_samples_from_batch = inner
        del model
        torch.cuda.empty_cache()

    lat_csv.close()
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, default=str))
    json.dump(summary, open(f"{ec.SUMM}/microbench_summary_{TAG}.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()

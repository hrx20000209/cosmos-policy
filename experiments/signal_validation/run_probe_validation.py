"""Validate the opt-in Cosmos internal probe on the original LIBERO checkpoint.

This is a measurement script, not a runtime policy.  It intentionally records
only pooled intermediate features and action outputs; Cosmos' value slot is not
extracted or used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.runtime.model_probe import SpatialTokenReducer, TokenProbeConfig
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation

DEFAULT_CHECKPOINT = "/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_STATS = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
DEFAULT_T5 = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"
BLOCK_IDS = [0, 7, 14, 21, 27]


def build_cfg(checkpoint: str, action_horizon: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        suite="libero",
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=checkpoint,
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True,
        num_third_person_images=1,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        normalize_proprio=True,
        unnormalize_actions=True,
        use_variance_scale=False,
        use_jpeg_compression=True,
        trained_with_image_aug=True,
        chunk_size=action_horizon,
        action_dim=7,
    )


def action_from_result(result: dict) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)


def tensor_bytes(features: list[torch.Tensor]) -> int:
    return int(sum(feature.numel() * feature.element_size() for feature in features))


def call_action(cfg, model, stats, obs, task, seed: int) -> tuple[np.ndarray, float, list[torch.Tensor]]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    model.sampler.step_timing_events = []
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = get_action(
        cfg,
        model,
        stats,
        obs,
        task,
        seed=seed,
        num_denoising_steps_action=1,
        generate_future_state_and_value_in_parallel=True,
        decode_future_state=False,
    )
    model._validation_last_result = result
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    features = list(getattr(model, "last_intermediate_features", None) or [])
    return action_from_result(result), elapsed_ms, features


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()) if arr.size else float("nan"),
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else float("nan"),
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "min_ms": float(arr.min()) if arr.size else float("nan"),
        "max_ms": float(arr.max()) if arr.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if "so101" in checkpoint.lower() or "finet" in checkpoint.lower():
        raise ValueError(f"benchmark validation refuses a finetuned/SO101 checkpoint: {checkpoint}")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)

    configure_repository_paths({"repositories": {"libero": "/home/rxhuang/Projects/LIBERO", "cosmos": str(REPO_ROOT)}})
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    init_t5_text_embeddings_cache(args.t5_embeddings)
    stats = load_dataset_stats(args.dataset_stats)
    cfg = build_cfg(checkpoint)
    model, train_config = get_model(cfg)
    model.eval()

    env = RealLiberoEnvironment(args.task_suite, args.task_id, 256)
    try:
        raw = env.reset(0)
        settle = np.zeros(7, dtype=np.float32)
        settle[-1] = -1.0
        for _ in range(10):
            raw, _, _, _ = env.step(settle)
        observation = extract_observation(raw, flip_vertical=True)
    finally:
        env.close()
    obs = {"primary_image": observation.primary_image, "wrist_image": observation.wrist_image, "proprio": observation.proprio}
    task = env.description if hasattr(env, "description") else ""

    # Warm up the exact baseline path before timing.  Probe is explicitly off.
    model.intermediate_feature_ids = None
    model.intermediate_feature_reducer = None
    model.last_intermediate_features = None
    _ = call_action(cfg, model, stats, obs, task, args.seed)

    off_times: list[float] = []
    off_actions: list[np.ndarray] = []
    off_mem: list[dict[str, float]] = []
    for _ in range(args.repeats):
        torch.cuda.reset_peak_memory_stats()
        action, elapsed, features = call_action(cfg, model, stats, obs, task, args.seed)
        if features:
            raise AssertionError("probe-off run unexpectedly produced intermediate features")
        off_times.append(elapsed)
        off_actions.append(action)
        off_mem.append({
            "allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        })

    reducer = SpatialTokenReducer(TokenProbeConfig(slot_indices=tuple(range(9))))
    model.intermediate_feature_ids = BLOCK_IDS
    model.intermediate_feature_reducer = reducer
    model.last_intermediate_features = None
    # Warm up with the probe enabled.
    _ = call_action(cfg, model, stats, obs, task, args.seed)

    on_times: list[float] = []
    on_actions: list[np.ndarray] = []
    on_mem: list[dict[str, float]] = []
    final_features: list[torch.Tensor] = []
    for _ in range(args.repeats):
        torch.cuda.reset_peak_memory_stats()
        action, elapsed, features = call_action(cfg, model, stats, obs, task, args.seed)
        if len(features) != len(BLOCK_IDS):
            raise AssertionError(f"expected {len(BLOCK_IDS)} feature tensors, got {len(features)}")
        for block_id, feature in zip(BLOCK_IDS, features):
            if tuple(feature.shape[:2]) != (1, 9):
                raise AssertionError(f"unexpected pooled feature shape at block {block_id}: {tuple(feature.shape)}")
        on_times.append(elapsed)
        on_actions.append(action)
        final_features = features
        on_mem.append({
            "allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        })

    runtime_result = getattr(model, "_validation_last_result")
    generated = runtime_result["generated_latent"]
    original_clean = runtime_result["orig_clean_latent_frames"]
    data_batch = runtime_result["data_batch"]
    latent_indices = {
        key: int(value)
        for key, value in runtime_result["latent_indices"].items()
    }
    runtime_layout = {
        "input_slots": {
            "0": "leading_temporal_placeholder",
            "1": "current_proprio_conditioned",
            "2": "current_wrist_image_conditioned",
            "3": "current_primary_image_conditioned",
        },
        "generated_slots": {
            "4": "action_chunk",
            "5": "future_proprio",
            "6": "future_wrist_image",
            "7": "future_primary_image",
            "8": "value_structural_only_excluded",
        },
        "latent_indices_from_runtime": latent_indices,
        "input_video_shape": list(data_batch["video"].shape),
        "input_video_dtype_after_normalization": str(data_batch["video"].dtype),
        "generated_latent_shape": list(generated.shape),
        "generated_latent_dtype": str(generated.dtype),
        "generated_latent_device": str(generated.device),
        "original_clean_latent_shape": list(original_clean.shape),
        "action_extraction_shape": [16, 7],
        "future_proprio_available_without_rgb_decode": True,
        "future_visual_latent_available_without_rgb_decode": True,
        "value_read_or_used": False,
        "latent_frame_token_grid": {"temporal": 9, "spatial_h": 14, "spatial_w": 14, "hidden_dim": 2048},
        "pooled_probe_shape_per_block": [1, 9, 2048],
    }

    max_off_on_abs = float(max(np.max(np.abs(a - b)) for a, b in zip(off_actions, on_actions)))
    max_off_repeat_abs = float(max(np.max(np.abs(a - off_actions[0])) for a in off_actions))
    feature_cpu_ms: list[float] = []
    feature_cpu_bytes: list[int] = []
    disk_ms: list[float] = []
    disk_bytes: list[int] = []
    with tempfile.TemporaryDirectory(prefix="cosmos_probe_validation_") as temp_dir:
        for repeat in range(args.repeats):
            # Re-run so each transfer measurement starts from a fresh GPU tensor.
            _, _, features = call_action(cfg, model, stats, obs, task, args.seed)
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            cpu_features = [feature.detach().cpu() for feature in features]
            torch.cuda.synchronize()
            transfer_ms = (time.perf_counter_ns() - start) / 1e6
            payload_bytes = tensor_bytes(cpu_features)
            feature_cpu_ms.append(transfer_ms)
            feature_cpu_bytes.append(payload_bytes)

            path = Path(temp_dir) / f"probe_{repeat}.pt"
            start = time.perf_counter_ns()
            torch.save({"block_ids": BLOCK_IDS, "features": cpu_features}, path)
            disk_ms.append((time.perf_counter_ns() - start) / 1e6)
            disk_bytes.append(path.stat().st_size)

    features_summary = {
        "block_ids": BLOCK_IDS,
        "shapes": [list(feature.shape) for feature in final_features],
        "dtype_on_gpu": [str(feature.dtype) for feature in final_features],
        "bytes_per_request_fp32": tensor_bytes(final_features),
    }
    result = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_suite": args.task_suite,
        "task_id": args.task_id,
        "denoising_steps": 1,
        "value_used": False,
        "probe": features_summary,
        "runtime_layout": runtime_layout,
        "baseline_off": {
            "timing": summarize(off_times),
            "peak_memory_mb": off_mem,
            "repeat_max_abs_action_difference": max_off_repeat_abs,
        },
        "probe_on": {
            "timing": summarize(on_times),
            "peak_memory_mb": on_mem,
            "repeat_max_abs_action_difference": float(max(np.max(np.abs(a - on_actions[0])) for a in on_actions)),
        },
        "equivalence": {
            "max_abs_action_difference_off_vs_on": max_off_on_abs,
            "max_abs_action_difference_off_repeat": max_off_repeat_abs,
            "allclose_atol_1e-6": bool(max_off_on_abs <= 1e-6),
        },
        "cpu_transfer": {
            "timing": summarize(feature_cpu_ms),
            "bytes": feature_cpu_bytes,
        },
        "disk_logging": {
            "timing": summarize(disk_ms),
            "bytes": disk_bytes,
            "format": "torch.save(compact pooled fp32 features)",
        },
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "model_config": {
            "min_num_conditional_frames": int(model.config.min_num_conditional_frames),
            "train_action_chunk": int(train_config.dataloader_train.dataset.chunk_size),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

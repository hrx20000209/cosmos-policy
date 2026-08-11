"""Collect raw aligned LIBERO states for the Cosmos latent-reuse experiment.

This is a data-collection pass only.  It runs the original pre-finetune LIBERO
checkpoint with one denoising step, does not read the value slot, and does not
install any intermediate-feature probe.  At each 16-step action request it
saves the current RGB/proprio, the predicted future visual/proprio slots, and
the real observation after executing the action chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation
from experiments.progressive_wam.task_stage import label_episode, stage_of_step

DEFAULT_CHECKPOINT = "/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_STATS = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
DEFAULT_T5 = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"


def build_cfg(checkpoint: str) -> SimpleNamespace:
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
        chunk_size=16,
        action_dim=7,
    )


def decode_future_proprio(latent: torch.Tensor, index: int, stats: dict) -> np.ndarray:
    frame = latent[:, :, index, :, :].reshape(latent.shape[0], -1)
    dim = int(np.asarray(stats["proprio_min"]).shape[0])
    copies = frame.shape[1] // dim
    if copies < 1:
        raise ValueError(f"future proprio latent is too small for dim={dim}: {tuple(frame.shape)}")
    normalized = frame[:, : copies * dim].reshape(frame.shape[0], copies, dim).mean(dim=1)
    normalized = normalized.detach().float().cpu().numpy()[0]
    minimum = np.asarray(stats["proprio_min"], dtype=np.float32)
    maximum = np.asarray(stats["proprio_max"], dtype=np.float32)
    return (0.5 * (normalized + 1.0) * (maximum - minimum) + minimum).astype(np.float32)


def policy_call(cfg, model, stats, observation, task: str, seed: int) -> dict:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    obs = {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }
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
        generate_future_state_and_value_in_parallel=False,
        decode_future_state=False,
    )
    torch.cuda.synchronize()
    finish = time.perf_counter_ns()
    actions = np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)
    latent = result["generated_latent"]
    indices = {key: int(value) for key, value in result["latent_indices"].items()}
    return {
        "actions": actions,
        "latent": latent,
        "indices": indices,
        "start_ns": int(start),
        "finish_ns": int(finish),
        "latency_ms": (finish - start) / 1e6,
    }


def run_episode(env, model, cfg, stats, task_id: int, seed: int, args) -> list[dict]:
    task = env.description
    raw = env.reset(0)
    settle = np.zeros(7, dtype=np.float32)
    settle[-1] = -1.0
    for _ in range(args.settle_steps):
        raw, _, _, _ = env.step(settle)

    requests: list[dict] = []
    all_proprios: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    control_step = 0

    while control_step < args.max_steps:
        current = extract_observation(raw, flip_vertical=True)
        result = policy_call(cfg, model, stats, current, task, seed)
        action = result["actions"]
        latent = result["latent"]
        indices = result["indices"]
        predicted_proprio = decode_future_proprio(latent, indices["future_proprio_latent_idx"], stats)
        predicted_wrist = (
            latent[0, :, indices["future_wrist_image_latent_idx"], :, :].detach().float().cpu().numpy().astype(np.float32)
        )
        predicted_primary = (
            latent[0, :, indices["future_image_latent_idx"], :, :].detach().float().cpu().numpy().astype(np.float32)
        )
        del latent

        prefix = min(args.execute_horizon, len(action))
        start_step = control_step
        target = None
        done = False
        for local_index in range(prefix):
            before = extract_observation(raw, flip_vertical=True)
            all_proprios.append(before.proprio.copy())
            all_actions.append(action[local_index].copy())
            raw, _, done, _ = env.step(action[local_index])
            after = extract_observation(raw, flip_vertical=True)
            if local_index + 1 == prefix:
                target = after
            control_step += 1
            if done or control_step >= args.max_steps:
                break

        # The downstream experiment intentionally uses only complete 16-step
        # aligned requests.  In particular, it never invents a target state.
        if target is not None and prefix == args.execute_horizon:
            requests.append(
                {
                    "task_id": int(task_id),
                    "task": task,
                    "control_step": int(start_step),
                    "target_execution_step": int(start_step + prefix),
                    "seed": int(seed),
                    "current_primary": np.ascontiguousarray(current.primary_image.copy()),
                    "current_wrist": np.ascontiguousarray(current.wrist_image.copy()),
                    "target_primary": np.ascontiguousarray(target.primary_image.copy()),
                    "target_wrist": np.ascontiguousarray(target.wrist_image.copy()),
                    "current_proprio": current.proprio.astype(np.float32).copy(),
                    "target_proprio": target.proprio.astype(np.float32).copy(),
                    "predicted_proprio": predicted_proprio,
                    "predicted_wrist": predicted_wrist,
                    "predicted_primary": predicted_primary,
                    "action": action.copy(),
                    "policy_latency_ms": float(result["latency_ms"]),
                    "value_used": False,
                    "intermediate_probe_used": False,
                }
            )
        if done or control_step >= args.max_steps:
            break

    props = np.stack(all_proprios) if all_proprios else np.empty((0, 9), dtype=np.float32)
    actions = np.stack(all_actions) if all_actions else np.empty((0, 7), dtype=np.float32)
    labels = label_episode(props, actions) if len(props) else []
    for request in requests:
        request["stage"] = stage_of_step(labels, request["control_step"])
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--execute-horizon", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--output", required=True, help="metadata JSON; arrays are saved beside it as .npz")
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if "so101" in checkpoint.lower() or "finet" in checkpoint.lower():
        raise ValueError(f"benchmark validation refuses a finetuned/SO101 checkpoint: {checkpoint}")
    configure_repository_paths({"repositories": {"libero": "/home/rxhuang/Projects/LIBERO", "cosmos": str(REPO_ROOT)}})
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    init_t5_text_embeddings_cache(args.t5_embeddings)
    stats = load_dataset_stats(args.dataset_stats)
    cfg = build_cfg(checkpoint)
    model, _ = get_model(cfg)
    model.eval()
    # Explicitly keep all optional diagnostics disabled for this experiment.
    model.intermediate_feature_ids = None
    model.intermediate_feature_reducer = None

    requests: list[dict] = []
    for task_id in args.task_ids:
        print(f"[reuse-trace] task={task_id} starting", flush=True)
        env = RealLiberoEnvironment(args.task_suite, task_id, 256)
        try:
            task_requests = run_episode(env, model, cfg, stats, task_id, args.seed, args)
        finally:
            env.close()
        requests.extend(task_requests)
        print(f"[reuse-trace] task={task_id} aligned_requests={len(task_requests)}", flush=True)

    if not requests:
        raise RuntimeError("no complete aligned requests were collected")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    array_path = output.with_suffix(".npz")
    np.savez_compressed(
        array_path,
        current_primary=np.stack([r.pop("current_primary") for r in requests]),
        current_wrist=np.stack([r.pop("current_wrist") for r in requests]),
        target_primary=np.stack([r.pop("target_primary") for r in requests]),
        target_wrist=np.stack([r.pop("target_wrist") for r in requests]),
        current_proprio=np.stack([r.pop("current_proprio") for r in requests]),
        target_proprio=np.stack([r.pop("target_proprio") for r in requests]),
        predicted_proprio=np.stack([r.pop("predicted_proprio") for r in requests]),
        predicted_wrist=np.stack([r.pop("predicted_wrist") for r in requests]),
        predicted_primary=np.stack([r.pop("predicted_primary") for r in requests]),
        action=np.stack([r.pop("action") for r in requests]),
    )
    metadata = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_suite": args.task_suite,
        "task_ids": [int(x) for x in args.task_ids],
        "denoising_steps": 1,
        "action_horizon": 16,
        "execute_horizon": int(args.execute_horizon),
        "value_used": False,
        "intermediate_probe_used": False,
        "latent_scale": "model.encode = tokenizer.encode(video) * model.sigma_data; generated latent uses the same model state scale",
        "slot_layout": {
            "0": "leading_temporal_placeholder",
            "1": "current_proprio_conditioned",
            "2": "current_wrist_image_conditioned",
            "3": "current_primary_image_conditioned",
            "4": "action_chunk",
            "5": "future_proprio",
            "6": "future_wrist_image",
            "7": "future_primary_image",
            "8": "value_structural_only_excluded",
        },
        "array_file": str(array_path),
        "requests": requests,
    }
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"aligned_requests": len(requests), "array_file": str(array_path)}, indent=2), flush=True)
    print(f"[reuse-trace] wrote {output}", flush=True)


if __name__ == "__main__":
    main()

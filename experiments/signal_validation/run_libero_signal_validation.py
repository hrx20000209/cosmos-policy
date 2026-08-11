"""Small, baseline-only Cosmos signal validation on six LIBERO episodes.

The policy is always the original pre-finetune LIBERO checkpoint.  This runner
does not implement a runtime intervention and does not read Cosmos' value slot.
It records generated action/future slots, compact DiT features, execution-step
alignment, visual latent errors against a constant-current baseline, and
fresh-vs-stale-camera action counterfactuals.
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

from cosmos_policy.runtime.model_probe import SpatialTokenReducer, TokenProbeConfig
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation
from experiments.progressive_wam.task_stage import label_episode, stage_of_step

DEFAULT_CHECKPOINT = "/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_STATS = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
DEFAULT_T5 = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"
BLOCK_IDS = [0, 7, 14, 21, 27]
VISUAL_HORIZONS = [1, 4, 8, 16]


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


def policy_call(cfg, model, stats, observation, task: str, seed: int):
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    obs = {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }
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
    torch.cuda.synchronize()
    finish = time.perf_counter_ns()
    model.sampler.step_timing_events = None
    actions = np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)
    latent = result["generated_latent"]
    indices = {key: int(value) for key, value in result["latent_indices"].items()}
    features = [feature.detach().float().cpu().numpy()[0] for feature in (model.last_intermediate_features or [])]
    return {
        "actions": actions,
        "latent": latent,
        "indices": indices,
        "features": features,
        "start_ns": int(start),
        "finish_ns": int(finish),
        "latency_ms": (finish - start) / 1e6,
    }


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


def hidden_convergence(features: list[np.ndarray]) -> list[list[float]]:
    if len(features) != len(BLOCK_IDS):
        raise ValueError(f"expected features for blocks {BLOCK_IDS}, got {len(features)}")
    final = features[-1]
    final_norm = np.linalg.norm(final, axis=-1)
    rows: list[list[float]] = []
    for feature in features:
        norm = np.linalg.norm(feature, axis=-1)
        cosine = np.sum(feature * final, axis=-1) / np.maximum(norm * final_norm, 1e-8)
        rows.append(cosine.astype(np.float32).tolist())
    return rows


def encode_slot_latents(
    model,
    cfg,
    current_wrist: np.ndarray,
    current_primary: np.ndarray,
    future_wrist: list[np.ndarray],
    future_primary: list[np.ndarray],
    slot: str,
) -> np.ndarray:
    """Encode full 33-frame slot sequences and return one VAE slot per sample.

    The Wan temporal kernel cannot encode a four-frame semantic slot in
    isolation.  Building the complete 9-slot sequence preserves the temporal
    context used by the actual Cosmos input path.
    """
    from cosmos_policy.experiments.robot.cosmos_utils import prepare_images_for_model

    if len(future_wrist) != len(future_primary):
        raise ValueError("future camera lists must have equal length")
    blank_source = np.zeros_like(current_primary)
    sequences = []
    for wrist, primary in zip(future_wrist, future_primary):
        processed = np.asarray(
            prepare_images_for_model([blank_source, current_wrist, current_primary, wrist, primary], cfg),
            dtype=np.uint8,
        )
        blank, curr_wrist, curr_primary, future_w, future_p = processed
        # 1 leading blank + 8 semantic slots duplicated four times.
        sequence = np.concatenate(
            [
                blank[None],
                np.repeat(blank[None], 4, axis=0),
                np.repeat(curr_wrist[None], 4, axis=0),
                np.repeat(curr_primary[None], 4, axis=0),
                np.repeat(blank[None], 4, axis=0),
                np.repeat(blank[None], 4, axis=0),
                np.repeat(future_w[None], 4, axis=0),
                np.repeat(future_p[None], 4, axis=0),
                np.repeat(blank[None], 4, axis=0),
            ],
            axis=0,
        )
        sequences.append(sequence)
    raw = np.stack(sequences, axis=0)
    video = torch.from_numpy(np.transpose(raw, (0, 4, 1, 2, 3))).to("cuda:0")
    video = video.to(dtype=model.tensor_kwargs["dtype"]) / 127.5 - 1.0
    with torch.inference_mode():
        encoded = model.encode(video).float()
    index = 7 if slot == "primary" else 6
    return encoded[:, :, index].detach().cpu().numpy().astype(np.float32)


def visual_metrics(model, cfg, current: object, targets: dict[int, object], predicted: np.ndarray, slot: str) -> tuple[list[dict], float]:
    horizons = [h for h in VISUAL_HORIZONS if h in targets]
    if not horizons:
        return [], 0.0
    start = time.perf_counter_ns()
    future_wrist = [targets[h].wrist_image if slot == "wrist" else current.wrist_image for h in horizons]
    future_primary = [targets[h].primary_image if slot == "primary" else current.primary_image for h in horizons]
    # The first sample is the constant-current future baseline.
    encoded = encode_slot_latents(
        model,
        cfg,
        current.wrist_image,
        current.primary_image,
        [current.wrist_image] + future_wrist,
        [current.primary_image] + future_primary,
        slot,
    )
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    current_latent = encoded[0]
    rows = []
    for position, horizon in enumerate(horizons, start=1):
        gt = encoded[position]
        e_pred = float(np.mean(np.abs(predicted - gt)))
        e_const = float(np.mean(np.abs(current_latent - gt)))
        rows.append(
            {
                "horizon_steps": int(horizon),
                "camera": slot,
                "e_pred_l1": e_pred,
                "e_const_l1": e_const,
                "gain_e_const_minus_e_pred": e_const - e_pred,
            }
        )
    return rows, elapsed_ms


def action_summary(actions: np.ndarray) -> dict[str, float]:
    first = np.asarray(actions, dtype=np.float64)
    second = np.diff(first, n=2, axis=0) if len(first) > 2 else np.zeros((0, first.shape[1]))
    return {
        "chunk_l2_norm": float(np.linalg.norm(first)),
        "chunk_mean_abs": float(np.mean(np.abs(first))),
        "chunk_max_abs": float(np.max(np.abs(first))),
        "chunk_jerk_mean": float(np.mean(np.linalg.norm(second, axis=1))) if len(second) else 0.0,
        "chunk_gripper_mean": float(np.mean(first[:, 6])),
        "chunk_gripper_sign_changes": int(np.sum(np.diff(np.sign(first[:, 6])) != 0)),
    }


def run_episode(env, model, cfg, stats, task_id: int, seed: int, args) -> dict:
    task = env.description
    raw = env.reset(0)
    settle = np.zeros(7, dtype=np.float32)
    settle[-1] = -1.0
    for _ in range(args.settle_steps):
        raw, _, _, _ = env.step(settle)

    requests: list[dict] = []
    counterfactuals: list[dict] = []
    all_proprios: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    previous_observation = None
    control_step = 0
    success = False
    termination = "max_steps"
    max_no_progress = 0
    no_progress_streak = 0

    while control_step < args.max_steps and not success:
        current = extract_observation(raw, flip_vertical=True)
        current_ts = int(current.timestamp_ns)
        result = policy_call(cfg, model, stats, current, task, seed)
        fresh_action = result["actions"]
        final_features = result["features"]
        latent = result["latent"]
        indices = result["indices"]
        predicted_proprio = decode_future_proprio(latent, indices["future_proprio_latent_idx"], stats)
        predicted_wrist = latent[0, :, indices["future_wrist_image_latent_idx"], :, :].detach().float().cpu().numpy()
        predicted_primary = latent[0, :, indices["future_image_latent_idx"], :, :].detach().float().cpu().numpy()
        del latent

        target_observations: dict[int, object] = {}
        # The baseline starts executing from the fresh result.  Camera
        # counterfactuals are run only after this chunk has completed.
        action_start_ns = int(result["finish_ns"])
        prefix = min(args.execute_horizon, len(fresh_action))
        request_actions = fresh_action[:prefix].copy()
        request_start_step = control_step
        for local_index in range(prefix):
            before = extract_observation(raw, flip_vertical=True)
            action = fresh_action[local_index].copy()
            all_proprios.append(before.proprio.copy())
            all_actions.append(action.copy())
            raw, _, done, _ = env.step(action)
            after = extract_observation(raw, flip_vertical=True)
            if local_index + 1 in VISUAL_HORIZONS:
                target_observations[local_index + 1] = after
            delta_eef = float(np.linalg.norm(after.proprio[2:5] - before.proprio[2:5]))
            if delta_eef < 1e-4:
                no_progress_streak += 1
            else:
                no_progress_streak = 0
            max_no_progress = max(max_no_progress, no_progress_streak)
            control_step += 1
            if done:
                success = bool(env.env.check_success()) if hasattr(env.env, "check_success") else True
                termination = "success" if success else "failure"
                break
            if control_step >= args.max_steps:
                break
        action_finish_ns = int(time.perf_counter_ns())
        target_observations = dict(target_observations)
        visual_primary, visual_primary_encode_ms = visual_metrics(
            model, cfg, current, target_observations, predicted_primary, "primary"
        )
        visual_wrist, visual_wrist_encode_ms = visual_metrics(
            model, cfg, current, target_observations, predicted_wrist, "wrist"
        )
        target16 = target_observations.get(prefix)
        future_error = None
        if target16 is not None:
            future_error = {
                "target_execution_step": int(request_start_step + prefix),
                "target_offset_steps": int(prefix),
                "predicted_proprio_l1": float(np.mean(np.abs(predicted_proprio - target16.proprio))),
                "predicted_proprio_l2": float(np.linalg.norm(predicted_proprio - target16.proprio)),
                "predicted_proprio": predicted_proprio.tolist(),
                "measured_proprio": target16.proprio.astype(np.float32).tolist(),
                "tracking_gap_l2": float(np.linalg.norm(predicted_proprio[2:5] - target16.proprio[2:5])),
            }

        # These are diagnostic-only calls.  Running them after execution keeps
        # the baseline's action age and queue age free of counterfactual delay.
        if previous_observation is not None:
            for camera in ("wrist", "primary"):
                cf_obs = {
                    "primary_image": current.primary_image,
                    "wrist_image": current.wrist_image,
                    "proprio": current.proprio,
                }
                if camera == "wrist":
                    cf_obs["wrist_image"] = previous_observation.wrist_image
                else:
                    cf_obs["primary_image"] = previous_observation.primary_image
                cf_observation = SimpleNamespace(
                    primary_image=cf_obs["primary_image"],
                    wrist_image=cf_obs["wrist_image"],
                    proprio=cf_obs["proprio"],
                )
                cf_result = policy_call(cfg, model, stats, cf_observation, task, seed)
                delta = cf_result["actions"] - fresh_action
                counterfactuals.append(
                    {
                        "task_id": int(task_id),
                        "control_step": int(control_step - prefix),
                        "camera": camera,
                        "stage": "pending_posthoc",
                        "action_delta_mean_l2": float(np.mean(np.linalg.norm(delta, axis=1))),
                        "action_delta_first_l2": float(np.linalg.norm(delta[0])),
                        "action_delta_prefix_l2": {
                            str(h): float(np.mean(np.linalg.norm(delta[:h], axis=1))) for h in [1, 4, 8, 16]
                        },
                        "fresh_vs_stale_same_seed": True,
                    }
                )
                del cf_result

        action_stats = action_summary(request_actions)
        request = {
            "task_id": int(task_id),
            "task": task,
            "control_step": int(request_start_step),
            "target_execution_step": int(request_start_step + prefix),
            "observation_timestamp_ns": current_ts,
            "inference_start_ns": int(result["start_ns"]),
            "inference_finish_ns": int(result["finish_ns"]),
            "action_start_ns": action_start_ns,
            "chunk_completion_ns": action_finish_ns,
            "policy_latency_ms": float(result["latency_ms"]),
            "observation_age_at_action_start_ms": (action_start_ns - current_ts) / 1e6,
            "action_age_ms": (action_start_ns - result["finish_ns"]) / 1e6,
            "queue_age_ms": (action_start_ns - result["finish_ns"]) / 1e6,
            "chunk_completion_ms": (action_finish_ns - action_start_ns) / 1e6,
            "executed_prefix_length": int(prefix),
            "chunk_complete": bool(prefix == args.execute_horizon),
            "current_proprio": current.proprio.astype(np.float32).tolist(),
            "gripper_qpos": current.proprio[:2].astype(np.float32).tolist(),
            "load_proxy": None,
            "load_proxy_available": False,
            "stall_streak_at_request": int(no_progress_streak),
            "action": action_stats,
            "predicted_future": future_error,
            "visual_future_primary": visual_primary,
            "visual_future_wrist": visual_wrist,
            "visual_primary_gt_vae_encode_ms": float(visual_primary_encode_ms),
            "visual_wrist_gt_vae_encode_ms": float(visual_wrist_encode_ms),
            "hidden_cosine_to_final_by_block_slot": hidden_convergence(final_features),
            "hidden_block_ids": BLOCK_IDS,
            "intermediate_action_projection_available": False,
            "value_used": False,
        }
        requests.append(request)
        previous_observation = current

    props = np.stack(all_proprios) if all_proprios else np.empty((0, 9), dtype=np.float32)
    actions = np.stack(all_actions) if all_actions else np.empty((0, 7), dtype=np.float32)
    labels = label_episode(props, actions) if len(props) else []
    for request in requests:
        request["stage"] = stage_of_step(labels, request["control_step"])
    for row in counterfactuals:
        row["stage"] = stage_of_step(labels, row["control_step"])

    return {
        "task_id": int(task_id),
        "task": task,
        "seed": int(seed),
        "success": bool(success),
        "termination": termination,
        "control_steps": int(control_step),
        "request_count": len(requests),
        "max_no_progress_streak": int(max_no_progress),
        "action_underflow": False,
        "stalled": bool(max_no_progress >= args.stall_steps),
        "requests": requests,
        "counterfactuals": counterfactuals,
        "stage_labels_source": "heuristic_from_proprio_and_executed_action_not_ground_truth",
    }


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
    parser.add_argument("--stall-steps", type=int, default=20)
    parser.add_argument("--output", required=True)
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
    model, train_config = get_model(cfg)
    model.eval()
    model.intermediate_feature_ids = BLOCK_IDS
    model.intermediate_feature_reducer = SpatialTokenReducer(TokenProbeConfig(slot_indices=tuple(range(9))))

    episodes = []
    all_requests = []
    all_counterfactuals = []
    for task_id in args.task_ids:
        print(f"[signal] task={task_id} starting", flush=True)
        env = RealLiberoEnvironment(args.task_suite, task_id, 256)
        try:
            episode = run_episode(env, model, cfg, stats, task_id, args.seed, args)
        finally:
            env.close()
        episodes.append({key: value for key, value in episode.items() if key not in ("requests", "counterfactuals")})
        all_requests.extend(episode["requests"])
        all_counterfactuals.extend(episode["counterfactuals"])
        print(
            f"[signal] task={task_id} success={episode['success']} steps={episode['control_steps']} requests={episode['request_count']}",
            flush=True,
        )

    result = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_suite": args.task_suite,
        "task_ids": [int(x) for x in args.task_ids],
        "episodes_per_task": 1,
        "denoising_steps": 1,
        "action_horizon": 16,
        "execute_horizon": int(args.execute_horizon),
        "value_used": False,
        "runtime_method_installed": False,
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
        "visual_horizon_alignment_sweep": VISUAL_HORIZONS,
        "future_proprio_target_rule": "target execution state after the executed action prefix; no inference-completion alignment",
        "stage_rule": "heuristic proprio/action labels; not ground-truth task phases",
        "episodes": episodes,
        "requests": all_requests,
        "camera_counterfactuals": all_counterfactuals,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"episodes": len(episodes), "requests": len(all_requests), "counterfactuals": len(all_counterfactuals), "successes": sum(int(e["success"]) for e in episodes)}, indent=2), flush=True)
    print(f"[signal] wrote {output}", flush=True)


if __name__ == "__main__":
    main()

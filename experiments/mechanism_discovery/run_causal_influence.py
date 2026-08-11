"""Causal modality interventions inside the pretrained Cosmos LIBERO WAM.

This is a diagnostic experiment, not a runtime policy.  At a physical state
reached after executing one 16-action chunk, it holds diffusion noise, language,
and all non-intervened modalities fixed while replacing current visual or
proprioceptive conditions.  Compact per-slot features are read after selected
DiT blocks; Cosmos' value slot is neither retained nor consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.runtime.model_probe import SpatialTokenReducer, TokenProbeConfig, replace_latent_slots
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation

DEFAULT_CHECKPOINT = "/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt"
DEFAULT_STATS = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
DEFAULT_T5 = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"
BLOCK_IDS = (0, 4, 8, 12, 16, 20, 24, 27)
PROBED_SLOTS = (1, 2, 3, 4, 5, 6, 7)
SLOT_NAMES = (
    "current_proprio",
    "current_wrist",
    "current_primary",
    "action",
    "future_proprio",
    "future_wrist",
    "future_primary",
)


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


def policy_call(
    cfg: SimpleNamespace,
    model: torch.nn.Module,
    stats: dict[str, Any],
    observation: Any,
    task: str,
    seed: int,
    latent_condition: torch.Tensor | None = None,
) -> dict[str, Any]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    obs = {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    result = get_action(
        cfg,
        model,
        stats,
        obs,
        task,
        seed=seed,
        randomize_seed=False,
        num_denoising_steps_action=1,
        generate_future_state_and_value_in_parallel=True,
        decode_future_state=False,
        skip_vae_encoding=latent_condition is not None,
        previous_generated_latent=latent_condition,
        skip_camera_preprocessing=latent_condition is not None,
    )
    torch.cuda.synchronize()
    features = torch.stack(list(model.last_intermediate_features or ()), dim=1)
    if features.shape[:3] != (1, len(BLOCK_IDS), len(PROBED_SLOTS)):
        raise RuntimeError(f"unexpected compact feature shape {tuple(features.shape)}")
    return {
        "actions": np.asarray(result["actions"], dtype=np.float32).reshape(16, 7),
        "features": features[0].detach().float().cpu().numpy(),
        "generated": result["generated_latent"].detach().clone(),
        "clean": result["orig_clean_latent_frames"].detach().clone(),
        "latency_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def intervention_metrics(fresh: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, Any]:
    delta = np.asarray(counterfactual["features"] - fresh["features"], dtype=np.float64)
    delta_l2 = np.linalg.norm(delta, axis=-1)
    reference_l2 = np.linalg.norm(np.asarray(fresh["features"], dtype=np.float64), axis=-1)
    normalized = delta_l2 / np.maximum(reference_l2, 1e-8)
    block_rows = []
    for block_position, block_id in enumerate(BLOCK_IDS):
        by_slot = {name: float(delta_l2[block_position, index]) for index, name in enumerate(SLOT_NAMES)}
        normalized_by_slot = {name: float(normalized[block_position, index]) for index, name in enumerate(SLOT_NAMES)}
        visual_source = 0.5 * (by_slot["current_wrist"] + by_slot["current_primary"])
        future_effect = np.mean([by_slot["future_proprio"], by_slot["future_wrist"], by_slot["future_primary"]])
        block_rows.append(
            {
                "block": int(block_id),
                "delta_action": by_slot["action"],
                "delta_current_visual": float(visual_source),
                "delta_current_proprio": by_slot["current_proprio"],
                "delta_future": float(future_effect),
                "action_to_current_visual_propagation": float(by_slot["action"] / max(visual_source, 1e-8)),
                "action_to_current_proprio_propagation": float(
                    by_slot["action"] / max(by_slot["current_proprio"], 1e-8)
                ),
                "slot_delta_l2": by_slot,
                "slot_normalized_delta": normalized_by_slot,
            }
        )
    action_delta = np.asarray(counterfactual["actions"] - fresh["actions"], dtype=np.float64)
    return {
        "blocks": block_rows,
        "output_action_mean_step_l2": float(np.mean(np.linalg.norm(action_delta, axis=1))),
        "output_action_first_l2": float(np.linalg.norm(action_delta[0])),
        "output_action_chunk_l2": float(np.linalg.norm(action_delta)),
        "latency_ms": float(counterfactual["latency_ms"]),
    }


def compact_summary(states: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = sorted({name for state in states for name in state["interventions"]})
    for name in names:
        available = [state["interventions"][name] for state in states if name in state["interventions"]]
        block_rows = []
        for block_position, block_id in enumerate(BLOCK_IDS):
            row: dict[str, Any] = {"block": int(block_id), "n": len(available)}
            for metric in ("delta_action", "delta_current_visual", "delta_current_proprio", "delta_future"):
                values = np.asarray([item["blocks"][block_position][metric] for item in available], dtype=np.float64)
                row[metric] = {
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            block_rows.append(row)
        action_values = np.asarray([item["output_action_mean_step_l2"] for item in available], dtype=np.float64)
        result[name] = {
            "n": len(available),
            "output_action_mean_step_l2": {
                "median": float(np.median(action_values)),
                "q25": float(np.quantile(action_values, 0.25)),
                "q75": float(np.quantile(action_values, 0.75)),
            },
            "blocks": block_rows,
        }
    return result


def execute_chunk(env: RealLiberoEnvironment, raw: dict[str, Any], actions: np.ndarray) -> tuple[dict[str, Any], bool]:
    success = False
    for action in actions:
        raw, _, done, _ = env.step(action)
        if done:
            success = bool(env.env.check_success()) if hasattr(env.env, "check_success") else True
            break
    return raw, success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--init-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--states-per-episode", type=int, default=12)
    parser.add_argument("--target-states", type=int, default=240)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if "so101" in checkpoint.lower() or "finet" in checkpoint.lower():
        raise ValueError(f"mechanism benchmark refuses a finetuned/SO101 checkpoint: {checkpoint}")
    if len(args.task_ids) < 8:
        raise ValueError("causal discovery requires at least eight tasks")
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
    model.intermediate_feature_ids = list(BLOCK_IDS)
    model.intermediate_feature_reducer = SpatialTokenReducer(TokenProbeConfig(slot_indices=PROBED_SLOTS))

    output = Path(args.output)
    raw_output = Path(args.raw_output) if args.raw_output else output.with_suffix(".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.exists():
        raise FileExistsError(f"refusing to overwrite incremental state output: {raw_output}")
    raw_handle = raw_output.open("a", encoding="utf-8")

    states: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    global_state_index = 0
    stop = False
    for init_index in args.init_indices:
        for task_id in args.task_ids:
            if len(states) >= args.target_states:
                stop = True
                break
            print(f"[causal] task={task_id} init={init_index} starting states={len(states)}", flush=True)
            env = RealLiberoEnvironment(args.task_suite, task_id, 256)
            episode_states = 0
            success = False
            try:
                raw = env.reset(init_index)
                settle = np.zeros(7, dtype=np.float32)
                settle[-1] = -1.0
                for _ in range(args.settle_steps):
                    raw, _, _, _ = env.step(settle)
                previous_observation = extract_observation(raw, flip_vertical=True)
                initial_seed = args.seed + global_state_index
                previous = policy_call(cfg, model, stats, previous_observation, env.description, initial_seed)
                raw, success = execute_chunk(env, raw, previous["actions"])

                while not success and episode_states < args.states_per_episode and len(states) < args.target_states:
                    current = extract_observation(raw, flip_vertical=True)
                    state_seed = args.seed + global_state_index + 1
                    fresh = policy_call(cfg, model, stats, current, env.description, state_seed)

                    predicted_visual = replace_latent_slots(fresh["clean"], previous["generated"], {2: 6, 3: 7})
                    cached_visual = replace_latent_slots(fresh["clean"], previous["clean"], {2: 2, 3: 3})
                    predicted_wrist = replace_latent_slots(fresh["clean"], previous["generated"], {2: 6})
                    predicted_primary = replace_latent_slots(fresh["clean"], previous["generated"], {3: 7})
                    stale_proprio_observation = SimpleNamespace(
                        primary_image=current.primary_image,
                        wrist_image=current.wrist_image,
                        proprio=previous_observation.proprio,
                    )
                    calls = {
                        "predicted_visual_fresh_proprio": policy_call(
                            cfg, model, stats, current, env.description, state_seed, predicted_visual
                        ),
                        "cached_visual_fresh_proprio": policy_call(
                            cfg, model, stats, current, env.description, state_seed, cached_visual
                        ),
                        "fresh_visual_stale_proprio": policy_call(
                            cfg, model, stats, stale_proprio_observation, env.description, state_seed, fresh["clean"]
                        ),
                        "fresh_primary_predicted_wrist": policy_call(
                            cfg, model, stats, current, env.description, state_seed, predicted_wrist
                        ),
                        "predicted_primary_fresh_wrist": policy_call(
                            cfg, model, stats, current, env.description, state_seed, predicted_primary
                        ),
                    }
                    state = {
                        "state_index": int(global_state_index),
                        "task_id": int(task_id),
                        "task": env.description,
                        "init_index": int(init_index),
                        "episode_transition_index": int(episode_states + 1),
                        "seed": int(state_seed),
                        "alignment_steps": 16,
                        "proprio_intervention_l2": float(
                            np.linalg.norm(current.proprio - previous_observation.proprio)
                        ),
                        "fresh_latency_ms": float(fresh["latency_ms"]),
                        "interventions": {name: intervention_metrics(fresh, value) for name, value in calls.items()},
                    }
                    states.append(state)
                    raw_handle.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")
                    raw_handle.flush()
                    episode_states += 1
                    global_state_index += 1
                    print(
                        f"[causal] state={global_state_index}/{args.target_states} task={task_id} init={init_index}",
                        flush=True,
                    )
                    previous_observation = current
                    previous = fresh
                    del calls
                    raw, success = execute_chunk(env, raw, fresh["actions"])
            finally:
                env.close()
            episodes.append(
                {
                    "task_id": int(task_id),
                    "task": env.description,
                    "init_index": int(init_index),
                    "states": int(episode_states),
                    "success": bool(success),
                }
            )
        if stop:
            break

    artifact = {
        "schema_version": 1,
        "experiment": "cosmos_causal_internal_influence_map",
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_suite": args.task_suite,
        "task_ids": [int(value) for value in args.task_ids],
        "init_indices": [int(value) for value in args.init_indices],
        "denoising_steps": 1,
        "execute_horizon": 16,
        "block_ids": list(BLOCK_IDS),
        "probed_slots": {str(slot): name for slot, name in zip(PROBED_SLOTS, SLOT_NAMES)},
        "state_count": len(states),
        "target_state_count": int(args.target_states),
        "value_used": False,
        "privileged_state_used": False,
        "runtime_scheduler_installed": False,
        "raw_state_jsonl": str(raw_output),
        "intervention_control": "same state, language, checkpoint, denoise=1, and diffusion seed; one modality source changed",
        "proprio_intervention": "previous measured proprio, not simulator state or synthetic phase label",
        "episodes": episodes,
        "summary": compact_summary(states),
        "states": states,
    }
    raw_handle.close()
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"states": len(states), "episodes": len(episodes), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

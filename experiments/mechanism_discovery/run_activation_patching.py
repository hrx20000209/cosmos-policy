"""Oracle activation-patching frontier for the pretrained Cosmos LIBERO WAM.

For the same executed state and diffusion seed, this experiment captures a
full-fresh hidden state at selected DiT blocks, patches selected temporal slots
into a predicted-visual run, and recomputes only the remaining blocks.  The
fresh prefix is oracle information: this script is a mechanism diagnostic and
does not implement a runtime scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.runtime.model_probe import LIBERO_PATCH_GROUPS, FullHiddenCapture, replace_latent_slots
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation
from experiments.mechanism_discovery.run_causal_influence import (
    DEFAULT_CHECKPOINT,
    DEFAULT_STATS,
    DEFAULT_T5,
    build_cfg,
    execute_chunk,
)

PATCH_BLOCKS = (4, 8, 12, 16, 20, 24)
TOTAL_BLOCKS = 28
ACTION_SLOT = 4


def observation_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def get_action_call(
    cfg: Any,
    model: torch.nn.Module,
    stats: dict[str, Any],
    observation: Any,
    task: str,
    seed: int,
    latent_condition: torch.Tensor | None = None,
) -> dict[str, Any]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    return get_action(
        cfg,
        model,
        stats,
        observation_dict(observation),
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


def extract_actions(latent: torch.Tensor, cfg: Any, stats: dict[str, Any]) -> np.ndarray:
    from cosmos_policy.experiments.robot.cosmos_utils import (
        extract_action_chunk_from_latent_sequence,
        unnormalize_actions,
    )

    indices = torch.full((latent.shape[0],), ACTION_SLOT, dtype=torch.int64, device=latent.device)
    actions = (
        extract_action_chunk_from_latent_sequence(
            latent,
            action_shape=(cfg.chunk_size, cfg.action_dim),
            action_indices=indices,
        )
        .float()
        .cpu()
        .numpy()
    )
    if cfg.unnormalize_actions:
        actions = unnormalize_actions(actions, stats)
    return np.asarray(actions, dtype=np.float32).reshape(latent.shape[0], cfg.chunk_size, cfg.action_dim)


def action_distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left - right, dtype=np.float64)
    return {
        "mean_step_l2": float(np.mean(np.linalg.norm(delta, axis=-1))),
        "first_step_l2": float(np.linalg.norm(delta[0])),
        "chunk_l2": float(np.linalg.norm(delta)),
    }


def summarize(states: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for block_id in PATCH_BLOCKS:
        block_summary: dict[str, Any] = {}
        for patch_name in LIBERO_PATCH_GROUPS:
            rows = [state["repairs"][str(block_id)][patch_name] for state in states]
            recovery = np.asarray([row["recovery_ratio"] for row in rows], dtype=np.float64)
            distance = np.asarray([row["distance_to_fresh"]["mean_step_l2"] for row in rows], dtype=np.float64)
            block_summary[patch_name] = {
                "n": len(rows),
                "recovery_ratio": {
                    "median": float(np.median(recovery)),
                    "q25": float(np.quantile(recovery, 0.25)),
                    "q75": float(np.quantile(recovery, 0.75)),
                    "mean": float(np.mean(recovery)),
                },
                "distance_to_fresh_mean_step_l2": {
                    "median": float(np.median(distance)),
                    "q25": float(np.quantile(distance, 0.25)),
                    "q75": float(np.quantile(distance, 0.75)),
                },
            }
        result[str(block_id)] = block_summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--init-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--states-per-episode", type=int, default=4)
    parser.add_argument("--target-states", type=int, default=30)
    parser.add_argument("--seed", type=int, default=791)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if "so101" in checkpoint.lower() or "finet" in checkpoint.lower():
        raise ValueError(f"mechanism benchmark refuses a finetuned/SO101 checkpoint: {checkpoint}")
    if len(args.task_ids) < 8:
        raise ValueError("activation-patching discovery requires at least eight tasks")

    output = Path(args.output)
    raw_output = Path(args.raw_output) if args.raw_output else output.with_suffix(".jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.exists():
        raise FileExistsError(f"refusing to overwrite incremental state output: {raw_output}")

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

    states: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    raw_handle = raw_output.open("a", encoding="utf-8")
    global_state_index = 0
    stop = False
    for init_index in args.init_indices:
        for task_id in args.task_ids:
            if len(states) >= args.target_states:
                stop = True
                break
            print(f"[patch] task={task_id} init={init_index} starting states={len(states)}", flush=True)
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
                model.intermediate_feature_ids = None
                model.intermediate_feature_reducer = None
                previous = get_action_call(
                    cfg, model, stats, previous_observation, env.description, args.seed + global_state_index
                )
                raw, success = execute_chunk(env, raw, np.asarray(previous["actions"], dtype=np.float32).reshape(16, 7))

                while not success and episode_states < args.states_per_episode and len(states) < args.target_states:
                    current = extract_observation(raw, flip_vertical=True)
                    state_seed = args.seed + global_state_index + 1

                    model.intermediate_feature_ids = list(PATCH_BLOCKS)
                    model.intermediate_feature_reducer = FullHiddenCapture()
                    torch.cuda.synchronize()
                    fresh_started = time.perf_counter_ns()
                    fresh = get_action_call(cfg, model, stats, current, env.description, state_seed)
                    torch.cuda.synchronize()
                    fresh_latency_ms = (time.perf_counter_ns() - fresh_started) / 1e6
                    fresh_hidden = {
                        block_id: hidden
                        for block_id, hidden in zip(PATCH_BLOCKS, model.last_intermediate_features, strict=True)
                    }
                    predicted_visual = replace_latent_slots(
                        fresh["orig_clean_latent_frames"], previous["generated_latent"], {2: 6, 3: 7}
                    )

                    model.intermediate_feature_ids = None
                    model.intermediate_feature_reducer = None
                    model.net.activation_patch_request = {
                        "fresh_hidden_by_block": fresh_hidden,
                        "slot_groups": dict(LIBERO_PATCH_GROUPS),
                    }
                    torch.cuda.synchronize()
                    patch_started = time.perf_counter_ns()
                    try:
                        speculative = get_action_call(
                            cfg, model, stats, current, env.description, state_seed, predicted_visual
                        )
                        patched_latents = model.last_activation_patch_latents
                    finally:
                        model.net.activation_patch_request = None
                    torch.cuda.synchronize()
                    patch_latency_ms = (time.perf_counter_ns() - patch_started) / 1e6
                    if not patched_latents or set(patched_latents) != set(PATCH_BLOCKS):
                        raise RuntimeError(f"incomplete activation patch outputs: {list(patched_latents or ())}")

                    fresh_actions = np.asarray(fresh["actions"], dtype=np.float32).reshape(16, 7)
                    speculative_actions = np.asarray(speculative["actions"], dtype=np.float32).reshape(16, 7)
                    speculative_distance = action_distance(speculative_actions, fresh_actions)
                    denominator = speculative_distance["mean_step_l2"]
                    repairs: dict[str, Any] = {}
                    for block_id in PATCH_BLOCKS:
                        repairs[str(block_id)] = {}
                        for patch_name in LIBERO_PATCH_GROUPS:
                            patched_actions = extract_actions(patched_latents[block_id][patch_name], cfg, stats)[0]
                            distance = action_distance(patched_actions, fresh_actions)
                            recovery = 1.0 - distance["mean_step_l2"] / max(denominator, 1e-8)
                            repairs[str(block_id)][patch_name] = {
                                "distance_to_fresh": distance,
                                "recovery_ratio": float(recovery),
                                "remaining_blocks": int(TOTAL_BLOCKS - block_id - 1),
                                "remaining_dit_flops_fraction": float((TOTAL_BLOCKS - block_id - 1) / TOTAL_BLOCKS),
                            }

                    extracted_spec = extract_actions(speculative["generated_latent"], cfg, stats)[0]
                    state = {
                        "state_index": int(global_state_index),
                        "task_id": int(task_id),
                        "task": env.description,
                        "init_index": int(init_index),
                        "episode_transition_index": int(episode_states + 1),
                        "seed": int(state_seed),
                        "speculative_distance_to_fresh": speculative_distance,
                        "generated_action_extraction_max_abs_error": float(
                            np.max(np.abs(extracted_spec - speculative_actions))
                        ),
                        "fresh_capture_latency_ms": float(fresh_latency_ms),
                        "all_repairs_latency_ms": float(patch_latency_ms),
                        "repairs": repairs,
                    }
                    states.append(state)
                    raw_handle.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")
                    raw_handle.flush()
                    episode_states += 1
                    global_state_index += 1
                    print(
                        f"[patch] state={global_state_index}/{args.target_states} task={task_id} init={init_index}",
                        flush=True,
                    )

                    previous_observation = current
                    previous = fresh
                    del fresh_hidden, patched_latents
                    raw, success = execute_chunk(env, raw, fresh_actions)
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

    raw_handle.close()
    artifact = {
        "schema_version": 1,
        "experiment": "cosmos_oracle_activation_repair_frontier",
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_suite": args.task_suite,
        "task_ids": [int(value) for value in args.task_ids],
        "init_indices": [int(value) for value in args.init_indices],
        "denoising_steps": 1,
        "execute_horizon": 16,
        "patch_blocks": list(PATCH_BLOCKS),
        "patch_groups": {name: list(slots) for name, slots in LIBERO_PATCH_GROUPS.items()},
        "state_count": len(states),
        "target_state_count": int(args.target_states),
        "value_used": False,
        "privileged_state_used": False,
        "runtime_scheduler_installed": False,
        "oracle_diagnostic": True,
        "oracle_reason": "fresh prefix hidden activations are required for each patch",
        "same_diffusion_seed_for_fresh_and_speculative": True,
        "raw_state_jsonl": str(raw_output),
        "episodes": episodes,
        "summary": summarize(states),
        "states": states,
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"states": len(states), "episodes": len(episodes), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

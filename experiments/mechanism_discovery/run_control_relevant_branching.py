"""Control-relevant future branching and oracle fresh-sensing value.

The pretrained Cosmos WAM first proposes an action.  That exact action slot is
then clamped while three possible future world states are sampled.  Each future
is promoted to the next current observation and queried with the same diffusion
seed.  Divergence among those next actions is compared with the action change
caused by the subsequently observed real state.  Cosmos' value slot is never
read or used.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from scipy import stats as scipy_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.runtime.model_probe import replace_latent_slots
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation
from experiments.mechanism_discovery.run_causal_influence import (
    DEFAULT_CHECKPOINT,
    DEFAULT_STATS,
    DEFAULT_T5,
    build_cfg,
    execute_chunk,
)

ACTION_SLOT = 4
FUTURE_PROPRIO_SLOT = 5
FUTURE_WRIST_SLOT = 6
FUTURE_PRIMARY_SLOT = 7


def call_policy(
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
        {
            "primary_image": observation.primary_image,
            "wrist_image": observation.wrist_image,
            "proprio": observation.proprio,
        },
        task,
        seed=seed,
        randomize_seed=False,
        num_denoising_steps_action=1,
        generate_future_state_and_value_in_parallel=False,
        decode_future_state=False,
        skip_vae_encoding=latent_condition is not None,
        previous_generated_latent=latent_condition,
        skip_camera_preprocessing=latent_condition is not None,
    )


def sample_action_conditioned_futures(
    cfg: Any,
    model: torch.nn.Module,
    action_result: dict[str, Any],
    seeds: list[int],
) -> list[torch.Tensor]:
    data_batch = dict(action_result["data_batch"])
    data_batch["num_conditional_frames"] = model.config.min_num_conditional_frames + 1
    futures = []
    for seed in seeds:
        futures.append(
            model.generate_samples_from_batch(
                data_batch,
                n_sample=1,
                num_steps=1,
                seed=seed,
                is_negative_prompt=False,
                use_variance_scale=cfg.use_variance_scale,
                skip_vae_encoding=True,
                previous_generated_latent=action_result["generated_latent"],
            )
        )
    return futures


def extract_normalized_proprio(latent: torch.Tensor, slot: int = FUTURE_PROPRIO_SLOT) -> np.ndarray:
    return latent[0, :, slot].reshape(-1)[:9].float().cpu().numpy()


def unnormalize_proprio(normalized: np.ndarray, dataset_stats: dict[str, Any]) -> np.ndarray:
    minimum = np.asarray(dataset_stats["proprio_min"], dtype=np.float32)
    maximum = np.asarray(dataset_stats["proprio_max"], dtype=np.float32)
    return 0.5 * (np.asarray(normalized, dtype=np.float32) + 1.0) * (maximum - minimum) + minimum


def future_as_current(
    clean_template: torch.Tensor,
    future: torch.Tensor,
    real_observation: Any,
    stats: dict[str, Any],
) -> tuple[torch.Tensor, Any, np.ndarray]:
    latent = replace_latent_slots(
        clean_template,
        future,
        {1: FUTURE_PROPRIO_SLOT, 2: FUTURE_WRIST_SLOT, 3: FUTURE_PRIMARY_SLOT},
    )
    normalized_q = extract_normalized_proprio(future)
    predicted_q = unnormalize_proprio(normalized_q, stats)
    observation = SimpleNamespace(
        primary_image=real_observation.primary_image,
        wrist_image=real_observation.wrist_image,
        proprio=predicted_q,
    )
    return latent, observation, predicted_q


def action_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=-1)))


def branching_metrics(actions: list[np.ndarray]) -> dict[str, float]:
    pairs = list(itertools.combinations(actions, 2))
    mean_step = [action_distance(left, right) for left, right in pairs]
    first = [float(np.linalg.norm(left[0] - right[0])) for left, right in pairs]
    eef = [float(np.mean(np.linalg.norm(left[:, :6] - right[:, :6], axis=-1))) for left, right in pairs]
    gripper = [float(np.mean(np.abs(left[:, 6] - right[:, 6]))) for left, right in pairs]
    endpoint = [float(np.linalg.norm(np.sum(left[:, :3] - right[:, :3], axis=0))) for left, right in pairs]
    return {
        "mean_pairwise_action_mean_step_l2": float(np.mean(mean_step)),
        "mean_pairwise_first_action_l2": float(np.mean(first)),
        "mean_pairwise_eef_mean_step_l2": float(np.mean(eef)),
        "mean_pairwise_gripper_abs": float(np.mean(gripper)),
        "mean_pairwise_translation_endpoint_l2": float(np.mean(endpoint)),
    }


def latent_pairwise_l1(futures: list[torch.Tensor], slots: tuple[int, ...]) -> float:
    distances = []
    for left, right in itertools.combinations(futures, 2):
        distances.append(float(torch.mean(torch.abs(left[:, :, slots] - right[:, :, slots])).item()))
    return float(np.mean(distances))


def extract_actions(latent: torch.Tensor, cfg: Any, stats: dict[str, Any]) -> np.ndarray:
    from cosmos_policy.experiments.robot.cosmos_utils import (
        extract_action_chunk_from_latent_sequence,
        unnormalize_actions,
    )

    indices = torch.full((1,), ACTION_SLOT, dtype=torch.int64, device=latent.device)
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
    return np.asarray(unnormalize_actions(actions, stats), dtype=np.float32)[0]


def correlation_summary(states: list[dict[str, Any]], task_ids: set[int] | None = None) -> dict[str, Any]:
    selected = states if task_ids is None else [state for state in states if state["task_id"] in task_ids]
    if not selected:
        return {"n": 0}
    target = np.asarray([state["oracle_fresh_sensing_value"] for state in selected], dtype=np.float64)
    metrics = {
        "control_relevant_branching": "branching.mean_pairwise_action_mean_step_l2",
        "future_visual_latent_l1": "baselines.future_visual_latent_pairwise_l1",
        "future_proprio_variation": "baselines.future_proprio_pairwise_l2",
        "action_magnitude": "baselines.action_magnitude",
        "action_jerk": "baselines.action_jerk",
        "proprio_prediction_error": "baselines.nominal_proprio_prediction_l2",
    }

    def lookup(state: dict[str, Any], path: str) -> float:
        value: Any = state
        for component in path.split("."):
            value = value[component]
        return float(value)

    result: dict[str, Any] = {"n": len(selected)}
    for name, path in metrics.items():
        values = np.asarray([lookup(state, path) for state in selected], dtype=np.float64)
        if len(values) < 3 or np.all(values == values[0]) or np.all(target == target[0]):
            rho, pvalue = float("nan"), float("nan")
        else:
            statistic = scipy_stats.spearmanr(values, target)
            rho, pvalue = float(statistic.statistic), float(statistic.pvalue)
        order = np.argsort(values)[::-1]
        topk: dict[str, float] = {}
        for fraction in (0.1, 0.25, 0.5):
            count = max(1, int(np.ceil(len(values) * fraction)))
            topk[str(fraction)] = float(np.mean(target[order[:count]]))
        result[name] = {"spearman_rho": rho, "spearman_p": pvalue, "top_budget_mean_value": topk}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--init-indices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--states-per-episode", type=int, default=6)
    parser.add_argument("--target-states", type=int, default=60)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1291)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if "so101" in checkpoint.lower() or "finet" in checkpoint.lower():
        raise ValueError(f"mechanism benchmark refuses a finetuned/SO101 checkpoint: {checkpoint}")
    if len(args.task_ids) < 8:
        raise ValueError("branching discovery requires at least eight tasks")
    if args.branches not in (2, 3):
        raise ValueError("the diagnostic uses two or three fixed-action future branches")

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
    global_index = 0
    stop = False
    for init_index in args.init_indices:
        for task_id in args.task_ids:
            if len(states) >= args.target_states:
                stop = True
                break
            print(f"[branch] task={task_id} init={init_index} starting states={len(states)}", flush=True)
            env = RealLiberoEnvironment(args.task_suite, task_id, 256)
            episode_states = 0
            success = False
            try:
                raw = env.reset(init_index)
                settle = np.zeros(7, dtype=np.float32)
                settle[-1] = -1.0
                for _ in range(args.settle_steps):
                    raw, _, _, _ = env.step(settle)
                current = extract_observation(raw, flip_vertical=True)

                while not success and episode_states < args.states_per_episode and len(states) < args.target_states:
                    action_seed = args.seed + global_index * 11
                    current_result = call_policy(cfg, model, stats, current, env.description, action_seed)
                    current_actions = np.asarray(current_result["actions"], dtype=np.float32).reshape(16, 7)
                    future_seeds = [action_seed + 1 + branch for branch in range(args.branches)]
                    futures = sample_action_conditioned_futures(cfg, model, current_result, future_seeds)
                    fixed_action_errors = [
                        float(np.max(np.abs(extract_actions(future, cfg, stats) - current_actions)))
                        for future in futures
                    ]

                    branch_actions = []
                    predicted_proprios = []
                    for branch_index, future in enumerate(futures):
                        latent, imagined_observation, predicted_q = future_as_current(
                            current_result["orig_clean_latent_frames"], future, current, stats
                        )
                        branch_result = call_policy(
                            cfg,
                            model,
                            stats,
                            imagined_observation,
                            env.description,
                            action_seed + 101,
                            latent,
                        )
                        branch_actions.append(np.asarray(branch_result["actions"], dtype=np.float32).reshape(16, 7))
                        predicted_proprios.append(predicted_q)
                        del branch_result

                    branching = branching_metrics(branch_actions)
                    raw, success = execute_chunk(env, raw, current_actions)
                    real_next = extract_observation(raw, flip_vertical=True)
                    fresh_next = call_policy(cfg, model, stats, real_next, env.description, action_seed + 101)
                    fresh_next_actions = np.asarray(fresh_next["actions"], dtype=np.float32).reshape(16, 7)
                    sensing_value = action_distance(branch_actions[0], fresh_next_actions)
                    proprio_pairwise = [
                        float(np.linalg.norm(left - right))
                        for left, right in itertools.combinations(predicted_proprios, 2)
                    ]

                    state = {
                        "state_index": int(global_index),
                        "task_id": int(task_id),
                        "task": env.description,
                        "init_index": int(init_index),
                        "episode_transition_index": int(episode_states),
                        "action_seed": int(action_seed),
                        "future_seeds": future_seeds,
                        "fixed_action_max_abs_errors": fixed_action_errors,
                        "branching": branching,
                        "oracle_fresh_sensing_value": float(sensing_value),
                        "baselines": {
                            "future_visual_latent_pairwise_l1": latent_pairwise_l1(
                                futures, (FUTURE_WRIST_SLOT, FUTURE_PRIMARY_SLOT)
                            ),
                            "future_proprio_pairwise_l2": float(np.mean(proprio_pairwise)),
                            "action_magnitude": float(np.mean(np.linalg.norm(current_actions, axis=-1))),
                            "action_jerk": float(np.mean(np.linalg.norm(np.diff(current_actions, axis=0), axis=-1))),
                            "nominal_proprio_prediction_l2": float(
                                np.linalg.norm(predicted_proprios[0] - real_next.proprio)
                            ),
                            "execution_age_actions": 16,
                        },
                    }
                    states.append(state)
                    raw_handle.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")
                    raw_handle.flush()
                    global_index += 1
                    episode_states += 1
                    print(
                        f"[branch] state={global_index}/{args.target_states} task={task_id} init={init_index}",
                        flush=True,
                    )
                    current = real_next
                    del futures, current_result, fresh_next
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
    discovery_ids = set(args.task_ids[:6])
    heldout_ids = set(args.task_ids[6:])
    artifact = {
        "schema_version": 1,
        "experiment": "cosmos_control_relevant_future_branching",
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_suite": args.task_suite,
        "task_ids": [int(value) for value in args.task_ids],
        "init_indices": [int(value) for value in args.init_indices],
        "denoising_steps": 1,
        "future_branches": int(args.branches),
        "action_conditioning": "native contiguous condition mask clamps the sampled action slot",
        "next_action_seed_shared_across_branches_and_real_state": True,
        "state_count": len(states),
        "target_state_count": int(args.target_states),
        "value_used": False,
        "value_slot_read": False,
        "privileged_state_used": False,
        "runtime_scheduler_installed": False,
        "discovery_task_ids": sorted(discovery_ids),
        "heldout_task_ids": sorted(heldout_ids),
        "raw_state_jsonl": str(raw_output),
        "episodes": episodes,
        "correlations_all": correlation_summary(states),
        "correlations_discovery": correlation_summary(states, discovery_ids),
        "correlations_heldout": correlation_summary(states, heldout_ids),
        "states": states,
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"states": len(states), "episodes": len(episodes), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

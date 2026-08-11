"""Low-rank structure and action recovery of fresh-minus-predicted visual state.

This diagnostic runs only after exact activation patching has shown a repairable
frontier.  It computes randomized rank-32 decompositions of the visual hidden
correction and injects rank 1/4/8/16/32 approximations before recomputing the
DiT suffix.  It is an oracle analysis, not a deployable correction model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.runtime.model_probe import FullHiddenCapture, replace_latent_slots
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation
from experiments.mechanism_discovery.run_activation_patching import extract_actions, get_action_call
from experiments.mechanism_discovery.run_causal_influence import (
    DEFAULT_CHECKPOINT,
    DEFAULT_STATS,
    DEFAULT_T5,
    build_cfg,
    execute_chunk,
)

BLOCKS = (4, 8, 12, 16)
RANKS = (1, 4, 8, 16, 32)
VISUAL_SLOTS = (2, 3)


def action_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=-1)))


def randomized_decomposition(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    matrix = matrix.float().cuda()
    q = min(max(RANKS), matrix.shape[0], matrix.shape[1])
    u, singular, v = torch.pca_lowrank(matrix, q=q, center=False, niter=2)
    total_energy = float(torch.sum(matrix.square()).item())
    return u, singular, v, total_energy


def energy_curve(singular: torch.Tensor, total_energy: float) -> dict[str, float]:
    squared = singular.double().square().cumsum(0).cpu().numpy()
    return {str(rank): float(squared[rank - 1] / max(total_energy, 1e-12)) for rank in RANKS}


def correction_branches(
    fresh: torch.Tensor,
    predicted: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[float]]:
    delta_visual = (fresh[:, VISUAL_SLOTS] - predicted[:, VISUAL_SLOTS]).reshape(-1, fresh.shape[-1])
    u, singular, v, total_energy = randomized_decomposition(delta_visual)
    branches: dict[str, torch.Tensor] = {}
    for rank in RANKS:
        approximation = (u[:, :rank] * singular[:rank]) @ v[:, :rank].T
        corrected = predicted.clone()
        corrected[:, VISUAL_SLOTS] = (
            predicted[:, VISUAL_SLOTS].float().cuda() + approximation.reshape_as(predicted[:, VISUAL_SLOTS])
        ).to(device="cpu", dtype=predicted.dtype)
        branches[f"rank_{rank}"] = corrected

    per_slot: dict[str, Any] = {}
    for slot, name in ((2, "current_wrist"), (3, "current_primary")):
        delta = (fresh[:, slot] - predicted[:, slot]).reshape(-1, fresh.shape[-1])
        _, slot_singular, _, slot_total = randomized_decomposition(delta)
        per_slot[name] = energy_curve(slot_singular, slot_total)
    structure = {
        "combined_visual_energy": energy_curve(singular, total_energy),
        "per_slot_energy": per_slot,
        "top32_residual_energy": float(
            max(0.0, 1.0 - torch.sum(singular.double().square()).item() / max(total_energy, 1e-12))
        ),
    }
    pooled = (fresh[:, VISUAL_SLOTS] - predicted[:, VISUAL_SLOTS]).float().mean(dim=(0, 1, 2, 3)).tolist()
    return branches, structure, pooled


def cross_state_energy(states: list[dict[str, Any]], block_id: int) -> dict[str, float]:
    matrix = torch.as_tensor([state["blocks"][str(block_id)]["pooled_visual_delta"] for state in states])
    _, singular, _, total = randomized_decomposition(matrix)
    available = [rank for rank in RANKS if rank <= singular.shape[0]]
    cumulative = singular.double().square().cumsum(0).cpu().numpy()
    return {str(rank): float(cumulative[rank - 1] / max(total, 1e-12)) for rank in available}


def summarize(states: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for block_id in BLOCKS:
        rows = [state["blocks"][str(block_id)] for state in states]
        result[str(block_id)] = {
            "combined_visual_energy_median": {
                str(rank): float(np.median([row["structure"]["combined_visual_energy"][str(rank)] for row in rows]))
                for rank in RANKS
            },
            "action_recovery_median": {
                name: float(np.median([row["action_recovery"][name] for row in rows]))
                for name in ("full_visual", *(f"rank_{rank}" for rank in RANKS))
            },
            "cross_state_pooled_energy": cross_state_energy(states, block_id),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--task-suite", default="libero_10")
    parser.add_argument("--task-ids", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--states-per-episode", type=int, default=1)
    parser.add_argument("--target-states", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2391)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output")
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    if "so101" in checkpoint.lower() or "finet" in checkpoint.lower():
        raise ValueError(f"mechanism benchmark refuses a finetuned/SO101 checkpoint: {checkpoint}")
    if len(args.task_ids) < 8:
        raise ValueError("low-rank diagnostic requires eight tasks")
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
    raw_handle = raw_output.open("a", encoding="utf-8")

    for task_id in args.task_ids:
        if len(states) >= args.target_states:
            break
        print(f"[rank] task={task_id} starting states={len(states)}", flush=True)
        env = RealLiberoEnvironment(args.task_suite, task_id, 256)
        success = False
        episode_states = 0
        try:
            raw = env.reset(args.init_index)
            settle = np.zeros(7, dtype=np.float32)
            settle[-1] = -1.0
            for _ in range(args.settle_steps):
                raw, _, _, _ = env.step(settle)
            previous_observation = extract_observation(raw, flip_vertical=True)
            model.intermediate_feature_ids = None
            previous = get_action_call(
                cfg, model, stats, previous_observation, env.description, args.seed + len(states)
            )
            raw, success = execute_chunk(env, raw, np.asarray(previous["actions"], dtype=np.float32).reshape(16, 7))

            while not success and episode_states < args.states_per_episode and len(states) < args.target_states:
                current = extract_observation(raw, flip_vertical=True)
                state_seed = args.seed + len(states) + 1
                model.intermediate_feature_ids = list(BLOCKS)
                model.intermediate_feature_reducer = FullHiddenCapture()
                fresh = get_action_call(cfg, model, stats, current, env.description, state_seed)
                fresh_hidden = dict(zip(BLOCKS, model.last_intermediate_features, strict=True))
                predicted_visual = replace_latent_slots(
                    fresh["orig_clean_latent_frames"], previous["generated_latent"], {2: 6, 3: 7}
                )
                predicted = get_action_call(cfg, model, stats, current, env.description, state_seed, predicted_visual)
                predicted_hidden = dict(zip(BLOCKS, model.last_intermediate_features, strict=True))

                hidden_branches: dict[int, dict[str, torch.Tensor]] = {}
                block_rows: dict[str, Any] = {}
                for block_id in BLOCKS:
                    branches, structure, pooled = correction_branches(
                        fresh_hidden[block_id], predicted_hidden[block_id]
                    )
                    hidden_branches[block_id] = branches
                    block_rows[str(block_id)] = {
                        "structure": structure,
                        "pooled_visual_delta": pooled,
                    }

                model.intermediate_feature_ids = None
                model.intermediate_feature_reducer = None
                model.net.activation_patch_request = {
                    "fresh_hidden_by_block": fresh_hidden,
                    "slot_groups": {"full_visual": VISUAL_SLOTS},
                    "hidden_branches_by_block": hidden_branches,
                }
                try:
                    repeated_predicted = get_action_call(
                        cfg, model, stats, current, env.description, state_seed, predicted_visual
                    )
                    patch_latents = model.last_activation_patch_latents
                finally:
                    model.net.activation_patch_request = None

                fresh_actions = np.asarray(fresh["actions"], dtype=np.float32).reshape(16, 7)
                predicted_actions = np.asarray(predicted["actions"], dtype=np.float32).reshape(16, 7)
                repeated_actions = np.asarray(repeated_predicted["actions"], dtype=np.float32).reshape(16, 7)
                denominator = action_distance(predicted_actions, fresh_actions)
                for block_id in BLOCKS:
                    recovery: dict[str, float] = {}
                    for branch_name, latent in patch_latents[block_id].items():
                        patched_actions = extract_actions(latent, cfg, stats)[0]
                        recovery[branch_name] = float(
                            1.0 - action_distance(patched_actions, fresh_actions) / max(denominator, 1e-8)
                        )
                    block_rows[str(block_id)]["action_recovery"] = recovery

                state = {
                    "state_index": len(states),
                    "task_id": int(task_id),
                    "task": env.description,
                    "init_index": int(args.init_index),
                    "seed": int(state_seed),
                    "speculative_distance_to_fresh": float(denominator),
                    "repeat_speculative_max_abs_error": float(np.max(np.abs(repeated_actions - predicted_actions))),
                    "blocks": block_rows,
                }
                states.append(state)
                raw_handle.write(json.dumps(state, ensure_ascii=False, separators=(",", ":")) + "\n")
                raw_handle.flush()
                episode_states += 1
                print(f"[rank] state={len(states)}/{args.target_states} task={task_id}", flush=True)
                previous_observation = current
                previous = fresh
                raw, success = execute_chunk(env, raw, fresh_actions)
        finally:
            env.close()

    raw_handle.close()
    artifact = {
        "schema_version": 1,
        "experiment": "cosmos_low_rank_visual_state_repair",
        "checkpoint": checkpoint,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_suite": args.task_suite,
        "task_ids": [int(value) for value in args.task_ids],
        "init_index": int(args.init_index),
        "denoising_steps": 1,
        "blocks": list(BLOCKS),
        "ranks": list(RANKS),
        "state_count": len(states),
        "target_state_count": int(args.target_states),
        "value_used": False,
        "privileged_state_used": False,
        "runtime_scheduler_installed": False,
        "oracle_diagnostic": True,
        "decomposition": "randomized uncentered PCA/SVD, q=32, niter=2",
        "raw_state_jsonl": str(raw_output),
        "summary": summarize(states),
        "states": states,
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"states": len(states), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

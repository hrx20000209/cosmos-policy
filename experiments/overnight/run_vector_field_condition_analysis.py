#!/usr/bin/env python3
"""Vector-field and directional condition-sensitivity analysis.

This diagnostic uses the same fixed LIBERO WAM states as the ablation.  It
captures the first EDM denoiser forward at identical seeds, compares the
fresh/predicted vector field on the action and future slots, and estimates a
directional finite difference for the visual condition.  It never reads the
value slot and does not introduce a runtime scheduler.
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

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from cosmos_policy.runtime.model_probe import SpatialTokenReducer, TokenProbeConfig
from experiments.libero_harness import configure_repository_paths, extract_observation
from experiments.overnight.run_feedback_assimilation_ablation import (
    FUTURE_SLOTS,
    TRAJECTORY_FILES,
    VISUAL_SLOTS,
    build_predicted_latent,
    load_states,
    obs_dict,
)
from experiments.progressive_wam.run_p1_trajectory_dump import (
    DEFAULT_CHECKPOINT,
    DEFAULT_STATS,
    DEFAULT_T5,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
)
from experiments.progressive_wam.run_p2_oracle import restore


BLOCK_IDS = (0, 4, 8, 12, 16, 20, 24, 27)
ACTION_SLOT = 4


def action_array(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(-1, 7)


def capture_one(model: Any, get_action: Any, cfg: Any, stats: dict[str, Any], observation: Any,
                state: dict[str, Any], *, latent: torch.Tensor | None, steps: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def hook(**kwargs: Any) -> None:
        x0 = kwargs["x0_pred_B_StateShape"].detach().float().cpu()
        noisy = kwargs.get("input_x_B_StateShape")
        if noisy is not None:
            noisy = noisy.detach().float().cpu()
        features = [feature.detach().float().cpu() for feature in (model.last_intermediate_features or [])]
        captured.append({
            "x0": x0,
            "noisy": noisy,
            "sigma": float(kwargs["sigma_cur_0"]),
            "features": torch.stack(features, dim=1).squeeze(0) if features else None,
            "solver_index": int(kwargs.get("i_th", len(captured))),
        })

    model.sampler.checkpoint_hook = hook
    model.intermediate_feature_ids = list(BLOCK_IDS)
    model.intermediate_feature_reducer = SpatialTokenReducer(TokenProbeConfig(slot_indices=(1, 2, 3, 4, 5, 6, 7)))
    try:
        result = get_action(
            cfg, model, stats, obs_dict(observation), state["task_description"],
            seed=state["seed"], randomize_seed=False,
            num_denoising_steps_action=steps,
            generate_future_state_and_value_in_parallel=False,
            decode_future_state=False,
            skip_vae_encoding=latent is not None,
            previous_generated_latent=latent,
            skip_camera_preprocessing=latent is not None,
        )
    finally:
        model.sampler.checkpoint_hook = None
        model.intermediate_feature_ids = None
        model.intermediate_feature_reducer = None
    if len(captured) != steps:
        raise RuntimeError(f"captured {len(captured)} forwards, expected {steps}")
    return result, captured[0]


def vector_metrics(fresh: dict[str, Any], predicted: dict[str, Any], fresh_cap: dict[str, Any], pred_cap: dict[str, Any]) -> dict[str, Any]:
    x_f = fresh_cap["x0"]
    x_p = pred_cap["x0"]
    sigma = max(float(fresh_cap["sigma"]), 1e-8)
    dv = (x_f - x_p) / sigma
    noisy_delta = None
    if fresh_cap["noisy"] is not None and pred_cap["noisy"] is not None:
        noisy_delta = float(torch.mean(torch.abs(fresh_cap["noisy"] - pred_cap["noisy"])).item())
    action_delta = action_array(fresh) - action_array(predicted)
    per_slot = {}
    for name, slots in (("action", (ACTION_SLOT,)), ("future", FUTURE_SLOTS), ("visual_condition", VISUAL_SLOTS)):
        value = dv[:, :, slots]
        per_slot[name] = {
            "vector_field_l2": float(torch.linalg.vector_norm(value).item()),
            "vector_field_mean_abs": float(torch.mean(torch.abs(value)).item()),
        }
    hidden_rows = []
    f_features = fresh_cap["features"]
    p_features = pred_cap["features"]
    if f_features is not None and p_features is not None:
        for index, block in enumerate(BLOCK_IDS):
            delta = f_features[index] - p_features[index]
            hidden_rows.append({
                "block": int(block),
                "current_visual_hidden_l2": float(torch.linalg.vector_norm(delta[[1, 2]]).item()),
                "action_hidden_l2": float(torch.linalg.vector_norm(delta[3]).item()),
                "future_hidden_l2": float(torch.linalg.vector_norm(delta[[4, 5, 6]]).item()),
                "all_nonvalue_hidden_l2": float(torch.linalg.vector_norm(delta).item()),
            })
    return {
        "sigma": sigma,
        "noisy_state_mean_abs_delta": noisy_delta,
        "vector_field_delta": per_slot,
        "hidden_delta_by_block": hidden_rows,
        "final_action_mean_step_l2": float(np.mean(np.linalg.norm(action_delta, axis=-1))),
        "final_action_first_step_l2": float(np.linalg.norm(action_delta[0])),
        "raw_visual_condition_mean_abs_delta": float(torch.mean(torch.abs(
            fresh["orig_clean_latent_frames"][:, :, VISUAL_SLOTS]
            - predicted["orig_clean_latent_frames"][:, :, VISUAL_SLOTS]
        )).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-states", type=int, default=20)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_T5)
    parser.add_argument("--libero-repo", default="/home/rxhuang/Projects/LIBERO")
    parser.add_argument("--action-horizon", type=int, default=16)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != KNOWN_CHECKPOINT_SHA256:
        raise RuntimeError(f"unexpected checkpoint hash {digest}")
    if "so101" in str(checkpoint).lower() or "finet" in str(checkpoint).lower():
        raise RuntimeError("vector analysis requires the original pre-finetune checkpoint")
    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(PROJECT)}})
    cfg = build_cfg(args)
    model, stats = load_model(cfg, args)
    model.eval()
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    all_states = load_states()
    selected = all_states[: args.target_states]
    env_cache: dict[tuple[str, int], Any] = {}
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(selected):
        key = (state["task_suite"], state["task_id"])
        if key not in env_cache:
            from experiments.libero_harness import RealLiberoEnvironment
            env_cache[key] = RealLiberoEnvironment(key[0], key[1], 256)
            env_cache[key].reset(state["episode_index"])
        env = env_cache[key]
        restore(env, state["sim_state"])
        observation = extract_observation(env.env.regenerate_obs_from_state(state["sim_state"]), flip_vertical=True)
        fresh, fresh_cap = capture_one(model, get_action, cfg, stats, observation, state, latent=None)
        predicted_latent = build_predicted_latent(fresh, state["previous_request"])
        predicted, pred_cap = capture_one(model, get_action, cfg, stats, observation, state, latent=predicted_latent)
        delta_visual = fresh["orig_clean_latent_frames"].detach().clone()
        delta_visual[:, :, VISUAL_SLOTS] = predicted_latent[:, :, VISUAL_SLOTS] + args.eps * (
            fresh["orig_clean_latent_frames"][:, :, VISUAL_SLOTS] - predicted_latent[:, :, VISUAL_SLOTS]
        )
        perturbed, pert_cap = capture_one(model, get_action, cfg, stats, observation, state, latent=delta_visual)
        base_action = action_array(predicted)
        pert_action = action_array(perturbed)
        directional = (pert_action - base_action) / max(args.eps, 1e-8)
        row = {
            "state_index": index,
            "task_suite": state["task_suite"],
            "task_id": state["task_id"],
            "task_description": state["task_description"],
            "episode_index": state["episode_index"],
            "control_step": state["control_step"],
            "seed": state["seed"],
            "vector_field": vector_metrics(fresh, predicted, fresh_cap, pred_cap),
            "finite_difference": {
                "eps": args.eps,
                "direction_norm": float(torch.linalg.vector_norm(
                    fresh["orig_clean_latent_frames"][:, :, VISUAL_SLOTS] - predicted_latent[:, :, VISUAL_SLOTS]
                ).item()),
                "action_jvp_mean_step_l2": float(np.mean(np.linalg.norm(directional, axis=-1))),
                "action_jvp_first_step_l2": float(np.linalg.norm(directional[0])),
                "hidden_delta_by_block": [
                    {
                        "block": int(block),
                        "current_visual_hidden_l2": float(torch.linalg.vector_norm(
                            (pert_cap["features"][i] - pred_cap["features"][i])[[1, 2]] / max(args.eps, 1e-8)
                        ).item()),
                        "action_hidden_l2": float(torch.linalg.vector_norm(
                            (pert_cap["features"][i] - pred_cap["features"][i])[3] / max(args.eps, 1e-8)
                        ).item()),
                    }
                    for i, block in enumerate(BLOCK_IDS)
                ],
            },
        }
        rows.append(row)
        print(json.dumps({"completed": index + 1, "total": len(selected), "task": state["task_id"]}), flush=True)
    for env in env_cache.values():
        env.close()
    payload = {
        "schema_version": "vector_field_condition_analysis_v1",
        "experiment": "same_noisy_state_vector_field_and_visual_condition_jvp",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "state_source": TRAJECTORY_FILES,
        "states": len(rows),
        "block_ids": BLOCK_IDS,
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
        "solver": "fixed native Cosmos sampler, denoise=1 for same-noisy-state comparison",
        "interpretation": "EDM vector field is represented by (x_s - x0) / sigma; no scheduler is reimplemented.",
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "states": len(rows)}), flush=True)


if __name__ == "__main__":
    main()

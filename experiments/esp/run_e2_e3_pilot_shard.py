#!/usr/bin/env python3
"""Run frozen E2 interventions and E3 prefix probes for one ESP pilot shard.

The shard consumes only recorded simulator states to reproduce observations.
Those states never enter a policy input.  E2 may use current fresh conditions
as its stated causal oracle; E3's deployable branch replaces only one visual
slot with a *previous* real condition and never reads the target fresh frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS, DEFAULT_T5_EMBEDDINGS, ORIGINAL_CHECKPOINT,
    atomic_write_json, build_model, checkpoint_contract, configure_libero,
    pair_metrics, read_jsonl, set_up_cuda,
)

ACTION_BLOCKS = (2, 4, 6, 8, 12)
VISUAL_SLOTS = ((2, "wrist"), (3, "primary"))


class PrefixExit(RuntimeError):
    pass


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def obs_hash(observation: Any) -> str:
    digest = hashlib.sha256()
    for value in (observation.primary_image, observation.wrist_image, observation.proprio):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def predicted_condition(previous_generated: torch.Tensor) -> torch.Tensor:
    result = previous_generated.detach().clone()
    result[:, :, 2] = result[:, :, 6]
    result[:, :, 3] = result[:, :, 7]
    return result


def action_diagnostics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    delta = left.astype(np.float64) - right.astype(np.float64)
    return {
        "pair": pair_metrics(left, right),
        "first4_mean_step_l2": float(np.linalg.norm(delta[:4], axis=1).mean()),
        "per_joint_mean_abs": [float(v) for v in np.abs(delta).mean(axis=0)],
        "gripper_sign_disagreement": float(np.mean(np.sign(left[:, 6]) != np.sign(right[:, 6]))),
    }


def load_request(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    episode = torch.load(Path(entry["collection_episode"]), map_location="cpu", weights_only=False)
    if episode.get("episode_key") != entry["episode_key"]:
        raise RuntimeError("collection episode provenance mismatch")
    source = episode["requests"][int(entry["source_request_index"])]
    target = episode["requests"][int(entry["target_request_index"])]
    if target.get("state_key") != entry["state_key"]:
        raise RuntimeError("target state key mismatch")
    if int(target["control_step"]) - int(source["control_step"]) != 16:
        raise RuntimeError("source and target are not action-aligned")
    return source, target


def render(entry: dict[str, Any], request: dict[str, Any]) -> Any:
    from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment
    from experiments.libero_harness import extract_observation
    from experiments.progressive_wam.run_p2_oracle import restore

    environment = ManifestLiberoEnvironment(entry, 256, None)
    try:
        environment.reset()
        state = np.asarray(request["sim_state"], dtype=np.float64)
        restore(environment, state)
        return extract_observation(environment.env.regenerate_obs_from_state(state), flip_vertical=True)
    finally:
        environment.close()


def observation_input(observation: Any) -> dict[str, Any]:
    return {"primary_image": observation.primary_image, "wrist_image": observation.wrist_image, "proprio": observation.proprio}


def run_action(
    cfg: Any, model: Any, stats: dict[str, Any], observation: Any, instruction: str, seed: int, *,
    previous: torch.Tensor | None, condition: torch.Tensor | None = None, capture_blocks: tuple[int, ...] = (),
) -> tuple[np.ndarray, torch.Tensor, dict[int, torch.Tensor], float, float]:
    """Normal F1/P1-like action call, optionally replacing condition once before block 0."""
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    def reducer(hidden: torch.Tensor, _: int) -> torch.Tensor:
        return hidden[:, 4].reshape(hidden.shape[0], -1, hidden.shape[-1]).detach()

    def transform(*, denoiser_forward_index: int, condition: Any) -> Any:
        if denoiser_forward_index == 0 and condition_value is not None:
            condition.gt_frames.copy_(condition_value.to(device=condition.gt_frames.device, dtype=condition.gt_frames.dtype))
        return condition

    condition_value = condition
    model.inference_condition_transform = transform if condition is not None else None
    model.intermediate_feature_ids = [block - 1 for block in capture_blocks] or None
    model.intermediate_feature_reducer = reducer if capture_blocks else None
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    start.record()
    try:
        result = get_action(
            cfg, model, stats, observation_input(observation), instruction, seed=seed, randomize_seed=False,
            num_denoising_steps_action=1, generate_future_state_and_value_in_parallel=False,
            decode_future_state=False, skip_vae_encoding=previous is not None,
            previous_generated_latent=previous, skip_camera_preprocessing=previous is not None,
        )
        end.record()
        torch.cuda.synchronize()
        features = {
            block: feature.detach().float().cpu().clone()
            for block, feature in zip(capture_blocks, model.last_intermediate_features or [])
        }
        return (
            np.asarray(result["actions"], dtype=np.float32), result["orig_clean_latent_frames"].detach().clone(),
            features, float(start.elapsed_time(end)), float((time.perf_counter_ns() - wall_start) / 1e6),
        )
    finally:
        model.inference_condition_transform = None
        model.intermediate_feature_ids = None
        model.intermediate_feature_reducer = None


def prefix_probe(
    cfg: Any, model: Any, stats: dict[str, Any], observation: Any, instruction: str, seed: int,
    condition: torch.Tensor, stop_block: int,
) -> tuple[torch.Tensor, float, float]:
    """Exact forward prefix; hook only observes output then exits without writing it."""
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    captured: dict[str, torch.Tensor] = {}

    def hook(_: Any, __: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["hidden"] = output[:, 4].reshape(output.shape[0], -1, output.shape[-1]).detach().float().cpu().clone()
        raise PrefixExit("intentional prefix boundary")

    handle = model.net.blocks[stop_block - 1].register_forward_hook(hook)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    start.record()
    try:
        try:
            get_action(
                cfg, model, stats, observation_input(observation), instruction, seed=seed, randomize_seed=False,
                num_denoising_steps_action=1, generate_future_state_and_value_in_parallel=False,
                decode_future_state=False, skip_vae_encoding=True, previous_generated_latent=condition,
                skip_camera_preprocessing=True,
            )
        except PrefixExit:
            pass
        end.record()
        torch.cuda.synchronize()
    finally:
        handle.remove()
        model.inference_condition_transform = None
    if "hidden" not in captured:
        raise RuntimeError("prefix hook did not observe action-token hidden")
    return captured["hidden"], float(start.elapsed_time(end)), float((time.perf_counter_ns() - wall_start) / 1e6)


def condition_scope(base: torch.Tensor, variant: torch.Tensor, slot: int) -> dict[str, Any]:
    changed = [index for index in range(base.shape[2]) if tensor_hash(base[:, :, index]) != tensor_hash(variant[:, :, index])]
    return {"changed_slots": changed, "only_target_slot_changed": changed == [slot]}


def process_state(
    entry: dict[str, Any], all_entries: dict[str, dict[str, Any]], cfg: Any, model: Any, stats: dict[str, Any],
) -> dict[str, Any]:
    source, target = load_request(entry)
    shuffle_entry = all_entries[entry["shuffle_state_key"]]
    _, shuffle_target = load_request(shuffle_entry)
    source_obs, target_obs, shuffle_obs = render(entry, source), render(entry, target), render(shuffle_entry, shuffle_target)
    previous_generated = torch.from_numpy(np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
    # This source F1 is pre-existing evidence from the preceding decision.  Its result is never replaced by target F1 in E3.
    _, previous_real, _, _, _ = run_action(cfg, model, stats, source_obs, entry["instruction"], int(entry["seed"]), previous=None)
    f1_action, target_real, _, f1_cuda_ms, f1_wall_ms = run_action(
        cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=None
    )
    f1_repeat, _, _, _, _ = run_action(cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=None)
    _, shuffle_real, _, _, _ = run_action(cfg, model, stats, shuffle_obs, shuffle_entry["instruction"], int(shuffle_entry["seed"]), previous=None)
    predicted = predicted_condition(previous_generated)
    e0_action, _, full_hidden, e0_cuda_ms, e0_wall_ms = run_action(
        cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=predicted, capture_blocks=ACTION_BLOCKS
    )
    camera_rows: list[dict[str, Any]] = []
    for slot, name in VISUAL_SLOTS:
        interventions: dict[str, dict[str, Any]] = {}
        for intervention, replacement in (("imagined", predicted), ("stale_previous_real", previous_real), ("shuffle_task_disjoint", shuffle_real)):
            variant = target_real.detach().clone()
            variant[:, :, slot] = replacement[:, :, slot]
            action, _, _, cuda_ms, wall_ms = run_action(
                cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=target_real, condition=variant
            )
            interventions[intervention] = {
                "action_distance_to_f1": action_diagnostics(action, f1_action),
                "condition_hash": tensor_hash(variant), "scope": condition_scope(target_real, variant, slot),
                "cuda_time_ms": cuda_ms, "wall_time_ms": wall_ms,
            }
        evidence = predicted.detach().clone()
        evidence[:, :, slot] = previous_real[:, :, slot]
        evidence_scope = condition_scope(predicted, evidence, slot)
        probes: dict[str, Any] = {}
        for block in ACTION_BLOCKS:
            h0, h0_cuda_ms, h0_wall_ms = prefix_probe(
                cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), predicted, block
            )
            hc, hc_cuda_ms, hc_wall_ms = prefix_probe(
                cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), evidence, block
            )
            scale = math.sqrt(float(h0.numel()))
            probes[str(block)] = {
                "delta_hidden_primary_rms": float(torch.linalg.vector_norm(hc - h0).item() / scale),
                "delta_hidden_cosine": float(1.0 - torch.nn.functional.cosine_similarity(h0.flatten(), hc.flatten(), dim=0).item()),
                "h0_norm_rms": float(torch.linalg.vector_norm(h0).item() / scale),
                "hc_norm_rms": float(torch.linalg.vector_norm(hc).item() / scale),
                "e0_full_forward_same_block_rms": float(torch.linalg.vector_norm(h0 - full_hidden[block]).item() / scale),
                "e0_cuda_time_ms": h0_cuda_ms, "ec_cuda_time_ms": hc_cuda_ms,
                "e0_wall_time_ms": h0_wall_ms, "ec_wall_time_ms": hc_wall_ms,
                "shape": list(h0.shape),
            }
        camera_rows.append({
            "camera_slot": slot, "camera_name": name, "e2": interventions,
            "e3": {"evidence_base": "predicted", "evidence_variant": "previous_real", "real_condition_age_actions": 16,
                   "condition_hash": tensor_hash(evidence), "scope": evidence_scope, "probes": probes},
        })
    return {
        "state_id": entry["state_key"], "task_uid": entry["task_uid"], "episode_id": entry["episode_key"],
        "seed": entry["seed"], "step_idx": entry["control_step"], "split": entry["split"],
        "source_state_id": source["state_key"], "shuffle_state_id": shuffle_entry["state_key"],
        "shuffle_task_uid": shuffle_entry["task_uid"], "observation_hash": obs_hash(target_obs),
        "source_observation_hash": obs_hash(source_obs), "proprio_hash": hashlib.sha256(np.ascontiguousarray(target_obs.proprio).tobytes()).hexdigest(),
        "f1_repeat": action_diagnostics(f1_repeat, f1_action), "f1_cuda_time_ms": f1_cuda_ms, "f1_wall_time_ms": f1_wall_ms,
        "e0_action_to_f1": action_diagnostics(e0_action, f1_action), "e0_cuda_time_ms": e0_cuda_ms, "e0_wall_time_ms": e0_wall_ms,
        "base_predicted_condition_hash": tensor_hash(predicted), "target_fresh_condition_hash": tensor_hash(target_real),
        "previous_real_condition_hash": tensor_hash(previous_real), "camera": camera_rows, "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=Path("reports/esp/ESP_PILOT_STATE_BANK.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--indices", default=None, help="comma-separated bank indices for a deterministic smoke subset")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard id")
    entries = read_jsonl(args.bank)
    selected = [entry for index, entry in enumerate(entries) if index % args.num_shards == args.shard_id]
    if args.indices is not None:
        indices = [int(value) for value in args.indices.split(",")]
        selected = [entries[index] for index in indices]
    all_entries = {entry["state_key"]: entry for entry in entries}
    checkpoint = checkpoint_contract(ORIGINAL_CHECKPOINT)
    gpu = set_up_cuda(0.40)
    configure_libero(selected[0])
    cfg, stats, model = build_model(checkpoint=ORIGINAL_CHECKPOINT, dataset_stats=DEFAULT_DATASET_STATS, t5_embeddings=DEFAULT_T5_EMBEDDINGS)
    completed: list[dict[str, Any]] = []
    try:
        for position, entry in enumerate(selected, 1):
            completed.append(process_state(entry, all_entries, cfg, model, stats))
            atomic_write_json(args.output, {"status": "RUNNING", "shard_id": args.shard_id, "num_shards": args.num_shards,
                                            "checkpoint": checkpoint, "gpu": gpu, "completed": completed, "expected": len(selected)})
            print(json.dumps({"shard": args.shard_id, "completed": position, "expected": len(selected)}), flush=True)
    finally:
        model = None
        torch.cuda.empty_cache()
    atomic_write_json(args.output, {"status": "PASS", "shard_id": args.shard_id, "num_shards": args.num_shards,
                                    "checkpoint": checkpoint, "gpu": gpu, "completed": completed, "expected": len(selected),
                                    "contract": {"denoising_steps": 1, "value_used": False, "finetuning_used": False,
                                                 "hidden_activation_patch_used": False, "deployable_esp_uses_current_fresh": False}})


if __name__ == "__main__":
    main()

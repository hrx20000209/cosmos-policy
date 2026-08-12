#!/usr/bin/env python3
"""E4/E5 paired-state collector using normal P1 summaries and cheap raw probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from experiments.esp.run_e2_e3_pilot_shard import load_request, render
from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS, DEFAULT_T5_EMBEDDINGS, ORIGINAL_CHECKPOINT, atomic_write_json,
    build_model, checkpoint_contract, configure_libero, pair_metrics, read_jsonl, set_up_cuda,
)

BLOCKS = (4, 8, 12, 16, 20, 24, 27)
SLOTS = (2, 3, 4, 5, 6, 7)


def predicted_condition(previous: torch.Tensor) -> torch.Tensor:
    value = previous.detach().clone()
    value[:, :, 2] = value[:, :, 6]
    value[:, :, 3] = value[:, :, 7]
    return value


def tensor_rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach().float().square())).item())


def action_geometry(actions: np.ndarray) -> dict[str, float]:
    a = np.asarray(actions, dtype=np.float64)
    delta = np.diff(a[:, :6], axis=0)
    accel = np.diff(delta, axis=0)
    return {
        "action_norm": float(np.linalg.norm(a[:, :6], axis=1).mean()),
        "endpoint_displacement": float(np.linalg.norm(a[-1, :6] - a[0, :6])),
        "action_curvature": float(np.linalg.norm(accel, axis=1).mean()) if len(accel) else 0.0,
        "action_jerk": float(np.linalg.norm(np.diff(accel, axis=0), axis=1).mean()) if len(accel) > 1 else 0.0,
        "gripper_transition": float(np.mean(np.sign(a[1:, 6]) != np.sign(a[:-1, 6]))),
    }


def cheap_probe(previous: Any, current: Any) -> dict[str, float]:
    """CPU-only 64px frame/gradient/Farneback descriptors for both cameras."""
    started = time.perf_counter_ns()
    result: dict[str, float] = {}
    actual_vectors = []
    for name in ("wrist", "primary"):
        old = getattr(previous, f"{name}_image")
        new = getattr(current, f"{name}_image")
        old_gray = cv2.cvtColor(cv2.resize(old, (64, 64)), cv2.COLOR_RGB2GRAY)
        new_gray = cv2.cvtColor(cv2.resize(new, (64, 64)), cv2.COLOR_RGB2GRAY)
        old_f, new_f = old_gray.astype(np.float32) / 255.0, new_gray.astype(np.float32) / 255.0
        flow = cv2.calcOpticalFlowFarneback(old_f, new_f, None, 0.5, 2, 15, 2, 5, 1.2, 0)
        magnitude = np.linalg.norm(flow, axis=-1)
        gx0, gy0 = np.gradient(old_f); gx1, gy1 = np.gradient(new_f)
        result[f"{name}_frame_diff"] = float(np.abs(new_f - old_f).mean())
        result[f"{name}_gradient_diff"] = float(np.abs(gx1 - gx0).mean() + np.abs(gy1 - gy0).mean())
        result[f"{name}_flow_mean"] = float(magnitude.mean())
        result[f"{name}_flow_p90"] = float(np.quantile(magnitude, .9))
        vector = flow.reshape(-1, 2).mean(axis=0)
        actual_vectors.append(vector)
    aggregate = np.mean(np.stack(actual_vectors), axis=0)
    result["flow_mean"] = float(np.mean([result["wrist_flow_mean"], result["primary_flow_mean"]]))
    result["frame_diff"] = float(np.mean([result["wrist_frame_diff"], result["primary_frame_diff"]]))
    result["gradient_diff"] = float(np.mean([result["wrist_gradient_diff"], result["primary_gradient_diff"]]))
    result["flow_direction_x"] = float(aggregate[0]); result["flow_direction_y"] = float(aggregate[1])
    result["probe_cpu_ms"] = float((time.perf_counter_ns() - started) / 1e6)
    return result


def run_route(cfg: Any, model: Any, stats: dict[str, Any], observation: Any, instruction: str, seed: int, *, previous: torch.Tensor | None, blocks: tuple[int, ...] = ()) -> tuple[np.ndarray, torch.Tensor, dict[str, float], float]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    def reducer(hidden: torch.Tensor, _: int) -> torch.Tensor:
        # All features arise from the ordinary post-block unified hidden tensor.
        pools = {slot: hidden[:, slot].mean(dim=(1, 2)).float() for slot in SLOTS}
        def rms(slot: int) -> torch.Tensor: return torch.sqrt(hidden[:, slot].float().square().mean(dim=(1,2,3)))
        def disp(slot: int) -> torch.Tensor: return hidden[:, slot].float().std(dim=(1,2,3))
        def cosine(a: int, b: int) -> torch.Tensor: return torch.nn.functional.cosine_similarity(pools[a], pools[b], dim=-1)
        return torch.stack([rms(4), disp(4), rms(6), rms(7), rms(2), rms(3), cosine(4, 6), cosine(4, 7), cosine(4, 2), cosine(4, 3), cosine(2, 6), cosine(3, 7)], dim=-1)
    model.intermediate_feature_ids = [block - 1 for block in blocks] or None
    model.intermediate_feature_reducer = reducer if blocks else None
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); start.record()
    try:
        result = get_action(
            cfg, model, stats, {"primary_image": observation.primary_image, "wrist_image": observation.wrist_image, "proprio": observation.proprio},
            instruction, seed=int(seed), randomize_seed=False, num_denoising_steps_action=1,
            generate_future_state_and_value_in_parallel=False, decode_future_state=False,
            skip_vae_encoding=previous is not None, previous_generated_latent=previous,
            skip_camera_preprocessing=previous is not None,
        )
        end.record(); torch.cuda.synchronize()
        features: dict[str, float] = {}
        for block, tensor in zip(blocks, model.last_intermediate_features or []):
            for index, value in enumerate(tensor[0].detach().cpu().tolist()):
                features[f"internal_b{block}_{index}"] = float(value)
        return np.asarray(result["actions"], dtype=np.float32), result["orig_clean_latent_frames"].detach().clone(), features, float(start.elapsed_time(end))
    finally:
        model.intermediate_feature_ids = None; model.intermediate_feature_reducer = None


def collect(entry: dict[str, Any], cfg: Any, model: Any, stats: dict[str, Any], *, repeat: bool) -> dict[str, Any]:
    source, target = load_request(entry)
    source_obs, target_obs = render(entry, source), render(entry, target)
    prev = torch.from_numpy(np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
    p1_action, _, internal, p1_ms = run_route(cfg, model, stats, target_obs, entry["instruction"], entry["seed"], previous=prev, blocks=BLOCKS)
    f1_action, fresh, _, f1_ms = run_route(cfg, model, stats, target_obs, entry["instruction"], entry["seed"], previous=None)
    p1_condition = predicted_condition(prev)
    innovation = tensor_rms(fresh[:, :, [2,3]] - p1_condition[:, :, [2,3]])
    action_error = float(pair_metrics(p1_action, f1_action)["mean_step_l2"])
    probe = cheap_probe(source_obs, target_obs)
    planned = p1_action[0, :2].astype(np.float64)
    actual = np.array([probe["flow_direction_x"], probe["flow_direction_y"]])
    denom = float(np.linalg.norm(planned) * np.linalg.norm(actual))
    probe["expected_motion"] = float(np.linalg.norm(planned))
    probe["motion_magnitude_error"] = float(abs(np.linalg.norm(actual) - np.linalg.norm(planned)))
    probe["motion_direction_error"] = float(1 - np.dot(planned, actual) / denom) if denom > 1e-9 else 1.0
    row = {
        "state_id": entry["state_key"], "task_uid": entry["task_uid"], "episode_id": entry["episode_key"], "seed": entry["seed"],
        "step_idx": entry["control_step"], "split": entry["split"], "prediction_age_actions": 16,
        "action_error_p1_f1": action_error, "visual_innovation_latent": innovation,
        "causal_sensitivity_target": action_error / max(innovation, 1e-6), "p1_cuda_ms": p1_ms, "f1_cuda_ms": f1_ms,
        **action_geometry(p1_action), **internal, **probe, "valid": bool(np.isfinite(innovation) and innovation >= 1e-6),
    }
    if repeat:
        p1_repeat, _, repeat_features, _ = run_route(cfg, model, stats, target_obs, entry["instruction"], entry["seed"], previous=prev, blocks=BLOCKS)
        p1_no_hook, _, _, _ = run_route(cfg, model, stats, target_obs, entry["instruction"], entry["seed"], previous=prev)
        f1_repeat, _, _, _ = run_route(cfg, model, stats, target_obs, entry["instruction"], entry["seed"], previous=None)
        row["p1_repeat_action_l2"] = float(pair_metrics(p1_action, p1_repeat)["mean_step_l2"])
        row["p1_hook_off_action_l2"] = float(pair_metrics(p1_action, p1_no_hook)["mean_step_l2"])
        row["f1_repeat_action_l2"] = float(pair_metrics(f1_action, f1_repeat)["mean_step_l2"])
        row["p1_hook_feature_max_abs_diff"] = float(max((abs(internal[k] - repeat_features[k]) for k in internal), default=0.0))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=Path("reports/semantic_risk/SEMANTIC_RISK_STATE_BANK.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", default="0,1")
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--repeat", action="store_true", help="repeat F1/P1 only for numerical-floor smoke or audit subsets")
    parser.add_argument("--resume", action="store_true", help="reuse already atomically written rows in --output")
    args = parser.parse_args()
    entries = read_jsonl(args.bank)
    if args.shard_id is not None or args.num_shards is not None:
        if args.shard_id is None or args.num_shards is None or not 0 <= args.shard_id < args.num_shards:
            raise ValueError("shard-id and num-shards must be provided together")
        selected = [entry for index, entry in enumerate(entries) if index % args.num_shards == args.shard_id]
    else:
        selected = [entries[int(index)] for index in args.indices.split(",")]
    checkpoint, gpu = checkpoint_contract(ORIGINAL_CHECKPOINT), set_up_cuda(.40)
    configure_libero(selected[0]); cfg, stats, model = build_model(checkpoint=ORIGINAL_CHECKPOINT, dataset_stats=DEFAULT_DATASET_STATS, t5_embeddings=DEFAULT_T5_EMBEDDINGS)
    rows = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        rows = list(existing.get("rows", []))
    completed_ids = {row["state_id"] for row in rows}
    pending = [entry for entry in selected if entry["state_key"] not in completed_ids]
    try:
        for position, entry in enumerate(pending, len(rows) + 1):
            rows.append(collect(entry, cfg, model, stats, repeat=args.repeat))
            atomic_write_json(args.output, {"status":"RUNNING", "checkpoint":checkpoint, "gpu":gpu, "denoising_steps":1, "value_used":False, "finetuning_used":False, "hidden_activation_patch_used":False, "additional_model_forward_for_e4":False, "rows":rows, "expected":len(selected)})
            print(json.dumps({"completed":position,"expected":len(selected)}), flush=True)
    finally:
        model = None; torch.cuda.empty_cache()
    payload = {"status":"PASS", "checkpoint":checkpoint, "gpu":gpu, "denoising_steps":1, "value_used":False, "finetuning_used":False, "hidden_activation_patch_used":False, "additional_model_forward_for_e4":False, "rows":rows, "expected":len(selected)}
    atomic_write_json(args.output, payload); print(json.dumps({"output":str(args.output),"states":len(rows)}))


if __name__ == "__main__":
    main()

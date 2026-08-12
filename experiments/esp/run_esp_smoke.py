#!/usr/bin/env python3
"""Minimal legal E2/E3 ESP smoke on recorded, task-disjoint state-bank rows.

This script never patches hidden states or uses a value head.  Its E2 oracle
uses target-current fresh slots only for the stated causal ablation.  Its E3
probe compares a predicted condition with a *previously encoded* physical
condition, so the deployable probe never reads target-current fresh evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS,
    DEFAULT_T5_EMBEDDINGS,
    ORIGINAL_CHECKPOINT,
    atomic_write_json,
    build_model,
    checkpoint_contract,
    configure_libero,
    pair_metrics,
    read_jsonl,
    set_up_cuda,
)


class PrefixExit(RuntimeError):
    pass


def digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def obs_hash(observation: Any) -> str:
    hasher = hashlib.sha256()
    for value in (observation.primary_image, observation.wrist_image, observation.proprio):
        hasher.update(np.ascontiguousarray(value).tobytes())
    return hasher.hexdigest()


def predicted_condition(previous: torch.Tensor) -> torch.Tensor:
    result = previous.detach().clone()
    result[:, :, 2] = result[:, :, 6]
    result[:, :, 3] = result[:, :, 7]
    return result


def load_request(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    episode = torch.load(Path(entry["collection_episode"]), map_location="cpu", weights_only=False)
    if episode.get("episode_key") != entry["episode_key"]:
        raise RuntimeError("collection episode key mismatch")
    source = episode["requests"][int(entry["source_request_index"])]
    target = episode["requests"][int(entry["target_request_index"])]
    if target.get("state_key") != entry["state_key"]:
        raise RuntimeError("state-bank provenance mismatch")
    if int(target["control_step"]) - int(source["control_step"]) != 16:
        raise RuntimeError("state-bank request pair is not 16-action aligned")
    return source, target


def render_request(entry: dict[str, Any], request: dict[str, Any], *, resolution: int = 256) -> Any:
    """Restore a recorded request only to recreate its camera/proprio observation."""
    from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment
    from experiments.libero_harness import extract_observation
    from experiments.progressive_wam.run_p2_oracle import restore

    environment = ManifestLiberoEnvironment(entry, resolution, None)
    try:
        environment.reset()
        sim_state = np.asarray(request["sim_state"], dtype=np.float64)
        restore(environment, sim_state)
        return extract_observation(environment.env.regenerate_obs_from_state(sim_state), flip_vertical=True)
    finally:
        environment.close()


def action_and_latent(
    cfg: Any, model: Any, stats: dict[str, Any], observation: Any, instruction: str, seed: int, *, previous: torch.Tensor | None,
    condition_transform: Any | None = None, capture_blocks: list[int] | None = None,
) -> tuple[np.ndarray, torch.Tensor, dict[int, torch.Tensor]]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    def reducer(hidden: torch.Tensor, _: int) -> torch.Tensor:
        # LIBERO action temporal slot 4; [B,14,14,2048] -> [B,196,2048].
        return hidden[:, 4].reshape(hidden.shape[0], -1, hidden.shape[-1]).detach()

    model.inference_condition_transform = condition_transform
    model.intermediate_feature_ids = capture_blocks
    model.intermediate_feature_reducer = reducer if capture_blocks else None
    try:
        result = get_action(
            cfg,
            model,
            stats,
            {"primary_image": observation.primary_image, "wrist_image": observation.wrist_image, "proprio": observation.proprio},
            instruction,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=1,
            generate_future_state_and_value_in_parallel=False,
            decode_future_state=False,
            skip_vae_encoding=previous is not None,
            previous_generated_latent=previous,
            skip_camera_preprocessing=previous is not None,
        )
        features = {
            block: feature.detach().float().cpu().clone()
            for block, feature in zip(capture_blocks or [], model.last_intermediate_features or [])
        }
        return np.asarray(result["actions"], dtype=np.float32), result["orig_clean_latent_frames"].detach().clone(), features
    finally:
        model.inference_condition_transform = None
        model.intermediate_feature_ids = None
        model.intermediate_feature_reducer = None


def prefix_probe(
    cfg: Any, model: Any, stats: dict[str, Any], observation: Any, instruction: str, seed: int, condition: torch.Tensor, stop_blocks: int,
) -> tuple[torch.Tensor, float]:
    """Run the exact normal prefix and abort after its post-block hidden tensor."""
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    captured: dict[str, torch.Tensor] = {}

    def hook(_: Any, __: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["hidden"] = output[:, 4].reshape(output.shape[0], -1, output.shape[-1]).detach().float().cpu().clone()
        raise PrefixExit("intentional exact-prefix stop")

    handle = model.net.blocks[stop_blocks - 1].register_forward_hook(hook)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        torch.cuda.synchronize()
        start.record()
        try:
            get_action(
                cfg,
                model,
                stats,
                {"primary_image": observation.primary_image, "wrist_image": observation.wrist_image, "proprio": observation.proprio},
                instruction,
                seed=seed,
                randomize_seed=False,
                num_denoising_steps_action=1,
                generate_future_state_and_value_in_parallel=False,
                decode_future_state=False,
                skip_vae_encoding=True,
                previous_generated_latent=condition,
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
        raise RuntimeError("early-exit hook did not capture action hidden")
    return captured["hidden"], float(start.elapsed_time(end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-index", type=Path, default=Path("reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/esp/E2_E3_SMOKE.json"))
    parser.add_argument("--indices", default="0,1")
    args = parser.parse_args()
    entries = read_jsonl(args.state_index)
    indices = [int(value) for value in args.indices.split(",")]
    selected = [entries[index] for index in indices]
    if len({entry["task_uid"] for entry in selected}) != 1:
        raise RuntimeError("smoke states must share a task")
    checkpoint = checkpoint_contract(ORIGINAL_CHECKPOINT)
    gpu = set_up_cuda(0.40)
    configure_libero(selected[0])
    cfg, stats, model = build_model(checkpoint=ORIGINAL_CHECKPOINT, dataset_stats=DEFAULT_DATASET_STATS, t5_embeddings=DEFAULT_T5_EMBEDDINGS)
    rows: list[dict[str, Any]] = []
    try:
        for entry in selected:
            source, target = load_request(entry)
            source_obs, target_obs = render_request(entry, source), render_request(entry, target)
            previous_generated = torch.from_numpy(np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
            # Prior source F1 is offline preparation of an already existing physical condition.
            _, previous_real, _ = action_and_latent(cfg, model, stats, source_obs, entry["instruction"], int(entry["seed"]), previous=None)
            f1_a, target_real, _ = action_and_latent(cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=None)
            f1_b, _, _ = action_and_latent(cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=None)
            predicted = predicted_condition(previous_generated)
            e0_action, _, full_hidden = action_and_latent(
                cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=predicted, capture_blocks=[1, 3]
            )
            camera_rows = []
            for slot, name in ((2, "wrist"), (3, "primary")):
                # E2 primary causal oracle: fresh target condition except this camera replaced by imagination.
                base_hashes = {str(i): digest(target_real[:, :, i]) for i in (2, 3)}
                imagined = target_real.detach().clone()
                imagined[:, :, slot] = predicted[:, :, slot]
                def e2_transform(*, denoiser_forward_index: int, condition: Any, value=imagined):
                    if denoiser_forward_index == 0:
                        condition.gt_frames.copy_(value.to(device=condition.gt_frames.device, dtype=condition.gt_frames.dtype))
                    return condition
                e2_action, _, _ = action_and_latent(cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=target_real, condition_transform=e2_transform)
                # Deployable ESP E^c: predicted E0 with only slot c from prior real source condition.
                evidence = predicted.detach().clone()
                evidence[:, :, slot] = previous_real[:, :, slot]
                def esp_transform(*, denoiser_forward_index: int, condition: Any, value=evidence):
                    if denoiser_forward_index == 0:
                        condition.gt_frames.copy_(value.to(device=condition.gt_frames.device, dtype=condition.gt_frames.dtype))
                    return condition
                ec_action, _, ec_hidden = action_and_latent(
                    cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), previous=predicted,
                    condition_transform=esp_transform, capture_blocks=[1, 3],
                )
                deltas = {
                    str(block + 1): float(torch.linalg.vector_norm(ec_hidden[block] - full_hidden[block]).item() / np.sqrt(ec_hidden[block].numel()))
                    for block in (1, 3)
                }
                camera_rows.append({
                    "camera_slot": slot, "camera_name": name,
                    "e2_imagined_causal": pair_metrics(e2_action, f1_a),
                    "esp_prior_real_vs_predicted_action": pair_metrics(ec_action, e0_action),
                    "esp_delta_hidden_rms": deltas,
                    "target_visual_slot_hashes": base_hashes,
                    "predicted_slot_hash": digest(predicted[:, :, slot]),
                    "prior_real_slot_hash": digest(previous_real[:, :, slot]),
                    "only_target_slot_changed": all(
                        digest(evidence[:, :, other]) == digest(predicted[:, :, other]) for other in (2, 3) if other != slot
                    ),
                })
            probes = {}
            for k in (2, 4):
                h0, cost0 = prefix_probe(cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), predicted, k)
                h1, cost1 = prefix_probe(cfg, model, stats, target_obs, entry["instruction"], int(entry["seed"]), predicted, k)
                probes[str(k)] = {
                    "shape": list(h0.shape), "repeat_rms": float(torch.linalg.vector_norm(h0 - h1).item() / np.sqrt(h0.numel())),
                    "full_forward_same_block_rms": float(
                        torch.linalg.vector_norm(h0 - full_hidden[k - 1]).item() / np.sqrt(h0.numel())
                    ),
                    "cuda_ms_mean": (cost0 + cost1) / 2,
                }
            rows.append({
                "state_id": entry["state_key"], "task_uid": entry["task_uid"], "split": entry["split"], "seed": entry["seed"],
                "target_observation_hash": obs_hash(target_obs), "source_observation_hash": obs_hash(source_obs),
                "f1_repeat": pair_metrics(f1_a, f1_b), "camera": camera_rows, "prefix_probes": probes,
            })
    finally:
        model = None
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1, "experiment": "ESP_E2_E3_smoke", "status": "PASS", **checkpoint,
        "denoising_steps": 1, "value_used": False, "hidden_activation_patch_used": False,
        "deployable_esp_uses_current_fresh": False, "oracle_e2_uses_current_fresh": True,
        "gpu": gpu, "states": rows,
        "notes": "E3 early exit is a passive post-block hook that aborts exact normal prefix execution; no hidden tensor is written or patched.",
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "states": len(rows)}))


if __name__ == "__main__":
    main()

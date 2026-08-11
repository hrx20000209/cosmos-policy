"""Validate LIBERO VAE prefix truncation and cross-request reuse feasibility.

This is a diagnostic only.  The deployed policy remains the native one-step
Cosmos policy and no value prediction is consumed.  For each restored state we
compare the native 33-frame VAE result with a causal 13-frame prefix (the four
conditional latent slots), then feed only that prefix back to the unchanged
DiT with the same seed and compare the resulting action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
)
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


def obs_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def timed_encode(model: torch.nn.Module, video: torch.Tensor, repeats: int) -> tuple[torch.Tensor, list[float]]:
    latencies = []
    result = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = model.encode(video)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - start) / 1e6)
    assert result is not None
    return result, latencies


def load_states(path: Path, count: int) -> list[tuple[dict, dict]]:
    source = path / "checkpoints.pt"
    if not source.exists():
        source = path / "checkpoints.partial.pt"
    episodes = torch.load(source, weights_only=False)
    selected = []
    # Prefer consecutive requests from one episode, because this also exposes
    # which latent slots actually change across control requests.
    for episode in episodes:
        for request in episode["requests"]:
            selected.append((episode, request))
            if len(selected) >= count:
                return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--timing-repeats", type=int, default=4)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json")
    parser.add_argument("--t5-embeddings", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--libero-repo", default="/home/rxhuang/Projects/LIBERO")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if "so101" in str(checkpoint).lower() or "finet" in str(checkpoint).lower():
        raise ValueError(f"refusing finetuned/SO101 checkpoint: {checkpoint}")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != KNOWN_CHECKPOINT_SHA256:
        raise ValueError(f"unexpected checkpoint SHA256 {digest}")

    configure_repository_paths({"repositories": {"libero": args.libero_repo, "cosmos": str(REPO_ROOT)}})
    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)
    model.eval()

    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    selected = load_states(Path(args.trajectory_dir), args.states)
    if not selected:
        raise RuntimeError("no completed trajectory states available")

    env_cache: dict[tuple[str, int], RealLiberoEnvironment] = {}
    rows = []
    conditional_latents = []
    try:
        for state_index, (episode, request) in enumerate(selected):
            key = (episode["task_suite"], int(episode["task_id"]))
            if key not in env_cache:
                env_cache[key] = RealLiberoEnvironment(key[0], key[1], 256)
                env_cache[key].reset(int(episode["episode_index"]))
            env = env_cache[key]
            sim_state = np.asarray(request["sim_state"], dtype=np.float64)
            restore(env, sim_state)
            raw = env.env.regenerate_obs_from_state(sim_state)
            observation = extract_observation(raw, flip_vertical=True)

            baseline = get_action(
                cfg,
                model,
                dataset_stats,
                obs_dict(observation),
                episode["task_description"],
                seed=int(request["seed"]),
                randomize_seed=False,
                num_denoising_steps_action=1,
                generate_future_state_and_value_in_parallel=True,
                decode_future_state=False,
            )
            video = baseline["data_batch"]["video"]
            if tuple(video.shape[1:3]) != (3, 33):
                raise RuntimeError(f"unexpected LIBERO video shape {tuple(video.shape)}")
            full_latent, full_ms = timed_encode(model, video, args.timing_repeats)
            prefix_latent, prefix_ms = timed_encode(model, video[:, :, :13], args.timing_repeats)
            if full_latent.shape[2] != 9 or prefix_latent.shape[2] != 4:
                raise RuntimeError(
                    f"unexpected latent shapes full={tuple(full_latent.shape)} prefix={tuple(prefix_latent.shape)}"
                )

            prefix_difference = (prefix_latent - full_latent[:, :, :4]).abs()
            native_clean = baseline["orig_clean_latent_frames"]
            direct_full_difference = (full_latent - native_clean).abs()
            prefix_padded = torch.zeros_like(full_latent)
            prefix_padded[:, :, :4] = prefix_latent
            reused = get_action(
                cfg,
                model,
                dataset_stats,
                obs_dict(observation),
                episode["task_description"],
                seed=int(request["seed"]),
                randomize_seed=False,
                num_denoising_steps_action=1,
                generate_future_state_and_value_in_parallel=True,
                decode_future_state=False,
                skip_vae_encoding=True,
                previous_generated_latent=prefix_padded,
                skip_camera_preprocessing=True,
            )
            baseline_action = np.asarray(baseline["actions"], dtype=np.float32)
            reused_action = np.asarray(reused["actions"], dtype=np.float32)
            action_difference = np.abs(reused_action - baseline_action)

            # Control: if the exact native conditional prefix is retained and
            # every non-conditional VAE slot is zeroed, the action must remain
            # identical.  This separates causal-prefix sufficiency from the
            # small numerical change caused by running a shorter VAE tensor.
            exact_prefix_padded = torch.zeros_like(native_clean)
            exact_prefix_padded[:, :, :4] = native_clean[:, :, :4]
            exact_reused = get_action(
                cfg,
                model,
                dataset_stats,
                obs_dict(observation),
                episode["task_description"],
                seed=int(request["seed"]),
                randomize_seed=False,
                num_denoising_steps_action=1,
                generate_future_state_and_value_in_parallel=True,
                decode_future_state=False,
                skip_vae_encoding=True,
                previous_generated_latent=exact_prefix_padded,
                skip_camera_preprocessing=True,
            )
            exact_action = np.asarray(exact_reused["actions"], dtype=np.float32)
            exact_action_difference = np.abs(exact_action - baseline_action)
            conditional_latents.append(full_latent[:, :, :4].detach().cpu().float())
            rows.append(
                {
                    "state_index": state_index,
                    "task_suite": key[0],
                    "task_id": key[1],
                    "episode_index": int(episode["episode_index"]),
                    "control_step": int(request["control_step"]),
                    "request_id": request["request_id"],
                    "video_shape": list(video.shape),
                    "full_latent_shape": list(full_latent.shape),
                    "prefix_latent_shape": list(prefix_latent.shape),
                    "prefix_bitwise_equal": bool(torch.equal(prefix_latent, full_latent[:, :, :4])),
                    "prefix_max_abs": float(prefix_difference.max().item()),
                    "prefix_mean_abs": float(prefix_difference.mean().item()),
                    "direct_full_vs_native_max_abs": float(direct_full_difference.max().item()),
                    "direct_full_vs_native_mean_abs": float(direct_full_difference.mean().item()),
                    "action_max_abs": float(action_difference.max()),
                    "action_mean_step_l2": float(np.mean(np.linalg.norm(reused_action - baseline_action, axis=-1))),
                    "exact_native_prefix_action_max_abs": float(exact_action_difference.max()),
                    "exact_native_prefix_action_mean_step_l2": float(
                        np.mean(np.linalg.norm(exact_action - baseline_action, axis=-1))
                    ),
                    "full_encode_ms": full_ms,
                    "prefix_encode_ms": prefix_ms,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    finally:
        for env in env_cache.values():
            env.close()

    transitions = []
    for index in range(1, len(conditional_latents)):
        previous = conditional_latents[index - 1]
        current = conditional_latents[index]
        slot_l1 = (current - previous).abs().mean(dim=(0, 1, 3, 4)).numpy()
        transitions.append(
            {
                "from_state": index - 1,
                "to_state": index,
                "slot_l1": [float(value) for value in slot_l1],
            }
        )

    full_samples = [value for row in rows for value in row["full_encode_ms"][1:]]
    prefix_samples = [value for row in rows for value in row["prefix_encode_ms"][1:]]
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "states": len(rows),
        "rows": rows,
        "cross_request_conditional_slot_change": transitions,
        "aggregate": {
            "all_prefix_bitwise_equal": all(row["prefix_bitwise_equal"] for row in rows),
            "max_prefix_abs": max(row["prefix_max_abs"] for row in rows),
            "max_action_abs": max(row["action_max_abs"] for row in rows),
            "max_exact_native_prefix_action_abs": max(
                row["exact_native_prefix_action_max_abs"] for row in rows
            ),
            "full_encode_median_ms_excluding_warmup": statistics.median(full_samples),
            "prefix_encode_median_ms_excluding_warmup": statistics.median(prefix_samples),
            "speedup": statistics.median(full_samples) / statistics.median(prefix_samples),
        },
        "interpretation_guardrail": (
            "The 33-frame axis is a modality/slot assembly (blank, proprio, wrist, primary, action, "
            "future modalities, value), not a chronological observation stream. Prefix truncation may remove "
            "non-conditioning VAE work, but it does not establish append-only caching across control requests."
        ),
        "value_used_as_signal": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()

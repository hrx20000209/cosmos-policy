"""Test fast real-proprio correction under predicted visual conditioning.

For each restored decision state:
  fresh visual + real q      (full reference)
  predicted visual + real q  (fast physical feedback)
  predicted visual + q_hat   (fully speculative state)

The difference between the latter two isolates the consequence of cheap real
proprio feedback.  This is characterization, not a threshold scheduler.
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
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.runtime.model_probe import SpatialTokenReducer, TokenProbeConfig  # noqa: E402
from experiments.libero_harness import RealLiberoEnvironment, configure_repository_paths, extract_observation  # noqa: E402
from experiments.progressive_wam.cosmos_hook import CosmosCheckpointCapture  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
)
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


BLOCKS = (0, 4, 8, 12, 16, 20, 24, 27)
SLOTS = tuple(range(1, 8))


def obs_dict(observation: Any, proprio: np.ndarray | None = None) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio if proprio is None else proprio,
    }


def action_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=-1)))


def decode_proprio(future: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    frame = np.asarray(future[:, 0], dtype=np.float32).reshape(-1)
    dimension = len(stats["proprio_min"])
    copies = len(frame) // dimension
    normalized = frame[: copies * dimension].reshape(copies, dimension).mean(axis=0)
    minimum = np.asarray(stats["proprio_min"], dtype=np.float32)
    maximum = np.asarray(stats["proprio_max"], dtype=np.float32)
    return (0.5 * (normalized + 1.0) * (maximum - minimum) + minimum).astype(np.float32)


def select(split_dirs: list[list[str]], states_per_task: int) -> list[tuple[str, dict, dict, dict]]:
    selected = []
    for split, directory_string in split_dirs:
        directory = Path(directory_string)
        source = directory / "checkpoints.pt"
        if not source.exists():
            source = directory / "checkpoints.partial.pt"
        episodes = torch.load(source, weights_only=False)
        by_task: dict[tuple[str, int], list[tuple[dict, dict, dict]]] = {}
        for episode in episodes:
            for index in range(1, len(episode["requests"])):
                key = (episode["task_suite"], int(episode["task_id"]))
                by_task.setdefault(key, []).append(
                    (episode, episode["requests"][index - 1], episode["requests"][index])
                )
        for key in sorted(by_task):
            rows = by_task[key]
            positions = np.linspace(0, len(rows) - 1, min(states_per_task, len(rows))).round().astype(int)
            selected.extend((split, *rows[int(position)]) for position in positions)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", action="append", nargs=2, metavar=("SPLIT", "DIR"), required=True)
    parser.add_argument("--states-per-task", type=int, default=2)
    parser.add_argument("--output", required=True)
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
    model.intermediate_feature_ids = list(BLOCKS)
    model.intermediate_feature_reducer = SpatialTokenReducer(TokenProbeConfig(slot_indices=SLOTS))
    capture = CosmosCheckpointCapture(
        model,
        cfg,
        dataset_stats,
        capture_future_latent=False,
        capture_value=False,
        capture_compact_hidden=True,
    )
    capture.install()
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    def call(observation: Any, episode: dict, request: dict, *, latent=None, proprio=None, name: str):
        with capture.request(name, 0.0) as checkpoints:
            result = get_action(
                cfg,
                model,
                dataset_stats,
                obs_dict(observation, proprio),
                episode["task_description"],
                seed=int(request["seed"]),
                randomize_seed=False,
                num_denoising_steps_action=1,
                generate_future_state_and_value_in_parallel=True,
                decode_future_state=False,
                skip_vae_encoding=latent is not None,
                previous_generated_latent=latent,
                skip_camera_preprocessing=latent is not None,
            )
        return result, checkpoints[0].compact_hidden.numpy().astype(np.float32)

    selected = select(args.split_dir, args.states_per_task)
    rows = []
    env_cache: dict[tuple[str, int], RealLiberoEnvironment] = {}
    try:
        for index, (split, episode, previous, request) in enumerate(selected):
            key = (episode["task_suite"], int(episode["task_id"]))
            if key not in env_cache:
                env_cache[key] = RealLiberoEnvironment(key[0], key[1], 256)
                env_cache[key].reset(int(episode["episode_index"]))
            env = env_cache[key]
            sim_state = np.asarray(request["sim_state"], dtype=np.float64)
            restore(env, sim_state)
            raw = env.env.regenerate_obs_from_state(sim_state)
            observation = extract_observation(raw, flip_vertical=True)
            fresh, fresh_hidden = call(
                observation, episode, request, name=f"proprio:{index}:fresh"
            )
            previous_schedule = previous["diagnostic_schedules"]
            deployment_key = 1 if 1 in previous_schedule else "1"
            previous_future = np.asarray(
                previous_schedule[deployment_key]["future_latents"][-1], dtype=np.float32
            )
            predicted_q = decode_proprio(previous_future, dataset_stats)
            predicted_condition = fresh["orig_clean_latent_frames"].detach().clone()
            source = torch.as_tensor(
                previous_future, device=predicted_condition.device, dtype=predicted_condition.dtype
            )
            predicted_condition[:, :, 2] = source[:, 1]
            predicted_condition[:, :, 3] = source[:, 2]
            real_q, real_hidden = call(
                observation,
                episode,
                request,
                latent=predicted_condition,
                name=f"proprio:{index}:pred_visual_real_q",
            )
            speculative, speculative_hidden = call(
                observation,
                episode,
                request,
                latent=predicted_condition,
                proprio=predicted_q,
                name=f"proprio:{index}:pred_visual_pred_q",
            )
            fresh_action = np.asarray(fresh["actions"], dtype=np.float32)
            real_action = np.asarray(real_q["actions"], dtype=np.float32)
            speculative_action = np.asarray(speculative["actions"], dtype=np.float32)
            before = action_distance(speculative_action, fresh_action)
            after = action_distance(real_action, fresh_action)
            row = {
                "state_index": index,
                "split": split,
                "task_suite": key[0],
                "task_id": key[1],
                "episode_index": int(episode["episode_index"]),
                "control_step": int(request["control_step"]),
                "proprio_innovation_l2": float(np.linalg.norm(np.asarray(observation.proprio) - predicted_q)),
                "speculative_to_fresh_action_l2": before,
                "real_proprio_to_fresh_action_l2": after,
                "real_proprio_action_change_l2": action_distance(real_action, speculative_action),
                "real_proprio_recovery": 1.0 - after / max(before, 1e-8),
                "hidden_real_vs_predicted_q_l2": np.linalg.norm(real_hidden - speculative_hidden, axis=-1).tolist(),
                "hidden_fresh_vs_predicted_visual_real_q_l2": np.linalg.norm(fresh_hidden - real_hidden, axis=-1).tolist(),
            }
            rows.append(row)
            print(
                f"[proprio] {index + 1}/{len(selected)} {split} {key} recovery={row['real_proprio_recovery']:.3f}",
                flush=True,
            )
    finally:
        capture.uninstall()
        for env in env_cache.values():
            env.close()

    summary = {}
    for split in sorted({row["split"] for row in rows}):
        subset = [row for row in rows if row["split"] == split]
        recovery = np.asarray([row["real_proprio_recovery"] for row in subset])
        before = np.asarray([row["speculative_to_fresh_action_l2"] for row in subset])
        after = np.asarray([row["real_proprio_to_fresh_action_l2"] for row in subset])
        innovation = np.asarray([row["proprio_innovation_l2"] for row in subset])
        summary[split] = {
            "n": len(subset),
            "median_recovery": float(np.median(recovery)),
            "q25_recovery": float(np.quantile(recovery, 0.25)),
            "q75_recovery": float(np.quantile(recovery, 0.75)),
            "positive_recovery_fraction": float(np.mean(recovery > 0)),
            "median_action_error_before": float(np.median(before)),
            "median_action_error_after": float(np.median(after)),
            "innovation_vs_recovery_spearman": float(
                spearmanr(innovation, recovery).statistic
            ),
        }
    result = {
        "experiment": "fast_real_proprio_correction_under_predicted_vision",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "states": len(rows),
        "summary": summary,
        "rows": rows,
        "value_used": False,
        "runtime_inputs": "predicted WAM visual state plus cheap real proprio; simulator state used only to restore offline states",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

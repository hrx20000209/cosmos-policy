"""Compute F1/P1/PP/PF/FF on exact states from the Fresh collection."""

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

from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment, install_libero_checkout, load_jsonl  # noqa: E402
from experiments.libero_harness import extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import build_cfg, load_model  # noqa: E402
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        t5_embeddings=args.t5_embeddings,
        action_horizon=16,
    )


def obs_dict(observation: Any) -> dict[str, Any]:
    return {"primary_image": observation.primary_image, "wrist_image": observation.wrist_image, "proprio": observation.proprio}


def action_array(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = left.reshape(-1).astype(np.float64), right.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def vector_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    a, b = left.reshape(-1).astype(np.float64), right.reshape(-1).astype(np.float64)
    return {
        "l2": float(np.linalg.norm(a - b) / np.sqrt(a.size)),
        "cosine": cosine(a, b),
        "norm_ratio": float(np.linalg.norm(a) / (np.linalg.norm(b) + 1e-12)),
        "first_action_l2": float(np.linalg.norm(a[:7] - b[:7])),
        "gripper_abs_error": float(abs(a[6] - b[6])),
        "gripper_sign_match": float(np.sign(a[6]) == np.sign(b[6])),
    }


def innovation_metrics(base: np.ndarray, solver: np.ndarray, feedback: np.ndarray) -> dict[str, float]:
    target = base["F1"] - base["P1"]
    solver_delta = base["PP"] - base["P1"]
    feedback_delta = base["PF"] - base["PP"]
    target_flat = target.reshape(-1).astype(np.float64)
    solver_flat = solver_delta.reshape(-1).astype(np.float64)
    feedback_flat = feedback_delta.reshape(-1).astype(np.float64)
    target_norm_sq = float(np.dot(target_flat, target_flat)) + 1e-12
    return {
        "target_norm": float(np.linalg.norm(target_flat) / np.sqrt(target_flat.size)),
        "solver_norm": float(np.linalg.norm(solver_flat) / np.sqrt(solver_flat.size)),
        "feedback_norm": float(np.linalg.norm(feedback_flat) / np.sqrt(feedback_flat.size)),
        "solver_target_cosine": float(np.dot(solver_flat, target_flat) / ((np.linalg.norm(solver_flat) * np.linalg.norm(target_flat)) + 1e-12)),
        "feedback_target_cosine": float(np.dot(feedback_flat, target_flat) / ((np.linalg.norm(feedback_flat) * np.linalg.norm(target_flat)) + 1e-12)),
        "feedback_target_projection": float(np.dot(feedback_flat, target_flat) / target_norm_sq),
        "feedback_target_sign_agreement": float(np.mean(np.sign(feedback_flat) == np.sign(target_flat))),
        "solver_target_projection": float(np.dot(solver_flat, target_flat) / target_norm_sq),
        "did_to_f1_l2": float(np.linalg.norm((base["P1"] + feedback_delta - base["F1"]).reshape(-1)) / np.sqrt(target_flat.size)),
        "did_less_error_than_pf": float(np.linalg.norm((base["P1"] + feedback_delta - base["F1"]).reshape(-1)) < np.linalg.norm((base["PF"] - base["F1"]).reshape(-1))),
        "did_less_error_than_pp": float(np.linalg.norm((base["P1"] + feedback_delta - base["F1"]).reshape(-1)) < np.linalg.norm((base["PP"] - base["F1"]).reshape(-1))),
    }


def call(
    cfg: Any,
    model: Any,
    stats: dict,
    observation: Any,
    instruction: str,
    seed: int,
    steps: int,
    previous_latent: torch.Tensor | None = None,
    persistent: bool = False,
) -> dict[str, Any]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    started = time.perf_counter()
    output = get_action(
        cfg,
        model,
        stats,
        obs_dict(observation),
        instruction,
        seed=seed,
        randomize_seed=False,
        num_denoising_steps_action=steps,
        generate_future_state_and_value_in_parallel=False,
        decode_future_state=False,
        skip_vae_encoding=previous_latent is not None,
        previous_generated_latent=previous_latent,
        skip_camera_preprocessing=previous_latent is not None and not persistent,
        persistent_visual_correction_prefix_frames=13 if persistent else None,
        persistent_visual_correction_arrival=1,
    )
    output["__latency_ms"] = (time.perf_counter() - started) * 1e3
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--split", choices=("discovery", "validation", "heldout"), default=None)
    parser.add_argument("--checkpoint", default="/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    parser.add_argument("--dataset-stats", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json")
    parser.add_argument("--t5-embeddings", default="/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/cosmos_libero_pro_t5_embeddings.pkl")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if "so101" in str(checkpoint).lower() or "finet" in str(checkpoint).lower() or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("refusing non-original checkpoint")
    rows = {row["episode_key"]: row for row in load_jsonl(args.manifest)}
    assigned = [row for row in rows.values() if int(row["episode_key"], 16) % args.num_shards == args.shard_index and (args.split is None or row["split"] == args.split)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = {path.stem for path in args.output_dir.glob("state_*.pt")}
    episode_files = list(args.collection_root.glob("group*/episode_*.pt")) + list(args.collection_root.glob("episode_*.pt"))
    collected = {path.stem.removeprefix("episode_"): path for path in episode_files}
    todo = [row for row in assigned if row["episode_key"] in collected]
    if not todo:
        print(json.dumps({"status": "no_collected_rows", "assigned": len(assigned)}), flush=True)
        return
    first = todo[0]
    install_libero_checkout(Path(first["libero_repo"]), Path(first["libero_config_path"]))
    import os
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    cfg_args = build_args(args)
    cfg = build_cfg(cfg_args)
    model, stats = load_model(cfg, cfg_args)
    model.eval()
    counts = {"episodes": 0, "states": 0, "skipped_states": 0, "failed_states": 0}
    for row in todo:
        episode_path = collected[row["episode_key"]]
        episode = torch.load(episode_path, weights_only=False)
        requests = episode["requests"]
        env = None
        try:
            env = ManifestLiberoEnvironment(row, 256, None)
            env.reset()
            for request_index in range(1, len(requests)):
                request = requests[request_index]
                state_key = request["state_key"]
                output_path = args.output_dir / f"state_{state_key.replace(':', '_')}.pt"
                if output_path.stem in done or output_path.exists():
                    counts["skipped_states"] += 1
                    continue
                restore(env, np.asarray(request["sim_state"], dtype=np.float64))
                raw = env.env.regenerate_obs_from_state(np.asarray(request["sim_state"], dtype=np.float64))
                observation = extract_observation(raw, flip_vertical=True)
                previous = torch.from_numpy(np.asarray(requests[request_index - 1]["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
                results: dict[str, dict[str, Any]] = {}
                with torch.inference_mode():
                    results["F1"] = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), 1)
                    results["P1"] = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), 1, previous)
                    results["PP"] = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), 2, previous)
                    results["PF"] = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), 2, previous, persistent=True)
                    results["FF"] = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), 2)
                actions = {name: action_array(value) for name, value in results.items()}
                future = {name: np.asarray(value["generated_latent"].detach().float().cpu().numpy(), dtype=np.float32)[:, :, (5, 6, 7)] for name, value in results.items()}
                metrics = {
                    "action_to_F1": {name: vector_metrics(value, actions["F1"]) for name, value in actions.items()},
                    "future_to_F1": {name: vector_metrics(value, future["F1"]) for name, value in future.items()},
                    "action_innovation": innovation_metrics(actions, actions["PP"], actions["PF"]),
                    "future_innovation": innovation_metrics(future, future["PP"], future["PF"]),
                    "latency_ms": {name: float(results[name].get("__latency_ms", 0.0)) for name in results},
                }
                output = {
                    "schema_version": 1,
                    "state_key": state_key,
                    "episode_key": row["episode_key"],
                    "split": row["split"],
                    "task_uid": row["task_uid"],
                    "task_name": row["task_name"],
                    "suite": row["suite"],
                    "init_state_index": row["init_state_index"],
                    "seed": row["seed"],
                    "request_index": request_index,
                    "control_step": request["control_step"],
                    "actions": actions,
                    "metrics": metrics,
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "value_used": False,
                    "privileged_runtime_state_used": False,
                    "scheduler_or_threshold_used": False,
                }
                temporary = output_path.with_suffix(".partial.pt")
                torch.save(output, temporary)
                temporary.replace(output_path)
                done.add(output_path.stem)
                counts["states"] += 1
                if counts["states"] % 10 == 0:
                    print(json.dumps({"shard": args.shard_index, **counts}), flush=True)
        except Exception as error:
            counts["failed_states"] += 1
            print(json.dumps({"episode": row["episode_key"][:12], "error": f"{type(error).__name__}:{error}"}), flush=True)
        finally:
            if env is not None:
                env.close()
        counts["episodes"] += 1
    summary = {"schema_version": 1, "shard_index": args.shard_index, "num_shards": args.num_shards, "split": args.split, "counts": counts, "checkpoint_sha256": CHECKPOINT_SHA256, "finished_at_ns": time.time_ns()}
    (args.output_dir / f"summary_shard{args.shard_index:02d}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

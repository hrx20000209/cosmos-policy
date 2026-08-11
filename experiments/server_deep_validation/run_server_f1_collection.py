"""Collect resumable one-step Fresh trajectories for server-side validation.

This is the only policy-driven state collection pass.  It records exact
MuJoCo snapshots and the generated non-value latent needed to construct P1 at
the next request, but it never reads Cosmos' value output or simulator state
inside policy inference.  Every episode is an independent output file, so a
worker can resume after interruption without replaying completed episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (  # noqa: E402
    ManifestLiberoEnvironment,
    install_libero_checkout,
    load_jsonl,
)
from experiments.libero_harness import extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import build_cfg, load_model  # noqa: E402


CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def args_for_build(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        t5_embeddings=args.t5_embeddings,
        action_horizon=args.action_horizon,
    )


def collect_episode(env: ManifestLiberoEnvironment, model: Any, dataset_stats: dict, row: dict, args: argparse.Namespace) -> dict:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    raw = env.reset()
    settle = np.zeros(7, dtype=np.float32)
    settle[-1] = float(args.settle_gripper_action)
    for _ in range(int(args.settle_steps)):
        raw, _, _, _ = env.step(settle)

    requests: list[dict[str, Any]] = []
    executed: list[np.ndarray] = []
    control_step = 0
    request_index = 0
    success = False
    started = time.perf_counter()
    while control_step < int(row["max_steps"]) and not success:
        if args.max_requests is not None and request_index >= int(args.max_requests):
            break
        observation = extract_observation(raw, flip_vertical=True)
        sim_state = np.asarray(env.env.get_sim_state(), dtype=np.float64)
        obs_dict = {
            "primary_image": observation.primary_image,
            "wrist_image": observation.wrist_image,
            "proprio": observation.proprio,
        }
        request_started = time.perf_counter()
        with torch.inference_mode():
            result = get_action(
                build_cfg(args_for_build(args)),
                model,
                dataset_stats,
                obs_dict,
                row["instruction"],
                seed=int(row["seed"]),
                randomize_seed=False,
                num_denoising_steps_action=1,
                generate_future_state_and_value_in_parallel=False,
                decode_future_state=False,
            )
        action_chunk = np.asarray(result["actions"], dtype=np.float32).reshape(args.action_horizon, 7)
        generated_latent = result["generated_latent"].detach().float().to("cpu").contiguous().numpy().astype(np.float16)
        requests.append(
            {
                "state_key": f"{row['episode_key']}:req{request_index}",
                "request_index": request_index,
                "control_step": control_step,
                "sim_state": sim_state,
                "proprio": np.asarray(observation.proprio, dtype=np.float32),
                "fresh_action": action_chunk,
                "generated_latent": generated_latent,
                "policy_latency_ms": (time.perf_counter() - request_started) * 1e3,
                "checkpoint_sha256": CHECKPOINT_SHA256,
            }
        )
        for action in action_chunk:
            executed.append(action.copy())
            raw, _, done, _ = env.step(action)
            control_step += 1
            if done:
                success = bool(env.env.check_success()) if hasattr(env.env, "check_success") else True
                break
            if control_step >= int(row["max_steps"]):
                break
        request_index += 1

    return {
        "schema_version": 1,
        "episode_key": row["episode_key"],
        "split": row["split"],
        "task_uid": row["task_uid"],
        "task_name": row["task_name"],
        "suite": row["suite"],
        "instruction": row["instruction"],
        "seed": row["seed"],
        "init_state_index": row["init_state_index"],
        "success": bool(success),
        "termination_reason": "success" if success else ("profile_request_cap" if args.max_requests is not None and request_index >= int(args.max_requests) else "max_steps"),
        "control_steps": control_step,
        "requests": requests,
        "executed_actions": np.stack(executed) if executed else np.zeros((0, 7), dtype=np.float32),
        "collection_wall_clock_s": time.perf_counter() - started,
        "value_used": False,
        "privileged_runtime_state_used": False,
        "denoising_steps": 1,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--checkpoint", default="/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    parser.add_argument("--dataset-stats", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json")
    parser.add_argument("--t5-embeddings", default="/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/cosmos_libero_pro_t5_embeddings.pkl")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--settle-gripper-action", type=float, default=-1.0)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--libero-repo", default="/home/rxhuang/Projects/LIBERO")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    checkpoint = Path(args.checkpoint).resolve()
    if "so101" in str(checkpoint).lower() or "finet" in str(checkpoint).lower():
        raise ValueError(f"refusing finetuned/SO101 checkpoint: {checkpoint}")
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")

    rows = [row for row in load_jsonl(args.manifest) if int(row["episode_key"], 16) % args.num_shards == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = {path.stem for path in args.output_dir.glob("episode_*.pt")}
    if not rows or all(row["episode_key"] in done for row in rows):
        print(json.dumps({"status": "already_complete", "shard": args.shard_index, "rows": len(rows)}), flush=True)
        return

    first = rows[0]
    install_libero_checkout(Path(first["libero_repo"]), Path(first["libero_config_path"]))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    cfg = build_cfg(args_for_build(args))
    model, dataset_stats = load_model(cfg, args_for_build(args))
    model.eval()
    counts = {"assigned": len(rows), "skipped": 0, "recorded": 0, "success": 0, "episode_failures": 0, "worker_errors": 0}
    started_ns = time.time_ns()
    for row in rows:
        target = args.output_dir / f"episode_{row['episode_key']}.pt"
        if target.exists():
            counts["skipped"] += 1
            continue
        env = None
        try:
            env = ManifestLiberoEnvironment(row, 256, None)
            episode = collect_episode(env, model, dataset_stats, row, args)
            temporary = target.with_suffix(".partial.pt")
            torch.save(episode, temporary)
            temporary.replace(target)
            counts["recorded"] += 1
            counts["success" if episode["success"] else "episode_failures"] += 1
            print(json.dumps({"episode": row["episode_key"][:12], "task": row["task_uid"], "split": row["split"], "requests": len(episode["requests"]), "success": episode["success"]}), flush=True)
        except Exception as error:
            counts["worker_errors"] += 1
            failure = {"episode_key": row["episode_key"], "task_uid": row["task_uid"], "error": f"{type(error).__name__}:{error}", "value_used": False}
            (args.output_dir / f"episode_{row['episode_key']}.failed.json").write_text(json.dumps(failure, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps(failure, ensure_ascii=False), flush=True)
        finally:
            if env is not None:
                env.close()
    summary = {
        "schema_version": 1,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "counts": counts,
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "value_used": False,
        "privileged_runtime_state_used": False,
        "denoising_steps": 1,
    }
    (args.output_dir / f"summary_shard{args.shard_index:02d}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    # Cosmos/TransformerEngine may leave CUDA helper threads alive after the
    # last JSON checkpoint.  The episode artifacts are already atomically
    # committed, so terminate this owned worker at the safe episode boundary
    # instead of making the supervisor wait indefinitely on interpreter
    # teardown.  No external process is affected.
    os._exit(0)


if __name__ == "__main__":
    main()

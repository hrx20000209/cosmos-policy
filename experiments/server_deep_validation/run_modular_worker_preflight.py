"""Memory-capped 1-state F1/P1 preflight for the modular WAM runtime.

The preflight is intentionally small: one previously recorded, exact decision
state and exactly two routes.  It proves that the *fresh* route really encodes
the observation while P1 reuses the prior joint latent, and records peak VRAM
before any multi-state oracle work is allowed.

It is not a closed-loop benchmark and it does not implement a scheduler.
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

from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment, install_libero_checkout, load_jsonl  # noqa: E402
from experiments.libero_harness import extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import build_cfg, load_model  # noqa: E402
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402
from experiments.server_deep_validation.run_server_ablation import call  # noqa: E402
from wam_runtime.fixed_policy_harness import ORIGINAL_COSMOS_CHECKPOINT_SHA256  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cfg_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        t5_embeddings=args.t5_embeddings,
        action_horizon=16,
    )


def action_distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "full_rmse": float(np.linalg.norm(delta) / np.sqrt(delta.size)),
        "mean_step_l2": float(np.linalg.norm(delta, axis=1).mean()),
        "first_action_l2": float(np.linalg.norm(delta[0])),
        "gripper_abs": float(abs(delta[0, 6])),
    }


def find_state(collection_root: Path, state_key: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    episode_key, _, suffix = state_key.partition(":req")
    if not episode_key or not suffix.isdigit():
        raise ValueError("state-key must have the form <episode_key>:req<index>")
    candidates = sorted(collection_root.rglob(f"episode_{episode_key}.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one collection episode for {episode_key}, got {len(candidates)}")
    episode = torch.load(candidates[0], map_location="cpu", weights_only=False)
    index = int(suffix)
    requests = episode.get("requests", [])
    if not 1 <= index < len(requests):
        raise IndexError(f"request {index} is unavailable in {episode_key}; requests={len(requests)}")
    source, target = requests[index - 1], requests[index]
    if target.get("state_key") != state_key:
        raise ValueError("state-key does not match collection request provenance")
    if int(target["control_step"]) - int(source["control_step"]) != 16:
        raise ValueError("preflight only accepts a physically aligned 16-action endpoint")
    return candidates[0], episode, source, target


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/server_deep_validation/manifests/full_scale_40_task.jsonl"))
    parser.add_argument("--collection-root", type=Path, default=Path("/data/rxhuang/wam_full_scale_server/queue_a/f1_collection"))
    parser.add_argument("--state-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--checkpoint", default="/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    parser.add_argument("--dataset-stats", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json")
    parser.add_argument("--t5-embeddings", default="/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/cosmos_libero_pro_t5_embeddings.pkl")
    args = parser.parse_args()
    if not 0.05 <= args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in [0.05, 1.0]")
    checkpoint = Path(args.checkpoint).resolve()
    if "so101" in str(checkpoint).lower() or "finetun" in str(checkpoint).lower():
        raise ValueError(f"refusing finetuned/SO101 checkpoint: {checkpoint}")
    if sha256(checkpoint) != ORIGINAL_COSMOS_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("this preflight requires CUDA")

    # CUDA_VISIBLE_DEVICES maps the selected physical card to logical device 0.
    device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    device_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    started = time.perf_counter()
    manifest = {row["episode_key"]: row for row in load_jsonl(args.manifest)}
    env: ManifestLiberoEnvironment | None = None
    result: dict[str, Any]
    try:
        episode_path, episode, source, target = find_state(args.collection_root, args.state_key)
        row = manifest.get(episode["episode_key"])
        if row is None:
            raise ValueError("collection episode does not appear in manifest")
        for key in ("split", "task_uid", "seed", "init_state_index"):
            if episode.get(key) != row.get(key):
                raise ValueError(f"manifest provenance mismatch: {key}")
        install_libero_checkout(Path(row["libero_repo"]), Path(row["libero_config_path"]))
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
        os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        cfg = build_cfg(cfg_args(args))
        model, stats = load_model(cfg, cfg_args(args))
        model.eval()
        env = ManifestLiberoEnvironment(row, 256, None)
        env.reset()
        sim_state = np.asarray(target["sim_state"], dtype=np.float64)
        restore(env, sim_state)
        observation = extract_observation(env.env.regenerate_obs_from_state(sim_state), flip_vertical=True)
        previous = torch.from_numpy(np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
        with torch.inference_mode():
            # F1 must encode the restored physical observation; it has no
            # previous latent and therefore cannot accidentally be P1.
            fresh = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), 1)
            predicted = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), 1, previous)
        fresh_action = np.asarray(fresh["actions"], dtype=np.float32).reshape(16, 7)
        predicted_action = np.asarray(predicted["actions"], dtype=np.float32).reshape(16, 7)
        baseline = action_distance(predicted_action, fresh_action)
        free_after, total_after = torch.cuda.mem_get_info(device)
        result = {
            "schema_version": 1,
            "experiment": "modular_wam_fresh_vs_predicted_preflight",
            "status": "PASS" if baseline["mean_step_l2"] > 1e-6 else "FAIL_ZERO_FRESH_BASELINE",
            "state_key": args.state_key,
            "episode_key": episode["episode_key"],
            "task_uid": row["task_uid"],
            "split": row["split"],
            "init_state_index": row["init_state_index"],
            "collection_episode": str(episode_path),
            "baseline_predicted_to_fresh": baseline,
            "route_contract": {
                "F1": {"skip_vae_encoding": False, "previous_generated_latent": False, "denoising_steps": 1},
                "P1": {"skip_vae_encoding": True, "previous_generated_latent": True, "denoising_steps": 1},
            },
            "offline_restore_used_for_reproducibility_only": True,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "checkpoint_sha256": ORIGINAL_COSMOS_CHECKPOINT_SHA256,
            "gpu": {
                "visible_device": device,
                "name": device_name,
                "memory_fraction_cap": args.memory_fraction,
                "cap_mb": total_memory * args.memory_fraction / 1024**2,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
                "free_after_mb": free_after / 1024**2,
                "total_mb": total_after / 1024**2,
            },
            "latency_ms": {"F1": float(fresh["__latency_ms"]), "P1": float(predicted["__latency_ms"])},
            "wall_clock_s": time.perf_counter() - started,
        }
    except Exception as error:
        result = {
            "schema_version": 1,
            "experiment": "modular_wam_fresh_vs_predicted_preflight",
            "status": "FAIL_RUNTIME",
            "state_key": args.state_key,
            "error": f"{type(error).__name__}:{error}",
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "checkpoint_sha256": ORIGINAL_COSMOS_CHECKPOINT_SHA256,
            "gpu": {
                "visible_device": device,
                "name": device_name,
                "memory_fraction_cap": args.memory_fraction,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
            },
            "wall_clock_s": time.perf_counter() - started,
        }
    finally:
        if env is not None:
            env.close()
    write_result(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    # The result is already atomically committed.  Avoid keeping CUDA helper
    # threads alive on a shared server after this single-state gate.
    os._exit(0)


if __name__ == "__main__":
    main()

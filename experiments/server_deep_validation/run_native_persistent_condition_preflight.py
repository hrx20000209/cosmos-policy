"""Small native persistent-visual-condition preflight for frozen Cosmos.

This is deliberately a one-state, denoise=1 test.  It compares:

* F1: a normal fresh call that VAE-encodes the full current observation;
* P1: speculative latent reuse with no fresh camera preprocessing; and
* PV0: the existing native ``persistent_visual_correction`` API, which starts
  from P1 but VAE-encodes only the causal fresh visual prefix and installs it
  before the one denoiser forward.

PV0 is not a hidden-activation patch and does not use Cosmos value.  The
recorded simulator state is used solely to reproduce a stored physical camera
observation; it is never supplied to the policy.
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
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment, install_libero_checkout, load_jsonl  # noqa: E402
from experiments.libero_harness import extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import build_cfg, load_model  # noqa: E402
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402
from wam_runtime.fixed_policy_harness import ORIGINAL_COSMOS_CHECKPOINT_SHA256  # noqa: E402


PREFIX_FRAMES = 13
ROUTES = ("F1", "P1", "PV0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cfg_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        t5_embeddings=args.t5_embeddings,
        action_horizon=16,
    )


def obs_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def action_array(result: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)


def action_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    per_step = np.linalg.norm(delta, axis=1)
    return {
        "full_rmse": float(np.linalg.norm(delta) / np.sqrt(delta.size)),
        "mean_step_l2": float(per_step.mean()),
        "first_action_l2": float(per_step[0]),
        "gripper_abs": float(abs(delta[0, 6])),
    }


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def find_state(collection_root: Path, state_key: str) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    episode_key, separator, suffix = state_key.partition(":req")
    if not separator or not suffix.isdigit():
        raise ValueError("state-key must have the form <episode_key>:req<index>")
    candidates = sorted(collection_root.rglob(f"episode_{episode_key}.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one collection episode for {episode_key}, got {len(candidates)}")
    episode = torch.load(candidates[0], map_location="cpu", weights_only=False)
    request_index = int(suffix)
    requests = episode.get("requests", [])
    if not 1 <= request_index < len(requests):
        raise IndexError(f"request {request_index} unavailable; requests={len(requests)}")
    source, target = requests[request_index - 1], requests[request_index]
    if target.get("state_key") != state_key:
        raise ValueError("state-key does not match collection provenance")
    if int(target["control_step"]) - int(source["control_step"]) != 16:
        raise ValueError("preflight requires a physically aligned 16-action endpoint")
    return candidates[0], episode, source, target


def route_contract(name: str) -> dict[str, Any]:
    common = {"denoising_steps": 1, "value_used": False, "scheduler_or_threshold_used": False}
    if name == "F1":
        return {**common, "skip_vae_encoding": False, "previous_generated_latent": False, "skip_camera_preprocessing": False}
    if name == "P1":
        return {**common, "skip_vae_encoding": True, "previous_generated_latent": True, "skip_camera_preprocessing": True}
    if name == "PV0":
        return {
            **common,
            "skip_vae_encoding": True,
            "previous_generated_latent": True,
            "skip_camera_preprocessing": False,
            "native_persistent_visual_correction": True,
            "fresh_visual_prefix_frames": PREFIX_FRAMES,
            "fresh_visual_arrival_denoiser_forward": 0,
            "async_predict_correct": False,
        }
    raise ValueError(name)


def run_route(
    name: str,
    *,
    cfg: Any,
    model: Any,
    stats: dict[str, Any],
    observation: Any,
    instruction: str,
    seed: int,
    previous: torch.Tensor,
) -> tuple[np.ndarray, dict[str, float]]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    contract = route_contract(name)
    metrics: dict[str, Any] = {}
    cfg._inference_metrics_sink = metrics
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        result = get_action(
            cfg,
            model,
            stats,
            obs_dict(observation),
            instruction,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=1,
            generate_future_state_and_value_in_parallel=False,
            decode_future_state=False,
            skip_vae_encoding=bool(contract["skip_vae_encoding"]),
            previous_generated_latent=previous if contract["previous_generated_latent"] else None,
            skip_camera_preprocessing=bool(contract["skip_camera_preprocessing"]),
            persistent_visual_correction_prefix_frames=contract.get("fresh_visual_prefix_frames"),
            persistent_visual_correction_arrival=int(contract.get("fresh_visual_arrival_denoiser_forward", 1)),
            async_predict_correct=bool(contract.get("async_predict_correct", False)),
        )
        torch.cuda.synchronize()
    finally:
        cfg._inference_metrics_sink = None
        model.inference_condition_transform = None
    metrics["wall_latency_ms"] = (time.perf_counter() - started) * 1e3
    numeric_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(float(value))
    }
    return action_array(result), numeric_metrics


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/server_deep_validation/manifests/full_scale_40_task.jsonl"))
    parser.add_argument("--collection-root", type=Path, default=Path("/data/rxhuang/wam_full_scale_server/queue_a/f1_collection"))
    parser.add_argument("--state-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--checkpoint", default="/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    parser.add_argument("--dataset-stats", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json")
    parser.add_argument("--t5-embeddings", default="/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/cosmos_libero_pro_t5_embeddings.pkl")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be >=1 and warmup must be >=0")
    if not 0.05 <= args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in [0.05, 1.0]")
    checkpoint = Path(args.checkpoint).resolve()
    if "so101" in str(checkpoint).lower() or "finetun" in str(checkpoint).lower():
        raise ValueError("refusing finetuned/SO101 checkpoint")
    if sha256(checkpoint) != ORIGINAL_COSMOS_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("this preflight requires CUDA")

    device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    env: ManifestLiberoEnvironment | None = None
    try:
        manifest = {row["episode_key"]: row for row in load_jsonl(args.manifest)}
        episode_path, episode, source, target = find_state(args.collection_root, args.state_key)
        row = manifest.get(episode["episode_key"])
        if row is None:
            raise ValueError("collection episode absent from manifest")
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
            for _ in range(args.warmup):
                run_route(
                    "P1",
                    cfg=cfg,
                    model=model,
                    stats=stats,
                    observation=observation,
                    instruction=row["instruction"],
                    seed=int(row["seed"]),
                    previous=previous,
                )
            trials: list[dict[str, Any]] = []
            for repeat_index in range(args.repeats):
                order = ROUTES[repeat_index % len(ROUTES) :] + ROUTES[: repeat_index % len(ROUTES)]
                outputs: dict[str, np.ndarray] = {}
                metrics: dict[str, dict[str, float]] = {}
                for name in order:
                    outputs[name], metrics[name] = run_route(
                        name,
                        cfg=cfg,
                        model=model,
                        stats=stats,
                        observation=observation,
                        instruction=row["instruction"],
                        seed=int(row["seed"]),
                        previous=previous,
                    )
                baseline = action_difference(outputs["P1"], outputs["F1"])
                native_error = action_difference(outputs["PV0"], outputs["F1"])
                trials.append(
                    {
                        "repeat_index": repeat_index,
                        "execution_order": list(order),
                        "baseline_p1_to_f1": baseline,
                        "native_pv0_to_f1": native_error,
                        "native_action_recovery": 1.0 - native_error["mean_step_l2"] / max(baseline["mean_step_l2"], 1e-8),
                        "latency_ms": {name: metrics[name].get("wall_latency_ms") for name in ROUTES},
                        "inference_metrics": metrics,
                    }
                )
        route_latency = {
            name: numeric_summary([float(trial["latency_ms"][name]) for trial in trials if trial["latency_ms"][name] is not None])
            for name in ROUTES
        }
        summary = {
            "baseline_p1_to_f1_mean_step_l2": numeric_summary([trial["baseline_p1_to_f1"]["mean_step_l2"] for trial in trials]),
            "native_pv0_to_f1_mean_step_l2": numeric_summary([trial["native_pv0_to_f1"]["mean_step_l2"] for trial in trials]),
            "native_action_recovery": numeric_summary([trial["native_action_recovery"] for trial in trials]),
            "route_wall_latency_ms": route_latency,
            "native_recovery_positive_all_repeats": all(trial["native_action_recovery"] > 0.0 for trial in trials),
        }
        free_after, total_memory = torch.cuda.mem_get_info(device)
        result: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "native_persistent_visual_condition_preflight",
            "status": "PASS_EXECUTED" if summary["baseline_p1_to_f1_mean_step_l2"]["median"] not in (None, 0.0) else "FAIL_ZERO_FRESH_BASELINE",
            "state_key": args.state_key,
            "episode_key": episode["episode_key"],
            "task_uid": row["task_uid"],
            "split": row["split"],
            "init_state_index": row["init_state_index"],
            "collection_episode": str(episode_path),
            "checkpoint_sha256": ORIGINAL_COSMOS_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "routes": {name: route_contract(name) for name in ROUTES},
            "native_interface": {
                "api": "get_action:persistent_visual_correction_prefix_frames",
                "fresh_visual_prefix_frames": PREFIX_FRAMES,
                "arrival_denoiser_forward": 0,
                "hidden_activation_patch_used": False,
                "fresh_prefix_oracle_used": False,
            },
            "offline_restore_used_for_reproducibility_only": True,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "trials": trials,
            "summary": summary,
            "gpu": {
                "visible_device": device,
                "name": torch.cuda.get_device_name(device),
                "memory_fraction_cap": args.memory_fraction,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
                "free_after_mb": free_after / 1024**2,
                "total_mb": total_memory / 1024**2,
            },
            "wall_clock_s": time.perf_counter() - started,
        }
    except Exception as error:
        result = {
            "schema_version": 1,
            "experiment": "native_persistent_visual_condition_preflight",
            "status": "FAIL_RUNTIME",
            "state_key": args.state_key,
            "checkpoint_sha256": ORIGINAL_COSMOS_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "error": f"{type(error).__name__}:{error}",
            "gpu": {
                "visible_device": device,
                "name": torch.cuda.get_device_name(device),
                "memory_fraction_cap": args.memory_fraction,
                "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
            },
            "wall_clock_s": time.perf_counter() - started,
        }
    finally:
        if env is not None:
            env.close()
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Shared, auditable primitives for the frozen native-PV0 overnight run.

The helpers in this module intentionally expose only three policy routes:

``F1``
    Re-encode the current physical camera observation.
``P1``
    Reuse the preceding generated joint latent.
``PV0``
    Reuse that latent while assimilating a causal, current visual prefix via
    Cosmos' native persistent-condition API before its sole denoiser forward.

The simulator state stored by the Foundation dataset is used *only* to
reconstruct a recorded camera observation for offline evaluation.  It is never
passed to the policy.  Cosmos' value output is never inspected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_CHECKPOINT = Path(
    "/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt"
)
ORIGINAL_CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
DEFAULT_DATASET_STATS = Path(
    "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
)
DEFAULT_T5_EMBEDDINGS = Path(
    "/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/"
    "cosmos_libero_pro_t5_embeddings.pkl"
)
PREFIX_FRAMES = 13
ACTION_HORIZON = 16
ACTION_DIM = 7
ROUTES = ("F1", "P1", "PV0")
PREFIX_LENGTHS = (4, 8, 12, 16)
EPS = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a portable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint_contract(checkpoint: Path = ORIGINAL_CHECKPOINT) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    lower = str(checkpoint).lower()
    if "so101" in lower or "finetun" in lower:
        raise ValueError(f"refusing SO101/finetuned checkpoint: {checkpoint}")
    actual = sha256(checkpoint)
    if actual != ORIGINAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            "original pre-finetune checkpoint SHA256 mismatch: "
            f"expected={ORIGINAL_CHECKPOINT_SHA256}, actual={actual}"
        )
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": actual, "finetuning_used": False}


def route_contract(route: str) -> dict[str, Any]:
    common = {
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_used": False,
        "scheduler_or_threshold_used": False,
        "hidden_activation_patch_used": False,
        "fresh_prefix_oracle_used": False,
    }
    if route == "F1":
        return {
            **common,
            "skip_vae_encoding": False,
            "previous_generated_latent": False,
            "skip_camera_preprocessing": False,
        }
    if route == "P1":
        return {
            **common,
            "skip_vae_encoding": True,
            "previous_generated_latent": True,
            "skip_camera_preprocessing": True,
        }
    if route == "PV0":
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
    raise ValueError(f"unknown route={route!r}")


def route_contract_passes(payload: Mapping[str, Any]) -> bool:
    routes = payload.get("route_contract") or payload.get("routes")
    if not isinstance(routes, Mapping):
        return False
    try:
        return (
            int(payload.get("denoising_steps", -1)) == 1
            and payload.get("checkpoint_sha256") == ORIGINAL_CHECKPOINT_SHA256
            and payload.get("value_used") is False
            and payload.get("privileged_runtime_state_used") is False
            and payload.get("scheduler_or_threshold_used") is False
            and routes["F1"] == route_contract("F1")
            and routes["P1"] == route_contract("P1")
            and routes["PV0"] == route_contract("PV0")
        )
    except (KeyError, TypeError):
        return False


def action_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(f"expected {ACTION_HORIZON}x{ACTION_DIM} actions, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("action array contains non-finite values")
    return np.ascontiguousarray(array)


def pair_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Fresh-policy discrepancy at whole-chunk and fixed execution prefixes."""

    left, right = action_array(left), action_array(right)
    delta = left.astype(np.float64) - right.astype(np.float64)
    step_l2 = np.linalg.norm(delta, axis=1)
    prefixes: dict[str, dict[str, float]] = {}
    for prefix in PREFIX_LENGTHS:
        prefix_delta = delta[:prefix]
        prefixes[str(prefix)] = {
            "rmse": float(np.linalg.norm(prefix_delta) / math.sqrt(prefix_delta.size)),
            "mean_step_l2": float(step_l2[:prefix].mean()),
            "endpoint_l2": float(step_l2[prefix - 1]),
            "first_action_l2": float(step_l2[0]),
            "first_gripper_abs": float(abs(delta[0, 6])),
            "first_gripper_sign_match": bool(np.sign(left[0, 6]) == np.sign(right[0, 6])),
            "prefix_gripper_sign_match_fraction": float(
                np.mean(np.sign(left[:prefix, 6]) == np.sign(right[:prefix, 6]))
            ),
        }
    return {
        "full_rmse": float(np.linalg.norm(delta) / math.sqrt(delta.size)),
        "mean_step_l2": float(step_l2.mean()),
        "first_action_l2": float(step_l2[0]),
        "end_action_l2": float(step_l2[-1]),
        "first_gripper_abs": float(abs(delta[0, 6])),
        "first_gripper_sign_match": bool(np.sign(left[0, 6]) == np.sign(right[0, 6])),
        "prefixes": prefixes,
    }


def recovery_metrics(p1_to_f1: Mapping[str, Any], pv0_to_f1: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    baseline = float(p1_to_f1["mean_step_l2"])
    candidate = float(pv0_to_f1["mean_step_l2"])
    output["absolute_gain_mean_step_l2"] = baseline - candidate
    output["native_beats_p1"] = bool(candidate < baseline)
    output["normalized_recovery_mean_step_l2"] = (
        float(1.0 - candidate / baseline) if baseline > EPS else None
    )
    prefixes: dict[str, Any] = {}
    for prefix in PREFIX_LENGTHS:
        p1_prefix = float(p1_to_f1["prefixes"][str(prefix)]["mean_step_l2"])
        pv0_prefix = float(pv0_to_f1["prefixes"][str(prefix)]["mean_step_l2"])
        prefixes[str(prefix)] = {
            "absolute_gain_mean_step_l2": p1_prefix - pv0_prefix,
            "native_beats_p1": bool(pv0_prefix < p1_prefix),
            "normalized_recovery_mean_step_l2": (
                float(1.0 - pv0_prefix / p1_prefix) if p1_prefix > EPS else None
            ),
        }
    output["prefixes"] = prefixes
    return output


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def set_up_cuda(memory_fraction: float) -> dict[str, Any]:
    if not 0.05 <= memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in [0.05, 1.0]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native PV0 evaluation")
    device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "visible_device": int(device),
        "name": torch.cuda.get_device_name(device),
        "memory_fraction_cap": float(memory_fraction),
        "free_before_mib": float(free / 2**20),
        "total_mib": float(total / 2**20),
        "physical_gpu": os.environ.get("EVAL_PHYSICAL_GPU"),
    }


def final_gpu_snapshot(device: int | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device() if device is None else device
    free, total = torch.cuda.mem_get_info(device)
    return {
        "visible_device": int(device),
        "name": torch.cuda.get_device_name(device),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
        "free_after_mib": float(free / 2**20),
        "total_mib": float(total / 2**20),
        "physical_gpu": os.environ.get("EVAL_PHYSICAL_GPU"),
    }


def configure_libero(row: Mapping[str, Any]) -> None:
    """Set per-process renderer paths before building an environment."""

    from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import install_libero_checkout

    install_libero_checkout(Path(row["libero_repo"]), Path(row["libero_config_path"]))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


def build_model(*, checkpoint: Path, dataset_stats: Path, t5_embeddings: Path) -> tuple[Any, dict[str, Any], Any]:
    """Load one frozen Cosmos worker and its minimal inference configuration."""

    from experiments.progressive_wam.run_p1_trajectory_dump import build_cfg, load_model

    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        dataset_stats=str(dataset_stats),
        t5_embeddings=str(t5_embeddings),
        action_horizon=ACTION_HORIZON,
    )
    cfg = build_cfg(args)
    model, stats = load_model(cfg, args)
    model.eval()
    return cfg, stats, model


def observation_dict(observation: Any) -> dict[str, Any]:
    return {
        "primary_image": observation.primary_image,
        "wrist_image": observation.wrist_image,
        "proprio": observation.proprio,
    }


def swapped_visual_observation(target: Any, source: Any) -> Any:
    """A fixed negative control: wrong fresh cameras, target proprio untouched."""

    return SimpleNamespace(
        primary_image=source.primary_image,
        wrist_image=source.wrist_image,
        proprio=target.proprio,
    )


def run_route(
    route: str,
    *,
    cfg: Any,
    model: Any,
    stats: Mapping[str, Any],
    observation: Any,
    instruction: str,
    seed: int,
    previous: torch.Tensor,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Execute a single legal route and return actions plus stage accounting."""

    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    contract = route_contract(route)
    metrics: dict[str, Any] = {}
    cfg._inference_metrics_sink = metrics
    device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    try:
        result = get_action(
            cfg,
            model,
            stats,
            observation_dict(observation),
            instruction,
            seed=int(seed),
            randomize_seed=False,
            num_denoising_steps_action=1,
            generate_future_state_and_value_in_parallel=False,
            decode_future_state=False,
            skip_vae_encoding=bool(contract["skip_vae_encoding"]),
            previous_generated_latent=previous if contract["previous_generated_latent"] else None,
            skip_camera_preprocessing=bool(contract["skip_camera_preprocessing"]),
            persistent_visual_correction_prefix_frames=contract.get("fresh_visual_prefix_frames"),
            persistent_visual_correction_arrival=int(contract.get("fresh_visual_arrival_denoiser_forward", 1)),
            async_predict_correct=False,
        )
        torch.cuda.synchronize(device)
    finally:
        cfg._inference_metrics_sink = None
        model.inference_condition_transform = None
        # The native route must leave no transform installed for the next route.
        if hasattr(model, "sampler"):
            model.sampler.x0_transform = None
    clean_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(float(value))
    }
    clean_metrics.update(
        {
            "wall_latency_ms": float((time.perf_counter_ns() - started) / 1e6),
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
        }
    )
    return action_array(result["actions"]), clean_metrics


def build_state_index(
    *, manifest_path: Path, collection_root: Path, output_path: Path
) -> dict[str, Any]:
    """Make a compact, deterministic index over the 3,801 aligned states."""

    from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import load_jsonl

    manifest_rows = load_jsonl(manifest_path)
    manifest_by_key = {str(row["episode_key"]): row for row in manifest_rows}
    if len(manifest_by_key) != len(manifest_rows):
        raise ValueError("manifest has duplicate episode keys")
    files = sorted(collection_root.rglob("episode_*.pt"))
    ordered: list[tuple[dict[str, Any], Path]] = []
    for path in files:
        episode_key = path.stem.removeprefix("episode_")
        row = manifest_by_key.get(episode_key)
        if row is not None:
            ordered.append((row, path))
    ordered.sort(key=lambda item: int(item[0].get("manifest_order", 0)))
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    for row, path in ordered:
        episode = torch.load(path, map_location="cpu", weights_only=False)
        if episode.get("checkpoint_sha256") != ORIGINAL_CHECKPOINT_SHA256:
            errors.append(f"wrong_checkpoint:{path}")
            continue
        requests = episode.get("requests", [])
        for target_index in range(1, len(requests)):
            source, target = requests[target_index - 1], requests[target_index]
            if int(target.get("control_step", -999)) - int(source.get("control_step", -999)) != ACTION_HORIZON:
                errors.append(f"unaligned:{episode.get('episode_key')}:{target_index}")
                continue
            entries.append(
                {
                    "state_key": str(target["state_key"]),
                    "episode_key": str(row["episode_key"]),
                    "collection_episode": str(path),
                    "source_request_index": int(target_index - 1),
                    "target_request_index": int(target_index),
                    "request_index": int(target.get("request_index", target_index)),
                    "control_step": int(target["control_step"]),
                    "split": str(row["split"]),
                    "task_uid": str(row["task_uid"]),
                    "task_name": str(row["task_name"]),
                    "suite": str(row["suite"]),
                    "instruction": str(row["instruction"]),
                    "init_state_index": int(row["init_state_index"]),
                    "seed": int(row["seed"]),
                    "manifest_order": int(row["manifest_order"]),
                    "libero_repo": str(row["libero_repo"]),
                    "libero_config_path": str(row["libero_config_path"]),
                    "bddl_path": str(row["bddl_path"]),
                    "init_path": str(row["init_path"]),
                }
            )
    entries.sort(key=lambda item: (item["manifest_order"], item["target_request_index"]))
    for index, entry in enumerate(entries):
        entry["global_index"] = index
    if len({entry["state_key"] for entry in entries}) != len(entries):
        errors.append("duplicate_state_key")
    if errors:
        raise RuntimeError(f"Foundation state index audit failed: {errors[:10]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "state_count": len(entries),
        "task_count": len({entry["task_uid"] for entry in entries}),
        "split_counts": {
            split: sum(entry["split"] == split for entry in entries)
            for split in ("discovery", "validation", "heldout")
        },
        "manifest": str(manifest_path),
        "collection_root": str(collection_root),
        "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_used": False,
    }
    atomic_write_json(output_path.with_name("state_index_audit.json"), audit)
    return audit


def build_condition_compile_anchors(*, state_index_path: Path, output_path: Path) -> dict[str, Any]:
    """Freeze one result-independent state per task and a cyclic visual control."""

    entries = read_jsonl(state_index_path)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_task.setdefault(str(entry["task_uid"]), []).append(entry)
    tasks = sorted(by_task)
    anchors: list[dict[str, Any]] = []
    for task in tasks:
        candidates = by_task[task]
        target = min(
            candidates,
            key=lambda item: hashlib.sha256(f"S4-anchor:{item['state_key']}".encode()).hexdigest(),
        )
        anchors.append({"target": target})
    for index, anchor in enumerate(anchors):
        partner = anchors[(index + 1) % len(anchors)]["target"]
        if partner["task_uid"] == anchor["target"]["task_uid"]:
            raise RuntimeError("S4 shuffled visual partner unexpectedly has same task")
        anchor["anchor_index"] = index
        anchor["shuffled_visual_source"] = partner
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for anchor in anchors:
            handle.write(json.dumps(anchor, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "selection": "per-task SHA256-minimum state key; independent of route outcomes",
        "shuffled_control": "next lexical task's frozen physical camera observation; target proprio preserved",
        "anchor_count": len(anchors),
        "task_count": len(tasks),
        "split_counts": {
            split: sum(anchor["target"]["split"] == split for anchor in anchors)
            for split in ("discovery", "validation", "heldout")
        },
        "state_index": str(state_index_path),
        "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_used": False,
    }
    atomic_write_json(output_path.with_name("condition_compile_anchor_audit.json"), audit)
    return audit


def record_path_for_state(output_dir: Path, entry: Mapping[str, Any]) -> Path:
    return output_dir / f"state_{int(entry['global_index']):05d}.json"


def record_path_for_anchor(output_dir: Path, anchor: Mapping[str, Any]) -> Path:
    return output_dir / f"anchor_{int(anchor['anchor_index']):03d}.json"


def valid_state_record(path: Path, entry: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "PASS"
        and int(payload.get("global_index", -1)) == int(entry["global_index"])
        and payload.get("state_key") == entry["state_key"]
        and route_contract_passes(payload)
        and set((payload.get("actions") or {}).keys()) == set(ROUTES)
    )


def valid_anchor_record(path: Path, anchor: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    target = anchor["target"]
    return (
        payload.get("status") == "PASS"
        and int(payload.get("anchor_index", -1)) == int(anchor["anchor_index"])
        and payload.get("state_key") == target["state_key"]
        and route_contract_passes(payload)
        and set((payload.get("actions") or {}).keys()) == {"F1", "P1", "PV0", "PV0_shuffled"}
    )

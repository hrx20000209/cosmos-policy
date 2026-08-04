#!/usr/bin/env python3
"""Resumable single-config Cosmos manifest worker."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from adapters.cosmos_adapter import CosmosAdapter
from experiments.libero_harness import EpisodeVideoWriter, extract_observation, load_yaml, run_episode
from runtime.runtime_metrics import JsonlWriter, ResourceSampler


class LockedJsonlWriter:
    """One-write append with an advisory cross-process lock."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, value: Any) -> None:
        line = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ContextTraceWriter:
    def __init__(self, writer: LockedJsonlWriter):
        self.writer = writer
        self.context: dict[str, Any] = {}

    def write(self, value: dict[str, Any]) -> None:
        record = dict(value)
        extra = record.get("extra") or {}
        for key in (
            "action_chunk_sha256",
            "action_chunk_min",
            "action_chunk_max",
            "action_chunk_mean",
            "action_chunk_std",
            "action_chunk_l2_norm",
            "action_chunk_nan_count",
            "action_chunk_inf_count",
            "action_chunk_saturation_ratio",
        ):
            if key in extra:
                record[key] = extra[key]
        # Canonical aliases requested by the experiment schema.  The original
        # timing fields remain intact for backward compatibility.
        record["chunk_index"] = int(record.get("control_step_id", 0)) // 16
        record["action_ready_timestamp_ns"] = record.get("inference_finish_ns")
        record["policy_total_ms"] = record.get("total_policy_request_latency_ms")
        record["DiT_total_ms"] = record.get("dit_denoising_latency_ms")
        record["preprocess_ms"] = record.get("preprocessing_latency_ms")
        record["action_decode_postprocess_ms"] = record.get("action_extraction_latency_ms")
        record["future_decode_ms"] = record.get("future_state_decode_latency_ms")
        record.update(self.context)
        self.writer.write(record)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def completed_keys(raw_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in raw_dir.glob("episodes.shard*.jsonl"):
        for row in load_jsonl(path):
            # A recorded failure is complete too: retry requires --retry-failures.
            if row.get("episode_key"):
                keys.add(row["episode_key"])
    return keys


def failed_keys(raw_dir: Path) -> set[str]:
    keys: set[str] = set()
    for path in raw_dir.glob("episodes.shard*.jsonl"):
        for row in load_jsonl(path):
            if row.get("episode_key") and row.get("termination_reason", "").startswith(
                ("error:", "fatal:", "validation_error:")
            ):
                keys.add(row["episode_key"])
    return keys


def install_libero_checkout(repo: Path, config_path: Path) -> None:
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path)
    for name in [key for key in sys.modules if key == "libero" or key.startswith("libero.")]:
        del sys.modules[name]
    outer = types.ModuleType("libero")
    outer.__path__ = [str(repo / "libero")]
    outer.__package__ = "libero"
    sys.modules["libero"] = outer


class ManifestLiberoEnvironment:
    def __init__(self, row: dict[str, Any], resolution: int, screenshot_path: Path | None):
        from libero.libero.envs import OffScreenRenderEnv

        self.row = row
        self.screenshot_path = screenshot_path
        init_path = Path(row["init_path"])
        try:
            self.initial_states = torch.load(init_path, weights_only=False)
        except TypeError:
            self.initial_states = torch.load(init_path)
        if len(self.initial_states) <= int(row["init_state_index"]):
            raise IndexError(
                f"init index {row['init_state_index']} out of range for {init_path} "
                f"({len(self.initial_states)} states)"
            )
        self.env = OffScreenRenderEnv(
            bddl_file_name=row["bddl_path"],
            camera_heights=resolution,
            camera_widths=resolution,
        )
        self.env.seed(int(row["seed"]))

    def reset(self) -> dict[str, Any]:
        self.env.reset()
        raw = self.env.set_init_state(self.initial_states[int(self.row["init_state_index"])])
        self.reset_observation_metadata = {
            "keys": sorted(raw.keys()),
            "agentview_image_shape": list(np.asarray(raw["agentview_image"]).shape),
            "wrist_image_shape": list(np.asarray(raw["robot0_eye_in_hand_image"]).shape),
            "proprio_shape": [
                int(
                    len(raw["robot0_gripper_qpos"])
                    + len(raw["robot0_eef_pos"])
                    + len(raw["robot0_eef_quat"])
                )
            ],
        }
        if self.screenshot_path is not None:
            import imageio.v3 as iio

            self.screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            observation = extract_observation(raw, True)
            frame = np.concatenate([observation.primary_image, observation.wrist_image], axis=1)
            iio.imwrite(self.screenshot_path, frame)
        return raw

    def step(self, action: np.ndarray):
        return self.env.step(np.asarray(action).tolist())

    def close(self) -> None:
        self.env.close()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_for_row(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    model = dict(base["model"])
    model["repo"] = str(PROJECT)
    model["config_file"] = "cosmos_policy/config/config.py"
    model["generation_mode"] = "action_only_usage"
    model["camera_flip_vertical"] = True
    evaluation = base["evaluation"]
    return {
        "model": model,
        "evaluation": {
            "task_suite": row["suite"],
            "settle_steps": int(evaluation["settle_steps"]),
            "settle_gripper_action": float(evaluation["settle_gripper_action"]),
            "resolution": int(evaluation["resolution"]),
            "max_steps": int(row["max_steps"]),
            "video_fps": int(evaluation["video_fps"]),
        },
        "history": {
            "observation_policy": "latest_only",
            "history_policy": "dense_recent",
            "length": 1,
            "frame_stride": 1,
            "buffer_capacity": 64,
        },
        "denoising": {
            "scheduler": "fixed",
            "steps": int(row["denoising_steps"]),
            "official_steps": 5,
            "minimum_steps": 1,
            "normal_steps": 3,
        },
        "execution_prefix": {"mode": "fixed", "length": 16, "buffer_strategy": "replace"},
        "pipeline": {"mode": "sync_baseline", "decode_future": False, "cuda_streams": 1},
    }


def validate_request_counts(row: dict[str, Any], traces: list[dict[str, Any]]) -> str | None:
    expected = int(row["denoising_steps"])
    for trace in traces:
        actual = int(trace.get("denoiser_forward_count", -1))
        timings = trace.get("per_denoising_step_latency_ms", [])
        if actual != expected:
            return f"denoiser_forward_count expected {expected}, got {actual}"
        if len(timings) != expected:
            return f"per-step timings expected {expected}, got {len(timings)}"
        if int(trace.get("selected_denoising_steps", -1)) != expected:
            return f"selected_denoising_steps expected {expected}"
        if int(trace.get("vae_decode_count", -1)) != 0:
            return "vae_decode_count must be zero in action_only_usage"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "experiments/cosmos_denoising_libero_pro/configs/sweep.yaml",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-reset-screenshot", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")

    base = load_yaml(args.config)
    rows = load_jsonl(args.manifest)
    config_ids = {row["config_id"] for row in rows}
    if len(config_ids) != 1:
        raise SystemExit(
            f"one worker must contain exactly one independent config_id, got {sorted(config_ids)}"
        )
    config_id = next(iter(config_ids))
    rows = [
        row
        for row in rows
        if int(row["episode_key"], 16) % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    output_root = Path(base["output_root"])
    raw_dir = output_root / "raw" / config_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    episode_writer = LockedJsonlWriter(raw_dir / f"episodes.shard{args.shard_index:02d}.jsonl")
    trace_writer = ContextTraceWriter(
        LockedJsonlWriter(raw_dir / f"requests.shard{args.shard_index:02d}.jsonl")
    )
    done = completed_keys(raw_dir)
    if args.retry_failures:
        done -= failed_keys(raw_dir)
    if rows and all(row["episode_key"] in done for row in rows):
        print(
            f"all {len(rows)} assigned episodes already complete; skipping model load",
            flush=True,
        )
        return

    # Every independent config process has a single pinned environment domain.
    first = rows[0] if rows else None
    if first is None:
        print("no rows assigned to this shard")
        return
    install_libero_checkout(Path(first["libero_repo"]), Path(first["libero_config_path"]))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(195)
    np.random.seed(195)
    torch.manual_seed(195)
    torch.cuda.manual_seed_all(195)

    adapter = CosmosAdapter(config_for_row(base, first)["model"])
    sampler = ResourceSampler(
        float(base["runtime"]["resource_interval_seconds"]),
        gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
    )
    sampler.start()
    started_ns = time.time_ns()
    counts = {"assigned": len(rows), "skipped": 0, "recorded": 0, "success": 0, "failed": 0}
    runtime_abort_reason: str | None = None
    try:
        for row in rows:
            if row["episode_key"] in done:
                counts["skipped"] += 1
                continue
            trace_writer.context = {
                key: row[key]
                for key in (
                    "episode_key",
                    "config_id",
                    "domain",
                    "perturbation_category",
                    "variant_id",
                    "suite",
                    "task_uid",
                    "seed",
                    "init_state_index",
                    "denoising_steps",
                )
            }
            trace_writer.context.update(
                {
                    "run_id": args.run_id,
                    "worker_pid": os.getpid(),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "physical_gpu_id": os.environ.get("EVAL_PHYSICAL_GPU"),
                }
            )
            run_token = f"{args.run_id}_{row['episode_key'][:16]}"
            action_path = output_root / "actions" / config_id / f"{run_token}.npy"
            video_path = (
                output_root / "videos" / config_id / f"{run_token}.mp4"
                if args.record_video
                else None
            )
            screenshot_path = (
                output_root / "videos" / config_id / f"{run_token}_reset.png"
                if args.record_reset_screenshot
                else None
            )
            env = None
            traces: list[dict[str, Any]] = []
            try:
                torch.cuda.reset_peak_memory_stats()
                env = ManifestLiberoEnvironment(
                    row, int(base["evaluation"]["resolution"]), screenshot_path
                )
                record, traces = run_episode(
                    adapter,
                    env,
                    config_for_row(base, row),
                    row["instruction"],
                    int(row["init_state_index"]),
                    int(row["seed"]),
                    trace_writer,
                    video_path=video_path,
                    action_trace_path=action_path,
                )
                validation_error = validate_request_counts(row, traces)
                if validation_error:
                    record["success"] = False
                    record["termination_reason"] = f"validation_error:{validation_error}"
            except Exception as error:
                if isinstance(error, torch.cuda.OutOfMemoryError) or (
                    "Already called Timer.start() once!" in str(error)
                ):
                    runtime_abort_reason = f"{type(error).__name__}:{error}"
                record = {
                    "success": False,
                    "environment_valid": False,
                    "episode_steps": 0,
                    "episode_wall_clock_time_s": 0.0,
                    "inference_count": len(traces),
                    "termination_reason": f"fatal:{type(error).__name__}:{error}",
                    "traceback": traceback.format_exc(),
                    "executed_actions_path": str(action_path),
                    "video_path": str(video_path) if video_path else None,
                }
            finally:
                if env is not None:
                    env.close()
            record.setdefault("environment_valid", True)
            record.update(row)
            record.update(
                {
                    "benchmark": row["domain"],
                    "base_task_id": row["global_task_index"],
                    "generated_task_id": row["task_uid"],
                    "perturbation_type": row["perturbation_category"],
                    "perturbation_variant_id": row["variant_id"],
                    "executed_actions": record.get("executed_actions_path"),
                    "policy_requests": len(traces),
                    "peak_GPU_memory_MB": torch.cuda.max_memory_allocated() / 2**20,
                    "checkpoint_sha256": base["model"]["checkpoint_sha256"],
                    "config_sha256": sha256(args.config),
                    "run_id": args.run_id,
                    "worker_pid": os.getpid(),
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "physical_gpu_id": os.environ.get("EVAL_PHYSICAL_GPU"),
                    "completed_at_ns": time.time_ns(),
                    "request_count": len(traces),
                    "request_forward_count_sum": sum(
                        int(trace.get("denoiser_forward_count", 0)) for trace in traces
                    ),
                    "vae_encode_count": sum(
                        int(trace.get("vae_encode_count", 0)) for trace in traces
                    ),
                    "vae_decode_count": sum(
                        int(trace.get("vae_decode_count", 0)) for trace in traces
                    ),
                    "total_dit_denoising_time_ms": sum(
                        float(trace.get("dit_denoising_latency_ms", 0.0)) for trace in traces
                    ),
                    "total_policy_time_ms": sum(
                        float(trace.get("total_policy_request_latency_ms", 0.0))
                        for trace in traces
                    ),
                    "mean_policy_latency_ms": (
                        float(
                            np.mean(
                                [
                                    float(trace.get("total_policy_request_latency_ms", 0.0))
                                    for trace in traces
                                ]
                            )
                        )
                        if traces
                        else None
                    ),
                    "actions_per_policy_request": (
                        float(record.get("episode_steps", 0) / len(traces)) if traces else None
                    ),
                    "policy_requests_per_second": (
                        float(len(traces) / record["episode_wall_clock_time_s"])
                        if record.get("episode_wall_clock_time_s", 0) > 0
                        else None
                    ),
                    "action_steps_per_second": (
                        float(record.get("episode_steps", 0) / record["episode_wall_clock_time_s"])
                        if record.get("episode_wall_clock_time_s", 0) > 0
                        else None
                    ),
                    "torch_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                    "torch_peak_memory_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                    "reset_screenshot_path": str(screenshot_path) if screenshot_path else None,
                }
            )
            if action_path.is_file():
                record["executed_actions_sha256_file"] = sha256(action_path)
            if env is not None:
                record["reset_observation_metadata"] = getattr(
                    env, "reset_observation_metadata", None
                )
            record["joint_latent_generation_enabled"] = True
            record["future_rgb_decode_enabled"] = False
            episode_writer.write(record)
            counts["recorded"] += 1
            counts["success" if record["success"] else "failed"] += 1
            print(
                f"[{counts['recorded']}/{len(rows) - counts['skipped']}] "
                f"{row['episode_key'][:8]} success={record['success']} "
                f"reason={record['termination_reason']}",
                flush=True,
            )
            if runtime_abort_reason is not None:
                print(
                    "aborting worker after unrecoverable CUDA/model timer state: "
                    f"{runtime_abort_reason}",
                    flush=True,
                )
                break
    finally:
        parameter_count = (
            sum(parameter.numel() for parameter in adapter.model.parameters())
            if adapter.model is not None
            else None
        )
        adapter.close()
        resources = sampler.stop()
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "config_id": config_id,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "counts": counts,
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "parameter_count": parameter_count,
        "resources": resources,
        "status": "aborted_corrupt_runtime" if runtime_abort_reason else "completed",
        "runtime_abort_reason": runtime_abort_reason,
    }
    summary_path = output_root / "summaries" / config_id / f"{args.run_id}.shard{args.shard_index:02d}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if runtime_abort_reason is not None:
        raise SystemExit(75)


if __name__ == "__main__":
    main()

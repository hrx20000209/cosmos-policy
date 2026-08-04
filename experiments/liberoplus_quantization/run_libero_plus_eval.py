#!/usr/bin/env python3
"""Unified, instrumented Cosmos Policy evaluation on LIBERO-Plus."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

EXP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = EXP_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import execution_metrics as metrics
import exp_common as ec
import quant_lib
from experiment_config import EvalConfig, load_config
from liberoplus_utils import (
    array_sha256,
    load_embedding_keys,
    load_task_list,
    resolve_instruction,
    task_list_sha256,
)

sys.path.insert(0, str(EXP_DIR))
from collect_hardware_metrics import PowerSampler, collect_hardware_info


@dataclass(eq=False)
class CosmosRuntimeConfig:
    suite: str = "libero"
    model_family: str = "cosmos"
    config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    trained_with_image_aug: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    deterministic: bool = True
    randomize_seed: bool = False
    seed: int = 195
    task_suite_name: str = "libero_10"
    inference_precision: str = "bf16"


class JsonlAppender:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", buffering=1)

    def write(self, row: dict[str, Any]) -> None:
        self.stream.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def _completed(path: Path) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    if not path.exists():
        return keys
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
                if row.get("record_status") == "completed":
                    keys.add(
                        (
                            row["config_hash"],
                            row["task_suite"],
                            str(row["task_id"]),
                            str(row["seed"]),
                            str(row["episode_index"]),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def _runtime_config(cfg: EvalConfig, seed: int) -> CosmosRuntimeConfig:
    return CosmosRuntimeConfig(
        ckpt_path=cfg.checkpoint,
        use_wrist_image=cfg.use_wrist_image,
        use_proprio=cfg.use_proprio,
        flip_images=cfg.flip_images,
        use_jpeg_compression=cfg.use_jpeg_compression,
        num_denoising_steps_action=cfg.denoising_steps,
        unnormalize_actions=cfg.unnormalize_actions,
        normalize_proprio=cfg.normalize_proprio,
        chunk_size=cfg.chunk_size,
        num_open_loop_steps=cfg.num_open_loop_steps,
        deterministic=cfg.deterministic,
        seed=seed,
        inference_precision="fp16" if cfg.precision in ("fp16", "float16") else "bf16",
    )


def _select_dynamic_parameters(
    cfg: EvalConfig,
    visual_delta: float,
    recent_actions: list[np.ndarray],
    no_progress_steps: int,
) -> tuple[int, int, str]:
    dynamic = cfg.dynamic_replan
    magnitude = float(np.linalg.norm(recent_actions[-1], ord=2)) if recent_actions else 0.0
    jerk = metrics.action_jerk(recent_actions)
    high_risk = (
        visual_delta >= dynamic.visual_change_high
        or magnitude >= dynamic.action_magnitude_high
        or jerk >= dynamic.action_jerk_high
        or no_progress_steps >= dynamic.no_progress_steps
    )
    medium_risk = visual_delta >= dynamic.visual_change_low
    candidates = sorted(dynamic.candidate_open_loop_steps)
    if high_risk:
        open_loop, risk = candidates[0], "high"
    elif medium_risk and len(candidates) > 1:
        open_loop, risk = candidates[len(candidates) // 2], "medium"
    else:
        open_loop, risk = candidates[-1], "low"
    denoise = cfg.denoising_steps
    if dynamic.dynamic_denoising:
        denoise = (
            dynamic.denoising_steps_high_risk
            if high_risk
            else dynamic.denoising_steps_low_risk
        )
    return open_loop, denoise, risk


def _classify_failure(error: Exception | None, success: bool, timed_out: bool) -> str:
    if success:
        return "success"
    if error is not None:
        message = str(error).lower()
        if "t5 embedding" in message:
            return "evaluation_error_missing_t5"
        if "nan" in message or "non-finite" in message:
            return "evaluation_error_invalid_action"
        return f"environment_or_evaluation_error:{type(error).__name__}"
    if timed_out:
        return "timeout"
    return "task_failure_unclassified"


def _process_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def warm_up_policy(
    cfg: EvalConfig,
    runtime: CosmosRuntimeConfig,
    env: Any,
    task_description: str,
    initial_state: np.ndarray,
    model: Any,
    dataset_stats: dict,
    resize_size: int,
) -> None:
    if cfg.warmup_policy_calls <= 0:
        return
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action
    from cosmos_policy.experiments.robot.libero.run_libero_eval import prepare_observation
    from cosmos_policy.utils.utils import set_seed_everywhere

    set_seed_everywhere(runtime.seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action(runtime.model_family))
    observation = prepare_observation(obs, resize_size, cfg.flip_images)
    for _ in range(cfg.warmup_policy_calls):
        get_action(
            runtime,
            model,
            dataset_stats,
            observation,
            task_description,
            seed=runtime.seed,
            randomize_seed=False,
            num_denoising_steps_action=cfg.denoising_steps,
            generate_future_state_and_value_in_parallel=False,
            batch_size=1,
        )
    torch.cuda.synchronize()


def run_episode(
    cfg: EvalConfig,
    runtime: CosmosRuntimeConfig,
    env: Any,
    task_description: str,
    initial_state: np.ndarray,
    model: Any,
    dataset_stats: dict,
    resize_size: int,
    task_meta: dict[str, Any],
    episode_index: int,
    episode_writer: JsonlAppender,
    chunk_writer: JsonlAppender,
    step_writer: JsonlAppender,
) -> dict[str, Any]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_dummy_action
    from cosmos_policy.experiments.robot.libero.run_libero_eval import (
        TASK_MAX_STEPS,
        prepare_observation,
    )
    from cosmos_policy.utils.utils import set_seed_everywhere

    seed = runtime.seed
    set_seed_everywhere(seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, done, _ = env.step(get_libero_dummy_action(runtime.model_family))
        if done:
            break

    max_steps = cfg.max_environment_steps or TASK_MAX_STEPS[task_meta["suite"]]
    action_queue: deque[tuple[np.ndarray, metrics.ChunkMetrics]] = deque()
    chunks: list[metrics.ChunkMetrics] = []
    recent_actions: list[np.ndarray] = []
    env_latencies: list[float] = []
    policy_wall_latencies: list[float] = []
    policy_gpu_latencies: list[float] = []
    denoise_latencies: list[float] = []
    previous_image: np.ndarray | None = None
    previous_executed_action: np.ndarray | None = None
    previous_plan_time: float | None = None
    replanning_intervals_ms: list[float] = []
    no_progress_steps = 0
    success = False
    timed_out = False
    error: Exception | None = None
    error_traceback: str | None = None
    control_step = 0
    early_replans = 0
    policy_calls = 0
    inference_total_ms = 0.0
    env_total_ms = 0.0
    episode_start = time.perf_counter()
    power = PowerSampler(device_index=0)
    if cfg.collect_power:
        power.start()
    torch.cuda.reset_peak_memory_stats()

    inner_denoise = model.generate_samples_from_batch
    denoise_timer = {"ms": 0.0}

    def timed_denoise(*args: Any, **kwargs: Any) -> Any:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = inner_denoise(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        denoise_timer["ms"] = start.elapsed_time(end)
        return output

    model.generate_samples_from_batch = timed_denoise
    try:
        while control_step < max_steps:
            now = time.perf_counter()
            if now - episode_start >= cfg.episode_timeout_seconds:
                timed_out = True
                break
            observation_start = time.perf_counter()
            observation = prepare_observation(obs, resize_size, cfg.flip_images)
            observation_prepare_ms = (time.perf_counter() - observation_start) * 1000.0
            delta = metrics.visual_change(previous_image, observation["primary_image"])
            if previous_image is not None and delta < cfg.dynamic_replan.visual_change_low:
                no_progress_steps += 1
            else:
                no_progress_steps = 0

            if cfg.dynamic_replan.enabled and action_queue:
                dynamic = cfg.dynamic_replan
                chunk_executed = action_queue[0][1].executed_actions
                should_replan = chunk_executed >= dynamic.min_open_loop_steps and (
                    delta >= dynamic.visual_change_high
                    or metrics.action_jerk(recent_actions) >= dynamic.action_jerk_high
                    or no_progress_steps >= dynamic.no_progress_steps
                )
                if should_replan:
                    action_queue.clear()
                    early_replans += 1

            if not action_queue:
                if cfg.dynamic_replan.enabled:
                    open_loop, denoise_steps, risk = _select_dynamic_parameters(
                        cfg, delta, recent_actions, no_progress_steps
                    )
                else:
                    open_loop, denoise_steps, risk = (
                        cfg.num_open_loop_steps,
                        cfg.denoising_steps,
                        "fixed",
                    )
                runtime.num_denoising_steps_action = denoise_steps
                runtime._inference_metrics_sink = {}
                denoise_timer["ms"] = 0.0
                torch.cuda.synchronize()
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                gpu_start.record()
                action_result = get_action(
                    runtime,
                    model,
                    dataset_stats,
                    observation,
                    task_description,
                    seed=seed,
                    randomize_seed=False,
                    num_denoising_steps_action=denoise_steps,
                    generate_future_state_and_value_in_parallel=False,
                    batch_size=1,
                )
                gpu_end.record()
                torch.cuda.synchronize()
                plan_time = time.perf_counter()
                wall_ms = (plan_time - wall_start) * 1000.0
                gpu_ms = gpu_start.elapsed_time(gpu_end)
                actions = np.asarray(action_result["actions"], dtype=np.float64)
                if actions.shape != (cfg.chunk_size, 7):
                    raise ValueError(
                        f"expected action chunk {(cfg.chunk_size, 7)}, got {actions.shape}"
                    )
                if not np.isfinite(actions).all():
                    raise ValueError("non-finite action generated")
                if np.allclose(actions, 0.0):
                    raise ValueError("all-zero action chunk generated")
                chunk = metrics.ChunkMetrics(
                    chunk_index=len(chunks),
                    generated_step=control_step,
                    generated_time_s=plan_time,
                    generated_actions=len(actions),
                    planned_open_loop_steps=open_loop,
                    denoising_steps=denoise_steps,
                )
                if previous_executed_action is not None:
                    metrics.set_boundary_metrics(chunk, previous_executed_action, actions[0])
                chunks.append(chunk)
                for action in actions[:open_loop]:
                    action_queue.append((action, chunk))
                if previous_plan_time is not None:
                    replanning_intervals_ms.append((plan_time - previous_plan_time) * 1000.0)
                previous_plan_time = plan_time
                policy_calls += 1
                inference_total_ms += wall_ms
                policy_wall_latencies.append(wall_ms)
                policy_gpu_latencies.append(gpu_ms)
                denoise_latencies.append(denoise_timer["ms"])
                step_writer.write(
                    {
                        "record_type": "policy_call",
                        "config_hash": cfg.config_hash,
                        "task_suite": task_meta["suite"],
                        "task_id": task_meta["id"],
                        "seed": seed,
                        "episode_index": episode_index,
                        "policy_call_index": policy_calls - 1,
                        "control_step": control_step,
                        "risk": risk,
                        "open_loop_steps": open_loop,
                        "denoising_steps": denoise_steps,
                        "policy_wall_ms": wall_ms,
                        "policy_gpu_ms": gpu_ms,
                        "denoising_gpu_ms": denoise_timer["ms"],
                        "observation_prepare_ms": observation_prepare_ms,
                        "preprocess_and_h2d_ms": runtime._inference_metrics_sink.get(
                            "preprocess_and_h2d_ms"
                        ),
                        "generation_wall_ms": runtime._inference_metrics_sink.get(
                            "generation_wall_ms"
                        ),
                        "postprocess_ms": runtime._inference_metrics_sink.get(
                            "postprocess_ms"
                        ),
                        "gpu_allocated_mb": torch.cuda.memory_allocated() / 1e6,
                    }
                )

            action, chunk = action_queue.popleft()
            execute_time = time.perf_counter()
            chunk.execute(control_step, execute_time)
            env_start = time.perf_counter()
            obs, reward, done, info = env.step(action.tolist())
            env_ms = (time.perf_counter() - env_start) * 1000.0
            env_total_ms += env_ms
            env_latencies.append(env_ms)
            recent_actions.append(action.copy())
            recent_actions = recent_actions[-3:]
            previous_executed_action = action.copy()
            previous_image = observation["primary_image"].copy()
            step_writer.write(
                {
                    "record_type": "environment_step",
                    "config_hash": cfg.config_hash,
                    "task_suite": task_meta["suite"],
                    "task_id": task_meta["id"],
                    "seed": seed,
                    "episode_index": episode_index,
                    "control_step": control_step,
                    "chunk_index": chunk.chunk_index,
                    "action_index_in_chunk": chunk.executed_actions - 1,
                    "observation_staleness_steps": chunk.staleness_steps[-1],
                    "observation_staleness_ms": chunk.staleness_ms[-1],
                    "environment_step_ms": env_ms,
                    "reward": float(reward),
                    "done": bool(done),
                }
            )
            control_step += 1
            if done:
                success = True
                break
    except Exception as caught:
        error = caught
        error_traceback = traceback.format_exc()
    finally:
        model.generate_samples_from_batch = inner_denoise

    chunk_summary = metrics.summarize_chunks(chunks)
    for chunk in chunks:
        chunk_writer.write(
            {
                "record_type": "action_chunk",
                "config_hash": cfg.config_hash,
                "task_suite": task_meta["suite"],
                "task_id": task_meta["id"],
                "seed": seed,
                "episode_index": episode_index,
                **chunk.as_dict(),
            }
        )
    episode_ms = (time.perf_counter() - episode_start) * 1000.0
    power_metrics = power.stop() if cfg.collect_power else {
        "power_measurement_supported": False,
        "measurement_note": "disabled by configuration",
    }
    failure_category = _classify_failure(error, success, timed_out)
    row = {
        "record_type": "episode",
        "record_status": "completed",
        "experiment_name": cfg.experiment_name,
        "config_hash": cfg.config_hash,
        "task_suite": task_meta["suite"],
        "task_id": task_meta["id"],
        "task_name": task_meta["name"],
        "perturbation_category": task_meta.get("category"),
        "difficulty": task_meta.get("difficulty_level"),
        "seed": seed,
        "episode_index": episode_index,
        "initial_state_index": task_meta["initial_state_index"],
        "initial_state_sha256": array_sha256(initial_state),
        "policy_instruction": task_description,
        "instruction_resolution": task_meta["instruction_resolution"],
        "success": success,
        "termination_reason": failure_category,
        "error": repr(error) if error else None,
        "traceback": error_traceback,
        "environment_steps": control_step,
        "policy_calls": policy_calls,
        "generated_action_chunks": len(chunks),
        "early_replans": early_replans,
        "episode_total_ms": episode_ms,
        "policy_inference_total_ms": inference_total_ms,
        "environment_total_ms": env_total_ms,
        "replanning_overhead_ratio": inference_total_ms / max(episode_ms, 1e-9),
        "mean_time_between_replanning_ms": (
            float(np.mean(replanning_intervals_ms)) if replanning_intervals_ms else None
        ),
        "policy_latency_ms": ec.latency_summary(policy_wall_latencies),
        "policy_gpu_latency_ms": ec.latency_summary(policy_gpu_latencies),
        "denoising_latency_ms": ec.latency_summary(denoise_latencies),
        "environment_step_latency_ms": ec.latency_summary(env_latencies),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_process_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "process_rss_mb_end": _process_rss_mb(),
        **chunk_summary,
        **power_metrics,
    }
    episode_writer.write(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", default="0/1", help="task shard i/N")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    shard_index, shard_count = (int(item) for item in args.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard must satisfy 0 <= i < N")

    cfg = load_config(args.config)
    tasks = load_task_list(cfg.task_list, cfg.task_names)
    tasks = [task for index, task in enumerate(tasks) if index % shard_count == shard_index]
    if args.limit:
        tasks = tasks[: args.limit]
    embedding_keys = load_embedding_keys(cfg.t5_embeddings_path)

    # Validate all instructions before loading the 2B model. This avoids a
    # silent on-demand UMT5-11B load after expensive initialization.
    from libero.libero import benchmark

    benchmark_cache: dict[str, Any] = {}
    resolved_tasks: list[dict[str, Any]] = []
    for task_meta in tasks:
        suite = task_meta["suite"]
        benchmark_cache.setdefault(suite, benchmark.get_benchmark_dict()[suite]())
        task = benchmark_cache[suite].get_task(task_meta["id"] - 1)
        try:
            instruction, method = resolve_instruction(
                task.language, task_meta.get("category", ""), embedding_keys
            )
        except KeyError:
            if cfg.fail_on_missing_t5:
                raise
            continue
        resolved_tasks.append(
            {
                **task_meta,
                "task_object": task,
                "policy_instruction": instruction,
                "instruction_resolution": method,
                "initial_state_index": task_meta.get(
                    "initial_state_index", cfg.initial_state_index
                ),
            }
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "config": cfg.as_dict(),
                    "config_hash": cfg.config_hash,
                    "task_list_sha256": task_list_sha256(cfg.task_list),
                    "resolved_tasks": [
                        {key: value for key, value in task.items() if key != "task_object"}
                        for task in resolved_tasks
                    ],
                },
                indent=2,
            )
        )
        return

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
    from cosmos_policy.experiments.robot.robot_utils import get_image_resize_size

    runtime = _runtime_config(cfg, cfg.seeds[0])
    torch.cuda.synchronize()
    load_start = time.perf_counter()
    model, cosmos_config = get_model(runtime)
    torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - load_start) * 1000.0
    if cfg.chunk_size != cosmos_config.dataloader_train.dataset.chunk_size:
        raise ValueError("evaluation chunk_size does not match checkpoint training config")
    model.eval()
    quant_lib.configure_model_precision(model, runtime.inference_precision)
    quant_audit = quant_lib.apply_quantization(
        model.net,
        cfg.quantization_mode,
        group_size=cfg.group_size,
        scope=cfg.quantization_scope,
        include_patterns=cfg.quantized_modules,
        exclude_patterns=cfg.excluded_modules,
    )
    quant_audit.update(quant_lib.verify_quantized(model.net))
    if cfg.real_quantization_required and not quant_audit["real_quantization"]:
        raise RuntimeError("configuration requires real quantization, but backend is not real")
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_embeddings_path, worker_id=0)
    resize_size = get_image_resize_size(runtime.model_family)

    suffix = f"{cfg.output_tag}_shard{shard_index}_of_{shard_count}"
    raw_dir = Path(ec.RAW)
    episode_path = raw_dir / f"episodes_{suffix}.jsonl"
    episode_writer = JsonlAppender(episode_path)
    chunk_writer = JsonlAppender(raw_dir / f"chunks_{suffix}.jsonl")
    step_writer = JsonlAppender(raw_dir / f"steps_{suffix}.jsonl")
    # A resumed run may use a different GPU/shard count. Read completion keys
    # from every prior shard for this output tag so finished episodes are not
    # rerun or double-counted when moving the sweep to another GPU set.
    done: set[tuple[str, ...]] = set()
    if cfg.skip_completed:
        for prior_episode_path in raw_dir.glob(
            f"episodes_{cfg.output_tag}_shard*_of_*.jsonl"
        ):
            done.update(_completed(prior_episode_path))
    run_metadata = {
        "experiment_name": cfg.experiment_name,
        "config": cfg.as_dict(),
        "config_hash": cfg.config_hash,
        "git_commit": ec.git_commit(),
        "task_list_sha256": task_list_sha256(cfg.task_list),
        "model_load_ms": model_load_ms,
        "checkpoint_file_size_bytes": Path(cfg.checkpoint).stat().st_size,
        "quantization": quant_audit,
        "hardware": collect_hardware_info(),
    }
    metadata_path = Path(ec.PROF) / f"run_{suffix}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(run_metadata, indent=2, default=_json_default) + "\n")

    try:
        warmed_up = False
        for task_meta in resolved_tasks:
            suite = task_meta["suite"]
            task_id = task_meta["id"] - 1
            task_suite = benchmark_cache[suite]
            initial_states = task_suite.get_task_init_states(task_id)
            env, _ = get_libero_env(task_meta["task_object"], "cosmos", resolution=256)
            try:
                base_state_index = task_meta["initial_state_index"]
                for seed in cfg.seeds:
                    runtime.seed = seed
                    runtime.task_suite_name = suite
                    for episode_index in range(cfg.episodes_per_task):
                        state_index = base_state_index + episode_index
                        key = (
                            cfg.config_hash,
                            suite,
                            str(task_meta["id"]),
                            str(seed),
                            str(episode_index),
                        )
                        if key in done:
                            continue
                        if state_index >= len(initial_states):
                            raise IndexError(
                                f"initial state {state_index} unavailable for {suite}:{task_meta['id']}"
                            )
                        if not warmed_up:
                            warm_up_policy(
                                cfg,
                                runtime,
                                env,
                                task_meta["policy_instruction"],
                                initial_states[state_index],
                                model,
                                dataset_stats,
                                resize_size,
                            )
                            warmed_up = True
                        task_meta["initial_state_index"] = state_index
                        row = run_episode(
                            cfg,
                            runtime,
                            env,
                            task_meta["policy_instruction"],
                            initial_states[state_index],
                            model,
                            dataset_stats,
                            resize_size,
                            task_meta,
                            episode_index,
                            episode_writer,
                            chunk_writer,
                            step_writer,
                        )
                        print(
                            f"{suite}:{task_meta['id']} seed={seed} episode={episode_index} "
                            f"success={int(row['success'])} steps={row['environment_steps']} "
                            f"calls={row['policy_calls']}"
                        )
            finally:
                env.close()
    finally:
        episode_writer.close()
        chunk_writer.close()
        step_writer.close()


if __name__ == "__main__":
    main()

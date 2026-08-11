from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from adapters.cosmos_adapter import CosmosAdapter
from adapters.lingbot_adapter import LingBotAdapter
from adapters.mock_adapter import MockAdapter
from runtime.action_buffer import ActionBuffer
from runtime.async_pipeline import ActionFirstPipeline, InferenceRequest
from runtime.denoising_scheduler import FixedScheduler, HeuristicScheduler, RandomMatchedScheduler, RuntimeState
from runtime.keyframe_selector import KeyframeSelector, pixel_change, proprio_change
from runtime.observation_buffer import Observation, ObservationBuffer
from runtime.runtime_metrics import (
    InferenceTrace,
    JsonlWriter,
    ResourceSampler,
    monotonic_ns,
    summarize_traces,
    write_json,
)

LOGGER = logging.getLogger("wam_libero")
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


class EpisodeVideoWriter:
    """Stream the exact post-flip model views to a side-by-side MP4."""

    def __init__(self, path: str | Path, fps: int):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            self.path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

    def append(self, observation: Observation) -> None:
        frame = np.concatenate([observation.primary_image, observation.wrist_image], axis=1)
        self.writer.append_data(np.ascontiguousarray(frame))

    def close(self) -> None:
        self.writer.close()


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    import yaml

    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"override must be key=value: {expression}")
        path, raw = expression.split("=", 1)
        target = config
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(raw)
    return config


def configure_repository_paths(config: dict[str, Any]) -> None:
    """Prefer the exact repositories recorded in the experiment config.

    This matters on machines that also have an editable LIBERO-plus install:
    importing that package can silently change environment code while still
    reading the same assets from ``~/.libero/config.yaml``.
    """

    repositories = config.get("repositories", {})
    for name in ("libero", "lingbot_va", "cosmos"):
        raw_path = repositories.get(name)
        if not raw_path:
            continue
        path = str(Path(raw_path).expanduser().resolve())
        if not Path(path).is_dir():
            raise FileNotFoundError(f"configured {name} repository does not exist: {path}")
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


class MockLiberoEnvironment:
    def __init__(self, seed: int, success_step: int = 8):
        self.rng = np.random.default_rng(seed)
        self.success_step = success_step
        self.steps = 0

    def reset(self, initial_state: Any = None) -> dict[str, Any]:
        self.steps = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.steps += 1
        done = self.steps >= self.success_step
        return self._obs(), float(done), done, {}

    def _obs(self) -> dict[str, Any]:
        value = np.uint8(min(self.steps * 8, 255))
        image = np.full((128, 128, 3), value, dtype=np.uint8)
        return {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image.copy(),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        }

    def close(self) -> None:
        pass


class RealLiberoEnvironment:
    def __init__(self, task_suite_name: str, task_id: int, resolution: int):
        import torch
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[task_suite_name]()
        self.task = suite.get_task(task_id)
        init_states_path = (
            Path(get_libero_path("init_states"))
            / self.task.problem_folder
            / self.task.init_states_file
        )
        if not init_states_path.is_file():
            raise FileNotFoundError(f"LIBERO init-state file is missing: {init_states_path}")
        # Official LIBERO init states contain NumPy objects. PyTorch 2.6 changed
        # torch.load's default to weights_only=True, which rejects these trusted
        # local benchmark files. Keep the unsafe mode scoped to this exact path.
        try:
            self.initial_states = torch.load(init_states_path, weights_only=False)
        except TypeError:
            self.initial_states = torch.load(init_states_path)
        self.description = self.task.language
        self.env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(task_id),
            camera_heights=resolution,
            camera_widths=resolution,
        )
        self.env.seed(0)

    def reset(self, episode_index: int) -> dict[str, Any]:
        self.env.reset()
        return self.env.set_init_state(self.initial_states[episode_index % len(self.initial_states)])

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return self.env.step(action.tolist())

    def close(self) -> None:
        self.env.close()


def extract_observation(raw: dict[str, Any], flip_vertical: bool) -> Observation:
    primary = np.asarray(raw["agentview_image"])
    wrist = np.asarray(raw["robot0_eye_in_hand_image"])
    if flip_vertical:
        primary = np.ascontiguousarray(primary[::-1])
        wrist = np.ascontiguousarray(wrist[::-1])
    proprio = np.concatenate(
        [raw["robot0_gripper_qpos"], raw["robot0_eef_pos"], raw["robot0_eef_quat"]]
    ).astype(np.float32)
    return Observation(monotonic_ns(), primary, wrist, proprio)


def build_adapter(model: str, config: dict[str, Any], mock: bool):
    adapter_config = config["model"]
    if mock:
        return MockAdapter(adapter_config)
    if model == "cosmos":
        return CosmosAdapter(adapter_config)
    if model == "lingbot_va":
        return LingBotAdapter(adapter_config)
    raise ValueError(f"unsupported model: {model}")


def build_scheduler(config: dict[str, Any], seed: int):
    denoising = config["denoising"]
    mode = denoising.get("scheduler", "fixed")
    if mode == "fixed":
        return FixedScheduler(int(denoising["steps"]))
    if mode == "heuristic":
        return HeuristicScheduler(
            int(denoising["minimum_steps"]),
            int(denoising["normal_steps"]),
            int(denoising["official_steps"]),
        )
    if mode == "random_matched":
        return RandomMatchedScheduler(list(map(int, denoising["choices"])), denoising.get("weights"), seed)
    raise ValueError(f"unknown denoising scheduler: {mode}")


def choose_prefix(config: dict[str, Any], output_actions: np.ndarray, state: RuntimeState) -> int:
    prefix = config["execution_prefix"]
    horizon = len(output_actions)
    mode = prefix.get("mode", "fixed")
    if mode == "fixed":
        return min(horizon, int(prefix.get("length", horizon)))
    candidates = sorted(set(min(horizon, int(value)) for value in prefix.get("candidates", [1, 2, 4, 8, horizon])))
    if mode == "proprio_change":
        score = state.proprio_change_score
    elif mode == "visual_change":
        score = state.image_change_score
    elif mode == "action_variance":
        score = float(np.mean(np.var(output_actions, axis=0)))
    elif mode == "task_stage":
        score = 1.0 if state.gripper_transition else state.image_change_score
    else:
        raise ValueError(f"unknown prefix mode: {mode}")
    if score > float(prefix.get("short_threshold", 0.05)):
        return candidates[0]
    if score > float(prefix.get("medium_threshold", 0.02)):
        return candidates[len(candidates) // 2]
    return candidates[-1]


def _trace_from_output(trace: InferenceTrace, output: Any) -> None:
    metrics = output.stage_metrics_ms
    trace.preprocessing_latency_ms = float(
        metrics.get(
            "preprocessing_ms",
            float(metrics.get("camera_preprocessing_ms", 0.0))
            + float(metrics.get("latent_assembly_h2d_ms", 0.0)),
        )
    )
    trace.vae_encoding_latency_ms = float(metrics.get("vae_encoding_ms", 0.0))
    trace.conditioning_latency_ms = float(metrics.get("conditioning_ms", 0.0))
    trace.per_denoising_step_latency_ms = list(
        map(float, metrics.get("per_denoising_step_latency_ms", []))
    )
    trace.dit_denoising_latency_ms = float(metrics.get("dit_denoising_ms", 0.0))
    trace.action_extraction_latency_ms = float(metrics.get("action_extraction_ms", 0.0))
    trace.denoiser_forward_count = int(output.denoiser_forward_count)
    trace.vae_encode_count = int(trace.vae_encoding_latency_ms > 0)
    trace.extra.update(
        {
            key: output.extra[key]
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
            )
            if key in output.extra
        }
    )
    trace.extra["non_overlapping_stage_ms"] = {
        key: float(metrics.get(key, 0.0))
        for key in (
            "camera_preprocessing_ms",
            "latent_assembly_h2d_ms",
            "vae_encoding_ms",
            "dit_denoising_ms",
            "generation_conditioning_overhead_ms",
            "action_extraction_ms",
            "postprocess_after_action_ms",
            "model_generate_inclusive_ms",
            "non_overlapping_stage_sum_ms",
            "unattributed_stage_ms",
        )
    }
    for key in (
        "visual_input_mode",
        "visual_source",
        "fresh_visual_request_count",
        "predicted_visual_request_count",
        "predict_correct_request_count",
        "native_persistent_request_count",
        "cached_visual_request_count",
        "rgb_preprocessing_count",
    ):
        if key in output.extra:
            trace.extra[key] = output.extra[key]


def run_episode(
    adapter: Any,
    env: Any,
    config: dict[str, Any],
    task_name: str,
    episode_index: int,
    seed: int,
    trace_writer: JsonlWriter,
    video_path: str | Path | None = None,
    action_trace_path: str | Path | None = None,
    inference_capture_callback: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode_id = f"{task_name}:{episode_index}:{seed}"
    adapter.reset(task_name, seed)
    raw = env.reset(episode_index) if isinstance(env, RealLiberoEnvironment) else env.reset()
    environment_time_ms = 0.0
    settle_steps = int(config["evaluation"].get("settle_steps", 10))
    settle_action = np.zeros(7, dtype=np.float32)
    settle_action[-1] = float(config["evaluation"].get("settle_gripper_action", 0.0))
    for _ in range(settle_steps):
        settle_start = monotonic_ns()
        raw, _, _, _ = env.step(settle_action)
        environment_time_ms += (monotonic_ns() - settle_start) / 1e6

    observations = ObservationBuffer(int(config["history"].get("buffer_capacity", 64)))
    selector = KeyframeSelector()
    actions = ActionBuffer()
    scheduler = build_scheduler(config, seed)
    pipeline = ActionFirstPipeline(config["pipeline"].get("mode", "sync_baseline"))
    traces: list[dict[str, Any]] = []
    pending_decodes: list[tuple[Any, dict[str, Any]]] = []
    executed_since_replan: list[np.ndarray] = []
    all_executed_actions: list[np.ndarray] = []
    context_since_replan: list[Observation] = []
    previous_observation: Observation | None = None
    previous_latency_ms = 0.0
    inference_count = 0
    all_policy_action_chunks: list[np.ndarray] = []
    episode_start = monotonic_ns()
    success = False
    termination_reason = "max_steps"
    max_steps = int(config["evaluation"].get("max_steps", MAX_STEPS.get(config["evaluation"]["task_suite"], 520)))
    control_step = 0
    no_progress_streak = 0
    maximum_no_progress_streak = 0
    video_writer = (
        EpisodeVideoWriter(video_path, int(config["evaluation"].get("video_fps", 20)))
        if video_path is not None
        else None
    )

    try:
        while control_step < max_steps:
            acquisition_start = monotonic_ns()
            current = extract_observation(raw, bool(config["model"].get("camera_flip_vertical", True)))
            if video_writer is not None:
                video_writer.append(current)
            acquisition_ms = (monotonic_ns() - acquisition_start) / 1e6
            observations.append(current)
            if config["model"].get("name") == "lingbot_va" and previous_observation is not None:
                capture_stride = int(config["history"].get("context_capture_stride_actions", 4))
                if control_step % max(1, capture_stride) == 0:
                    context_since_replan.append(current)

            if actions.occupancy == 0:
                if executed_since_replan:
                    adapter.update_context(context_since_replan, np.stack(executed_since_replan))
                    executed_since_replan.clear()
                    context_since_replan.clear()

                snapshot = observations.snapshot()
                observation_policy = config["history"].get("observation_policy", "latest_only")
                selected = selector.choose_current(snapshot, observation_policy, int(config["history"].get("frame_stride", 1)))
                history_length = int(config["history"].get("length", 1))
                history_policy = config["history"].get("history_policy", "dense_recent")
                selected_history, history_stats = selector.select_history(
                    snapshot,
                    history_length,
                    int(config["history"].get("frame_stride", 1)),
                    history_policy,
                )
                if config["model"].get("name") == "cosmos":
                    selected_history = [selected]
                    history_stats.history_span_seconds = 0.0
                    history_stats.average_frame_interval = 0.0
                    history_stats.maximum_frame_interval = 0.0

                image_score = pixel_change(previous_observation, selected) if previous_observation is not None else 0.0
                proprio_score = proprio_change(previous_observation, selected) if previous_observation is not None else 0.0
                state = RuntimeState(
                    image_change_score=image_score,
                    proprio_change_score=proprio_score,
                    remaining_action_buffer=0,
                    previous_inference_latency_ms=previous_latency_ms,
                    observation_age_ms=(monotonic_ns() - selected.timestamp_ns) / 1e6,
                )
                steps = scheduler.select_steps(state)
                request = InferenceRequest.create(selected.timestamp_ns, episode_id, control_step)
                inference_start = monotonic_ns()
                trace = InferenceTrace(
                    request.request_id,
                    episode_id,
                    control_step,
                    selected.timestamp_ns,
                    inference_start,
                    observation_age_at_inference_start_ms=(inference_start - selected.timestamp_ns) / 1e6,
                    image_acquisition_latency_ms=acquisition_ms,
                    selected_denoising_steps=steps,
                    history_length=len(selected_history),
                    history_span_seconds=history_stats.history_span_seconds,
                    average_frame_interval=history_stats.average_frame_interval,
                    maximum_frame_interval=history_stats.maximum_frame_interval,
                )
                result = pipeline.submit(
                    request,
                    lambda: adapter.infer(selected, request, steps, selected_history),
                    adapter.decode_future if config["pipeline"].get("decode_future", False) else None,
                )
                output = result.policy_output
                trace.finalize(result.action_ready_ns)
                _trace_from_output(trace, output)
                trace.future_state_decode_latency_ms = result.future_decode_latency_ms
                trace.vae_decode_count = int(result.future_decode_latency_ms > 0)
                previous_latency_ms = trace.total_policy_request_latency_ms
                state.gripper_transition = bool(np.any(np.diff(np.sign(output.actions[:, -1])) != 0))
                prefix_length = choose_prefix(config, output.actions, state)
                all_policy_action_chunks.append(
                    np.ascontiguousarray(output.actions, dtype=np.float32).copy()
                )
                trace.executed_prefix_length = prefix_length
                installed = actions.install(
                    request.request_id,
                    output.actions,
                    prefix_length=prefix_length,
                    strategy=config["execution_prefix"].get("buffer_strategy", "replace"),
                    latest_valid_request_id=request.request_id,
                )
                if not installed:
                    trace.stale = True
                if inference_capture_callback is not None:
                    inference_capture_callback(
                        observation=selected,
                        output=output,
                        trace=trace,
                        request=request,
                        prefix_length=prefix_length,
                    )
                trace.action_buffer_occupancy = actions.occupancy
                inference_count += 1
                previous_observation = selected
                trace_dict = asdict(trace)
                traces.append(trace_dict)
                if result.decode_future is not None:
                    pending_decodes.append((result.decode_future, trace_dict))

            action = actions.pop()
            if action is None:
                termination_reason = "action_buffer_underflow"
                break
            action_start = monotonic_ns()
            if traces and traces[-1]["action_execution_timestamp_ns"] == 0:
                traces[-1]["action_execution_timestamp_ns"] = action_start
                traces[-1]["observation_age_at_action_start_ms"] = (
                    action_start - traces[-1]["observation_timestamp_ns"]
                ) / 1e6
                traces[-1]["action_age_ms"] = (action_start - traces[-1]["inference_finish_ns"]) / 1e6
            previous_eef = np.asarray(raw.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
            environment_start = monotonic_ns()
            raw, _, done, _ = env.step(action)
            environment_time_ms += (monotonic_ns() - environment_start) / 1e6
            current_eef = np.asarray(raw.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
            if float(np.linalg.norm(current_eef - previous_eef)) < 1e-4:
                no_progress_streak += 1
                maximum_no_progress_streak = max(maximum_no_progress_streak, no_progress_streak)
            else:
                no_progress_streak = 0
            executed_since_replan.append(action.copy())
            all_executed_actions.append(action.copy())
            control_step += 1
            if done:
                success = True
                termination_reason = "success"
                break
    except Exception as error:
        LOGGER.exception("episode failed")
        termination_reason = f"error:{type(error).__name__}:{error}"
    finally:
        pipeline.close()
        if video_writer is not None:
            video_writer.close()
        for future, trace_dict in pending_decodes:
            decoded = future.result()
            trace_dict["future_state_decode_latency_ms"] = decoded.elapsed_ms
            trace_dict["vae_decode_count"] = int(not decoded.discarded)
            trace_dict["cancelled"] = bool(decoded.discarded)

    for trace_dict in traces:
        trace_writer.write(trace_dict)

    actions.clear()
    action_array = np.stack(all_executed_actions) if all_executed_actions else np.empty((0, 7))
    canonical_actions = np.ascontiguousarray(action_array, dtype=np.float32)
    canonical_chunks = (
        np.ascontiguousarray(np.stack(all_policy_action_chunks), dtype=np.float32)
        if all_policy_action_chunks
        else np.empty((0, 16, 7), dtype=np.float32)
    )
    action_chunks_path = None
    if action_trace_path is not None:
        action_trace_path = Path(action_trace_path)
        action_trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(action_trace_path, canonical_actions, allow_pickle=False)
        action_chunks_path = action_trace_path.with_name(
            f"{action_trace_path.stem}.chunks.npy"
        )
        np.save(action_chunks_path, canonical_chunks, allow_pickle=False)
    smoothness = float(np.mean(np.linalg.norm(np.diff(action_array, axis=0), axis=1))) if len(action_array) > 1 else 0.0
    jerk = float(np.mean(np.linalg.norm(np.diff(action_array, n=3, axis=0), axis=1))) if len(action_array) > 3 else 0.0
    chunk_boundaries = list(range(16, len(action_array), 16))
    boundary_discontinuity = (
        float(
            np.mean(
                [
                    np.linalg.norm(action_array[index] - action_array[index - 1])
                    for index in chunk_boundaries
                ]
            )
        )
        if chunk_boundaries
        else 0.0
    )
    gripper = np.sign(action_array[:, -1]) if len(action_array) else np.empty(0)
    transition_indices = np.flatnonzero(np.diff(gripper) != 0) + 1 if len(gripper) > 1 else np.empty(0, dtype=int)
    chatter_count = (
        int(np.sum(np.diff(transition_indices) <= 2)) if len(transition_indices) > 1 else 0
    )
    control_frequency_hz = float(config["evaluation"].get("control_frequency_hz") or 20.0)
    record = {
        "task_name": task_name,
        "episode_index": episode_index,
        "random_seed": seed,
        "success": success,
        "episode_steps": control_step,
        "episode_wall_clock_time_s": (monotonic_ns() - episode_start) / 1e9,
        "episode_time_ms": (monotonic_ns() - episode_start) / 1e6,
        "environment_time_ms": environment_time_ms,
        "termination_reason": termination_reason,
        "inference_count": inference_count,
        "action_buffer_underflow_count": actions.stats.underflow_count,
        "stale_inference_count": actions.stats.stale_inference_count,
        "cancelled_inference_count": pipeline.registry.cancelled_count,
        "discarded_observation_count": observations.discarded_count,
        "wasted_speculative_compute_time_ms": pipeline.wasted_speculative_compute_ms,
        "wasted_unused_actions": actions.stats.wasted_unused_actions,
        "action_smoothness": smoothness,
        "action_jerk": jerk,
        "chunk_boundary_action_discontinuity": boundary_discontinuity,
        "gripper_transition_count": int(len(transition_indices)),
        "gripper_chatter_count": chatter_count,
        "action_saturation_ratio": (
            float(np.mean(np.abs(action_array) >= 0.999)) if len(action_array) else 0.0
        ),
        "no_progress_duration_s": maximum_no_progress_streak / control_frequency_hz,
        "total_future_decode_ms": float(
            sum(float(trace.get("future_state_decode_latency_ms", 0.0)) for trace in traces)
        ),
        "executed_action_sha256": hashlib.sha256(canonical_actions.tobytes()).hexdigest(),
        "action_chunks_sha256": hashlib.sha256(canonical_chunks.tobytes()).hexdigest(),
        "executed_actions_path": str(action_trace_path) if action_trace_path is not None else None,
        "action_chunks_path": str(action_chunks_path) if action_chunks_path is not None else None,
        "video_path": str(video_path) if video_path is not None else None,
    }
    return record, traces


def repository_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()
        except Exception as error:
            return f"unavailable: {error}"

    return {
        "path": str(path),
        "commit_sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def checkpoint_size_bytes(checkpoint: str | None) -> int | None:
    if not checkpoint:
        return None
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return None


def adapter_parameter_count(adapter: Any) -> int | None:
    module = getattr(adapter, "model", None)
    if module is None and getattr(adapter, "server", None) is not None:
        module = getattr(adapter.server, "transformer", None)
    if module is None or not hasattr(module, "parameters"):
        return None
    return sum(parameter.numel() for parameter in module.parameters())


def environment_report(config: dict[str, Any]) -> str:
    lines = [f"Python: {sys.version}", f"Platform: {platform.platform()}"]
    try:
        import torch

        lines.extend([f"PyTorch: {torch.__version__}", f"CUDA: {torch.version.cuda}"])
    except Exception as error:
        lines.append(f"PyTorch: unavailable ({error})")
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        lines.append("GPU:\n" + output)
    except Exception as error:
        lines.append(f"GPU: unavailable ({error})")
    lines.append(f"Checkpoint: {config['model'].get('checkpoint', '')}")
    return "\n".join(lines) + "\n"


def run_experiment(model: str, config: dict[str, Any], output_dir: str | Path, mock: bool = False) -> Path:
    configure_repository_paths(config)
    run_id = config.get("run_id") or f"{model}-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    run_dir = Path(output_dir) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    import yaml

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "environment.txt").write_text(environment_report(config), encoding="utf-8")
    repos = config.get("repositories", {})
    states = [repository_state(path) for path in repos.values() if path]
    (run_dir / "git_state.txt").write_text(
        "\n".join(json.dumps(state, ensure_ascii=False) for state in states)
        + f"\ncheckpoint={config['model'].get('checkpoint', '')}\n"
        + environment_report(config),
        encoding="utf-8",
    )
    stdout_stream = (run_dir / "stdout.log").open("a", encoding="utf-8")
    previous_stdout, previous_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(previous_stdout, stdout_stream), Tee(previous_stderr, stdout_stream)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(stdout_handler)
    LOGGER.setLevel(logging.INFO)

    adapter = build_adapter(model, config, mock)
    episodes_writer = JsonlWriter(run_dir / "episodes.jsonl")
    trace_writer = JsonlWriter(run_dir / "inference_trace.jsonl")
    sampler = ResourceSampler(float(config["metrics"].get("resource_interval_seconds", 0.5)))
    sampler.start()
    episode_records: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    evaluation = config["evaluation"]
    task_ids = list(map(int, evaluation.get("task_ids", [0, 1])))
    seeds = list(map(int, evaluation.get("seeds", [195, 196, 197])))
    episodes_per_task = int(evaluation.get("episodes_per_task", len(seeds)))

    fatal_error: Exception | None = None
    parameter_count: int | None = None
    try:
        LOGGER.info("starting run_id=%s model=%s mock=%s", run_id, model, mock)
        for task_id in task_ids:
            if mock:
                task_name = f"mock_task_{task_id}"
            else:
                env_probe = RealLiberoEnvironment(evaluation["task_suite"], task_id, int(evaluation.get("resolution", 128)))
                task_name = env_probe.description
                env_probe.close()
            for episode_index in range(episodes_per_task):
                seed = seeds[episode_index % len(seeds)]
                random.seed(seed)
                np.random.seed(seed)
                env = (
                    MockLiberoEnvironment(seed)
                    if mock
                    else RealLiberoEnvironment(evaluation["task_suite"], task_id, int(evaluation.get("resolution", 128)))
                )
                try:
                    video_path = (
                        run_dir / "videos" / f"task{task_id:02d}_episode{episode_index:03d}_seed{seed}.mp4"
                        if bool(evaluation.get("record_video", False))
                        else None
                    )
                    action_trace_path = (
                        run_dir / "actions" / f"task{task_id:02d}_episode{episode_index:03d}_seed{seed}.npy"
                        if bool(evaluation.get("record_actions", True))
                        else None
                    )
                    record, traces = run_episode(
                        adapter,
                        env,
                        config,
                        task_name,
                        episode_index,
                        seed,
                        trace_writer,
                        video_path=video_path,
                        action_trace_path=action_trace_path,
                    )
                finally:
                    env.close()
                episodes_writer.write(record)
                episode_records.append(record)
                all_traces.extend(traces)
    except Exception as error:
        fatal_error = error
        LOGGER.exception("run failed before all episodes completed")
    finally:
        parameter_count = adapter_parameter_count(adapter)
        adapter.close()
        resources = sampler.stop()
        LOGGER.info("finished run_id=%s", run_id)
        LOGGER.removeHandler(stdout_handler)
        sys.stdout, sys.stderr = previous_stdout, previous_stderr
        stdout_stream.close()

    success = [int(item["success"]) for item in episode_records]
    summary = {
        "run_id": run_id,
        "model": model,
        "mock": mock,
        "status": "failed" if fatal_error else "completed",
        "fatal_error": f"{type(fatal_error).__name__}: {fatal_error}" if fatal_error else None,
        "episodes": len(episode_records),
        "aggregate_success_rate": float(np.mean(success)) if success else None,
        "episode_completion_time_s": {
            "mean": float(np.mean([item["episode_wall_clock_time_s"] for item in episode_records]))
            if episode_records
            else None
        },
        "latency": summarize_traces(all_traces, int(config["metrics"].get("warmup_requests", 10))),
        "resources": resources,
        "fairness": {
            "checkpoint": config["model"].get("checkpoint"),
            "checkpoint_size_bytes": checkpoint_size_bytes(config["model"].get("checkpoint")),
            "parameter_count": parameter_count,
            "precision": config["model"].get("precision"),
            "image_resolution": config["model"].get("image_resolution"),
            "camera_views": config["model"].get("camera_views"),
            "history_length": config["history"].get("length"),
            "frame_stride": config["history"].get("frame_stride"),
            "action_horizon": config["model"].get("action_horizon"),
            "executed_horizon_mode": config["execution_prefix"].get("mode"),
            "executed_horizon": config["execution_prefix"].get("length"),
            "cache_enabled": model == "lingbot_va",
            "control_frequency_hz": config["evaluation"].get("control_frequency_hz"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    if fatal_error is not None:
        raise RuntimeError(f"run failed; diagnostics saved to {run_dir}: {fatal_error}") from fatal_error
    return run_dir

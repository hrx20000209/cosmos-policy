#!/usr/bin/env python3
"""Run one resumable frozen-Cosmos closed-loop PV0 episode.

The process deliberately owns one manifest row and one route.  This keeps
multi-GPU scheduling recoverable and prevents a long LIBERO rollout from
blocking unrelated work.  It supports the optional, preregistered short action
interruption used by S5; that interruption changes only execution outcomes,
never a policy input or route-selection rule.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adapters.cosmos_adapter import CosmosAdapter
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (
    ManifestLiberoEnvironment,
    config_for_row,
    install_libero_checkout,
    load_jsonl,
)
from experiments.libero_harness import load_yaml, run_episode
from pv0_overnight_common import ORIGINAL_CHECKPOINT, ORIGINAL_CHECKPOINT_SHA256, checkpoint_contract


MODES = ("fresh", "predicted_reuse", "native_persistent", "pv0_r0", "pv0_r1", "pv0_r2", "pv0_r3")


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def route_contract(mode: str) -> dict[str, Any]:
    common = {
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "hidden_activation_patch_used": False,
        "fresh_prefix_oracle_used": False,
    }
    if mode == "fresh":
        return {**common, "bootstrap_and_followups": "fresh camera preprocessing and VAE encoding"}
    if mode == "predicted_reuse":
        return {
            **common,
            "bootstrap": "fresh camera preprocessing and VAE encoding",
            "followups": "prior generated visual latent; no camera preprocessing",
        }
    if mode == "native_persistent":
        return {
            **common,
            "bootstrap": "fresh camera preprocessing and VAE encoding",
            "followups": "prior generated latent + native causal fresh visual prefix",
            "native_interface": "get_action:persistent_visual_correction_prefix_frames",
            "fresh_visual_prefix_frames": 13,
            "fresh_visual_arrival_denoiser_forward": 0,
        }
    if mode in {"pv0_r0", "pv0_r1", "pv0_r2", "pv0_r3"}:
        patterns = {
            "pv0_r0": ["PV0"],
            "pv0_r1": ["PV0", "P1"],
            "pv0_r2": ["PV0", "P1", "P1"],
            "pv0_r3": ["PV0", "P1", "P1", "P1"],
        }
        return {
            **common,
            "bootstrap": "fresh camera preprocessing and VAE encoding",
            "followup_pattern": patterns[mode],
            "PV0": "prior generated latent + native causal 13-frame fresh visual prefix",
            "P1": "prior generated visual latent without camera preprocessing",
            "fresh_visual_prefix_frames": 13,
            "fresh_visual_arrival_denoiser_forward": 0,
        }
    raise ValueError(mode)


def expected_visual_modes(mode: str, request_count: int) -> list[str]:
    if request_count <= 0:
        return []
    if mode in {"pv0_r0", "pv0_r1", "pv0_r2", "pv0_r3"}:
        patterns = {
            "pv0_r0": ["native_persistent"],
            "pv0_r1": ["native_persistent", "predicted"],
            "pv0_r2": ["native_persistent", "predicted", "predicted"],
            "pv0_r3": ["native_persistent", "predicted", "predicted", "predicted"],
        }
        sequence = patterns[mode]
        return ["fresh", *[sequence[(index - 1) % len(sequence)] for index in range(1, request_count)]]
    followup = {"fresh": "fresh", "predicted_reuse": "predicted", "native_persistent": "native_persistent"}[mode]
    return ["fresh", *([followup] * (request_count - 1))]


def validate_trace_contract(mode: str, traces: list[dict[str, Any]]) -> dict[str, Any]:
    visual_modes = [str((trace.get("extra") or {}).get("visual_input_mode")) for trace in traces]
    forwards = [int(trace.get("denoiser_forward_count", -1)) for trace in traces]
    expected = expected_visual_modes(mode, len(traces))
    if visual_modes != expected:
        raise RuntimeError(f"visual route mismatch: expected={expected}, actual={visual_modes}")
    if any(forward != 1 for forward in forwards):
        raise RuntimeError(f"one-denoise contract failed: {forwards}")
    if mode == "native_persistent" and len(traces) > 1:
        native = sum(int((trace.get("extra") or {}).get("native_persistent_request_count", 0)) for trace in traces)
        if native != len(traces) - 1:
            raise RuntimeError(f"native persistent count mismatch: expected={len(traces)-1}, actual={native}")
    return {
        "status": "PASS",
        "visual_input_modes": visual_modes,
        "visual_input_mode_counts": dict(Counter(visual_modes)),
        "denoiser_forward_counts": forwards,
    }


def attach_request_alignment(
    feedback: list[dict[str, Any]], traces: list[dict[str, Any]], settle_steps: int
) -> None:
    """Attach only request/chunk bookkeeping to raw execution telemetry.

    The association is derived from control-step indices and the committed
    prefix length; it introduces no environment state or task information.
    """
    for sample in feedback:
        control_step = int(sample["environment_step"]) - int(settle_steps)
        sample["control_step_after_settle"] = control_step
        sample["request_id"] = None
        sample["request_control_step"] = None
        sample["nominal_action_index"] = None
        if control_step < 0:
            continue
        for trace in traces:
            start = int(trace["control_step_id"])
            length = int(trace["executed_prefix_length"])
            if start <= control_step < start + length:
                sample["request_id"] = str(trace["request_id"])
                sample["request_control_step"] = start
                sample["nominal_action_index"] = control_step - start
                break


class InferenceSeedAdapter(CosmosAdapter):
    """Keep the physical manifest seed fixed while optionally changing Cosmos noise."""

    def __init__(self, config: dict[str, Any], inference_seed_offset: int):
        super().__init__(config)
        self.inference_seed_offset = int(inference_seed_offset)

    def reset(self, task_description: str, seed: int) -> None:
        super().reset(task_description, int(seed) + self.inference_seed_offset)


class ExecutionFeedbackEnvironment:
    """Record controller-visible execution telemetry around every environment step.

    The record intentionally contains only robot proprioception and the action
    handed to the environment.  In particular, it never reads object poses,
    contacts, rewards, task predicates, or success.  This makes the resulting
    ``execution_feedback`` directly usable by a future runtime fast loop.

    ``zero_motion_preserve_gripper`` remains an optional, pre-registered S5
    perturbation.  It changes what is executed, never what the policy sees.
    """

    def __init__(self, base: Any, *, settle_steps: int, start_step: int | None, length: int):
        self.base = base
        self.interruption_enabled = start_step is not None
        self.trigger_start = int(settle_steps) + int(start_step or 0)
        self.trigger_end = self.trigger_start + int(length)
        self.total_steps = 0
        self.events: list[dict[str, Any]] = []
        self.execution_feedback: list[dict[str, Any]] = []
        self._raw: dict[str, Any] | None = None

    @staticmethod
    def _proprio(raw: dict[str, Any]) -> dict[str, list[float]]:
        return {
            "eef_pos": np.asarray(raw["robot0_eef_pos"], dtype=np.float32).tolist(),
            "eef_quat": np.asarray(raw["robot0_eef_quat"], dtype=np.float32).tolist(),
            "gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).tolist(),
        }

    def reset(self) -> Any:
        self.total_steps = 0
        self.events = []
        self.execution_feedback = []
        self._raw = self.base.reset()
        return self._raw

    def step(self, action: np.ndarray) -> Any:
        if self._raw is None:
            raise RuntimeError("execution feedback wrapper stepped before reset")
        planned = np.asarray(action, dtype=np.float32)
        executed = planned.copy()
        interrupted = self.interruption_enabled and self.trigger_start <= self.total_steps < self.trigger_end
        if interrupted:
            # A short hold preserves the requested gripper command but removes
            # Cartesian/joint progress.  It is an outcome mismatch, not an
            # observation/action heuristic and is identical for all routes.
            executed[:-1] = 0.0
            self.events.append(
                {
                    "environment_step": self.total_steps,
                    "planned_action": planned.tolist(),
                    "executed_action": executed.tolist(),
                    "kind": "zero_motion_preserve_gripper",
                }
            )
        before = self._proprio(self._raw)
        next_raw, reward, done, info = self.base.step(executed)
        after = self._proprio(next_raw)
        self.execution_feedback.append(
            {
                "environment_step": int(self.total_steps),
                "planned_action": planned.tolist(),
                "executed_action": executed.tolist(),
                "arm_command_changed_by_interruption": bool(interrupted),
                "before": before,
                "after": after,
            }
        )
        self._raw = next_raw
        self.total_steps += 1
        return next_raw, reward, done, info

    def close(self) -> None:
        self.base.close()


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def select_row(manifest: Path, episode_key: str) -> dict[str, Any]:
    matches = [row for row in load_jsonl(manifest) if row["episode_key"] == episode_key]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {episode_key}, got {len(matches)}")
    return matches[0]


def preserve_prior_failure(path: Path) -> None:
    if not path.is_file():
        return
    payload: dict[str, Any] | None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    if payload and payload.get("status") == "PASS":
        return
    archived = path.with_name(f"{path.stem}.prior_failure_{time.time_ns()}{path.suffix}")
    path.replace(archived)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--episode-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().with_name("server_sweep.yaml")
    )
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--execution-prefix", type=int, default=16, choices=(4, 8, 12, 16))
    parser.add_argument("--inference-seed-offset", type=int, default=0)
    parser.add_argument("--interruption-start-step", type=int, default=None)
    parser.add_argument("--interruption-length", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in [0.05, 1.0]")
    if args.interruption_start_step is not None and args.interruption_start_step < 0:
        raise ValueError("interruption-start-step must be non-negative")
    if args.interruption_length < 1:
        raise ValueError("interruption-length must be positive")
    checkpoint = checkpoint_contract(ORIGINAL_CHECKPOINT)
    if args.output.is_file():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if prior.get("status") == "PASS":
            print(json.dumps({"status": "PASS_ALREADY_COMPLETE", "output": str(args.output)}), flush=True)
            return
        preserve_prior_failure(args.output)
    if args.trace_output is not None and args.trace_output.is_file():
        preserve_prior_failure(args.trace_output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    row = select_row(args.manifest, args.episode_key)
    if int(row["denoising_steps"]) != 1:
        raise RuntimeError("manifest violates one-denoise contract")
    # EGL is the validated headless backend for the shared-GPU PV0 workers.
    # OSMesa can fail before policy construction on this server's PyOpenGL build.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    install_libero_checkout(Path(row["libero_repo"]), Path(row["libero_config_path"]))
    base = load_yaml(args.config)
    config = config_for_row({**base, "model": {**base["model"], "closed_loop_mode": args.mode}}, row)
    config["execution_prefix"] = {"mode": "fixed", "length": int(args.execution_prefix), "buffer_strategy": "replace"}
    if args.max_steps is not None:
        config["evaluation"]["max_steps"] = min(int(args.max_steps), int(row["max_steps"]))
    if int(config["denoising"]["steps"]) != 1 or config["denoising"]["scheduler"] != "fixed":
        raise RuntimeError("closed-loop worker must use fixed denoise=1")
    traces: list[dict[str, Any]] = []

    class Writer:
        def write(self, value: dict[str, Any]) -> None:
            traces.append(value)

    environment: Any = None
    adapter: InferenceSeedAdapter | None = None
    started = time.time_ns()
    try:
        base_environment = ManifestLiberoEnvironment(row, int(config["evaluation"]["resolution"]), None)
        environment = ExecutionFeedbackEnvironment(
            base_environment,
            settle_steps=int(config["evaluation"].get("settle_steps", 0)),
            start_step=(int(args.interruption_start_step) if args.interruption_start_step is not None else None),
            length=int(args.interruption_length),
        )
        adapter = InferenceSeedAdapter(config["model"], args.inference_seed_offset)
        action_path = args.output.parent / "actions" / f"{args.mode}_{row['episode_key']}_seedoff{args.inference_seed_offset}.npy"
        action_path.parent.mkdir(parents=True, exist_ok=True)
        record, _ = run_episode(
            adapter,
            environment,
            config,
            row["instruction"],
            int(row["init_state_index"]),
            int(row["seed"]),
            Writer(),
            action_trace_path=action_path,
        )
        torch.cuda.synchronize(device)
        trace_contract = validate_trace_contract(args.mode, traces)
        attach_request_alignment(
            environment.execution_feedback, traces, int(config["evaluation"].get("settle_steps", 0))
        )
        request_latency = [float(trace.get("total_policy_request_latency_ms", 0.0)) for trace in traces]
        model_latency = [
            float(((trace.get("extra") or {}).get("non_overlapping_stage_ms") or {}).get("model_generate_inclusive_ms", 0.0))
            for trace in traces
        ]
        free_after, _ = torch.cuda.mem_get_info(device)
        interruption = None
        if args.interruption_start_step is not None:
            interruption = {
                "kind": "zero_motion_preserve_gripper",
                "control_step_after_settle": int(args.interruption_start_step),
                "length": int(args.interruption_length),
                "events": environment.events,
                "event_count": len(environment.events),
            }
        payload: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "PV0_closed_loop_episode",
            "status": "PASS",
            "phase_role": "S5_action_outcome_preflight" if interruption is not None else "closed_loop",
            "mode": args.mode,
            "route_contract": route_contract(args.mode),
            "manifest_row": {
                key: row[key]
                for key in (
                    "episode_key",
                    "task_uid",
                    "task_name",
                    "suite",
                    "split",
                    "instruction",
                    "init_state_index",
                    "seed",
                    "max_steps",
                    "denoising_steps",
                )
            },
            **checkpoint,
            "inference_seed": int(row["seed"]) + int(args.inference_seed_offset),
            "inference_seed_offset": int(args.inference_seed_offset),
            "execution_prefix": int(args.execution_prefix),
            "value_used": False,
            "privileged_runtime_state_input": False,
            "adaptive_scheduler_used": False,
            "threshold_used": False,
            "hidden_activation_patch_used": False,
            "fresh_prefix_oracle_used": False,
            "action_outcome_perturbation": interruption,
            "execution_feedback_contract": {
                "runtime_observables_only": True,
                "contains": ["planned_action", "executed_action", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                "excludes": ["object_pose", "contact", "reward", "success", "task_predicate", "ground_truth_subgoal"],
                "action_semantics": "7D OSC_POSE relative EEF delta plus gripper command; arm-only scaling is legal, gripper is preserved",
            },
            "execution_feedback": environment.execution_feedback,
            "gpu": {
                "visible_device_index": int(device),
                "physical_gpu": os.environ.get("EVAL_PHYSICAL_GPU"),
                "device_name": torch.cuda.get_device_name(device),
                "memory_fraction_cap": float(args.memory_fraction),
                "memory_free_before_mib": float(free_before / 2**20),
                "memory_free_after_mib": float(free_after / 2**20),
                "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
                "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
                "total_mib": float(total_memory / 2**20),
            },
            "record": record,
            "trace_contract": trace_contract,
            "request_latency_ms": numeric_summary(request_latency),
            "model_generate_inclusive_ms": numeric_summary(model_latency),
            "trace_count": len(traces),
            "action_path": str(action_path),
            "shared_gpu_latency_caveat": True,
            "started_at_ns": started,
            "finished_at_ns": time.time_ns(),
        }
        atomic_write(args.output, payload)
        if args.trace_output is not None:
            atomic_write(args.trace_output, {"schema_version": 1, "status": "PASS", "traces": traces})
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "episode_key": row["episode_key"],
                    "success": bool(record.get("success")),
                    "termination_reason": record.get("termination_reason"),
                    "requests": len(traces),
                    "output": str(args.output),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as error:
        payload = {
            "schema_version": 1,
            "experiment": "PV0_closed_loop_episode",
            "status": "FAIL_RUNTIME",
            "mode": args.mode,
            "episode_key": args.episode_key,
            "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_input": False,
            "adaptive_scheduler_used": False,
            "error": f"{type(error).__name__}:{error}",
            "traceback": traceback.format_exc(limit=12),
            "trace_count": len(traces),
            "failed_at_ns": time.time_ns(),
        }
        atomic_write(args.output, payload)
        raise
    finally:
        if environment is not None:
            environment.close()
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    main()

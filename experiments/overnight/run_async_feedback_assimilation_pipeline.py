#!/usr/bin/env python3
"""Measure cross-action-execution action-commit latency.

This is a host-level closed-loop pipeline.  While the current action chunk is
being executed, the next request runs in a worker.  ``async_fresh`` is the
essential baseline: it receives the same execution overlap window without
using predicted visual content.  ``predict_correct_async`` additionally uses
Cosmos' predicted future visual condition for the first denoiser forward and
assimilates the fresh VAE prefix before the second forward.

The simulator is stepped with an explicit real-time dwell per action step so
the experiment measures a controller execution window.  It does not claim
physical robot timing; those numbers are reserved for the later SO101 gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.cosmos_adapter import CosmosAdapter  # noqa: E402
from runtime.async_pipeline import InferenceRequest  # noqa: E402
from runtime.observation_buffer import Observation  # noqa: E402
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (  # noqa: E402
    ManifestLiberoEnvironment,
    install_libero_checkout,
)
from experiments.libero_harness import extract_observation, load_yaml  # noqa: E402
from experiments.overnight.run_predict_correct_p3_closed_loop import (  # noqa: E402
    PRO_CONFIG,
    PRO_REPO,
    SWEEP_CONFIG,
    TASKS,
    build_rows,
)


CHECKPOINT = Path("/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def observation_from_raw(raw: dict[str, Any]) -> Observation:
    extracted = extract_observation(raw, flip_vertical=True)
    return Observation(
        timestamp_ns=time.monotonic_ns(),
        primary_image=extracted.primary_image,
        wrist_image=extracted.wrist_image,
        proprio=extracted.proprio,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_one(adapter: CosmosAdapter, observation: Observation, episode_id: str, control_step: int, seed: int):
    request = InferenceRequest.create(observation.timestamp_ns, episode_id, control_step)
    started_ns = time.monotonic_ns()
    output = adapter.infer(observation, request, denoising_steps=1)
    finished_ns = time.monotonic_ns()
    return output, {
        "request_id": request.request_id,
        "observation_timestamp_ns": observation.timestamp_ns,
        "inference_start_ns": started_ns,
        "inference_finish_ns": finished_ns,
        "seed": seed,
    }


def run_episode(
    row: dict[str, Any],
    base: dict[str, Any],
    mode: str,
    *,
    action_step_duration_ms: float,
    settle_steps: int,
    max_steps: int,
    resolution: int,
    adapter: CosmosAdapter | None = None,
) -> dict[str, Any]:
    env = ManifestLiberoEnvironment(row, resolution, None)
    owns_adapter = adapter is None
    if adapter is None:
        adapter_mode = mode if mode in {"predict_correct", "predict_correct_async"} else "fresh"
        if mode == "predicted_reuse":
            adapter_mode = "predicted_reuse"
        adapter = CosmosAdapter({
            **base["model"],
            "closed_loop_mode": adapter_mode,
            "predict_correct_steps": 2,
        })
    assert adapter is not None
    adapter.reset(row["instruction"], int(row["seed"]))
    episode_id = f"{row['label']}|{mode}"
    records: list[dict[str, Any]] = []
    success = False
    control_step = 0
    raw = env.reset()
    settle = np.zeros(7, dtype=np.float32)
    settle[-1] = -1.0
    for _ in range(settle_steps):
        raw, _, _, _ = env.step(settle.tolist())

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"feedback-{mode}")
    try:
        initial_observation = observation_from_raw(raw)
        initial_output, initial_timing = infer_one(adapter, initial_observation, episode_id, control_step, int(row["seed"]))
        current_output = initial_output
        current_action_start_ns = time.monotonic_ns()
        request_index = 0
        while control_step < max_steps and not success:
            current_execution_start_ns = time.monotonic_ns()
            future: Future | None = None
            next_observation: Observation | None = None
            next_submission_ns: int | None = None
            action_chunk = np.asarray(current_output.actions, dtype=np.float32)
            action_steps = min(len(action_chunk), max_steps - control_step)
            action_done = False
            next_request_context: dict[str, Any] | None = None

            for action_index in range(action_steps):
                step_action = action_chunk[action_index]
                raw, _, done, _ = env.step(step_action.tolist())
                control_step += 1
                time.sleep(max(action_step_duration_ms, 0.0) / 1000.0)
                if action_index == 0 and control_step < max_steps and not done:
                    next_observation = observation_from_raw(raw)
                    next_submission_ns = time.monotonic_ns()
                    next_request_context = {
                        "control_step": control_step,
                        "observation": next_observation,
                    }
                    if mode != "sync_fresh":
                        future = executor.submit(
                            infer_one,
                            adapter,
                            next_observation,
                            episode_id,
                            control_step,
                            int(row["seed"]),
                        )
                if done:
                    success = bool(env.env.check_success()) if hasattr(env.env, "check_success") else True
                    action_done = True
                    break

            execution_end_ns = time.monotonic_ns()
            if action_done or control_step >= max_steps:
                records.append({
                    "request_index": request_index,
                    "control_step": control_step,
                    "mode": mode,
                    "current_execution_start_ns": current_execution_start_ns,
                    "current_execution_end_ns": execution_end_ns,
                    "action_steps": action_steps,
                    "terminal": True,
                })
                break

            if mode == "sync_fresh":
                assert next_observation is not None
                next_output, timing = infer_one(adapter, next_observation, episode_id, control_step, int(row["seed"]))
                next_ready_ns = timing["inference_finish_ns"]
                next_submission_ns = timing["inference_start_ns"]
            else:
                assert future is not None
                next_output, timing = future.result()
                next_ready_ns = timing["inference_finish_ns"]
            commit_ns = time.monotonic_ns()
            records.append({
                "request_index": request_index,
                "next_request_index": request_index + 1,
                "current_control_step": control_step - action_steps,
                "next_control_step": control_step,
                "mode": mode,
                "observation_timestamp_ns": timing["observation_timestamp_ns"],
                "inference_start_ns": timing["inference_start_ns"],
                "inference_finish_ns": timing["inference_finish_ns"],
                "action_ready_ns": next_ready_ns,
                "action_commit_ns": commit_ns,
                "current_execution_start_ns": current_execution_start_ns,
                "current_execution_end_ns": execution_end_ns,
                "observation_age_ms": max((timing["inference_start_ns"] - timing["observation_timestamp_ns"]) / 1e6, 0.0),
                "action_commit_latency_ms": (commit_ns - timing["observation_timestamp_ns"]) / 1e6,
                "inference_latency_ms": (timing["inference_finish_ns"] - timing["inference_start_ns"]) / 1e6,
                "action_age_at_commit_ms": max((commit_ns - next_ready_ns) / 1e6, 0.0),
                "execution_overlap_ms": max((min(execution_end_ns, next_ready_ns) - max(current_execution_start_ns, timing["inference_start_ns"])) / 1e6, 0.0),
                "commit_wait_after_execution_ms": max((commit_ns - execution_end_ns) / 1e6, 0.0),
                "buffer_starvation": bool(next_ready_ns > execution_end_ns),
                "next_stage_metrics_ms": next_output.stage_metrics_ms,
                "next_denoiser_forward_count": next_output.denoiser_forward_count,
            })
            current_output = next_output
            request_index += 1
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        if owns_adapter:
            adapter.close()
        env.close()

    nonterminal = [record for record in records if not record.get("terminal")]
    return {
        "task_label": row["base_label"],
        "task_name": row["task_name"],
        "suite": row["suite"],
        "mode": mode,
        "init_state_index": row["init_state_index"],
        "seed": row["seed"],
        "success": success,
        "episode_steps": control_step,
        "request_count": len(records),
        "metrics": {
            "commit_latency_ms": {
                "p50": float(np.percentile([r["action_commit_latency_ms"] for r in nonterminal], 50)) if nonterminal else None,
                "p95": float(np.percentile([r["action_commit_latency_ms"] for r in nonterminal], 95)) if nonterminal else None,
            },
            "inference_latency_ms": {
                "p50": float(np.percentile([r["inference_latency_ms"] for r in nonterminal], 50)) if nonterminal else None,
                "p95": float(np.percentile([r["inference_latency_ms"] for r in nonterminal], 95)) if nonterminal else None,
            },
            "execution_overlap_ms": {
                "p50": float(np.percentile([r["execution_overlap_ms"] for r in nonterminal], 50)) if nonterminal else None,
                "p95": float(np.percentile([r["execution_overlap_ms"] for r in nonterminal], 95)) if nonterminal else None,
            },
            "commit_wait_after_execution_ms": {
                "p50": float(np.percentile([r["commit_wait_after_execution_ms"] for r in nonterminal], 50)) if nonterminal else None,
                "p95": float(np.percentile([r["commit_wait_after_execution_ms"] for r in nonterminal], 95)) if nonterminal else None,
            },
            "buffer_starvation_rate": float(np.mean([r["buffer_starvation"] for r in nonterminal])) if nonterminal else None,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-indices", default="0,2,4,6,8,9")
    parser.add_argument("--init-states", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-step-duration-ms", type=float, default=20.0)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--modes",
        default="sync_fresh,async_fresh,predicted_reuse,predict_correct_async",
        help="comma-separated modes; run one mode per process when parallelizing",
    )
    args = parser.parse_args()
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash mismatch")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    install_libero_checkout(PRO_REPO, PRO_CONFIG)
    base = load_yaml(SWEEP_CONFIG)
    task_indices = [int(value) for value in args.task_indices.split(",") if value.strip()]
    init_states = {int(value) for value in args.init_states.split(",") if value.strip()}
    rows = [row for row in build_rows(task_indices) if row["init_state_index"] in init_states]
    allowed_modes = {"sync_fresh", "async_fresh", "predicted_reuse", "predict_correct", "predict_correct_async"}
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not modes or any(mode not in allowed_modes for mode in modes):
        raise ValueError(f"unknown mode in {modes}; allowed={sorted(allowed_modes)}")
    episodes = []
    started_ns = time.time_ns()
    for mode in modes:
        adapter_mode = mode if mode in {"predict_correct", "predict_correct_async"} else "fresh"
        if mode == "predicted_reuse":
            adapter_mode = "predicted_reuse"
        adapter = CosmosAdapter({
            **base["model"],
            "closed_loop_mode": adapter_mode,
            "predict_correct_steps": 2,
        })
        try:
            for row in rows:
                print(json.dumps({"mode": mode, "task": row["label"]}, ensure_ascii=False), flush=True)
                episodes.append(run_episode(
                    row,
                    base,
                    mode,
                    action_step_duration_ms=args.action_step_duration_ms,
                    settle_steps=args.settle_steps,
                    max_steps=int(row["max_steps"]),
                    resolution=args.resolution,
                    adapter=adapter,
                ))
        finally:
            adapter.close()
    summary = {}
    for mode in modes:
        subset = [episode for episode in episodes if episode["mode"] == mode]
        summary[mode] = {
            "episodes": len(subset),
            "successes": sum(bool(episode["success"]) for episode in subset),
            "commit_latency_p50_ms": float(np.nanmedian([episode["metrics"]["commit_latency_ms"]["p50"] for episode in subset])),
            "commit_latency_p95_ms": float(np.nanmedian([episode["metrics"]["commit_latency_ms"]["p95"] for episode in subset])),
            "execution_overlap_p50_ms": float(np.nanmedian([episode["metrics"]["execution_overlap_ms"]["p50"] for episode in subset])),
            "buffer_starvation_rate_mean": float(np.nanmean([episode["metrics"]["buffer_starvation_rate"] for episode in subset])),
        }
    payload = {
        "schema_version": "async-feedback-assimilation-pipeline-v1",
        "experiment": "cross_request_action_commit_pipeline",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "modes": modes,
        "task_indices": task_indices,
        "init_states": sorted(init_states),
        "action_step_duration_ms": args.action_step_duration_ms,
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "summary": summary,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

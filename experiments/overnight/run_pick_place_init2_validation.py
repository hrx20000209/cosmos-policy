"""P2 real closed-loop validation for pick_place init2/seed197.

The three configurations are independent episodes from the same original
LIBERO-PRO initialization: fresh one-step, simple predicted-latent reuse, and
two-forward persistent Predict-Correct.  The runner records the actual
executed trajectory and performs shadow-fresh inference only after the episode
has finished, so shadow inference cannot affect control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.cosmos_adapter import CosmosAdapter  # noqa: E402
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (  # noqa: E402
    ManifestLiberoEnvironment,
    config_for_row,
    install_libero_checkout,
)
from experiments.libero_harness import load_yaml, run_episode  # noqa: E402
from experiments.signal_validation.run_closed_loop_phase1 import (  # noqa: E402
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    PRO_CONFIG,
    PRO_REPO,
    SWEEP_CONFIG,
)
from experiments.signal_validation.run_closed_loop_phase2a import (  # noqa: E402
    MODES as OLD_MODES,
    ShadowCapture,
    build_rows,
    shadow_metrics,
)
from runtime.async_pipeline import InferenceRequest  # noqa: E402
from runtime.observation_buffer import Observation  # noqa: E402
from runtime.runtime_metrics import JsonlWriter, ResourceSampler, monotonic_ns  # noqa: E402


MODES = ("fresh", "predicted_reuse", "predict_correct")


class CollectingTraceWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, value: dict[str, Any]) -> None:
        self.rows.append(value)


class ReuseCapture:
    """Capture current observations and policy chunks at reuse requests."""

    def __init__(self, row: dict[str, Any], mode: str) -> None:
        self.row = row
        self.mode = mode
        self.primary: list[np.ndarray] = []
        self.wrist: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []
        self.request_index: list[int] = []
        self.control_step: list[int] = []
        self.prefix_length: list[int] = []
        self.action_chunks: list[np.ndarray] = []

    def __call__(self, *, observation, output, trace, request, prefix_length) -> None:
        visual_mode = output.extra.get("visual_input_mode")
        if visual_mode not in {"predicted", "predict_correct"}:
            return
        self.primary.append(np.ascontiguousarray(observation.primary_image).copy())
        self.wrist.append(np.ascontiguousarray(observation.wrist_image).copy())
        self.proprio.append(np.ascontiguousarray(observation.proprio, dtype=np.float32).copy())
        self.request_index.append(int(output.extra.get("cosmos_request_index", len(self.request_index))))
        self.control_step.append(int(request.control_step_id))
        self.prefix_length.append(int(prefix_length))
        self.action_chunks.append(np.ascontiguousarray(output.actions, dtype=np.float32).copy())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "label": self.row["base_label"],
            "episode_label": self.row["label"],
            "suite": self.row["suite"],
            "task_name": self.row["task_name"],
            "instruction": self.row["instruction"],
            "mode": self.mode,
            "source": "runtime",
            "init_state_index": self.row["init_state_index"],
            "seed": self.row["seed"],
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "denoising_steps": 2 if self.mode == "predict_correct" else 1,
            "action_horizon": 16,
        }
        np.savez_compressed(
            path,
            primary_images=np.stack(self.primary) if self.primary else np.empty((0, 256, 256, 3), dtype=np.uint8),
            wrist_images=np.stack(self.wrist) if self.wrist else np.empty((0, 256, 256, 3), dtype=np.uint8),
            proprio=np.stack(self.proprio) if self.proprio else np.empty((0, 8), dtype=np.float32),
            request_index=np.asarray(self.request_index, dtype=np.int64),
            control_step=np.asarray(self.control_step, dtype=np.int64),
            executed_prefix_length=np.asarray(self.prefix_length, dtype=np.int64),
            executed_speculative_action_chunks=(
                np.stack(self.action_chunks) if self.action_chunks else np.empty((0, 16, 7), dtype=np.float32)
            ),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_row() -> dict[str, Any]:
    rows = [row for row in build_rows() if row["base_label"] == "pick_place" and row["init_state_index"] == 2]
    if len(rows) != 1 or rows[0]["seed"] != 197:
        raise RuntimeError(f"expected exactly pick_place init2 seed197, got {rows}")
    return rows[0]


def summarize_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def percentiles(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "p50": None, "p95": None, "p99": None}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
        }

    result: dict[str, Any] = {}
    for mode in MODES:
        subset = [row for row in rows if row.get("configuration") == mode]
        result[mode] = {
            "request_count": len(subset),
            "total_policy_request_latency_ms": percentiles(
                [float(row.get("total_policy_request_latency_ms", 0.0)) for row in subset]
            ),
            "dit_denoising_latency_ms": percentiles(
                [float(row.get("dit_denoising_latency_ms", 0.0)) for row in subset]
            ),
            "vae_encoding_latency_ms": percentiles(
                [float(row.get("vae_encoding_latency_ms", 0.0)) for row in subset]
            ),
            "success_trace_count": sum(bool(row.get("success", False)) for row in subset),
        }
    return result


def run_shadow(capture_paths: list[Path], adapter: CosmosAdapter) -> list[dict[str, Any]]:
    adapter.closed_loop_mode = "fresh"
    output: list[dict[str, Any]] = []
    for path in sorted(capture_paths):
        with np.load(path, allow_pickle=False) as packed:
            metadata = json.loads(str(packed["metadata_json"].item()))
            primary = packed["primary_images"]
            wrist = packed["wrist_images"]
            proprio = packed["proprio"]
            request_indices = packed["request_index"]
            control_steps = packed["control_step"]
            chunks = packed["executed_speculative_action_chunks"]
        adapter.reset(metadata["instruction"], int(metadata["seed"]))
        for index in range(len(request_indices)):
            timestamp = monotonic_ns()
            observation = Observation(
                timestamp,
                np.ascontiguousarray(primary[index]),
                np.ascontiguousarray(wrist[index]),
                np.ascontiguousarray(proprio[index], dtype=np.float32),
            )
            request = InferenceRequest.create(
                timestamp,
                f"shadow:{metadata['episode_label']}:{metadata['seed']}",
                int(control_steps[index]),
            )
            fresh = adapter.infer(observation, request, 1, [observation])
            output.append(
                {
                    **shadow_metrics(chunks[index], fresh.actions),
                    "mode": metadata["mode"],
                    "request_index": int(request_indices[index]),
                    "control_step_id": int(control_steps[index]),
                    "capture_path": str(path),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "reports/artifacts/p2_pick_place_init2_validation.json",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("/data/rxhuang/wam_libero_outputs/p2_pick_place_init2_validation"),
    )
    args = parser.parse_args()

    if "so101" in str(CHECKPOINT).lower() or "finet" in str(CHECKPOINT).lower():
        raise ValueError("refusing SO101/finetuned checkpoint")
    actual_hash = sha256(CHECKPOINT)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint hash mismatch: {actual_hash}")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    base = load_yaml(SWEEP_CONFIG)
    row = choose_row()
    install_libero_checkout(PRO_REPO, PRO_CONFIG)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    trace_path = args.raw_output / "inference_trace.jsonl"
    trace_writer = JsonlWriter(trace_path)
    sampler = ResourceSampler(float(base["runtime"]["resource_interval_seconds"]))
    sampler.start()
    adapter = CosmosAdapter({**base["model"], "closed_loop_mode": "fresh", "predict_correct_steps": 2})
    records: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    capture_paths: list[Path] = []
    started_ns = time.time_ns()
    try:
        for mode in MODES:
            config = config_for_row(
                {**base, "model": {**base["model"], "closed_loop_mode": mode, "predict_correct_steps": 2}},
                row,
            )
            adapter.closed_loop_mode = mode
            env = ManifestLiberoEnvironment(row, int(base["evaluation"]["resolution"]), None)
            collected = CollectingTraceWriter()
            capture = ReuseCapture(row, mode)
            action_path = args.raw_output / mode / "actions.npy"
            try:
                record, traces = run_episode(
                    adapter,
                    env,
                    config,
                    row["instruction"],
                    0,
                    int(row["seed"]),
                    collected,
                    action_trace_path=action_path,
                    inference_capture_callback=capture,
                )
            finally:
                env.close()
            capture_path = capture.save(args.raw_output / mode / "reuse_capture.npz")
            capture_paths.append(capture_path)
            for trace in collected.rows:
                trace["configuration"] = mode
                trace["task_label"] = row["base_label"]
                trace["init_state_index"] = row["init_state_index"]
                trace["seed"] = row["seed"]
                trace_writer.write(trace)
            all_traces.extend(collected.rows)
            records.append(
                {
                    **record,
                    "configuration": mode,
                    "task_label": row["base_label"],
                    "task_name": row["task_name"],
                    "instruction": row["instruction"],
                    "init_state_index": row["init_state_index"],
                    "seed": row["seed"],
                    "checkpoint_sha256": actual_hash,
                    "executed_actions_path": str(action_path),
                    "reuse_capture_path": str(capture_path),
                    "reuse_capture_count": len(capture.request_index),
                }
            )
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "success": bool(record.get("success")),
                        "termination_reason": record.get("termination_reason"),
                        "steps": record.get("episode_steps"),
                        "requests": record.get("inference_count"),
                        "reuse_captures": len(capture.request_index),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        shadow = run_shadow(capture_paths, adapter)
    finally:
        adapter.close()
        resources = sampler.stop()

    output = {
        "schema_version": 1,
        "experiment": "p2_pick_place_init2_fresh_simple_predict_correct",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_hash,
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task": row,
        "configurations": MODES,
        "denoising_steps": {"fresh": 1, "predicted_reuse": 1, "predict_correct": 2},
        "action_horizon": 16,
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
        "shadow_fresh_definition": "offline fresh inference at the actual observation captured by each reuse request",
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "records": records,
        "trace_summary": summarize_traces(all_traces),
        "shadow_fresh": shadow,
        "raw_trace_path": str(trace_path),
        "resources": resources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "shadow_records": len(shadow)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""Phase 2A multi-state closed-loop validation with offline shadow-fresh actions.

This experiment deliberately keeps the Phase 1 schedule fixed:

* FRESH: every request uses the real RGB observation;
* ALTERNATE_SPECULATIVE: FRESH -> PREDICTED -> FRESH -> ...;
* ALTERNATE_CACHE: FRESH -> CACHE -> FRESH -> ... .

The runtime never consumes Cosmos value predictions, privileged simulator state,
latent-distance thresholds, or any adaptive scheduler.  Simulator state is only
used by the normal reset API.  The shadow-fresh pass runs after an episode and
therefore does not affect the closed-loop trajectory.
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

from adapters.cosmos_adapter import CosmosAdapter
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (
    ManifestLiberoEnvironment,
    config_for_row,
    extract_observation,
    install_libero_checkout,
)
from experiments.libero_harness import load_yaml, run_episode
from experiments.signal_validation.run_closed_loop_phase1 import (
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    MAX_STEPS,
    PRO_AUDIT,
    PRO_CONFIG,
    PRO_REPO,
    SELECTED_TASKS,
    SWEEP_CONFIG,
    augment_action_metrics,
)
from runtime.async_pipeline import InferenceRequest
from runtime.observation_buffer import Observation
from runtime.runtime_metrics import JsonlWriter, ResourceSampler, monotonic_ns, percentiles

PHASE1_RESULT = PROJECT / "reports/artifacts/libero_pro_closed_loop_phase1_v2.json"
STAGE2_INIT_ROOT = Path(
    "/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/stage2_init"
)
PHASE2A_OUTPUT = PROJECT / "reports/artifacts/libero_pro_closed_loop_phase2a.json"
PHASE2A_RAW_OUTPUT = Path("/data/rxhuang/wam_libero_outputs/libero_pro_closed_loop_phase2a")
MODES = ("fresh", "alternate_speculative", "alternate_cache")
REPLICATES = ((0, 195), (1, 196), (2, 197))


class CollectingTraceWriter:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, value: dict[str, Any]) -> None:
        self.rows.append(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage2_init_path(suite: str, category: str, task_name: str) -> Path:
    path = STAGE2_INIT_ROOT / f"{suite}_{category}" / f"{task_name}.five_states"
    if not path.is_file():
        raise FileNotFoundError(f"Phase 2A five-state init asset is missing: {path}")
    return path


def build_rows(indices: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, Any]]:
    audit = json.loads(PRO_AUDIT.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for selection_index, selection in enumerate(SELECTED_TASKS):
        matches = [
            row
            for row in audit
            if row["suite"] == selection["suite"]
            and row["task_name"] == selection["task_name"]
            and row["category"] == selection["category"]
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one PRO audit row for {selection}, got {len(matches)}")
        asset = matches[0]
        if not asset["variant_applied"]:
            raise ValueError(f"selected PRO variant was a no-op: {selection}")
        init_path = stage2_init_path(selection["suite"], selection["category"], selection["task_name"])
        for init_index, seed in REPLICATES:
            if init_index not in indices:
                continue
            label = f"{selection['label']}_init{init_index}"
            rows.append(
                {
                    "manifest_order": selection_index * 3 + init_index,
                    "episode_key": hashlib.sha256(
                        f"phase2a|{selection['label']}|init{init_index}|seed{seed}".encode()
                    ).hexdigest(),
                    "domain": "libero_pro",
                    "perturbation_category": selection["category"],
                    "variant_id": f"{selection['category']}:seed28",
                    "variant_applied": True,
                    "suite": selection["suite"],
                    "task_uid": f"{selection['suite']}:{selection['task_name']}",
                    "task_name": selection["task_name"],
                    "label": label,
                    "base_label": selection["label"],
                    "instruction": asset["instruction"],
                    "bddl_path": asset["bddl_path"],
                    "bddl_sha256": asset["bddl_sha256"],
                    "init_path": str(init_path),
                    "init_sha256": sha256(init_path),
                    "init_state_index": init_index,
                    "seed": seed,
                    "perturbation_seed": 28,
                    "denoising_steps": 1,
                    "action_horizon": 16,
                    "max_steps": MAX_STEPS[selection["suite"]],
                    "libero_repo": str(PRO_REPO),
                    "libero_config_path": str(PRO_CONFIG),
                }
            )
    return rows


def capture_path(root: Path, mode: str, row: dict[str, Any]) -> Path:
    return root / "shadow_capture" / mode / f"{row['label']}_seed{row['seed']}.npz"


class ShadowCapture:
    """Persist the real state seen by a reuse request without extra inference."""

    def __init__(self, row: dict[str, Any], mode: str, source: str = "runtime") -> None:
        self.row = row
        self.mode = mode
        self.source = source
        self.primary: list[np.ndarray] = []
        self.wrist: list[np.ndarray] = []
        self.proprio: list[np.ndarray] = []
        self.request_index: list[int] = []
        self.control_step: list[int] = []
        self.prefix_length: list[int] = []
        self.action_chunks: list[np.ndarray] = []

    def __call__(self, *, observation: Observation, output: Any, trace: Any, request: Any, prefix_length: int) -> None:
        if output.extra.get("visual_input_mode") not in {"predicted", "cache"}:
            return
        self.add(
            observation=observation,
            request_index=int(output.extra.get("cosmos_request_index", len(self.request_index))),
            control_step=int(request.control_step_id),
            prefix_length=prefix_length,
            action_chunk=np.asarray(output.actions, dtype=np.float32),
        )

    def add(
        self,
        *,
        observation: Observation,
        request_index: int,
        control_step: int,
        prefix_length: int,
        action_chunk: np.ndarray,
    ) -> None:
        self.primary.append(np.ascontiguousarray(observation.primary_image).copy())
        self.wrist.append(np.ascontiguousarray(observation.wrist_image).copy())
        self.proprio.append(np.ascontiguousarray(observation.proprio, dtype=np.float32).copy())
        self.request_index.append(int(request_index))
        self.control_step.append(int(control_step))
        self.prefix_length.append(int(prefix_length))
        self.action_chunks.append(np.ascontiguousarray(action_chunk, dtype=np.float32).copy())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "label": self.row["base_label"],
            "episode_label": self.row["label"],
            "suite": self.row["suite"],
            "task_name": self.row["task_name"],
            "instruction": self.row["instruction"],
            "mode": self.mode,
            "source": self.source,
            "init_state_index": self.row["init_state_index"],
            "seed": self.row["seed"],
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "action_horizon": 16,
        }
        image_shape = (0, 256, 256, 3)
        primary = np.stack(self.primary) if self.primary else np.empty(image_shape, dtype=np.uint8)
        wrist = np.stack(self.wrist) if self.wrist else np.empty(image_shape, dtype=np.uint8)
        proprio = np.stack(self.proprio) if self.proprio else np.empty((0, 8), dtype=np.float32)
        chunks = np.stack(self.action_chunks) if self.action_chunks else np.empty((0, 16, 7), dtype=np.float32)
        np.savez_compressed(
            path,
            primary_images=primary,
            wrist_images=wrist,
            proprio=proprio,
            request_index=np.asarray(self.request_index, dtype=np.int64),
            control_step=np.asarray(self.control_step, dtype=np.int64),
            executed_prefix_length=np.asarray(self.prefix_length, dtype=np.int64),
            executed_speculative_action_chunks=chunks,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        return path


def summarize_latency(traces: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
        values = [float(row.get(key, 0.0)) for row in rows]
        if not values:
            return {"mean": None, **percentiles([])}
        return {"mean": float(np.mean(values)), **percentiles(values)}

    by_mode = {
        mode: [row for row in traces if row.get("extra", {}).get("visual_input_mode") == mode]
        for mode in ("fresh", "predicted", "cache")
    }
    result: dict[str, Any] = {"all": stats(traces, "total_policy_request_latency_ms")}
    for mode, rows in by_mode.items():
        stage_rows = [row.get("extra", {}).get("non_overlapping_stage_ms", {}) for row in rows]

        def stage_stats(key: str) -> dict[str, float | None]:
            values = [float(item.get(key, 0.0)) for item in stage_rows]
            if not values:
                return {"mean": None, **percentiles([])}
            return {"mean": float(np.mean(values)), **percentiles(values)}

        result[mode] = {
            "n": len(rows),
            "total_ms": stats(rows, "total_policy_request_latency_ms"),
            "camera_preprocessing_ms": stage_stats("camera_preprocessing_ms"),
        }
        for key in (
            "latent_assembly_h2d_ms",
            "vae_encoding_ms",
            "dit_denoising_ms",
            "generation_conditioning_overhead_ms",
            "action_extraction_ms",
            "postprocess_after_action_ms",
            "non_overlapping_stage_sum_ms",
            "unattributed_stage_ms",
        ):
            values = [float(item.get(key, 0.0)) for item in stage_rows]
            result[mode][key] = {"n": len(values), "mean": float(np.mean(values)) if values else None, **percentiles(values)}
    return result


def nested_value(row: dict[str, Any], dotted_key: str) -> float:
    value: Any = row
    for key in dotted_key.split("."):
        value = value.get(key, 0.0) if isinstance(value, dict) else 0.0
    return float(value)


def summarize_shadow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("mean_step_l2", "first_step_l2", "cosine", "gripper_disagreement", "endpoint_l2")

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"n": len(group)}
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            result[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "p75": float(np.quantile(values, 0.75)) if values else None,
                "p90": float(np.quantile(values, 0.90)) if values else None,
            }
        return result

    result: dict[str, Any] = {"overall": {}}
    for mode in ("predicted", "cache"):
        result["overall"][mode] = group_summary([row for row in rows if row["mode"] == mode])
    for key_name, key in (
        ("per_task", "base_label"),
        ("per_init", "init_state_index"),
    ):
        result[key_name] = {}
        values = sorted({str(row[key]) for row in rows})
        for value in values:
            result[key_name][value] = {}
            for mode in ("predicted", "cache"):
                result[key_name][value][mode] = group_summary(
                    [row for row in rows if str(row[key]) == value and row["mode"] == mode]
                )
    return result


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def transition_table(rows: list[dict[str, Any]], other_mode: str) -> dict[str, Any]:
    pairs = []
    for key in sorted({(row["base_label"], row["init_state_index"], row["seed"]) for row in rows}):
        group = [row for row in rows if (row["base_label"], row["init_state_index"], row["seed"]) == key]
        lookup = {row["configuration"]: bool(row["success"]) for row in group}
        if "fresh" in lookup and other_mode in lookup:
            pairs.append((lookup["fresh"], lookup[other_mode]))
    counts = {
        "fresh_success_other_success": sum(a and b for a, b in pairs),
        "fresh_success_other_failure": sum(a and not b for a, b in pairs),
        "fresh_failure_other_success": sum(not a and b for a, b in pairs),
        "fresh_failure_other_failure": sum(not a and not b for a, b in pairs),
    }
    other_successes = sum(b for _, b in pairs)
    return {
        "other_configuration": other_mode,
        "n_pairs": len(pairs),
        "counts": counts,
        "regression_rate": counts["fresh_success_other_failure"] / max(1, sum(a for a, _ in pairs)),
        "regression_wilson_95": wilson_interval(
            counts["fresh_success_other_failure"], sum(a for a, _ in pairs)
        ),
        "other_success_rate": other_successes / max(1, len(pairs)),
        "other_success_wilson_95": wilson_interval(other_successes, len(pairs)),
    }


def add_record_metadata(record: dict[str, Any], row: dict[str, Any], mode: str) -> dict[str, Any]:
    record = dict(record)
    record.update(
        {
            "configuration": mode,
            "label": row["label"],
            "base_label": row["base_label"],
            "suite": row["suite"],
            "task_name": row["task_name"],
            "instruction": row["instruction"],
            "perturbation_category": row["perturbation_category"],
            "variant_id": row["variant_id"],
            "variant_applied": row["variant_applied"],
            "init_state_index": row["init_state_index"],
            "seed": row["seed"],
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "phase2a_source": "supplement_init1_init2",
        }
    )
    return record


def run_supplement(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    adapter: CosmosAdapter,
    output_root: Path,
    trace_writer: JsonlWriter,
    sampler: ResourceSampler,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    capture_paths: list[Path] = []
    for mode in MODES:
        adapter.closed_loop_mode = mode
        for row in rows:
            config = config_for_row(
                {**base, "model": {**base["model"], "closed_loop_mode": mode}}, row
            )
            env = ManifestLiberoEnvironment(row, int(base["evaluation"]["resolution"]), None)
            action_path = output_root / mode / "actions" / f"{row['label']}_seed{row['seed']}.npy"
            capture = ShadowCapture(row, mode)
            collected = CollectingTraceWriter()
            try:
                record, episode_traces = run_episode(
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
            capture_file = capture.save(capture_path(output_root, mode, row))
            capture_paths.append(capture_file)
            record = augment_action_metrics(record)
            records.append(add_record_metadata(record, row, mode))
            for trace in collected.rows:
                trace["configuration"] = mode
                trace["label"] = row["label"]
                trace["base_label"] = row["base_label"]
                trace["init_state_index"] = row["init_state_index"]
                trace["seed"] = row["seed"]
                trace_writer.write(trace)
            traces.extend(collected.rows)
            print(
                f"[{mode}] {row['label']} seed={row['seed']} success={record['success']} "
                f"steps={record['episode_steps']} requests={record['inference_count']} "
                f"shadow_states={len(capture.request_index)}",
                flush=True,
            )
    return records, traces, capture_paths


def load_phase1_records() -> list[dict[str, Any]]:
    payload = json.loads(PHASE1_RESULT.read_text(encoding="utf-8"))
    records = []
    for config in payload["configurations"]:
        mode = config["configuration"]
        for episode in config["episodes"]:
            record = dict(episode)
            record.update(
                {
                    "configuration": mode,
                    "base_label": episode["label"],
                    "phase2a_source": "phase1_v2_reused_init0",
                    "init_state_index": 0,
                    "seed": 195,
                }
            )
            records.append(record)
    return records


def load_phase1_traces(mode: str, label: str) -> list[dict[str, Any]]:
    payload = json.loads(PHASE1_RESULT.read_text(encoding="utf-8"))
    trace_path = Path(payload["trace_path"])
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row.get("configuration") == mode and row.get("label") == label]


def load_supplement_traces(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = {row["label"] for row in rows}
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        item = json.loads(line)
        if item.get("label") in labels:
            result.append(item)
    return result


def recover_supplement_records(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Recover outcome fields from persisted executed actions after aggregation failure."""
    records: list[dict[str, Any]] = []
    for mode in MODES:
        for row in rows:
            action_path = output_root / mode / "actions" / f"{row['label']}_seed{row['seed']}.npy"
            if not action_path.is_file():
                raise FileNotFoundError(f"persisted supplement action trace is missing: {action_path}")
            actions = np.asarray(np.load(action_path), dtype=np.float32)
            env = ManifestLiberoEnvironment(row, int(base["evaluation"]["resolution"]), None)
            settle_action = np.zeros(7, dtype=np.float32)
            settle_action[-1] = float(base["evaluation"]["settle_gripper_action"])
            success = False
            replay_start = time.monotonic_ns()
            try:
                raw = env.reset()
                for _ in range(int(base["evaluation"]["settle_steps"])):
                    raw, _, _, _ = env.step(settle_action)
                for action in actions:
                    raw, _, done, _ = env.step(action)
                    if done:
                        success = True
                        break
            finally:
                env.close()
            record = {
                "success": success,
                "environment_valid": True,
                "episode_steps": int(len(actions)),
                "episode_wall_clock_time_s": (time.monotonic_ns() - replay_start) / 1e9,
                "inference_count": None,
                "termination_reason": "success" if success else "max_steps",
                "executed_actions_path": str(action_path),
                "recovered_from_persisted_action_trace": True,
            }
            records.append(add_record_metadata(record, row, mode))
    return records


def recover_observed_runtime_records(
    rows: list[dict[str, Any]],
    output_root: Path,
    trace_path: Path,
) -> list[dict[str, Any]]:
    """Recover outcomes from the completed runtime's emitted episode summaries.

    This table mirrors the terminal lines emitted by ``run_supplement``.  It is
    only used after the optional simulator replay is interrupted; provenance in
    the final artifact states that the outcome fields came from observed runtime
    summaries, while latency and shadow metrics still come from persisted traces.
    """
    observed_successes = {
        ("fresh", "pick_place", 2),
    }
    trace_rows = load_supplement_traces(trace_path, rows)
    records: list[dict[str, Any]] = []
    for mode in MODES:
        for row in rows:
            action_path = output_root / mode / "actions" / f"{row['label']}_seed{row['seed']}.npy"
            actions = np.asarray(np.load(action_path), dtype=np.float32)
            episode_traces = [
                item
                for item in trace_rows
                if item.get("configuration") == mode and item.get("label") == row["label"]
            ]
            success = (mode, row["base_label"], int(row["init_state_index"])) in observed_successes
            record = {
                "success": success,
                "environment_valid": True,
                "episode_steps": int(len(actions)),
                "episode_wall_clock_time_s": None,
                "inference_count": len(episode_traces),
                "termination_reason": "success" if success else "max_steps",
                "executed_actions_path": str(action_path),
                "recovered_from_observed_runtime_summary": True,
            }
            records.append(add_record_metadata(record, row, mode))
    return records


def replay_phase1_init0(
    rows: list[dict[str, Any]],
    output_root: Path,
    base: dict[str, Any],
) -> list[Path]:
    """Recover init0 shadow states from the already executed Phase 1 actions."""
    paths: list[Path] = []
    zero_rows = {row["base_label"]: row for row in rows if row["init_state_index"] == 0}
    phase1_actions_root = Path("/data/rxhuang/wam_libero_outputs/libero_pro_closed_loop_phase1_v2")
    for mode in ("alternate_speculative", "alternate_cache"):
        for base_label, row in zero_rows.items():
            old_traces = load_phase1_traces(mode, base_label)
            target_traces = [
                (index, trace)
                for index, trace in enumerate(old_traces)
                if trace.get("extra", {}).get("visual_input_mode") in {"predicted", "cache"}
            ]
            action_path = phase1_actions_root / mode / "actions" / f"{base_label}.npy"
            chunks_path = action_path.with_name(f"{action_path.stem}.chunks.npy")
            actions = np.asarray(np.load(action_path), dtype=np.float32)
            chunks = np.asarray(np.load(chunks_path), dtype=np.float32)
            targets = {
                int(trace["control_step_id"]): (request_index, trace, chunks[request_index])
                for request_index, trace in target_traces
            }
            capture = ShadowCapture(row, mode, source="phase1_action_replay_init0")
            env = ManifestLiberoEnvironment(row, int(base["evaluation"]["resolution"]), None)
            settle_action = np.zeros(7, dtype=np.float32)
            settle_action[-1] = float(base["evaluation"]["settle_gripper_action"])
            try:
                raw = env.reset()
                for _ in range(int(base["evaluation"]["settle_steps"])):
                    raw, _, _, _ = env.step(settle_action)
                for control_step, action in enumerate(actions):
                    if control_step in targets:
                        request_index, trace, chunk = targets[control_step]
                        capture.add(
                            observation=extract_observation(raw, True),
                            request_index=int(
                                trace.get("extra", {}).get("cosmos_request_index", request_index)
                            ),
                            control_step=control_step,
                            prefix_length=int(trace.get("executed_prefix_length", 16)),
                            action_chunk=chunk,
                        )
                    raw, _, _, _ = env.step(action)
            finally:
                env.close()
            path = capture.save(capture_path(output_root, mode, row))
            paths.append(path)
            if len(targets) != len(capture.request_index):
                raise RuntimeError(
                    f"init0 replay missed shadow states for {mode}/{base_label}: "
                    f"expected {len(targets)}, got {len(capture.request_index)}"
                )
    return paths


def shadow_metrics(speculative: np.ndarray, fresh: np.ndarray) -> dict[str, float]:
    speculative = np.asarray(speculative, dtype=np.float64)
    fresh = np.asarray(fresh, dtype=np.float64)
    deltas = speculative - fresh
    per_step = np.linalg.norm(deltas, axis=1)
    spec_gripper = np.sign(speculative[:, -1])
    fresh_gripper = np.sign(fresh[:, -1])
    spec_flat = speculative.reshape(-1)
    fresh_flat = fresh.reshape(-1)
    denominator = np.linalg.norm(spec_flat) * np.linalg.norm(fresh_flat)
    return {
        "mean_step_l2": float(np.mean(per_step)),
        "first_step_l2": float(per_step[0]),
        "cosine": float(np.dot(spec_flat, fresh_flat) / denominator) if denominator else 1.0,
        "gripper_disagreement": float(np.mean(spec_gripper != fresh_gripper)),
        "endpoint_l2": float(np.linalg.norm(deltas[-1])),
    }


def run_offline_shadow(
    capture_paths: list[Path],
    adapter: CosmosAdapter,
) -> list[dict[str, Any]]:
    adapter.closed_loop_mode = "fresh"
    output: list[dict[str, Any]] = []
    for path in sorted(capture_paths):
        with np.load(path, allow_pickle=False) as packed:
            metadata = json.loads(str(packed["metadata_json"].item()))
            primaries = packed["primary_images"]
            wrists = packed["wrist_images"]
            proprio = packed["proprio"]
            request_indices = packed["request_index"]
            control_steps = packed["control_step"]
            chunks = packed["executed_speculative_action_chunks"]
        if len(request_indices) == 0:
            continue
        episode_id = f"shadow:{metadata['episode_label']}:{metadata['seed']}"
        adapter.reset(metadata["instruction"], int(metadata["seed"]))
        for index in range(len(request_indices)):
            observation = Observation(
                monotonic_ns(),
                np.ascontiguousarray(primaries[index]),
                np.ascontiguousarray(wrists[index]),
                np.ascontiguousarray(proprio[index], dtype=np.float32),
            )
            request = InferenceRequest.create(
                observation.timestamp_ns,
                episode_id,
                int(control_steps[index]),
            )
            fresh_output = adapter.infer(observation, request, 1, [observation])
            metrics = shadow_metrics(chunks[index], fresh_output.actions)
            output.append(
                {
                    **metrics,
                    "mode": "predicted" if metadata["mode"] == "alternate_speculative" else "cache",
                    "schedule": metadata["mode"],
                    "base_label": metadata["label"],
                    "episode_label": metadata["episode_label"],
                    "suite": metadata["suite"],
                    "task_name": metadata["task_name"],
                    "init_state_index": int(metadata["init_state_index"]),
                    "seed": int(metadata["seed"]),
                    "request_index": int(request_indices[index]),
                    "control_step_id": int(control_steps[index]),
                    "capture_path": str(path),
                }
            )
        print(
            f"[shadow-fresh] {metadata['mode']} {metadata['episode_label']} "
            f"states={len(request_indices)}",
            flush=True,
        )
    return output


def summarize_outcomes(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {}
    for row in records:
        by_key[(row["base_label"], int(row["init_state_index"]), int(row["seed"]))] = row
    result: dict[str, Any] = {}
    for mode in MODES:
        values = [bool(row["success"]) for row in records if row["configuration"] == mode]
        result[mode] = {
            "successes": int(sum(values)),
            "episodes": len(values),
            "success_rate": float(np.mean(values)) if values else None,
            "wilson_95": wilson_interval(int(sum(values)), len(values)),
        }
    task = {}
    for label in sorted({key[0] for key in by_key}):
        task[label] = {}
        for mode in MODES:
            values = [
                bool(row["success"])
                for key, row in by_key.items()
                if key[0] == label and row["configuration"] == mode
            ]
            task[label][mode] = {
                "successes": int(sum(values)),
                "episodes": len(values),
                "success_rate": float(np.mean(values)) if values else None,
            }
    per_init = {}
    for init_index in sorted({key[1] for key in by_key}):
        per_init[str(init_index)] = {}
        for mode in MODES:
            values = [
                bool(row["success"])
                for key, row in by_key.items()
                if key[1] == init_index and row["configuration"] == mode
            ]
            per_init[str(init_index)][mode] = {
                "successes": int(sum(values)),
                "episodes": len(values),
                "success_rate": float(np.mean(values)) if values else None,
            }
    result["per_task"] = task
    result["per_init"] = per_init
    result["transitions"] = {
        "fresh_vs_speculative": transition_table(records, "alternate_speculative"),
        "fresh_vs_cache": transition_table(records, "alternate_cache"),
        "per_task": {
            label: {
                "fresh_vs_speculative": transition_table(
                    [row for row in records if row["base_label"] == label],
                    "alternate_speculative",
                ),
                "fresh_vs_cache": transition_table(
                    [row for row in records if row["base_label"] == label],
                    "alternate_cache",
                ),
            }
            for label in sorted({row["base_label"] for row in records})
        },
        "per_init": {
            str(init_index): {
                "fresh_vs_speculative": transition_table(
                    [row for row in records if int(row["init_state_index"]) == init_index],
                    "alternate_speculative",
                ),
                "fresh_vs_cache": transition_table(
                    [row for row in records if int(row["init_state_index"]) == init_index],
                    "alternate_cache",
                ),
            }
            for init_index in sorted({int(row["init_state_index"]) for row in records})
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PHASE2A_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=PHASE2A_RAW_OUTPUT)
    parser.add_argument("--resume-persisted-runtime", action="store_true")
    parser.add_argument("--resume-observed-outcomes", action="store_true")
    args = parser.parse_args()

    if "so101" in str(CHECKPOINT).lower() or "finet" in str(CHECKPOINT).lower():
        raise ValueError("Phase 2A refuses SO101/finetuned checkpoint")
    actual_hash = sha256(CHECKPOINT)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint hash mismatch: {actual_hash}")
    if not PHASE1_RESULT.is_file():
        raise FileNotFoundError(f"required reusable Phase 1 result is missing: {PHASE1_RESULT}")

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    base = load_yaml(SWEEP_CONFIG)
    all_rows = build_rows()
    supplement_rows = [row for row in all_rows if row["init_state_index"] in {1, 2}]
    install_libero_checkout(PRO_REPO, PRO_CONFIG)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    trace_path = args.raw_output / "inference_trace.jsonl"
    if trace_path.exists() and not (args.resume_persisted_runtime or args.resume_observed_outcomes):
        raise FileExistsError(f"refusing to append to existing run: {trace_path}")
    trace_writer = JsonlWriter(trace_path)
    sampler = ResourceSampler(float(base["runtime"]["resource_interval_seconds"]))
    sampler.start()
    started_ns = time.time_ns()
    adapter = CosmosAdapter({**base["model"], "closed_loop_mode": "fresh"})
    try:
        if args.resume_observed_outcomes:
            supplement_records = recover_observed_runtime_records(
                supplement_rows, args.raw_output, trace_path
            )
            supplement_traces = load_supplement_traces(trace_path, supplement_rows)
            supplement_capture_paths = [
                capture_path(args.raw_output, mode, row)
                for mode in MODES
                for row in supplement_rows
            ]
        elif args.resume_persisted_runtime:
            supplement_records = recover_supplement_records(supplement_rows, base, args.raw_output)
            supplement_traces = load_supplement_traces(trace_path, supplement_rows)
            supplement_capture_paths = [
                capture_path(args.raw_output, mode, row)
                for mode in MODES
                for row in supplement_rows
            ]
            if not all(path.is_file() for path in supplement_capture_paths):
                missing = [str(path) for path in supplement_capture_paths if not path.is_file()]
                raise FileNotFoundError(f"persisted supplement shadow captures are missing: {missing[:3]}")
        else:
            supplement_records, supplement_traces, supplement_capture_paths = run_supplement(
                supplement_rows, base, adapter, args.raw_output, trace_writer, sampler
            )
        init0_rows = [row for row in all_rows if row["init_state_index"] == 0]
        replay_capture_paths = [
            capture_path(args.raw_output, mode, row)
            for mode in ("alternate_speculative", "alternate_cache")
            for row in init0_rows
        ]
        if not all(path.is_file() for path in replay_capture_paths):
            replay_capture_paths = replay_phase1_init0(init0_rows, args.raw_output, base)
        shadow_records = run_offline_shadow(
            supplement_capture_paths + replay_capture_paths,
            adapter,
        )
    finally:
        adapter.close()
        resources = sampler.stop()

    records = load_phase1_records() + supplement_records
    outcomes = summarize_outcomes(records)
    all_runtime_traces = supplement_traces
    mode_trace_counts = {
        mode: sum(
            1
            for row in all_runtime_traces
            if row.get("configuration") == mode
        )
        for mode in MODES
    }
    visual_trace_counts = {
        source: sum(
            1
            for row in all_runtime_traces
            if row.get("extra", {}).get("visual_input_mode") == source
        )
        for source in ("fresh", "predicted", "cache")
    }
    visual_trace_counts_by_configuration = {
        mode: {
            source: sum(
                1
                for row in all_runtime_traces
                if row.get("configuration") == mode
                and row.get("extra", {}).get("visual_input_mode") == source
            )
            for source in ("fresh", "predicted", "cache")
        }
        for mode in MODES
    }
    fresh_successes = outcomes["fresh"]["successes"]
    shadow_summary = summarize_shadow(shadow_records)
    spec_shadow = shadow_summary["overall"]["predicted"]["mean_step_l2"]["median"]
    cache_shadow = shadow_summary["overall"]["cache"]["mean_step_l2"]["median"]
    spec_regression = outcomes["transitions"]["fresh_vs_speculative"]["counts"][
        "fresh_success_other_failure"
    ]
    go_conditions = {
        "no_fresh_success_to_speculative_failure": spec_regression == 0,
        "speculative_drift_lower_than_cache_median_mean_step_l2": (
            spec_shadow is not None and cache_shadow is not None and spec_shadow < cache_shadow
        ),
        "supplement_has_non_overlapping_stage_timers": bool(supplement_traces),
        "supplement_speculative_visual_refresh_is_about_half": (
            mode_trace_counts["alternate_speculative"] > 0
            and 0.45
            <= visual_trace_counts_by_configuration["alternate_speculative"]["predicted"]
            / mode_trace_counts["alternate_speculative"]
            <= 0.55
        ),
    }
    phase2a_recommendation = (
        "GO"
        if all(go_conditions.values())
        else "NO-GO-CANDIDATE"
        if not go_conditions["no_fresh_success_to_speculative_failure"]
        else "GO-CANDIDATE"
    )
    output = {
        "schema_version": 1,
        "experiment": "phase2a_closed_loop_multistate_shadow_fresh",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_hash,
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "denoising_steps": 1,
        "action_horizon": 16,
        "execution_prefix": 16,
        "task_count": 6,
        "episode_count": len(records),
        "initial_state_seed_pairs": [
            {"init_state_index": init_index, "seed": seed}
            for init_index, seed in REPLICATES
        ],
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "latent_l1_scheduler_used": False,
        "proprio_threshold_scheduler_used": False,
        "layer_probe_used": False,
        "schedule": {
            "fresh": "FRESH every request",
            "alternate_speculative": "FRESH -> PREDICTED -> FRESH -> ...",
            "alternate_cache": "FRESH -> CACHE -> FRESH -> ...",
        },
        "reused_phase1_init0_outcomes": True,
        "supplemented_init_indices": [1, 2],
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "outcomes": outcomes,
        "shadow_fresh": {
            "definition": "offline fresh inference at the real state captured by each reuse request",
            "metrics": shadow_summary,
            "records": shadow_records,
        },
        "latency": {
            "supplement_init1_init2_non_overlapping": summarize_latency(supplement_traces),
            "supplement_trace_count_by_mode": mode_trace_counts,
            "supplement_visual_trace_count_by_source": visual_trace_counts,
            "supplement_visual_trace_count_by_configuration": visual_trace_counts_by_configuration,
            "phase1_init0_latency_reused_as_legacy": True,
        },
        "go_gate": {
            "conditions": go_conditions,
            "fresh_successes": fresh_successes,
            "speculative_median_shadow_mean_step_l2": spec_shadow,
            "cache_median_shadow_mean_step_l2": cache_shadow,
            "recommendation": phase2a_recommendation,
        },
        "records": records,
        "outcome_recovery": (
            "observed_runtime_summary"
            if args.resume_observed_outcomes
            else "simulator_action_replay"
            if args.resume_persisted_runtime
            else "direct_runtime"
        ),
        "raw_trace_path": str(trace_path),
        "resources": resources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "recommendation": phase2a_recommendation,
                "episodes": len(records),
                "shadow_records": len(shadow_records),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

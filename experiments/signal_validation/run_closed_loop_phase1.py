"""Phase 1 causal closed-loop test for Cosmos future-latent reuse on LIBERO-PRO.

The three configurations are run as independent episodes from the same
LIBERO-PRO reset state:

* fresh: every request reads RGB and runs RGB -> VAE -> DiT;
* alternate_speculative: FRESH -> PREDICTED -> FRESH -> ...;
* alternate_cache: FRESH -> CACHE -> FRESH -> ... .

This runner deliberately uses no Cosmos value, privileged simulator state, new
proxy, layer probe, or latent-distance scheduler. Simulator state is used only
through the normal reset/init-state API. Ground-truth observations remain in
the environment and are passed to the policy only as real proprio in reuse
requests.
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
    install_libero_checkout,
)
from experiments.libero_harness import load_yaml, run_episode
from runtime.runtime_metrics import JsonlWriter, ResourceSampler, percentiles

CHECKPOINT = Path("/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
EXPECTED_CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
SWEEP_CONFIG = PROJECT / "experiments/cosmos_denoising_libero_pro/configs/sweep.yaml"
PRO_REPO = Path("/data/rxhuang/repos/LIBERO-PRO")
PRO_CONFIG = PROJECT / "experiments/cosmos_denoising_libero_pro/configs/libero_pro_config"
PRO_AUDIT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/libero_pro_asset_audit.json")
DEFAULT_OUTPUT = PROJECT / "reports/artifacts/libero_pro_closed_loop_phase1.json"
RAW_OUTPUT = Path("/data/rxhuang/wam_libero_outputs/libero_pro_closed_loop_phase1")

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}

# The selected variants are all present in the official PRO asset audit and
# have variant_applied=True. Their task semantics cover the requested phases.
SELECTED_TASKS = (
    {
        "label": "pick_spatial",
        "suite": "libero_spatial",
        "task_name": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "category": "task",
    },
    {
        "label": "pick_place",
        "suite": "libero_object",
        "task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "category": "object",
    },
    {
        "label": "open_close",
        "suite": "libero_goal",
        "task_name": "open_the_middle_drawer_of_the_cabinet",
        "category": "task",
    },
    {
        "label": "contact_switch",
        "suite": "libero_goal",
        "task_name": "turn_on_the_stove",
        "category": "task",
    },
    {
        "label": "multi_stage",
        "suite": "libero_10",
        "task_name": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "category": "environment",
    },
    {
        "label": "spatial_rearrangement",
        "suite": "libero_10",
        "task_name": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "category": "position",
    },
)


class CollectingTraceWriter:
    """Capture run_episode traces so configuration labels are added once."""

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


def build_rows() -> list[dict[str, Any]]:
    audit = json.loads(PRO_AUDIT.read_text(encoding="utf-8"))
    rows = []
    for index, selection in enumerate(SELECTED_TASKS):
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
        rows.append(
            {
                "manifest_order": index,
                "episode_key": hashlib.sha256(
                    f"phase1|{selection['label']}|seed195|init0".encode()
                ).hexdigest(),
                "domain": "libero_pro",
                "perturbation_category": selection["category"],
                "variant_id": f"{selection['category']}:seed28",
                "variant_applied": True,
                "suite": selection["suite"],
                "task_uid": f"{selection['suite']}:{selection['task_name']}",
                "task_name": selection["task_name"],
                "label": selection["label"],
                "instruction": asset["instruction"],
                "bddl_path": asset["bddl_path"],
                "bddl_sha256": asset["bddl_sha256"],
                "init_path": asset["init_path"],
                "init_sha256": asset["init_sha256"],
                "init_state_index": 0,
                "seed": 195,
                "perturbation_seed": 28,
                "denoising_steps": 1,
                "action_horizon": 16,
                "max_steps": MAX_STEPS[selection["suite"]],
                "libero_repo": str(PRO_REPO),
                "libero_config_path": str(PRO_CONFIG),
            }
        )
    return rows


def summarize_latency(traces: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(key: str) -> dict[str, float | None]:
        values = [float(row.get(key, 0.0)) for row in traces]
        if not values:
            return {"mean": None, **percentiles([])}
        return {"mean": float(np.mean(values)), **percentiles(values)}

    def stats_for(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
        values = [float(row.get(key, 0.0)) for row in rows]
        if not values:
            return {"mean": None, **percentiles([])}
        return {"mean": float(np.mean(values)), **percentiles(values)}

    fresh = [row for row in traces if row.get("extra", {}).get("visual_input_mode") == "fresh"]
    predicted = [row for row in traces if row.get("extra", {}).get("visual_input_mode") == "predicted"]
    cached = [row for row in traces if row.get("extra", {}).get("visual_input_mode") == "cache"]
    predict_correct = [
        row for row in traces if row.get("extra", {}).get("visual_input_mode") == "predict_correct"
    ]
    return {
        "all": stats("total_policy_request_latency_ms"),
        "fresh_requests": stats_for(fresh, "total_policy_request_latency_ms"),
        "predicted_requests": stats_for(predicted, "total_policy_request_latency_ms"),
        "cached_requests": stats_for(cached, "total_policy_request_latency_ms"),
        "predict_correct_requests": stats_for(predict_correct, "total_policy_request_latency_ms"),
        "vae_ms": stats("vae_encoding_latency_ms"),
        "dit_ms": stats("dit_denoising_latency_ms"),
        "preprocess_ms": stats("preprocessing_latency_ms"),
        "action_postprocess_ms": stats("action_extraction_latency_ms"),
    }


def augment_action_metrics(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["executed_actions_path"])
    actions = np.asarray(np.load(path), dtype=np.float64)
    if len(actions):
        magnitudes = np.linalg.norm(actions, axis=1)
        record["action_magnitude_mean"] = float(np.mean(magnitudes))
        record["action_magnitude_p95"] = float(np.quantile(magnitudes, 0.95))
        record["gripper_mean"] = float(np.mean(actions[:, -1]))
    else:
        record["action_magnitude_mean"] = None
        record["action_magnitude_p95"] = None
        record["gripper_mean"] = None
    return record


def run_configuration(
    mode: str,
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    adapter: CosmosAdapter,
    output_root: Path,
    trace_writer: JsonlWriter,
    sampler: ResourceSampler,
) -> dict[str, Any]:
    adapter.closed_loop_mode = mode
    task_records = []
    config_traces = []
    config_start = time.monotonic_ns()
    for row in rows:
        config = config_for_row({**base, "model": {**base["model"], "closed_loop_mode": mode}}, row)
        env = ManifestLiberoEnvironment(row, int(base["evaluation"]["resolution"]), None)
        action_path = output_root / mode / "actions" / f"{row['label']}.npy"
        collected_writer = CollectingTraceWriter()
        try:
            record, traces = run_episode(
                adapter,
                env,
                config,
                row["instruction"],
                0,
                int(row["seed"]),
                collected_writer,
                action_trace_path=action_path,
            )
        finally:
            env.close()
        record = augment_action_metrics(record)
        record.update(
            {
                "configuration": mode,
                "label": row["label"],
                "suite": row["suite"],
                "task_name": row["task_name"],
                "instruction": row["instruction"],
                "perturbation_category": row["perturbation_category"],
                "variant_id": row["variant_id"],
                "variant_applied": row["variant_applied"],
                "init_state_index": row["init_state_index"],
                "seed": row["seed"],
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            }
        )
        for trace in collected_writer.rows:
            trace["configuration"] = mode
            trace["label"] = row["label"]
            trace_writer.write(trace)
        task_records.append(record)
        config_traces.extend(traces)
        print(
            f"[{mode}] {row['label']} success={record['success']} "
            f"steps={record['episode_steps']} requests={record['inference_count']}",
            flush=True,
        )
    elapsed_s = (time.monotonic_ns() - config_start) / 1e9
    visual_counts = {
        "fresh": sum(int(row.get("extra", {}).get("fresh_visual_request_count", 0)) for row in config_traces),
        "predicted": sum(int(row.get("extra", {}).get("predicted_visual_request_count", 0)) for row in config_traces),
        "cached": sum(int(row.get("extra", {}).get("cached_visual_request_count", 0)) for row in config_traces),
        "predict_correct": sum(
            int(row.get("extra", {}).get("predict_correct_request_count", 0)) for row in config_traces
        ),
        "rgb_preprocessing": sum(int(row.get("extra", {}).get("rgb_preprocessing_count", 0)) for row in config_traces),
    }
    return {
        "configuration": mode,
        "episodes": task_records,
        "success_rate": float(np.mean([bool(row["success"]) for row in task_records])),
        "successes": int(sum(bool(row["success"]) for row in task_records)),
        "episode_wall_time_s": float(sum(float(row["episode_wall_clock_time_s"]) for row in task_records)),
        "configuration_wall_time_s": elapsed_s,
        "total_control_steps": int(sum(int(row["episode_steps"]) for row in task_records)),
        "total_wam_requests": int(sum(int(row["inference_count"]) for row in task_records)),
        "vae_calls": int(sum(int(row.get("vae_encode_count", 0)) for row in config_traces)),
        "latency": summarize_latency(config_traces),
        "visual_counts": visual_counts,
        "traces": config_traces,
        "resource_samples_so_far": len(sampler.samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path, default=RAW_OUTPUT)
    parser.add_argument("--task-limit", type=int, default=6)
    args = parser.parse_args()

    if "so101" in str(CHECKPOINT).lower() or "finet" in str(CHECKPOINT).lower():
        raise ValueError("Phase 1 refuses SO101/finetuned checkpoint")
    actual_hash = sha256(CHECKPOINT)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint hash mismatch: {actual_hash}")
    if args.task_limit != 6:
        raise ValueError("Phase 1 is fixed to the requested six tasks")

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    base = load_yaml(SWEEP_CONFIG)
    rows = build_rows()
    install_libero_checkout(PRO_REPO, PRO_CONFIG)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    trace_path = args.raw_output / "inference_trace.jsonl"
    if trace_path.exists():
        raise FileExistsError(f"refusing to append to existing run: {trace_path}")
    trace_writer = JsonlWriter(trace_path)
    sampler = ResourceSampler(float(base["runtime"]["resource_interval_seconds"]))
    sampler.start()
    adapter = CosmosAdapter({**base["model"], "closed_loop_mode": "fresh"})
    config_results = []
    started_ns = time.time_ns()
    try:
        for mode in ("fresh", "alternate_speculative", "alternate_cache"):
            config_results.append(run_configuration(mode, rows, base, adapter, args.raw_output, trace_writer, sampler))
    finally:
        adapter.close()
        resources = sampler.stop()

    by_label = {
        row["label"]: {
            result["configuration"]: next(
                episode for episode in result["episodes"] if episode["label"] == row["label"]
            )
            for result in config_results
        }
        for row in rows
    }
    fresh_success = {label: bool(values["fresh"]["success"]) for label, values in by_label.items()}
    predicted_success = {
        label: bool(values["alternate_speculative"]["success"]) for label, values in by_label.items()
    }
    cache_success = {label: bool(values["alternate_cache"]["success"]) for label, values in by_label.items()}
    predicted_regressions = [label for label in by_label if fresh_success[label] and not predicted_success[label]]
    cache_regressions = [label for label in by_label if fresh_success[label] and not cache_success[label]]
    phase1_recommendation = (
        "NO-GO: speculative regression on a fresh-success task"
        if predicted_regressions
        else "GO-CANDIDATE: no fresh-success regression; proceed only after reviewing traces"
    )
    output = {
        "schema_version": 1,
        "experiment": "phase1_closed_loop_latent_reuse",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_hash,
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "denoising_steps": 1,
        "action_horizon": 16,
        "execution_prefix": 16,
        "task_count": len(rows),
        "task_selection": rows,
        "same_initial_state_across_configurations": True,
        "same_task_text_across_configurations": True,
        "value_used": False,
        "privileged_state_runtime_input": False,
        "new_offline_proxy_used": False,
        "new_layer_probe_used": False,
        "latent_l1_scheduler_used": False,
        "proprio_threshold_scheduler_used": False,
        "schedule": {
            "fresh": "FRESH every request",
            "alternate_speculative": "FRESH -> PREDICTED -> FRESH -> ...",
            "alternate_cache": "FRESH -> CACHE -> FRESH -> ...",
        },
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "configurations": config_results,
        "paired_success": {
            "fresh": fresh_success,
            "alternate_speculative": predicted_success,
            "alternate_cache": cache_success,
            "speculative_regressions_vs_fresh": predicted_regressions,
            "cache_regressions_vs_fresh": cache_regressions,
            "phase1_recommendation": phase1_recommendation,
        },
        "resources": resources,
        "trace_path": str(trace_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "recommendation": phase1_recommendation}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

"""P3 multi-task paired closed-loop validation.

Runs Fresh, simple predicted-latent reuse, and persistent Predict-Correct on
12 fixed LIBERO-PRO tasks and three fixed initial states/seeds.  Each process
owns one visible GPU and receives whole tasks so all configurations for a
task/state remain paired in the same artifact.
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
    MAX_STEPS,
    PRO_AUDIT,
    PRO_CONFIG,
    PRO_REPO,
    SWEEP_CONFIG,
)
from experiments.overnight.run_pick_place_init2_validation import (  # noqa: E402
    MODES,
    ReuseCapture,
    summarize_traces,
)
from runtime.runtime_metrics import JsonlWriter, ResourceSampler  # noqa: E402


TASKS = (
    ("pick_spatial", "libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", "task"),
    ("pick_spatial_center", "libero_spatial", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", "position"),
    ("pick_place", "libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket", "object"),
    ("pick_bbq", "libero_object", "pick_up_the_bbq_sauce_and_place_it_in_the_basket", "position"),
    ("open_close", "libero_goal", "open_the_middle_drawer_of_the_cabinet", "task"),
    ("open_top", "libero_goal", "open_the_top_drawer_and_put_the_bowl_inside", "position"),
    ("contact_switch", "libero_goal", "turn_on_the_stove", "task"),
    ("rack_place", "libero_goal", "put_the_wine_bottle_on_the_rack", "position"),
    ("multi_stage", "libero_10", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "environment"),
    ("multi_microwave", "libero_10", "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", "language"),
    ("living_multi", "libero_10", "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", "object"),
    ("study_book", "libero_10", "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", "position"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rows(task_indices: list[int], init_indices: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, Any]]:
    audit = json.loads(PRO_AUDIT.read_text(encoding="utf-8"))
    root = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/stage2_init")
    rows: list[dict[str, Any]] = []
    for task_index in task_indices:
        label, suite, task_name, category = TASKS[task_index]
        matches = [
            item for item in audit
            if item["suite"] == suite and item["task_name"] == task_name and item["category"] == category
        ]
        if len(matches) != 1 or not matches[0]["variant_applied"]:
            raise RuntimeError(f"invalid P3 audit match for {label}: {len(matches)}")
        asset = matches[0]
        init_path = root / f"{suite}_{category}" / f"{task_name}.five_states"
        if not init_path.is_file():
            raise FileNotFoundError(init_path)
        for init_index, seed in ((index, 195 + index) for index in init_indices):
            rows.append(
                {
                    "manifest_order": task_index * 3 + init_index,
                    "episode_key": hashlib.sha256(f"p3|{label}|init{init_index}|seed{seed}".encode()).hexdigest(),
                    "domain": "libero_pro",
                    "perturbation_category": category,
                    "variant_id": f"{category}:seed28",
                    "variant_applied": True,
                    "suite": suite,
                    "task_uid": f"{suite}:{task_name}",
                    "task_name": task_name,
                    "label": f"{label}_init{init_index}",
                    "base_label": label,
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
                    "max_steps": MAX_STEPS[suite],
                    "libero_repo": str(PRO_REPO),
                    "libero_config_path": str(PRO_CONFIG),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--task-indices", default=None, help="comma-separated task indices; default is shard split")
    parser.add_argument("--init-indices", default="0,1,2")
    parser.add_argument("--modes", default=",".join(MODES))
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if "so101" in str(CHECKPOINT).lower() or "finet" in str(CHECKPOINT).lower():
        raise ValueError("refusing SO101/finetuned checkpoint")
    actual_hash = sha256(CHECKPOINT)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint hash mismatch: {actual_hash}")

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    requested_modes = tuple(value.strip() for value in args.modes.split(",") if value.strip())
    if not requested_modes or any(mode not in MODES for mode in requested_modes):
        raise ValueError(f"unknown mode in {requested_modes}; allowed={MODES}")
    init_indices = tuple(int(value) for value in args.init_indices.split(",") if value.strip())
    if not init_indices or any(index < 0 or index > 4 for index in init_indices):
        raise ValueError(f"init indices must be in [0,4], got {init_indices}")
    task_indices = (
        [int(value) for value in args.task_indices.split(",") if value.strip()]
        if args.task_indices is not None
        else list(range(args.shard_index, len(TASKS), args.shard_count))
    )
    rows = build_rows(task_indices, init_indices)
    base = load_yaml(SWEEP_CONFIG)
    install_libero_checkout(PRO_REPO, PRO_CONFIG)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    trace_path = args.raw_output / "inference_trace.jsonl"
    trace_writer = JsonlWriter(trace_path)
    sampler = ResourceSampler(float(base["runtime"]["resource_interval_seconds"]))
    sampler.start()
    adapter = CosmosAdapter({**base["model"], "closed_loop_mode": "fresh", "predict_correct_steps": 2})
    records: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    captures: list[str] = []
    started_ns = time.time_ns()
    try:
        for mode in requested_modes:
            for row in rows:
                adapter.closed_loop_mode = mode
                config = config_for_row(
                    {**base, "model": {**base["model"], "closed_loop_mode": mode, "predict_correct_steps": 2}},
                    row,
                )
                env = ManifestLiberoEnvironment(row, int(base["evaluation"]["resolution"]), None)
                collected = []

                class Writer:
                    def write(self, item):
                        collected.append(item)

                capture = ReuseCapture(row, mode)
                action_path = args.raw_output / mode / "actions" / f"{row['label']}_seed{row['seed']}.npy"
                try:
                    record, traces = run_episode(
                        adapter,
                        env,
                        config,
                        row["instruction"],
                        0,
                        int(row["seed"]),
                        Writer(),
                        action_trace_path=action_path,
                        inference_capture_callback=capture,
                    )
                finally:
                    env.close()
                capture_path = capture.save(
                    args.raw_output / mode / "shadow_capture" / f"{row['label']}_seed{row['seed']}.npz"
                )
                captures.append(str(capture_path))
                for trace in collected:
                    trace["configuration"] = mode
                    trace["task_label"] = row["base_label"]
                    trace["init_state_index"] = row["init_state_index"]
                    trace["seed"] = row["seed"]
                    trace_writer.write(trace)
                all_traces.extend(collected)
                record = {
                    **record,
                    "configuration": mode,
                    "task_label": row["base_label"],
                    "task_name": row["task_name"],
                    "suite": row["suite"],
                    "instruction": row["instruction"],
                    "init_state_index": row["init_state_index"],
                    "seed": row["seed"],
                    "checkpoint_sha256": actual_hash,
                    "executed_actions_path": str(action_path),
                    "reuse_capture_path": str(capture_path),
                    "reuse_capture_count": len(capture.request_index),
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "mode": mode,
                            "task": row["label"],
                            "success": bool(record.get("success")),
                            "reason": record.get("termination_reason"),
                            "steps": record.get("episode_steps"),
                            "requests": record.get("inference_count"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        adapter.close()
        resources = sampler.stop()

    outcomes: dict[str, Any] = {}
    for mode in requested_modes:
        subset = [row for row in records if row["configuration"] == mode]
        outcomes[mode] = {
            "episodes": len(subset),
            "successes": sum(bool(row.get("success")) for row in subset),
            "success_rate": float(np.mean([bool(row.get("success")) for row in subset])) if subset else None,
            "by_task": {
                label: {
                    "episodes": len([row for row in subset if row["task_label"] == label]),
                    "successes": sum(bool(row.get("success")) for row in subset if row["task_label"] == label),
                }
                for label, *_ in TASKS
                if any(row["task_label"] == label for row in subset)
            },
        }
    output = {
        "schema_version": 1,
        "experiment": "p3_predict_correct_multitask_closed_loop",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": actual_hash,
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_count_total": len(TASKS),
        "task_indices_in_shard": task_indices,
        "tasks_in_shard": [TASKS[index] for index in task_indices],
        "initial_state_seed_pairs": [{"init_state_index": i, "seed": 195 + i} for i in init_indices],
        "configurations": requested_modes,
        "denoising_steps": {"fresh": 1, "predicted_reuse": 1, "predict_correct": 2},
        "action_horizon": 16,
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "outcomes": outcomes,
        "trace_summary": summarize_traces(all_traces),
        "records": records,
        "capture_paths": captures,
        "raw_trace_path": str(trace_path),
        "resources": resources,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "episodes": len(records)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

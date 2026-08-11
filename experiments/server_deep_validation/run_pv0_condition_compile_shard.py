#!/usr/bin/env python3
"""Fixed-interface 40-task native-PV0 condition-compile validation.

S4 does not search blocks or interfaces.  It uses the already fixed native
fresh-visual prefix pathway and compares it with an intentionally mismatched
fresh visual prefix from a different frozen task.  That contrast tests whether
the current physical visual condition, rather than generic extra model work,
is responsible for fidelity to F1.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pv0_overnight_common import (
    DEFAULT_DATASET_STATS,
    DEFAULT_T5_EMBEDDINGS,
    ORIGINAL_CHECKPOINT,
    ORIGINAL_CHECKPOINT_SHA256,
    atomic_write_json,
    build_model,
    checkpoint_contract,
    configure_libero,
    final_gpu_snapshot,
    pair_metrics,
    read_jsonl,
    record_path_for_anchor,
    recovery_metrics,
    route_contract,
    run_route,
    set_up_cuda,
    swapped_visual_observation,
    valid_anchor_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True, help="Exclusive anchor index")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--checkpoint", type=Path, default=ORIGINAL_CHECKPOINT)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_DATASET_STATS)
    parser.add_argument("--t5-embeddings", type=Path, default=DEFAULT_T5_EMBEDDINGS)
    return parser.parse_args()


def load_request(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    episode = torch.load(Path(entry["collection_episode"]), map_location="cpu", weights_only=False)
    if episode.get("episode_key") != entry["episode_key"]:
        raise RuntimeError("collection episode key mismatch")
    requests = episode.get("requests", [])
    source_index = int(entry["source_request_index"])
    target_index = int(entry["target_request_index"])
    source, target = requests[source_index], requests[target_index]
    if target.get("state_key") != entry["state_key"]:
        raise RuntimeError("state-index provenance mismatch")
    if int(target["control_step"]) - int(source["control_step"]) != 16:
        raise RuntimeError("condition-compile anchor is not physically aligned")
    return source, target


def render_observation(entry: dict[str, Any], *, resolution: int = 256) -> Any:
    from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment
    from experiments.libero_harness import extract_observation
    from experiments.progressive_wam.run_p2_oracle import restore

    _, target = load_request(entry)
    environment = ManifestLiberoEnvironment(entry, resolution, None)
    try:
        environment.reset()
        sim_state = np.asarray(target["sim_state"], dtype=np.float64)
        restore(environment, sim_state)
        return extract_observation(environment.env.regenerate_obs_from_state(sim_state), flip_vertical=True)
    finally:
        environment.close()


def write_failure(path: Path, anchor: dict[str, Any], error: BaseException, gpu: dict[str, Any]) -> None:
    target = anchor["target"]
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "experiment": "S4_fixed_native_condition_compile",
            "status": "FAIL_ANCHOR",
            "anchor_index": int(anchor["anchor_index"]),
            "state_key": target["state_key"],
            "task_uid": target["task_uid"],
            "split": target["split"],
            "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "error": f"{type(error).__name__}:{error}",
            "traceback": traceback.format_exc(limit=8),
            "gpu": gpu,
            "failed_at_ns": time.time_ns(),
        },
    )


def main() -> None:
    args = parse_args()
    if args.start_index < 0 or args.end_index <= args.start_index:
        raise ValueError("require 0 <= start-index < end-index")
    checkpoint = checkpoint_contract(args.checkpoint)
    anchors = read_jsonl(args.anchors)
    if args.end_index > len(anchors):
        raise ValueError(f"end-index={args.end_index} exceeds {len(anchors)} anchors")
    assigned = anchors[args.start_index : args.end_index]
    if any(int(anchor["anchor_index"]) != args.start_index + offset for offset, anchor in enumerate(assigned)):
        raise RuntimeError("anchor index must be consecutive and ordered")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [anchor for anchor in assigned if not valid_anchor_record(record_path_for_anchor(args.output_dir, anchor), anchor)]
    summary_path = args.output_dir / f"summary_{args.start_index:03d}_{args.end_index:03d}.json"
    if not pending:
        atomic_write_json(
            summary_path,
            {
                "schema_version": 1,
                "experiment": "S4_fixed_native_condition_compile",
                "status": "PASS_ALREADY_COMPLETE",
                "start_index": args.start_index,
                "end_index": args.end_index,
                "assigned": len(assigned),
                "skipped_existing": len(assigned),
                **checkpoint,
                "denoising_steps": 1,
                "value_used": False,
                "privileged_runtime_state_used": False,
                "scheduler_or_threshold_used": False,
            },
        )
        return

    gpu = set_up_cuda(float(args.memory_fraction))
    configure_libero(pending[0]["target"])
    cfg, stats, model = build_model(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        t5_embeddings=args.t5_embeddings,
    )
    completed = 0
    skipped = len(assigned) - len(pending)
    started = time.monotonic()
    try:
        for local_index, anchor in enumerate(pending, start=1):
            output_path = record_path_for_anchor(args.output_dir, anchor)
            if valid_anchor_record(output_path, anchor):
                skipped += 1
                continue
            target = anchor["target"]
            shuffled = anchor["shuffled_visual_source"]
            try:
                target_source, _ = load_request(target)
                target_observation = render_observation(target)
                shuffled_observation = render_observation(shuffled)
                previous = torch.from_numpy(
                    np.asarray(target_source["generated_latent"], dtype=np.float16).astype(np.float32)
                ).cuda()
                actions: dict[str, np.ndarray] = {}
                route_metrics: dict[str, dict[str, Any]] = {}
                with torch.inference_mode():
                    for route in ("P1", "F1", "PV0"):
                        actions[route], route_metrics[route] = run_route(
                            route,
                            cfg=cfg,
                            model=model,
                            stats=stats,
                            observation=target_observation,
                            instruction=target["instruction"],
                            seed=int(target["seed"]),
                            previous=previous,
                        )
                    actions["PV0_shuffled"], route_metrics["PV0_shuffled"] = run_route(
                        "PV0",
                        cfg=cfg,
                        model=model,
                        stats=stats,
                        observation=swapped_visual_observation(target_observation, shuffled_observation),
                        instruction=target["instruction"],
                        seed=int(target["seed"]),
                        previous=previous,
                    )
                p1_to_f1 = pair_metrics(actions["P1"], actions["F1"])
                correct_to_f1 = pair_metrics(actions["PV0"], actions["F1"])
                shuffled_to_f1 = pair_metrics(actions["PV0_shuffled"], actions["F1"])
                payload = {
                    "schema_version": 1,
                    "experiment": "S4_fixed_native_condition_compile",
                    "status": "PASS",
                    "anchor_index": int(anchor["anchor_index"]),
                    "state_key": target["state_key"],
                    "episode_key": target["episode_key"],
                    "task_uid": target["task_uid"],
                    "task_name": target["task_name"],
                    "suite": target["suite"],
                    "split": target["split"],
                    "init_state_index": int(target["init_state_index"]),
                    "seed": int(target["seed"]),
                    "shuffled_visual_source": {
                        key: shuffled[key]
                        for key in ("state_key", "episode_key", "task_uid", "task_name", "suite", "split")
                    },
                    "shuffled_control_contract": {
                        "selection": "frozen cyclic different-task camera prefix",
                        "target_proprio_preserved": True,
                        "target_instruction_preserved": True,
                        "policy_receives_only_camera_and_proprio": True,
                    },
                    **checkpoint,
                    "denoising_steps": 1,
                    "route_contract": {
                        "F1": route_contract("F1"),
                        "P1": route_contract("P1"),
                        "PV0": route_contract("PV0"),
                    },
                    "offline_restore_used_for_reproducibility_only": True,
                    "value_used": False,
                    "privileged_runtime_state_used": False,
                    "scheduler_or_threshold_used": False,
                    "actions": {route: values.tolist() for route, values in actions.items()},
                    "metrics": {
                        "p1_to_f1": p1_to_f1,
                        "pv0_correct_to_f1": correct_to_f1,
                        "pv0_shuffled_to_f1": shuffled_to_f1,
                        "correct_recovery": recovery_metrics(p1_to_f1, correct_to_f1),
                        "correct_beats_shuffled": bool(
                            correct_to_f1["mean_step_l2"] < shuffled_to_f1["mean_step_l2"]
                        ),
                        "correct_minus_shuffled_mean_step_l2": float(
                            correct_to_f1["mean_step_l2"] - shuffled_to_f1["mean_step_l2"]
                        ),
                    },
                    "route_metrics": route_metrics,
                    "gpu": final_gpu_snapshot(),
                    "shared_gpu_latency_caveat": True,
                    "completed_at_ns": time.time_ns(),
                }
                atomic_write_json(output_path, payload)
                completed += 1
                print(
                    json.dumps(
                        {
                            "event": "anchor_complete",
                            "anchor_index": anchor["anchor_index"],
                            "completed": completed,
                            "pending": len(pending),
                            "elapsed_s": round(time.monotonic() - started, 2),
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                write_failure(output_path, anchor, error, final_gpu_snapshot())
                raise
    finally:
        model = None
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "experiment": "S4_fixed_native_condition_compile",
        "status": "PASS",
        "start_index": args.start_index,
        "end_index": args.end_index,
        "assigned": len(assigned),
        "completed": completed,
        "skipped_existing": skipped,
        **checkpoint,
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_used": False,
        "scheduler_or_threshold_used": False,
        "gpu": final_gpu_snapshot(),
        "elapsed_s": time.monotonic() - started,
        "finished_at_ns": time.time_ns(),
    }
    atomic_write_json(summary_path, payload)
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()

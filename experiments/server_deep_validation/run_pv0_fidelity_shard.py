#!/usr/bin/env python3
"""Run a small, resumable shard of the 3,801-state native-PV0 fidelity test.

Every state evaluates the frozen one-step F1, P1, and native PV0 paths from
the same restored physical observation and same stochastic seed.  The driver
does not expose simulator state to Cosmos; state restoration only recreates the
camera/proprio observation recorded in the Foundation V2 collection.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    ROUTES,
    atomic_write_json,
    build_model,
    checkpoint_contract,
    configure_libero,
    final_gpu_snapshot,
    pair_metrics,
    read_jsonl,
    record_path_for_state,
    recovery_metrics,
    route_contract,
    run_route,
    set_up_cuda,
    valid_state_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--end-index", type=int, required=True, help="Exclusive global-state index")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--checkpoint", type=Path, default=ORIGINAL_CHECKPOINT)
    parser.add_argument("--dataset-stats", type=Path, default=DEFAULT_DATASET_STATS)
    parser.add_argument("--t5-embeddings", type=Path, default=DEFAULT_T5_EMBEDDINGS)
    return parser.parse_args()


def source_and_target(entry: dict[str, Any], episode_cache: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    requests = episode_cache.get("requests", [])
    source_index = int(entry["source_request_index"])
    target_index = int(entry["target_request_index"])
    if not 0 <= source_index < target_index < len(requests):
        raise IndexError(f"invalid request indices in {entry['state_key']}")
    source, target = requests[source_index], requests[target_index]
    if target.get("state_key") != entry["state_key"]:
        raise RuntimeError("state-index provenance mismatch")
    if int(target.get("control_step", -999)) - int(source.get("control_step", -999)) != 16:
        raise RuntimeError("state is no longer physically aligned by a 16-action prefix")
    return source, target


def write_failure(output_path: Path, entry: dict[str, Any], error: BaseException, gpu: dict[str, Any]) -> None:
    atomic_write_json(
        output_path,
        {
            "schema_version": 1,
            "experiment": "S1_pv0_full_scale_fidelity",
            "status": "FAIL_STATE",
            "global_index": int(entry["global_index"]),
            "state_key": entry["state_key"],
            "episode_key": entry["episode_key"],
            "task_uid": entry["task_uid"],
            "split": entry["split"],
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
    entries = read_jsonl(args.state_index)
    if args.end_index > len(entries):
        raise ValueError(f"end-index={args.end_index} exceeds {len(entries)} indexed states")
    assigned = entries[args.start_index : args.end_index]
    if any(int(entry["global_index"]) != args.start_index + offset for offset, entry in enumerate(assigned)):
        raise RuntimeError("state index must be consecutive and globally ordered")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [entry for entry in assigned if not valid_state_record(record_path_for_state(args.output_dir, entry), entry)]
    summary_path = args.output_dir / f"summary_{args.start_index:05d}_{args.end_index:05d}.json"
    if not pending:
        atomic_write_json(
            summary_path,
            {
                "schema_version": 1,
                "experiment": "S1_pv0_full_scale_fidelity",
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
    configure_libero(pending[0])
    from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment
    from experiments.libero_harness import extract_observation
    from experiments.progressive_wam.run_p2_oracle import restore

    cfg, stats, model = build_model(
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        t5_embeddings=args.t5_embeddings,
    )
    current_episode_key: str | None = None
    current_episode: dict[str, Any] | None = None
    environment: ManifestLiberoEnvironment | None = None
    completed = 0
    skipped = len(assigned) - len(pending)
    failures = 0
    started = time.monotonic()
    try:
        for local_index, entry in enumerate(pending, start=1):
            output_path = record_path_for_state(args.output_dir, entry)
            if valid_state_record(output_path, entry):
                skipped += 1
                continue
            try:
                if current_episode_key != entry["episode_key"]:
                    if environment is not None:
                        environment.close()
                        environment = None
                    current_episode = torch.load(
                        Path(entry["collection_episode"]), map_location="cpu", weights_only=False
                    )
                    if current_episode.get("episode_key") != entry["episode_key"]:
                        raise RuntimeError("collection episode key mismatch")
                    if current_episode.get("checkpoint_sha256") != ORIGINAL_CHECKPOINT_SHA256:
                        raise RuntimeError("collection episode checkpoint mismatch")
                    environment = ManifestLiberoEnvironment(entry, 256, None)
                    current_episode_key = str(entry["episode_key"])
                if current_episode is None or environment is None:
                    raise RuntimeError("episode/environment cache unexpectedly absent")
                source, target = source_and_target(entry, current_episode)
                environment.reset()
                sim_state = np.asarray(target["sim_state"], dtype=np.float64)
                restore(environment, sim_state)
                observation = extract_observation(
                    environment.env.regenerate_obs_from_state(sim_state), flip_vertical=True
                )
                previous = torch.from_numpy(
                    np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)
                ).cuda()
                # Rotate the legal call order, preserving deterministic seed semantics
                # while avoiding an accidental fixed warm-cache ordering artifact.
                shift = int(entry["global_index"]) % len(ROUTES)
                order = ROUTES[shift:] + ROUTES[:shift]
                actions: dict[str, np.ndarray] = {}
                route_metrics: dict[str, dict[str, Any]] = {}
                with torch.inference_mode():
                    for route in order:
                        actions[route], route_metrics[route] = run_route(
                            route,
                            cfg=cfg,
                            model=model,
                            stats=stats,
                            observation=observation,
                            instruction=entry["instruction"],
                            seed=int(entry["seed"]),
                            previous=previous,
                        )
                p1_to_f1 = pair_metrics(actions["P1"], actions["F1"])
                pv0_to_f1 = pair_metrics(actions["PV0"], actions["F1"])
                payload = {
                    "schema_version": 1,
                    "experiment": "S1_pv0_full_scale_fidelity",
                    "status": "PASS",
                    "global_index": int(entry["global_index"]),
                    "state_key": entry["state_key"],
                    "episode_key": entry["episode_key"],
                    "task_uid": entry["task_uid"],
                    "task_name": entry["task_name"],
                    "suite": entry["suite"],
                    "split": entry["split"],
                    "init_state_index": int(entry["init_state_index"]),
                    "seed": int(entry["seed"]),
                    "request_index": int(entry["request_index"]),
                    "control_step": int(entry["control_step"]),
                    "collection_episode": entry["collection_episode"],
                    **checkpoint,
                    "denoising_steps": 1,
                    "route_contract": {route: route_contract(route) for route in ROUTES},
                    "execution_order": list(order),
                    "offline_restore_used_for_reproducibility_only": True,
                    "value_used": False,
                    "privileged_runtime_state_used": False,
                    "scheduler_or_threshold_used": False,
                    "actions": {route: actions[route].tolist() for route in ROUTES},
                    "metrics": {
                        "p1_to_f1": p1_to_f1,
                        "pv0_to_f1": pv0_to_f1,
                        "recovery": recovery_metrics(p1_to_f1, pv0_to_f1),
                    },
                    "route_metrics": route_metrics,
                    "gpu": final_gpu_snapshot(),
                    "completed_at_ns": time.time_ns(),
                    "shared_gpu_latency_caveat": True,
                }
                atomic_write_json(output_path, payload)
                completed += 1
                if local_index % 5 == 0 or local_index == len(pending):
                    print(
                        json.dumps(
                            {
                                "event": "state_complete",
                                "global_index": entry["global_index"],
                                "completed": completed,
                                "pending": len(pending),
                                "elapsed_s": round(time.monotonic() - started, 2),
                            }
                        ),
                        flush=True,
                    )
            except Exception as error:
                failures += 1
                write_failure(output_path, entry, error, final_gpu_snapshot())
                # CUDA OOM leaves the context unreliable.  Let the supervisor
                # save the log/snapshot, raise its required-VRAM estimate, and
                # requeue on a safer GPU.
                if "out of memory" in str(error).lower() or "cuda" in str(error).lower():
                    raise
                raise
    finally:
        if environment is not None:
            environment.close()
        model = None
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "experiment": "S1_pv0_full_scale_fidelity",
        "status": "PASS" if failures == 0 else "FAILURES_RECORDED",
        "start_index": args.start_index,
        "end_index": args.end_index,
        "assigned": len(assigned),
        "completed": completed,
        "skipped_existing": skipped,
        "failures": failures,
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

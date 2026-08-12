#!/usr/bin/env python3
"""E12-A/B paired shadow collection: P1 (+frozen score), F1 reference, PV0 correction.

One row per legal H=16-aligned physical state.  All three routes see the same
task, simulator state, observation, proprio, instruction, seed, checkpoint and
denoise count; only the route differs.  Simulator state reconstructs the camera
observation and never enters the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.esp.run_e2_e3_pilot_shard import load_request, render
from experiments.p1_semantic_verify.common import (
    ACTION_GEOMETRY_FEATURES,
    BLOCKS,
    action_geometry,
    array_hash,
    frozen_score,
    load_frozen_model,
    observation_hash,
    predicted_condition,
    risk,
    run_route,
)
from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS,
    DEFAULT_T5_EMBEDDINGS,
    ORIGINAL_CHECKPOINT,
    atomic_write_json,
    build_model,
    checkpoint_contract,
    configure_libero,
    read_jsonl,
    route_contract,
    set_up_cuda,
)


def collect(entry: dict[str, Any], cfg: Any, model: Any, stats: Any, frozen: dict[str, Any],
            *, e4_variant_diagnostic: bool = False) -> dict[str, Any]:
    source, target = load_request(entry)
    gap = int(target["control_step"]) - int(source["control_step"])
    if gap != 16:
        raise RuntimeError(f"illegal temporal gap {gap} for {entry['state_key']}")
    observation = render(entry, target)
    previous = torch.from_numpy(
        np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)
    ).cuda()

    kwargs = dict(
        cfg=cfg, model=model, stats=stats, observation=observation,
        instruction=entry["instruction"], seed=int(entry["seed"]), previous_generated=previous,
    )
    p1 = run_route("P1", capture_blocks=BLOCKS, **kwargs)
    f1 = run_route("F1", **kwargs)
    pv0 = run_route("PV0", **kwargs)

    score = frozen_score(frozen, p1["internal"], p1["actions"])
    risk_p1 = risk(p1["actions"], f1["actions"])
    risk_pv0 = risk(pv0["actions"], f1["actions"])
    geometry = action_geometry(p1["actions"])

    row: dict[str, Any] = {
        "state_id": entry["state_key"],
        "task_id": entry["task_uid"],
        "episode_id": entry["episode_key"],
        "init_id": int(entry["init_state_index"]),
        "seed": int(entry["seed"]),
        "split": entry["e12_split"],
        "control_step": int(entry["control_step"]),
        "source_index": int(entry["source_request_index"]),
        "target_index": int(entry["target_request_index"]),
        "temporal_gap_actions": gap,
        "s_p1": score,
        "risk_p1_full": risk_p1["full"],
        "risk_p1_first": risk_p1["first"],
        "risk_p1_first4": risk_p1["first4"],
        "risk_p1_gripper_disagreement": risk_p1["gripper_sign_disagreement"],
        "risk_pv0_full": risk_pv0["full"],
        "risk_pv0_first": risk_pv0["first"],
        "risk_pv0_first4": risk_pv0["first4"],
        "pv0_gain": risk_p1["full"] - risk_pv0["full"],
        "pv0_relative_recovery": (
            float(1.0 - risk_pv0["full"] / risk_p1["full"]) if risk_p1["full"] > 1e-12 else None
        ),
        "action_p1_sha256": array_hash(p1["actions"]),
        "action_f1_sha256": array_hash(f1["actions"]),
        "action_pv0_sha256": array_hash(pv0["actions"]),
        "observation_hash": observation_hash(observation),
        "proprio_hash": array_hash(np.asarray(observation.proprio)),
        "instruction_hash": hashlib.sha256(entry["instruction"].encode()).hexdigest(),
        "p1_cuda_ms": p1["cuda_ms"],
        "f1_cuda_ms": f1["cuda_ms"],
        "pv0_cuda_ms": pv0["cuda_ms"],
        "valid": True,
        "invalid_reason": None,
    }
    for name in ACTION_GEOMETRY_FEATURES:
        row[name] = float(geometry[name])
    for index, value in enumerate(risk_p1["per_joint_mean_abs"]):
        row[f"risk_p1_joint{index}"] = float(value)
    row.update({name: float(value) for name, value in p1["internal"].items()})

    if e4_variant_diagnostic:
        # Discovery-only diagnostic. The E4/E11-A collection variant conditions on
        # the RAW prior generated latent (no slot 6/7 -> 2/3 move), which is not
        # the deployed P1 route. Recorded to attribute a possible NO-GO between
        # task transfer and route-variant change; never a candidate policy here.
        from experiments.semantic_risk.run_semantic_shard import run_route as e4_route

        variant_action, _, variant_internal, _ = e4_route(
            cfg, model, stats, observation, entry["instruction"], int(entry["seed"]),
            previous=previous, blocks=BLOCKS,
        )
        variant_action = np.asarray(variant_action, dtype=np.float32).reshape(16, 7)
        row["s_p1_e4variant"] = frozen_score(frozen, variant_internal, variant_action)
        row["risk_p1_e4variant_full"] = float(risk(variant_action, f1["actions"])["full"])

    finite = all(
        np.isfinite(row[key]) for key in ("s_p1", "risk_p1_full", "risk_pv0_full")
    )
    if not finite:
        row["valid"] = False
        row["invalid_reason"] = "non_finite_score_or_risk"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--e4-variant-diagnostic", action="store_true",
                        help="discovery only: also score the E4/E11-A raw-latent collection variant")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard id")

    entries = read_jsonl(args.bank)
    selected = [entry for index, entry in enumerate(entries) if index % args.num_shards == args.shard_id]
    if args.limit is not None:
        selected = selected[: args.limit]

    frozen = load_frozen_model()
    contract = checkpoint_contract(ORIGINAL_CHECKPOINT)
    gpu = set_up_cuda(args.memory_fraction)
    configure_libero(selected[0])
    cfg, stats, model = build_model(
        checkpoint=ORIGINAL_CHECKPOINT, dataset_stats=DEFAULT_DATASET_STATS, t5_embeddings=DEFAULT_T5_EMBEDDINGS
    )

    rows: list[dict[str, Any]] = []
    if args.output.exists():
        try:
            rows = list(json.loads(args.output.read_text(encoding="utf-8")).get("rows", []))
        except (OSError, json.JSONDecodeError):
            rows = []
    done = {row["state_id"] for row in rows}
    pending = [entry for entry in selected if entry["state_key"] not in done]

    header = {
        "schema_version": 1,
        "experiment": "E12A_shadow",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        **contract,
        "gpu": gpu,
        "denoising_steps": 1,
        "value_used": False,
        "finetuning_used": False,
        "privileged_runtime_state_used": False,
        "scheduler_or_threshold_used": False,
        "hidden_activation_patch_used": False,
        "additional_model_forward_for_score": False,
        "e4_variant_diagnostic": bool(args.e4_variant_diagnostic),
        "deployed_p1_route": "predicted_reuse: predicted future slots 6/7 moved into current slots 2/3",
        "frozen_score_checksum": frozen["checksum_sha256"],
        "route_contract": {route: route_contract(route) for route in ("F1", "P1", "PV0")},
        "expected": len(selected),
    }
    try:
        for position, entry in enumerate(pending, len(rows) + 1):
            try:
                rows.append(collect(entry, cfg, model, stats, frozen,
                                    e4_variant_diagnostic=args.e4_variant_diagnostic))
            except Exception as error:  # keep failures visible, never silently dropped
                rows.append({
                    "state_id": entry["state_key"], "task_id": entry["task_uid"],
                    "episode_id": entry["episode_key"], "split": entry["e12_split"],
                    "valid": False, "invalid_reason": f"{type(error).__name__}:{error}",
                })
            atomic_write_json(args.output, {**header, "status": "RUNNING", "rows": rows})
            print(json.dumps({"shard": args.shard_id, "completed": position, "expected": len(selected)}), flush=True)
    finally:
        model = None
        torch.cuda.empty_cache()

    atomic_write_json(args.output, {**header, "status": "PASS", "rows": rows})
    print(json.dumps({"output": str(args.output), "states": len(rows),
                      "valid": sum(bool(row.get("valid")) for row in rows)}))


if __name__ == "__main__":
    main()

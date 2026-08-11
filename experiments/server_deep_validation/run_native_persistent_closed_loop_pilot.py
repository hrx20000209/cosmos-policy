#!/usr/bin/env python3
"""Run one baseline-eligible, paired, native-PV0 closed-loop LIBERO episode.

This driver is intentionally small.  It runs exactly one task/init/mode per
process so F1, P1, and PV0 can be placed on separate GPUs without changing the
physical initial state used by any route.  PV0 is the existing native Cosmos
``persistent_visual_correction`` path: it starts each post-bootstrap request
from the prior predicted latent, encodes a causal fresh visual prefix, and
installs that prefix before the only denoiser forward.

The policy gets only regular camera/proprio observations from ``run_episode``.
It uses the frozen pre-finetune checkpoint, no value signal, no privileged
simulator state, no threshold, and fixed one-step denoising.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.cosmos_adapter import CosmosAdapter  # noqa: E402
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (  # noqa: E402
    ManifestLiberoEnvironment,
    config_for_row,
    install_libero_checkout,
    load_jsonl,
)
from experiments.libero_harness import load_yaml, run_episode  # noqa: E402
from wam_runtime.fixed_policy_harness import ORIGINAL_COSMOS_CHECKPOINT_SHA256  # noqa: E402


CHECKPOINT = Path("/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
# The candidate was selected from an independent Fresh-only collection that
# uses this exact manifest row (including BDDL, instruction, init state, and
# checkpoint), rather than from the older P3 task family where similarly named
# tasks can carry a different perturbation asset.
BASELINE_ELIGIBLE_TASK = "put_the_bowl_on_the_stove"
HELDOUT_BASELINE_ELIGIBLE_TASK = (
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_"
    "put_the_chocolate_pudding_to_the_right_of_the_plate"
)
BASELINE_ELIGIBLE_ROWS = {
    (BASELINE_ELIGIBLE_TASK, 0, 195): {
        "episode_key": "969c6e03b29ae497e09bb08e9e13932d34fd7f17ca40c0854dc1e2e3d9ff06e4",
        "record": Path(
            "/data/rxhuang/wam_full_scale_server/queue_a/f1_collection/group06/"
            "episode_969c6e03b29ae497e09bb08e9e13932d34fd7f17ca40c0854dc1e2e3d9ff06e4.pt"
        ),
    },
    (BASELINE_ELIGIBLE_TASK, 1, 196): {
        "episode_key": "6702c4c0fa238d102fd3bf992ba6875b91694555eebe8395f7acc23f2496cd39",
        "record": Path(
            "/data/rxhuang/wam_full_scale_server/queue_a/f1_collection/group03/"
            "episode_6702c4c0fa238d102fd3bf992ba6875b91694555eebe8395f7acc23f2496cd39.pt"
        ),
    },
    (HELDOUT_BASELINE_ELIGIBLE_TASK, 1, 196): {
        "episode_key": "09b74eb0218102b7529a23346bf053161af14d89f14c0de2cfa88fe3f28edd14",
        "record": Path(
            "/data/rxhuang/wam_full_scale_server/queue_a/f1_collection/group00/"
            "episode_09b74eb0218102b7529a23346bf053161af14d89f14c0de2cfa88fe3f28edd14.pt"
        ),
    },
    (HELDOUT_BASELINE_ELIGIBLE_TASK, 2, 197): {
        "episode_key": "29844c13638a6e701f589ad1c0863dfced45463d5b7cc22d8f85cd7c5d461a69",
        "record": Path(
            "/data/rxhuang/wam_full_scale_server/queue_a/f1_collection/group03/"
            "episode_29844c13638a6e701f589ad1c0863dfced45463d5b7cc22d8f85cd7c5d461a69.pt"
        ),
    },
}
DEFAULT_BASELINE_ELIGIBLE_INIT = 0
DEFAULT_BASELINE_ELIGIBLE_SEED = 195
MODES = ("fresh", "predicted_reuse", "native_persistent")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def route_contract(mode: str) -> dict[str, Any]:
    common = {
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
    }
    if mode == "fresh":
        return {
            **common,
            "bootstrap_and_followups": "full fresh camera preprocessing and VAE encoding",
        }
    if mode == "predicted_reuse":
        return {
            **common,
            "bootstrap": "full fresh camera preprocessing and VAE encoding",
            "followups": "prior generated visual latent only; camera preprocessing skipped",
        }
    if mode == "native_persistent":
        return {
            **common,
            "bootstrap": "full fresh camera preprocessing and VAE encoding",
            "followups": "prior generated latent base + native fresh causal visual-prefix assimilation",
            "native_interface": "get_action:persistent_visual_correction_prefix_frames",
            "fresh_visual_prefix_frames": 13,
            "fresh_visual_arrival_denoiser_forward": 0,
            "hidden_activation_patch_used": False,
            "fresh_prefix_oracle_used": False,
        }
    raise ValueError(f"unknown mode {mode!r}")


def expected_visual_modes(mode: str, request_count: int) -> list[str]:
    if request_count <= 0:
        return []
    followup = {
        "fresh": "fresh",
        "predicted_reuse": "predicted",
        "native_persistent": "native_persistent",
    }[mode]
    return ["fresh", *([followup] * (request_count - 1))]


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_row(args: argparse.Namespace) -> dict[str, Any]:
    matches = [
        row
        for row in load_jsonl(args.manifest)
        if row["task_name"] == args.task_name
        and int(row["init_state_index"]) == args.init_state_index
        and int(row["seed"]) == args.seed
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one manifest row for "
            f"task={args.task_name!r}, init={args.init_state_index}, seed={args.seed}; got {len(matches)}"
        )
    return matches[0]


def validate_baseline_eligibility(row: dict[str, Any], record_path: Path) -> dict[str, Any]:
    """Verify the independent, exact-row Fresh success used for task selection.

    This record is audit provenance only; its observations or simulator state
    are never passed to the runtime policy in this pilot.
    """
    if not record_path.is_file():
        raise FileNotFoundError(f"missing baseline-eligibility record: {record_path}")
    prior = torch.load(record_path, map_location="cpu", weights_only=False)
    expected = {
        "episode_key": row["episode_key"],
        "task_name": row["task_name"],
        "instruction": row["instruction"],
        "init_state_index": int(row["init_state_index"]),
        "seed": int(row["seed"]),
        "denoising_steps": 1,
        "checkpoint_sha256": ORIGINAL_COSMOS_CHECKPOINT_SHA256,
    }
    for key, value in expected.items():
        if prior.get(key) != value:
            raise RuntimeError(
                f"baseline-eligibility provenance mismatch for {key}: "
                f"expected {value!r}, got {prior.get(key)!r}"
            )
    if not bool(prior.get("success")):
        raise RuntimeError("baseline-eligibility Fresh record was not successful")
    if bool(prior.get("value_used")) or bool(prior.get("privileged_runtime_state_used")):
        raise RuntimeError("baseline-eligibility record violates no-value/no-privileged-state contract")
    return {
        "source": str(record_path),
        "selection_rule": "exact manifest row previously succeeded under Fresh-only original-checkpoint collection",
        "source_success": bool(prior["success"]),
        "source_control_steps": int(prior["control_steps"]),
        "source_termination_reason": str(prior["termination_reason"]),
        "source_value_used": bool(prior["value_used"]),
        "source_privileged_runtime_state_used": bool(prior["privileged_runtime_state_used"]),
    }


def validate_trace_contract(mode: str, traces: list[dict[str, Any]]) -> dict[str, Any]:
    visual_modes = [str((trace.get("extra") or {}).get("visual_input_mode")) for trace in traces]
    forwards = [int(trace.get("denoiser_forward_count", -1)) for trace in traces]
    expected = expected_visual_modes(mode, len(traces))
    if visual_modes != expected:
        raise RuntimeError(f"visual route mismatch: expected {expected}, got {visual_modes}")
    if any(count != 1 for count in forwards):
        raise RuntimeError(f"one-denoise contract failed: {forwards}")
    if mode == "native_persistent" and len(traces) > 1:
        native_count = sum(
            int((trace.get("extra") or {}).get("native_persistent_request_count", 0))
            for trace in traces
        )
        if native_count != len(traces) - 1:
            raise RuntimeError(
                f"native persistent request count mismatch: expected {len(traces) - 1}, got {native_count}"
            )
    return {
        "visual_input_modes": visual_modes,
        "visual_input_mode_counts": dict(Counter(visual_modes)),
        "denoiser_forward_counts": forwards,
        "trace_contract": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/server_deep_validation/manifests/full_scale_40_task.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/server_deep_validation/server_sweep.yaml",
    )
    parser.add_argument("--task-name", default=BASELINE_ELIGIBLE_TASK)
    parser.add_argument("--init-state-index", type=int, default=DEFAULT_BASELINE_ELIGIBLE_INIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASELINE_ELIGIBLE_SEED)
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.trace_output is not None and args.trace_output.exists():
        raise FileExistsError(f"refusing to overwrite {args.trace_output}")
    if not 0.05 <= args.memory_fraction <= 1.0:
        raise ValueError("memory-fraction must be in [0.05, 1.0]")
    selection_key = (args.task_name, args.init_state_index, args.seed)
    eligibility = BASELINE_ELIGIBLE_ROWS.get(selection_key)
    if eligibility is None:
        raise ValueError(
            "this small pilot is deliberately restricted to exact baseline-eligible task/init/seed rows; "
            f"got {selection_key!r}"
        )
    if "so101" in str(CHECKPOINT).lower() or "finetun" in str(CHECKPOINT).lower():
        raise ValueError("refusing finetuned/SO101 checkpoint")
    if sha256(CHECKPOINT) != ORIGINAL_COSMOS_CHECKPOINT_SHA256:
        raise RuntimeError("original pre-finetune checkpoint SHA256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    device = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
    torch.cuda.empty_cache()
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)

    row = selected_row(args)
    if row["episode_key"] != eligibility["episode_key"]:
        raise RuntimeError("manifest episode key does not match the baseline-eligible provenance")
    baseline_eligibility = validate_baseline_eligibility(row, eligibility["record"])
    if int(row["denoising_steps"]) != 1:
        raise RuntimeError(f"manifest violates PV0 one-denoise contract: {row['denoising_steps']}")
    install_libero_checkout(Path(row["libero_repo"]), Path(row["libero_config_path"]))
    base = load_yaml(args.config)
    config = config_for_row(
        {**base, "model": {**base["model"], "closed_loop_mode": args.mode}},
        row,
    )
    if args.max_steps is not None:
        config["evaluation"]["max_steps"] = min(int(args.max_steps), int(row["max_steps"]))
    if int(config["denoising"]["steps"]) != 1 or config["denoising"]["scheduler"] != "fixed":
        raise RuntimeError("pilot must use fixed one-denoise policy")

    traces: list[dict[str, Any]] = []

    class Writer:
        def write(self, value: dict[str, Any]) -> None:
            traces.append(value)

    environment: ManifestLiberoEnvironment | None = None
    adapter: CosmosAdapter | None = None
    started = time.time_ns()
    try:
        environment = ManifestLiberoEnvironment(row, int(config["evaluation"]["resolution"]), None)
        adapter = CosmosAdapter(config["model"])
        action_path = args.output.parent / "actions" / f"{args.mode}_{row['episode_key']}.npy"
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
        request_latencies = [float(trace["total_policy_request_latency_ms"]) for trace in traces]
        model_latencies = [
            float(((trace.get("extra") or {}).get("non_overlapping_stage_ms") or {}).get("model_generate_inclusive_ms", 0.0))
            for trace in traces
        ]
        free_after, _ = torch.cuda.mem_get_info(device)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "experiment": "native_persistent_closed_loop_pilot",
            "pilot_scale": "one_baseline_eligible_task_x_one_init_x_one_mode",
            "mode": args.mode,
            "route_contract": route_contract(args.mode),
            "baseline_eligibility": baseline_eligibility,
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
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": ORIGINAL_COSMOS_CHECKPOINT_SHA256,
            "finetuning_used": False,
            "value_used": False,
            "privileged_runtime_state_input": False,
            "adaptive_scheduler_used": False,
            "threshold_used": False,
            "hidden_activation_patch_used": False,
            "fresh_prefix_oracle_used": False,
            "gpu": {
                "visible_device_index": int(device),
                "device_name": torch.cuda.get_device_name(device),
                "memory_fraction_cap": float(args.memory_fraction),
                "memory_free_before_mib": float(free_before / 2**20),
                "memory_free_after_mib": float(free_after / 2**20),
                "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
                "total_mib": float(total_bytes / 2**20),
            },
            "record": record,
            "trace_contract": trace_contract,
            "request_latency_ms": percentile_summary(request_latencies),
            "model_generate_inclusive_ms": percentile_summary(model_latencies),
            "trace_count": len(traces),
            "action_path": str(action_path),
            "shared_gpu_latency_caveat": True,
            "started_at_ns": started,
            "finished_at_ns": time.time_ns(),
        }
        atomic_write(args.output, payload)
        if args.trace_output is not None:
            atomic_write(args.trace_output, {"schema_version": 1, "traces": traces})
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "success": bool(record.get("success")),
                    "termination_reason": record.get("termination_reason"),
                    "requests": len(traces),
                    "output": str(args.output),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        if environment is not None:
            environment.close()
        if adapter is not None:
            adapter.close()


if __name__ == "__main__":
    main()

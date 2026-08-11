#!/usr/bin/env python3
"""Audit the deliberately small native-PV0 closed-loop pilot.

Only result files in the pilot root are considered.  Earlier attempts placed
under attempt-specific subdirectories are preserved as provenance but excluded
from the success gate.  The audit verifies routing and the frozen-policy
contract before it reports any success statistic.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODES = ("fresh", "predicted_reuse", "native_persistent")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def expected_visual_modes(mode: str, count: int) -> list[str]:
    if count <= 0:
        return []
    followup = {
        "fresh": "fresh",
        "predicted_reuse": "predicted",
        "native_persistent": "native_persistent",
    }[mode]
    return ["fresh", *([followup] * (count - 1))]


def validate_result(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode"))
    if mode not in MODES:
        raise ValueError(f"{path}: unsupported mode {mode!r}")
    if payload.get("experiment") != "native_persistent_closed_loop_pilot":
        raise ValueError(f"{path}: unexpected experiment label")
    contract = payload.get("route_contract") or {}
    if int(contract.get("denoising_steps", -1)) != 1:
        raise ValueError(f"{path}: not one-denoise")
    for key in ("value_used", "privileged_runtime_state_input", "adaptive_scheduler_used", "threshold_used"):
        if bool(contract.get(key)):
            raise ValueError(f"{path}: contract violates {key}=False")
    if bool(payload.get("hidden_activation_patch_used")) or bool(payload.get("fresh_prefix_oracle_used")):
        raise ValueError(f"{path}: oracle or hidden patch was used")
    eligibility = payload.get("baseline_eligibility") or {}
    if not bool(eligibility.get("source_success")):
        raise ValueError(f"{path}: missing exact-row Fresh-success eligibility")
    if bool(eligibility.get("source_value_used")) or bool(eligibility.get("source_privileged_runtime_state_used")):
        raise ValueError(f"{path}: eligibility provenance violates frozen-policy contract")
    trace_path = path.with_name(f"{path.stem}_traces.json")
    trace_payload = load_json(trace_path)
    traces = trace_payload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError(f"{trace_path}: missing traces")
    visual_modes = [str((trace.get("extra") or {}).get("visual_input_mode")) for trace in traces]
    if visual_modes != expected_visual_modes(mode, len(traces)):
        raise ValueError(f"{path}: visual route mismatch {visual_modes}")
    forwards = [int(trace.get("denoiser_forward_count", -1)) for trace in traces]
    if any(count != 1 for count in forwards):
        raise ValueError(f"{path}: one-denoise route mismatch {forwards}")
    if mode == "native_persistent":
        if contract.get("native_interface") != "get_action:persistent_visual_correction_prefix_frames":
            raise ValueError(f"{path}: missing native persistent interface")
        if int(contract.get("fresh_visual_prefix_frames", -1)) != 13:
            raise ValueError(f"{path}: unexpected fresh prefix length")
        if int(contract.get("fresh_visual_arrival_denoiser_forward", -1)) != 0:
            raise ValueError(f"{path}: fresh prefix did not arrive before the only forward")
        native_count = sum(
            int((trace.get("extra") or {}).get("native_persistent_request_count", 0))
            for trace in traces
        )
        if native_count != len(traces) - 1:
            raise ValueError(f"{path}: native request count mismatch")
    manifest = payload.get("manifest_row") or {}
    record = payload.get("record") or {}
    if int(manifest.get("denoising_steps", -1)) != 1:
        raise ValueError(f"{path}: manifest does not specify one denoise")
    if len(traces) != int(record.get("inference_count", -1)):
        raise ValueError(f"{path}: trace and record inference counts differ")
    warm = traces[1:]
    return {
        "path": str(path),
        "mode": mode,
        "task_uid": str(manifest["task_uid"]),
        "task_name": str(manifest["task_name"]),
        "split": str(manifest["split"]),
        "episode_key": str(manifest["episode_key"]),
        "init_state_index": int(manifest["init_state_index"]),
        "seed": int(manifest["seed"]),
        "success": bool(record.get("success")),
        "termination_reason": str(record.get("termination_reason")),
        "episode_steps": int(record.get("episode_steps", 0)),
        "inference_count": len(traces),
        "visual_input_mode_counts": dict(Counter(visual_modes)),
        "warm_request_latency_ms": summary([float(trace["total_policy_request_latency_ms"]) for trace in warm]),
        "warm_model_generate_inclusive_ms": summary(
            [
                float(
                    ((trace.get("extra") or {}).get("non_overlapping_stage_ms") or {}).get(
                        "model_generate_inclusive_ms", 0.0
                    )
                )
                for trace in warm
            ]
        ),
        "baseline_eligibility": eligibility,
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/modular_wam_runtime/native_persistent_closed_loop_pilot"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    candidate_payloads = []
    for path in sorted(args.input_dir.glob("*_init*.json")):
        if path.name.endswith("_traces.json"):
            continue
        payload = load_json(path)
        # Prior audit outputs can share the ``*_init*.json`` naming pattern;
        # only raw route outputs belong in the current paired audit.
        if payload.get("experiment") == "native_persistent_closed_loop_pilot":
            candidate_payloads.append((path, payload))
    if not candidate_payloads:
        raise FileNotFoundError(f"no root pilot result files under {args.input_dir}")
    candidates = [path for path, _ in candidate_payloads]
    records = [validate_result(path, payload) for path, payload in candidate_payloads]
    groups: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["episode_key"], record["init_state_index"], record["seed"])
        if record["mode"] in groups[key]:
            raise ValueError(f"duplicate route {record['mode']} for paired group {key}")
        groups[key][record["mode"]] = record
    incomplete = [key for key, group in groups.items() if set(group) != set(MODES)]
    if incomplete:
        raise ValueError(f"incomplete paired groups: {incomplete}")

    paired = []
    for key, group in sorted(groups.items()):
        fresh_success = bool(group["fresh"]["success"])
        p1_success = bool(group["predicted_reuse"]["success"])
        pv0_success = bool(group["native_persistent"]["success"])
        paired.append(
            {
                "episode_key": key[0],
                "task_uid": group["fresh"]["task_uid"],
                "task_name": group["fresh"]["task_name"],
                "split": group["fresh"]["split"],
                "init_state_index": key[1],
                "seed": key[2],
                "fresh_success": fresh_success,
                "p1_success": p1_success,
                "pv0_success": pv0_success,
                "pv0_preserves_fresh_success": (not fresh_success) or pv0_success,
                "p1_preserves_fresh_success": (not fresh_success) or p1_success,
            }
        )

    task_count = len({item["task_uid"] for item in paired})
    all_fresh_success = all(item["fresh_success"] for item in paired)
    all_pv0_preserve = all(item["pv0_preserves_fresh_success"] for item in paired)
    if len(paired) >= 2 and all_fresh_success and all_pv0_preserve:
        status = (
            "M3_CLOSED_LOOP_GO_CANDIDATE_TWO_TASK_SMALL_SCALE"
            if task_count >= 2
            else "M3_CLOSED_LOOP_GO_CANDIDATE_ONE_TASK_TWO_INIT"
        )
    else:
        status = "M3_CLOSED_LOOP_INCONCLUSIVE"
    payload = {
        "schema_version": 1,
        "experiment": "native_persistent_closed_loop_pilot_audit",
        "status": status,
        "reason": (
            "Every included group uses an exact-row Fresh-success provenance record and passes the native one-denoise route contract. "
            "This remains a deliberately small paired pilot, not a task-general success-rate estimate."
        ),
        "included_result_files": [str(path) for path in candidates],
        "excluded_attempt_directories": [
            str(path) for path in sorted(args.input_dir.glob("attempt*")) if path.is_dir()
        ],
        "route_records": records,
        "paired_groups": paired,
        "summary": {
            "paired_group_count": len(paired),
            "task_count": task_count,
            "splits": dict(Counter(item["split"] for item in paired)),
            "successes_by_mode": {
                mode: int(sum(bool(group[mode]["success"]) for group in groups.values()))
                for mode in MODES
            },
            "fresh_success_count": int(sum(item["fresh_success"] for item in paired)),
            "pv0_preserves_fresh_success_count": int(sum(item["pv0_preserves_fresh_success"] for item in paired)),
            "p1_preserves_fresh_success_count": int(sum(item["p1_preserves_fresh_success"] for item in paired)),
            "pv0_success_when_p1_fails_count": int(
                sum((not item["p1_success"]) and item["pv0_success"] for item in paired)
            ),
            "pv0_recovers_fresh_success_vs_p1_count": int(
                sum(
                    item["fresh_success"] and (not item["p1_success"]) and item["pv0_success"]
                    for item in paired
                )
            ),
            "timing_interpretation": (
                "Warm-request timings are retained per route, but are not cross-route latency evidence because routes ran on different shared GPUs. "
                "Use the paired same-worker native-condition microbatch for the server model-cost latency claim."
            ),
        },
    }
    atomic_write(args.output, payload)
    print(
        json.dumps(
            {
                "status": status,
                "paired_groups": len(paired),
                "tasks": task_count,
                "successes_by_mode": payload["summary"]["successes_by_mode"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Continuously audit the frozen overnight PV0 Phase-A experiment.

This program is deliberately analysis-only.  It never invokes Cosmos, never
reads a value prediction, and never places simulator state into a policy input.
It turns independently restartable S1/S4 artifacts and the audited Foundation
V2 F1/P1/PP/PF/FF collection into task-balanced Phase-A evidence and a frozen
GO/NO-GO decision.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from pv0_overnight_common import (
    ORIGINAL_CHECKPOINT_SHA256,
    atomic_write_json,
    json_safe,
    numeric_summary,
    read_jsonl,
)


FIDELITY_EPSILON = 0.05
S1_MAX_P95_MEAN_STEP_L2 = 0.15
S1_MIN_TASK_BALANCED_FIDELITY = 0.90
S1_MIN_TASK_BALANCED_P1_WIN = 0.90
S1_MIN_TASK_BALANCED_GRIPPER_MATCH = 0.95
S4_MIN_ALL_TASK_CORRECT_WIN = 0.70
S4_MIN_HELDOUT_CORRECT_WIN = 0.60
S2_MIN_HELDOUT_COSINE = 0.50
S2_MIN_HELDOUT_PROJECTION = 0.50
S2_MIN_HELDOUT_PF_WIN = 0.60
PREFIXES = (4, 8, 12, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-index", type=Path, required=True)
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--s4-dir", type=Path, required=True)
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("/data/rxhuang/wam_full_scale_server/queue_b/ablation"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true", help="Only refresh S2 decomposition.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def bool_summary(values: Iterable[Any]) -> dict[str, Any]:
    numeric = [float(bool(value)) for value in values if value is not None]
    result = numeric_summary(numeric)
    result["fraction"] = result.pop("mean")
    return result


def task_balanced(rows: Iterable[dict[str, Any]], extractor: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = extractor(row)
        if finite(value):
            grouped[str(row["task_uid"])].append(float(value))
    per_task = {task: float(np.mean(values)) for task, values in grouped.items() if values}
    return {
        "tasks": len(per_task),
        "per_task": per_task,
        "summary": numeric_summary(per_task.values()),
    }


def split_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output = {"discovery": [], "validation": [], "heldout": []}
    for row in rows:
        split = str(row.get("split"))
        output.setdefault(split, []).append(row)
    return output


def path_metric(row: Mapping[str, Any], *keys: str) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def summarize_s1_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def value(route: str, field: str) -> Callable[[dict[str, Any]], Any]:
        return lambda row: path_metric(row, "metrics", route, field)

    def prefix(route: str, k: int, field: str) -> Callable[[dict[str, Any]], Any]:
        return lambda row: path_metric(row, "metrics", route, "prefixes", str(k), field)

    def recovery(k: int | None, field: str) -> Callable[[dict[str, Any]], Any]:
        keys = ("metrics", "recovery") if k is None else ("metrics", "recovery", "prefixes", str(k))
        return lambda row: path_metric(row, *keys, field)

    result: dict[str, Any] = {
        "states": len(rows),
        "tasks": len({row["task_uid"] for row in rows}),
        "p1_to_f1_mean_step_l2": numeric_summary(value("p1_to_f1", "mean_step_l2")(row) for row in rows),
        "pv0_to_f1_mean_step_l2": numeric_summary(value("pv0_to_f1", "mean_step_l2")(row) for row in rows),
        "pv0_fidelity_at_epsilon": bool_summary(
            value("pv0_to_f1", "mean_step_l2")(row) <= FIDELITY_EPSILON
            for row in rows
            if finite(value("pv0_to_f1", "mean_step_l2")(row))
        ),
        "pv0_beats_p1": bool_summary(recovery(None, "native_beats_p1")(row) for row in rows),
        "pv0_first_gripper_sign_match": bool_summary(
            value("pv0_to_f1", "first_gripper_sign_match")(row) for row in rows
        ),
        "task_balanced": {
            "p1_to_f1_mean_step_l2": task_balanced(rows, value("p1_to_f1", "mean_step_l2")),
            "pv0_to_f1_mean_step_l2": task_balanced(rows, value("pv0_to_f1", "mean_step_l2")),
            "pv0_fidelity_at_epsilon": task_balanced(
                rows,
                lambda row: float(value("pv0_to_f1", "mean_step_l2")(row) <= FIDELITY_EPSILON)
                if finite(value("pv0_to_f1", "mean_step_l2")(row))
                else None,
            ),
            "pv0_beats_p1": task_balanced(
                rows,
                lambda row: float(bool(recovery(None, "native_beats_p1")(row)))
                if recovery(None, "native_beats_p1")(row) is not None
                else None,
            ),
            "pv0_first_gripper_sign_match": task_balanced(
                rows,
                lambda row: float(bool(value("pv0_to_f1", "first_gripper_sign_match")(row)))
                if value("pv0_to_f1", "first_gripper_sign_match")(row) is not None
                else None,
            ),
        },
        "execution_prefixes": {},
    }
    for k in PREFIXES:
        result["execution_prefixes"][str(k)] = {
            "p1_to_f1_mean_step_l2": numeric_summary(prefix("p1_to_f1", k, "mean_step_l2")(row) for row in rows),
            "pv0_to_f1_mean_step_l2": numeric_summary(prefix("pv0_to_f1", k, "mean_step_l2")(row) for row in rows),
            "pv0_fidelity_at_epsilon": bool_summary(
                prefix("pv0_to_f1", k, "mean_step_l2")(row) <= FIDELITY_EPSILON
                for row in rows
                if finite(prefix("pv0_to_f1", k, "mean_step_l2")(row))
            ),
            "pv0_beats_p1": bool_summary(recovery(k, "native_beats_p1")(row) for row in rows),
            "pv0_first_gripper_sign_match": bool_summary(
                prefix("pv0_to_f1", k, "first_gripper_sign_match")(row) for row in rows
            ),
            "task_balanced": {
                "pv0_fidelity_at_epsilon": task_balanced(
                    rows,
                    lambda row, prefix_k=k: float(
                        prefix("pv0_to_f1", prefix_k, "mean_step_l2")(row) <= FIDELITY_EPSILON
                    )
                    if finite(prefix("pv0_to_f1", prefix_k, "mean_step_l2")(row))
                    else None,
                ),
                "pv0_beats_p1": task_balanced(
                    rows,
                    lambda row, prefix_k=k: float(bool(recovery(prefix_k, "native_beats_p1")(row)))
                    if recovery(prefix_k, "native_beats_p1")(row) is not None
                    else None,
                ),
                "pv0_first_gripper_sign_match": task_balanced(
                    rows,
                    lambda row, prefix_k=k: float(
                        bool(prefix("pv0_to_f1", prefix_k, "first_gripper_sign_match")(row))
                    )
                    if prefix("pv0_to_f1", prefix_k, "first_gripper_sign_match")(row) is not None
                    else None,
                ),
            },
        }
    return result


def s1_records(state_index: list[dict[str, Any]], directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    expected = {int(entry["global_index"]): entry for entry in state_index}
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[int] = set()
    for path in sorted(directory.glob("state_*.json")):
        payload = read_json(path)
        if payload is None:
            errors.append(f"unreadable:{path.name}")
            continue
        index = payload.get("global_index")
        if not isinstance(index, int) or index not in expected:
            errors.append(f"unexpected_index:{path.name}")
            continue
        if index in seen:
            errors.append(f"duplicate_index:{index}")
            continue
        seen.add(index)
        entry = expected[index]
        if payload.get("status") != "PASS":
            errors.append(f"not_pass:{index}:{payload.get('status')}")
            continue
        if payload.get("state_key") != entry["state_key"]:
            errors.append(f"state_key_mismatch:{index}")
            continue
        if payload.get("checkpoint_sha256") != ORIGINAL_CHECKPOINT_SHA256:
            errors.append(f"checkpoint_mismatch:{index}")
            continue
        if int(payload.get("denoising_steps", -1)) != 1:
            errors.append(f"denoise_mismatch:{index}")
            continue
        if payload.get("value_used") is not False or payload.get("privileged_runtime_state_used") is not False:
            errors.append(f"policy_contract_mismatch:{index}")
            continue
        records.append(payload)
    missing = sorted(set(expected) - {int(record["global_index"]) for record in records})
    errors.extend(f"missing:{index}" for index in missing[:100])
    return records, errors


def summarize_s1(state_index: list[dict[str, Any]], records: list[dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    by_split = split_rows(records)
    expected_by_split = split_rows(state_index)
    summaries = {split: summarize_s1_split(by_split.get(split, [])) for split in expected_by_split}
    coverage = {
        split: {
            "expected_states": len(expected_by_split[split]),
            "completed_states": len(by_split.get(split, [])),
            "expected_tasks": len({row["task_uid"] for row in expected_by_split[split]}),
            "completed_tasks": len({row["task_uid"] for row in by_split.get(split, [])}),
        }
        for split in expected_by_split
    }
    complete = len(records) == len(state_index) and all(
        coverage[split]["completed_tasks"] == coverage[split]["expected_tasks"] for split in coverage
    )
    heldout = summaries["heldout"]
    heldout_task = heldout["task_balanced"]
    criteria = {
        "complete_3801_state_coverage": complete,
        "heldout_task_balanced_fidelity_at_0_05_at_least_0_90": (
            heldout_task["pv0_fidelity_at_epsilon"]["summary"]["mean"] is not None
            and heldout_task["pv0_fidelity_at_epsilon"]["summary"]["mean"] >= S1_MIN_TASK_BALANCED_FIDELITY
        ),
        "heldout_task_balanced_pv0_beats_p1_at_least_0_90": (
            heldout_task["pv0_beats_p1"]["summary"]["mean"] is not None
            and heldout_task["pv0_beats_p1"]["summary"]["mean"] >= S1_MIN_TASK_BALANCED_P1_WIN
        ),
        "heldout_p95_mean_step_l2_at_most_0_15": (
            heldout["pv0_to_f1_mean_step_l2"]["p95"] is not None
            and heldout["pv0_to_f1_mean_step_l2"]["p95"] <= S1_MAX_P95_MEAN_STEP_L2
        ),
        "heldout_task_balanced_first_gripper_sign_match_at_least_0_95": (
            heldout_task["pv0_first_gripper_sign_match"]["summary"]["mean"] is not None
            and heldout_task["pv0_first_gripper_sign_match"]["summary"]["mean"] >= S1_MIN_TASK_BALANCED_GRIPPER_MATCH
        ),
    }
    status = "GO" if all(criteria.values()) else ("INCOMPLETE" if not complete else "NO_GO")
    return {
        "schema_version": 1,
        "experiment": "S1_pv0_full_scale_fidelity_analysis",
        "status": status,
        "frozen_tolerance": {"mean_step_l2_epsilon": FIDELITY_EPSILON},
        "criteria": criteria,
        "coverage": coverage,
        "splits": summaries,
        "records": len(records),
        "expected_records": len(state_index),
        "audit_error_count": len(errors),
        "audit_errors": errors[:200],
        "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_used": False,
        "scheduler_or_threshold_used": False,
    }


def summarize_s3(state_index: list[dict[str, Any]], records: list[dict[str, Any]], s1: Mapping[str, Any]) -> dict[str, Any]:
    by_split = split_rows(records)
    expected_by_split = split_rows(state_index)
    coverage_complete = len(records) == len(state_index)
    splits: dict[str, Any] = {}
    for split, rows in by_split.items():
        split_summary: dict[str, Any] = {
            "states": len(rows),
            "tasks": len({row["task_uid"] for row in rows}),
            "prefixes": {},
        }
        for k in PREFIXES:
            def metric(row: Mapping[str, Any], route: str, field: str) -> Any:
                return path_metric(row, "metrics", route, "prefixes", str(k), field)

            def recovery(row: Mapping[str, Any], field: str) -> Any:
                return path_metric(row, "metrics", "recovery", "prefixes", str(k), field)

            pv0_error = [metric(row, "pv0_to_f1", "mean_step_l2") for row in rows]
            split_summary["prefixes"][str(k)] = {
                "p1_to_f1_mean_step_l2": numeric_summary(metric(row, "p1_to_f1", "mean_step_l2") for row in rows),
                "pv0_to_f1_mean_step_l2": numeric_summary(pv0_error),
                "pv0_fidelity_at_epsilon": bool_summary(
                    value <= FIDELITY_EPSILON for value in pv0_error if finite(value)
                ),
                "pv0_beats_p1": bool_summary(recovery(row, "native_beats_p1") for row in rows),
                "first_gripper_sign_match": bool_summary(
                    metric(row, "pv0_to_f1", "first_gripper_sign_match") for row in rows
                ),
                "task_balanced": {
                    "pv0_fidelity_at_epsilon": task_balanced(
                        rows,
                        lambda row: float(metric(row, "pv0_to_f1", "mean_step_l2") <= FIDELITY_EPSILON)
                        if finite(metric(row, "pv0_to_f1", "mean_step_l2"))
                        else None,
                    ),
                    "pv0_beats_p1": task_balanced(
                        rows,
                        lambda row: float(bool(recovery(row, "native_beats_p1")))
                        if recovery(row, "native_beats_p1") is not None
                        else None,
                    ),
                    "first_gripper_sign_match": task_balanced(
                        rows,
                        lambda row: float(bool(metric(row, "pv0_to_f1", "first_gripper_sign_match")))
                        if metric(row, "pv0_to_f1", "first_gripper_sign_match") is not None
                        else None,
                    ),
                },
            }
        splits[split] = split_summary
    heldout = splits.get("heldout", {"prefixes": {}})
    criteria: dict[str, bool] = {"complete_3801_state_coverage": coverage_complete}
    for k in PREFIXES:
        item = heldout["prefixes"].get(str(k), {})
        task = item.get("task_balanced", {})
        fidelity = path_metric(task, "pv0_fidelity_at_epsilon", "summary", "mean")
        wins = path_metric(task, "pv0_beats_p1", "summary", "mean")
        grip = path_metric(task, "first_gripper_sign_match", "summary", "mean")
        criteria[f"heldout_k{k}_task_balanced_fidelity_at_least_0_90"] = finite(fidelity) and fidelity >= S1_MIN_TASK_BALANCED_FIDELITY
        criteria[f"heldout_k{k}_task_balanced_pv0_beats_p1_at_least_0_90"] = finite(wins) and wins >= S1_MIN_TASK_BALANCED_P1_WIN
        criteria[f"heldout_k{k}_task_balanced_gripper_match_at_least_0_95"] = finite(grip) and grip >= S1_MIN_TASK_BALANCED_GRIPPER_MATCH
    status = "GO" if all(criteria.values()) else ("INCOMPLETE" if not coverage_complete else "NO_GO")
    return {
        "schema_version": 1,
        "experiment": "S3_execution_prefix_alignment",
        "status": status,
        "fixed_prefix_lengths": list(PREFIXES),
        "frozen_tolerance": {"mean_step_l2_epsilon": FIDELITY_EPSILON},
        "criteria": criteria,
        "coverage": {
            split: {
                "expected_states": len(expected_by_split[split]),
                "completed_states": len(by_split.get(split, [])),
                "expected_tasks": len({row["task_uid"] for row in expected_by_split[split]}),
                "completed_tasks": len({row["task_uid"] for row in by_split.get(split, [])}),
            }
            for split in expected_by_split
        },
        "splits": splits,
        "source_s1_status": s1.get("status"),
        "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_used": False,
        "scheduler_or_threshold_used": False,
    }


def raw_ablation_rows(root: Path, expected_keys: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("state_*.pt")):
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            errors.append(f"unreadable:{path}:{type(error).__name__}")
            continue
        state_key = str(raw.get("state_key", ""))
        if state_key in seen:
            errors.append(f"duplicate:{state_key}")
            continue
        seen.add(state_key)
        if state_key not in expected_keys:
            errors.append(f"unexpected:{state_key}")
            continue
        if raw.get("checkpoint_sha256") != ORIGINAL_CHECKPOINT_SHA256:
            errors.append(f"checkpoint:{state_key}")
            continue
        if raw.get("value_used") is not False or raw.get("privileged_runtime_state_used") is not False:
            errors.append(f"contract:{state_key}")
            continue
        metrics = raw.get("metrics") or {}
        innovation = metrics.get("action_innovation") or {}
        action_to_f1 = metrics.get("action_to_F1") or {}
        try:
            rows.append(
                {
                    "state_key": state_key,
                    "task_uid": str(raw["task_uid"]),
                    "split": str(raw["split"]),
                    "feedback_target_cosine": innovation.get("feedback_target_cosine"),
                    "feedback_target_projection": innovation.get("feedback_target_projection"),
                    "feedback_target_sign_agreement": innovation.get("feedback_target_sign_agreement"),
                    "solver_target_cosine": innovation.get("solver_target_cosine"),
                    "solver_target_projection": innovation.get("solver_target_projection"),
                    "p1_to_f1_l2": path_metric(action_to_f1, "P1", "l2"),
                    "pp_to_f1_l2": path_metric(action_to_f1, "PP", "l2"),
                    "pf_to_f1_l2": path_metric(action_to_f1, "PF", "l2"),
                    "did_to_f1_l2": innovation.get("did_to_f1_l2"),
                    "did_less_error_than_pf": innovation.get("did_less_error_than_pf"),
                    "did_less_error_than_pp": innovation.get("did_less_error_than_pp"),
                    "p1_first_action_l2": path_metric(action_to_f1, "P1", "first_action_l2"),
                    "pf_first_action_l2": path_metric(action_to_f1, "PF", "first_action_l2"),
                    "p1_gripper_sign_match": path_metric(action_to_f1, "P1", "gripper_sign_match"),
                    "pf_gripper_sign_match": path_metric(action_to_f1, "PF", "gripper_sign_match"),
                }
            )
        except KeyError:
            errors.append(f"missing_metrics:{state_key}")
    missing = expected_keys - {row["state_key"] for row in rows}
    errors.extend(f"missing:{key}" for key in sorted(missing)[:100])
    return rows, errors


def summarize_s2_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        return bool_summary(predicate(row) for row in rows)

    outputs: dict[str, Any] = {
        "states": len(rows),
        "tasks": len({row["task_uid"] for row in rows}),
        "feedback_target_cosine": numeric_summary(row["feedback_target_cosine"] for row in rows),
        "feedback_target_projection": numeric_summary(row["feedback_target_projection"] for row in rows),
        "feedback_target_sign_agreement": numeric_summary(row["feedback_target_sign_agreement"] for row in rows),
        "solver_target_cosine": numeric_summary(row["solver_target_cosine"] for row in rows),
        "solver_target_projection": numeric_summary(row["solver_target_projection"] for row in rows),
        "p1_to_f1_l2": numeric_summary(row["p1_to_f1_l2"] for row in rows),
        "pf_to_f1_l2": numeric_summary(row["pf_to_f1_l2"] for row in rows),
        "pf_beats_p1": rate(
            lambda row: finite(row["pf_to_f1_l2"])
            and finite(row["p1_to_f1_l2"])
            and float(row["pf_to_f1_l2"]) < float(row["p1_to_f1_l2"])
        ),
        "did_less_error_than_pf": bool_summary(row["did_less_error_than_pf"] for row in rows),
        "task_balanced": {},
    }
    for field in (
        "feedback_target_cosine",
        "feedback_target_projection",
        "feedback_target_sign_agreement",
        "solver_target_cosine",
        "solver_target_projection",
        "p1_to_f1_l2",
        "pf_to_f1_l2",
    ):
        outputs["task_balanced"][field] = task_balanced(rows, lambda row, name=field: row[name])
    outputs["task_balanced"]["pf_beats_p1"] = task_balanced(
        rows,
        lambda row: float(
            finite(row["pf_to_f1_l2"])
            and finite(row["p1_to_f1_l2"])
            and float(row["pf_to_f1_l2"]) < float(row["p1_to_f1_l2"])
        ),
    )
    p1 = np.asarray([float(row["p1_to_f1_l2"]) for row in rows if finite(row["p1_to_f1_l2"])])
    tails: dict[str, Any] = {}
    for fraction in (0.05, 0.10):
        if p1.size:
            threshold = float(np.quantile(p1, 1.0 - fraction))
            tail = [row for row in rows if finite(row["p1_to_f1_l2"]) and float(row["p1_to_f1_l2"]) >= threshold]
            tails[f"top_{int(fraction * 100):02d}_pct_p1_error"] = {
                "threshold": threshold,
                "states": len(tail),
                "pf_beats_p1": bool_summary(
                    finite(row["pf_to_f1_l2"])
                    and float(row["pf_to_f1_l2"]) < float(row["p1_to_f1_l2"])
                    for row in tail
                ),
                "feedback_target_cosine": numeric_summary(row["feedback_target_cosine"] for row in tail),
                "feedback_target_projection": numeric_summary(row["feedback_target_projection"] for row in tail),
            }
    outputs["p1_error_tails"] = tails
    return outputs


def summarize_s2(state_index: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    raw, errors = raw_ablation_rows(root, {entry["state_key"] for entry in state_index})
    by_split = split_rows(raw)
    expected = split_rows(state_index)
    summaries = {split: summarize_s2_split(by_split.get(split, [])) for split in expected}
    complete = len(raw) == len(state_index) and not any(error.startswith("missing:") for error in errors)
    heldout = summaries["heldout"]["task_balanced"]
    cosine = heldout["feedback_target_cosine"]["summary"]["median"]
    projection = heldout["feedback_target_projection"]["summary"]["median"]
    pf_wins = heldout["pf_beats_p1"]["summary"]["median"]
    criteria = {
        "complete_3801_state_coverage": complete,
        "heldout_task_balanced_feedback_cosine_median_at_least_0_50": finite(cosine) and cosine >= S2_MIN_HELDOUT_COSINE,
        "heldout_task_balanced_feedback_projection_median_at_least_0_50": finite(projection) and projection >= S2_MIN_HELDOUT_PROJECTION,
        "heldout_task_balanced_pf_beats_p1_median_at_least_0_60": finite(pf_wins) and pf_wins >= S2_MIN_HELDOUT_PF_WIN,
    }
    status = "GO" if all(criteria.values()) else ("INCOMPLETE" if not complete else "NO_GO")
    return {
        "schema_version": 1,
        "experiment": "S2_predictive_prior_decomposition",
        "status": status,
        "role": "compute-matched mechanism diagnostic only; PF is not promoted as a runtime route",
        "criteria": criteria,
        "source_ablation_root": str(root),
        "records": len(raw),
        "expected_records": len(state_index),
        "splits": summaries,
        "audit_error_count": len(errors),
        "audit_errors": errors[:200],
        "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_used": False,
        "scheduler_or_threshold_used": False,
    }


def s4_records(directory: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[int] = set()
    for path in sorted(directory.glob("anchor_*.json")):
        payload = read_json(path)
        if payload is None:
            errors.append(f"unreadable:{path.name}")
            continue
        index = payload.get("anchor_index")
        if not isinstance(index, int) or index in seen:
            errors.append(f"invalid_or_duplicate:{path.name}")
            continue
        seen.add(index)
        if payload.get("status") != "PASS":
            errors.append(f"not_pass:{index}:{payload.get('status')}")
            continue
        if payload.get("checkpoint_sha256") != ORIGINAL_CHECKPOINT_SHA256 or int(payload.get("denoising_steps", -1)) != 1:
            errors.append(f"contract:{index}")
            continue
        records.append(payload)
    return records, errors


def summarize_s4(records: list[dict[str, Any]], errors: list[str], expected_tasks: int) -> dict[str, Any]:
    by_split = split_rows(records)
    splits: dict[str, Any] = {}
    for split, rows in by_split.items():
        correct = [path_metric(row, "metrics", "pv0_correct_to_f1", "mean_step_l2") for row in rows]
        shuffled = [path_metric(row, "metrics", "pv0_shuffled_to_f1", "mean_step_l2") for row in rows]
        p1 = [path_metric(row, "metrics", "p1_to_f1", "mean_step_l2") for row in rows]
        splits[split] = {
            "anchors": len(rows),
            "tasks": len({row["task_uid"] for row in rows}),
            "pv0_correct_to_f1_mean_step_l2": numeric_summary(correct),
            "pv0_shuffled_to_f1_mean_step_l2": numeric_summary(shuffled),
            "p1_to_f1_mean_step_l2": numeric_summary(p1),
            "correct_beats_shuffled": bool_summary(
                path_metric(row, "metrics", "correct_beats_shuffled") for row in rows
            ),
            "correct_beats_p1": bool_summary(
                finite(path_metric(row, "metrics", "pv0_correct_to_f1", "mean_step_l2"))
                and finite(path_metric(row, "metrics", "p1_to_f1", "mean_step_l2"))
                and path_metric(row, "metrics", "pv0_correct_to_f1", "mean_step_l2")
                < path_metric(row, "metrics", "p1_to_f1", "mean_step_l2")
                for row in rows
            ),
        }
    for split in ("discovery", "validation", "heldout"):
        splits.setdefault(split, {"anchors": 0, "tasks": 0})
    complete = len(records) == expected_tasks and len({row["task_uid"] for row in records}) == expected_tasks
    all_win = bool_summary(path_metric(row, "metrics", "correct_beats_shuffled") for row in records)["fraction"]
    heldout_win = splits["heldout"].get("correct_beats_shuffled", {}).get("fraction")
    criteria = {
        "one_frozen_anchor_per_40_tasks": complete,
        "all_task_correct_visual_beats_shuffled_visual_at_least_0_70": finite(all_win) and all_win >= S4_MIN_ALL_TASK_CORRECT_WIN,
        "heldout_correct_visual_beats_shuffled_visual_at_least_0_60": finite(heldout_win) and heldout_win >= S4_MIN_HELDOUT_CORRECT_WIN,
    }
    status = "GO" if all(criteria.values()) else ("INCOMPLETE" if not complete else "NO_GO")
    return {
        "schema_version": 1,
        "experiment": "S4_fixed_native_condition_compile",
        "status": status,
        "fixed_hypothesis": "current causal fresh visual prefix is the native condition carrier; no blocks/interfaces were searched",
        "criteria": criteria,
        "records": len(records),
        "expected_tasks": expected_tasks,
        "splits": splits,
        "audit_error_count": len(errors),
        "audit_errors": errors[:200],
        "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
        "denoising_steps": 1,
        "value_used": False,
        "privileged_runtime_state_used": False,
        "scheduler_or_threshold_used": False,
    }


def failure_localization(records: list[dict[str, Any]], directory: Path) -> dict[str, Any]:
    """Useful diagnosis for either a complete NO-GO or partial live result."""

    if not records:
        return {"status": "NO_RECORDS"}
    p1_errors = np.asarray(
        [path_metric(row, "metrics", "p1_to_f1", "mean_step_l2") for row in records], dtype=np.float64
    )
    pv0_errors = np.asarray(
        [path_metric(row, "metrics", "pv0_to_f1", "mean_step_l2") for row in records], dtype=np.float64
    )
    per_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        per_task[str(row["task_uid"])].append(row)
    task_map = []
    for task, rows in sorted(per_task.items()):
        task_map.append(
            {
                "task_uid": task,
                "split": rows[0]["split"],
                "states": len(rows),
                "p1_to_f1_mean_step_l2": numeric_summary(
                    path_metric(row, "metrics", "p1_to_f1", "mean_step_l2") for row in rows
                ),
                "pv0_to_f1_mean_step_l2": numeric_summary(
                    path_metric(row, "metrics", "pv0_to_f1", "mean_step_l2") for row in rows
                ),
                "pv0_failure_at_epsilon_fraction": bool_summary(
                    path_metric(row, "metrics", "pv0_to_f1", "mean_step_l2") > FIDELITY_EPSILON for row in rows
                )["fraction"],
                "pv0_beats_p1_fraction": bool_summary(
                    path_metric(row, "metrics", "recovery", "native_beats_p1") for row in rows
                )["fraction"],
                "first_gripper_mismatch_fraction": bool_summary(
                    not bool(path_metric(row, "metrics", "pv0_to_f1", "first_gripper_sign_match")) for row in rows
                )["fraction"],
            }
        )
    tails: dict[str, Any] = {}
    for fraction in (0.05, 0.10):
        threshold = float(np.quantile(p1_errors, 1.0 - fraction))
        selected = [row for row, error in zip(records, p1_errors, strict=True) if error >= threshold]
        tails[f"top_{int(fraction * 100):02d}_pct_p1_error"] = {
            "p1_threshold": threshold,
            "states": len(selected),
            "pv0_to_f1_mean_step_l2": numeric_summary(
                path_metric(row, "metrics", "pv0_to_f1", "mean_step_l2") for row in selected
            ),
            "pv0_failure_at_epsilon_fraction": bool_summary(
                path_metric(row, "metrics", "pv0_to_f1", "mean_step_l2") > FIDELITY_EPSILON for row in selected
            )["fraction"],
            "pv0_beats_p1_fraction": bool_summary(
                path_metric(row, "metrics", "recovery", "native_beats_p1") for row in selected
            )["fraction"],
        }
    ordered = sorted(records, key=lambda row: path_metric(row, "metrics", "pv0_to_f1", "mean_step_l2"), reverse=True)
    representative = [
        {
            "state_key": row["state_key"],
            "task_uid": row["task_uid"],
            "split": row["split"],
            "pv0_to_f1_mean_step_l2": path_metric(row, "metrics", "pv0_to_f1", "mean_step_l2"),
            "p1_to_f1_mean_step_l2": path_metric(row, "metrics", "p1_to_f1", "mean_step_l2"),
            "artifact": str(directory / f"state_{int(row['global_index']):05d}.json"),
        }
        for row in ordered[:20]
    ]
    return {
        "status": "READY",
        "frozen_failure_epsilon": FIDELITY_EPSILON,
        "records": len(records),
        "overall": {
            "p1_to_f1_mean_step_l2": numeric_summary(p1_errors),
            "pv0_to_f1_mean_step_l2": numeric_summary(pv0_errors),
            "first_action_vs_late_chunk": {
                "first_action_l2": numeric_summary(
                    path_metric(row, "metrics", "pv0_to_f1", "first_action_l2") for row in records
                ),
                "k4_mean_step_l2": numeric_summary(
                    path_metric(row, "metrics", "pv0_to_f1", "prefixes", "4", "mean_step_l2") for row in records
                ),
                "k16_mean_step_l2": numeric_summary(
                    path_metric(row, "metrics", "pv0_to_f1", "prefixes", "16", "mean_step_l2") for row in records
                ),
            },
            "first_gripper_mismatch_fraction": bool_summary(
                not bool(path_metric(row, "metrics", "pv0_to_f1", "first_gripper_sign_match")) for row in records
            )["fraction"],
        },
        "per_task_failure_map": task_map,
        "p1_error_tails": tails,
        "representative_trace_bundles": representative,
    }


def phase_decision(s1: Mapping[str, Any], s2: Mapping[str, Any], s3: Mapping[str, Any], s4: Mapping[str, Any]) -> dict[str, Any]:
    statuses = {"S1": s1.get("status"), "S2": s2.get("status"), "S3": s3.get("status"), "S4": s4.get("status")}
    if all(status == "GO" for status in statuses.values()):
        status = "GO"
        rationale = "All frozen Phase-A fidelity, execution-prefix, decomposition, and condition-control gates passed."
    elif any(status == "INCOMPLETE" for status in statuses.values()):
        status = "INCOMPLETE"
        rationale = "At least one required Phase-A artifact family is incomplete; no closed-loop promotion yet."
    elif any(status is None for status in statuses.values()):
        status = "INCOMPLETE_INFRA"
        rationale = "A required Phase-A analysis artifact is unavailable."
    else:
        status = "NO_GO"
        rationale = "At least one frozen Phase-A gate failed; promote only failure localization, not a new method."
    return {
        "schema_version": 1,
        "experiment": "PV0_PHASE_A_DECISION",
        "status": status,
        "rationale": rationale,
        "phase_statuses": statuses,
        "gates": {
            "S1": s1.get("criteria"),
            "S2": s2.get("criteria"),
            "S3": s3.get("criteria"),
            "S4": s4.get("criteria"),
        },
        "frozen_protocol": {
            "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "new_interface_search_used": False,
        },
        "generated_at_ns": time.time_ns(),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_index = read_jsonl(args.state_index)
    if len({entry["state_key"] for entry in state_index}) != len(state_index):
        raise RuntimeError("state index has duplicate state keys")
    s2 = summarize_s2(state_index, args.ablation_root)
    atomic_write_json(args.output_dir / "S2_PREDICTIVE_PRIOR_DECOMPOSITION.json", s2)
    if args.static_only:
        print(json.dumps({"S2": s2["status"], "records": s2["records"]}), flush=True)
        return
    records, s1_errors = s1_records(state_index, args.s1_dir)
    s1 = summarize_s1(state_index, records, s1_errors)
    s3 = summarize_s3(state_index, records, s1)
    anchors, s4_errors = s4_records(args.s4_dir)
    s4 = summarize_s4(anchors, s4_errors, len({entry["task_uid"] for entry in state_index}))
    failure = failure_localization(records, args.s1_dir)
    decision = phase_decision(s1, s2, s3, s4)
    atomic_write_json(args.output_dir / "S1_PV0_FULL_SCALE_FIDELITY.json", s1)
    atomic_write_json(args.output_dir / "S3_EXECUTION_PREFIX_ALIGNMENT.json", s3)
    atomic_write_json(args.output_dir / "S4_CONDITION_COMPILE_VALIDATION.json", s4)
    atomic_write_json(args.output_dir / "PV0_FAILURE_LOCALIZATION.json", failure)
    atomic_write_json(args.output_dir / "PHASE_A_DECISION.json", decision)
    print(
        json.dumps(
            {
                "S1": s1["status"],
                "S2": s2["status"],
                "S3": s3["status"],
                "S4": s4["status"],
                "PHASE_A": decision["status"],
                "S1_records": len(records),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

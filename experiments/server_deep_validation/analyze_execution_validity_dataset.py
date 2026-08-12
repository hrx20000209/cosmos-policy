#!/usr/bin/env python3
"""Analyze task-disjoint, shadow-labeled R2 execution-validity data.

All features are formed from feedback strictly preceding a decision.  F1/P1/
PV0 actions are retrospective labels; they never appear in a runtime feature.
The only learned object is a discovery-frozen linear risk score.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr


EPS = 1e-12
FEATURES = (
    "prediction_age_actions",
    "reuse_depth",
    "tracking_direction_residual",
    "windowed_physical_displacement",
    "progress_consistency_ratio",
    "chunk_boundary_discontinuity",
    "gripper_switch_count",
    "gripper_qpos_delta",
    "planned_action_norm",
    "planned_action_jerk",
)
PHYSICAL_SCORE_FEATURES = tuple(
    feature for feature in FEATURES if feature not in {"prediction_age_actions", "reuse_depth"}
)
CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def finite(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values)


def summary(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    x = x[finite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.1)),
        "p90": float(np.quantile(x, 0.9)),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / den) if den > EPS else float("nan")


def spearman(score: np.ndarray, target: np.ndarray) -> float | None:
    keep = finite(score) & finite(target)
    if keep.sum() < 4 or len(np.unique(score[keep])) < 2 or len(np.unique(target[keep])) < 2:
        return None
    value = spearmanr(score[keep], target[keep]).statistic
    return float(value) if np.isfinite(value) else None


def auc(score: np.ndarray, target: np.ndarray) -> float | None:
    keep = finite(score) & np.isfinite(target)
    score, target = score[keep], target[keep].astype(bool)
    positives, negatives = int(target.sum()), int((~target).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[target].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def task_macro(rows: list[dict[str, Any]], score_key: str, target_key: str) -> float | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_uid"]].append(row)
    results = [
        spearman(np.asarray([row[score_key] for row in group]), np.asarray([row[target_key] for row in group]))
        for group in grouped.values()
    ]
    results = [result for result in results if result is not None]
    return float(np.mean(results)) if results else None


def request_context(traces: list[dict[str, Any]], control_step: int) -> tuple[float, float]:
    prior = sorted((trace for trace in traces if int(trace["control_step_id"]) < control_step), key=lambda x: int(x["control_step_id"]))
    physical = [
        trace
        for trace in prior
        if (trace.get("extra") or {}).get("visual_input_mode") in {"fresh", "native_persistent"}
    ]
    last_physical = physical[-1] if physical else None
    last_step = int(last_physical["control_step_id"]) if last_physical else 0
    p1_count = sum(
        (trace.get("extra") or {}).get("visual_input_mode") == "predicted"
        and int(trace["control_step_id"]) > last_step
        for trace in prior
    )
    return float(max(control_step - last_step, 0)), float(p1_count)


def feedback_features(feedback: list[dict[str, Any]], control_step: int, window: int) -> dict[str, float]:
    prior = sorted(
        (
            item
            for item in feedback
            if item.get("control_step_after_settle") is not None
            and 0 <= int(item["control_step_after_settle"]) < control_step
        ),
        key=lambda item: int(item["control_step_after_settle"]),
    )[-window:]
    if not prior:
        return {feature: float("nan") for feature in FEATURES if feature not in {"prediction_age_actions", "reuse_depth"}}
    planned = np.asarray([item["planned_action"] for item in prior], dtype=np.float64)
    eef_before = np.asarray(prior[0]["before"]["eef_pos"], dtype=np.float64)
    eef_after = np.asarray(prior[-1]["after"]["eef_pos"], dtype=np.float64)
    actual_delta = eef_after - eef_before
    nominal_delta = planned[:, :3].sum(axis=0)
    alignment = cosine(actual_delta, nominal_delta)
    actual_norm, nominal_norm = float(np.linalg.norm(actual_delta)), float(np.linalg.norm(nominal_delta))
    gripper_commands = np.sign(planned[:, 6])
    gripper_qpos = np.asarray([item["after"]["gripper_qpos"] for item in prior], dtype=np.float64)
    previous = prior[-1]
    previous_last = np.asarray(previous["planned_action"], dtype=np.float64)
    boundary = float("nan")
    if len(prior) >= 2:
        request_ids = [item.get("request_id") for item in prior]
        boundaries = [index for index in range(1, len(prior)) if request_ids[index] != request_ids[index - 1]]
        if boundaries:
            index = boundaries[-1]
            boundary = float(np.linalg.norm(planned[index] - planned[index - 1]))
    return {
        "tracking_direction_residual": 1.0 - alignment if np.isfinite(alignment) else float("nan"),
        "windowed_physical_displacement": actual_norm,
        "progress_consistency_ratio": actual_norm / max(nominal_norm, EPS),
        "chunk_boundary_discontinuity": boundary,
        "gripper_switch_count": float(np.count_nonzero(np.diff(gripper_commands) != 0)),
        "gripper_qpos_delta": float(np.linalg.norm(gripper_qpos[-1] - gripper_qpos[0])),
        "planned_action_norm": float(np.mean(np.linalg.norm(planned[:, :6], axis=1))),
        "planned_action_jerk": float(np.mean(np.linalg.norm(np.diff(planned[:, :6], axis=0), axis=1))) if len(planned) > 1 else 0.0,
        "last_planned_action_norm": float(np.linalg.norm(previous_last[:6])),
    }


def load_rows(episodes: Path, window: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(episodes.glob("pv0_r2_*.json")):
        if path.name.endswith("_traces.json") or ".prior_failure_" in path.name:
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("status") != "PASS" or raw.get("mode") != "pv0_r2":
            raise RuntimeError(f"invalid episode: {path}")
        if raw.get("checkpoint_sha256") != CHECKPOINT_SHA256 or raw.get("value_used") is not False:
            raise RuntimeError(f"frozen checkpoint/value contract violation: {path}")
        if raw.get("shadow_labels_runtime_policy_input") is not False:
            raise RuntimeError(f"shadow label leaked into runtime: {path}")
        trace_path = path.with_name(f"{path.stem}_traces.json")
        try:
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
            traces = trace_payload["traces"]
        except (OSError, json.JSONDecodeError, KeyError) as error:
            raise RuntimeError(f"missing/invalid trace payload for {path}: {error}") from error
        feedback = raw["execution_feedback"]
        manifest = raw["manifest_row"]
        for label in raw.get("shadow_validity_labels", []):
            if label.get("shadow_only") is not True or label.get("value_used") is not False:
                raise RuntimeError(f"invalid shadow label: {path}")
            control = int(label["control_step"])
            age, depth = request_context(traces, control)
            row = {
                "episode_key": manifest["episode_key"],
                "task_uid": manifest["task_uid"],
                "split": manifest["split"],
                "init_state_index": int(manifest["init_state_index"]),
                "control_step": control,
                "p1_risk": float(label["p1_to_f1_rmse"]),
                "pv0_residual_risk": float(label["pv0_to_f1_rmse"]),
                "pv0_correction_gain": float(label["pv0_correction_gain"]),
                "prediction_age_actions": age,
                "reuse_depth": depth,
                **feedback_features(feedback, control, window),
            }
            rows.append(row)
    return rows


def learn_linear_score(discovery: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    """Fit only physical execution features; age/depth remain explicit baselines."""
    x = np.asarray([[row[name] for name in PHYSICAL_SCORE_FEATURES] for row in discovery], dtype=np.float64)
    y = np.asarray([row["p1_risk"] for row in discovery], dtype=np.float64)
    medians = np.nanmedian(x, axis=0)
    # Some short-window signals, notably a chunk boundary, can be unavailable
    # at every decision in a fixed 16-action prefix.  A frozen neutral fill
    # retains the feature schema without manufacturing a risk signal.
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isfinite(x), x, medians)
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales = np.where(scales > EPS, scales, 1.0)
    z = (x - means) / scales
    weights = np.linalg.solve(z.T @ z + 1e-3 * np.eye(z.shape[1]), z.T @ y)
    return (
        {name: float(weight) for name, weight in zip(PHYSICAL_SCORE_FEATURES, weights)},
        {name: float(value) for name, value in zip(PHYSICAL_SCORE_FEATURES, medians)},
    )


def assign_score(rows: list[dict[str, Any]], weights: dict[str, float], medians: dict[str, float], discovery: list[dict[str, Any]]) -> None:
    d = np.asarray([[row[name] for name in PHYSICAL_SCORE_FEATURES] for row in discovery], dtype=np.float64)
    d = np.where(np.isfinite(d), d, np.asarray([medians[name] for name in PHYSICAL_SCORE_FEATURES]))
    means, scales = d.mean(axis=0), d.std(axis=0)
    scales = np.where(scales > EPS, scales, 1.0)
    vector = np.asarray([weights[name] for name in PHYSICAL_SCORE_FEATURES])
    for row in rows:
        values = np.asarray([row[name] for name in PHYSICAL_SCORE_FEATURES], dtype=np.float64)
        values = np.where(np.isfinite(values), values, np.asarray([medians[name] for name in PHYSICAL_SCORE_FEATURES]))
        row["feedback_linear_score"] = float(((values - means) / scales) @ vector)


def evaluate(rows: list[dict[str, Any]], score_key: str, threshold_10: float, threshold_20: float) -> dict[str, Any]:
    score = np.asarray([row[score_key] for row in rows], dtype=np.float64)
    risk = np.asarray([row["p1_risk"] for row in rows], dtype=np.float64)
    high10, high20 = risk >= threshold_10, risk >= threshold_20
    result = {
        "spearman_p1_risk": spearman(score, risk),
        "task_balanced_spearman_p1_risk": task_macro(rows, score_key, "p1_risk"),
        "auc_discovery_frozen_top10_risk": auc(score, high10),
        "auc_discovery_frozen_top20_risk": auc(score, high20),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-actions", type=int, default=16)
    args = parser.parse_args()
    rows = load_rows(args.episodes, args.window_actions)
    split = {name: [row for row in rows if row["split"] == name] for name in ("discovery", "validation", "heldout")}
    if not all(split.values()):
        raise RuntimeError("all task-disjoint splits require at least one shadow label")
    weights, medians = learn_linear_score(split["discovery"])
    assign_score(rows, weights, medians, split["discovery"])
    for episode_rows in defaultdict(list, {key: [row for row in rows if row["episode_key"] == key] for key in {row["episode_key"] for row in rows}}).values():
        episode_rows.sort(key=lambda row: row["control_step"])
        for index, row in enumerate(episode_rows):
            row["next_p1_risk"] = episode_rows[index + 1]["p1_risk"] if index + 1 < len(episode_rows) else float("nan")
    threshold10 = float(np.quantile([row["p1_risk"] for row in split["discovery"]], 0.9))
    threshold20 = float(np.quantile([row["p1_risk"] for row in split["discovery"]], 0.8))
    baselines = {
        "AGE_ONLY": "prediction_age_actions",
        "REUSE_DEPTH_ONLY": "reuse_depth",
        "ACTION_NORM": "planned_action_norm",
        "JERK": "planned_action_jerk",
        "EXECUTION_FEEDBACK_LINEAR": "feedback_linear_score",
    }
    results = {
        name: {baseline: evaluate(values, score, threshold10, threshold20) for baseline, score in baselines.items()}
        for name, values in split.items()
    }
    lead_time = {
        name: {
            baseline: {
                "one_request_ahead_spearman": spearman(
                    np.asarray([row[score] for row in values]), np.asarray([row["next_p1_risk"] for row in values])
                ),
                "lead_actions": 16,
            }
            for baseline, score in baselines.items()
        }
        for name, values in split.items()
    }
    feedback = results["heldout"]["EXECUTION_FEEDBACK_LINEAR"]
    age = results["heldout"]["AGE_ONLY"]
    direction_consistent = (
        results["validation"]["EXECUTION_FEEDBACK_LINEAR"]["task_balanced_spearman_p1_risk"] is not None
        and feedback["task_balanced_spearman_p1_risk"] is not None
        and results["validation"]["EXECUTION_FEEDBACK_LINEAR"]["task_balanced_spearman_p1_risk"] > 0
        and feedback["task_balanced_spearman_p1_risk"] > 0
    )
    beats_age = (
        feedback["task_balanced_spearman_p1_risk"] is not None
        and age["task_balanced_spearman_p1_risk"] is not None
        and feedback["task_balanced_spearman_p1_risk"] > age["task_balanced_spearman_p1_risk"]
    )
    decision = "EXECUTION_FEEDBACK_GO_CANDIDATE" if direction_consistent and beats_age else "FEEDBACK_ADAPTATION_NO_GO"
    dataset_path = args.output_dir / "EXECUTION_VALIDITY_DATASET.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "dataset": "EXECUTION_VALIDITY_DATASET",
        "rows": {name: len(values) for name, values in split.items()},
        "tasks": {name: len({row["task_uid"] for row in values}) for name, values in split.items()},
        "feature_window_actions": args.window_actions,
        "controller_visible_features": list(FEATURES),
        "physical_score_features_excluding_age_and_depth": list(PHYSICAL_SCORE_FEATURES),
        "label_definition": {"p1_risk": "D(A_P1,A_F1)", "pv0_residual_risk": "D(A_PV0,A_F1)"},
        "runtime_oracle_inputs": False,
        "linear_score_frozen_on": "discovery",
        "linear_score_weights": weights,
        "imputation_medians_frozen_on": medians,
        "high_risk_thresholds_frozen_on_discovery": {"top10": threshold10, "top20": threshold20},
        "baseline_evaluation": results,
        "one_request_ahead_lead_time": lead_time,
        "decision": decision,
        "decision_rule": "Feedback must be directionally task-balanced on validation/heldout and beat AGE_ONLY on heldout task-balanced Spearman.",
        "limitations": [
            "Small 12-task x 2-init shadow collection; it is a gate, not a full-promotion result.",
            "Shadow F1/P1/PV0 actions are retrospective labels and are never runtime inputs.",
            "No action scaling, retiming, task identity, object pose, contact, reward, or success is used as a runtime feature.",
        ],
    }
    write_json(args.output_dir / "EXECUTION_VALIDITY_FEATURE_ANALYSIS.json", payload)
    md = [
        "# Execution-validity shadow dataset (small task-disjoint gate)",
        "",
        f"- rows: discovery={len(split['discovery'])}, validation={len(split['validation'])}, heldout={len(split['heldout'])}",
        f"- tasks: discovery={len({r['task_uid'] for r in split['discovery']})}, validation={len({r['task_uid'] for r in split['validation']})}, heldout={len({r['task_uid'] for r in split['heldout']})}",
        f"- decision: **{decision}**",
        "- Labels are offline F1/P1/PV0 comparisons. Runtime features are only proprioception and planned/executed actions preceding the decision.",
    ]
    (args.output_dir / "EXECUTION_VALIDITY_FEATURE_ANALYSIS_ZH.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"rows": payload["rows"], "decision": decision}, ensure_ascii=False))


if __name__ == "__main__":
    main()

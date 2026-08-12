#!/usr/bin/env python3
"""Frozen, task-disjoint discovery for PV0-centered execution feedback.

This program is intentionally offline.  It reads the already collected
Foundation V2 F1/P1/PF action bundles and writes only reports.  F1 is used
solely as a retrospective diagnostic target; no emitted policy consumes it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


HORIZON, ARM_DIMS, GRIPPER_DIM, EPS = 16, slice(0, 6), 6, 1e-12
CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def finite(x: object) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def summary(xs: list[float]) -> dict[str, Any]:
    a = np.asarray([x for x in xs if finite(x)], dtype=np.float64)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)), "p10": float(np.quantile(a, .1)), "p90": float(np.quantile(a, .9))}


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) / math.sqrt(np.asarray(a).size))


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > EPS else None


def spearman(a: list[float], b: list[float]) -> float | None:
    x, y = np.asarray(a), np.asarray(b)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return None
    value = spearmanr(x[mask], y[mask]).statistic
    return float(value) if np.isfinite(value) else None


def auc(score: list[float], target: list[float], fraction: float) -> float | None:
    x, y = np.asarray(score), np.asarray(target)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 8:
        return None
    positive = y >= np.quantile(y, 1 - fraction)
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return None
    ranks = rankdata(x)
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def task_macro(rows: list[dict[str, Any]], feature: str, target: str) -> float | None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_uid"]].append(row)
    values = [spearman([r[feature] for r in group], [r[target] for r in group]) for group in grouped.values()]
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def scaled_error(p1: np.ndarray, f1: np.ndarray) -> tuple[float, float, float | None]:
    """Least-squares arm-only alpha; gripper is never changed."""
    p, f = p1[:, ARM_DIMS], f1[:, ARM_DIMS]
    denom = float(np.sum(p * p))
    alpha = float(np.sum(p * f) / denom) if denom > EPS else 1.0
    alpha = float(np.clip(alpha, 0.0, 2.0))
    base, scaled = l2(p, f), l2(alpha * p, f)
    recovery = (base - scaled) / base if base > EPS else None
    return alpha, recovery if recovery is not None else float("nan"), cosine(p, f)


def boundary_jump(chunks: list[np.ndarray], index: int) -> float:
    if index <= 0 or index >= len(chunks):
        return float("nan")
    return float(np.linalg.norm(chunks[index - 1][-1] - chunks[index][0]))


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("state_*.pt")):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if raw.get("checkpoint_sha256") != CHECKPOINT_SHA256 or raw.get("value_used") is not False or raw.get("privileged_runtime_state_used") is not False:
            raise RuntimeError(f"frozen contract violation: {path}")
        actions = {k: np.asarray(v, dtype=np.float64) for k, v in raw["actions"].items()}
        if set(actions) != {"F1", "P1", "PP", "PF", "FF"} or any(v.shape != (HORIZON, 7) for v in actions.values()):
            raise RuntimeError(f"unexpected action schema: {path}")
        p1_risk = l2(actions["P1"], actions["F1"])
        pv0_residual = l2(actions["PF"], actions["F1"])
        gain = (p1_risk - pv0_residual) / p1_risk if p1_risk > EPS else 0.0
        alpha, recovery, direction = scaled_error(actions["P1"], actions["F1"])
        jerk = float(np.mean(np.linalg.norm(np.diff(actions["P1"][:, ARM_DIMS], axis=0), axis=1)))
        row = {
            "state_key": raw["state_key"], "episode_key": raw["episode_key"], "split": raw["split"], "task_uid": raw["task_uid"],
            "request_index": int(raw["request_index"]), "control_step": int(raw["control_step"]),
            "p1_risk": p1_risk, "pv0_residual_risk": pv0_residual, "pv0_gain": gain,
            "p1_action_norm": float(np.mean(np.linalg.norm(actions["P1"][:, ARM_DIMS], axis=1))), "p1_jerk": jerk,
            "episode_progress": float(raw["control_step"]), "executed_prefix_fraction": 1.0,
            "gripper_switches": int(np.count_nonzero(np.diff(np.sign(actions["P1"][:, GRIPPER_DIM])))),
            "p1_f1_direction_cosine": direction, "oracle_alpha": alpha, "oracle_scale_recovery": recovery,
            "first_action_scale_recovery": (l2(actions["P1"][:1, ARM_DIMS], actions["F1"][:1, ARM_DIMS]) - l2(alpha * actions["P1"][:1, ARM_DIMS], actions["F1"][:1, ARM_DIMS])) / max(l2(actions["P1"][:1, ARM_DIMS], actions["F1"][:1, ARM_DIMS]), EPS),
            "actions": actions,
        }
        rows.append(row)
    return rows


def attach_temporal_features(rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["episode_key"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: r["request_index"])
        chunks = [r["actions"]["P1"] for r in group]
        for i, row in enumerate(group):
            row["reuse_depth"] = float(i)
            row["p1_boundary_jump"] = boundary_jump(chunks, i)


def feature_report(rows: list[dict[str, Any]], feature: str) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for target in ("p1_risk", "pv0_gain", "pv0_residual_risk"):
        x, y = [r[feature] for r in rows], [r[target] for r in rows]
        report[target] = {"spearman": spearman(x, y), "task_balanced_spearman": task_macro(rows, feature, target), "top10_auc": auc(x, y, .1), "top20_auc": auc(x, y, .2)}
    return report


def frozen_global_alpha(discovery: list[dict[str, Any]]) -> float:
    # Fits only the discovery route; a single alpha is the strongest allowed baseline.
    numer = sum(float(np.sum(r["actions"]["P1"][:, ARM_DIMS] * r["actions"]["F1"][:, ARM_DIMS])) for r in discovery)
    denom = sum(float(np.sum(r["actions"]["P1"][:, ARM_DIMS] ** 2)) for r in discovery)
    return float(np.clip(numer / denom, 0.0, 2.0)) if denom > EPS else 1.0


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ablation-root", type=Path, default=Path("/data/rxhuang/wam_full_scale_server/queue_b/ablation"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/pv0_execution_feedback"))
    args = p.parse_args()
    rows = load_rows(args.ablation_root)
    attach_temporal_features(rows)
    split = {name: [r for r in rows if r["split"] == name] for name in ("discovery", "validation", "heldout")}
    features = ["p1_boundary_jump", "reuse_depth", "p1_action_norm", "p1_jerk", "gripper_switches", "episode_progress"]
    landscape = {name: {feature: feature_report(values, feature) for feature in features} for name, values in split.items()}
    alpha = frozen_global_alpha(split["discovery"])
    alpha_eval: dict[str, Any] = {}
    for name, values in split.items():
        global_recovery = []
        for row in values:
            p1, f1 = row["actions"]["P1"][:, ARM_DIMS], row["actions"]["F1"][:, ARM_DIMS]
            base, adjusted = l2(p1, f1), l2(alpha * p1, f1)
            global_recovery.append((base - adjusted) / base if base > EPS else float("nan"))
        alpha_eval[name] = {"oracle_alpha": summary([r["oracle_alpha"] for r in values]), "oracle_recovery": summary([r["oracle_scale_recovery"] for r in values]), "global_alpha_recovery": summary(global_recovery), "direction_cosine": summary([r["p1_f1_direction_cosine"] for r in values])}
    heldout_oracle = alpha_eval["heldout"]["oracle_recovery"]["median"] or 0.0
    heldout_direction = alpha_eval["heldout"]["direction_cosine"]["median"] or 0.0
    decision = "GO" if heldout_oracle >= .30 and heldout_direction >= .70 else "ACTION_PROGRESS_NO_GO"
    result = {"schema_version": 1, "frozen_contract": {"checkpoint_sha256": CHECKPOINT_SHA256, "denoise": 1, "value_used": False, "runtime_f1_oracle_used": False}, "rows": {k: len(v) for k, v in split.items()}, "feature_landscape": landscape, "action_progress_oracle": {"global_alpha_frozen_on": "discovery", "global_alpha": alpha, "by_split": alpha_eval, "decision": decision, "kill_gate": {"minimum_oracle_recovery": .30, "minimum_direction_cosine": .70}}, "limitations": ["Foundation V2 stores F1/P1/PF action bundles but not tick-level actual EEF feedback; tracking/progress estimator requires new telemetry pilot.", "PF is the frozen one-step fresh-condition oracle/precursor, not a claim that offline F1 targets are runtime available."]}
    write_json(args.output_dir / "CORRECTION_DEMAND_AND_ACTION_PROGRESS.json", result)
    md = ["# PV0 执行反馈：冻结离线 discovery", "", f"- states: discovery={len(split['discovery'])}, validation={len(split['validation'])}, heldout={len(split['heldout'])}", f"- global arm-only α（仅 discovery 冻结）: {alpha:.4f}", f"- action-progress gate: **{decision}**", "", "该阶段只使用 F1/P1/PF 的离线 action comparison，F1 不会进入 runtime。旧资产没有逐 tick EEF telemetry，因此 tracking/progress 的闭环证据将在新 telemetry pilot 中测量。"]
    (args.output_dir / "CORRECTION_DEMAND_AND_ACTION_PROGRESS_ZH.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "decision": decision, "global_alpha": alpha, "sha256": hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()}))


if __name__ == "__main__":
    main()

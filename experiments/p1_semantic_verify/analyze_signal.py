#!/usr/bin/env python3
"""E12-A analysis: does the frozen P1-native score predict its own action risk?

Discovery mode confirms direction and freezes operating points.  Validation mode
only evaluates what discovery froze.  Nothing here refits the frozen score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.p1_semantic_verify.common import ACTION_GEOMETRY_FEATURES
from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json

BUDGETS = (0.20, 0.30, 0.40)
BOOTSTRAP = 10000
SEED = 20260812


def load_rows(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"shard not complete: {path} status={payload.get('status')}")
        rows.extend(payload["rows"])
    frame = pd.DataFrame(rows)
    if frame.state_id.duplicated().any():
        raise RuntimeError("duplicate state_id across shards")
    return frame


def task_balanced_rho(frame: pd.DataFrame, score: str, target: str) -> tuple[float | None, dict[str, float]]:
    per_task: dict[str, float] = {}
    for task, group in frame.groupby("task_id"):
        if group[score].nunique() > 1 and group[target].nunique() > 1:
            per_task[task] = float(spearmanr(group[score], group[target]).statistic)
    if not per_task:
        return None, {}
    return float(np.mean(list(per_task.values()))), per_task


def hierarchical_indices(frame: pd.DataFrame, rng: np.random.Generator) -> list[np.ndarray]:
    """Resample tasks, then episodes within task, then states within episode.

    Returns one positional-index array per resampled task, so a statistic can be
    computed task-balanced without rebuilding a DataFrame per draw.
    """

    structure = [
        [np.asarray(indices) for indices in task_rows.groupby("episode_id").indices.values()]
        for _, task_rows in frame.groupby("task_id")
    ]
    picked = rng.integers(0, len(structure), size=len(structure))
    draws: list[np.ndarray] = []
    for task_position in picked:
        episodes = structure[task_position]
        chosen = rng.integers(0, len(episodes), size=len(episodes))
        parts = []
        for episode_position in chosen:
            indices = episodes[episode_position]
            parts.append(rng.choice(indices, size=len(indices), replace=True))
        draws.append(np.concatenate(parts))
    return draws


def task_balanced_from_indices(columns: dict[str, np.ndarray], draws: list[np.ndarray],
                               score: str, target: str) -> float | None:
    values = []
    for indices in draws:
        a, b = columns[score][indices], columns[target][indices]
        if len(np.unique(a)) > 1 and len(np.unique(b)) > 1:
            values.append(spearmanr(a, b).statistic)
    return float(np.mean(values)) if values else None


def bootstrap_statistics(frame: pd.DataFrame, statistic: Any, draws: int = BOOTSTRAP) -> dict[str, Any]:
    """Hierarchical (task -> episode -> state) bootstrap; 128 states are never IID."""

    rng = np.random.default_rng(SEED)
    columns = {name: frame[name].to_numpy() for name in frame.columns
               if frame[name].dtype.kind in "fiub"}
    values: list[float] = []
    for _ in range(draws):
        value = statistic(columns, hierarchical_indices(frame, rng))
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"n_draws": 0, "mean": None, "ci95": [None, None]}
    array = np.asarray(values)
    return {
        "n_draws": len(values),
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        "fraction_positive": float(np.mean(array > 0)),
    }


def fit_action_only(train: pd.DataFrame, target: str = "risk_p1_full") -> dict[str, Any]:
    """Ridge lambda=10 on the 5 frozen action-geometry features, log1p target."""

    features = list(ACTION_GEOMETRY_FEATURES)
    x = train[features].to_numpy(dtype=np.float64)
    y = np.log1p(train[target].to_numpy(dtype=np.float64))
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    coef = np.linalg.solve(z.T @ z + 10.0 * np.eye(len(features)), z.T @ (y - y.mean()))
    return {
        "feature_names": features, "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "coefficients": coef.tolist(), "intercept": float(y.mean()),
        "ridge_lambda": 10.0, "target": f"log1p({target})", "output_transform": "expm1(linear_score)",
        "training_rows": int(len(train)), "training_split": "discovery",
    }


def apply_action_only(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    x = frame[model["feature_names"]].to_numpy(dtype=np.float64)
    mean, scale, coef = (np.asarray(model[key], dtype=np.float64)
                         for key in ("feature_mean", "feature_scale", "coefficients"))
    return np.expm1(((x - mean) / scale) @ coef + float(model["intercept"]))


def leave_one_task_out(frame: pd.DataFrame, target: str = "risk_p1_full") -> np.ndarray:
    """Honest in-split ACTION_ONLY predictions: never fit on the task it scores."""

    predictions = np.full(len(frame), np.nan)
    for task in frame.task_id.unique():
        mask = (frame.task_id == task).to_numpy()
        model = fit_action_only(frame[~mask], target)
        predictions[mask] = apply_action_only(model, frame[mask])
    return predictions


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks over ties
    values = np.concatenate([positive, negative])
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[: len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    hits = np.cumsum(labels)
    precision = hits / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


def group_contrast(frame: pd.DataFrame, score: str, target: str, quantile: float = 0.20) -> dict[str, Any]:
    """High-S vs low-S mean target, computed task-balanced."""

    highs, lows = [], []
    for _, group in frame.groupby("task_id"):
        count = max(1, int(round(len(group) * quantile)))
        ordered = group.sort_values(score)
        lows.append(float(ordered.head(count)[target].mean()))
        highs.append(float(ordered.tail(count)[target].mean()))
    return {
        "quantile": quantile,
        "low_s_mean": float(np.mean(lows)),
        "high_s_mean": float(np.mean(highs)),
        "difference": float(np.mean(highs) - np.mean(lows)),
        "per_task_high_minus_low": [float(h - l) for h, l in zip(highs, lows)],
        "tasks_with_positive_contrast": int(sum(h > l for h, l in zip(highs, lows))),
    }


def evaluate(frame: pd.DataFrame, score: str, action_only: np.ndarray) -> dict[str, Any]:
    work = frame.copy()
    work["action_only"] = action_only
    rho_s, per_task_s = task_balanced_rho(work, score, "risk_p1_full")
    rho_a, per_task_a = task_balanced_rho(work, "action_only", "risk_p1_full")
    rho_gain, per_task_gain = task_balanced_rho(work, score, "pv0_gain")
    rho_pv0, _ = task_balanced_rho(work, score, "risk_pv0_full")

    labels = np.zeros(len(work), dtype=int)
    for _, group in work.groupby("task_id"):
        cutoff = group.risk_p1_full.quantile(0.80)
        labels[work.index.get_indexer(group.index[group.risk_p1_full >= cutoff])] = 1

    def statistic_s(columns: dict[str, np.ndarray], draws: list[np.ndarray]) -> float | None:
        return task_balanced_from_indices(columns, draws, score, "risk_p1_full")

    def statistic_delta(columns: dict[str, np.ndarray], draws: list[np.ndarray]) -> float | None:
        left = task_balanced_from_indices(columns, draws, score, "risk_p1_full")
        right = task_balanced_from_indices(columns, draws, "action_only", "risk_p1_full")
        return None if left is None or right is None else left - right

    def statistic_contrast(columns: dict[str, np.ndarray], draws: list[np.ndarray]) -> float | None:
        """Task-balanced high-S minus low-S mean risk under the same resample."""

        highs, lows = [], []
        for indices in draws:
            order = indices[np.argsort(columns[score][indices], kind="mergesort")]
            count = max(1, int(round(len(order) * 0.20)))
            lows.append(columns["risk_p1_full"][order[:count]].mean())
            highs.append(columns["risk_p1_full"][order[-count:]].mean())
        return float(np.mean(highs) - np.mean(lows)) if highs else None

    return {
        "states": int(len(work)),
        "tasks": int(work.task_id.nunique()),
        "task_balanced_spearman_s_vs_risk_p1": rho_s,
        "per_task_spearman_s_vs_risk_p1": per_task_s,
        "tasks_with_positive_rho": int(sum(value > 0 for value in per_task_s.values())),
        "task_balanced_spearman_action_only_vs_risk_p1": rho_a,
        "per_task_spearman_action_only": per_task_a,
        "absolute_rho_improvement_over_action_only": (
            None if rho_s is None or rho_a is None else float(rho_s - rho_a)
        ),
        "task_balanced_spearman_s_vs_pv0_gain": rho_gain,
        "per_task_spearman_s_vs_pv0_gain": per_task_gain,
        "task_balanced_spearman_s_vs_risk_pv0": rho_pv0,
        "auroc_top20_percent_risk": auroc(work[score].to_numpy(), labels),
        "pr_auc_top20_percent_risk": average_precision(work[score].to_numpy(), labels),
        "risk_contrast_high_vs_low_s": group_contrast(work, score, "risk_p1_full"),
        "pv0_gain_contrast_high_vs_low_s": group_contrast(work, score, "pv0_gain"),
        "bootstrap_rho_s": bootstrap_statistics(work, statistic_s),
        "bootstrap_rho_delta_vs_action_only": bootstrap_statistics(work, statistic_delta),
        "bootstrap_risk_contrast": bootstrap_statistics(work, statistic_contrast),
        "pv0_correction_quality": {
            "risk_p1_full": {k: float(v) for k, v in work.risk_p1_full.describe().items()},
            "risk_pv0_full": {k: float(v) for k, v in work.risk_pv0_full.describe().items()},
            "pv0_gain_positive_fraction": float((work.pv0_gain > 0).mean()),
            "pv0_relative_recovery_median": float(work.pv0_relative_recovery.median()),
            "pv0_worse_than_p1_states": int((work.pv0_gain < 0).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("discovery", "validation"))
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-only-model", type=Path,
                        default=Path("reports/p1_semantic_verify/E12_ACTION_ONLY_MODEL.json"),
                        help="written in discovery mode, read in validation mode")
    parser.add_argument("--candidates", type=Path,
                        default=Path("reports/p1_semantic_verify/E12B_POLICY_CANDIDATES.json"))
    args = parser.parse_args()

    frame = load_rows(args.shards)
    invalid = frame[~frame.valid.astype(bool)]
    frame = frame[frame.valid.astype(bool)].reset_index(drop=True)
    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.parquet, index=False)

    if args.split == "discovery":
        # Reported ACTION_ONLY is leave-one-task-out so discovery is not an
        # in-sample comparison; the model shipped to validation is fit on all
        # discovery rows, exactly once, before validation is ever read.
        action_only = leave_one_task_out(frame)
        model = fit_action_only(frame)
        atomic_write_json(args.action_only_model, model)
        action_only_note = "leave-one-task-out within discovery"
    else:
        model = json.loads(args.action_only_model.read_text(encoding="utf-8"))
        action_only = apply_action_only(model, frame)
        action_only_note = "frozen discovery-fit model applied to validation"

    result = evaluate(frame, "s_p1", action_only)
    result["action_only_protocol"] = action_only_note

    if "s_p1_e4variant" in frame.columns and frame.s_p1_e4variant.notna().any():
        rho_variant, per_task_variant = task_balanced_rho(frame, "s_p1_e4variant", "risk_p1_e4variant_full")
        rho_cross, _ = task_balanced_rho(frame, "s_p1_e4variant", "risk_p1_full")
        result["diagnostic_e4_collection_variant"] = {
            "note": "DISCOVERY-ONLY DIAGNOSTIC. Not a candidate policy in this round.",
            "task_balanced_spearman_variant_score_vs_variant_risk": rho_variant,
            "per_task": per_task_variant,
            "task_balanced_spearman_variant_score_vs_deployed_p1_risk": rho_cross,
            "score_rank_agreement_deployed_vs_variant": float(
                spearmanr(frame.s_p1, frame.s_p1_e4variant).statistic
            ),
            "risk_rank_agreement_deployed_vs_variant": float(
                spearmanr(frame.risk_p1_full, frame.risk_p1_e4variant_full).statistic
            ),
        }

    if args.split == "discovery":
        thresholds = {}
        for budget in BUDGETS:
            thresholds[f"{int(budget * 100)}"] = {
                "correction_budget": budget,
                "global_s_threshold": float(frame.s_p1.quantile(1.0 - budget)),
                "global_action_only_threshold": float(np.quantile(action_only, 1.0 - budget)),
                "rule": "correct when score > threshold; a single global cut, never per task",
            }
        candidates = {
            "schema_version": 1,
            "status": "FROZEN_FROM_DISCOVERY",
            "source": str(args.parquet),
            "discovery_states": int(len(frame)),
            "score": "frozen E4 INTERNAL_PLUS_ACTION read from the deployed P1 forward",
            "score_checksum": "3299a5ec2308f990ff68e6f80f2d28ca922b4aef8c72c2f708da139bd6745d4b",
            "operating_points": thresholds,
            "action_only_model": str(args.action_only_model),
            "periodic_baseline_period": {
                f"{int(budget * 100)}": max(1, int(round(1.0 / budget))) for budget in BUDGETS
            },
            "note": "Thresholds are global and frozen here. Validation and heldout only evaluate them.",
        }
        atomic_write_json(args.candidates, candidates)
        result["frozen_operating_points"] = thresholds

    payload = {
        "schema_version": 1,
        "experiment": "E12A",
        "split": args.split,
        "status": "PASS",
        "frozen_score_modified": False,
        "invalid_rows": int(len(invalid)),
        "invalid_reasons": invalid.invalid_reason.tolist() if len(invalid) else [],
        "bootstrap_draws": BOOTSTRAP,
        "bootstrap_scheme": "hierarchical task -> episode -> state",
        "parquet": str(args.parquet),
        "result": result,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "split": args.split, "states": result["states"],
        "rho_s": result["task_balanced_spearman_s_vs_risk_p1"],
        "rho_action_only": result["task_balanced_spearman_action_only_vs_risk_p1"],
        "delta": result["absolute_rho_improvement_over_action_only"],
        "rho_s_vs_pv0_gain": result["task_balanced_spearman_s_vs_pv0_gain"],
        "tasks_positive": result["tasks_with_positive_rho"],
    }, indent=2))


if __name__ == "__main__":
    main()

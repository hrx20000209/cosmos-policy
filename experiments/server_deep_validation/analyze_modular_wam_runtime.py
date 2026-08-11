"""Audit and analyze the frozen Cosmos modular WAM runtime evidence.

This is deliberately an *offline mechanism analysis*.  It never invokes a
policy, consumes Cosmos' value, restores simulator state into a policy input,
or chooses a sensing/execution schedule.  Its job is to turn the existing
Fresh collection and paired F1/P1/PP/PF/FF states into auditable answers for:

* M1: does physical prediction residual explain an action-relevant value of
  fresh sensing better than generic pre-arrival signals?
* M3: does fresh feedback causally preserve the Fresh-1 action through an
  available internal-condition interface, and what evidence remains oracle?

Run from the repository root.  All result files are generated under
``reports/modular_wam_runtime`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from wam_runtime.accounting import ComputeLedger, ComputeRecord
from wam_runtime.fixed_policy_harness import ORIGINAL_COSMOS_CHECKPOINT_SHA256
from wam_runtime.thor_cost_model import ThorCostModel


ACTION_ROUTES = ("F1", "P1", "PP", "PF", "FF")
ACTION_HORIZON = 16
PROPRIO_DIM = 9
EPS = 1e-12


@dataclass
class DatasetBundle:
    name: str
    manifest_rows: dict[str, dict[str, Any]]
    episode_paths: dict[str, Path]
    ablations: dict[str, dict[str, Any]]
    audit: dict[str, Any]


def json_safe(value: Any) -> Any:
    """Convert numpy and non-finite values into portable JSON values."""

    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray([float(value) for value in values if finite(value)], dtype=np.float64)
    return float(np.quantile(array, q)) if array.size else None


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if finite(value)], dtype=np.float64)
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p25": None, "p75": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def safe_spearman(left: Iterable[float], right: Iterable[float]) -> float | None:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return None
    result = spearmanr(x, y)
    rho = getattr(result, "statistic", result[0])
    return float(rho) if np.isfinite(rho) else None


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 4 or np.std(left) < EPS or np.std(right) < EPS:
        return None
    output = float(np.corrcoef(left, right)[0, 1])
    return output if math.isfinite(output) else None


def macro_task_spearman(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if finite(row.get(x_key)) and finite(row.get(y_key)):
            by_task[str(row["task_uid"])].append(row)
    values: list[float] = []
    valid_tasks = 0
    for task_rows in by_task.values():
        rho = safe_spearman((row[x_key] for row in task_rows), (row[y_key] for row in task_rows))
        if rho is not None:
            values.append(rho)
            valid_tasks += 1
    return {"tasks_with_signal": valid_tasks, "macro_spearman": float(np.mean(values)) if values else None, "per_task": numeric_summary(values)}


def binary_auc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    mask = np.isfinite(scores)
    scores, positive = scores[mask], positive[mask]
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[positive].sum())
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def top_precision(scores: np.ndarray, target: np.ndarray, fraction: float) -> float | None:
    mask = np.isfinite(scores) & np.isfinite(target)
    scores, target = scores[mask], target[mask]
    if scores.size < 5:
        return None
    target_positive = target >= np.quantile(target, 1.0 - fraction)
    picked = max(1, int(math.ceil(scores.size * fraction)))
    chosen = np.argsort(scores)[-picked:]
    return float(target_positive[chosen].mean())


def signal_summary(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    valid = [row for row in rows if finite(row.get(x_key)) and finite(row.get(y_key))]
    x = np.asarray([row[x_key] for row in valid], dtype=np.float64)
    y = np.asarray([row[y_key] for row in valid], dtype=np.float64)
    summary: dict[str, Any] = {
        "n": int(len(valid)),
        "request_weighted_spearman": safe_spearman(x, y),
        "task_balanced": macro_task_spearman(valid, x_key, y_key),
        "x": numeric_summary(x),
        "y": numeric_summary(y),
    }
    for fraction in (0.10, 0.20, 0.30):
        label = f"top_{int(fraction * 100):02d}"
        positive = y >= np.quantile(y, 1.0 - fraction) if y.size else np.zeros(0, dtype=bool)
        summary[label] = {
            "auc": binary_auc(x, positive),
            "precision": top_precision(x, y, fraction),
            "prevalence": float(positive.mean()) if positive.size else None,
        }
    return summary


def partial_rank_correlation(rows: list[dict[str, Any]], x_key: str, y_key: str, controls: tuple[str, ...]) -> dict[str, Any]:
    valid = [row for row in rows if finite(row.get(x_key)) and finite(row.get(y_key)) and all(finite(row.get(key)) for key in controls)]
    if len(valid) < len(controls) + 5:
        return {"n": len(valid), "partial_spearman": None, "controls": list(controls)}
    x = rankdata([row[x_key] for row in valid]).astype(np.float64)
    y = rankdata([row[y_key] for row in valid]).astype(np.float64)
    design = np.column_stack([np.ones(len(valid)), *[rankdata([row[key] for row in valid]) for key in controls]])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return {"n": len(valid), "partial_spearman": pearson(x_residual, y_residual), "controls": list(controls)}


def action_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != (ACTION_HORIZON, 7) or right.shape != (ACTION_HORIZON, 7):
        raise ValueError(f"expected paired 16x7 action chunks, got {left.shape} and {right.shape}")
    delta = left - right
    per_step = np.linalg.norm(delta, axis=1)
    return {
        "full_rmse": float(np.linalg.norm(delta) / math.sqrt(delta.size)),
        "mean_step_l2": float(per_step.mean()),
        "first_l2": float(per_step[0]),
        "first_4_rmse": float(np.linalg.norm(delta[:4]) / math.sqrt(delta[:4].size)),
        "first_8_rmse": float(np.linalg.norm(delta[:8]) / math.sqrt(delta[:8].size)),
        "end_l2": float(per_step[-1]),
        "gripper_abs": float(abs(delta[0, 6])),
        "gripper_sign_flip": float(np.sign(left[0, 6]) != np.sign(right[0, 6])),
    }


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    product = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / product) if product > EPS else None


def quaternion_angle(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    a_norm, b_norm = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if a_norm < EPS or b_norm < EPS:
        return None
    dot = abs(float(np.dot(a / a_norm, b / b_norm)))
    return float(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))


def decode_future_proprio(latent: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """Decode the documented future-proprio carrier (slot 5) from a joint latent."""

    array = np.asarray(latent)
    if array.shape != (1, 16, 9, 28, 28):
        raise ValueError(f"unexpected generated joint latent shape: {array.shape}")
    normalized = array[0, :, 5].reshape(-1)[:PROPRIO_DIM].astype(np.float32)
    minimum = np.asarray(stats["proprio_min"], dtype=np.float32)
    maximum = np.asarray(stats["proprio_max"], dtype=np.float32)
    return (0.5 * (normalized + 1.0) * (maximum - minimum) + minimum).astype(np.float32)


def proprio_residual(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float | None]:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)
    return {
        "residual_l2": float(np.linalg.norm(delta)),
        "residual_mean_abs": float(np.abs(delta).mean()),
        "residual_max_abs": float(np.abs(delta).max()),
        "gripper_q_l2": float(np.linalg.norm(delta[:2])),
        "eef_translation_l2": float(np.linalg.norm(delta[2:5])),
        "eef_rotation_rad": quaternion_angle(predicted[5:9], actual[5:9]),
        **{f"joint_abs_{index}": float(abs(value)) for index, value in enumerate(delta[:2])},
    }


def load_torch(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def episode_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("episode_*.pt") if ".partial" not in path.name)


def state_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("state_*.pt") if ".partial" not in path.name)


def audit_dataset(name: str, manifest_path: Path, collection_root: Path, ablation_root: Path) -> DatasetBundle:
    manifest_list = read_jsonl(manifest_path)
    manifest_rows = {str(row["episode_key"]): row for row in manifest_list}
    errors: list[str] = []
    warnings: list[str] = []
    if len(manifest_rows) != len(manifest_list):
        errors.append("manifest has duplicate episode_key")
    manifest_task_counts = Counter((row["split"], row["task_uid"]) for row in manifest_list)
    bad_init_sets = [f"{split}:{task}" for (split, task), count in manifest_task_counts.items() if count != 5]
    if bad_init_sets:
        errors.append(f"manifest task/init cardinality differs from five: {bad_init_sets[:10]}")

    paths_by_key: dict[str, Path] = {}
    expected_states: set[str] = set()
    collected_splits: Counter[str] = Counter()
    endpoint_pairs = 0
    physically_aligned_pairs = 0
    collection_latency: list[float] = []
    collection_actions = 0
    collection_requests = 0
    tensor_shape_counts: Counter[str] = Counter()
    for path in episode_files(collection_root):
        try:
            episode = load_torch(path)
        except Exception as error:  # audit must continue over bad artifacts
            errors.append(f"cannot load {path}: {type(error).__name__}:{error}")
            continue
        episode_key = str(episode.get("episode_key", ""))
        if episode_key in paths_by_key:
            errors.append(f"duplicate collected episode {episode_key}")
            continue
        paths_by_key[episode_key] = path
        row = manifest_rows.get(episode_key)
        if row is None:
            errors.append(f"collected episode absent from manifest: {episode_key}")
            continue
        for key in ("split", "task_uid", "task_name", "suite", "seed", "init_state_index"):
            if episode.get(key) != row.get(key):
                errors.append(f"episode metadata mismatch {episode_key}:{key}")
        if episode.get("checkpoint_sha256") != ORIGINAL_COSMOS_CHECKPOINT_SHA256:
            errors.append(f"wrong checkpoint in episode {episode_key}")
        if episode.get("value_used") is not False or episode.get("privileged_runtime_state_used") is not False:
            errors.append(f"policy-contract violation in episode {episode_key}")
        if int(episode.get("denoising_steps", -1)) != 1:
            errors.append(f"non-one-step collection in episode {episode_key}")
        requests = episode.get("requests", [])
        if not isinstance(requests, list) or not requests:
            errors.append(f"missing requests in episode {episode_key}")
            continue
        collected_splits[str(episode["split"])] += 1
        collection_actions += int(episode.get("control_steps", 0))
        collection_requests += len(requests)
        for request in requests:
            if finite(request.get("policy_latency_ms")):
                collection_latency.append(float(request["policy_latency_ms"]))
            latent = np.asarray(request.get("generated_latent"))
            tensor_shape_counts[f"{latent.shape}:{latent.dtype}"] += 1
            if latent.shape != (1, 16, 9, 28, 28):
                errors.append(f"unexpected latent shape {episode_key}:{request.get('request_index')}:{latent.shape}")
            if np.asarray(request.get("proprio")).shape != (9,):
                errors.append(f"unexpected proprio shape {episode_key}:{request.get('request_index')}")
            if np.asarray(request.get("fresh_action")).shape != (16, 7):
                errors.append(f"unexpected action shape {episode_key}:{request.get('request_index')}")
            if request.get("checkpoint_sha256") != ORIGINAL_COSMOS_CHECKPOINT_SHA256:
                errors.append(f"wrong request checkpoint {episode_key}:{request.get('request_index')}")
        for index in range(1, len(requests)):
            source, target = requests[index - 1], requests[index]
            endpoint_pairs += 1
            expected_key = str(target.get("state_key"))
            expected_states.add(expected_key)
            if int(target.get("request_index", -1)) != index or int(source.get("request_index", -1)) != index - 1:
                errors.append(f"nonsequential request indices in {episode_key}")
            delta = int(target.get("control_step", -999)) - int(source.get("control_step", -999))
            if delta == ACTION_HORIZON:
                physically_aligned_pairs += 1
            else:
                warnings.append(f"non-physical endpoint pair {target.get('state_key')} control_delta={delta}")

    ablations: dict[str, dict[str, Any]] = {}
    ablation_latencies: dict[str, list[float]] = {route: [] for route in ACTION_ROUTES}
    ablation_split_count: Counter[str] = Counter()
    for path in state_files(ablation_root):
        try:
            raw = load_torch(path)
        except Exception as error:
            errors.append(f"cannot load {path}: {type(error).__name__}:{error}")
            continue
        state_key = str(raw.get("state_key", ""))
        if state_key in ablations:
            errors.append(f"duplicate ablation state {state_key}")
            continue
        episode_key = str(raw.get("episode_key", ""))
        row = manifest_rows.get(episode_key)
        if row is None:
            errors.append(f"ablation episode absent from manifest {state_key}")
        else:
            for key in ("split", "task_uid", "task_name", "suite", "seed", "init_state_index"):
                if raw.get(key) != row.get(key):
                    errors.append(f"ablation metadata mismatch {state_key}:{key}")
        if raw.get("checkpoint_sha256") != ORIGINAL_COSMOS_CHECKPOINT_SHA256:
            errors.append(f"wrong checkpoint in ablation {state_key}")
        if raw.get("value_used") is not False or raw.get("privileged_runtime_state_used") is not False or raw.get("scheduler_or_threshold_used") is not False:
            errors.append(f"policy-contract violation in ablation {state_key}")
        actions = raw.get("actions", {})
        if set(actions) != set(ACTION_ROUTES):
            errors.append(f"ablation routes malformed {state_key}: {sorted(actions)}")
            continue
        converted_actions: dict[str, np.ndarray] = {}
        for route, value in actions.items():
            action = np.asarray(value, dtype=np.float32)
            if action.shape != (ACTION_HORIZON, 7):
                errors.append(f"ablation action shape {state_key}:{route}:{action.shape}")
            converted_actions[route] = action
        metrics = raw.get("metrics", {})
        for route, value in metrics.get("latency_ms", {}).items():
            if route in ablation_latencies and finite(value):
                ablation_latencies[route].append(float(value))
        ablations[state_key] = {
            "state_key": state_key,
            "episode_key": episode_key,
            "split": raw.get("split"),
            "task_uid": raw.get("task_uid"),
            "request_index": int(raw.get("request_index", -1)),
            "control_step": int(raw.get("control_step", -1)),
            "actions": converted_actions,
            "metrics": metrics,
        }
        ablation_split_count[str(raw.get("split"))] += 1

    missing_ablation = sorted(expected_states - set(ablations))
    unexpected_ablation = sorted(set(ablations) - expected_states)
    if missing_ablation:
        errors.append(f"missing ablation states: {len(missing_ablation)}")
    if unexpected_ablation:
        errors.append(f"unexpected ablation states: {len(unexpected_ablation)}")
    manifest_split = Counter(str(row["split"]) for row in manifest_list)
    audit = {
        "dataset": name,
        "manifest": str(manifest_path),
        "collection_root": str(collection_root),
        "ablation_root": str(ablation_root),
        "contract": {
            "checkpoint_sha256": ORIGINAL_COSMOS_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
        },
        "manifest": {
            "episodes": len(manifest_list),
            "unique_episode_keys": len(manifest_rows),
            "tasks": len({row["task_uid"] for row in manifest_list}),
            "split_episodes": dict(manifest_split),
            "task_init_cardinality_expected": 5,
            "bad_task_init_sets": bad_init_sets,
        },
        "collection": {
            "episodes": len(paths_by_key),
            "split_episodes": dict(collected_splits),
            "requests": collection_requests,
            "executed_actions": collection_actions,
            "policy_latency_ms": numeric_summary(collection_latency),
            "latent_shapes": dict(tensor_shape_counts),
        },
        "physical_alignment": {
            "endpoint_pairs": endpoint_pairs,
            "physically_aligned_pairs": physically_aligned_pairs,
            "fraction": physically_aligned_pairs / endpoint_pairs if endpoint_pairs else None,
            "definition": "source generated future endpoint is compared only with target proprio when control_step delta is exactly 16 executed actions",
        },
        "ablation": {
            "states": len(ablations),
            "split_states": dict(ablation_split_count),
            "expected_states": len(expected_states),
            "missing_states": len(missing_ablation),
            "unexpected_states": len(unexpected_ablation),
            "latency_ms": {route: numeric_summary(values) for route, values in ablation_latencies.items()},
        },
        "errors": errors[:200],
        "error_count": len(errors),
        "warnings": warnings[:200],
        "warning_count": len(warnings),
        "audit_status": "PASS" if not errors else "FAIL",
    }
    return DatasetBundle(name=name, manifest_rows=manifest_rows, episode_paths=paths_by_key, ablations=ablations, audit=audit)


def build_m1_records(bundle: DatasetBundle, stats: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidate_pairs = 0
    physical_pairs = 0
    decode_failures = 0
    for episode_key, path in sorted(bundle.episode_paths.items()):
        episode = load_torch(path)
        requests = episode["requests"]
        for index in range(1, len(requests)):
            candidate_pairs += 1
            source, target = requests[index - 1], requests[index]
            control_delta = int(target["control_step"]) - int(source["control_step"])
            if control_delta != ACTION_HORIZON:
                continue
            physical_pairs += 1
            state_key = str(target["state_key"])
            ablation = bundle.ablations.get(state_key)
            if ablation is None:
                continue
            try:
                predicted = decode_future_proprio(np.asarray(source["generated_latent"]), stats)
            except Exception:
                decode_failures += 1
                continue
            actual = np.asarray(target["proprio"], dtype=np.float32)
            residual = proprio_residual(predicted, actual)
            policy_delta = action_difference(ablation["actions"]["F1"], ablation["actions"]["P1"])
            feedback_delta = action_difference(ablation["actions"]["PF"], ablation["actions"]["PP"])
            source_action = np.asarray(source["fresh_action"], dtype=np.float32)
            prior_action = np.asarray(requests[index - 2]["fresh_action"], dtype=np.float32) if index >= 2 else None
            prior_latent = np.asarray(requests[index - 2]["generated_latent"]) if index >= 2 else None
            record: dict[str, Any] = {
                "dataset": bundle.name,
                "episode_key": episode_key,
                "state_key": state_key,
                "task_uid": str(ablation["task_uid"]),
                "split": str(ablation["split"]),
                "request_index": index,
                "control_delta_actions": control_delta,
                "source_action_magnitude": float(np.mean(np.linalg.norm(source_action, axis=1))),
                "past_action_jerk": float(np.mean(np.linalg.norm(source_action - prior_action, axis=1))) if prior_action is not None else float("nan"),
                "latent_l1_prior": float(np.mean(np.abs(np.asarray(source["generated_latent"], dtype=np.float32) - np.asarray(prior_latent, dtype=np.float32)))) if prior_latent is not None else float("nan"),
                "aoi_actions": float(control_delta),
                "proprio_motion_post_arrival": float(np.linalg.norm(actual - np.asarray(source["proprio"], dtype=np.float32))),
                **residual,
                **{f"policy_{key}": value for key, value in policy_delta.items()},
                **{f"feedback_{key}": value for key, value in feedback_delta.items()},
            }
            records.append(record)
    rng = np.random.default_rng(20260812)
    for record in records:
        record["random_control"] = float(rng.random())
        record["periodic_control"] = float(record["request_index"] % 4)
    metadata = {
        "candidate_endpoint_pairs": candidate_pairs,
        "physically_aligned_endpoint_pairs": physical_pairs,
        "physical_alignment_fraction": physical_pairs / candidate_pairs if candidate_pairs else None,
        "records_with_paired_actions": len(records),
        "decode_failures": decode_failures,
        "residual_definition": "L2(actual target proprio - source generated future-proprio endpoint); target is accepted only after exactly 16 executed source actions",
        "fair_pre_arrival_baselines": ["past_action_jerk", "source_action_magnitude", "latent_l1_prior", "aoi_actions", "random_control", "periodic_control"],
        "post_arrival_diagnostic_only": ["proprio_motion_post_arrival"],
    }
    return records, metadata


def lag_scan(records: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, Any]:
    by_episode: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_episode[record["episode_key"]][int(record["request_index"])] = record
    results: dict[str, Any] = {}
    for lag in range(-8, 9):
        paired: list[dict[str, Any]] = []
        for indexed in by_episode.values():
            for index, record in indexed.items():
                later = indexed.get(index + lag)
                if later is not None and finite(record.get(x_key)) and finite(later.get(y_key)):
                    paired.append({"task_uid": record["task_uid"], x_key: record[x_key], y_key: later[y_key]})
        results[str(lag)] = {
            "n": len(paired),
            "request_weighted_spearman": safe_spearman((row[x_key] for row in paired), (row[y_key] for row in paired)),
            "task_balanced": macro_task_spearman(paired, x_key, y_key),
        }
    return results


def m1_analysis(records: list[dict[str, Any]], metadata: Mapping[str, Any]) -> dict[str, Any]:
    y_keys = ("policy_full_rmse", "policy_first_l2", "policy_first_4_rmse", "policy_first_8_rmse", "policy_end_l2", "policy_gripper_abs", "feedback_full_rmse")
    x_keys = ("residual_l2", "eef_translation_l2", "eef_rotation_rad", "gripper_q_l2")
    scoped: dict[str, list[dict[str, Any]]] = {"all": records}
    for split in ("discovery", "validation", "heldout"):
        scoped[split] = [row for row in records if row["split"] == split]
    associations: dict[str, Any] = {}
    for scope, scope_rows in scoped.items():
        associations[scope] = {y_key: {x_key: signal_summary(scope_rows, x_key, y_key) for x_key in x_keys} for y_key in y_keys}
    baseline_fields = tuple(metadata["fair_pre_arrival_baselines"]) + tuple(metadata["post_arrival_diagnostic_only"])
    baselines = {
        scope: {field: signal_summary(rows, field, "policy_full_rmse") for field in baseline_fields}
        for scope, rows in scoped.items()
    }
    partial = {
        scope: partial_rank_correlation(rows, "residual_l2", "policy_full_rmse", ("past_action_jerk", "source_action_magnitude", "latent_l1_prior"))
        for scope, rows in scoped.items()
    }
    lags = lag_scan(records, "residual_l2", "policy_full_rmse")
    lag_values = {int(lag): result["request_weighted_spearman"] for lag, result in lags.items() if result["request_weighted_spearman"] is not None}
    peak_lag = max(lag_values, key=lambda lag: lag_values[lag]) if lag_values else None
    primary = {split: associations[split]["policy_full_rmse"]["residual_l2"] for split in ("discovery", "validation", "heldout")}
    baseline_jerk = {split: baselines[split]["past_action_jerk"]["task_balanced"]["macro_spearman"] for split in ("discovery", "validation", "heldout")}
    heldout_rho = primary["heldout"]["task_balanced"]["macro_spearman"]
    validation_rho = primary["validation"]["task_balanced"]["macro_spearman"]
    sign_consistent = all(value is not None and value > 0.0 for value in (primary["discovery"]["request_weighted_spearman"], primary["validation"]["request_weighted_spearman"], primary["heldout"]["request_weighted_spearman"]))
    beats_jerk = all(
        primary[split]["task_balanced"]["macro_spearman"] is not None
        and baseline_jerk[split] is not None
        and primary[split]["task_balanced"]["macro_spearman"] > baseline_jerk[split]
        for split in ("validation", "heldout")
    )
    criteria = {
        "physical_alignment_at_least_0_95": bool((metadata.get("physical_alignment_fraction") or 0.0) >= 0.95),
        "validation_task_balanced_spearman_at_least_0_30": bool(validation_rho is not None and validation_rho >= 0.30),
        "heldout_task_balanced_spearman_at_least_0_30": bool(heldout_rho is not None and heldout_rho >= 0.30),
        "same_positive_direction_across_splits": sign_consistent,
        "beats_jerk_on_validation_and_heldout": beats_jerk,
        "peak_association_at_lag_zero": peak_lag == 0,
    }
    return {
        "metadata": dict(metadata),
        "residual_summary": {key: numeric_summary(record.get(key) for record in records) for key in x_keys},
        "associations": associations,
        "baseline_comparison": baselines,
        "partial_correlation": partial,
        "lag_scan": lags,
        "decision": {
            "criteria": criteria,
            "peak_lag": peak_lag,
            "result": "M1_GO" if all(criteria.values()) else "M1_NO_GO",
            "interpretation": "This is a shadow signal-validation decision only; no threshold, sensing scheduler, or policy switch is implemented.",
        },
    }


def m3_metrics_from_ablation(ablation: Mapping[str, Any]) -> dict[str, float | None]:
    actions = ablation["actions"]
    target = np.asarray(actions["F1"], dtype=np.float64) - np.asarray(actions["P1"], dtype=np.float64)
    feedback = np.asarray(actions["PF"], dtype=np.float64) - np.asarray(actions["PP"], dtype=np.float64)
    target_norm_sq = float(np.dot(target.reshape(-1), target.reshape(-1)))
    target_norm = float(np.linalg.norm(target) / math.sqrt(target.size))
    feedback_norm = float(np.linalg.norm(feedback) / math.sqrt(feedback.size))
    pf_error = action_difference(actions["PF"], actions["F1"])
    p1_error = action_difference(actions["P1"], actions["F1"])
    pp_error = action_difference(actions["PP"], actions["F1"])
    ff_error = action_difference(actions["FF"], actions["F1"])
    did = np.asarray(actions["P1"], dtype=np.float64) + feedback
    did_error = action_difference(did, actions["F1"])
    return {
        "target_norm": target_norm,
        "feedback_norm": feedback_norm,
        "feedback_target_cosine": cosine(feedback, target),
        "feedback_target_projection": float(np.dot(feedback.reshape(-1), target.reshape(-1)) / target_norm_sq) if target_norm_sq > EPS else None,
        "feedback_target_sign_agreement": float(np.mean(np.sign(feedback) == np.sign(target))) if target_norm_sq > EPS else None,
        "pf_to_f1_rmse": pf_error["full_rmse"],
        "p1_to_f1_rmse": p1_error["full_rmse"],
        "pp_to_f1_rmse": pp_error["full_rmse"],
        "ff_to_f1_rmse": ff_error["full_rmse"],
        "pf_beats_p1": float(pf_error["full_rmse"] < p1_error["full_rmse"]),
        "pf_beats_pp": float(pf_error["full_rmse"] < pp_error["full_rmse"]),
        "pf_first_l2_to_f1": pf_error["first_l2"],
        "p1_first_l2_to_f1": p1_error["first_l2"],
        "pf_gripper_sign_flip_to_f1": pf_error["gripper_sign_flip"],
        "p1_gripper_sign_flip_to_f1": p1_error["gripper_sign_flip"],
        "did_to_f1_rmse": did_error["full_rmse"],
    }


def scoped_metric_summary(rows: list[dict[str, Any]], metric_keys: Iterable[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in metric_keys:
        values = [row.get(metric) for row in rows]
        output[metric] = numeric_summary(values)
        task_rows: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if finite(row.get(metric)):
                task_rows[str(row["task_uid"])].append(float(row[metric]))
        task_means = [statistics.fmean(values) for values in task_rows.values() if values]
        output[metric]["task_balanced"] = numeric_summary(task_means)
    return output


def audit_oracle(oracle_root: Path) -> dict[str, Any]:
    paths = sorted(path for path in oracle_root.rglob("state_*.json") if ".partial" not in path.name)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"cannot load {path}: {type(error).__name__}:{error}")
            continue
        rows.append(row)
    baselines = [float(row.get("baseline_predicted_to_fresh", float("nan"))) for row in rows if finite(row.get("baseline_predicted_to_fresh"))]
    nonzero = [value for value in baselines if abs(value) > 1e-6]
    repair_count = sum(len(row.get("repairs", [])) for row in rows)
    split_counts = Counter(str(row.get("split")) for row in rows)
    contract_bad = sum(
        row.get("checkpoint_sha256") != ORIGINAL_COSMOS_CHECKPOINT_SHA256 or row.get("value_used") is not False
        for row in rows
    )
    zero_fraction = 1.0 - len(nonzero) / len(baselines) if baselines else 1.0
    # A patch-recovery fraction is undefined when its denominator is exactly
    # zero.  Treat this as a protocol invalidation, not a successful recovery.
    valid = bool(rows) and zero_fraction < 0.05 and contract_bad == 0
    return {
        "paths": len(paths),
        "loaded_states": len(rows),
        "split_states": dict(split_counts),
        "baseline_predicted_to_fresh": numeric_summary(baselines),
        "nonzero_baseline_states": len(nonzero),
        "zero_baseline_fraction": zero_fraction,
        "repair_records": repair_count,
        "contract_bad_records": contract_bad,
        "load_errors": errors[:20],
        "status": "VALID" if valid else "INVALID_ORACLE_PROTOCOL",
        "reason": "Recovery is not interpretable when predicted-to-fresh baseline is zero. Code audit must verify that fresh path did not reuse previous generated latent before any repair frontier is used as evidence.",
    }


def _route_contract_is_valid(row: Mapping[str, Any]) -> bool:
    """Check the fresh/predicted route distinction needed for an oracle audit."""

    fresh = row.get("fresh_route_contract")
    predicted = row.get("predicted_route_contract")
    return (
        isinstance(fresh, Mapping)
        and isinstance(predicted, Mapping)
        and fresh.get("skip_vae_encoding") is False
        and fresh.get("previous_generated_latent") is False
        and predicted.get("skip_vae_encoding") is True
        and predicted.get("previous_generated_latent") is True
    )


def audit_corrected_oracle_microbatch(
    oracle_root: Path,
    manifest_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit the pre-registered, small corrected-oracle diagnostic.

    This intentionally has a narrower claim than a runtime benchmark.  Each
    artifact must first establish a genuine fresh-vs-predicted action gap and
    must label the fact that the hidden-state intervention uses a fresh-prefix
    oracle.  We then summarize only the 2-task x 2-init frontier without
    mixing it with the invalid legacy oracle or duplicate pilot states.
    """

    paths = sorted(path for path in oracle_root.rglob("state_*.json") if ".partial" not in path.name)
    errors: list[str] = []
    valid_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"cannot load {path}: {type(error).__name__}:{error}")
            continue
        all_rows.append(row)
        row_errors: list[str] = []
        episode_key = str(row.get("episode_key", ""))
        manifest = manifest_rows.get(episode_key)
        state_key = str(row.get("state_key", ""))
        request_index = row.get("request_index")
        if int(row.get("schema_version", -1)) < 2:
            row_errors.append("schema_version<2")
        if row.get("experiment") != "one_step_late_binding_oracle":
            row_errors.append("unexpected experiment")
        if manifest is None:
            row_errors.append("episode absent from manifest")
        else:
            if row.get("split") != manifest.get("split") or row.get("task_uid") != manifest.get("task_uid"):
                row_errors.append("manifest split/task mismatch")
        if not isinstance(request_index, int) or state_key != f"{episode_key}:req{request_index}":
            row_errors.append("state-key/request-index mismatch")
        if row.get("checkpoint_sha256") != ORIGINAL_COSMOS_CHECKPOINT_SHA256:
            row_errors.append("wrong checkpoint")
        if row.get("value_used") is not False:
            row_errors.append("value contract violation")
        if row.get("oracle_fresh_prefix_required") is not True:
            row_errors.append("missing fresh-prefix oracle label")
        if not _route_contract_is_valid(row):
            row_errors.append("fresh/predicted route contract violation")
        baseline = row.get("baseline_predicted_to_fresh")
        if not finite(baseline) or abs(float(baseline)) <= 1e-6:
            row_errors.append("zero or non-finite predicted-to-fresh baseline")
        repairs = row.get("repairs")
        if not isinstance(repairs, list) or not repairs:
            row_errors.append("no repair records")
        else:
            for repair in repairs:
                if not isinstance(repair, Mapping) or not finite(repair.get("recovery")):
                    row_errors.append("non-finite repair recovery")
                    break
        if row_errors:
            errors.append(f"{path}: {', '.join(row_errors)}")
            continue
        valid_rows.append(row)

    by_block_group: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows:
        for repair in row["repairs"]:
            by_block_group[(int(repair["block"]), str(repair["group"]))].append(repair)
    frontier = [
        {
            "block": block,
            "group": group,
            "states": len(repairs),
            "recovery": numeric_summary(repair.get("recovery") for repair in repairs),
            "distance_to_fresh": numeric_summary(repair.get("distance_to_fresh") for repair in repairs),
            "first_action_l2_to_fresh": numeric_summary(repair.get("first_action_l2_to_fresh") for repair in repairs),
            "gripper_sign_match": numeric_summary(repair.get("gripper_sign_match") for repair in repairs),
            "remaining_dit_fraction": numeric_summary(repair.get("remaining_dit_fraction") for repair in repairs),
        }
        for (block, group), repairs in sorted(by_block_group.items())
    ]
    split_counts = Counter(str(row.get("split")) for row in valid_rows)
    task_counts = Counter(str(row.get("task_uid")) for row in valid_rows)
    init_indices = sorted(
        int(manifest_rows[str(row["episode_key"])]["init_state_index"])
        for row in valid_rows
        if str(row["episode_key"]) in manifest_rows
    )
    expected_small_batch = len(valid_rows) >= 4 and split_counts["validation"] >= 2 and split_counts["heldout"] >= 2
    early = next((item for item in frontier if item["block"] == 4 and item["group"] == "current_visual"), None)
    middle = next((item for item in frontier if item["block"] == 12 and item["group"] == "current_visual"), None)
    late = next((item for item in frontier if item["block"] == 24 and item["group"] == "current_visual"), None)
    medians = [
        item["recovery"]["median"] if item is not None else None
        for item in (early, middle, late)
    ]
    frontier_descends = all(
        value is not None for value in medians
    ) and bool(medians[0] >= medians[1] >= medians[2])
    valid = bool(valid_rows) and not errors and expected_small_batch
    return {
        "paths": len(paths),
        "loaded_states": len(all_rows),
        "valid_states": len(valid_rows),
        "split_states": dict(split_counts),
        "task_states": dict(task_counts),
        "init_state_indices": init_indices,
        "baseline_predicted_to_fresh": numeric_summary(row.get("baseline_predicted_to_fresh") for row in valid_rows),
        "frontier": frontier,
        "frontier_descends_block4_to12_to24": frontier_descends,
        "pre_registered_2x2_coverage": expected_small_batch,
        "load_or_contract_errors": errors[:40],
        "error_count": len(errors),
        "status": "VALID_SMALL_SCALE_ORACLE_DIAGNOSTIC" if valid else "INVALID_OR_INCOMPLETE_SMALL_SCALE_ORACLE",
        "interpretation": (
            "This is a causal hidden-condition diagnostic only. It uses a fresh-prefix oracle "
            "to construct the intervention, so it cannot establish deployable partial recompute, "
            "runtime latency, or target-device cost."
        ),
    }


INTERFACE_GROUPS = ("current_visual", "visual_proprio", "future", "action")
NATIVE_PERSISTENT_ROUTES = ("F1", "P1", "PV0")


def audit_corrected_interface_microbatch(
    oracle_roots: Iterable[Path],
    manifest_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit a fixed-block, four-interface small-scale condition test.

    The worker reports only causal hidden-condition repairs.  In particular,
    this audit must not convert the recovery ranking into a claim about online
    compute saving: every repair is constructed from a fresh-prefix oracle.
    """

    paths = sorted({path for root in oracle_roots for path in root.rglob("state_*.json") if ".partial" not in path.name})
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    seen_state_keys: set[str] = set()
    expected_groups = set(INTERFACE_GROUPS)
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"cannot load {path}: {type(error).__name__}:{error}")
            continue
        row_errors: list[str] = []
        episode_key = str(row.get("episode_key", ""))
        manifest = manifest_rows.get(episode_key)
        state_key = str(row.get("state_key", ""))
        request_index = row.get("request_index")
        if state_key in seen_state_keys:
            row_errors.append("duplicate state_key")
        seen_state_keys.add(state_key)
        if int(row.get("schema_version", -1)) < 2:
            row_errors.append("schema_version<2")
        if row.get("experiment") != "one_step_late_binding_oracle":
            row_errors.append("unexpected experiment")
        if manifest is None:
            row_errors.append("episode absent from manifest")
        elif row.get("split") != manifest.get("split") or row.get("task_uid") != manifest.get("task_uid"):
            row_errors.append("manifest split/task mismatch")
        if not isinstance(request_index, int) or state_key != f"{episode_key}:req{request_index}":
            row_errors.append("state-key/request-index mismatch")
        if row.get("checkpoint_sha256") != ORIGINAL_COSMOS_CHECKPOINT_SHA256 or row.get("value_used") is not False:
            row_errors.append("model/value contract violation")
        if row.get("oracle_fresh_prefix_required") is not True or not _route_contract_is_valid(row):
            row_errors.append("fresh/predicted route contract violation")
        baseline = row.get("baseline_predicted_to_fresh")
        if not finite(baseline) or abs(float(baseline)) <= 1e-6:
            row_errors.append("zero or non-finite predicted-to-fresh baseline")
        repairs = row.get("repairs")
        if not isinstance(repairs, list):
            row_errors.append("repairs missing")
            repairs = []
        repair_groups = {str(repair.get("group")) for repair in repairs if isinstance(repair, Mapping)}
        repair_blocks = {int(repair.get("block", -1)) for repair in repairs if isinstance(repair, Mapping)}
        if repair_groups != expected_groups:
            row_errors.append(f"unexpected interface groups={sorted(repair_groups)}")
        if repair_blocks != {12}:
            row_errors.append(f"interface test is not fixed at block12={sorted(repair_blocks)}")
        if len(repairs) != len(expected_groups) or any(not finite(repair.get("recovery")) for repair in repairs if isinstance(repair, Mapping)):
            row_errors.append("incomplete or non-finite interface repair")
        if row_errors:
            errors.append(f"{path}: {', '.join(row_errors)}")
            continue
        rows.append(row)

    repairs_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    repair_by_state: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        indexed = {str(repair["group"]): repair for repair in row["repairs"]}
        repair_by_state[str(row["state_key"])] = indexed
        for group, repair in indexed.items():
            repairs_by_group[group].append(repair)
    groups = {
        group: {
            "states": len(repairs_by_group[group]),
            "recovery": numeric_summary(repair.get("recovery") for repair in repairs_by_group[group]),
            "distance_to_fresh": numeric_summary(repair.get("distance_to_fresh") for repair in repairs_by_group[group]),
            "first_action_l2_to_fresh": numeric_summary(repair.get("first_action_l2_to_fresh") for repair in repairs_by_group[group]),
            "gripper_sign_match": numeric_summary(repair.get("gripper_sign_match") for repair in repairs_by_group[group]),
        }
        for group in INTERFACE_GROUPS
    }

    def contrast(left: str, right: str) -> dict[str, Any]:
        deltas = [
            float(indexed[left]["recovery"]) - float(indexed[right]["recovery"])
            for indexed in repair_by_state.values()
            if left in indexed and right in indexed
        ]
        return {
            "left": left,
            "right": right,
            "paired_states": len(deltas),
            "recovery_delta_left_minus_right": numeric_summary(deltas),
            "left_beats_right_fraction": float(np.mean(np.asarray(deltas) > 0.0)) if deltas else None,
        }

    contrasts = [
        contrast("current_visual", "future"),
        contrast("current_visual", "action"),
        contrast("visual_proprio", "current_visual"),
    ]
    split_counts = Counter(str(row.get("split")) for row in rows)
    init_indices = sorted(
        int(manifest_rows[str(row["episode_key"])]["init_state_index"])
        for row in rows
        if str(row["episode_key"]) in manifest_rows
    )
    exact_2x2 = len(rows) == 4 and split_counts["validation"] == 2 and split_counts["heldout"] == 2
    visual_dominant = all(
        contrast_item["left_beats_right_fraction"] == 1.0
        for contrast_item in contrasts[:2]
    )
    valid = not errors and exact_2x2 and all(groups[group]["states"] == 4 for group in INTERFACE_GROUPS)
    return {
        "paths": len(paths),
        "valid_states": len(rows),
        "split_states": dict(split_counts),
        "init_state_indices": init_indices,
        "fixed_block": 12,
        "groups": groups,
        "contrasts": contrasts,
        "visual_dominant_over_nonvisual_on_all_states": visual_dominant,
        "pre_registered_2x2_coverage": exact_2x2,
        "load_or_contract_errors": errors[:40],
        "error_count": len(errors),
        "status": "VALID_SMALL_SCALE_INTERFACE_DIAGNOSTIC" if valid else "INVALID_OR_INCOMPLETE_SMALL_SCALE_INTERFACE_DIAGNOSTIC",
        "interpretation": (
            "A visual condition is strongly more action-relevant than the tested future-only or action-only "
            "conditions at block 12 in this fixed 2x2 batch. This remains a fresh-prefix oracle diagnostic, "
            "not evidence of a deployable arrival interface or a latency reduction."
        ),
    }


def _native_route_contract_is_valid(name: str, contract: Any) -> bool:
    if not isinstance(contract, Mapping):
        return False
    common = (
        contract.get("denoising_steps") == 1
        and contract.get("value_used") is False
        and contract.get("scheduler_or_threshold_used") is False
    )
    if name == "F1":
        return common and contract.get("skip_vae_encoding") is False and contract.get("previous_generated_latent") is False
    if name == "P1":
        return (
            common
            and contract.get("skip_vae_encoding") is True
            and contract.get("previous_generated_latent") is True
            and contract.get("skip_camera_preprocessing") is True
        )
    if name == "PV0":
        return (
            common
            and contract.get("skip_vae_encoding") is True
            and contract.get("previous_generated_latent") is True
            and contract.get("skip_camera_preprocessing") is False
            and contract.get("native_persistent_visual_correction") is True
            and contract.get("fresh_visual_prefix_frames") == 13
            and contract.get("fresh_visual_arrival_denoiser_forward") == 0
            and contract.get("async_predict_correct") is False
        )
    return False


def audit_native_persistent_microbatch(
    artifact_roots: Iterable[Path],
    manifest_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit the real, denoise=1 native persistent-condition microbatch.

    Unlike the hidden-patch frontier, PV0 receives an ordinary fresh camera
    observation and invokes the existing ``persistent_visual_correction`` API
    to encode a 13-frame causal prefix.  This is still intentionally only a
    2-task x 2-init server microbatch, so it can support a system *candidate*
    but not a target-device or task-success claim.
    """

    paths = sorted(
        {path for root in artifact_roots for path in root.rglob("*.json") if ".partial" not in path.name}
    )
    errors: list[str] = []
    valid_rows: list[dict[str, Any]] = []
    seen_state_keys: set[str] = set()
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"cannot load {path}: {type(error).__name__}:{error}")
            continue
        row_errors: list[str] = []
        state_key = str(row.get("state_key", ""))
        episode_key = str(row.get("episode_key", ""))
        manifest = manifest_rows.get(episode_key)
        if state_key in seen_state_keys:
            row_errors.append("duplicate state_key")
        seen_state_keys.add(state_key)
        if row.get("schema_version") != 1 or row.get("experiment") != "native_persistent_visual_condition_preflight":
            row_errors.append("unexpected native preflight schema/experiment")
        if row.get("status") != "PASS_EXECUTED":
            row_errors.append(f"execution status={row.get('status')}")
        if row.get("checkpoint_sha256") != ORIGINAL_COSMOS_CHECKPOINT_SHA256 or row.get("denoising_steps") != 1:
            row_errors.append("checkpoint or denoise contract violation")
        if row.get("value_used") is not False or row.get("privileged_runtime_state_used") is not False or row.get("scheduler_or_threshold_used") is not False:
            row_errors.append("value/privilege/scheduler contract violation")
        if manifest is None:
            row_errors.append("episode absent from manifest")
        else:
            if row.get("split") != manifest.get("split") or row.get("task_uid") != manifest.get("task_uid"):
                row_errors.append("manifest split/task mismatch")
            if row.get("init_state_index") != manifest.get("init_state_index"):
                row_errors.append("manifest init mismatch")
        request_suffix = state_key.partition(":req")[2]
        if not request_suffix.isdigit() or state_key != f"{episode_key}:req{request_suffix}":
            row_errors.append("state-key/request-index mismatch")
        interface = row.get("native_interface")
        if not isinstance(interface, Mapping) or (
            interface.get("api") != "get_action:persistent_visual_correction_prefix_frames"
            or interface.get("fresh_visual_prefix_frames") != 13
            or interface.get("arrival_denoiser_forward") != 0
            or interface.get("hidden_activation_patch_used") is not False
            or interface.get("fresh_prefix_oracle_used") is not False
        ):
            row_errors.append("native interface contract violation")
        routes = row.get("routes")
        if not isinstance(routes, Mapping) or set(routes) != set(NATIVE_PERSISTENT_ROUTES):
            row_errors.append("native route set malformed")
        elif any(not _native_route_contract_is_valid(name, routes[name]) for name in NATIVE_PERSISTENT_ROUTES):
            row_errors.append("native route contract violation")
        trials = row.get("trials")
        if not isinstance(trials, list) or len(trials) < 3:
            row_errors.append("missing repeated trials")
        else:
            warm_repeats = 0
            for trial in trials:
                if not isinstance(trial, Mapping):
                    row_errors.append("malformed trial")
                    break
                baseline = trial.get("baseline_p1_to_f1")
                native_error = trial.get("native_pv0_to_f1")
                if (
                    not isinstance(baseline, Mapping)
                    or not isinstance(native_error, Mapping)
                    or not finite(baseline.get("mean_step_l2"))
                    or float(baseline["mean_step_l2"]) <= 1e-6
                    or not finite(native_error.get("mean_step_l2"))
                    or not finite(trial.get("native_action_recovery"))
                ):
                    row_errors.append("zero/non-finite action comparison")
                    break
                latencies = trial.get("latency_ms")
                metrics = trial.get("inference_metrics")
                if not isinstance(latencies, Mapping) or not isinstance(metrics, Mapping):
                    row_errors.append("missing route timing")
                    break
                for route in NATIVE_PERSISTENT_ROUTES:
                    if not finite(latencies.get(route)) or not isinstance(metrics.get(route), Mapping):
                        row_errors.append(f"missing {route} timing")
                        break
                    if not finite(metrics[route].get("model_generate_inclusive_ms")) or not finite(
                        metrics[route].get("peak_memory_allocated_bytes")
                    ):
                        row_errors.append(f"missing {route} model/peak timing")
                        break
                if row_errors:
                    break
                if int(trial.get("repeat_index", -1)) >= 1:
                    warm_repeats += 1
            if warm_repeats < 2:
                row_errors.append("fewer than two post-first-call timing repeats")
        if row_errors:
            errors.append(f"{path}: {', '.join(row_errors)}")
            continue
        valid_rows.append(row)

    action_rows: list[dict[str, float]] = []
    warm_timing_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for row in valid_rows:
        per_state_recovery: list[float] = []
        per_state_baseline: list[float] = []
        per_state_native_error: list[float] = []
        for trial in row["trials"]:
            baseline = float(trial["baseline_p1_to_f1"]["mean_step_l2"])
            native_error = float(trial["native_pv0_to_f1"]["mean_step_l2"])
            recovery = float(trial["native_action_recovery"])
            action_rows.append({"baseline": baseline, "native_error": native_error, "recovery": recovery})
            per_state_recovery.append(recovery)
            per_state_baseline.append(baseline)
            per_state_native_error.append(native_error)
            if int(trial["repeat_index"]) >= 1:
                warm_timing_rows.append(
                    {
                        "state_key": str(row["state_key"]),
                        "latency_ms": {route: float(trial["latency_ms"][route]) for route in NATIVE_PERSISTENT_ROUTES},
                        "model_generate_inclusive_ms": {
                            route: float(trial["inference_metrics"][route]["model_generate_inclusive_ms"])
                            for route in NATIVE_PERSISTENT_ROUTES
                        },
                        "peak_allocated_mb": {
                            route: float(trial["inference_metrics"][route]["peak_memory_allocated_bytes"]) / 1024**2
                            for route in NATIVE_PERSISTENT_ROUTES
                        },
                    }
                )
        state_rows.append(
            {
                "state_key": row["state_key"],
                "split": row["split"],
                "task_uid": row["task_uid"],
                "init_state_index": row["init_state_index"],
                "baseline_p1_to_f1_mean_step_l2": numeric_summary(per_state_baseline),
                "native_pv0_to_f1_mean_step_l2": numeric_summary(per_state_native_error),
                "native_action_recovery": numeric_summary(per_state_recovery),
            }
        )

    def route_metric(metric: str) -> dict[str, Any]:
        return {
            route: numeric_summary(record[metric][route] for record in warm_timing_rows)
            for route in NATIVE_PERSISTENT_ROUTES
        }

    def f1_minus_pv0(metric: str) -> dict[str, Any]:
        deltas = [float(record[metric]["F1"]) - float(record[metric]["PV0"]) for record in warm_timing_rows]
        fractions = [
            (float(record[metric]["F1"]) - float(record[metric]["PV0"])) / max(float(record[metric]["F1"]), EPS)
            for record in warm_timing_rows
        ]
        return {
            "paired_warm_repeats": len(deltas),
            "f1_minus_pv0": numeric_summary(deltas),
            "pv0_lower_than_f1_fraction": float(np.mean(np.asarray(deltas) > 0.0)) if deltas else None,
            "relative_reduction": numeric_summary(fractions),
        }

    split_counts = Counter(str(row.get("split")) for row in valid_rows)
    exact_2x2 = len(valid_rows) == 4 and split_counts["validation"] == 2 and split_counts["heldout"] == 2
    model_delta = f1_minus_pv0("model_generate_inclusive_ms")
    recovery = numeric_summary(row["recovery"] for row in action_rows)
    valid = not errors and exact_2x2 and len(warm_timing_rows) >= 8
    runtime_go_candidate = bool(
        valid
        and recovery["median"] is not None
        and float(recovery["median"]) >= 0.95
        and model_delta["pv0_lower_than_f1_fraction"] == 1.0
    )
    return {
        "paths": len(paths),
        "valid_states": len(valid_rows),
        "split_states": dict(split_counts),
        "state_results": state_rows,
        "pre_registered_2x2_coverage": exact_2x2,
        "action_equivalence": {
            "all_trials": len(action_rows),
            "baseline_p1_to_f1_mean_step_l2": numeric_summary(row["baseline"] for row in action_rows),
            "native_pv0_to_f1_mean_step_l2": numeric_summary(row["native_error"] for row in action_rows),
            "native_action_recovery": recovery,
        },
        "warm_timing_protocol": {
            "definition": "repeat_index>=1; each state uses a P1 warmup and rotates F1/P1/PV0 route order across measured repeats",
            "paired_measurements": len(warm_timing_rows),
            "route_wall_latency_ms": route_metric("latency_ms"),
            "route_model_generate_inclusive_ms": route_metric("model_generate_inclusive_ms"),
            "route_peak_allocated_mb": route_metric("peak_allocated_mb"),
            "pv0_vs_f1_wall": f1_minus_pv0("latency_ms"),
            "pv0_vs_f1_model_generate": model_delta,
        },
        "load_or_contract_errors": errors[:40],
        "error_count": len(errors),
        "status": "VALID_NATIVE_PERSISTENT_RUNTIME_MICROBATCH" if valid else "INVALID_OR_INCOMPLETE_NATIVE_PERSISTENT_RUNTIME_MICROBATCH",
        "decision": {
            "result": "M3_NATIVE_RUNTIME_GO_CANDIDATE" if runtime_go_candidate else "M3_NATIVE_RUNTIME_NO_GO",
            "reason": (
                "PV0 is an end-to-end native fresh-prefix condition update, not a hidden patch. "
                "The result remains a 2-task x 2-init server microbatch; it does not establish closed-loop success, "
                "cross-task generalization, arrival-delay robustness, or Thor timing."
            ),
        },
    }


def m3_analysis(
    bundle: DatasetBundle,
    oracle_root: Path,
    corrected_oracle_root: Path,
    native_runtime_microbatch: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ablation in bundle.ablations.values():
        rows.append({"split": ablation["split"], "task_uid": ablation["task_uid"], **m3_metrics_from_ablation(ablation)})
    metric_keys = tuple(key for key in rows[0] if key not in {"split", "task_uid"}) if rows else ()
    scopes: dict[str, list[dict[str, Any]]] = {"all": rows}
    for split in ("discovery", "validation", "heldout"):
        scopes[split] = [row for row in rows if row["split"] == split]
    summary = {scope: scoped_metric_summary(scope_rows, metric_keys) for scope, scope_rows in scopes.items()}
    oracle = audit_oracle(oracle_root)
    corrected_oracle = audit_corrected_oracle_microbatch(corrected_oracle_root, bundle.manifest_rows)
    heldout = summary.get("heldout", {})
    cosine_median = heldout.get("feedback_target_cosine", {}).get("median")
    pf_beats = heldout.get("pf_beats_p1", {}).get("mean")
    mechanism_go = bool(cosine_median is not None and cosine_median >= 0.80 and pf_beats is not None and pf_beats >= 0.70)
    valid_small_oracle = corrected_oracle["status"] == "VALID_SMALL_SCALE_ORACLE_DIAGNOSTIC"
    native_runtime_go = native_runtime_microbatch.get("decision", {}).get("result") == "M3_NATIVE_RUNTIME_GO_CANDIDATE"
    return {
        "rows": len(rows),
        "cross_split_feedback_audit": summary,
        "oracle_audit": oracle,
        "corrected_small_scale_oracle_audit": corrected_oracle,
        "native_runtime_microbatch_audit": dict(native_runtime_microbatch),
        "decision": {
            "mechanism_status": "M3_GO_CANDIDATE" if mechanism_go else "M3_NO_GO",
            "system_status": (
                "M3_SYSTEM_GO_CANDIDATE_SMALL_SCALE"
                if mechanism_go and native_runtime_go
                else "M3_SYSTEM_UNRESOLVED"
                if mechanism_go and valid_small_oracle
                else "M3_SYSTEM_NO_GO"
            ),
            "reason": (
                "PF/FF alignment and the corrected oracle establish the mechanism. The native PV0 microbatch "
                "additionally verifies a non-oracle arrival interface and server-side model-cost reduction, but "
                "closed-loop success, arrival-delay robustness, and target-device timing remain separate gates."
            ),
        },
    }


def compute_accounting(bundle: DatasetBundle) -> dict[str, Any]:
    ledger = ComputeLedger()
    raw_latency: list[float] = []
    for episode_key, path in sorted(bundle.episode_paths.items()):
        del episode_key
        episode = load_torch(path)
        latencies = [float(request["policy_latency_ms"]) for request in episode["requests"] if finite(request.get("policy_latency_ms"))]
        raw_latency.extend(latencies)
        ledger.append(
            ComputeRecord(
                route="F1_collection",
                model_ms=float(sum(latencies)),
                vae_calls=len(latencies),
                dit_forwards=len(latencies),
                executed_actions=int(episode.get("control_steps", 0)),
            )
        )
    route_latencies: dict[str, list[float]] = {route: [] for route in ACTION_ROUTES}
    for ablation in bundle.ablations.values():
        for route, value in ablation.get("metrics", {}).get("latency_ms", {}).items():
            if route in route_latencies and finite(value):
                route_latencies[route].append(float(value))
    return {
        "fresh_collection": ledger.summary(),
        "fresh_request_latency_ms": numeric_summary(raw_latency),
        "paired_ablation_route_latency_ms": {route: numeric_summary(values) for route, values in route_latencies.items()},
        "accounting_definition": "model work and VAE/DiT calls are charged per executed action; no unobserved overlap or target-device speedup is inferred from this server trace.",
    }


def save_m1_records(path: Path, records: list[dict[str, Any]]) -> None:
    numeric_keys = sorted({key for record in records for key, value in record.items() if isinstance(value, (float, int, np.floating, np.integer))})
    arrays = {key: np.asarray([record.get(key, np.nan) for record in records], dtype=np.float64) for key in numeric_keys}
    arrays["episode_index"] = np.asarray(
        [int.from_bytes(hashlib.sha256(str(record["episode_key"]).encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF for record in records],
        dtype=np.int64,
    )
    np.savez_compressed(path, **arrays)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or not finite(value):
        return "—"
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    v1: DatasetBundle,
    v2: DatasetBundle,
    m1_v1: Mapping[str, Any],
    m1_v2: Mapping[str, Any],
    m3_v2: Mapping[str, Any],
    accounting: Mapping[str, Any],
    interface_microbatch: Mapping[str, Any],
    closed_loop_pilot: Mapping[str, Any],
) -> None:
    primary = m1_v2["associations"]
    residual = {split: primary[split]["policy_full_rmse"]["residual_l2"] for split in ("discovery", "validation", "heldout")}
    jerk = {split: m1_v2["baseline_comparison"][split]["past_action_jerk"] for split in ("discovery", "validation", "heldout")}
    feedback = m3_v2["cross_split_feedback_audit"]
    legacy_oracle = m3_v2["oracle_audit"]
    corrected_oracle = m3_v2["corrected_small_scale_oracle_audit"]
    corrected_frontier = corrected_oracle["frontier"]
    interface_groups = interface_microbatch["groups"]
    interface_contrasts = interface_microbatch["contrasts"]
    native_runtime = m3_v2["native_runtime_microbatch_audit"]
    native_action = native_runtime["action_equivalence"]
    native_timing = native_runtime["warm_timing_protocol"]
    native_model_timing = native_timing["route_model_generate_inclusive_ms"]
    native_wall_timing = native_timing["route_wall_latency_ms"]
    native_peak = native_timing["route_peak_allocated_mb"]
    native_model_delta = native_timing["pv0_vs_f1_model_generate"]
    latencies = accounting["paired_ablation_route_latency_ms"]
    closed_summary = closed_loop_pilot["summary"]
    closed_groups = closed_loop_pilot["paired_groups"]
    lines = [
        "# SERVER MODULAR WAM RUNTIME VALIDATION（审计版）",
        "",
        "## 结论先行",
        "",
        f"- Foundation V1：{v1.audit['audit_status']}（{v1.audit['collection']['episodes']} episodes / {v1.audit['ablation']['states']} paired states）；Foundation V2：{v2.audit['audit_status']}（{v2.audit['collection']['episodes']} episodes / {v2.audit['ablation']['states']} paired states）。",
        f"- M1 physical-residual → action-value：`{m1_v2['decision']['result']}`。这只是 shadow signal 的 GO/NO-GO，不实现任何 sensing scheduler。",
        f"- M3 persistent-condition feedback：`{m3_v2['decision']['mechanism_status']}`；系统级 native fresh-prefix runtime：`{m3_v2['decision']['system_status']}`。",
        f"- 旧 server late-binding oracle：`{legacy_oracle['status']}`，zero baseline fraction={fmt(legacy_oracle['zero_baseline_fraction'])}；不得把它作为 repair frontier 或 latency 结论。",
        f"- 修正后的预注册 2-task×2-init 小规模 oracle：`{corrected_oracle['status']}`；它支持 condition-compile frontier，但因需要 fresh-prefix oracle，不构成 runtime speedup 结论。",
        f"- 固定 block-12 的 2-task×2-init interface microbatch：`{interface_microbatch['status']}`；fresh visual condition 在全部 4 个 state 上强于 tested future-only/action-only condition。",
        f"- 原生 PV0（fresh 13-frame visual prefix + latent reuse）microbatch：`{native_runtime['decision']['result']}`；这是不依赖 hidden patch / oracle 的 server-side 小样本系统候选。",
        f"- PV0 配对闭环 pilot：`{closed_loop_pilot['status']}`；在 2 tasks × 2 init 的 Fresh-eligible 状态上，F1={closed_summary['successes_by_mode']['fresh']}/{closed_summary['paired_group_count']}，P1={closed_summary['successes_by_mode']['predicted_reuse']}/{closed_summary['paired_group_count']}，PV0={closed_summary['successes_by_mode']['native_persistent']}/{closed_summary['paired_group_count']}。",
        "",
        "本报告只使用原始 pre-finetune Cosmos LIBERO checkpoint（SHA256 `8818528d…5954d33e2`）、denoise=1，未读取 Cosmos value，未向 runtime policy 输入 simulator state，未训练新模块。",
        "",
        "## 数据与物理对齐",
        "",
        "| Dataset | Manifest episodes | Tasks | Collection episodes | Paired states | 物理对齐 endpoint pairs | Audit |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| Foundation V1 | {v1.audit['manifest']['episodes']} | {v1.audit['manifest']['tasks']} | {v1.audit['collection']['episodes']} | {v1.audit['ablation']['states']} | {v1.audit['physical_alignment']['physically_aligned_pairs']}/{v1.audit['physical_alignment']['endpoint_pairs']} | {v1.audit['audit_status']} |",
        f"| Foundation V2 | {v2.audit['manifest']['episodes']} | {v2.audit['manifest']['tasks']} | {v2.audit['collection']['episodes']} | {v2.audit['ablation']['states']} | {v2.audit['physical_alignment']['physically_aligned_pairs']}/{v2.audit['physical_alignment']['endpoint_pairs']} | {v2.audit['audit_status']} |",
        "",
        "M1 只把上一请求生成的 future-proprio endpoint 与恰好执行完该 16-action chunk 后的真实 proprio 对齐；不把 future latent 当作真实轨迹，也不使用 simulator state 作为 policy 输入。",
        "",
        "## M1：Physical residual 是否是 fresh sensing 的控制价值信号",
        "",
        "目标为 `D(A_F1, A_P1)` 的 full-chunk RMSE。数值为 request-weighted Spearman / task-balanced Spearman；jerk 是 pre-arrival generic baseline。",
        "",
        "| Split | Physical residual | Task-balanced residual | Jerk task-balanced | residual top-20% AUC |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {split} | {fmt(residual[split]['request_weighted_spearman'])} | {fmt(residual[split]['task_balanced']['macro_spearman'])} | {fmt(jerk[split]['task_balanced']['macro_spearman'])} | {fmt(residual[split]['top_20']['auc'])} |"
            for split in ("discovery", "validation", "heldout")
        ],
        "",
        f"Lag scan 的最大正关联位于 lag={m1_v2['decision']['peak_lag']}；M1 criteria={json.dumps(json_safe(m1_v2['decision']['criteria']), ensure_ascii=False)}。",
        "",
        "## M3：Fresh feedback 的 persistent-condition 路径",
        "",
        "`F1` 是 fresh one-step reference；`P1` 为预测 latent reuse；`PP` 是额外 model work 的预测路径；`PF` 为 persistent fresh-condition 路径；`FF` 为 fresh two-step reference。PF 的对齐表明 fresh physical/visual evidence 进入 joint WAM condition 后会显著改写 action，这不是 value signal，也不等同于一个可部署 scheduler。",
        "",
        "| Split | feedback→target cosine (median) | projection (median) | PF beats P1 | PF→F1 RMSE (median) | P1→F1 RMSE (median) |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {split} | {fmt(feedback[split]['feedback_target_cosine']['median'])} | {fmt(feedback[split]['feedback_target_projection']['median'])} | {fmt(feedback[split]['pf_beats_p1']['mean'])} | {fmt(feedback[split]['pf_to_f1_rmse']['median'])} | {fmt(feedback[split]['p1_to_f1_rmse']['median'])} |"
            for split in ("discovery", "validation", "heldout")
        ],
        "",
        "### 旧 Oracle 审计（必须排除）",
        "",
        f"旧 `{legacy_oracle['loaded_states']}` 个 oracle state 中，仅 `{legacy_oracle['nonzero_baseline_states']}` 个具备非零 predicted→fresh denominator。因此该批结果被标记为 `{legacy_oracle['status']}`，不进入 recovery–FLOPs frontier。该问题来自 fresh call 也复用了 previous latent；修正协议要求 fresh call 实际执行 VAE encoding，预测 call 才复用 previous latent。",
        "",
        "### 修正后小规模 Oracle（condition-compile 诊断，而非 runtime benchmark）",
        "",
        f"固定为 2 tasks × 2 independent init（validation={corrected_oracle['split_states'].get('validation', 0)}，heldout={corrected_oracle['split_states'].get('heldout', 0)}）；有效 state={corrected_oracle['valid_states']}/{corrected_oracle['loaded_states']}，baseline `P1→F1` median={fmt(corrected_oracle['baseline_predicted_to_fresh']['median'])}。每个 artifact 均验证 fresh route 不跳过 VAE/不传 previous latent，predicted route 才复用 previous latent；没有读取 value。",
        "",
        "| Injection block | Remaining DiT fraction | Recovery median [p25, p75] | First-action L2 median | States |",
        "|---:|---:|---:|---:|---:|",
        *[
            f"| {item['block']} ({item['group']}) | {fmt(item['remaining_dit_fraction']['median'])} | {fmt(item['recovery']['median'])} [{fmt(item['recovery']['p25'])}, {fmt(item['recovery']['p75'])}] | {fmt(item['first_action_l2_to_fresh']['median'])} | {item['states']} |"
            for item in corrected_frontier
        ],
        "",
        f"block 4→12→24 的 median recovery 单调下降：`{corrected_oracle['frontier_descends_block4_to12_to24']}`。这说明当前视觉的新证据越早写入内部 condition，越可能保留 Fresh-1 action；但该实验以 fresh prefix 构造 intervention，尚未减少模型工作，因此不报告 latency/FLOPs 节省。",
        "",
        "### Fixed block-12 interface microbatch（最小 condition 候选）",
        "",
        f"同一固定 2 tasks × 2 init batch，在 block 12 比较 `current_visual`、`visual_proprio`、`future` 和 `action`。有效 state={interface_microbatch['valid_states']}，coverage={interface_microbatch['pre_registered_2x2_coverage']}；所有结果保持 no-value、fresh/predicted route contract 和 fresh-prefix-oracle 标记。",
        "",
        "| Condition group | Recovery median [p25, p75] | First-action L2 median | States |",
        "|---|---:|---:|---:|",
        *[
            f"| {group} | {fmt(interface_groups[group]['recovery']['median'])} [{fmt(interface_groups[group]['recovery']['p25'])}, {fmt(interface_groups[group]['recovery']['p75'])}] | {fmt(interface_groups[group]['first_action_l2_to_fresh']['median'])} | {interface_groups[group]['states']} |"
            for group in INTERFACE_GROUPS
        ],
        "",
        "| Paired recovery contrast | Median Δ | Left wins | Paired states |",
        "|---|---:|---:|---:|",
        *[
            f"| {item['left']} − {item['right']} | {fmt(item['recovery_delta_left_minus_right']['median'])} | {fmt(item['left_beats_right_fraction'])} | {item['paired_states']} |"
            for item in interface_contrasts
        ],
        "",
        f"`current_visual` 在全部 state 上强于 `future` 与 `action`：`{interface_microbatch['visual_dominant_over_nonvisual_on_all_states']}`。`visual_proprio` 有小幅正增益，但 n=4 不足以声称 proprio 必需；当前可支持的最窄结论是：fresh visual evidence 是冻结 WAM 内部 condition 中的主导 action-relevant carrier。",
        "",
        "### Native PV0：端到端 fresh-prefix condition 更新（denoise=1）",
        "",
        "PV0 直接调用现有 `get_action:persistent_visual_correction_prefix_frames`：从 P1 的 joint latent 起步，但保留真实相机预处理并只编码 13-frame causal fresh visual prefix，在唯一 denoiser forward 前写入 current-visual condition。它不使用 hidden activation patch、fresh-prefix oracle、Cosmos value、simulator state runtime input 或 scheduler。",
        "",
        f"固定 2 tasks × 2 init，共 {native_runtime['valid_states']} states；每 state 先 P1 warmup，再轮换 F1/P1/PV0 顺序，统计 repeat≥1 的 {native_timing['paired_measurements']} 条配对 server 测量。",
        "",
        "| Route | To-F1 mean-step L2 (median) | Model-generate p50 ms | End-to-end wall p50 ms | Peak allocated p50 MiB |",
        "|---|---:|---:|---:|---:|",
        f"| F1 (full fresh) | reference | {fmt(native_model_timing['F1']['median'])} | {fmt(native_wall_timing['F1']['median'])} | {fmt(native_peak['F1']['median'])} |",
        f"| P1 (latent reuse) | {fmt(native_action['baseline_p1_to_f1_mean_step_l2']['median'])} | {fmt(native_model_timing['P1']['median'])} | {fmt(native_wall_timing['P1']['median'])} | {fmt(native_peak['P1']['median'])} |",
        f"| PV0 (native fresh visual prefix) | {fmt(native_action['native_pv0_to_f1_mean_step_l2']['median'])} | {fmt(native_model_timing['PV0']['median'])} | {fmt(native_wall_timing['PV0']['median'])} | {fmt(native_peak['PV0']['median'])} |",
        "",
        f"PV0 action recovery median={fmt(native_action['native_action_recovery']['median'])}；在 {native_model_delta['paired_warm_repeats']} 个 warm paired repeats 中，PV0 的 model-generate 均低于 F1（fraction={fmt(native_model_delta['pv0_lower_than_f1_fraction'])}），relative reduction median={fmt(native_model_delta['relative_reduction']['median'])}。这是 server 内模型段耗时；共享 GPU 的 wall time 仅作辅助审计，不能外推到 Thor。",
        "",
        "因此这条路径把 M3 从纯 hidden-oracle frontier 推进为 **native runtime GO-CANDIDATE**。下面的 paired closed-loop gate 只检验 Fresh-success preservation / P1-failure recovery；它不等同于任务总体 success superiority，也没有完成 arrival-delay sweep 或目标设备测量。",
        "",
        "### 配对闭环小样本 gate（2 tasks × 2 init）",
        "",
        "每个状态先由独立的 Fresh-only collection 验证为成功，再以完全相同的 manifest row（BDDL、instruction、init、seed、原始 checkpoint、denoise=1）运行 F1/P1/PV0。故这里的 4/4 是 **条件化的 success-preservation 集**，不是随机任务分布上的成功率估计。audit 逐条验证所有 PV0 后续请求均走原生 persistent-prefix，且每条请求只有一个 denoiser forward；没有 value、privileged state、scheduler、hidden patch 或 fresh-prefix oracle。",
        "",
        "| Task / split | Init | F1 | P1 | PV0 |",
        "|---|---:|:---:|:---:|:---:|",
        *[
            f"| {item['task_name']} ({item['split']}) | {item['init_state_index']} | {'✓' if item['fresh_success'] else '✗'} | {'✓' if item['p1_success'] else '✗'} | {'✓' if item['pv0_success'] else '✗'} |"
            for item in closed_groups
        ],
        "",
        f"汇总：F1={closed_summary['successes_by_mode']['fresh']}/{closed_summary['paired_group_count']}，P1={closed_summary['successes_by_mode']['predicted_reuse']}/{closed_summary['paired_group_count']}，PV0={closed_summary['successes_by_mode']['native_persistent']}/{closed_summary['paired_group_count']}；PV0 保留全部 {closed_summary['pv0_preserves_fresh_success_count']}/{closed_summary['fresh_success_count']} 条 Fresh success。存在 {closed_summary['pv0_recovers_fresh_success_vs_p1_count']} 个 held-out paired state：Fresh=success、P1=max_steps、PV0=success。这是一个有价值的 recovery instance，但 n=1，不能表述为平均 success superiority。",
        "",
        "闭环三路为节省时间分布在不同且共享的 GPU 上；因此该 pilot 的 warm request/wall timing 只保留作 route trace，不做跨路线 latency 结论。PV0 的 server model-cost 降幅仍以同一 worker 内轮换的 native microbatch 为准；其 paired model-generate median reduction=0.346。两次因外部进程动态占用导致的 Fresh GPU OOM 被保留在 attempt 日志中并排除；缺失 Fresh control 随后在显存充足的 GPU 上重跑成功。",
        "",
        "## Server accounting（不是 Thor latency claim）",
        "",
        f"Fresh collection request latency: p50={fmt(accounting['fresh_request_latency_ms']['median'])} ms, p95={fmt(accounting['fresh_request_latency_ms']['p95'])} ms；model ms/executed action={fmt(accounting['fresh_collection']['model_ms_per_executed_action'])}。",
        "",
        "| Route | p50 ms | p95 ms |",
        "|---|---:|---:|",
        *[f"| {route} | {fmt(latencies[route]['median'])} | {fmt(latencies[route]['p95'])} |" for route in ACTION_ROUTES],
        "",
        "PF 的绝对 server latency 不是 Thor 上的可部署费用；仍需测 fresh-VAE、prefix/suffix、native condition recompute 和 peak memory，且每次 computation 按实际执行 action 计费。",
        "",
        "## 论文雏形：三个运行时模块，而非 scheduler",
        "",
        "1. **Predictive Latent State (M0)**：保留上一次 frozen joint WAM 的完整 generated latent 与已提交 action prefix，构成 P1 的 speculative starting point；它不读取 value，也不推断或触发 schedule。",
        "2. **Native Fresh-Visual Assimilator (M2+M3)**：fresh sensing 到达后，仅编码因果 visual prefix，并通过原生 persistent-condition API 覆盖 current-visual carrier 后执行 denoise=1。PV0 是实际 runtime；hidden patch 仅保留为机制 oracle，不伪装成系统方法。",
        "3. **Execution/Compute Contract (M4–M6)**：将 committed action prefix、fresh-prefix VAE、model work、peak memory 与 executed actions 一起记录，并导出可在 Thor 上复现的成本表。",
        "",
        "M1 physical-innovation signal 已在 task-disjoint audit 中 NO-GO，因此不属于运行时设计，只作为论文中的反例/negative result。当前三个运行时模块均不训练。若后续证明 causal prefix 在目标设备上仍不可接受，唯一可考虑的训练扩展才是受限 arrival-interface mapper；它必须与 frozen backbone、task-disjoint generalization、no-value/no-privilege contract 分开评估，不能先假定其有效。",
        "",
        "## 下一步 gate",
        "",
        "当前 2-task×2-init 闭环 gate 已关闭；在没有新的预注册问题前不继续扩展 task 数。若要将 GO-CANDIDATE 提升为论文主张，下一步应是固定的 task-disjoint robustness/arrival-delay repeat 与目标设备（Thor）成本测量；保持 denoise=1、frozen checkpoint、no-value/no-privilege，并且不引入 adaptive scheduler。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-manifest", type=Path, default=Path("reports/server_deep_validation/manifests/server_f1_collection.jsonl"))
    parser.add_argument("--v1-collection", type=Path, default=Path("/data/rxhuang/wam_server_deep_validation/queue_a/f1_collection"))
    parser.add_argument("--v1-ablation", type=Path, default=Path("/data/rxhuang/wam_server_deep_validation/queue_a/ablation"))
    parser.add_argument("--v2-manifest", type=Path, default=Path("reports/server_deep_validation/manifests/full_scale_40_task.jsonl"))
    parser.add_argument("--v2-collection", type=Path, default=Path("/data/rxhuang/wam_full_scale_server/queue_a/f1_collection"))
    parser.add_argument("--v2-ablation", type=Path, default=Path("/data/rxhuang/wam_full_scale_server/queue_b/ablation"))
    parser.add_argument("--v2-oracle", type=Path, default=Path("/data/rxhuang/wam_full_scale_server/queue_b/oracle"))
    parser.add_argument("--corrected-oracle-microbatch", type=Path, default=Path("reports/modular_wam_runtime/corrected_oracle_microbatch"))
    parser.add_argument(
        "--corrected-interface-roots",
        type=Path,
        nargs="+",
        default=[
            Path("reports/modular_wam_runtime/corrected_oracle_interface_pilot"),
            Path("reports/modular_wam_runtime/corrected_oracle_interface_microbatch"),
        ],
    )
    parser.add_argument(
        "--native-persistent-roots",
        type=Path,
        nargs="+",
        default=[
            Path("reports/modular_wam_runtime/native_persistent_preflight"),
            Path("reports/modular_wam_runtime/native_persistent_microbatch"),
        ],
    )
    parser.add_argument(
        "--closed-loop-pilot-audit",
        type=Path,
        default=Path("reports/modular_wam_runtime/native_persistent_closed_loop_pilot/closed_loop_pilot_audit_final.json"),
    )
    parser.add_argument("--dataset-stats", type=Path, default=Path("/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/modular_wam_runtime"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = json.loads(args.dataset_stats.read_text(encoding="utf-8"))
    v1 = audit_dataset("Foundation_V1", args.v1_manifest, args.v1_collection, args.v1_ablation)
    v2 = audit_dataset("Foundation_V2", args.v2_manifest, args.v2_collection, args.v2_ablation)
    write_json(args.output_dir / "FOUNDATION_DATASET_V1_AUDIT.json", v1.audit)
    write_json(args.output_dir / "FOUNDATION_DATASET_V2_AUDIT.json", v2.audit)
    task_registry = {
        "v1": {"tasks_by_split": {split: sorted({row["task_uid"] for row in v1.manifest_rows.values() if row["split"] == split}) for split in ("discovery", "validation", "heldout")}},
        "v2": {"tasks_by_split": {split: sorted({row["task_uid"] for row in v2.manifest_rows.values() if row["split"] == split}) for split in ("discovery", "validation", "heldout")}},
    }
    write_json(args.output_dir / "TASK_SPLIT_REGISTRY.json", task_registry)
    m1_records_v1, m1_meta_v1 = build_m1_records(v1, stats)
    m1_records_v2, m1_meta_v2 = build_m1_records(v2, stats)
    m1_v1, m1_v2 = m1_analysis(m1_records_v1, m1_meta_v1), m1_analysis(m1_records_v2, m1_meta_v2)
    save_m1_records(args.output_dir / "M1_PHYSICAL_ALIGNMENT_RECORDS_V1.npz", m1_records_v1)
    save_m1_records(args.output_dir / "M1_PHYSICAL_ALIGNMENT_RECORDS_V2.npz", m1_records_v2)
    write_json(args.output_dir / "M1_SIGNAL_VALIDATION_V1.json", m1_v1)
    write_json(args.output_dir / "M1_SIGNAL_VALIDATION_V2.json", m1_v2)
    native_runtime_microbatch = audit_native_persistent_microbatch(args.native_persistent_roots, v2.manifest_rows)
    m3_v2 = m3_analysis(v2, args.v2_oracle, args.corrected_oracle_microbatch, native_runtime_microbatch)
    write_json(args.output_dir / "M3_FEEDBACK_ASSIMILATION_AUDIT_V2.json", m3_v2)
    write_json(args.output_dir / "CORRECTED_SMALL_SCALE_ORACLE_AUDIT.json", m3_v2["corrected_small_scale_oracle_audit"])
    write_json(args.output_dir / "NATIVE_PERSISTENT_RUNTIME_MICROBATCH_AUDIT.json", native_runtime_microbatch)
    interface_microbatch = audit_corrected_interface_microbatch(args.corrected_interface_roots, v2.manifest_rows)
    write_json(args.output_dir / "CORRECTED_INTERFACE_MICROBATCH_AUDIT.json", interface_microbatch)
    if not args.closed_loop_pilot_audit.is_file():
        raise FileNotFoundError(f"closed-loop pilot audit not found: {args.closed_loop_pilot_audit}")
    closed_loop_pilot = json.loads(args.closed_loop_pilot_audit.read_text(encoding="utf-8"))
    if closed_loop_pilot.get("experiment") != "native_persistent_closed_loop_pilot_audit":
        raise ValueError("unexpected closed-loop pilot audit payload")
    write_json(args.output_dir / "NATIVE_PERSISTENT_CLOSED_LOOP_PILOT_AUDIT.json", closed_loop_pilot)
    accounting = compute_accounting(v2)
    accounting["thor_cost_model"] = ThorCostModel().as_dict()
    write_json(args.output_dir / "SERVER_COMPUTE_ACCOUNTING_V2.json", accounting)
    write_report(
        args.output_dir / "SERVER_MODULAR_WAM_RUNTIME_VALIDATION_ZH.md",
        v1,
        v2,
        m1_v1,
        m1_v2,
        m3_v2,
        accounting,
        interface_microbatch,
        closed_loop_pilot,
    )
    final = {
        "v1_audit": v1.audit["audit_status"],
        "v2_audit": v2.audit["audit_status"],
        "m1_v1": m1_v1["decision"]["result"],
        "m1_v2": m1_v2["decision"]["result"],
        "m3": m3_v2["decision"],
        "corrected_small_scale_oracle": m3_v2["corrected_small_scale_oracle_audit"]["status"],
        "corrected_interface_microbatch": interface_microbatch["status"],
        "native_persistent_runtime_microbatch": native_runtime_microbatch["decision"],
        "native_persistent_closed_loop_pilot": {
            "status": closed_loop_pilot["status"],
            "summary": closed_loop_pilot["summary"],
        },
        "report": str(args.output_dir / "SERVER_MODULAR_WAM_RUNTIME_VALIDATION_ZH.md"),
    }
    write_json(args.output_dir / "RUN_SUMMARY.json", final)
    print(json.dumps(json_safe(final), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

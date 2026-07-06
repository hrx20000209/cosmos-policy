"""汇总 SO101 parallel sweep，并按离线动作指标选择每个实验的最佳 checkpoint。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

CHECKPOINTS = ("iter_000000500", "iter_000001000", "iter_000001500", "iter_000002000")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_root", type=Path, required=True)
    parser.add_argument("--episode_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _finite_mean(values: list[float | None]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(finite) if finite else math.nan


def _dict_mean(data: dict[str, float | None]) -> float:
    return _finite_mean(list(data.values()))


def _episode_jsons(root: Path) -> list[Path]:
    return sorted(root.glob("episode_*/episode_*_stride_*.json"))


def _row(experiment: str, checkpoint: str, sample_path: Path, episode_root: Path) -> dict[str, Any]:
    sample = _load(sample_path)
    episode_paths = _episode_jsons(episode_root)
    if len(episode_paths) != 5:
        raise RuntimeError(f"{experiment}/{checkpoint} full-episode 结果数={len(episode_paths)}，期望 5")
    episodes = [_load(path) for path in episode_paths]
    row = {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "step": int(checkpoint.rsplit("_", 1)[1]),
        "sample_first5_mae_mean": float(sample["first_5_step_mae"]),
        "sample_full50_mae_mean": float(sample["full_50_step_mae"]),
        "sample_direction_agreement_mean": _dict_mean(sample["per_joint_direction_agreement"]),
        "sample_gripper_mae": float(sample["gripper_mae"]),
        "sample_mean_collapse_flag": bool(sample.get("possible_temporal_action_collapse", False)),
        "full_episode_count": len(episodes),
        "full_first_query_first5_mae_mean": _finite_mean(
            [item.get("first_query_first_5_step_mae") for item in episodes]
        ),
        "full_all_query_first5_mae_mean": _finite_mean(
            [item.get("all_query_first_5_step_mae") for item in episodes]
        ),
        "full_stitched_mae_mean": _finite_mean([item.get("stitched_trajectory_mae") for item in episodes]),
        "full_direction_agreement_mean": _finite_mean(
            [_dict_mean(item["per_joint_direction_agreement"]) for item in episodes]
        ),
        "full_gripper_mae_mean": _finite_mean([item.get("gripper_mae") for item in episodes]),
        "full_episode_jump_score": _finite_mean([item.get("normalized_action_jump_score") for item in episodes]),
        "full_mean_collapse_rate": statistics.mean(
            bool(item.get("mean_collapse_risk", False)) for item in episodes
        ),
        "full_constant_output_rate": statistics.mean(
            bool(item.get("constant_output_flag", False)) for item in episodes
        ),
        "full_discontinuity_rate": statistics.mean(
            bool(item.get("obvious_action_discontinuity", False)) for item in episodes
        ),
        "full_reverse_gripper_rate": statistics.mean(
            bool(item.get("obvious_reverse_gripper", False)) for item in episodes
        ),
    }
    return row


def _best_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """严格按用户指定优先级做 lexicographic checkpoint 选择。"""
    return (
        row["sample_first5_mae_mean"],
        -row["sample_direction_agreement_mean"],
        row["sample_gripper_mae"],
        row["full_episode_jump_score"],
        row["sample_mean_collapse_flag"] or row["full_mean_collapse_rate"] > 0,
        row["full_constant_output_rate"] > 0,
        row["full_discontinuity_rate"] > 0,
    )


def _analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda item: item["step"])
    first5 = [row["sample_first5_mae_mean"] for row in ordered]
    best = min(ordered, key=_best_key)
    best_to_last = ordered[-1]["sample_first5_mae_mean"] / max(best["sample_first5_mae_mean"], 1e-9)
    step_1000 = next(row for row in ordered if row["step"] == 1000)
    later = [row for row in ordered if row["step"] > 1000]
    overfit_after_1000 = any(
        row["sample_first5_mae_mean"] > step_1000["sample_first5_mae_mean"] * 1.05 for row in later
    )
    return {
        "best_checkpoint": best["checkpoint"],
        "best_step": best["step"],
        "stable_first5_improvement": all(next_value <= value for value, next_value in zip(first5, first5[1:])),
        "overfit_after_1000": overfit_after_1000,
        "step_2000_worse_than_best": best_to_last > 1.05,
        "step_500_vs_base": "unknown_base_not_evaluated",
        "resume_recommendation": best["checkpoint"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field, "")
            values.append(f"{value:.5f}" if isinstance(value, float) else str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body]) + "\n"


def main() -> None:
    args = parse_args()
    experiments = sorted(path.name for path in args.sample_root.iterdir() if path.is_dir() and path.name != "logs")
    if not experiments:
        raise RuntimeError(f"没有发现 sweep experiment：{args.sample_root}")
    dynamics = []
    summary = []
    for experiment in experiments:
        experiment_rows = []
        for checkpoint in CHECKPOINTS:
            sample_path = args.sample_root / experiment / checkpoint / "sample_level" / "summary.json"
            episode_root = args.episode_root / experiment / checkpoint
            experiment_rows.append(_row(experiment, checkpoint, sample_path, episode_root))
        analysis = _analysis(experiment_rows)
        for row in experiment_rows:
            row.update(analysis)
            row["is_best_checkpoint"] = row["checkpoint"] == analysis["best_checkpoint"]
        dynamics.extend(experiment_rows)
        best_row = next(row for row in experiment_rows if row["is_best_checkpoint"])
        summary.append(best_row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    dynamics_fields = list(dynamics[0].keys())
    summary_fields = list(summary[0].keys())
    _write_csv(args.output_root / "training_dynamics.csv", dynamics, dynamics_fields)
    _write_csv(args.output_root / "summary.csv", summary, summary_fields)
    (args.output_root / "training_dynamics.md").write_text(_markdown_table(dynamics, dynamics_fields))
    (args.output_root / "summary.md").write_text(_markdown_table(summary, summary_fields))
    print(f"已汇总 {len(experiments)} 个实验；输出目录：{args.output_root}")
    for row in summary:
        print(
            f"{row['experiment']}: best={row['best_checkpoint']}, "
            f"first5={row['sample_first5_mae_mean']:.4f}, "
            f"direction={row['sample_direction_agreement_mean']:.4f}"
        )


if __name__ == "__main__":
    main()

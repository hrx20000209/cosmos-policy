"""汇总 action-focused sweep 的 sampling、full-episode 与 teacher-forced 指标。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def _mean(values: list[float | None]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.mean(finite) if finite else math.nan


def _dict_mean(data: dict[str, float | None]) -> float:
    return _mean(list(data.values()))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def _trainable(log_path: Path) -> tuple[int | None, float | None, bool]:
    if not log_path.is_file():
        return None, None, False
    text = log_path.read_text(errors="replace")
    matches = re.findall(r"可训练参数=([0-9,]+)；比例=([0-9.]+)%", text)
    oom = "CUDA out of memory" in text or "torch.OutOfMemoryError" in text
    if not matches:
        return None, None, oom
    count, ratio = matches[-1]
    return int(count.replace(",", "")), float(ratio) / 100.0, oom


def _recommendation(row: dict[str, Any]) -> str:
    if row["oom"]:
        return "OOM；不作为候选"
    if not math.isfinite(row["sample_first5_mae"]):
        return "结果不完整"
    if row["mean_collapse_flag"] or row["constant_output_flag"]:
        return "存在 collapse/constant output；禁止真机"
    if row["direction_agreement"] < 0.6:
        return "方向一致率不足；继续诊断 sampling/representation"
    return "离线候选；仍需更多 held-out 验证"


def _parse_manifest(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open() as file:
        for raw in file:
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\n").split("\t")
            if len(fields) != 9:
                raise RuntimeError(f"manifest 行应有 9 列，实际 {len(fields)}：{raw.rstrip()}")
            rows.append(
                dict(
                    exp_name=fields[0],
                    finetune_mode=fields[1],
                    loss_mode=fields[2],
                    action_loss_multiplier=fields[3],
                    normalize_masked_loss=fields[4],
                    max_step=fields[5],
                    run_dir=fields[6],
                    gpu=fields[7],
                    train_log=fields[8],
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    for experiment in _parse_manifest(args.manifest):
        trainable_params, trainable_ratio, oom = _trainable(Path(experiment["train_log"]))
        max_step = int(experiment["max_step"])
        for step in range(500, max_step + 1, 500):
            checkpoint = f"iter_{step:09d}"
            root = args.eval_root / experiment["exp_name"] / checkpoint
            sample = _load(root / "sample_level" / "summary.json")
            teacher = _load(root / "teacher_forced" / "summary.json")
            episode_files = sorted((root / "full_episode").glob("episode_*/episode_*_stride_10.json"))
            episodes = [_load(path) for path in episode_files]
            directions = [_dict_mean(item.get("per_joint_direction_agreement", {})) for item in episodes]
            row = {
                "exp_name": experiment["exp_name"],
                "finetune_mode": experiment["finetune_mode"],
                "trainable_params": trainable_params,
                "trainable_ratio": trainable_ratio,
                "loss_mode": experiment["loss_mode"],
                "action_loss_multiplier": int(experiment["action_loss_multiplier"]),
                "normalize_masked_loss": experiment["normalize_masked_loss"].lower() == "true",
                "checkpoint_step": step,
                "sample_first5_mae": float(sample.get("first_5_step_mae", math.nan)),
                "sample_full50_mae": float(sample.get("full_50_step_mae", math.nan)),
                "full_episode_mae": _mean([item.get("stitched_trajectory_mae") for item in episodes]),
                "direction_agreement": _mean(directions),
                "gripper_mae": _mean([item.get("gripper_mae") for item in episodes]),
                "gripper_direction_agreement": _mean(
                    [item.get("gripper_direction_agreement") for item in episodes]
                ),
                "scale_ratio_mean": _mean(
                    [_dict_mean(item.get("per_joint_scale_ratio", {})) for item in episodes]
                ),
                "temporal_scale_ratio_mean": _mean(
                    [_dict_mean(item.get("per_joint_mean_chunk_temporal_scale_ratio", {})) for item in episodes]
                ),
                "full_episode_jump_score": _mean(
                    [item.get("normalized_action_jump_score") for item in episodes]
                ),
                "constant_output_flag": any(item.get("constant_output_flag", False) for item in episodes),
                "mean_collapse_flag": bool(sample.get("possible_temporal_action_collapse", False))
                or any(item.get("mean_collapse_risk", False) for item in episodes),
                "teacher_forced_low_sigma_mae": float(
                    teacher.get("low_sigma", {}).get("mae", math.nan)
                ),
                "teacher_forced_random_sigma_mae": float(
                    teacher.get("random_training_sigma", {}).get("mae", math.nan)
                ),
                "full_episode_count": len(episodes),
                "oom": oom,
            }
            row["recommendation"] = _recommendation(row)
            rows.append(row)

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        def minimum(value: float) -> float:
            return value if math.isfinite(value) else math.inf

        def maximum(value: float) -> float:
            return -value if math.isfinite(value) else math.inf

        return (
            minimum(row["teacher_forced_low_sigma_mae"]),
            minimum(row["sample_first5_mae"]),
            maximum(row["direction_agreement"]),
            minimum(row["gripper_mae"]),
            minimum(row["full_episode_jump_score"]),
            row["mean_collapse_flag"],
            row["constant_output_flag"],
        )

    rows.sort(key=sort_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(f"{row[field]:.5f}" if isinstance(row[field], float) else str(row[field]) for field in fields)
            + " |"
        )
    (args.output_dir / "summary.md").write_text("\n".join([header, separator, *body]) + "\n")
    print(f"汇总 {len(rows)} 个 checkpoint 行到 {args.output_dir}")


if __name__ == "__main__":
    main()

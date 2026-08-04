#!/usr/bin/env python3
"""Estimate wall time, GPU-hours, and disk use from completed formal episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")


def load_complete_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    manifest = load_complete_jsonl(args.manifest)
    expected = {row["episode_key"] for row in manifest}
    latest: dict[str, dict] = {}
    for path in OUTPUT.glob("raw/*/episodes.shard*.jsonl"):
        for row in load_complete_jsonl(path):
            if row.get("episode_key") in expected:
                latest[row["episode_key"]] = row

    wall_seconds = [
        float(row["episode_wall_clock_time_s"])
        for row in latest.values()
        if row.get("episode_wall_clock_time_s") is not None
    ]
    mean_seconds = sum(wall_seconds) / len(wall_seconds) if wall_seconds else 0.0
    remaining = len(expected) - len(latest)
    estimated_remaining_wall_seconds = remaining * mean_seconds / max(args.workers, 1)
    by_step: dict[int, list[float]] = defaultdict(list)
    for row in latest.values():
        if row.get("episode_wall_clock_time_s") is not None:
            by_step[int(row["denoising_steps"])].append(float(row["episode_wall_clock_time_s"]))

    disk = shutil.disk_usage(OUTPUT)
    result = {
        "phase": args.phase,
        "expected_episodes": len(expected),
        "completed_episodes": len(latest),
        "remaining_episodes": remaining,
        "mean_episode_wall_seconds": mean_seconds,
        "observed_gpu_hours": sum(wall_seconds) / 3600.0,
        "projected_total_gpu_hours": len(expected) * mean_seconds / 3600.0,
        "estimated_remaining_wall_minutes": estimated_remaining_wall_seconds / 60.0,
        "workers": args.workers,
        "mean_episode_wall_seconds_by_step": {
            str(step): sum(values) / len(values) for step, values in sorted(by_step.items())
        },
        "output_bytes": size_bytes(OUTPUT),
        "filesystem_free_bytes": disk.free,
        "filesystem_total_bytes": disk.total,
    }
    target = (
        Path(__file__).resolve().parents[1]
        / "summaries"
        / f"runtime_estimate_{args.phase}.json"
    )
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = Path(__file__).resolve().parents[1] / "reports/runtime_estimate.md"
    mib = result["output_bytes"] / 2**20
    free_tib = result["filesystem_free_bytes"] / 2**40
    by_step_lines = "\n".join(
        f"| {step} | {seconds:.2f} |"
        for step, seconds in result["mean_episode_wall_seconds_by_step"].items()
    )
    report.write_text(
        "# 正式评测运行预算\n\n"
        f"- 阶段：`{args.phase}`\n"
        f"- 已完成：{len(latest)}/{len(expected)} episodes\n"
        f"- 已完成样本平均墙钟时间：{mean_seconds:.2f} 秒/episode\n"
        f"- 预计总 GPU 时间：{result['projected_total_gpu_hours']:.2f} GPU·小时\n"
        f"- 按 {args.workers} 个并行 worker 估计剩余纯执行时间："
        f"{result['estimated_remaining_wall_minutes']:.2f} 分钟\n"
        f"- 当前实验输出占用：{mib:.2f} MiB\n"
        f"- 输出文件系统剩余：{free_tib:.2f} TiB\n\n"
        "该估算以已完成 episode 的实际墙钟时间线性外推，不包含后续分片的模型"
        "重复加载、任务难度分布变化和失败重试开销，因此只作资源预算，不作完成时间承诺。\n\n"
        "| Denoising steps | 平均 episode 墙钟时间（秒） |\n"
        "|---:|---:|\n"
        f"{by_step_lines}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

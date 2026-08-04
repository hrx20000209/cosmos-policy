#!/usr/bin/env python3
"""Launch independent per-config workers over a GPU pool."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT / "experiments/cosmos_denoising_libero_pro"
RUNNER = EXPERIMENT / "scripts/run_manifest.py"
SPLITTER = EXPERIMENT / "scripts/split_manifest.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT / "configs/sweep.yaml",
    )
    parser.add_argument("--phase", required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-reset-screenshot", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--limit-per-config", type=int)
    parser.add_argument("--shards-per-config", type=int, default=1)
    args = parser.parse_args()
    split_dir = EXPERIMENT / "manifests/split" / args.phase
    subprocess.run(
        [sys.executable, str(SPLITTER), str(args.manifest), "--output-dir", str(split_dir)],
        check=True,
        cwd=PROJECT,
    )
    index = json.loads((split_dir / "index.json").read_text(encoding="utf-8"))
    job_list = [
        (config_id, metadata, shard_index)
        for config_id, metadata in sorted(index.items())
        for shard_index in range(args.shards_per_config)
    ]
    random.Random(195).shuffle(job_list)
    jobs = deque(job_list)
    gpu_pool = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_pool:
        raise SystemExit("empty GPU pool")
    # A GPU may appear more than once to create independent simulator/model
    # slots. Jobs still come from this single queue, so no config/shard is
    # assigned twice.
    active: dict[int, tuple[subprocess.Popen, object, str, str, str]] = {}
    logs = EXPERIMENT / "logs" / args.phase
    logs.mkdir(parents=True, exist_ok=True)
    launch_stamp = time.strftime("%Y%m%d-%H%M%S")
    failures = []
    while jobs or active:
        for slot, gpu in enumerate(gpu_pool):
            if slot in active or not jobs:
                continue
            config_id, metadata, shard_index = jobs.popleft()
            run_id = (
                f"{args.phase}-{config_id}-shard{shard_index:02d}-"
                f"{launch_stamp}-{os.getpid()}"
            )
            log_path = logs / f"{run_id}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(RUNNER),
                "--manifest",
                metadata["path"],
                "--run-id",
                run_id,
                "--config",
                str(args.config.resolve()),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(args.shards_per_config),
            ]
            if args.record_video:
                command.append("--record-video")
            if args.record_reset_screenshot:
                command.append("--record-reset-screenshot")
            if args.retry_failures:
                command.append("--retry-failures")
            if args.limit_per_config is not None:
                command.extend(["--limit", str(args.limit_per_config)])
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "EVAL_PHYSICAL_GPU": gpu,
                    "MUJOCO_GL": "osmesa",
                    "PYOPENGL_PLATFORM": "osmesa",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "TOKENIZERS_PARALLELISM": "false",
                    "LD_LIBRARY_PATH": (
                        "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu:"
                        + environment.get("LD_LIBRARY_PATH", "")
                    ),
                }
            )
            print(f"launch gpu={gpu} config={config_id} episodes={metadata['episodes']} log={log_path}")
            process = subprocess.Popen(
                command,
                cwd=PROJECT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active[slot] = (
                process,
                log_handle,
                f"{config_id}:shard{shard_index}",
                str(log_path),
                gpu,
            )
        time.sleep(2)
        for slot, (
            process,
            log_handle,
            config_id,
            log_path,
            gpu,
        ) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            print(f"finish gpu={gpu} config={config_id} returncode={return_code}")
            if return_code != 0:
                failures.append(
                    {"gpu": gpu, "config_id": config_id, "returncode": return_code, "log": log_path}
                )
            del active[slot]
    result = {
        "phase": args.phase,
        "manifest": str(args.manifest.resolve()),
        "gpus": gpu_pool,
        "configs": len(index),
        "shards_per_config": args.shards_per_config,
        "failures": failures,
    }
    result_path = logs / f"launch_summary_{launch_stamp}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

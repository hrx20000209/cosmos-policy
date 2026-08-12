#!/usr/bin/env python3
"""Resumable VRAM-aware supervisor for the small R2 shadow-label collection.

It never touches processes it did not start.  Each worker runs the frozen R2
backbone, while its post-control F1/P1/PV0 computations are written as offline
labels only.  A completed JSON payload is the sole completion marker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REQUIRED_FREE_MIB, RESERVE_MIB = 14_000, 500


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def free_gpus() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return [
        int(index.strip())
        for line in output.splitlines()
        for index, free in [line.split(",")]
        if int(free.strip()) >= REQUIRED_FREE_MIB + RESERVE_MIB
    ]


def valid_output(path: Path) -> bool:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    labels = row.get("shadow_validity_labels", [])
    return (
        row.get("status") == "PASS"
        and row.get("mode") == "pv0_r2"
        and row.get("shadow_labels_runtime_policy_input") is False
        and len(labels) >= 1
        and all(
            label.get("shadow_only") is True
            and label.get("denoising_steps_per_shadow_route") == 1
            and label.get("value_used") is False
            for label in labels
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=4.0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    episodes, logs = run_dir / "episodes", run_dir / "logs"
    episodes.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line]
    jobs = [
        {"id": row["episode_key"], "episode_key": row["episode_key"], "split": row["split"], "task_uid": row["task_uid"], "attempts": 0, "status": "pending"}
        for row in rows
    ]
    if len({job["id"] for job in jobs}) != len(jobs):
        raise RuntimeError("duplicate episode key")
    running: dict[int, tuple[subprocess.Popen, dict, object]] = {}
    state_path = run_dir / "COLLECTION_STATE.json"
    while True:
        for gpu, (process, job, log) in list(running.items()):
            if process.poll() is None:
                continue
            log.close()
            output = episodes / f"pv0_r2_{job['episode_key']}.json"
            if process.returncode == 0 and valid_output(output):
                job["status"] = "completed"
            elif job["attempts"] < 2:
                job["status"] = "pending"
            else:
                job["status"] = "failed"
            running.pop(gpu)
        for gpu in free_gpus():
            if gpu in running:
                continue
            job = next((item for item in jobs if item["status"] == "pending"), None)
            if job is None:
                break
            job["attempts"] += 1
            job["status"] = "running"
            output = episodes / f"pv0_r2_{job['episode_key']}.json"
            trace = episodes / f"pv0_r2_{job['episode_key']}_traces.json"
            command = [
                sys.executable,
                "experiments/server_deep_validation/run_pv0_closed_loop_episode.py",
                "--mode", "pv0_r2",
                "--episode-key", job["episode_key"],
                "--output", str(output),
                "--trace-output", str(trace),
                "--manifest", str(args.manifest.resolve()),
                "--memory-fraction", "0.35",
                "--shadow-validity-labels",
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "EVAL_PHYSICAL_GPU": str(gpu),
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
                    "MUJOCO_EGL_DEVICE_ID": str(gpu),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            log = (logs / f"pv0_r2_{job['episode_key']}.attempt{job['attempts']}.log").open("w", encoding="utf-8")
            running[gpu] = (subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT), job, log)
        atomic_write(
            state_path,
            {
                "schema_version": 1,
                "collection": "EXECUTION_VALIDITY_DATASET_small_shadow_r2",
                "manifest": str(args.manifest.resolve()),
                "route_executed": "pv0_r2",
                "shadow_labels_runtime_policy_input": False,
                "denoising_steps": 1,
                "value_used": False,
                "required_free_mib": REQUIRED_FREE_MIB,
                "reserve_mib": RESERVE_MIB,
                "running_gpus": sorted(running),
                "counts": {state: sum(job["status"] == state for job in jobs) for state in ("pending", "running", "completed", "failed")},
                "jobs": jobs,
            },
        )
        if not running and not any(job["status"] == "pending" for job in jobs):
            break
        time.sleep(args.poll_seconds)
    print(json.dumps(json.loads(state_path.read_text(encoding="utf-8")), ensure_ascii=False))


if __name__ == "__main__":
    main()

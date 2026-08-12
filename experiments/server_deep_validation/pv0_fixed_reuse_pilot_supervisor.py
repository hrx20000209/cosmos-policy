#!/usr/bin/env python3
"""Small, idempotent, free-VRAM-scheduled PV0/P1 fixed-reuse pilot.

The supervisor launches at most one Cosmos worker per eligible GPU, never
signals external processes, and retries a failed shard once.  It intentionally
contains no adaptive rule or action scaling: its only job is to select the
safe fixed reuse depth before controller discovery.
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
# ``native_persistent`` is R0 (PV0 always); do not run its identical alias
# ``pv0_r0`` as a redundant episode.
ROUTES = ("fresh", "native_persistent", "pv0_r1", "pv0_r2", "pv0_r3")
# Empirical telemetry preflight for this exact frozen 2B/denoise=1 worker
# reached about 14.0 GiB peak.  On a shared card, 14.5 GiB free leaves a
# measured 0.5 GiB model margin plus ~0.4 GiB scheduler margin.  This admits
# correctness shards on partially occupied GPUs without relying on utilization.
REQUIRED_FREE_MIB, RESERVE_MIB = 14_000, 500


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def free_gpus() -> list[int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    result = []
    for line in output.splitlines():
        index, free = (part.strip() for part in line.split(","))
        if int(free) >= REQUIRED_FREE_MIB + RESERVE_MIB:
            result.append(int(index))
    return result


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    heldout_tasks = sorted({row["task_uid"] for row in rows if row["split"] == "heldout"})
    if len(heldout_tasks) < 8:
        raise RuntimeError("expected at least eight heldout tasks")
    selected = set(heldout_tasks[:8])
    chosen = []
    for task in heldout_tasks[:8]:
        options = sorted((row for row in rows if row["task_uid"] == task), key=lambda row: row["init_state_index"])
        chosen.append(options[0])
    if len(chosen) != 8 or {row["task_uid"] for row in chosen} != selected:
        raise RuntimeError("pilot task selection failed")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--routes",
        default=",".join(ROUTES),
        help="comma-separated fixed routes; stale_r2 is used for the matched-stale collection",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    episodes, logs = run_dir / "episodes", run_dir / "logs"
    episodes.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    selected_routes = tuple(item.strip() for item in args.routes.split(",") if item.strip())
    if not selected_routes:
        raise RuntimeError("at least one route is required")
    jobs = [
        {"id": f"{route}:{row['episode_key']}", "route": route, "row": row, "attempts": 0, "status": "pending"}
        for row in load_rows(args.manifest)
        for route in selected_routes
    ]
    running: dict[int, tuple[subprocess.Popen, dict, object]] = {}
    state_path = run_dir / "PILOT_STATE.json"
    while True:
        for gpu, (process, job, log) in list(running.items()):
            if process.poll() is None:
                continue
            log.close()
            output = episodes / f"{job['route']}_{job['row']['episode_key']}.json"
            passed = process.returncode == 0 and output.is_file() and json.loads(output.read_text(encoding="utf-8")).get("status") == "PASS"
            if passed:
                job["status"] = "completed"
            elif job["attempts"] < 2:
                job["status"] = "pending"
            else:
                job["status"] = "failed"
            running.pop(gpu)
        for gpu in free_gpus():
            if gpu in running:
                continue
            pending = next((job for job in jobs if job["status"] == "pending"), None)
            if pending is None:
                break
            pending["attempts"] += 1
            pending["status"] = "running"
            row, route = pending["row"], pending["route"]
            output = episodes / f"{route}_{row['episode_key']}.json"
            trace = episodes / f"{route}_{row['episode_key']}_traces.json"
            command = [
                sys.executable, "experiments/server_deep_validation/run_pv0_closed_loop_episode.py",
                "--mode", route, "--episode-key", row["episode_key"], "--output", str(output), "--trace-output", str(trace),
                "--manifest", str(args.manifest.resolve()), "--memory-fraction", "0.35",
            ]
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "EVAL_PHYSICAL_GPU": str(gpu), "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json", "MUJOCO_EGL_DEVICE_ID": str(gpu), "PYTHONUNBUFFERED": "1"})
            log = (logs / f"{route}_{row['episode_key']}.attempt{pending['attempts']}.log").open("w", encoding="utf-8")
            running[gpu] = (subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT), pending, log)
        atomic_json(state_path, {"schema_version": 1, "pilot": "fixed_pv0_p1_reuse_depth", "routes": selected_routes, "selected_tasks": sorted({job['row']['task_uid'] for job in jobs}), "jobs": [{key: value for key, value in job.items() if key != "row"} for job in jobs], "running_gpus": sorted(running), "counts": {status: sum(job['status'] == status for job in jobs) for status in ("pending", "running", "completed", "failed")}, "runtime_only_feedback": True, "denoising_steps": 1, "value_used": False})
        if not running and not any(job["status"] == "pending" for job in jobs):
            break
        time.sleep(args.poll_seconds)
    print(json.dumps(json.loads(state_path.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()

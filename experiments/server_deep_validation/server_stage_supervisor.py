"""Launch a resumable post-collection stage when Queue A is complete."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from server_queue_supervisor import gpu_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/home/rxhuang/Projects/cosmos-policy"))
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default="/home/rxhuang/Projects/cosmos-policy/.venv/bin/python")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--hard-stop-epoch", type=float, required=True)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    args = parser.parse_args()
    args.status_dir.mkdir(parents=True, exist_ok=True)
    stage_name = args.script.stem
    active: dict[int, dict] = {}
    finished: list[dict] = []
    failed_attempts: dict[int, int] = {}

    def write_status(reason: str) -> None:
        status = {
            "timestamp_ns": time.time_ns(),
            "reason": reason,
            "hard_stop_epoch": args.hard_stop_epoch,
            "wait_for": str(args.wait_for),
            "wait_satisfied": args.wait_for.exists(),
            "active": active,
            "finished": finished,
            "gpu_snapshot": gpu_snapshot(),
        }
        target = args.status_dir / f"stage_status_{stage_name}.json"
        temporary = target.with_suffix(".partial.json")
        temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(target)

    while not args.wait_for.exists() and time.time() < args.hard_stop_epoch:
        write_status("waiting_for_queue_a")
        time.sleep(args.poll_interval)
    if not args.wait_for.exists():
        write_status("hard_stop_before_stage")
        return

    while time.time() < args.hard_stop_epoch:
        # A completed stage summary is an atomic safe boundary.  Terminate
        # only our own child if CUDA helper threads keep the interpreter alive.
        for pid, item in list(active.items()):
            summary = args.output_root / item["label"] / f"summary_shard{item['group']:02d}.json"
            if summary.is_file():
                try:
                    value = json.loads(summary.read_text(encoding="utf-8"))
                    counts = value.get("counts", {})
                    complete = int(counts.get("failed_states", 0)) == 0
                except Exception:
                    complete = False
                if complete:
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        for pid, item in list(active.items()):
            result = os.waitpid(pid, os.WNOHANG)
            if result != (0, 0):
                summary = args.output_root / item["label"] / f"summary_shard{item['group']:02d}.json"
                complete = False
                try:
                    value = json.loads(summary.read_text(encoding="utf-8"))
                    complete = int(value.get("counts", {}).get("failed_states", 0)) == 0
                except Exception:
                    complete = False
                record = {**item, "returncode": result[1], "complete": complete, "finished_ns": time.time_ns()}
                if complete:
                    finished.append(record)
                else:
                    failed_attempts[item["group"]] = failed_attempts.get(item["group"], 0) + 1
                    if failed_attempts[item["group"]] >= 2:
                        record["terminal_failed"] = True
                        finished.append(record)
                del active[pid]
        free = [row["gpu"] for row in gpu_snapshot() if row["free_for_this_run"]]
        reserved = {int(item["gpu"]) for item in active.values()}
        free = [gpu for gpu in free if gpu not in reserved]
        for group in range(args.groups):
            if len(active) >= len(free) + len(active):
                break
            label = f"group{group:02d}"
            if any(item["group"] == group for item in active.values()) or any(item["group"] == group and not item.get("terminal_failed") for item in finished):
                continue
            summary = args.output_root / label / f"summary_shard{group:02d}.json"
            if summary.exists():
                try:
                    value = json.loads(summary.read_text(encoding="utf-8"))
                    if int(value.get("counts", {}).get("failed_states", 0)) == 0:
                        finished.append({"group": group, "preexisting_summary": True, "finished_ns": time.time_ns()})
                        continue
                except Exception:
                    pass
            if not free:
                break
            gpu = free.pop(0)
            log = args.output_root / ".." / "logs" / f"ablation_group{group:02d}_gpu{gpu}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                args.python_bin,
                str(args.script),
                "--manifest", str(args.manifest),
                "--collection-root", str(args.collection_root),
                "--output-dir", str(args.output_root / label),
                "--shard-index", str(group),
                "--num-shards", str(args.groups),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["EVAL_PHYSICAL_GPU"] = str(gpu)
            environment.setdefault("MUJOCO_GL", "osmesa")
            environment.setdefault("PYOPENGL_PLATFORM", "osmesa")
            osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
            environment["LD_LIBRARY_PATH"] = osmesa + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
            with log.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(command, cwd=str(args.repo), env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[process.pid] = {"pid": process.pid, "gpu": gpu, "group": group, "label": label, "started_ns": time.time_ns(), "log": str(log)}
        write_status("poll")
        if len(finished) >= args.groups and not active:
            (args.status_dir / f"{stage_name}_complete.json").write_text(json.dumps({"finished_ns": time.time_ns(), "groups": args.groups}, indent=2) + "\n", encoding="utf-8")
            write_status("ablation_complete")
            return
        time.sleep(args.poll_interval)

    for pid in list(active):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    write_status("hard_stop")


if __name__ == "__main__":
    main()

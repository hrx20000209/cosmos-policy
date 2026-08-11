"""Non-destructive GPU-aware supervisor for the deep-validation queues.

The supervisor launches only on GPUs with no compute process and <=1 GB
allocated memory.  It never kills or signals a process it did not launch.  A
fixed eight-way episode shard lets newly-free GPUs pick up pending work without
duplicating a running shard.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def gpu_snapshot() -> list[dict[str, Any]]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    pmon = subprocess.check_output(["nvidia-smi", "pmon", "-c", "1", "-s", "um"], text=True)
    compute: dict[int, list[dict[str, Any]]] = {}
    for line in pmon.splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        gpu, pid, kind = int(fields[0]), int(fields[1]), fields[2]
        if kind == "C":
            compute.setdefault(gpu, []).append({"pid": pid, "kind": kind, "command": fields[-1]})
    rows = []
    for line in query.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) < 4:
            continue
        gpu, used, total, util = map(int, values[:4])
        rows.append(
            {
                "gpu": gpu,
                "memory_used_mb": used,
                "memory_total_mb": total,
                "utilization_gpu": util,
                "compute_processes": compute.get(gpu, []),
                "free_for_this_run": used <= 1024 and not compute.get(gpu),
            }
        )
    return rows


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.status_dir = args.status_dir
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.active: dict[int, dict[str, Any]] = {}
        self.profile_slots = {0: 3, 1: 4, 2: 6}
        self.profile_gpu: int | None = None
        # Make the supervisor resumable across an agent/session restart.  The
        # profiling stage is immutable once its atomic marker exists; without
        # this check a restarted supervisor needlessly relaunches the density
        # workers and can race with the safe-boundary watchdog.
        self.profile_done = (self.status_dir / "profiling_complete.json").is_file()
        self.main_started = (self.status_dir / "queue_a_complete.json").is_file()
        self.last_status = 0.0
        self.rows = read_rows(args.manifest)
        self.expected_by_group = {
            group: sum(int(row["episode_key"], 16) % args.main_groups == group for row in self.rows)
            for group in range(args.main_groups)
        }

    def write_status(self, reason: str) -> None:
        snapshot = {
            "timestamp_ns": time.time_ns(),
            "reason": reason,
            "hard_stop_epoch": self.args.hard_stop_epoch,
            "hard_stop_reached": time.time() >= self.args.hard_stop_epoch,
            "profile_done": self.profile_done,
            "main_started": self.main_started,
            "active": self.active,
            "expected_by_group": self.expected_by_group,
            "gpu_snapshot": gpu_snapshot(),
        }
        target = self.status_dir / "supervisor_status.json"
        temporary = target.with_suffix(".partial.json")
        temporary.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(target)
        checkpoint = self.status_dir / f"status_{int(time.time())}.json"
        checkpoint.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def free_gpus(self) -> list[int]:
        reserved = {int(item["gpu"]) for item in self.active.values()}
        return [row["gpu"] for row in gpu_snapshot() if row["free_for_this_run"] and row["gpu"] not in reserved]

    def launch(self, gpu: int, kind: str, slot: int) -> None:
        if kind == "profile":
            shard = self.profile_slots[slot]
            output = self.args.raw_root / "profiling" / f"worker{slot}"
            command = [
                self.args.python_bin,
                str(self.args.collector),
                "--manifest",
                str(self.args.manifest),
                "--output-dir",
                str(output),
                "--shard-index",
                str(shard),
                "--num-shards",
                "160",
                "--max-requests",
                "20",
            ]
            label = f"profile{slot}"
        else:
            group = slot
            output = self.args.raw_root / "queue_a" / "f1_collection" / f"group{group:02d}"
            command = [
                self.args.python_bin,
                str(self.args.collector),
                "--manifest",
                str(self.args.manifest),
                "--output-dir",
                str(output),
                "--shard-index",
                str(group),
                "--num-shards",
                str(self.args.main_groups),
            ]
            label = f"group{group:02d}"
        log_path = self.args.raw_root / "logs" / f"{label}_gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["EVAL_PHYSICAL_GPU"] = str(gpu)
        environment.setdefault("MUJOCO_GL", "osmesa")
        environment.setdefault("PYOPENGL_PLATFORM", "osmesa")
        osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
        environment["LD_LIBRARY_PATH"] = osmesa + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        process = subprocess.Popen(
            command,
            cwd=str(self.args.repo),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        self.active[process.pid] = {"pid": process.pid, "gpu": gpu, "kind": kind, "slot": slot, "label": label, "log": str(log_path), "started_ns": time.time_ns()}

    def reap(self) -> list[dict[str, Any]]:
        finished = []
        for pid, item in list(self.active.items()):
            code = os.waitpid(pid, os.WNOHANG)
            if code == (0, 0):
                continue
            finished.append({**item, "returncode": code[1], "finished_ns": time.time_ns()})
            del self.active[pid]
        return finished

    def profile_complete(self) -> bool:
        if self.active and any(item["kind"] == "profile" for item in self.active.values()):
            return False
        return all(
            any(item.get("label") == f"profile{slot}" for item in self._finished)
            for slot in self.profile_slots
        )

    def group_complete(self, group: int) -> bool:
        output = self.args.raw_root / "queue_a" / "f1_collection" / f"group{group:02d}"
        summary = output / f"summary_shard{group:02d}.json"
        if not summary.is_file():
            return False
        try:
            value = json.loads(summary.read_text(encoding="utf-8"))
            counts = value.get("counts", {})
            return int(counts.get("recorded", 0)) + int(counts.get("skipped", 0)) >= self.expected_by_group[group] and int(counts.get("worker_errors", 0)) == 0
        except Exception:
            return False

    def run(self) -> None:
        self._finished: list[dict[str, Any]] = []
        while True:
            now = time.time()
            # Some CUDA/TransformerEngine helper threads can remain alive
            # after the final summary has been atomically written. That is a
            # safe boundary for terminating an owned worker and releasing its
            # GPU for the next fixed shard.
            for pid, item in list(self.active.items()):
                if item["kind"] == "main" and self.group_complete(int(item["slot"])):
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            finished = self.reap()
            self._finished.extend(finished)
            if finished:
                (self.status_dir / "completed_jobs.jsonl").open("a", encoding="utf-8").write(
                    "".join(json.dumps(item) + "\n" for item in finished)
                )
            if now >= self.args.hard_stop_epoch:
                self.write_status("hard_stop")
                for pid, item in list(self.active.items()):
                    try:
                        os.killpg(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                break

            free = self.free_gpus()
            if not self.profile_done:
                for slot in self.profile_slots:
                    if any(item["kind"] == "profile" and item["slot"] == slot for item in self.active.values()):
                        continue
                    if any(item.get("label") == f"profile{slot}" for item in self._finished):
                        continue
                    if slot == 0:
                        if not free:
                            break
                        self.profile_gpu = free.pop(0)
                    elif self.profile_gpu is None:
                        break
                    # Slots 1 and 2 intentionally share the first profiling
                    # GPU. This is the short 1/2/3-workers-per-GPU density
                    # test; an OOM is recorded as a capacity result.
                    self.launch(self.profile_gpu, "profile", slot)
                if self.profile_complete():
                    self.profile_done = True
                    (self.status_dir / "profiling_complete.json").write_text(json.dumps({"finished_ns": time.time_ns(), "jobs": self._finished}, indent=2) + "\n", encoding="utf-8")
            elif not self.main_started:
                self.main_started = True

            if self.profile_done:
                running_groups = {int(item["slot"]) for item in self.active.values() if item["kind"] == "main"}
                for group in range(self.args.main_groups):
                    if group in running_groups or self.group_complete(group):
                        continue
                    if not free:
                        break
                    self.launch(free.pop(0), "main", group)

            if self.last_status == 0.0 or now - self.last_status >= self.args.status_interval:
                self.write_status("poll")
                self.last_status = now
            if self.profile_done and all(self.group_complete(group) for group in range(self.args.main_groups)) and not self.active:
                self.write_status("all_main_groups_complete")
                (self.status_dir / "queue_a_complete.json").write_text(
                    json.dumps({"finished_ns": time.time_ns(), "groups": self.args.main_groups}, indent=2) + "\n",
                    encoding="utf-8",
                )
                break
            time.sleep(self.args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/home/rxhuang/Projects/cosmos-policy"))
    parser.add_argument("--collector", type=Path, default=Path("/home/rxhuang/Projects/cosmos-policy/experiments/server_deep_validation/run_server_f1_collection.py"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("/data/rxhuang/wam_server_deep_validation"))
    parser.add_argument("--status-dir", type=Path, default=Path("/home/rxhuang/Projects/cosmos-policy/reports/server_deep_validation/checkpoints"))
    parser.add_argument("--python-bin", default="/home/rxhuang/Projects/cosmos-policy/.venv/bin/python")
    parser.add_argument("--main-groups", type=int, default=8)
    parser.add_argument("--hard-stop-epoch", type=float, required=True)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--status-interval", type=float, default=1200.0)
    args = parser.parse_args()
    Supervisor(args).run()


if __name__ == "__main__":
    main()

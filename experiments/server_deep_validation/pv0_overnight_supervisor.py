#!/usr/bin/env python3
"""Persistent, GPU-memory-aware supervisor for the frozen PV0 overnight run.

The supervisor is intentionally non-destructive: it never signals, pauses,
renices, resets, or otherwise changes a process that it did not start.  GPU
placement is based on live free VRAM plus a calibrated safety reserve, not on
utilization or a fixed workers-per-GPU rule.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from pv0_overnight_common import (
    ORIGINAL_CHECKPOINT_SHA256,
    atomic_write_json,
    build_condition_compile_anchors,
    build_state_index,
    read_jsonl,
    record_path_for_anchor,
    record_path_for_state,
    valid_anchor_record,
    valid_state_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "reports/pv0_overnight"
DEFAULT_MANIFEST = REPO_ROOT / "reports/server_deep_validation/manifests/full_scale_40_task.jsonl"
DEFAULT_COLLECTION = Path("/data/rxhuang/wam_full_scale_server/queue_a/f1_collection")
DEFAULT_ABLATION = Path("/data/rxhuang/wam_full_scale_server/queue_b/ablation")
STATE_SHARD_SIZE = 48
ANCHOR_SHARD_SIZE = 5
INITIAL_REQUIRED_VRAM_MIB = 8200
MIN_RESERVE_MIB = 3000
CALIBRATION_HEADROOM_MIB = 1024
OOM_GPU_COOLDOWN_S = 5.0 * 60.0
OOM_GPU_EXTRA_RESERVE_MIB = 2048
ANALYSIS_MIN_INTERVAL_S = 120.0
HEARTBEAT_INTERVAL_S = 30.0
GPU_SNAPSHOT_INTERVAL_S = 8.0
HEALTH_AUDIT_INTERVAL_S = 30.0 * 60.0
HARD_END_GRACE_S = 5.0 * 60.0


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"ERROR:{type(error).__name__}:{error}"


def current_git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as error:
            return f"unavailable:{type(error).__name__}:{error}"

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status": git("status", "--short"),
    }


def gpu_snapshot(our_pids: set[int]) -> list[dict[str, Any]]:
    """Record all GPUs and separate supervisor-owned from external compute."""

    rows_raw = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    process_raw = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    by_uuid: dict[str, list[dict[str, Any]]] = {}
    if not process_raw.startswith("ERROR:"):
        for line in process_raw.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4 or not fields[1].isdigit():
                continue
            pid = int(fields[1])
            owner = "unknown"
            try:
                owner = subprocess.check_output(["ps", "-o", "user=", "-p", str(pid)], text=True).strip() or "unknown"
            except Exception:
                pass
            by_uuid.setdefault(fields[0], []).append(
                {
                    "pid": pid,
                    "process_name": fields[2],
                    "used_memory_mib": int(fields[3]) if fields[3].isdigit() else None,
                    "owner": owner,
                    "ownership": "ours" if pid in our_pids else "external",
                }
            )
    output: list[dict[str, Any]] = []
    if rows_raw.startswith("ERROR:"):
        return [{"status": "ERROR", "error": rows_raw}]
    for line in rows_raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8 or not fields[0].isdigit():
            continue
        index, uuid, name, total, used, free, util, temp = fields
        processes = by_uuid.get(uuid, [])
        output.append(
            {
                "gpu": int(index),
                "uuid": uuid,
                "name": name,
                "total_mib": int(total),
                "used_mib": int(used),
                "free_mib": int(free),
                "utilization_percent": int(util),
                "temperature_c": int(temp),
                "all_compute_processes": processes,
                "our_processes": [process for process in processes if process["ownership"] == "ours"],
                "external_processes": [process for process in processes if process["ownership"] == "external"],
            }
        )
    return output


class Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = args.run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.failure_dir = self.run_dir / "failures"
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir = self.run_dir / "manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.state_index_path = self.manifest_dir / "foundation_v2_state_index.jsonl"
        self.anchor_path = self.manifest_dir / "s4_condition_compile_anchors.jsonl"
        self.state_path = self.run_dir / "run_state.json"
        self.pid_path = self.run_dir / "supervisor.pid"
        # Do not resolve this symlink: virtual environments rely on the
        # executable's original ``.venv/bin/python`` location to locate their
        # site-packages (including torch).
        self.python = sys.executable
        self.handles: dict[str, subprocess.Popen[str]] = {}
        self.stop_requested = False
        self.hard_end_seen_monotonic: float | None = None
        self.last_heartbeat = 0.0
        self.last_gpu_snapshot = 0.0
        self.last_analysis_launch = 0.0
        self.last_analysis_state_count = -1
        self.last_artifact_audit = 0.0
        self.last_success_monotonic = time.monotonic()
        self.last_progress_monotonic = time.monotonic()
        self.last_gpu_data: list[dict[str, Any]] = []
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        existing = read_json(self.state_path)
        if existing and existing.get("schema_version") == 1 and existing.get("run_kind") == "pv0_overnight":
            self.state = existing
            self.state.setdefault("jobs", {})
            self.state.setdefault("gpu_seconds_used", 0.0)
            self.state.setdefault("memory_calibration_mib", {})
            self.state.setdefault("measured_peak_mib", {})
            self.state.setdefault("gpu_oom_cooldown_until_epoch", {})
            self.state.setdefault("gpu_oom_extra_reserve_mib", {})
            self.state.setdefault("events", [])
            self._recover_running_jobs()
            return
        started_wall = time.time()
        started_mono = time.monotonic()
        self.state = {
            "schema_version": 1,
            "run_kind": "pv0_overnight",
            "run_start_time": utc_now(),
            "run_start_epoch": started_wall,
            "run_start_monotonic": started_mono,
            "soft_end_epoch": started_wall + float(self.args.duration_hours) * 3600.0 * 0.95,
            "hard_end_epoch": started_wall + float(self.args.duration_hours) * 3600.0,
            "duration_hours": float(self.args.duration_hours),
            "repository": current_git_state(),
            "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "jobs": {},
            "gpu_seconds_used": 0.0,
            "memory_calibration_mib": {
                "S1": INITIAL_REQUIRED_VRAM_MIB,
                "S4": INITIAL_REQUIRED_VRAM_MIB,
                "S5": INITIAL_REQUIRED_VRAM_MIB,
                "CLOSED_LOOP": INITIAL_REQUIRED_VRAM_MIB,
            },
            "measured_peak_mib": {},
            "gpu_oom_cooldown_until_epoch": {},
            "gpu_oom_extra_reserve_mib": {},
            "events": [],
            "phase_a_promoted": False,
            "phase_a_terminal": None,
            "finalized": False,
        }
        self._event("initialized", "new autonomous run initialized")
        self._ensure_manifests()
        self._migrate_memory_policy_if_needed()
        self._create_initial_jobs()
        self._save_state()

    def _migrate_memory_policy_if_needed(self) -> None:
        """Replace legacy global OOM inflation with measured per-GPU guards.

        An OOM while an unrelated process suddenly consumes the remaining
        memory says little about the Cosmos worker's intrinsic residency.  The
        old policy promoted that transient race to a global family requirement,
        which unnecessarily excluded otherwise-safe shared GPUs.  Keep the
        measured model requirement global and attach the additional guard to
        the GPU that actually raced.
        """

        if int(self.state.get("memory_policy_version", 0)) >= 2:
            return
        recovered: dict[str, int] = {}
        for family, root, entries, path_for in (
            ("S1", self.run_dir / "s1_fidelity/states", self.state_index, record_path_for_state),
            ("S4", self.run_dir / "s4_condition_compile/anchors", self.anchors, record_path_for_anchor),
        ):
            peaks: list[float] = []
            for entry in entries:
                payload = read_json(path_for(root, entry)) or {}
                for route in (payload.get("route_metrics") or {}).values():
                    peak = route.get("peak_reserved_mib") if isinstance(route, Mapping) else None
                    if isinstance(peak, (int, float)):
                        peaks.append(float(peak))
            if not peaks:
                continue
            observed = int(math.ceil(max(peaks)))
            required = max(INITIAL_REQUIRED_VRAM_MIB, observed + CALIBRATION_HEADROOM_MIB)
            self.state["measured_peak_mib"][family] = observed
            self.state["memory_calibration_mib"][family] = required
            recovered[family] = required
            for job in self.state["jobs"].values():
                if job.get("family") == family and job.get("status") == "pending":
                    job["required_vram_mib"] = required
        self.state["memory_policy_version"] = 2
        if recovered:
            detail = ", ".join(f"{family}={required}MiB" for family, required in sorted(recovered.items()))
            self._event("memory_policy_migration", f"restored measured worker requirements: {detail}")

    def _recover_running_jobs(self) -> None:
        for job in self.state["jobs"].values():
            if job.get("status") == "running" and not pid_alive(int(job.get("pid", -1))):
                job["status"] = "pending"
                job["recovered_from_dead_supervisor"] = True
                job.pop("pid", None)
                job.pop("gpu", None)
                self._event("requeue_recovered", f"requeued {job['id']} after supervisor restart")
        self._ensure_manifests()
        self._migrate_memory_policy_if_needed()
        # Job commands are persisted so an overnight run can survive a
        # supervisor restart.  The first supervisor version serialized the
        # resolved system Python path before the virtualenv-path correction.
        # Migrate only our script invocations, leaving any future non-Python
        # commands untouched.
        for job in self.state["jobs"].values():
            command = job.get("command")
            if (
                isinstance(command, list)
                and len(command) >= 2
                and isinstance(command[1], str)
                and command[1].startswith("experiments/server_deep_validation/")
                and command[0] != self.python
            ):
                old_python = command[0]
                command[0] = self.python
                self._event(
                    "repair_worker_interpreter",
                    f"updated {job['id']} interpreter from {old_python} to {self.python}",
                )
        # A supervisor version prior to this guard resolved the venv Python
        # symlink, launching workers through the system interpreter.  Preserve
        # its failure artifacts but automatically requeue only that known
        # infrastructure bootstrap error after the corrected supervisor starts.
        for job in self.state["jobs"].values():
            if job.get("status") != "terminal_failure":
                continue
            log_tail = self._read_log_tail(job)
            if "ModuleNotFoundError: No module named 'torch'" in log_tail:
                job.update(
                    {
                        "status": "pending",
                        "attempts": 0,
                        "last_failure": "REQUEUED_AFTER_VENV_PATH_FIX",
                        "requeued_at": utc_now(),
                    }
                )
                self._event("requeue_venv_fix", f"requeued {job['id']} after virtualenv-path repair")
            elif (
                not job.get("renderer_env_repaired")
                and (
                    "mujoco.osmesa" in log_tail
                    or "glGetError" in log_tail
                )
            ):
                job.update(
                    {
                        "status": "pending",
                        "attempts": 0,
                        "last_failure": "REQUEUED_AFTER_EGL_RENDERER_FIX",
                        "renderer_env_repaired": True,
                        "requeued_at": utc_now(),
                    }
                )
                self._event("requeue_renderer_fix", f"requeued {job['id']} after EGL renderer repair")
        self._save_state()

    def _ensure_manifests(self) -> None:
        if not self.state_index_path.is_file():
            audit = build_state_index(
                manifest_path=self.args.manifest,
                collection_root=self.args.collection_root,
                output_path=self.state_index_path,
            )
            self._event("state_index", f"built state index: {audit['state_count']} states")
        if not self.anchor_path.is_file():
            audit = build_condition_compile_anchors(
                state_index_path=self.state_index_path,
                output_path=self.anchor_path,
            )
            self._event("s4_anchors", f"built condition-compile anchors: {audit['anchor_count']} tasks")
        self.state_index = read_jsonl(self.state_index_path)
        self.anchors = read_jsonl(self.anchor_path)
        self.manifest_rows = read_jsonl(self.args.manifest)

    def _add_job(
        self,
        *,
        job_id: str,
        family: str,
        priority: int,
        command: list[str],
        completion: dict[str, Any],
        depends_on: list[str] | None = None,
        required_vram_mib: int = INITIAL_REQUIRED_VRAM_MIB,
        max_attempts: int = 3,
        cpu_only: bool = False,
    ) -> None:
        if job_id in self.state["jobs"]:
            return
        self.state["jobs"][job_id] = {
            "id": job_id,
            "family": family,
            "priority": int(priority),
            "command": command,
            "completion": completion,
            "depends_on": depends_on or [],
            "required_vram_mib": int(required_vram_mib),
            "max_attempts": int(max_attempts),
            "cpu_only": bool(cpu_only),
            "attempts": 0,
            "status": "pending",
            "created_at": utc_now(),
        }

    def _s1_command(self, start: int, end: int) -> list[str]:
        return [
            self.python,
            "experiments/server_deep_validation/run_pv0_fidelity_shard.py",
            "--state-index",
            str(self.state_index_path),
            "--output-dir",
            str(self.run_dir / "s1_fidelity/states"),
            "--start-index",
            str(start),
            "--end-index",
            str(end),
            "--memory-fraction",
            "0.40",
        ]

    def _s4_command(self, start: int, end: int) -> list[str]:
        return [
            self.python,
            "experiments/server_deep_validation/run_pv0_condition_compile_shard.py",
            "--anchors",
            str(self.anchor_path),
            "--output-dir",
            str(self.run_dir / "s4_condition_compile/anchors"),
            "--start-index",
            str(start),
            "--end-index",
            str(end),
            "--memory-fraction",
            "0.40",
        ]

    def _analysis_command(self, *, static_only: bool = False) -> list[str]:
        command = [
            self.python,
            "experiments/server_deep_validation/analyze_pv0_phase_a.py",
            "--state-index",
            str(self.state_index_path),
            "--s1-dir",
            str(self.run_dir / "s1_fidelity/states"),
            "--s4-dir",
            str(self.run_dir / "s4_condition_compile/anchors"),
            "--ablation-root",
            str(self.args.ablation_root),
            "--output-dir",
            str(self.run_dir),
        ]
        if static_only:
            command.append("--static-only")
        return command

    def _create_initial_jobs(self) -> None:
        self._add_job(
            job_id="s2_static_decomposition",
            family="S2",
            priority=0,
            command=self._analysis_command(static_only=True),
            completion={"kind": "analysis_static"},
            cpu_only=True,
        )
        smoke_end = min(3, len(self.state_index))
        self._add_job(
            job_id="s1_smoke_00000",
            family="S1",
            priority=0,
            command=self._s1_command(0, smoke_end),
            completion={"kind": "s1", "start": 0, "end": smoke_end},
        )
        self._add_job(
            job_id="s4_smoke_000",
            family="S4",
            priority=0,
            command=self._s4_command(0, 1),
            completion={"kind": "s4", "start": 0, "end": 1},
            depends_on=["s1_smoke_00000"],
        )
        for start in range(0, len(self.state_index), STATE_SHARD_SIZE):
            end = min(start + STATE_SHARD_SIZE, len(self.state_index))
            self._add_job(
                job_id=f"s1_{start:05d}_{end:05d}",
                family="S1",
                priority=1,
                command=self._s1_command(start, end),
                completion={"kind": "s1", "start": start, "end": end},
                depends_on=["s1_smoke_00000"],
            )
        for start in range(0, len(self.anchors), ANCHOR_SHARD_SIZE):
            end = min(start + ANCHOR_SHARD_SIZE, len(self.anchors))
            self._add_job(
                job_id=f"s4_{start:03d}_{end:03d}",
                family="S4",
                priority=4,
                command=self._s4_command(start, end),
                completion={"kind": "s4", "start": start, "end": end},
                depends_on=["s4_smoke_000"],
            )

    def _event(self, kind: str, message: str) -> None:
        event = {"timestamp": utc_now(), "kind": kind, "message": message}
        self.state.setdefault("events", []).append(event)
        self.state["events"] = self.state["events"][-200:]

    def _save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        self.state["elapsed_hours"] = (time.monotonic() - float(self.state["run_start_monotonic"])) / 3600.0
        self.state["remaining_hours"] = max(0.0, (float(self.state["hard_end_epoch"]) - time.time()) / 3600.0)
        atomic_write_json(self.state_path, self.state)

    def _our_pids(self) -> set[int]:
        return {
            int(job["pid"])
            for job in self.state["jobs"].values()
            if job.get("status") == "running" and isinstance(job.get("pid"), int)
        }

    def _update_gpu_snapshot(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_gpu_snapshot < GPU_SNAPSHOT_INTERVAL_S:
            return
        self.last_gpu_data = gpu_snapshot(self._our_pids())
        atomic_write_json(
            self.run_dir / "gpu_pool_live.json",
            {
                "timestamp": utc_now(),
                "elapsed_hours": (now - float(self.state["run_start_monotonic"])) / 3600.0,
                "gpus": self.last_gpu_data,
                "policy": "eligible iff free_vram > calibrated_required_vram + safety_reserve; external processes are never modified",
            },
        )
        self.last_gpu_snapshot = now

    def _safety_reserve(self, required_mib: int) -> int:
        # Conservatively retain 30% of measured model residency, never under 3 GiB.
        return max(MIN_RESERVE_MIB, int(math.ceil(required_mib * 0.30)))

    def _family_required_vram(self, job: Mapping[str, Any]) -> int:
        family = str(job["family"])
        return max(
            int(job.get("required_vram_mib", INITIAL_REQUIRED_VRAM_MIB)),
            int(self.state.get("memory_calibration_mib", {}).get(family, INITIAL_REQUIRED_VRAM_MIB)),
        )

    def _active_gpu_ids(self) -> set[int]:
        return {
            int(job["gpu"])
            for job in self.state["jobs"].values()
            if job.get("status") == "running" and job.get("gpu") is not None
        }

    def _eligible_gpus(self, job: Mapping[str, Any]) -> list[dict[str, Any]]:
        if job.get("cpu_only"):
            return []
        required = self._family_required_vram(job)
        reserve = self._safety_reserve(required)
        active = self._active_gpu_ids()
        now_epoch = time.time()
        cooldowns = self.state.get("gpu_oom_cooldown_until_epoch", {})
        gpu_extra_reserve = self.state.get("gpu_oom_extra_reserve_mib", {})
        candidates = []
        for row in self.last_gpu_data:
            if not isinstance(row.get("gpu"), int) or row["gpu"] in active:
                continue
            gpu_key = str(row["gpu"])
            if float(cooldowns.get(gpu_key, 0.0)) > now_epoch:
                continue
            if job.get("requires_clean_gpu") and (
                int(row.get("used_mib", 10**9)) > 1024 or row.get("all_compute_processes")
            ):
                continue
            extra = int(gpu_extra_reserve.get(gpu_key, 0))
            if int(row.get("free_mib", 0)) > required + reserve + extra:
                candidates.append(row)
        return sorted(candidates, key=lambda row: int(row["free_mib"]), reverse=True)

    def _dependencies_done(self, job: Mapping[str, Any]) -> bool:
        return all(self.state["jobs"].get(job_id, {}).get("status") == "completed" for job_id in job.get("depends_on", []))

    def _ready_jobs(self, *, cpu_only: bool) -> list[dict[str, Any]]:
        return sorted(
            [
                job
                for job in self.state["jobs"].values()
                if job.get("status") == "pending"
                and bool(job.get("cpu_only")) == cpu_only
                and self._dependencies_done(job)
            ],
            key=lambda job: (int(job["priority"]), str(job["id"])),
        )

    def _s1_done_count(self) -> int:
        root = self.run_dir / "s1_fidelity/states"
        return sum(valid_state_record(record_path_for_state(root, entry), entry) for entry in self.state_index)

    def _s4_done_count(self) -> int:
        root = self.run_dir / "s4_condition_compile/anchors"
        return sum(valid_anchor_record(record_path_for_anchor(root, anchor), anchor) for anchor in self.anchors)

    def _s4_should_get_a_slot(self) -> bool:
        if self._s1_done_count() < 4:
            return False
        if any(job.get("status") == "running" and job.get("family") == "S4" for job in self.state["jobs"].values()):
            return False
        return bool(self._ready_jobs(cpu_only=False))

    def _select_gpu_job(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        ready = self._ready_jobs(cpu_only=False)
        if not ready:
            return None
        # Preserve S1 priority, but once its smoke and a few shards have passed,
        # reserve an occasional slot for fixed S4 rather than serializing it until
        # thousands of states finish.
        if self._s4_should_get_a_slot():
            s4 = [job for job in ready if job["family"] == "S4"]
            if s4:
                candidates = self._eligible_gpus(s4[0])
                if candidates:
                    return s4[0], candidates[0]
        for job in ready:
            candidates = self._eligible_gpus(job)
            if candidates:
                return job, candidates[0]
        return None

    def _launch(self, job: dict[str, Any], gpu: int | None) -> None:
        attempt = int(job.get("attempts", 0)) + 1
        log_path = self.logs_dir / f"{job['id']}.attempt{attempt}.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if gpu is None:
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["EVAL_PHYSICAL_GPU"] = str(gpu)
            # OSMesa is unavailable to this account on the shared host.  Use
            # NVIDIA's headless EGL implementation and pin its renderer to
            # the same physical device as the Cosmos worker.  This is an
            # execution-environment setting, not a policy input.
            env["MUJOCO_GL"] = "egl"
            env["PYOPENGL_PLATFORM"] = "egl"
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
            env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                job["command"],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        self.handles[job["id"]] = process
        job.update(
            {
                "status": "running",
                "attempts": attempt,
                "pid": int(process.pid),
                "gpu": gpu,
                "log": str(log_path),
                "started_at": utc_now(),
                "started_monotonic": time.monotonic(),
            }
        )
        self.last_progress_monotonic = time.monotonic()
        self._event("launch", f"{job['id']} attempt={attempt} gpu={gpu}")

    def _read_log_tail(self, job: Mapping[str, Any]) -> str:
        path = Path(str(job.get("log", "")))
        try:
            data = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return data[-12000:]

    def _classify_failure(self, job: Mapping[str, Any], log_tail: str, returncode: int | None) -> str:
        text = log_tail.lower()
        if "out of memory" in text or "cuda error: out of memory" in text:
            return "OOM_RETRY"
        if any(token in text for token in ("osmesa", "egl", "renderer", "resource temporarily unavailable", "filelock")):
            return "TRANSIENT"
        if any(token in text for token in ("filenotfound", "bddl", "manifest", "state-index provenance")):
            return "DATA"
        if any(token in text for token in ("checkpoint", "model", "shape mismatch")):
            return "MODEL"
        if returncode == 0:
            return "INCOMPLETE_ARTIFACT"
        return "UNKNOWN"

    def _artifact_complete(self, job: Mapping[str, Any]) -> tuple[bool, str]:
        completion = job["completion"]
        kind = completion["kind"]
        if kind == "s1":
            root = self.run_dir / "s1_fidelity/states"
            entries = self.state_index[int(completion["start"]) : int(completion["end"])]
            return all(valid_state_record(record_path_for_state(root, entry), entry) for entry in entries), "S1 state records"
        if kind == "s4":
            root = self.run_dir / "s4_condition_compile/anchors"
            anchors = self.anchors[int(completion["start"]) : int(completion["end"])]
            return all(valid_anchor_record(record_path_for_anchor(root, anchor), anchor) for anchor in anchors), "S4 anchor records"
        if kind == "analysis_static":
            payload = read_json(self.run_dir / "S2_PREDICTIVE_PRIOR_DECOMPOSITION.json")
            return payload is not None and payload.get("status") in {"GO", "NO_GO"}, "S2 analysis"
        if kind == "analysis_refresh":
            return (self.run_dir / "PHASE_A_DECISION.json").is_file(), "Phase-A analysis"
        if kind == "closed_loop":
            payload = read_json(Path(completion["output"]))
            if payload is None or payload.get("status") != "PASS":
                return False, "closed-loop output"
            valid = (
                payload.get("checkpoint_sha256") == ORIGINAL_CHECKPOINT_SHA256
                and payload.get("value_used") is False
                and payload.get("privileged_runtime_state_input") is False
                and payload.get("adaptive_scheduler_used") is False
                and int(payload.get("route_contract", {}).get("denoising_steps", -1)) == 1
                and payload.get("trace_contract", {}).get("status") == "PASS"
            )
            return valid, "closed-loop contract"
        if kind == "clean_latency":
            payload = read_json(Path(completion["output"]))
            return payload is not None and payload.get("status") == "PASS_EXECUTED", "clean latency output"
        return False, f"unknown completion kind={kind}"

    def _calibrate_memory_from_artifact(self, job: Mapping[str, Any]) -> None:
        completion = job["completion"]
        kind = completion["kind"]
        peaks: list[float] = []
        if kind == "s1":
            root = self.run_dir / "s1_fidelity/states"
            for entry in self.state_index[int(completion["start"]) : int(completion["end"])]:
                payload = read_json(record_path_for_state(root, entry)) or {}
                for route in (payload.get("route_metrics") or {}).values():
                    if isinstance(route, Mapping) and isinstance(route.get("peak_reserved_mib"), (int, float)):
                        peaks.append(float(route["peak_reserved_mib"]))
        elif kind == "s4":
            root = self.run_dir / "s4_condition_compile/anchors"
            for anchor in self.anchors[int(completion["start"]) : int(completion["end"])]:
                payload = read_json(record_path_for_anchor(root, anchor)) or {}
                for route in (payload.get("route_metrics") or {}).values():
                    if isinstance(route, Mapping) and isinstance(route.get("peak_reserved_mib"), (int, float)):
                        peaks.append(float(route["peak_reserved_mib"]))
        elif kind == "closed_loop":
            payload = read_json(Path(completion["output"])) or {}
            peak = (payload.get("gpu") or {}).get("peak_reserved_mib")
            if isinstance(peak, (int, float)):
                peaks.append(float(peak))
        if peaks:
            observed = int(math.ceil(max(peaks)))
            family = str(job["family"])
            current_observed = int(self.state.get("measured_peak_mib", {}).get(family, 0))
            measured = max(current_observed, observed)
            self.state.setdefault("measured_peak_mib", {})[family] = measured
            self.state["memory_calibration_mib"][family] = max(
                INITIAL_REQUIRED_VRAM_MIB,
                measured + CALIBRATION_HEADROOM_MIB,
            )

    def _finish_job(self, job: dict[str, Any], returncode: int | None) -> None:
        elapsed = max(0.0, time.monotonic() - float(job.get("started_monotonic", time.monotonic())))
        if job.get("gpu") is not None:
            self.state["gpu_seconds_used"] = float(self.state.get("gpu_seconds_used", 0.0)) + elapsed
        complete, description = self._artifact_complete(job)
        if returncode == 0 and complete:
            job.update({"status": "completed", "finished_at": utc_now(), "returncode": returncode, "elapsed_s": elapsed})
            self.handles.pop(job["id"], None)
            self._calibrate_memory_from_artifact(job)
            atomic_append_jsonl(
                self.run_dir / "completed_jobs.jsonl",
                {"timestamp": utc_now(), "job": job, "artifact": description},
            )
            self.last_success_monotonic = time.monotonic()
            self.last_progress_monotonic = time.monotonic()
            self._event("complete", f"{job['id']} elapsed={elapsed:.1f}s")
            self.state["last_successful_artifact_time"] = utc_now()
            self.state["last_progress_time"] = utc_now()
            return
        tail = self._read_log_tail(job)
        classification = self._classify_failure(job, tail, returncode)
        failure_payload = {
            "timestamp": utc_now(),
            "job": job,
            "returncode": returncode,
            "classification": classification,
            "artifact_check": description,
            "log_tail": tail,
            "gpu_snapshot": self.last_gpu_data,
        }
        atomic_write_json(self.failure_dir / f"{job['id']}.attempt{job['attempts']}.json", failure_payload)
        attempts = int(job["attempts"])
        self.handles.pop(job["id"], None)
        if classification == "OOM_RETRY":
            # Preserve the measured worker requirement.  The failure snapshot
            # may reflect a co-tenant's abrupt allocation rather than a larger
            # model.  Increase the safety guard only for that physical GPU and
            # let work stealing choose a different card during its cooldown.
            gpu = job.get("gpu")
            if isinstance(gpu, int):
                key = str(gpu)
                cooldowns = self.state.setdefault("gpu_oom_cooldown_until_epoch", {})
                extras = self.state.setdefault("gpu_oom_extra_reserve_mib", {})
                cooldowns[key] = max(float(cooldowns.get(key, 0.0)), time.time() + OOM_GPU_COOLDOWN_S)
                extras[key] = max(int(extras.get(key, 0)), OOM_GPU_EXTRA_RESERVE_MIB)
                self._event(
                    "gpu_oom_guard",
                    f"GPU{gpu} cooldown={OOM_GPU_COOLDOWN_S:.0f}s extra_reserve={extras[key]}MiB",
                )
        retry_allowed = attempts < int(job["max_attempts"])
        if retry_allowed:
            job.update(
                {
                    "status": "pending",
                    "last_failure": classification,
                    "last_returncode": returncode,
                    "last_failure_at": utc_now(),
                }
            )
            atomic_append_jsonl(self.run_dir / "retry_jobs.jsonl", failure_payload)
            self._event("retry", f"{job['id']} classification={classification} attempt={attempts}")
        else:
            job.update(
                {
                    "status": "terminal_failure",
                    "last_failure": "TERMINAL_INFRA_FAILURE" if classification == "OOM_RETRY" else classification,
                    "last_returncode": returncode,
                    "finished_at": utc_now(),
                }
            )
            atomic_append_jsonl(self.run_dir / "failed_jobs.jsonl", failure_payload)
            self._event("terminal_failure", f"{job['id']} classification={classification}")

    def _poll_workers(self) -> None:
        for job in list(self.state["jobs"].values()):
            if job.get("status") != "running":
                continue
            process = self.handles.get(job["id"])
            if process is not None:
                returncode = process.poll()
                if returncode is not None:
                    self._finish_job(job, returncode)
            elif not pid_alive(int(job.get("pid", -1))):
                # A worker survived a supervisor restart but has now exited.  Its
                # artifact is authoritative; the historical exit status is not.
                self._finish_job(job, None)

    def _maybe_enqueue_analysis(self) -> None:
        now = time.monotonic()
        s1_done = self._s1_done_count()
        cpu_active = any(job.get("status") == "running" and job.get("cpu_only") for job in self.state["jobs"].values())
        if cpu_active:
            return
        changed = s1_done >= self.last_analysis_state_count + 3
        if not changed and now - self.last_analysis_launch < ANALYSIS_MIN_INTERVAL_S:
            return
        job_id = f"analysis_refresh_{int(time.time())}"
        self._add_job(
            job_id=job_id,
            family="ANALYSIS",
            priority=0,
            command=self._analysis_command(),
            completion={"kind": "analysis_refresh"},
            cpu_only=True,
        )
        self.last_analysis_launch = now
        self.last_analysis_state_count = s1_done

    def _phase_a_decision(self) -> dict[str, Any] | None:
        return read_json(self.run_dir / "PHASE_A_DECISION.json")

    def _find_preflight_rows(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        lookup = {(row["task_name"], int(row["init_state_index"])): row for row in self.manifest_rows}
        first = lookup.get(("put_the_bowl_on_the_stove", 0))
        heldout_name = "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
        second = [
            lookup.get(("put_the_bowl_on_the_stove", 0)),
            lookup.get(("put_the_bowl_on_the_stove", 1)),
            lookup.get((heldout_name, 1)),
            lookup.get((heldout_name, 2)),
        ]
        if first is None or any(row is None for row in second):
            # Fallback is still frozen and manifest-order based, not outcome based.
            ordered = sorted(self.manifest_rows, key=lambda row: int(row["manifest_order"]))
            first = ordered[0]
            second = ordered[:4]
        return first, [row for row in second if row is not None]

    def _closed_loop_command(
        self,
        *,
        row: Mapping[str, Any],
        mode: str,
        output: Path,
        trace_output: Path,
        prefix: int = 16,
        seed_offset: int = 0,
        interruption_start: int | None = None,
    ) -> list[str]:
        command = [
            self.python,
            "experiments/server_deep_validation/run_pv0_closed_loop_episode.py",
            "--mode",
            mode,
            "--episode-key",
            str(row["episode_key"]),
            "--output",
            str(output),
            "--trace-output",
            str(trace_output),
            "--manifest",
            str(self.args.manifest),
            "--memory-fraction",
            "0.40",
            "--execution-prefix",
            str(prefix),
            "--inference-seed-offset",
            str(seed_offset),
        ]
        if interruption_start is not None:
            command.extend(["--interruption-start-step", str(interruption_start), "--interruption-length", "4"])
        return command

    def _add_closed_loop_job(
        self,
        *,
        phase: str,
        row: Mapping[str, Any],
        mode: str,
        depends_on: list[str],
        seed_offset: int = 0,
        interruption_start: int | None = None,
        priority: int = 6,
    ) -> str:
        suffix = f"{mode}_{row['episode_key']}_seedoff{seed_offset}"
        output = self.run_dir / phase / "episodes" / f"{suffix}.json"
        trace = self.run_dir / phase / "traces" / f"{suffix}.json"
        job_id = f"{phase}_{suffix}"
        self._add_job(
            job_id=job_id,
            family="S5" if phase == "s5_action_outcome" else "CLOSED_LOOP",
            priority=priority,
            command=self._closed_loop_command(
                row=row,
                mode=mode,
                output=output,
                trace_output=trace,
                seed_offset=seed_offset,
                interruption_start=interruption_start,
            ),
            completion={"kind": "closed_loop", "output": str(output)},
            depends_on=depends_on,
        )
        return job_id

    def _promote_phase_a_go(self) -> None:
        if self.state.get("phase_a_promoted"):
            return
        first, second = self._find_preflight_rows()
        s5_jobs = [
            self._add_closed_loop_job(
                phase="s5_action_outcome",
                row=first,
                mode=mode,
                depends_on=[],
                interruption_start=16,
                priority=5,
            )
            for mode in ("fresh", "predicted_reuse", "native_persistent")
        ]
        preflight_one = [
            self._add_closed_loop_job(
                phase="preflight_1x1",
                row=first,
                mode=mode,
                depends_on=s5_jobs,
                priority=6,
            )
            for mode in ("fresh", "predicted_reuse", "native_persistent")
        ]
        preflight_two = []
        for row in second:
            for mode in ("fresh", "predicted_reuse", "native_persistent"):
                preflight_two.append(
                    self._add_closed_loop_job(
                        phase="preflight_2x2",
                        row=row,
                        mode=mode,
                        depends_on=preflight_one,
                        priority=6,
                    )
                )
        for row in self.manifest_rows:
            for mode in ("fresh", "predicted_reuse", "native_persistent"):
                self._add_closed_loop_job(
                    phase="phase_b_600",
                    row=row,
                    mode=mode,
                    depends_on=preflight_two,
                    priority=7,
                )
        self.state["phase_a_promoted"] = True
        self._event("phase_a_go", "S5, closed-loop preflights, and 600-episode queue activated")

    def _maybe_advance_phase(self) -> None:
        decision = self._phase_a_decision()
        if decision is None:
            return
        status = decision.get("status")
        if status == "GO":
            self._promote_phase_a_go()
        elif status in {"NO_GO", "INCOMPLETE_INFRA"}:
            self.state["phase_a_terminal"] = status
            if not self.state.get("phase_a_terminal_noted"):
                self._event("phase_a_terminal", f"Phase-A status={status}; closed-loop promotion disabled")
                self.state["phase_a_terminal_noted"] = True

    def _all_dependencies_completed(self, prefix: str) -> bool:
        relevant = [job for job in self.state["jobs"].values() if str(job["id"]).startswith(prefix)]
        return bool(relevant) and all(job.get("status") == "completed" for job in relevant)

    def _maybe_add_seed_confirmation(self) -> None:
        if not self._all_dependencies_completed("phase_b_600_") or self.state.get("seed_confirmation_added"):
            return
        # Only after the 600 fixed-seed episodes complete: identify exactly the
        # Fresh-success or F1/PV0-disagreement scenarios, then repeat F1/PV0
        # with three new inference seeds while physical initial state stays fixed.
        selected: list[dict[str, Any]] = []
        for row in self.manifest_rows:
            root = self.run_dir / "phase_b_600/episodes"
            fresh = read_json(root / f"fresh_{row['episode_key']}_seedoff0.json")
            pv0 = read_json(root / f"native_persistent_{row['episode_key']}_seedoff0.json")
            if fresh is None or pv0 is None:
                continue
            fresh_success = bool((fresh.get("record") or {}).get("success"))
            pv0_success = bool((pv0.get("record") or {}).get("success"))
            if fresh_success or fresh_success != pv0_success:
                selected.append(row)
        for row in selected:
            for seed_offset in (1000, 2000, 3000):
                for mode in ("fresh", "native_persistent"):
                    self._add_closed_loop_job(
                        phase="seed_confirmation",
                        row=row,
                        mode=mode,
                        depends_on=[],
                        seed_offset=seed_offset,
                        priority=8,
                    )
        self.state["seed_confirmation_added"] = True
        self.state["seed_confirmation_scenarios"] = len(selected)
        self._event("seed_confirmation", f"added 3-seed repeats for {len(selected)} selected scenarios")

    def _maybe_add_clean_latency(self) -> None:
        if not self.state.get("seed_confirmation_added") or self.state.get("clean_latency_added"):
            return
        # Hold a formal latency microbenchmark until a genuinely clean GPU is
        # observed. Correctness jobs continue on shared GPUs in the meantime.
        clean = [
            row
            for row in self.last_gpu_data
            if isinstance(row.get("gpu"), int)
            and int(row.get("used_mib", 10**9)) <= 1024
            and not row.get("all_compute_processes")
        ]
        if not clean:
            return
        entry = self.state_index[0]
        output = self.run_dir / "clean_latency/preflight_repeats.json"
        command = [
            self.python,
            "experiments/server_deep_validation/run_native_persistent_condition_preflight.py",
            "--state-key",
            str(entry["state_key"]),
            "--output",
            str(output),
            "--repeats",
            "25",
            "--warmup",
            "5",
            "--memory-fraction",
            "0.50",
        ]
        self._add_job(
            job_id="clean_latency_preflight_repeats",
            family="CLOSED_LOOP",
            priority=9,
            command=command,
            completion={"kind": "clean_latency", "output": str(output)},
            required_vram_mib=INITIAL_REQUIRED_VRAM_MIB,
        )
        self.state["jobs"]["clean_latency_preflight_repeats"]["requires_clean_gpu"] = True
        self.state["clean_latency_added"] = True
        self._event("clean_latency", f"clean latency job queued after GPU{clean[0]['gpu']} became clean")

    def _maybe_launch_cpu(self) -> None:
        if any(job.get("status") == "running" and job.get("cpu_only") for job in self.state["jobs"].values()):
            return
        ready = self._ready_jobs(cpu_only=True)
        if ready:
            self._launch(ready[0], None)

    def _closed_loop_cpu_allows_more(self) -> bool:
        active = sum(
            job.get("status") == "running" and job.get("family") in {"S5", "CLOSED_LOOP"}
            for job in self.state["jobs"].values()
        )
        if active == 0:
            return True
        cpu_count = os.cpu_count() or 1
        try:
            load1 = os.getloadavg()[0]
        except OSError:
            load1 = 0.0
        # The renderer is CPU-heavy.  Avoid adding a fourth simulator worker
        # when host load is already high; offline S1/S4 is still free to run.
        return active < 3 and load1 < cpu_count * 0.85

    def _maybe_launch_gpu(self) -> None:
        remaining_s = float(self.state["hard_end_epoch"]) - time.time()
        # All workers are deliberately bounded, but avoid a fresh launch when
        # fewer than fifteen minutes remain.  This honors the soft-end rule
        # without needlessly idling through its entire final half hour.
        if remaining_s < 15.0 * 60.0:
            return
        # An external process can change its residency between periodic polls.
        # Refresh immediately before placement, then fill every independent
        # eligible card in this scheduling pass rather than waiting one poll
        # per GPU.  ``_active_gpu_ids`` prevents double-placement.
        self._update_gpu_snapshot(force=True)
        for _ in range(max(1, len(self.last_gpu_data))):
            candidate = self._select_gpu_job()
            if candidate is None:
                return
            job, gpu = candidate
            if job["family"] in {"S5", "CLOSED_LOOP"} and not self._closed_loop_cpu_allows_more():
                return
            self._launch(job, int(gpu["gpu"]))

    def _write_artifact_audit(self) -> None:
        now = time.monotonic()
        if now - self.last_artifact_audit < HEARTBEAT_INTERVAL_S:
            return
        s1_done = self._s1_done_count()
        s4_done = self._s4_done_count()
        counts = Counter(job.get("status") for job in self.state["jobs"].values())
        payload = {
            "timestamp": utc_now(),
            "state_index_expected": len(self.state_index),
            "s1_valid_records": s1_done,
            "s4_expected_anchors": len(self.anchors),
            "s4_valid_records": s4_done,
            "job_status_counts": dict(counts),
            "phase_a_decision": self._phase_a_decision(),
            "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
        }
        atomic_write_json(self.run_dir / "artifact_audit.json", payload)
        self.last_artifact_audit = now

    def _write_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self.last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        counts = Counter(job.get("status") for job in self.state["jobs"].values())
        running = [
            {"id": job["id"], "family": job["family"], "pid": job.get("pid"), "gpu": job.get("gpu")}
            for job in self.state["jobs"].values()
            if job.get("status") == "running"
        ]
        payload = {
            "timestamp": utc_now(),
            "elapsed_hours": (now - float(self.state["run_start_monotonic"])) / 3600.0,
            "remaining_hours": max(0.0, (float(self.state["hard_end_epoch"]) - time.time()) / 3600.0),
            "current_phase": self._phase_a_decision().get("status") if self._phase_a_decision() else "PHASE_A_RUNNING",
            "pending_jobs": int(counts.get("pending", 0)),
            "running_jobs": running,
            "completed_jobs": int(counts.get("completed", 0)),
            "failed_jobs": int(counts.get("terminal_failure", 0)),
            "gpu_status": self.last_gpu_data,
            "last_successful_artifact_time": self.state.get("last_successful_artifact_time", utc_now()),
            "last_progress_time": self.state.get("last_progress_time", utc_now()),
            "gpu_hours_used": float(self.state.get("gpu_seconds_used", 0.0)) / 3600.0,
        }
        atomic_write_json(self.run_dir / "heartbeat.json", payload)
        self.last_heartbeat = now

    def _write_live_report(self) -> None:
        decision = self._phase_a_decision() or {}
        s1 = read_json(self.run_dir / "S1_PV0_FULL_SCALE_FIDELITY.json") or {}
        s2 = read_json(self.run_dir / "S2_PREDICTIVE_PRIOR_DECOMPOSITION.json") or {}
        s3 = read_json(self.run_dir / "S3_EXECUTION_PREFIX_ALIGNMENT.json") or {}
        s4 = read_json(self.run_dir / "S4_CONDITION_COMPILE_VALIDATION.json") or {}
        counts = Counter(job.get("status") for job in self.state["jobs"].values())
        gpu_lines = []
        for row in self.last_gpu_data:
            if not isinstance(row.get("gpu"), int):
                continue
            gpu_lines.append(
                f"- GPU{row['gpu']}: free {row['free_mib']} MiB / {row['total_mib']} MiB, "
                f"util {row['utilization_percent']}%, ours={len(row['our_processes'])}, external={len(row['external_processes'])}"
            )
        phase_b = [job for job in self.state["jobs"].values() if str(job["id"]).startswith("phase_b_600_")]
        phase_b_completed = sum(job.get("status") == "completed" for job in phase_b)
        text = "\n".join(
            [
                "# PV0 Overnight Live Report",
                "",
                f"更新时间：{utc_now()}  ",
                f"运行已过：{(time.monotonic() - float(self.state['run_start_monotonic'])) / 3600.0:.2f} h；"
                f"剩余：{max(0.0, (float(self.state['hard_end_epoch']) - time.time()) / 3600.0):.2f} h。",
                "",
                "## 当前科学门控",
                "",
                f"- Phase-A：`{decision.get('status', 'INCOMPLETE')}`",
                f"- S1 3801-state PV0 fidelity：`{s1.get('status', 'PENDING')}`",
                f"- S2 predictive-prior decomposition：`{s2.get('status', 'PENDING')}`",
                f"- S3 execution-prefix alignment：`{s3.get('status', 'PENDING')}`",
                f"- S4 fixed condition-compile control：`{s4.get('status', 'PENDING')}`",
                "",
                "## 进度",
                "",
                f"- S1 valid state artifacts：{self._s1_done_count()} / {len(self.state_index)}",
                f"- S4 valid anchor artifacts：{self._s4_done_count()} / {len(self.anchors)}",
                f"- Phase-B closed-loop：{phase_b_completed} / {len(phase_b)} episodes（仅在 Phase-A GO 后启动）",
                f"- Jobs：pending={counts.get('pending', 0)}, running={counts.get('running', 0)}, completed={counts.get('completed', 0)}, terminal_failed={counts.get('terminal_failure', 0)}",
                f"- 累积已记录 GPU-hours：{float(self.state.get('gpu_seconds_used', 0.0)) / 3600.0:.2f}",
                "",
                "## GPU pool",
                "",
                *(gpu_lines or ["- 正在等待可读取的 GPU snapshot。"]),
                "",
                "## 纪律与解释边界",
                "",
                "- 仅使用原始 pre-finetune Cosmos checkpoint，denoise=1；不读取 Cosmos value。",
                "- PV0 是 native fresh visual-prefix persistent-condition 路径；没有 hidden patch、fresh-prefix oracle、scheduler 或 privileged runtime state。",
                "- 共享 GPU 的闭环 timing 不会写成正式 latency claim；清洁 GPU benchmark 会单独等待可用卡。",
                "",
                "## 当前暂定结论",
                "",
                decision.get("rationale", "正在积累可审计数据，尚未达到 Phase-A 自动决策条件。"),
                "",
            ]
        )
        (self.run_dir / "OVERNIGHT_LIVE_REPORT.md").write_text(text, encoding="utf-8")
        (self.run_dir / "overnight_summary.md").write_text(text, encoding="utf-8")

    def _health_audit_if_stalled(self) -> None:
        now = time.monotonic()
        if now - self.last_success_monotonic <= HEALTH_AUDIT_INTERVAL_S:
            return
        self._write_artifact_audit()
        self._event("health_audit", "no completed artifact for >30 minutes; refreshed audit and retained pending queue")
        self.last_success_monotonic = now

    def _write_final_reports(self) -> None:
        decision = self._phase_a_decision() or {}
        s1 = read_json(self.run_dir / "S1_PV0_FULL_SCALE_FIDELITY.json") or {}
        s2 = read_json(self.run_dir / "S2_PREDICTIVE_PRIOR_DECOMPOSITION.json") or {}
        s3 = read_json(self.run_dir / "S3_EXECUTION_PREFIX_ALIGNMENT.json") or {}
        s4 = read_json(self.run_dir / "S4_CONDITION_COMPILE_VALIDATION.json") or {}
        counts = Counter(job.get("status") for job in self.state["jobs"].values())
        phase_b = [job for job in self.state["jobs"].values() if str(job["id"]).startswith("phase_b_600_")]
        phase_b_done = sum(job.get("status") == "completed" for job in phase_b)
        status = decision.get("status", "INCOMPLETE")
        result = "\n".join(
            [
                "# PV0 Overnight Autonomous Results",
                "",
                f"## CURRENT DECISION: {status}",
                "",
                f"- S1 status: `{s1.get('status', 'PENDING')}` ({self._s1_done_count()}/{len(self.state_index)} state artifacts)",
                f"- S2 status: `{s2.get('status', 'PENDING')}`",
                f"- S3 status: `{s3.get('status', 'PENDING')}`",
                f"- S4 status: `{s4.get('status', 'PENDING')}` ({self._s4_done_count()}/{len(self.anchors)} anchors)",
                f"- S5 status: `{self._all_dependencies_completed('s5_action_outcome_')}`",
                f"- 600-episode status: {phase_b_done}/{len(phase_b)} completed",
                f"- jobs completed={counts.get('completed', 0)}, failed={counts.get('terminal_failure', 0)}, retries={sum(int(job.get('attempts', 0)) > 1 for job in self.state['jobs'].values())}",
                f"- recorded GPU-hours={float(self.state.get('gpu_seconds_used', 0.0)) / 3600.0:.2f}",
                "",
                "## Scientific contract",
                "",
                "Original pre-finetune Cosmos checkpoint only; fixed denoise=1; no Cosmos value, privileged runtime state, scheduler, threshold, hidden patch, or new interface search.",
                "",
                "## Remaining work",
                "",
                f"Pending jobs: {counts.get('pending', 0)}; active jobs: {counts.get('running', 0)}; terminal infrastructure failures: {counts.get('terminal_failure', 0)}.",
                "",
                decision.get("rationale", "No final Phase-A decision artifact was available."),
                "",
            ]
        )
        (self.run_dir / "OVERNIGHT_PV0_RESULTS_ZH.md").write_text(result, encoding="utf-8")
        morning = "\n".join(
            [
                "# Morning README",
                "",
                f"1. 完成情况：S1 {self._s1_done_count()}/{len(self.state_index)}，S4 {self._s4_done_count()}/{len(self.anchors)}；Phase-B {phase_b_done}/{len(phase_b)}。",
                f"2. 最强正面结果：{decision.get('rationale', '待完整分析。')}",
                "3. 最强负面结果：见 `PV0_FAILURE_LOCALIZATION.json` 与任一 failed job artifact。",
                f"4. PV0 是否通过 3801-state test：{s1.get('status', 'PENDING')}。",
                f"5. execution-prefix alignment：{s3.get('status', 'PENDING')}。",
                "6. physical disturbance recovery：见 S5 artifacts；它不使用或训练 scheduler。",
                f"7. 600 episodes 是否启动/完成：{len(phase_b) > 0}/{phase_b_done == len(phase_b) and bool(phase_b)}。",
                f"8. 当前决策：{status}。",
                "9. 下一步：若 GO，审阅完整 paired closed-loop 与 clean-GPU cost；若 NO_GO，使用已生成的 failure localization 作为机制负结果，不扩展新方法。",
                "",
            ]
        )
        (self.run_dir / "MORNING_README.md").write_text(morning, encoding="utf-8")

    def _finalize_if_needed(self) -> bool:
        now_epoch = time.time()
        if self.stop_requested or now_epoch >= float(self.state["hard_end_epoch"]):
            if self.hard_end_seen_monotonic is None:
                self.hard_end_seen_monotonic = time.monotonic()
                self._event("hard_end", "hard end reached; no new jobs will be launched")
            active = any(job.get("status") == "running" for job in self.state["jobs"].values())
            if active and time.monotonic() - self.hard_end_seen_monotonic < HARD_END_GRACE_S:
                return False
            self._write_artifact_audit()
            self._write_live_report()
            self._write_final_reports()
            self.state["finalized"] = True
            self._save_state()
            return True
        return False

    def run(self) -> None:
        self.pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        self._event("supervisor_alive", f"pid={os.getpid()}")
        self._save_state()
        while True:
            self._update_gpu_snapshot()
            self._poll_workers()
            self._maybe_enqueue_analysis()
            self._maybe_advance_phase()
            self._maybe_add_seed_confirmation()
            self._maybe_add_clean_latency()
            if not self.stop_requested and time.time() < float(self.state["hard_end_epoch"]):
                self._maybe_launch_cpu()
                self._maybe_launch_gpu()
            self._write_artifact_audit()
            self._write_heartbeat()
            self._write_live_report()
            self._health_audit_if_stalled()
            self._save_state()
            if self._finalize_if_needed():
                return
            time.sleep(float(self.args.poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--collection-root", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--duration-hours", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_hours <= 0:
        raise ValueError("duration-hours must be positive")
    supervisor = Supervisor(args)

    def request_stop(signum: int, _frame: Any) -> None:
        supervisor.stop_requested = True
        supervisor._event("signal", f"received signal {signum}; entering graceful finalization")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    supervisor.run()


if __name__ == "__main__":
    main()

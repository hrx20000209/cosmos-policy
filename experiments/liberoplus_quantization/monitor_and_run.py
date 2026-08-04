#!/usr/bin/env python3
"""Wait for genuinely idle GPUs, then run the resumable Phase 0-3 pipeline."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

EXP = Path(__file__).resolve().parent
REPO = EXP.parents[1]
LOG_DIR = Path(os.environ.get("LIBEROPLUS_MONITOR_LOG_DIR", EXP / "logs/monitor"))
STATUS_PATH = LOG_DIR / "status.json"
LOCK_PATH = LOG_DIR / "monitor.lock"
STOP_PATH = LOG_DIR / "monitor.stop"
PIPELINE_LOG = LOG_DIR / "pipeline.log"

CHECK_INTERVAL_SECONDS = int(os.environ.get("LIBEROPLUS_GPU_CHECK_INTERVAL", "60"))
REQUIRED_CONSECUTIVE_CHECKS = int(
    os.environ.get("LIBEROPLUS_GPU_CONSECUTIVE_CHECKS", "3")
)
MIN_FREE_MEMORY_MIB = int(os.environ.get("LIBEROPLUS_GPU_MIN_FREE_MIB", "12000"))
MAX_UTILIZATION_PERCENT = int(os.environ.get("LIBEROPLUS_GPU_MAX_UTIL", "10"))
GPU_IDS_RAW = os.environ.get("LIBEROPLUS_GPU_IDS", "")
GPU_ALLOWLIST = (
    {int(item.strip()) for item in GPU_IDS_RAW.split(",") if item.strip()}
    if GPU_IDS_RAW
    else None
)
MAX_GPUS = min(4, len(GPU_ALLOWLIST)) if GPU_ALLOWLIST else 4

BASELINE = ["baseline_bf16.yaml"]
QUANTIZATION = [
    "baseline_fp16.yaml",
    "quant_int8.yaml",
    "quant_int8_w8a8.yaml",
    "quant_int4.yaml",
    "quant_int4_fake.yaml",
]
BRANCH = [
    "branch_quant_vision.yaml",
    "branch_quant_action.yaml",
    "branch_quant_attention.yaml",
    "branch_quant_mlp.yaml",
    "branch_backbone_int4_action_high.yaml",
]
ASYNC_BF16 = [
    "async_openloop_1.yaml",
    "async_openloop_2.yaml",
    "async_openloop_4.yaml",
    "async_openloop_8.yaml",
    "async_openloop_16.yaml",
]
ASYNC_INT8 = [
    "async_int8_openloop_1.yaml",
    "async_int8_openloop_2.yaml",
    "async_int8_openloop_4.yaml",
    "async_int8_openloop_8.yaml",
    "async_int8_openloop_16.yaml",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    temporary.replace(path)


class Monitor:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.started_at = now()
        self.completed_configs: list[str] = []
        if STATUS_PATH.exists():
            try:
                previous_status = json.loads(STATUS_PATH.read_text())
                self.completed_configs = list(
                    previous_status.get("completed_configs", [])
                )
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        self.failed_configs: dict[str, str] = {}
        self.current_config: str | None = None
        self.last_gpu_stats: list[dict[str, int]] = []
        self.message = "initializing"

    def status(self, state: str, **extra: Any) -> None:
        atomic_json(
            STATUS_PATH,
            {
                "state": state,
                "pid": os.getpid(),
                "started_at": self.started_at,
                "updated_at": now(),
                "current_config": self.current_config,
                "completed_configs": self.completed_configs,
                "failed_configs": self.failed_configs,
                "gpu_policy": {
                    "check_interval_seconds": CHECK_INTERVAL_SECONDS,
                    "required_consecutive_checks": REQUIRED_CONSECUTIVE_CHECKS,
                    "min_free_memory_mib": MIN_FREE_MEMORY_MIB,
                    "max_utilization_percent": MAX_UTILIZATION_PERCENT,
                    "max_gpus": MAX_GPUS,
                    "allowed_gpu_ids": (
                        sorted(GPU_ALLOWLIST) if GPU_ALLOWLIST is not None else None
                    ),
                },
                "last_gpu_stats": self.last_gpu_stats,
                "message": self.message,
                **extra,
            },
        )

    def query_gpus(self) -> list[dict[str, int]]:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in output.splitlines():
            index, free, utilization = (int(item.strip()) for item in line.split(","))
            if GPU_ALLOWLIST is not None and index not in GPU_ALLOWLIST:
                continue
            rows.append(
                {
                    "index": index,
                    "free_memory_mib": free,
                    "utilization_percent": utilization,
                }
            )
        return rows

    def wait_for_gpus(self) -> list[int]:
        streaks: dict[int, int] = {}
        while True:
            if STOP_PATH.exists():
                self.message = f"stop file detected: {STOP_PATH}"
                self.status("stopped")
                raise SystemExit(0)
            try:
                self.last_gpu_stats = self.query_gpus()
            except Exception as error:
                self.message = f"nvidia-smi query failed: {error}"
                self.status("waiting_for_gpu")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            ready_now = set()
            for gpu in self.last_gpu_stats:
                index = gpu["index"]
                if (
                    gpu["free_memory_mib"] >= MIN_FREE_MEMORY_MIB
                    and gpu["utilization_percent"] <= MAX_UTILIZATION_PERCENT
                ):
                    streaks[index] = streaks.get(index, 0) + 1
                    ready_now.add(index)
                else:
                    streaks[index] = 0
            stable = [
                gpu
                for gpu in self.last_gpu_stats
                if gpu["index"] in ready_now
                and streaks[gpu["index"]] >= REQUIRED_CONSECUTIVE_CHECKS
            ]
            stable.sort(key=lambda item: item["free_memory_mib"], reverse=True)
            if stable:
                selected = [item["index"] for item in stable[:MAX_GPUS]]
                self.message = f"selected stable idle GPUs: {selected}"
                self.status("gpu_ready", selected_gpus=selected)
                return selected
            progress = max(streaks.values(), default=0)
            self.message = (
                "waiting for GPU: "
                f"best idle streak {progress}/{REQUIRED_CONSECUTIVE_CHECKS}"
            )
            self.status("waiting_for_gpu")
            time.sleep(CHECK_INTERVAL_SECONDS)

    def run_command(self, command: list[str], log_name: str) -> None:
        log_path = LOG_DIR / log_name
        with log_path.open("a", buffering=1) as log:
            log.write(f"\n[{now()}] COMMAND {' '.join(command)}\n")
            result = subprocess.run(
                command,
                cwd=REPO,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"command failed with exit {result.returncode}; see {log_path}"
            )

    def run_config(self, config: str) -> None:
        gpus = self.wait_for_gpus()
        self.current_config = config
        self.message = f"running {config} on GPUs {gpus}"
        self.status("running", selected_gpus=gpus)
        command = [
            str(REPO / ".venv/bin/python"),
            str(EXP / "scripts/sweep_runner.py"),
            "--configs",
            config,
            "--gpus",
            ",".join(str(gpu) for gpu in gpus),
        ]
        self.run_command(command, f"{Path(config).stem}.log")
        self.completed_configs.append(config)
        self.current_config = None
        self.message = f"completed {config}"
        self.status("between_configs")

    def run_config_resilient(self, config: str, *, required: bool = False) -> bool:
        if config in self.completed_configs:
            return True
        try:
            self.run_config(config)
            return True
        except Exception as error:
            self.failed_configs[config] = str(error)
            self.current_config = None
            self.message = f"failed {config}; continuing pipeline: {error}"
            self.status("config_failed")
            if required:
                raise
            return False

    def aggregate(self) -> None:
        self.run_command(
            [str(REPO / ".venv/bin/python"), str(EXP / "aggregate_results.py")],
            "aggregate.log",
        )
        self.run_command(
            [str(REPO / ".venv/bin/python"), str(EXP / "plot_results.py")],
            "plot.log",
        )

    def summary_row(self, experiment_name: str) -> dict[str, str] | None:
        path = EXP / "summaries/overall_summary.csv"
        if not path.exists() or path.stat().st_size == 0:
            return None
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row["experiment_name"] == experiment_name:
                    return row
        return None

    def baseline_gate(self) -> float:
        self.aggregate()
        row = self.summary_row("baseline_bf16")
        if row is None:
            raise RuntimeError("baseline summary row is missing")
        completed = int(float(row["completed_episodes"]))
        success_rate = float(row["success_rate"])
        if completed < 4:
            raise RuntimeError(f"baseline completed only {completed}/4 episodes")
        episodes_path = EXP / "summaries/episodes.csv"
        with episodes_path.open() as stream:
            errors = [
                row
                for row in csv.DictReader(stream)
                if row["experiment_name"] == "baseline_bf16"
                and (
                    row["termination_reason"].startswith(
                        "environment_or_evaluation_error"
                    )
                    or row["termination_reason"].startswith("evaluation_error")
                )
            ]
        if errors:
            raise RuntimeError(f"baseline has {len(errors)} evaluation errors")
        if success_rate < 0.50:
            raise RuntimeError(
                f"baseline success {success_rate:.3f} is below the 0.50 pilot gate"
            )
        return success_rate

    def int8_async_allowed(self, baseline_success_rate: float) -> bool:
        self.aggregate()
        row = self.summary_row("quant_int8_weight_only")
        if row is None or int(float(row["completed_episodes"])) < 4:
            return False
        return float(row["success_rate"]) >= baseline_success_rate - 0.25

    def run(self) -> None:
        with PIPELINE_LOG.open("a", buffering=1) as pipeline_log:
            pipeline_log.write(f"\n[{now()}] monitor started pid={os.getpid()}\n")
        for config in BASELINE:
            self.run_config_resilient(config, required=True)
        baseline_success_rate = self.baseline_gate()
        for config in QUANTIZATION:
            self.run_config_resilient(config)
        for config in BRANCH:
            self.run_config_resilient(config)
        for config in ASYNC_BF16:
            self.run_config_resilient(config)
        run_int8_async = self.int8_async_allowed(baseline_success_rate)
        if run_int8_async:
            for config in ASYNC_INT8:
                self.run_config_resilient(config)
        else:
            self.message = (
                "INT8 open-loop sweep skipped: INT8 pilot was missing or dropped "
                "more than one success out of four versus BF16."
            )
            self.status("between_configs", int8_async_skipped=True)
        self.aggregate()
        self.message = (
            "Phase 0-3 unattended pipeline completed"
            if not self.failed_configs
            else "Phase 0-3 pipeline completed with failed configurations"
        )
        final_state = "completed" if not self.failed_configs else "completed_with_failures"
        self.status(final_state, int8_async_skipped=not run_int8_async)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock_stream = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another monitor holds {LOCK_PATH}", file=sys.stderr)
        raise SystemExit(2)
    monitor = Monitor()
    try:
        monitor.run()
    except SystemExit:
        raise
    except Exception as error:
        monitor.message = str(error)
        monitor.status("failed", traceback=traceback.format_exc())
        with PIPELINE_LOG.open("a") as log:
            log.write(f"[{now()}] FAILED: {error}\n{traceback.format_exc()}\n")
        raise
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()


if __name__ == "__main__":
    main()

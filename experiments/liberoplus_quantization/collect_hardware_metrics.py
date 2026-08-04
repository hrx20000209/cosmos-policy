"""Hardware metadata and optional NVML power sampling."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


def _command(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"unavailable: {error}"


def collect_hardware_info() -> dict:
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "nvidia_smi": _command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ]
        ),
    }
    try:
        import psutil

        info.update(
            {
                "system_memory_bytes": psutil.virtual_memory().total,
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
            }
        )
    except ImportError:
        info["psutil"] = "unavailable"
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    except ImportError:
        info["torch"] = "unavailable"
    return info


@dataclass
class PowerSampler:
    """Integrate NVML power samples.

    This is an estimate from board-power samples, not a laboratory-grade energy
    measurement. ``supported`` and ``measurement_note`` are always logged so it
    cannot be mistaken for a direct energy counter.
    """

    device_index: int = 0
    interval_seconds: float = 0.1
    samples_watts: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    supported: bool = False
    error: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            physical_index = self.device_index
            if visible and visible.split(",")[0].strip().isdigit():
                physical_index = int(visible.split(",")[0])
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
            self.supported = True
        except Exception as error:
            self.error = str(error)
            return

        self._stop.clear()

        def sample() -> None:
            while not self._stop.is_set():
                try:
                    self.timestamps.append(time.perf_counter())
                    self.samples_watts.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                except Exception as error:
                    self.error = str(error)
                    break
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        energy_joules = 0.0
        for index in range(1, len(self.samples_watts)):
            dt = self.timestamps[index] - self.timestamps[index - 1]
            energy_joules += 0.5 * (self.samples_watts[index] + self.samples_watts[index - 1]) * dt
        return {
            "power_measurement_supported": self.supported,
            "average_power_watts": (
                sum(self.samples_watts) / len(self.samples_watts) if self.samples_watts else None
            ),
            "energy_joules_estimated": energy_joules if len(self.samples_watts) >= 2 else None,
            "power_samples": len(self.samples_watts),
            "power_error": self.error,
            "measurement_note": (
                "NVML board-power integration at 10 Hz; estimate, not a direct energy counter."
                if self.supported
                else "Power measurement unavailable; no energy conclusion is supported."
            ),
        }


if __name__ == "__main__":
    destination = Path("experiments/liberoplus_quantization/system_info.json")
    destination.write_text(json.dumps(collect_hardware_info(), indent=2) + "\n")
    print(destination)

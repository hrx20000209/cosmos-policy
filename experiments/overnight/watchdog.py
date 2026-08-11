"""Non-invasive GPU/process watchdog for the autonomous WAM run.

The watchdog records state only.  It never kills, pauses, or renices a process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def gpu_rows() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    rows = []
    for line in output.splitlines():
        index, memory, utilization, temperature, power = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "memory_used_mib": int(memory),
                "utilization_percent": int(utilization),
                "temperature_c": int(temperature),
                "power_w": float(power),
            }
        )
    return rows


def pid_state(pid_file: Path) -> dict[str, object]:
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return {"pid_file": str(pid_file), "pid": None, "alive": False}
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
    return {"pid_file": str(pid_file), "pid": pid, "alive": alive}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pid-dir", required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--duration-seconds", type=float, default=43200.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pid_dir = Path(args.pid_dir)
    deadline = time.monotonic() + args.duration_seconds
    with output.open("a", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            record = {
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
                "gpus": gpu_rows(),
                "owned_managers": [pid_state(path) for path in sorted(pid_dir.glob("*.manager.pid"))],
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            handle.flush()
            time.sleep(min(args.interval_seconds, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()

"""Record a non-destructive snapshot of the server GPU pool."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def main() -> None:
    output = Path("reports/server_deep_validation/manifests/GPU_POOL.json")
    query = run(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    )
    processes: dict[int, list[dict[str, str]]] = {}
    try:
        compute = run("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits")
        for row in csv.reader(compute.splitlines(), skipinitialspace=True):
            if len(row) < 3:
                continue
            pid = row[1].strip()
            try:
                user = run("ps", "-o", "user=", "-p", pid).strip()
            except Exception:
                user = "unknown"
            processes.setdefault(-1, []).append({"gpu_uuid": row[0].strip(), "pid": pid, "process": row[2].strip(), "user": user})
    except Exception:
        pass
    gpus = []
    for row in csv.reader(query.splitlines(), skipinitialspace=True):
        if len(row) < 7:
            continue
        index = int(row[0].strip())
        used = int(row[3].strip())
        total = int(row[2].strip())
        gpus.append(
            {
                "gpu_id": index,
                "name": row[1].strip(),
                "memory_total_mb": total,
                "memory_used_mb": used,
                "memory_free_mb": int(row[4].strip()),
                "utilization_gpu_percent": int(row[5].strip()),
                "temperature_c": int(row[6].strip()),
                "truly_free_by_memory": used <= 1024,
            }
        )
    payload = {
        "schema_version": 1,
        "captured_at_ns": time.time_ns(),
        "policy": "never kill or signal external processes",
        "gpus": gpus,
        "compute_processes": processes.get(-1, []),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

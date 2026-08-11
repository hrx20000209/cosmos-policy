"""Release owned workers after their atomic summary boundary is written."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import psutil


ROOTS = (Path("/data/rxhuang/wam_server_deep_validation/queue_a/f1_collection"), Path("/data/rxhuang/wam_server_deep_validation/queue_a/ablation"), Path("/data/rxhuang/wam_server_deep_validation/queue_b/oracle"), Path("/data/rxhuang/wam_server_deep_validation/queue_d/latency"), Path("/data/rxhuang/wam_server_deep_validation/queue_f/closed_loop"))


def main() -> None:
    while True:
        for process in psutil.process_iter(["pid", "cmdline"]):
            try:
                command = process.info.get("cmdline") or []
                if not any(item.endswith("run_server_f1_collection.py") or item.endswith("run_server_ablation.py") or item.endswith("run_server_late_binding_oracle.py") or item.endswith("run_server_latency_matrix.py") or item.endswith("run_server_closed_loop_benchmark.py") for item in command):
                    continue
                output_index = command.index("--output-dir") if "--output-dir" in command else -1
                if output_index < 0 or output_index + 1 >= len(command):
                    continue
                output = Path(command[output_index + 1])
                # This watchdog belongs to the original deep-validation run.
                # Do not act on a later full-scale run that happens to use the
                # same runner name; process ownership alone is insufficient
                # once multiple experiment roots coexist.
                try:
                    output_resolved = output.resolve()
                    if not any(output_resolved == root.resolve() or root.resolve() in output_resolved.parents for root in ROOTS):
                        continue
                except OSError:
                    continue
                complete = False
                if "run_server_f1_collection.py" in " ".join(command):
                    complete = (output / f"summary_shard{int(command[command.index('--shard-index') + 1]):02d}.json").is_file()
                elif "run_server_ablation.py" in " ".join(command):
                    complete = (output / f"summary_shard{int(command[command.index('--shard-index') + 1]):02d}.json").is_file()
                else:
                    complete = any(output.glob("summary_shard*.json"))
                if complete:
                    try:
                        os.killpg(process.info["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, KeyError):
                continue
        time.sleep(30)


if __name__ == "__main__":
    main()

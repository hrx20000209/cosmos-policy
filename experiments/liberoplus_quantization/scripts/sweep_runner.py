"""Launch one or more configs, optionally sharded over GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
EVALUATOR = EXP / "run_libero_plus_eval.py"


def run_configs(configs: list[Path], gpus: list[int], dry_run: bool, limit: int) -> None:
    env_base = os.environ.copy()
    magick_home = env_base.get(
        "MAGICK_HOME", "/data/rxhuang/envs/imagemagick"
    )
    library_path = env_base.get("LD_LIBRARY_PATH", "")
    env_base.update(
        {
            "MUJOCO_GL": "egl",
            "__EGL_VENDOR_LIBRARY_FILENAMES": (
                "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
            ),
            "PYOPENGL_PLATFORM": "egl",
            "TOKENIZERS_PARALLELISM": "false",
            "MAGICK_HOME": magick_home,
            "LD_LIBRARY_PATH": (
                f"{library_path}:{magick_home}/lib"
                if library_path
                else f"{magick_home}/lib"
            ),
        }
    )
    for config in configs:
        processes: list[subprocess.Popen] = []
        for shard_index, gpu in enumerate(gpus):
            command = [
                str(REPO / ".venv/bin/python"),
                str(EVALUATOR),
                "--config",
                str(config),
                "--shard",
                f"{shard_index}/{len(gpus)}",
            ]
            if limit:
                command.extend(["--limit", str(limit)])
            if dry_run:
                command.append("--dry-run")
            env = dict(env_base)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
            print(" ".join(command), f"[GPU {gpu}]", flush=True)
            processes.append(subprocess.Popen(command, cwd=REPO, env=env))
        failures = [process.wait() for process in processes]
        if any(failures):
            raise SystemExit(f"config {config.name} failed with codes {failures}")


def parse_and_run(default_names: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=default_names)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config_dir = EXP / "configs"
    configs = [
        Path(name) if Path(name).is_absolute() else config_dir / name
        for name in args.configs
    ]
    run_configs(configs, [int(item) for item in args.gpus.split(",")], args.dry_run, args.limit)


if __name__ == "__main__":
    parse_and_run(sys.argv[1:])

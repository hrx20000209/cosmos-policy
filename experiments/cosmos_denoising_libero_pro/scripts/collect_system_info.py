#!/usr/bin/env python3
"""Collect immutable experiment provenance without modifying user configs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT / "experiments/cosmos_denoising_libero_pro"


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as error:
        return f"ERROR: {type(error).__name__}: {error}"


def git_state(path: Path) -> dict:
    return {
        "path": str(path),
        "commit": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "branch": command("git", "-C", str(path), "branch", "--show-current"),
        "status": command("git", "-C", str(path), "status", "--short"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    packages = {}
    for name in (
        "torch",
        "numpy",
        "robosuite",
        "mujoco",
        "PyOpenGL",
        "transformers",
        "scipy",
        "matplotlib",
        "pandas",
        "imageio",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    paths = {
        "checkpoint": Path(
            "/data/rxhuang/models/cosmos-policy-libero-2b/"
            "Cosmos-Policy-LIBERO-Predict2-2B.pt"
        ),
        "dataset_stats": Path(
            "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
        ),
        "base_t5": Path(
            "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"
        ),
    }
    report = {
        "collected_at": command("date", "--iso-8601=seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": packages,
        "nvidia_smi": command("nvidia-smi"),
        "gpus": command(
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ),
        "repositories": [
            git_state(PROJECT),
            git_state(Path("/home/rxhuang/Projects/LIBERO")),
            git_state(Path("/data/rxhuang/repos/LIBERO-PRO")),
            git_state(Path("/data/rxhuang/LIBERO-plus")),
        ],
        "files": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for key, path in paths.items()
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "LD_LIBRARY_PATH",
                "LIBERO_CONFIG_PATH",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "home_libero_config": Path("~/.libero/config.yaml").expanduser().read_text(
            encoding="utf-8"
        ),
        "import_conflict": (
            "The environment's editable default resolves libero to /data/rxhuang/LIBERO-plus; "
            "workers install an explicit outer-package shim pinned to each manifest row."
        ),
    }
    json_path = EXPERIMENT / "system_info.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    text = [
        f"collected_at: {report['collected_at']}",
        f"python: {report['python']}",
        f"platform: {report['platform']}",
        "packages: " + json.dumps(packages, ensure_ascii=False),
        "gpus:",
        report["gpus"],
        "repositories:",
        *[json.dumps(value, ensure_ascii=False) for value in report["repositories"]],
        "files:",
        json.dumps(report["files"], ensure_ascii=False, indent=2),
        "default ~/.libero/config.yaml:",
        report["home_libero_config"],
        "isolation:",
        report["import_conflict"],
    ]
    (EXPERIMENT / "system_info.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "status": "completed"}, indent=2))


if __name__ == "__main__":
    main()

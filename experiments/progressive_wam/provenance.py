"""Reproducibility metadata captured with every progressive-WAM run."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(repo: str | Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"<unavailable: {exc}>"


def repository_state(repo: str | Path) -> dict[str, Any]:
    return {
        "path": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "uncommitted_diff_stat": _git(repo, "diff", "--stat"),
    }


def file_sha256(path: str | Path, max_bytes: int | None = None) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    read = 0
    with p.open("rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
            read += len(block)
            if max_bytes is not None and read >= max_bytes:
                break
    return digest.hexdigest()


def hardware_report() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            info["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except Exception as exc:  # pragma: no cover
        info["torch"] = f"<unavailable: {exc}>"
    return info


def run_provenance(
    repos: dict[str, str | Path],
    checkpoint: str | Path | None,
    config: dict[str, Any],
    *,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Everything needed to re-run this experiment, per the reproducibility spec."""
    return {
        "config": config,
        "repositories": {name: repository_state(path) for name, path in repos.items()},
        "checkpoint": {
            "path": str(checkpoint) if checkpoint else None,
            "size_bytes": Path(checkpoint).stat().st_size if checkpoint and Path(checkpoint).is_file() else None,
            # Hashing 3.9 GB costs ~20 s; callers pass a known-good digest instead
            # of paying it on every run.
            "sha256": checkpoint_sha256,
        },
        "hardware": hardware_report(),
        "argv": sys.argv,
    }


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str))

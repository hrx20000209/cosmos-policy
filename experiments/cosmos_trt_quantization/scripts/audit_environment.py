#!/usr/bin/env python3
"""Print a complete, non-mutating environment audit for this experiment."""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODULES = [
    "torch",
    "torchao",
    "tensorrt",
    "tensorrt_rtx",
    "torch_tensorrt",
    "modelopt",
    "transformer_engine",
    "onnx",
    "onnxruntime",
]
TOOLS = ["nvidia-smi", "nvcc", "trtexec", "nsys", "ncu", "uv"]


def run(*args: str) -> str:
    try:
        return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()
    except Exception as exc:
        return f"ERROR: {exc!r}"


print(f"timestamp={datetime.now().astimezone().isoformat()}")
print(f"hostname={platform.node()}")
print(f"platform={platform.platform()}")
print(f"python={sys.version.replace(os.linesep, ' ')}")
print(f"python_executable={sys.executable}")
print(f"cwd={Path.cwd()}")
print(f"git_branch={run('git', 'branch', '--show-current')}")
print(f"git_commit={run('git', 'rev-parse', 'HEAD')}")
print("\n[tools]")
for name in TOOLS:
    path = shutil.which(name)
    print(f"{name}={path or 'NOT AVAILABLE'}")
    if path and name in {"nvcc", "trtexec", "nsys", "ncu", "uv"}:
        flag = "--version"
        print(run(path, flag).splitlines()[0:4])

print("\n[python modules]")
for name in MODULES:
    try:
        module = importlib.import_module(name)
        print(f"{name}={getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=NOT AVAILABLE {exc!r}")

try:
    import torch

    print("\n[torch/cuda]")
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_runtime={torch.version.cuda}")
    print(f"cudnn={torch.backends.cudnn.version()}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"visible_gpu_count={torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(
                f"gpu[{index}]={props.name}; capability={props.major}.{props.minor}; "
                f"memory_bytes={props.total_memory}; multiprocessors={props.multi_processor_count}"
            )
except Exception as exc:
    print(f"torch_cuda_audit_error={exc!r}")

print("\n[nvidia-smi]")
print(run("nvidia-smi"))
print("\n[nvcc]")
nvcc = shutil.which("nvcc") or "/usr/local/cuda-12.6/bin/nvcc"
print(run(nvcc, "--version"))


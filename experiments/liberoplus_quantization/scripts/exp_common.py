"""Common infrastructure for the LIBERO-Plus quantization experiment.

Pure, model-agnostic helpers:
  - resumable append-safe CSV writers (trials.csv, inference_steps.csv)
  - config hashing + git commit capture
  - CudaTimer: accurate GPU timing via CUDA events (+ wall clock)
  - memory stat helpers

Everything downstream (success-rate eval, latency microbench, replanning) writes
through these so the raw schema stays identical and reproducible.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from typing import Any

# ----------------------------------------------------------------------------- paths
REPO = "/home/rxhuang/Projects/cosmos-policy"
EXP = f"{REPO}/experiments/liberoplus_quantization"
RAW = f"{EXP}/raw"           # symlink -> /data/rxhuang/liberoplus_quantization/raw
SUMM = f"{EXP}/summaries"
PROF = f"{EXP}/profiles"
LOGS = f"{EXP}/logs"

TRIALS_FIELDS = [
    "model_family", "checkpoint", "quant_mode", "quant_backend", "hardware_accelerated",
    "task_id", "task_name", "task_suite", "perturbation_category", "difficulty",
    "perturbation_level", "seed", "success", "termination_reason",
    "environment_steps", "executed_actions", "policy_calls",
    "episode_total_ms", "inference_total_ms", "environment_total_ms",
    "peak_memory_mb", "git_commit", "config_hash",
]

INFER_FIELDS = [
    "quant_mode", "task_id", "episode_id", "policy_call_index",
    "chunk_size", "num_open_loop_steps",
    "preprocess_ms", "h2d_ms", "model_forward_ms", "denoising_ms",
    "action_decode_ms", "postprocess_ms", "total_policy_step_ms",
    "amortized_ms_per_action", "gpu_memory_allocated_mb", "gpu_memory_reserved_mb",
]


def git_commit(repo: str = REPO) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


class CsvAppender:
    """Append-safe CSV writer. Writes header once; flushes every row so a killed
    run keeps all completed rows (resumability)."""

    def __init__(self, path: str, fields: list[str]):
        self.path = path
        self.fields = fields
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self.fh = open(path, "a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
            self.fh.flush()

    def write(self, row: dict[str, Any]):
        self.w.writerow(row)
        self.fh.flush()
        os.fsync(self.fh.fileno())

    def close(self):
        self.fh.close()


def completed_keys(path: str, key_fields: list[str]) -> set[tuple]:
    """Read an existing CSV and return the set of key tuples already present, so a
    resumed run can skip finished work."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for r in csv.DictReader(f):
            done.add(tuple(r.get(k, "") for k in key_fields))
    return done


# ----------------------------------------------------------------------------- timing
class CudaTimer:
    """Accurate GPU timing via CUDA events with correct synchronization.

    Usage:
        t = CudaTimer()
        with t.section("model_forward"): ...     # GPU-timed
        t.wall_start(); ...; wall_ms = t.wall_stop()
    Records both per-section GPU ms and can be reset per policy call.
    """

    def __init__(self):
        import torch
        self.torch = torch
        self.sections: dict[str, float] = {}
        self._wall0 = None

    @contextmanager
    def section(self, name: str):
        torch = self.torch
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            self.sections[name] = self.sections.get(name, 0.0) + start.elapsed_time(end)
        else:
            t0 = time.perf_counter_ns()
            yield
            self.sections[name] = self.sections.get(name, 0.0) + (time.perf_counter_ns() - t0) / 1e6

    def wall_start(self):
        self._wall0 = time.perf_counter_ns()

    def wall_stop(self) -> float:
        return (time.perf_counter_ns() - self._wall0) / 1e6

    def reset(self):
        self.sections = {}


def mem_stats_mb() -> dict[str, float]:
    import torch
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak_allocated": 0.0, "peak_reserved": 0.0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1e6,
        "reserved": torch.cuda.memory_reserved() / 1e6,
        "peak_allocated": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved": torch.cuda.max_memory_reserved() / 1e6,
    }


def reset_peak_mem():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def pctl(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def latency_summary(vals: list[float]) -> dict[str, float]:
    import statistics
    if not vals:
        return {k: float("nan") for k in
                ["count", "mean", "std", "p50", "p90", "p95", "p99", "min", "max"]}
    return {
        "count": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "p50": pctl(vals, 0.50), "p90": pctl(vals, 0.90),
        "p95": pctl(vals, 0.95), "p99": pctl(vals, 0.99),
        "min": min(vals), "max": max(vals),
    }

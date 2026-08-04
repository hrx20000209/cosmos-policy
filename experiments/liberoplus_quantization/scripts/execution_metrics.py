"""Model-independent action-chunk execution metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ChunkMetrics:
    chunk_index: int
    generated_step: int
    generated_time_s: float
    generated_actions: int
    planned_open_loop_steps: int
    denoising_steps: int
    executed_actions: int = 0
    discarded_actions: int = 0
    staleness_steps: list[int] = field(default_factory=list)
    staleness_ms: list[float] = field(default_factory=list)
    boundary_l1: float | None = None
    boundary_l2: float | None = None
    boundary_per_dimension: list[float] = field(default_factory=list)

    def execute(self, current_step: int, current_time_s: float) -> None:
        self.executed_actions += 1
        self.staleness_steps.append(current_step - self.generated_step)
        self.staleness_ms.append((current_time_s - self.generated_time_s) * 1000.0)

    def finish(self) -> None:
        self.discarded_actions = self.generated_actions - self.executed_actions
        if self.discarded_actions < 0:
            raise ValueError("executed actions cannot exceed generated actions")

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "generated_step": self.generated_step,
            "generated_actions": self.generated_actions,
            "planned_open_loop_steps": self.planned_open_loop_steps,
            "denoising_steps": self.denoising_steps,
            "executed_actions": self.executed_actions,
            "discarded_actions": self.discarded_actions,
            "stale_action_ratio": self.discarded_actions / max(self.generated_actions, 1),
            "observation_staleness_steps_mean": _mean(self.staleness_steps),
            "observation_staleness_steps_max": max(self.staleness_steps, default=0),
            "observation_staleness_ms_mean": _mean(self.staleness_ms),
            "observation_staleness_ms_max": max(self.staleness_ms, default=0.0),
            "action_discontinuity_l1": self.boundary_l1,
            "action_discontinuity_l2": self.boundary_l2,
            "action_discontinuity_per_dimension": self.boundary_per_dimension,
        }


def set_boundary_metrics(chunk: ChunkMetrics, previous_action: Any, next_action: Any) -> None:
    previous = np.asarray(previous_action, dtype=np.float64)
    current = np.asarray(next_action, dtype=np.float64)
    difference = np.abs(current - previous)
    chunk.boundary_l1 = float(np.linalg.norm(current - previous, ord=1))
    chunk.boundary_l2 = float(np.linalg.norm(current - previous, ord=2))
    chunk.boundary_per_dimension = difference.tolist()


def summarize_chunks(chunks: list[ChunkMetrics]) -> dict[str, Any]:
    for chunk in chunks:
        chunk.finish()
    generated = sum(chunk.generated_actions for chunk in chunks)
    executed = sum(chunk.executed_actions for chunk in chunks)
    discarded = sum(chunk.discarded_actions for chunk in chunks)
    discontinuity_l1 = [chunk.boundary_l1 for chunk in chunks if chunk.boundary_l1 is not None]
    discontinuity_l2 = [chunk.boundary_l2 for chunk in chunks if chunk.boundary_l2 is not None]
    stale_steps = [item for chunk in chunks for item in chunk.staleness_steps]
    stale_ms = [item for chunk in chunks for item in chunk.staleness_ms]
    return {
        "generated_actions": generated,
        "executed_actions": executed,
        "discarded_actions": discarded,
        "stale_action_ratio": discarded / max(generated, 1),
        "valid_executed_actions_per_chunk": executed / max(len(chunks), 1),
        "action_discontinuity_l1_mean": _mean(discontinuity_l1),
        "action_discontinuity_l2_mean": _mean(discontinuity_l2),
        "observation_staleness_steps_mean": _mean(stale_steps),
        "observation_staleness_steps_max": max(stale_steps, default=0),
        "observation_staleness_ms_mean": _mean(stale_ms),
        "observation_staleness_ms_max": max(stale_ms, default=0.0),
    }


def visual_change(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 0.0
    left = np.asarray(previous, dtype=np.float32) / 255.0
    right = np.asarray(current, dtype=np.float32) / 255.0
    return float(np.mean(np.abs(right - left)))


def action_jerk(actions: list[np.ndarray]) -> float:
    if len(actions) < 3:
        return 0.0
    a, b, c = (np.asarray(item, dtype=np.float64) for item in actions[-3:])
    return float(np.linalg.norm((c - b) - (b - a), ord=2))


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


"""M4: immutable execution-side state, distinct from predictive state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionState:
    episode_key: str
    control_step: int
    committed_actions: np.ndarray
    latest_proprio: np.ndarray

    def __post_init__(self) -> None:
        actions = np.asarray(self.committed_actions)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"expected [N,7] committed actions, got {actions.shape}")
        if np.asarray(self.latest_proprio).shape != (9,):
            raise ValueError("latest_proprio must be 9-D")


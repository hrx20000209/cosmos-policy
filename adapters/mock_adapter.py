from __future__ import annotations

import time
from typing import Any

import numpy as np

from adapters.base import PolicyOutput
from runtime.async_pipeline import InferenceRequest
from runtime.observation_buffer import Observation


class MockAdapter:
    """Deterministic adapter for harness and metrics smoke tests only."""

    model_name = "mock"

    def __init__(self, config: dict[str, Any]):
        self.action_horizon = int(config.get("action_horizon", 16))
        self.action_dim = int(config.get("action_dim", 7))

    def reset(self, task_description: str, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def infer(self, observation: Observation, request: InferenceRequest, denoising_steps: int, history=None) -> PolicyOutput:
        start = time.monotonic_ns()
        actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        actions[:, -1] = -1.0
        elapsed = (time.monotonic_ns() - start) / 1e6
        return PolicyOutput(
            actions=actions,
            denoiser_forward_count=denoising_steps,
            stage_metrics_ms={"total_ms": elapsed, "dit_denoising_ms": elapsed},
        )

    def update_context(self, observations: list[Observation], predicted_actions: np.ndarray) -> dict[str, Any]:
        return {}

    def decode_future(self, output: PolicyOutput) -> Any:
        return None

    def close(self) -> None:
        pass


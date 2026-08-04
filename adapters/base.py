from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from runtime.async_pipeline import InferenceRequest
from runtime.observation_buffer import Observation


@dataclass
class PolicyOutput:
    actions: np.ndarray
    denoiser_forward_count: int
    generated_latent: Any = None
    future_state: Any = None
    value: float | None = None
    stage_metrics_ms: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


class PolicyAdapter(Protocol):
    model_name: str
    action_horizon: int

    def reset(self, task_description: str, seed: int) -> None: ...

    def infer(
        self,
        observation: Observation,
        request: InferenceRequest,
        denoising_steps: int,
        history: list[Observation] | None = None,
    ) -> PolicyOutput: ...

    def update_context(self, observations: list[Observation], predicted_actions: np.ndarray) -> dict[str, Any]: ...

    def decode_future(self, output: PolicyOutput) -> Any: ...

    def close(self) -> None: ...


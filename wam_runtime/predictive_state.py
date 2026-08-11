"""M0: a provenance-carrying predictive state emitted by a frozen WAM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PredictiveState:
    """A speculative joint latent at one execution boundary.

    ``latent`` is intentionally opaque to the runtime contract.  The policy
    owns its meaning; the runtime only records when it was generated and which
    action prefix it corresponds to.  This prevents an offline simulator state
    from quietly becoming a policy input.
    """

    episode_key: str
    request_index: int
    control_step: int
    latent: np.ndarray
    generated_action: np.ndarray
    checkpoint_sha256: str

    @classmethod
    def from_request(cls, episode_key: str, request: Mapping[str, Any]) -> "PredictiveState":
        latent = np.asarray(request["generated_latent"])
        action = np.asarray(request["fresh_action"], dtype=np.float32)
        if latent.ndim != 5:
            raise ValueError(f"expected [B,C,T,H,W] joint latent, got {latent.shape}")
        if action.shape != (16, 7):
            raise ValueError(f"expected a 16x7 action chunk, got {action.shape}")
        return cls(
            episode_key=str(episode_key),
            request_index=int(request["request_index"]),
            control_step=int(request["control_step"]),
            latent=latent,
            generated_action=action,
            checkpoint_sha256=str(request["checkpoint_sha256"]),
        )

    @property
    def next_control_step(self) -> int:
        return self.control_step + int(self.generated_action.shape[0])


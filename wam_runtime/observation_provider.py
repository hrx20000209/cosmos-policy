"""M2: runtime-safe fresh-observation provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ObservationSource = Literal["camera", "proprio", "recorded_offline"]


@dataclass(frozen=True)
class ObservationPacket:
    """A physical observation that may be consumed by a WAM encoder.

    ``recorded_offline`` is permitted only for reproducible analysis; it must
    never be passed to a live policy as simulator privilege.
    """

    control_step: int
    proprio: np.ndarray
    primary_image: np.ndarray | None = None
    wrist_image: np.ndarray | None = None
    source: ObservationSource = "camera"
    arrival_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        if np.asarray(self.proprio).shape != (9,):
            raise ValueError(f"expected LIBERO 9-D proprio, got {np.asarray(self.proprio).shape}")


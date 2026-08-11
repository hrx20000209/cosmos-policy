"""A measurement container for a Jetson AGX Thor export, not a latency claim."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ThorCostModel:
    fresh_vae_ms: float | None = None
    dit_prefix_ms: float | None = None
    dit_suffix_ms: float | None = None
    native_condition_recompute_ms: float | None = None
    memory_peak_mb: float | None = None
    device: str = "unmeasured"

    def complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.fresh_vae_ms,
                self.dit_prefix_ms,
                self.dit_suffix_ms,
                self.native_condition_recompute_ms,
                self.memory_peak_mb,
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "complete": self.complete()}

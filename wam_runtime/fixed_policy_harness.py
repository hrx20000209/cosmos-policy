"""M5: invariants for a frozen, non-value Cosmos policy experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ORIGINAL_COSMOS_CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


@dataclass(frozen=True)
class FixedPolicyContract:
    checkpoint: Path
    checkpoint_sha256: str
    denoising_steps: int = 1
    value_used: bool = False
    privileged_runtime_state_used: bool = False
    scheduler_or_threshold_used: bool = False

    def validate(self) -> None:
        name = str(self.checkpoint).lower()
        if "so101" in name or "finetun" in name:
            raise ValueError(f"refusing a finetuned/SO101 checkpoint: {self.checkpoint}")
        if self.checkpoint_sha256 != ORIGINAL_COSMOS_CHECKPOINT_SHA256:
            raise ValueError("checkpoint digest is not the original pre-finetune Cosmos policy")
        if self.denoising_steps != 1:
            raise ValueError("the modular validation contract fixes denoising_steps=1")
        if self.value_used:
            raise ValueError("Cosmos value is excluded from this runtime")
        if self.privileged_runtime_state_used:
            raise ValueError("simulator state is audit-only and cannot be a runtime policy input")
        if self.scheduler_or_threshold_used:
            raise ValueError("this validation measures mechanisms, not a scheduler")


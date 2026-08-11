"""M3: declarative feedback-assimilation interfaces for a frozen WAM.

This file intentionally does not implement a trust scheduler.  It records the
causal interface being tested and whether it is an oracle or a realizable
runtime path, so reports cannot accidentally promote an oracle to a system.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InterfaceCandidate(str, Enum):
    FULL_FRESH_RESTART = "full_fresh_restart"
    NATIVE_CONDITION_RECOMPUTE = "native_condition_recompute"
    PERSISTENT_VISUAL_CONDITION = "persistent_visual_condition"
    HIDDEN_STATE_PATCH_ORACLE = "hidden_state_patch_oracle"
    NOISY_LATENT_PATCH_ORACLE = "noisy_latent_patch_oracle"


@dataclass(frozen=True)
class RepairPlan:
    candidate: InterfaceCandidate
    injected_slots: tuple[int, ...]
    handoff_block: int | None
    requires_fresh_vae: bool
    oracle: bool
    rationale: str

    @property
    def realizable(self) -> bool:
        return not self.oracle


class AssimilationBackend:
    """Registry used by experiments and export bundles, not a policy chooser."""

    def plan(self, candidate: InterfaceCandidate, *, handoff_block: int | None = None) -> RepairPlan:
        if candidate is InterfaceCandidate.FULL_FRESH_RESTART:
            return RepairPlan(candidate, (1, 2, 3), None, True, False, "reference Fresh-1 recomputation")
        if candidate is InterfaceCandidate.NATIVE_CONDITION_RECOMPUTE:
            return RepairPlan(candidate, (1, 2, 3), None, True, False, "re-encode fresh condition through native API")
        if candidate is InterfaceCandidate.PERSISTENT_VISUAL_CONDITION:
            return RepairPlan(candidate, (1, 2, 3), None, True, False, "native persistent-condition feedback path")
        if candidate is InterfaceCandidate.HIDDEN_STATE_PATCH_ORACLE:
            return RepairPlan(candidate, (1, 2, 3), handoff_block, False, True, "offline causal diagnostic")
        if candidate is InterfaceCandidate.NOISY_LATENT_PATCH_ORACLE:
            return RepairPlan(candidate, (1, 2, 3), handoff_block, False, True, "offline causal diagnostic")
        raise ValueError(f"unsupported assimilation candidate: {candidate}")


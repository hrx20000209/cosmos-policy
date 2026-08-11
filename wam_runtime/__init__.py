"""Minimal, policy-agnostic runtime contracts for frozen world-action models.

The package deliberately contains no scheduler or value head.  It makes the
state and accounting boundaries used by the modular WAM experiments explicit:
prediction, fresh observation, feedback assimilation, execution, and cost.
"""

from .accounting import ComputeLedger, ComputeRecord
from .assimilation_backend import AssimilationBackend, InterfaceCandidate, RepairPlan
from .execution_state import ExecutionState
from .fixed_policy_harness import ORIGINAL_COSMOS_CHECKPOINT_SHA256, FixedPolicyContract
from .observation_provider import ObservationPacket
from .predictive_state import PredictiveState

__all__ = [
    "AssimilationBackend",
    "ComputeLedger",
    "ComputeRecord",
    "ExecutionState",
    "FixedPolicyContract",
    "InterfaceCandidate",
    "ObservationPacket",
    "ORIGINAL_COSMOS_CHECKPOINT_SHA256",
    "PredictiveState",
    "RepairPlan",
]

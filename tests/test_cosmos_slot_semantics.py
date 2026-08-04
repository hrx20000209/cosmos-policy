import numpy as np
import pytest

from adapters.cosmos_adapter import COSMOS_LIBERO_SLOT_SEMANTICS, CosmosAdapter
from runtime.observation_buffer import Observation


def make_observation(timestamp: int) -> Observation:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return Observation(timestamp, image, image, np.zeros(9, dtype=np.float32))


def test_libero_slots_are_modalities_not_history():
    assert COSMOS_LIBERO_SLOT_SEMANTICS[2] == "current_wrist_image"
    assert COSMOS_LIBERO_SLOT_SEMANTICS[3] == "current_primary_image"
    assert COSMOS_LIBERO_SLOT_SEMANTICS[4] == "action_chunk"
    with pytest.raises(ValueError, match="must not replace"):
        CosmosAdapter.validate_history([make_observation(1), make_observation(2)])


import numpy as np
import pytest

from runtime.keyframe_selector import KeyframeSelector
from runtime.observation_buffer import Observation, ObservationBuffer


def obs(timestamp: int, value: int = 0) -> Observation:
    image = np.full((8, 8, 3), value, dtype=np.uint8)
    return Observation(timestamp, image, image, np.array([value], dtype=np.float32))


def test_observation_buffer_is_ordered_and_bounded():
    buffer = ObservationBuffer(2)
    buffer.append(obs(1))
    buffer.append(obs(2))
    buffer.append(obs(3))
    assert [item.timestamp_ns for item in buffer.snapshot()] == [2, 3]
    assert buffer.discarded_count == 1
    with pytest.raises(ValueError):
        buffer.append(obs(3))


def test_sparse_history_reports_time_span():
    selector = KeyframeSelector()
    selected, stats = selector.select_history([obs(i * 1_000_000_000) for i in range(8)], 3, 2, "uniform_sparse")
    assert [item.timestamp_ns // 1_000_000_000 for item in selected] == [3, 5, 7]
    assert stats.history_span_seconds == 4.0
    assert stats.average_frame_interval == 2.0


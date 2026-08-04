import numpy as np

from runtime.action_buffer import ActionBuffer


def test_prefix_keeps_first_actions_not_tail():
    buffer = ActionBuffer()
    chunk = np.arange(40, dtype=np.float32).reshape(10, 4)
    assert buffer.install("r1", chunk, prefix_length=3, latest_valid_request_id="r1")
    assert np.array_equal(buffer.pop(), chunk[0])
    assert np.array_equal(buffer.pop(), chunk[1])
    assert np.array_equal(buffer.pop(), chunk[2])


def test_stale_request_is_rejected():
    buffer = ActionBuffer()
    chunk = np.zeros((2, 7), dtype=np.float32)
    assert not buffer.install("old", chunk, prefix_length=1, latest_valid_request_id="new")
    assert buffer.stats.stale_inference_count == 1
    assert buffer.occupancy == 0


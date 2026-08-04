import numpy as np

from adapters.mock_adapter import MockAdapter
from runtime.async_pipeline import InferenceRequest
from runtime.observation_buffer import Observation


def test_adapter_reports_exact_requested_denoiser_calls():
    adapter = MockAdapter({"action_horizon": 4})
    adapter.reset("task", 0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    observation = Observation(1, image, image, np.zeros(9, dtype=np.float32))
    request = InferenceRequest.create(1, "episode", 0)
    output = adapter.infer(observation, request, 1)
    assert output.denoiser_forward_count == 1


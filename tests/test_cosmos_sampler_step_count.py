import pytest
import torch

from cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler


@pytest.mark.parametrize("requested_steps", [1, 2, 3, 4, 5, 8])
def test_cosmos_sampler_performs_exact_requested_denoiser_forwards(requested_steps):
    calls = 0

    def denoiser(noisy, sigma):
        nonlocal calls
        calls += 1
        return torch.zeros_like(noisy)

    sampler = CosmosPolicySampler()
    result = sampler(
        denoiser,
        torch.ones(1, 2),
        num_steps=requested_steps,
        sigma_min=4.0,
        sigma_max=80.0,
    )

    assert result.shape == (1, 2)
    assert calls == requested_steps


def test_cosmos_sampler_rejects_non_positive_step_count():
    with pytest.raises(ValueError, match="at least 1"):
        CosmosPolicySampler()(lambda noisy, sigma: noisy, torch.ones(1, 2), num_steps=0)

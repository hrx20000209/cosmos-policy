"""Unit tests for the progressive-denoising checkpoint instrumentation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence
from cosmos_policy.modules.cosmos_sampler import CosmosPolicySampler
from experiments.progressive_wam import metrics
from experiments.progressive_wam.checkpoint import CheckpointRecorder
from experiments.progressive_wam.cosmos_hook import CosmosCheckpointCapture
from experiments.progressive_wam.task_stage import StageThresholds, label_episode


LIBERO_SIGMA_MIN = 4.0
LIBERO_SIGMA_MAX = 80.0
STATE_SHAPE = (1, 16, 9, 4, 4)


def _linear_denoiser(scale: float = 0.5):
    """A cheap stand-in for the DiT: contracts toward a fixed target."""
    target = torch.full(STATE_SHAPE, 0.25, dtype=torch.float64)

    def fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        s = sigma.reshape(-1, *([1] * (x.dim() - 1))).to(x.dtype)
        return target + scale * (x - target) / (1.0 + s)

    return fn


def _run_sampler(num_steps: int, hook=None):
    sampler = CosmosPolicySampler()
    if hook is not None:
        sampler.checkpoint_hook = hook
    x_sigma_max = torch.randn(STATE_SHAPE, dtype=torch.float64) * LIBERO_SIGMA_MAX
    out = sampler.forward(
        _linear_denoiser(),
        x_sigma_max,
        num_steps=num_steps,
        sigma_min=LIBERO_SIGMA_MIN,
        sigma_max=LIBERO_SIGMA_MAX,
    )
    return out


@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 5, 8])
def test_checkpoint_count_equals_denoiser_forwards(num_steps: int) -> None:
    """Every public step must produce exactly one checkpoint, in every branch."""
    seen: list[dict] = []
    _run_sampler(num_steps, hook=lambda **kw: seen.append(kw))
    assert len(seen) == num_steps


@pytest.mark.parametrize("num_steps", [1, 2, 3, 5, 8])
def test_hook_does_not_change_output(num_steps: int) -> None:
    torch.manual_seed(0)
    without = _run_sampler(num_steps)
    torch.manual_seed(0)
    with_hook = _run_sampler(num_steps, hook=lambda **kw: None)
    assert torch.equal(without, with_hook)


def test_checkpoint_sigmas_are_the_official_libero_grid() -> None:
    seen: list[dict] = []
    _run_sampler(5, hook=lambda **kw: seen.append(kw))
    sigmas = [float(kw["sigma_cur_0"]) for kw in seen]
    # 5-step LIBERO grid, see reports/progressive_denoising_audit.md section 0.
    expected = [80.0, 42.2911, 20.9724, 9.6183, 4.0]
    assert sigmas == pytest.approx(expected, abs=1e-3)
    assert [kw.get("is_terminal_clean", False) for kw in seen] == [False, False, False, False, True]


def test_first_checkpoint_matches_standalone_one_step() -> None:
    """The j=1 predicted-clean of any schedule equals standalone_one_step output.

    This is the EDM grid endpoint property the audit relies on; if it ever breaks,
    the P1 comparison between 'configuration' and 'checkpoint' is invalid.
    """
    torch.manual_seed(7)
    x_sigma_max = torch.randn(STATE_SHAPE, dtype=torch.float64) * LIBERO_SIGMA_MAX
    outputs = {}
    for num_steps in (1, 2, 3, 5, 8):
        seen: list[dict] = []
        sampler = CosmosPolicySampler()
        sampler.checkpoint_hook = lambda **kw: seen.append(kw)
        sampler.forward(
            _linear_denoiser(),
            x_sigma_max.clone(),
            num_steps=num_steps,
            sigma_min=LIBERO_SIGMA_MIN,
            sigma_max=LIBERO_SIGMA_MAX,
        )
        outputs[num_steps] = seen[0]["x0_pred_B_StateShape"]
    reference = outputs[1]
    for num_steps, first in outputs.items():
        assert torch.allclose(first, reference, atol=1e-12), f"{num_steps}-step j=1 differs from standalone_1"


def test_extract_action_matches_official_path() -> None:
    """The hook's GPU-side extraction must equal the official extraction helper."""
    capture = CosmosCheckpointCapture(
        model=SimpleNamespace(sampler=SimpleNamespace()),
        cfg=SimpleNamespace(chunk_size=16, action_dim=7),
        dataset_stats={},
        capture_future_latent=False,
    )
    torch.manual_seed(3)
    latent = torch.randn(1, 16, 9, 28, 28, dtype=torch.float64)
    official = extract_action_chunk_from_latent_sequence(
        latent, action_shape=(16, 7), action_indices=torch.tensor([4])
    )
    ours = capture._extract_action(latent)
    assert torch.allclose(ours.to(torch.float64), official.squeeze(0).to(torch.float64), atol=1e-6)


def test_eps_parameterization_is_consistent_with_x0() -> None:
    """EDM identity used in the audit: x0 = x_t - sigma * eps_pred."""
    torch.manual_seed(11)
    xt = torch.randn(2, 4, dtype=torch.float64)
    x0 = torch.randn(2, 4, dtype=torch.float64)
    sigma = torch.tensor([[3.0], [7.0]], dtype=torch.float64)
    eps = (xt - x0) / sigma
    assert torch.allclose(xt - sigma * eps, x0, atol=1e-12)


def test_flow_matching_predicted_clean_formula() -> None:
    """LingBot: x0 = x_sigma - sigma * v, and step(to_final=True) computes it."""
    scheduler_path = Path("/home/rxhuang/Projects/lingbot-va/wan_va/utils/scheduler.py")
    if not scheduler_path.is_file():
        pytest.skip("LingBot-VA repository not available")
    # Loaded by path: importing `wan_va.utils` as a package drags in the
    # websocket server, which this environment does not have and does not need.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_lingbot_scheduler", scheduler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    FlowMatchScheduler = module.FlowMatchScheduler

    scheduler = FlowMatchScheduler(num_inference_steps=50, shift=0.05)
    torch.manual_seed(5)
    x0 = torch.randn(1, 7, 4, 4, 1, dtype=torch.float32)
    noise = torch.randn_like(x0)
    for step_index in (0, 7, 25, 49):
        t = scheduler.timesteps[step_index : step_index + 1]
        sigma = scheduler.sigmas[step_index]
        x_sigma = (1 - sigma) * x0 + sigma * noise
        v = noise - x0  # exactly scheduler.training_target
        manual = x_sigma - sigma * v
        via_api = scheduler.step(v, t, x_sigma, to_final=True)
        assert torch.allclose(manual, x0, atol=1e-5)
        assert torch.allclose(via_api, manual, atol=1e-6)


def test_recorder_requires_open_request() -> None:
    recorder = CheckpointRecorder("cosmos", capture_cuda_events=False)
    with pytest.raises(RuntimeError):
        recorder.record(denoise_stage="joint", sigma=1.0, predicted_clean_action=torch.zeros(16, 7))
    recorder.begin_request("r0", 0.0)
    recorder.record(denoise_stage="joint", sigma=80.0, predicted_clean_action=torch.zeros(16, 7))
    with pytest.raises(RuntimeError):
        recorder.begin_request("r1", 0.0)
    out = recorder.finish_request()
    assert len(out) == 1 and out[0].denoiser_forward_count == 1


def test_hook_rejects_unexpected_callback_signature() -> None:
    capture = CosmosCheckpointCapture(
        model=SimpleNamespace(sampler=SimpleNamespace()),
        cfg=SimpleNamespace(chunk_size=16, action_dim=7),
        dataset_stats={},
    )
    capture.recorder.capture_cuda_events = False
    capture.recorder.begin_request("r0", 0.0)
    with pytest.raises(RuntimeError, match="missing expected keys"):
        capture._hook(i_th=0)


def test_horizon_weighted_error_favours_near_term() -> None:
    scale = np.ones(7)
    final = np.zeros((16, 7))
    near_bad = final.copy()
    near_bad[0] = 1.0
    far_bad = final.copy()
    far_bad[15] = 1.0
    e_near = metrics.horizon_weighted_error(near_bad, final, scale, horizon=16, decay=0.5)
    e_far = metrics.horizon_weighted_error(far_bad, final, scale, horizon=16, decay=0.5)
    assert e_near > e_far
    uniform_near = metrics.horizon_weighted_error(near_bad, final, scale, horizon=16, decay=1.0)
    uniform_far = metrics.horizon_weighted_error(far_bad, final, scale, horizon=16, decay=1.0)
    assert uniform_near == pytest.approx(uniform_far)


def test_metric_bundle_is_exact_on_identical_chunks() -> None:
    scale = np.ones(7)
    torch.manual_seed(1)
    chunk = np.random.RandomState(0).randn(16, 7)
    out = metrics.checkpoint_metrics(chunk, chunk, scale)
    assert out["normalized_l1"] == pytest.approx(0.0)
    assert out["cosine"] == pytest.approx(1.0)
    assert out["sign_agreement"] == pytest.approx(1.0)
    assert out["gripper_agreement"] == pytest.approx(1.0)


def test_endpoint_error_uses_cumulative_delta() -> None:
    final = np.zeros((16, 7))
    inter = np.zeros((16, 7))
    inter[:4, 0] = 0.1  # four steps of +0.1 in x
    pose = metrics.endpoint_pose_error(inter, final, horizon=4)
    assert pose["endpoint_translation_error"] == pytest.approx(0.4)
    pose_h1 = metrics.endpoint_pose_error(inter, final, horizon=1)
    assert pose_h1["endpoint_translation_error"] == pytest.approx(0.1)


def test_stage_labels_cover_expected_phases() -> None:
    steps = 40
    proprio = np.zeros((steps, 9))
    actions = np.zeros((steps, 7))
    # Phase 1: fast free-space motion with the gripper open.
    proprio[:15, 0] = 0.04
    proprio[:15, 1] = 0.04
    proprio[:15, 2] = np.linspace(0, 0.3, 15)
    actions[:15, 6] = -1.0
    # Phase 2: gripper closes.
    proprio[15:25, 0] = np.linspace(0.04, 0.01, 10)
    proprio[15:25, 1] = np.linspace(0.04, 0.01, 10)
    proprio[15:25, 2] = 0.3
    actions[15:25, 6] = 1.0
    # Phase 3: transport while holding.
    proprio[25:, 0] = 0.01
    proprio[25:, 1] = 0.01
    proprio[25:, 2] = np.linspace(0.3, 0.6, steps - 25)
    actions[25:, 6] = 1.0

    labels = label_episode(proprio, actions, StageThresholds())
    assert len(labels) == steps
    assert "gripper_transition" in labels
    assert "free_space_motion" in labels
    assert "transport" in labels

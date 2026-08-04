"""Unified denoising-checkpoint instrumentation for Cosmos Policy and LingBot-VA.

The recorder is deliberately allocation-light and synchronisation-free on the hot
path: every checkpoint keeps small GPU-resident tensors plus a CUDA event, and a
single host transfer happens when the request finishes.  Doing a ``.cpu()`` per
checkpoint would serialise the denoiser loop and destroy the very latency numbers
this experiment is meant to measure.

See ``reports/progressive_denoising_audit.md`` for the parameterisation audit that
justifies how ``predicted_clean_action`` is obtained for each model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


DENOISE_STAGE_ACTION = "action"
DENOISE_STAGE_VIDEO = "video"
DENOISE_STAGE_JOINT = "joint"


@dataclass
class DenoisingCheckpoint:
    """One full denoiser forward, expressed in model-independent terms.

    ``predicted_clean_action`` is always the *predicted-clean* action, never a
    noisy solver state.  ``noisy_state`` carries the solver state that produced
    it, and is ``None`` unless capture was explicitly requested.
    """

    request_id: str
    model_name: str
    denoise_stage: str
    video_step: Optional[int]
    action_step: Optional[int]
    sigma: float
    noisy_state: Optional[torch.Tensor]
    predicted_clean_action: torch.Tensor
    predicted_future_latent: Optional[torch.Tensor]
    value_prediction: Optional[torch.Tensor]
    observation_timestamp: float
    checkpoint_timestamp: float
    denoiser_forward_count: int
    is_terminal_clean: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_cpu(self) -> "DenoisingCheckpoint":
        def move(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if value is None:
                return None
            return value.detach().to("cpu", copy=True)

        return DenoisingCheckpoint(
            request_id=self.request_id,
            model_name=self.model_name,
            denoise_stage=self.denoise_stage,
            video_step=self.video_step,
            action_step=self.action_step,
            sigma=self.sigma,
            noisy_state=move(self.noisy_state),
            predicted_clean_action=move(self.predicted_clean_action),
            predicted_future_latent=move(self.predicted_future_latent),
            value_prediction=move(self.value_prediction),
            observation_timestamp=self.observation_timestamp,
            checkpoint_timestamp=self.checkpoint_timestamp,
            denoiser_forward_count=self.denoiser_forward_count,
            is_terminal_clean=self.is_terminal_clean,
            extra=dict(self.extra),
        )


class CheckpointRecorder:
    """Collects checkpoints for one request without synchronising the GPU.

    Usage::

        recorder = CheckpointRecorder(model_name="cosmos")
        recorder.begin_request("req-0", observation_timestamp=t0)
        ...                                  # hooks call recorder.record(...)
        checkpoints = recorder.finish_request()   # single sync + host copy
    """

    def __init__(self, model_name: str, capture_cuda_events: bool = True) -> None:
        self.model_name = model_name
        self.capture_cuda_events = capture_cuda_events and torch.cuda.is_available()
        self._request_id: Optional[str] = None
        self._observation_timestamp: float = 0.0
        self._pending: list[DenoisingCheckpoint] = []
        self._events: list[Any] = []
        self._start_event: Any = None
        self._forward_count: int = 0

    @property
    def active(self) -> bool:
        return self._request_id is not None

    @property
    def forward_count(self) -> int:
        return self._forward_count

    def begin_request(self, request_id: str, observation_timestamp: float) -> None:
        if self._request_id is not None:
            raise RuntimeError(
                f"CheckpointRecorder already has an open request {self._request_id!r}; "
                "finish_request() must be called before starting another one"
            )
        self._request_id = request_id
        self._observation_timestamp = observation_timestamp
        self._pending = []
        self._events = []
        self._forward_count = 0
        if self.capture_cuda_events:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_event = None

    def record(
        self,
        *,
        denoise_stage: str,
        sigma: float,
        predicted_clean_action: torch.Tensor,
        video_step: Optional[int] = None,
        action_step: Optional[int] = None,
        noisy_state: Optional[torch.Tensor] = None,
        predicted_future_latent: Optional[torch.Tensor] = None,
        value_prediction: Optional[torch.Tensor] = None,
        is_terminal_clean: bool = False,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._request_id is None:
            raise RuntimeError("CheckpointRecorder.record() called outside an open request")
        self._forward_count += 1
        if self.capture_cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self._events.append(event)
        else:
            self._events.append(None)
        self._pending.append(
            DenoisingCheckpoint(
                request_id=self._request_id,
                model_name=self.model_name,
                denoise_stage=denoise_stage,
                video_step=video_step,
                action_step=action_step,
                sigma=float(sigma),
                noisy_state=noisy_state,
                predicted_clean_action=predicted_clean_action,
                predicted_future_latent=predicted_future_latent,
                value_prediction=value_prediction,
                observation_timestamp=self._observation_timestamp,
                # Host-side enqueue time. It is NOT the kernel completion time;
                # the CUDA-event elapsed value written in finish_request() is.
                checkpoint_timestamp=time.perf_counter(),
                denoiser_forward_count=self._forward_count,
                is_terminal_clean=is_terminal_clean,
                extra=dict(extra or {}),
            )
        )

    def finish_request(self) -> list[DenoisingCheckpoint]:
        if self._request_id is None:
            raise RuntimeError("CheckpointRecorder.finish_request() called without an open request")
        if self.capture_cuda_events and self._events:
            torch.cuda.synchronize()
            for checkpoint, event in zip(self._pending, self._events):
                if event is not None and self._start_event is not None:
                    checkpoint.extra["gpu_elapsed_ms_from_request_start"] = float(
                        self._start_event.elapsed_time(event)
                    )
        results = [checkpoint.to_cpu() for checkpoint in self._pending]
        self._request_id = None
        self._pending = []
        self._events = []
        self._start_event = None
        return results

    def abort_request(self) -> None:
        self._request_id = None
        self._pending = []
        self._events = []
        self._start_event = None
        self._forward_count = 0

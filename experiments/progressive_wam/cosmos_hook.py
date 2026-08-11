"""Cosmos Policy checkpoint capture.

Cosmos uses EDM x0-parameterisation, so ``x0_fn`` already returns the
predicted-clean latent at every solver step (``policy_video2world_model.py:456``,
``self.scaling = EDMScaling(sigma_data)``).  No scheduler maths is re-implemented
here: the checkpoint action is exactly what the official extraction path would
produce if the sampler had stopped at that step.

The hook is attached as ``model.sampler.checkpoint_hook``; ``CosmosPolicySampler``
short-circuits on ``None`` so an uninstalled hook leaves the sampler byte-for-byte
on its original control flow.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import torch

from .checkpoint import DENOISE_STAGE_JOINT, CheckpointRecorder


# LIBERO with proprio + 1 wrist + 1 third-person camera. Verified against the
# `latent_indices` dict that get_action() returns; see `assert_indices_match`.
LIBERO_LATENT_INDICES = {
    "current_proprio_latent_idx": 1,
    "current_wrist_image_latent_idx": 2,
    "current_image_latent_idx": 3,
    "action_latent_idx": 4,
    "future_proprio_latent_idx": 5,
    "future_wrist_image_latent_idx": 6,
    "future_image_latent_idx": 7,
    "value_latent_idx": 8,
}

FUTURE_SLOT_KEYS = (
    "future_proprio_latent_idx",
    "future_wrist_image_latent_idx",
    "future_image_latent_idx",
)


class CosmosCheckpointCapture:
    """Installs a per-denoiser-forward hook on a Cosmos policy model."""

    def __init__(
        self,
        model: Any,
        cfg: Any,
        dataset_stats: dict,
        *,
        latent_indices: Optional[dict[str, int]] = None,
        capture_future_latent: bool = True,
        capture_noisy_state: bool = False,
        capture_value: bool = False,
        capture_compact_hidden: bool = False,
        unnormalize: bool = True,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.dataset_stats = dataset_stats
        self.latent_indices = dict(latent_indices or LIBERO_LATENT_INDICES)
        self.capture_future_latent = capture_future_latent
        self.capture_noisy_state = capture_noisy_state
        # Value is structurally present in Cosmos Policy, but mechanism/runtime
        # experiments must not read it unless an explicitly separate planning
        # study opts in.  The overnight WAM study keeps this False.
        self.capture_value = capture_value
        self.capture_compact_hidden = capture_compact_hidden
        self.unnormalize = unnormalize
        self.recorder = CheckpointRecorder(model_name="cosmos")
        self._installed = False
        self._chunk_size = int(getattr(cfg, "chunk_size", 16))
        self._action_dim = int(getattr(cfg, "action_dim", 7))

    # ------------------------------------------------------------------ setup

    def install(self) -> None:
        if self._installed:
            return
        sampler = self.model.sampler
        if getattr(sampler, "checkpoint_hook", None) is not None:
            raise RuntimeError("a checkpoint_hook is already installed on this sampler")
        sampler.checkpoint_hook = self._hook
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self.model.sampler.checkpoint_hook = None
        self._installed = False

    @contextmanager
    def request(self, request_id: str, observation_timestamp: float):
        """Scopes one policy call; yields the list of checkpoints on exit."""
        self.recorder.begin_request(request_id, observation_timestamp)
        holder: list[Any] = []
        try:
            yield holder
        except BaseException:
            self.recorder.abort_request()
            raise
        holder.extend(self.recorder.finish_request())

    def assert_indices_match(self, reported: dict[str, Any]) -> None:
        """Fail fast if the model's real slot layout differs from our assumption."""
        for key, expected in self.latent_indices.items():
            if key not in reported:
                continue
            actual = int(reported[key])
            if actual != expected:
                raise RuntimeError(
                    f"latent slot mismatch for {key}: capture assumed {expected}, model reported {actual}"
                )

    # ------------------------------------------------------------------- hook

    def _hook(self, **kwargs: Any) -> None:
        # `differential_equation_solver` calls `callback_fn(**locals())`, so the
        # key names below are contractual with res_sampler.py. Missing keys are a
        # real upstream change and must not be silently tolerated.
        missing = [key for key in ("i_th", "sigma_cur_0", "x0_pred_B_StateShape") if key not in kwargs]
        if missing:
            raise RuntimeError(
                f"sampler callback is missing expected keys {missing}; "
                "res_sampler.differential_equation_solver changed its locals"
            )
        x0_pred = kwargs["x0_pred_B_StateShape"]
        sigma = kwargs["sigma_cur_0"]
        step_index = int(kwargs["i_th"])
        is_terminal = bool(kwargs.get("is_terminal_clean", False))

        action = self._extract_action(x0_pred)
        value = self._extract_value(x0_pred) if self.capture_value else None
        future = self._extract_future(x0_pred) if self.capture_future_latent else None
        compact_hidden = self._extract_compact_hidden() if self.capture_compact_hidden else None
        noisy = None
        if self.capture_noisy_state:
            state = kwargs.get("input_x_B_StateShape")
            if state is not None:
                noisy = state.detach().to(torch.float16)

        self.recorder.record(
            denoise_stage=DENOISE_STAGE_JOINT,
            sigma=float(sigma),
            predicted_clean_action=action,
            action_step=self.recorder.forward_count + 1,
            video_step=self.recorder.forward_count + 1,
            noisy_state=noisy,
            predicted_future_latent=future,
            value_prediction=value,
            compact_hidden=compact_hidden,
            is_terminal_clean=is_terminal,
            extra={"solver_index": step_index},
        )

    # -------------------------------------------------------------- extraction

    def _extract_action(self, latent: torch.Tensor) -> torch.Tensor:
        """Predicted-clean action chunk, normalised layout ``(chunk, action_dim)``.

        Mirrors ``extract_action_chunk_from_latent_sequence`` exactly (mean over
        the repeated chunk copies packed into the action latent frame) but keeps
        the result on the GPU so the hot path never synchronises.
        """
        idx = self.latent_indices["action_latent_idx"]
        frame = latent[:, :, idx, :, :]  # (B, C, H, W)
        batch = frame.shape[0]
        flat = frame.reshape(batch, -1)
        num_elements = self._chunk_size * self._action_dim
        num_copies = flat.shape[1] // num_elements
        if num_copies < 1:
            raise ValueError(
                f"action latent has {flat.shape[1]} elements, too few for a "
                f"({self._chunk_size}, {self._action_dim}) action chunk"
            )
        chunks = flat[:, : num_copies * num_elements].reshape(
            batch, num_copies, self._chunk_size, self._action_dim
        )
        return chunks.mean(dim=1).to(torch.float32).squeeze(0)

    def _extract_value(self, latent: torch.Tensor) -> Optional[torch.Tensor]:
        idx = self.latent_indices.get("value_latent_idx")
        if idx is None or idx < 0:
            return None
        frame = latent[:, :, idx, :, :]
        return frame.reshape(frame.shape[0], -1).mean(dim=1).to(torch.float32)

    def _extract_future(self, latent: torch.Tensor) -> Optional[torch.Tensor]:
        slots = [self.latent_indices[key] for key in FUTURE_SLOT_KEYS if self.latent_indices.get(key, -1) >= 0]
        if not slots:
            return None
        # Stored as fp16: these are only used for relative distance metrics in P7,
        # and fp32 would double the dump size for no measurable benefit.
        return latent[:, :, slots, :, :].to(torch.float16).squeeze(0)

    def _extract_compact_hidden(self) -> Optional[torch.Tensor]:
        """Capture the current denoiser forward's reduced block features.

        The model callback runs immediately after ``x0_fn`` returns, while
        ``last_intermediate_features`` still belongs to that exact denoising
        stage.  Full token maps are deliberately rejected here.
        """
        features = self.model.last_intermediate_features
        if not features:
            return None
        if any(feature.ndim != 3 for feature in features):
            shapes = [tuple(feature.shape) for feature in features]
            raise ValueError(f"compact hidden capture requires [B,T,D] features, got {shapes}")
        stacked = torch.stack(features, dim=1)
        if stacked.shape[0] != 1:
            raise ValueError(f"compact hidden capture expects batch size one, got {stacked.shape[0]}")
        return stacked.squeeze(0).detach().to(torch.float16)

    # ------------------------------------------------------------ postprocess

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Applies the official min-max inverse transform used by ``get_action``."""
        if not self.unnormalize:
            return np.asarray(action, dtype=np.float32)
        from cosmos_policy.experiments.robot.cosmos_utils import unnormalize_actions

        arr = np.asarray(action, dtype=np.float32)
        squeezed = arr.reshape(1, *arr.shape) if arr.ndim == 2 else arr
        out = unnormalize_actions(squeezed, self.dataset_stats)
        return np.asarray(out, dtype=np.float32).reshape(arr.shape)

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from adapters.base import PolicyOutput
from runtime.async_pipeline import InferenceRequest
from runtime.observation_buffer import Observation


class LingBotAdapter:
    model_name = "lingbot_va"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.repo = Path(config.get("repo", "/home/rxhuang/Projects/lingbot-va")).resolve()
        self.action_horizon = int(config.get("action_horizon", 16))
        self.server = None
        self._initial = True
        self._last_native_action: np.ndarray | None = None

    @staticmethod
    def _native_observation(item: Observation) -> dict[str, np.ndarray]:
        return {
            "observation.images.agentview_rgb": item.primary_image,
            "observation.images.eye_in_hand_rgb": item.wrist_image,
        }

    def _ensure_loaded(self) -> None:
        if self.server is not None:
            return
        checkpoint = self.config.get("checkpoint")
        if not checkpoint:
            raise RuntimeError("LingBot adapter requires a LIBERO checkpoint path")
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"LingBot LIBERO checkpoint does not exist: {checkpoint_path}. "
                "Do not substitute the base, RoboTwin, or SO101 checkpoint."
            )
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        wan_va_root = self.repo / "wan_va"
        if str(wan_va_root) not in sys.path:
            sys.path.insert(0, str(wan_va_root))
        from wan_va.configs import VA_CONFIGS
        from wan_va.wan_va_server import VA_Server

        config_name = self.config.get("config_name", "libero")
        job_config = VA_CONFIGS[config_name]
        job_config.wan22_pretrained_model_name_or_path = checkpoint
        if self.config.get("transformer_path"):
            job_config.transformer_path = self.config["transformer_path"]
        job_config.enable_offload = bool(self.config.get("enable_offload", False))
        self.server = VA_Server(job_config)
        import torch

        self.decode_stream = torch.cuda.Stream(priority=0)
        native_horizon = int(job_config.frame_chunk_size * job_config.action_per_frame)
        if native_horizon != self.action_horizon:
            raise ValueError(f"checkpoint action horizon is {native_horizon}, configured {self.action_horizon}")

    def reset(self, task_description: str, seed: int) -> None:
        self._ensure_loaded()
        self.server.infer({"reset": True, "prompt": task_description})
        self._initial = True
        self._last_native_action = None

    def infer(
        self,
        observation: Observation,
        request: InferenceRequest,
        denoising_steps: int,
        history: list[Observation] | None = None,
    ) -> PolicyOutput:
        # Baseline server only reads obs inside _infer for frame_st_id==0.
        # Later real observations enter through update_context/KV-cache update.
        explicit = history or [observation]
        payload = {
            "obs": [self._native_observation(item) for item in explicit],
            "video_inference_steps": int(self.config.get("video_inference_steps", 20)),
            "action_inference_steps": denoising_steps,
            "return_profile": True,
            "save_intermediates": False,
        }
        start = time.monotonic_ns()
        result = self.server.infer(payload)
        total_ms = (time.monotonic_ns() - start) / 1e6
        native = np.asarray(result["pred_action"], dtype=np.float32)  # [7, latent_frames, actions_per_frame]
        self._last_native_action = native
        actions = native.transpose(1, 2, 0).reshape(-1, native.shape[0])
        was_initial = self._initial
        if was_initial:
            # The first latent action frame is the zero conditioning
            # placeholder in the official client, not an executable action.
            actions = actions[int(self.server.job_config.action_per_frame) :]
        timing = result.get("server_timing", {})
        video_calls = int(timing.get("video_transformer_calls", payload["video_inference_steps"] + 1))
        action_calls = int(timing.get("action_transformer_calls", denoising_steps + 1))
        metrics = {
            "total_ms": total_ms,
            "vae_encoding_ms": float(timing.get("obs_encode_s", 0.0)) * 1000,
            "video_denoising_ms": float(timing.get("video_loop_s", 0.0)) * 1000,
            "dit_denoising_ms": (
                float(timing.get("video_loop_s", 0.0)) + float(timing.get("action_loop_s", 0.0))
            )
            * 1000,
            "action_extraction_ms": float(timing.get("postprocess_s", 0.0)) * 1000,
        }
        self._initial = False
        return PolicyOutput(
            actions=actions,
            denoiser_forward_count=video_calls + action_calls,
            generated_latent=result.get("pred_latent"),
            stage_metrics_ms=metrics,
            extra={
                "video_transformer_calls": video_calls,
                "action_transformer_calls": action_calls,
                "initial_placeholder_actions_dropped": int(self.server.job_config.action_per_frame) if was_initial else 0,
            },
        )

    def update_context(self, observations: list[Observation], predicted_actions: np.ndarray) -> dict[str, Any]:
        if self.server is None or self._last_native_action is None or not observations:
            return {}
        action_per_frame = int(self.server.job_config.action_per_frame)
        executed = np.asarray(predicted_actions, dtype=np.float32)
        if executed.ndim == 2 and executed.shape[1] == self._last_native_action.shape[0]:
            remainder = len(executed) % action_per_frame
            if remainder:
                executed = np.concatenate(
                    [executed, np.repeat(executed[-1:], action_per_frame - remainder, axis=0)],
                    axis=0,
                )
            native_action = executed.reshape(-1, action_per_frame, executed.shape[1]).transpose(2, 0, 1)
        else:
            native_action = self._last_native_action
        payload = {
            "obs": [self._native_observation(item) for item in observations],
            "compute_kv_cache": True,
            "pred_action": native_action,
            "save_intermediates": False,
        }
        return self.server.infer(payload)

    def decode_future(self, output: PolicyOutput) -> Any:
        if output.generated_latent is None:
            return None
        import torch

        latent = torch.as_tensor(output.generated_latent, device=self.server.device, dtype=self.server.dtype)
        with torch.inference_mode():
            with torch.cuda.stream(self.decode_stream):
                return self.server.decode_one_video(latent, "np")

    def close(self) -> None:
        self.server = None

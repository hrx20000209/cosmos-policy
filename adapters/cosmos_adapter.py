from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from adapters.base import PolicyOutput
from runtime.async_pipeline import InferenceRequest
from runtime.observation_buffer import Observation


COSMOS_LIBERO_SLOT_SEMANTICS = {
    0: "temporal_vae_leading_placeholder",
    1: "current_proprio",
    2: "current_wrist_image",
    3: "current_primary_image",
    4: "action_chunk",
    5: "future_proprio",
    6: "future_wrist_image",
    7: "future_primary_image",
    8: "value",
}


class CosmosAdapter:
    """Low-intrusion wrapper around Cosmos Policy's existing LIBERO path."""

    model_name = "cosmos"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.action_horizon = int(config.get("action_horizon", 16))
        self.model = None
        self.dataset_stats = None
        self.cfg = None

    @staticmethod
    def validate_history(history: list[Observation] | None) -> None:
        if history is not None and len(history) > 1:
            raise ValueError(
                "Cosmos baseline consumes one complete current observation; "
                "historical frames must not replace wrist/primary/future latent slots"
            )

    def _ensure_loaded(self) -> None:
        if self.model is not None:
            return
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        required = ("checkpoint", "dataset_stats_path", "t5_embeddings_path")
        missing = [key for key in required if not self.config.get(key)]
        if missing:
            raise RuntimeError(f"Cosmos adapter missing required config: {', '.join(missing)}")
        defaults = dict(
            suite="libero",
            config="cosmos_predict2_2b_480p_libero__inference_only",
            ckpt_path=self.config["checkpoint"],
            config_file=self.config.get("config_file", "cosmos_policy/config/config.py"),
            use_third_person_image=True,
            num_third_person_images=1,
            use_wrist_image=True,
            num_wrist_images=1,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            use_variance_scale=False,
            use_jpeg_compression=True,
            trained_with_image_aug=True,
            chunk_size=self.action_horizon,
            action_dim=7,
        )
        defaults.update(self.config.get("native_config", {}))
        self.cfg = SimpleNamespace(**defaults)
        init_t5_text_embeddings_cache(self.config["t5_embeddings_path"])
        self.dataset_stats = load_dataset_stats(self.config["dataset_stats_path"])
        self.model, train_config = get_model(self.cfg)
        import torch

        self.decode_stream = torch.cuda.Stream(priority=0)
        train_horizon = int(train_config.dataloader_train.dataset.chunk_size)
        if train_horizon != self.action_horizon:
            raise ValueError(f"checkpoint action horizon is {train_horizon}, configured {self.action_horizon}")

    def reset(self, task_description: str, seed: int) -> None:
        self._ensure_loaded()
        self.task_description = task_description
        self.seed = seed

    def infer(
        self,
        observation: Observation,
        request: InferenceRequest,
        denoising_steps: int,
        history: list[Observation] | None = None,
    ) -> PolicyOutput:
        self.validate_history(history)
        from cosmos_policy.experiments.robot.cosmos_utils import get_action

        obs = {
            "primary_image": observation.primary_image,
            "wrist_image": observation.wrist_image,
            "proprio": observation.proprio,
        }
        metrics: dict[str, float] = {}
        self.cfg._inference_metrics_sink = metrics
        self.model.sampler.step_timing_events = []
        start = time.monotonic_ns()
        try:
            result = get_action(
                self.cfg,
                self.model,
                self.dataset_stats,
                obs,
                self.task_description,
                seed=self.seed,
                randomize_seed=bool(self.config.get("native_config", {}).get("randomize_seed", False)),
                num_denoising_steps_action=denoising_steps,
                generate_future_state_and_value_in_parallel=self.config.get("generation_mode", "joint_parallel")
                != "autoregressive_future",
                decode_future_state=False,
            )
            step_events = self.model.sampler.step_timing_events
            metrics["per_denoising_step_latency_ms"] = [
                float(start_event.elapsed_time(end_event)) for start_event, end_event in step_events
            ]
        finally:
            self.model.sampler.step_timing_events = None
        total_ms = (time.monotonic_ns() - start) / 1e6
        metrics.setdefault("total_ms", total_ms)
        metrics.setdefault("dit_denoising_ms", metrics.get("generation_wall_ms", 0.0))
        metrics.setdefault("action_extraction_ms", metrics.get("postprocess_ms", 0.0))
        actions = np.asarray(result["actions"], dtype=np.float32).reshape(self.action_horizon, -1)
        canonical_actions = np.ascontiguousarray(actions, dtype=np.float32)
        return PolicyOutput(
            actions=actions,
            denoiser_forward_count=denoising_steps,
            generated_latent=result.get("generated_latent"),
            future_state=result.get("future_image_predictions"),
            value=result.get("value_prediction"),
            stage_metrics_ms=metrics,
            extra={
                "action_chunk_sha256": hashlib.sha256(canonical_actions.tobytes()).hexdigest(),
                "action_chunk_min": float(canonical_actions.min()),
                "action_chunk_max": float(canonical_actions.max()),
                "action_chunk_mean": float(canonical_actions.mean()),
                "action_chunk_std": float(canonical_actions.std()),
                "action_chunk_l2_norm": float(np.linalg.norm(canonical_actions)),
                "action_chunk_nan_count": int(np.isnan(canonical_actions).sum()),
                "action_chunk_inf_count": int(np.isinf(canonical_actions).sum()),
                "action_chunk_saturation_ratio": float(
                    np.mean(np.abs(canonical_actions) >= 0.999)
                ),
                "latent_indices": result.get("latent_indices", {}),
                "orig_clean_latent_frames": result.get("orig_clean_latent_frames"),
                "data_batch": result.get("data_batch"),
            },
        )

    def update_context(self, observations: list[Observation], predicted_actions: np.ndarray) -> dict[str, Any]:
        return {}

    def decode_future(self, output: PolicyOutput) -> Any:
        if output.generated_latent is None:
            return None
        if self.config.get("generation_mode", "joint_parallel") == "autoregressive_future":
            from cosmos_policy.experiments.robot.cosmos_utils import get_future_state_prediction
            import torch

            indices = output.extra["latent_indices"]
            with torch.inference_mode():
                result = get_future_state_prediction(
                    self.cfg,
                    self.model,
                    output.extra["data_batch"],
                    output.generated_latent,
                    output.extra["orig_clean_latent_frames"],
                    indices["future_proprio_latent_idx"],
                    indices["future_wrist_image_latent_idx"],
                    indices["future_wrist_image2_latent_idx"],
                    indices["future_image_latent_idx"],
                    indices["future_image2_latent_idx"],
                    seed=self.seed,
                    num_denoising_steps_future_state=int(self.config.get("future_state_steps", 1)),
                )
            return result["future_image_predictions"]
        from cosmos_policy.experiments.robot.cosmos_utils import get_future_images_from_generated_samples
        import torch

        indices = output.extra["latent_indices"]
        replacements = [0, 1, 4, 5]
        with torch.inference_mode():
            with torch.cuda.stream(self.decode_stream):
                return get_future_images_from_generated_samples(
                    self.model,
                    output.generated_latent.clone(),
                    self.cfg,
                    output.extra["orig_clean_latent_frames"],
                    replacements,
                    indices["future_wrist_image_latent_idx"],
                    indices["future_wrist_image2_latent_idx"],
                    indices["future_image_latent_idx"],
                    indices["future_image2_latent_idx"],
                    temporal_compression_factor=4,
                )

    def close(self) -> None:
        self.model = None

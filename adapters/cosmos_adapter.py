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
        self.closed_loop_mode = str(config.get("closed_loop_mode", "fresh")).lower()
        if self.closed_loop_mode not in {
            "fresh",
            "alternate_speculative",
            "alternate_cache",
            "predict_correct",
            "predict_correct_async",
            "predicted_reuse",
            "native_persistent",
            "pv0_r0",
            "pv0_r1",
            "pv0_r2",
            "pv0_r3",
            "stale_r2",
        }:
            raise ValueError(f"unknown closed_loop_mode={self.closed_loop_mode!r}")
        self.request_index = 0
        self.previous_real_latent = None
        self.previous_generated_latent = None
        self.last_physical_condition_latent = None
        self._shadow_prior_generated_latent = None

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
        self.request_index = 0
        self.previous_real_latent = None
        self.previous_generated_latent = None
        self.last_physical_condition_latent = None
        self._shadow_prior_generated_latent = None

    @staticmethod
    def _predicted_visual_latent(latent):
        """Move predicted LIBERO future visual content into current slots.

        The slot positions remain current positions 2/3.  Only latent content
        is copied from future wrist/primary slots 6/7; positional encodings and
        the native conditional mask are still created by the Cosmos path.
        """
        import torch

        if latent is None or not isinstance(latent, torch.Tensor):
            raise RuntimeError("predicted visual latent is unavailable")
        if latent.ndim != 5 or latent.shape[2] != 9:
            raise ValueError(f"expected LIBERO latent [B,C,9,H,W], got {tuple(latent.shape)}")
        result = latent.detach().clone()
        result[:, :, 2] = result[:, :, 6]
        result[:, :, 3] = result[:, :, 7]
        return result

    def _visual_input_for_request(self):
        fixed_patterns = {
            "pv0_r0": ("native_persistent",),
            "pv0_r1": ("native_persistent", "predicted"),
            "pv0_r2": ("native_persistent", "predicted", "predicted"),
            "pv0_r3": ("native_persistent", "predicted", "predicted", "predicted"),
        }
        if self.closed_loop_mode in fixed_patterns:
            if self.request_index == 0:
                return "fresh", None
            if self.previous_generated_latent is None:
                raise RuntimeError("fixed PV0/P1 route has no previous generated latent")
            pattern = fixed_patterns[self.closed_loop_mode]
            visual_mode = pattern[(self.request_index - 1) % len(pattern)]
            return visual_mode, self._predicted_visual_latent(self.previous_generated_latent)
        if self.closed_loop_mode == "stale_r2":
            if self.request_index == 0:
                return "fresh", None
            if self.previous_generated_latent is None:
                raise RuntimeError("stale R2 route has no prior generated latent")
            # R2 cadence is PV0, STALE, STALE.  The stale route deliberately
            # retains the most recent *physical* PV0-conditioned joint state;
            # unlike P1, it never moves generated future slots into current
            # slots.  Thus sensing and request cadence match R2 exactly.
            phase = (self.request_index - 1) % 3
            if phase == 0:
                return "native_persistent", self._predicted_visual_latent(self.previous_generated_latent)
            if self.last_physical_condition_latent is None:
                raise RuntimeError("stale R2 requested before a physical PV0 condition was captured")
            return "stale_physical", self.last_physical_condition_latent.detach().clone()
        if self.closed_loop_mode == "fresh":
            return "fresh", None
        if self.closed_loop_mode in {"predict_correct", "predict_correct_async"}:
            if self.request_index == 0:
                return "fresh", None
            if self.previous_generated_latent is None:
                raise RuntimeError("predict-correct request has no previous generated latent")
            return "predict_correct", self._predicted_visual_latent(self.previous_generated_latent)
        if self.closed_loop_mode == "predicted_reuse":
            if self.request_index == 0:
                return "fresh", None
            if self.previous_generated_latent is None:
                raise RuntimeError("predicted-reuse request has no previous generated latent")
            return "predicted", self._predicted_visual_latent(self.previous_generated_latent)
        if self.closed_loop_mode == "native_persistent":
            if self.request_index == 0:
                return "fresh", None
            if self.previous_generated_latent is None:
                raise RuntimeError("native-persistent request has no previous generated latent")
            # Keep the predicted latent as the base condition, then ask the
            # native Cosmos input path to assimilate a causal fresh visual
            # prefix before its sole denoiser forward.  This is intentionally
            # distinct from predict_correct, whose two-forward correction is
            # not compatible with the one-denoise PV0 contract.
            return "native_persistent", self._predicted_visual_latent(self.previous_generated_latent)
        if self.request_index % 2 == 0:
            return "fresh", None
        if self.closed_loop_mode == "alternate_speculative":
            if self.previous_generated_latent is None:
                raise RuntimeError("alternate speculative request has no previous generated latent")
            return "predicted", self._predicted_visual_latent(self.previous_generated_latent)
        if self.previous_real_latent is None:
            raise RuntimeError("alternate cache request has no previous real visual latent")
        return "cache", self.previous_real_latent.detach().clone()

    def infer(
        self,
        observation: Observation,
        request: InferenceRequest,
        denoising_steps: int,
        history: list[Observation] | None = None,
    ) -> PolicyOutput:
        self.validate_history(history)
        from cosmos_policy.experiments.robot.cosmos_utils import get_action

        visual_input_mode, reused_latent = self._visual_input_for_request()
        effective_denoising_steps = (
            int(self.config.get("predict_correct_steps", 2))
            if visual_input_mode == "predict_correct"
            else denoising_steps
        )
        if visual_input_mode == "predict_correct" and effective_denoising_steps < 2:
            raise ValueError("predict-correct requires at least two denoiser forwards")
        obs = {
            "primary_image": observation.primary_image,
            "wrist_image": observation.wrist_image,
            "proprio": observation.proprio,
        }
        metrics: dict[str, float] = {}
        self.cfg._inference_metrics_sink = metrics
        self.model.sampler.step_timing_events = []
        original_encode = None
        async_runtime = self.closed_loop_mode == "predict_correct_async" and visual_input_mode == "predict_correct"
        if visual_input_mode in {"fresh", "predict_correct", "native_persistent"} and not async_runtime:
            import torch

            original_encode = self.model.encode

            def timed_encode(state):
                torch.cuda.synchronize()
                encode_start = time.monotonic_ns()
                output = original_encode(state)
                torch.cuda.synchronize()
                metrics["vae_encoding_ms"] = (time.monotonic_ns() - encode_start) / 1e6
                return output

            self.model.encode = timed_encode
        else:
            metrics["vae_encoding_ms"] = 0.0
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
                num_denoising_steps_action=effective_denoising_steps,
                generate_future_state_and_value_in_parallel=False,
                decode_future_state=False,
                skip_vae_encoding=reused_latent is not None,
                previous_generated_latent=reused_latent,
                skip_camera_preprocessing=(
                    reused_latent is not None
                    and visual_input_mode not in {"predict_correct", "native_persistent"}
                ),
                persistent_visual_correction_prefix_frames=(
                    13 if visual_input_mode in {"predict_correct", "native_persistent"} else None
                ),
                persistent_visual_correction_arrival=(
                    0 if visual_input_mode == "native_persistent" else 1
                ),
                async_predict_correct=async_runtime,
                async_visual_arrival_delay_ms=float(self.config.get("async_visual_arrival_delay_ms", 0.0)),
            )
            if async_runtime and "async_gpu_timeline" in metrics:
                metrics["per_denoising_step_latency_ms"] = [
                    float(item["duration_ms"])
                    for item in metrics["async_gpu_timeline"]
                    if item["stage"].startswith("dit_forward_")
                ]
                metrics["vae_encoding_ms"] = float(metrics.get("async_prefix_gpu_ms", 0.0))
            else:
                step_events = self.model.sampler.step_timing_events
                metrics["per_denoising_step_latency_ms"] = [
                    float(start_event.elapsed_time(end_event)) for start_event, end_event in step_events
                ]
        finally:
            if original_encode is not None and "encode" in self.model.__dict__:
                del self.model.__dict__["encode"]
            self.model.sampler.step_timing_events = None
        total_ms = (time.monotonic_ns() - start) / 1e6
        metrics.setdefault("total_ms", total_ms)
        metrics["model_generate_inclusive_ms"] = float(metrics.get("model_generate_inclusive_ms", 0.0))
        metrics["dit_denoising_ms"] = float(sum(metrics.get("per_denoising_step_latency_ms", [])))
        metrics["generation_conditioning_overhead_ms"] = max(
            metrics["model_generate_inclusive_ms"]
            - float(metrics.get("vae_encoding_ms", 0.0))
            - metrics["dit_denoising_ms"],
            0.0,
        )
        metrics.setdefault("action_extraction_ms", metrics.get("postprocess_ms", 0.0))
        metrics.setdefault("camera_preprocessing_ms", 0.0)
        metrics.setdefault("latent_assembly_h2d_ms", 0.0)
        metrics.setdefault("postprocess_after_action_ms", 0.0)
        stage_total_ms = sum(
            float(metrics.get(key, 0.0))
            for key in (
                "camera_preprocessing_ms",
                "latent_assembly_h2d_ms",
                "vae_encoding_ms",
                "dit_denoising_ms",
                "generation_conditioning_overhead_ms",
                "action_extraction_ms",
                "postprocess_after_action_ms",
            )
        )
        metrics["non_overlapping_stage_sum_ms"] = stage_total_ms
        metrics["unattributed_stage_ms"] = metrics["total_ms"] - stage_total_ms
        actions = np.asarray(result["actions"], dtype=np.float32).reshape(self.action_horizon, -1)
        canonical_actions = np.ascontiguousarray(actions, dtype=np.float32)
        cosmos_request_index = self.request_index
        # Preserve the speculative prior for an optional *post-control* shadow
        # label.  Do this before the main request advances its generated state.
        self._shadow_prior_generated_latent = (
            self.previous_generated_latent.detach().clone()
            if self.previous_generated_latent is not None
            else None
        )
        if visual_input_mode == "fresh":
            self.previous_real_latent = result["orig_clean_latent_frames"].detach().clone()
        if visual_input_mode == "native_persistent":
            physical_condition = result.get("persistent_condition_latent")
            if physical_condition is None:
                raise RuntimeError("native PV0 request did not return its physical condition latent")
            self.last_physical_condition_latent = physical_condition.detach().clone()
        self.previous_generated_latent = result["generated_latent"].detach().clone()
        self.request_index += 1
        return PolicyOutput(
            actions=actions,
            denoiser_forward_count=effective_denoising_steps,
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
                "visual_input_mode": visual_input_mode,
                "visual_source": visual_input_mode,
                "cosmos_request_index": cosmos_request_index,
                "fresh_visual_request_count": int(visual_input_mode == "fresh"),
                "fresh_sensing_request_count": int(
                    visual_input_mode in {"fresh", "predict_correct", "native_persistent"}
                ),
                "predicted_visual_request_count": int(visual_input_mode == "predicted"),
                "predict_correct_request_count": int(visual_input_mode == "predict_correct"),
                "native_persistent_request_count": int(visual_input_mode == "native_persistent"),
                "stale_physical_request_count": int(visual_input_mode == "stale_physical"),
                "cached_visual_request_count": int(visual_input_mode == "cache"),
                "rgb_preprocessing_count": int(
                    visual_input_mode in {"fresh", "predict_correct", "native_persistent"}
                ),
            },
        )

    def infer_shadow_validity_labels(self, observation: Observation) -> dict[str, Any] | None:
        """Compute retrospective F1/P1/PV0 labels without affecting control.

        This is deliberately callable only *after* the main route selected and
        installed its action chunk.  Its outputs are offline labels used for
        task-disjoint feature discovery; they are never returned to
        ``_visual_input_for_request`` or used to select a route.
        """
        if self.previous_generated_latent is None:
            return None
        import torch

        from cosmos_policy.experiments.robot.cosmos_utils import get_action

        # ``previous_generated_latent`` was replaced by the just-completed
        # main request.  The relevant speculative prior for this observation
        # is saved before that replacement in ``infer`` below.
        prior = getattr(self, "_shadow_prior_generated_latent", None)
        if prior is None:
            return None
        obs = {
            "primary_image": observation.primary_image,
            "wrist_image": observation.wrist_image,
            "proprio": observation.proprio,
        }

        def action_for(mode: str) -> np.ndarray:
            if mode == "F1":
                kwargs = {
                    "skip_vae_encoding": False,
                    "previous_generated_latent": None,
                    "skip_camera_preprocessing": False,
                    "persistent_visual_correction_prefix_frames": None,
                }
            elif mode == "P1":
                kwargs = {
                    "skip_vae_encoding": True,
                    "previous_generated_latent": self._predicted_visual_latent(prior),
                    "skip_camera_preprocessing": True,
                    "persistent_visual_correction_prefix_frames": None,
                }
            elif mode == "PV0":
                kwargs = {
                    "skip_vae_encoding": True,
                    "previous_generated_latent": self._predicted_visual_latent(prior),
                    "skip_camera_preprocessing": False,
                    "persistent_visual_correction_prefix_frames": 13,
                }
            else:
                raise ValueError(mode)
            result = get_action(
                self.cfg,
                self.model,
                self.dataset_stats,
                obs,
                self.task_description,
                seed=self.seed,
                randomize_seed=False,
                num_denoising_steps_action=1,
                generate_future_state_and_value_in_parallel=False,
                decode_future_state=False,
                persistent_visual_correction_arrival=0 if mode == "PV0" else 1,
                async_predict_correct=False,
                async_visual_arrival_delay_ms=0.0,
                **kwargs,
            )
            return np.asarray(result["actions"], dtype=np.float32).reshape(self.action_horizon, -1)

        with torch.inference_mode():
            actions = {name: action_for(name) for name in ("F1", "P1", "PV0")}

        def rmse(left: np.ndarray, right: np.ndarray) -> float:
            return float(np.linalg.norm(left - right) / np.sqrt(left.size))
        p1_risk = rmse(actions["P1"], actions["F1"])
        pv0_residual = rmse(actions["PV0"], actions["F1"])
        return {
            "shadow_only": True,
            "denoising_steps_per_shadow_route": 1,
            "value_used": False,
            "actions": {name: value.tolist() for name, value in actions.items()},
            "p1_to_f1_rmse": p1_risk,
            "pv0_to_f1_rmse": pv0_residual,
            "pv0_correction_gain": (p1_risk - pv0_residual) / p1_risk if p1_risk > 1e-12 else 0.0,
        }

    def update_context(self, observations: list[Observation], predicted_actions: np.ndarray) -> dict[str, Any]:
        return {}

    def decode_future(self, output: PolicyOutput) -> Any:
        if output.generated_latent is None:
            return None
        if self.config.get("generation_mode", "joint_parallel") == "autoregressive_future":
            import torch

            from cosmos_policy.experiments.robot.cosmos_utils import get_future_state_prediction

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
        import torch

        from cosmos_policy.experiments.robot.cosmos_utils import get_future_images_from_generated_samples

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

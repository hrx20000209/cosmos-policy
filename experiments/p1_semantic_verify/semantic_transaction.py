"""Transactional speculative-P1 adapter for closed-loop semantic verify-and-correct.

The transaction is *speculate-without-mutating, commit-on-accept*.  The
speculative P1 forward writes nothing durable — not adapter cache, not RNG, not
model hooks — so the reject path is exact by construction rather than by a
restore that could drift.  See ``reports/p1_semantic_verify/TRANSACTION_CONTRACT.json``
and the source audit in ``reports/p1_semantic_verify/AUDIT.md``.

Route semantics are inherited unchanged from ``CosmosAdapter``:
bootstrap request 0 is F1; accepted decisions are exactly ``predicted_reuse``;
rejected decisions are exactly ``native_persistent`` at the same physical
observation.  No K != 16 commitment, no new scheduler family, no threshold
learned at runtime.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from adapters.base import PolicyOutput
from adapters.cosmos_adapter import CosmosAdapter
from experiments.p1_semantic_verify.common import BLOCKS, frozen_score, load_frozen_model, semantic_reducer


class SemanticVerifyAdapter(CosmosAdapter):
    """P1-native semantic verify-and-correct with a frozen decision threshold.

    ``verify_mode``:
      ``semantic``      -- accept when the frozen 89-feature P1 score <= threshold
      ``action_only``   -- accept when the frozen-family action-geometry ridge
                           score <= threshold (matched-budget baseline)
      ``periodic``      -- accept unless the decision index hits a fixed period
                           (matched-budget baseline; no score is read)
    """

    def __init__(self, config: dict[str, Any], *, verify_mode: str = "semantic",
                 threshold: float | None = None, period: int | None = None,
                 action_only_model: dict[str, Any] | None = None):
        super().__init__({**config, "closed_loop_mode": "predicted_reuse"})
        if verify_mode not in {"semantic", "action_only", "periodic"}:
            raise ValueError(f"unknown verify_mode={verify_mode!r}")
        if verify_mode in {"semantic", "action_only"} and threshold is None:
            raise ValueError("semantic/action_only verify requires a frozen threshold")
        if verify_mode == "periodic" and (period is None or period < 1):
            raise ValueError("periodic verify requires a positive frozen period")
        if verify_mode == "action_only" and action_only_model is None:
            raise ValueError("action_only verify requires the frozen discovery ridge model")
        self.verify_mode = verify_mode
        self.threshold = float(threshold) if threshold is not None else None
        self.period = int(period) if period is not None else None
        self.action_only_model = action_only_model
        self.frozen = load_frozen_model()
        self.decisions: list[dict[str, Any]] = []
        self.decision_index = 0

    # ------------------------------------------------------------------ #
    def reset(self, task_description: str, seed: int) -> None:
        super().reset(task_description, seed)
        self.decisions = []
        self.decision_index = 0

    # ------------------------------------------------------------------ #
    def _score(self, internal: dict[str, float], actions: np.ndarray) -> float:
        if self.verify_mode == "action_only":
            from experiments.p1_semantic_verify.common import action_geometry

            model = self.action_only_model
            values = action_geometry(actions)
            x = np.asarray([values[name] for name in model["feature_names"]], dtype=np.float64)
            mean, scale, coef = (
                np.asarray(model[key], dtype=np.float64) for key in ("feature_mean", "feature_scale", "coefficients")
            )
            return float(np.expm1(((x - mean) / scale) @ coef + float(model["intercept"])))
        return frozen_score(self.frozen, internal, actions)

    def _speculative_p1(self, observation: Any) -> dict[str, Any]:
        """One P1 forward with frozen feature capture and zero durable effect."""

        from cosmos_policy.experiments.robot.cosmos_utils import get_action

        prior = self._predicted_visual_latent(self.previous_generated_latent)
        capture = self.verify_mode == "semantic"
        self.model.intermediate_feature_ids = [block - 1 for block in BLOCKS] if capture else None
        self.model.intermediate_feature_reducer = semantic_reducer if capture else None
        started = time.monotonic_ns()
        try:
            result = get_action(
                self.cfg, self.model, self.dataset_stats,
                {
                    "primary_image": observation.primary_image,
                    "wrist_image": observation.wrist_image,
                    "proprio": observation.proprio,
                },
                self.task_description, seed=self.seed, randomize_seed=False,
                num_denoising_steps_action=1,
                generate_future_state_and_value_in_parallel=False, decode_future_state=False,
                skip_vae_encoding=True, previous_generated_latent=prior,
                skip_camera_preprocessing=True,
                persistent_visual_correction_prefix_frames=None,
                persistent_visual_correction_arrival=1,
                async_predict_correct=False,
            )
            internal: dict[str, float] = {}
            for block, tensor in zip(BLOCKS, self.model.last_intermediate_features or []):
                for index, value in enumerate(tensor[0].detach().cpu().tolist()):
                    internal[f"internal_b{block}_{index}"] = float(value)
        finally:
            self.model.intermediate_feature_ids = None
            self.model.intermediate_feature_reducer = None
            self.model.last_intermediate_features = None
        return {
            "result": result,
            "internal": internal,
            "actions": np.asarray(result["actions"], dtype=np.float32).reshape(self.action_horizon, -1),
            "latency_ms": float((time.monotonic_ns() - started) / 1e6),
        }

    # ------------------------------------------------------------------ #
    def infer(self, observation: Any, request: Any, denoising_steps: int,
              history: list[Any] | None = None) -> PolicyOutput:
        # Bootstrap and any request without a legal prior fall through to the
        # inherited route (request 0 is F1 by CosmosAdapter's own contract).
        if self.request_index == 0 or self.previous_generated_latent is None:
            output = super().infer(observation, request, denoising_steps, history)
            self.decisions.append({
                "decision_idx": self.decision_index, "control_step": int(request.control_step_id),
                "mode": "bootstrap_f1", "s_p1": None, "threshold": self.threshold,
                "decision": "bootstrap", "accepted_p1": False, "corrected_pv0": False,
                "speculative_calls": 0, "discarded_speculative_calls": 0,
                "speculative_latency_ms": 0.0, "correction_latency_ms": 0.0,
                "model_latency_ms": float(output.stage_metrics_ms.get("total_ms", 0.0)),
            })
            self.decision_index += 1
            return output

        decision_started = time.monotonic_ns()
        if self.verify_mode == "periodic":
            speculative = None
            score = None
            accept = (self.decision_index % self.period) != 0
        else:
            speculative = self._speculative_p1(observation)
            score = self._score(speculative["internal"], speculative["actions"])
            accept = score <= self.threshold

        speculative_latency = float(speculative["latency_ms"]) if speculative is not None else 0.0

        if accept and speculative is not None:
            # ---- COMMIT the speculative P1 exactly as predicted_reuse would ----
            result = speculative["result"]
            actions = speculative["actions"]
            self._shadow_prior_generated_latent = self.previous_generated_latent.detach().clone()
            self.previous_generated_latent = result["generated_latent"].detach().clone()
            self.request_index += 1
            correction_latency = 0.0
            visual_mode = "predicted"
        else:
            # ---- REJECT (nothing was written) or periodic correction ----
            correction_started = time.monotonic_ns()
            saved_mode = self.closed_loop_mode
            self.closed_loop_mode = "native_persistent"
            try:
                output = super().infer(observation, request, denoising_steps, history)
            finally:
                self.closed_loop_mode = saved_mode
            correction_latency = float((time.monotonic_ns() - correction_started) / 1e6)
            self.decisions.append({
                "decision_idx": self.decision_index, "control_step": int(request.control_step_id),
                "mode": self.verify_mode, "s_p1": score, "threshold": self.threshold,
                "decision": "correct", "accepted_p1": False, "corrected_pv0": True,
                "speculative_calls": int(speculative is not None),
                "discarded_speculative_calls": int(speculative is not None),
                "speculative_latency_ms": speculative_latency,
                "correction_latency_ms": correction_latency,
                "decision_latency_ms": float((time.monotonic_ns() - decision_started) / 1e6),
                "model_latency_ms": float(output.stage_metrics_ms.get("total_ms", 0.0)),
            })
            self.decision_index += 1
            output.extra["semantic_decision"] = "correct"
            output.extra["semantic_score"] = score
            output.extra["speculative_discarded"] = bool(speculative is not None)
            return output

        canonical = np.ascontiguousarray(actions, dtype=np.float32)
        metrics = {
            "total_ms": speculative_latency,
            "model_generate_inclusive_ms": speculative_latency,
            "vae_encoding_ms": 0.0,
        }
        self.decisions.append({
            "decision_idx": self.decision_index, "control_step": int(request.control_step_id),
            "mode": self.verify_mode, "s_p1": score, "threshold": self.threshold,
            "decision": "accept", "accepted_p1": True, "corrected_pv0": False,
            "speculative_calls": 1, "discarded_speculative_calls": 0,
            "speculative_latency_ms": speculative_latency, "correction_latency_ms": 0.0,
            "decision_latency_ms": float((time.monotonic_ns() - decision_started) / 1e6),
            "model_latency_ms": speculative_latency,
        })
        self.decision_index += 1
        return PolicyOutput(
            actions=actions,
            denoiser_forward_count=1,
            generated_latent=result.get("generated_latent"),
            future_state=result.get("future_image_predictions"),
            value=None,
            stage_metrics_ms=metrics,
            extra={
                "action_chunk_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
                "action_chunk_nan_count": int(np.isnan(canonical).sum()),
                "action_chunk_inf_count": int(np.isinf(canonical).sum()),
                "latent_indices": result.get("latent_indices", {}),
                "orig_clean_latent_frames": result.get("orig_clean_latent_frames"),
                "data_batch": result.get("data_batch"),
                "visual_input_mode": visual_mode,
                "visual_source": visual_mode,
                "cosmos_request_index": self.request_index - 1,
                "fresh_visual_request_count": 0,
                "fresh_sensing_request_count": 0,
                "predicted_visual_request_count": 1,
                "native_persistent_request_count": 0,
                "rgb_preprocessing_count": 0,
                "semantic_decision": "accept",
                "semantic_score": score,
                "speculative_discarded": False,
            },
        )

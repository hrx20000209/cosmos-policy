"""Shared primitives for E12 P1-native semantic verify-and-correct.

Three route primitives (F1 / P1 / PV0) under the frozen one-denoise contract,
plus the frozen E4 semantic score read out of an ordinary P1 forward.

Nothing here refits, reselects, or re-normalizes the frozen score.  The feature
reducer is byte-identical in behaviour to
``experiments/semantic_risk/run_semantic_shard.py::run_route``'s reducer; SMOKE-B
asserts that numerically rather than trusting the copy.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiments.semantic_risk.run_semantic_shard import BLOCKS, SLOTS, action_geometry
from experiments.server_deep_validation.pv0_overnight_common import (
    PREFIX_FRAMES,
    pair_metrics,
    route_contract,
)

FROZEN_MODEL_PATH = Path("reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json")
FROZEN_CHECKSUM = "3299a5ec2308f990ff68e6f80f2d28ca922b4aef8c72c2f708da139bd6745d4b"
ACTION_GEOMETRY_FEATURES = (
    "action_norm",
    "endpoint_displacement",
    "action_curvature",
    "action_jerk",
    "gripper_transition",
)
ROUTES = ("P1", "F1", "PV0")


# --------------------------------------------------------------------------- #
# frozen score
# --------------------------------------------------------------------------- #
def load_frozen_model(path: Path = FROZEN_MODEL_PATH) -> dict[str, Any]:
    """Load and self-verify the frozen E4 INTERNAL_PLUS_ACTION ridge score."""

    model = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in model.items() if key != "checksum_sha256"}
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    if actual != model["checksum_sha256"] or actual != FROZEN_CHECKSUM:
        raise RuntimeError(f"frozen sensitivity model checksum mismatch: {actual}")
    if len(model["feature_names"]) != 89:
        raise RuntimeError(f"expected 89 frozen features, got {len(model['feature_names'])}")
    return model


def frozen_score(frozen: Mapping[str, Any], internal: Mapping[str, float], actions: np.ndarray) -> float:
    """Exact float64 evaluation of the frozen score. No approximation is allowed."""

    values = {**internal, **action_geometry(actions)}
    x = np.asarray([values[name] for name in frozen["feature_names"]], dtype=np.float64)
    mean, scale, coef = (
        np.asarray(frozen[key], dtype=np.float64) for key in ("feature_mean", "feature_scale", "coefficients")
    )
    return float(np.expm1(((x - mean) / scale) @ coef + float(frozen["intercept"])))


def frozen_score_from_row(frozen: Mapping[str, Any], row: Mapping[str, float]) -> float:
    x = np.asarray([row[name] for name in frozen["feature_names"]], dtype=np.float64)
    mean, scale, coef = (
        np.asarray(frozen[key], dtype=np.float64) for key in ("feature_mean", "feature_scale", "coefficients")
    )
    return float(np.expm1(((x - mean) / scale) @ coef + float(frozen["intercept"])))


# --------------------------------------------------------------------------- #
# route primitives
# --------------------------------------------------------------------------- #
def predicted_condition(previous_generated: torch.Tensor) -> torch.Tensor:
    """Move predicted future visual slots 6/7 into current slots 2/3."""

    if previous_generated.ndim != 5 or previous_generated.shape[2] != 9:
        raise ValueError(f"expected [B,C,9,H,W] latent, got {tuple(previous_generated.shape)}")
    result = previous_generated.detach().clone()
    result[:, :, 2] = result[:, :, 6]
    result[:, :, 3] = result[:, :, 7]
    return result


def semantic_reducer(hidden: torch.Tensor, _: int) -> torch.Tensor:
    """The frozen E4 passive post-block reducer: 12 scalars per selected block."""

    pools = {slot: hidden[:, slot].mean(dim=(1, 2)).float() for slot in SLOTS}

    def rms(slot: int) -> torch.Tensor:
        return torch.sqrt(hidden[:, slot].float().square().mean(dim=(1, 2, 3)))

    def disp(slot: int) -> torch.Tensor:
        return hidden[:, slot].float().std(dim=(1, 2, 3))

    def cosine(a: int, b: int) -> torch.Tensor:
        return torch.nn.functional.cosine_similarity(pools[a], pools[b], dim=-1)

    return torch.stack(
        [
            rms(4), disp(4), rms(6), rms(7), rms(2), rms(3),
            cosine(4, 6), cosine(4, 7), cosine(4, 2), cosine(4, 3), cosine(2, 6), cosine(3, 7),
        ],
        dim=-1,
    )


def run_route(
    route: str,
    *,
    cfg: Any,
    model: Any,
    stats: Mapping[str, Any],
    observation: Any,
    instruction: str,
    seed: int,
    previous_generated: torch.Tensor | None,
    capture_blocks: tuple[int, ...] = (),
) -> dict[str, Any]:
    """One legal route forward.

    ``previous_generated`` is the raw prior generated joint latent; the
    predicted-condition slot move is applied here so every reuse route sees the
    identical base condition.
    """

    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    contract = route_contract(route)
    if route == "F1":
        reused = None
    else:
        if previous_generated is None:
            raise RuntimeError(f"route {route} requires a prior generated latent")
        reused = predicted_condition(previous_generated)

    model.intermediate_feature_ids = [block - 1 for block in capture_blocks] or None
    model.intermediate_feature_reducer = semantic_reducer if capture_blocks else None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter_ns()
    start_event.record()
    try:
        result = get_action(
            cfg,
            model,
            stats,
            {
                "primary_image": observation.primary_image,
                "wrist_image": observation.wrist_image,
                "proprio": observation.proprio,
            },
            instruction,
            seed=int(seed),
            randomize_seed=False,
            num_denoising_steps_action=1,
            generate_future_state_and_value_in_parallel=False,
            decode_future_state=False,
            skip_vae_encoding=bool(contract["skip_vae_encoding"]),
            previous_generated_latent=reused,
            skip_camera_preprocessing=bool(contract["skip_camera_preprocessing"]),
            persistent_visual_correction_prefix_frames=contract.get("fresh_visual_prefix_frames"),
            persistent_visual_correction_arrival=int(contract.get("fresh_visual_arrival_denoiser_forward", 1)),
            async_predict_correct=False,
        )
        end_event.record()
        torch.cuda.synchronize()
        internal: dict[str, float] = {}
        for block, tensor in zip(capture_blocks, model.last_intermediate_features or []):
            for index, value in enumerate(tensor[0].detach().cpu().tolist()):
                internal[f"internal_b{block}_{index}"] = float(value)
    finally:
        model.intermediate_feature_ids = None
        model.intermediate_feature_reducer = None
        model.last_intermediate_features = None
        model.inference_condition_transform = None
        if hasattr(model, "sampler"):
            model.sampler.x0_transform = None

    return {
        "route": route,
        "actions": np.ascontiguousarray(np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)),
        "generated_latent": result["generated_latent"].detach().clone(),
        "orig_clean_latent_frames": result["orig_clean_latent_frames"].detach().clone(),
        "persistent_condition_latent": (
            result["persistent_condition_latent"].detach().clone()
            if result.get("persistent_condition_latent") is not None
            else None
        ),
        "internal": internal,
        "cuda_ms": float(start_event.elapsed_time(end_event)),
        "wall_ms": float((time.perf_counter_ns() - wall_start) / 1e6),
    }


# --------------------------------------------------------------------------- #
# hashing / risk helpers
# --------------------------------------------------------------------------- #
def observation_hash(observation: Any) -> str:
    digest = hashlib.sha256()
    for value in (observation.primary_image, observation.wrist_image, observation.proprio):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def risk(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    """Project-standard action discrepancy; primary field is ``mean_step_l2``."""

    metrics = pair_metrics(left, right)
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return {
        "full": float(metrics["mean_step_l2"]),
        "first": float(metrics["first_action_l2"]),
        "first4": float(metrics["prefixes"]["4"]["mean_step_l2"]),
        "per_joint_mean_abs": [float(v) for v in np.abs(delta).mean(axis=0)],
        "gripper_sign_disagreement": float(
            np.mean(np.sign(np.asarray(left)[:, 6]) != np.sign(np.asarray(right)[:, 6]))
        ),
    }


PREFIX = PREFIX_FRAMES
__all__ = [
    "ACTION_GEOMETRY_FEATURES", "BLOCKS", "PREFIX", "ROUTES", "action_geometry", "array_hash",
    "frozen_score", "frozen_score_from_row", "load_frozen_model", "observation_hash",
    "predicted_condition", "risk", "run_route", "semantic_reducer", "tensor_hash",
]

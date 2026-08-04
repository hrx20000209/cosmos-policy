"""Convergence / agreement metrics between an intermediate and a final action chunk.

All functions take actions in *physical* (un-normalised) units with shape
``(horizon, action_dim)`` unless stated otherwise.  For LIBERO the layout is
``[dx, dy, dz, drx, dry, drz, gripper]``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


LIBERO_TRANSLATION_DIMS = (0, 1, 2)
LIBERO_ROTATION_DIMS = (3, 4, 5)
LIBERO_GRIPPER_DIM = 6

EPS = 1e-8


def _as_chunk(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (horizon, action_dim) actions, got shape {arr.shape}")
    return arr


def per_dim_scale(dataset_stats: dict, action_dim: int) -> np.ndarray:
    """Per-dimension normaliser ``s_k`` for scale-free errors.

    Uses the dataset action std, which is the only scale in the checkpoint that
    reflects how much a dimension actually moves.  Falls back to 1.0 for degenerate
    dimensions so a near-constant channel cannot manufacture huge normalised error.
    """
    std = np.asarray(dataset_stats["actions_std"], dtype=np.float64)[:action_dim]
    return np.where(std > 1e-4, std, 1.0)


def normalized_l1(intermediate: np.ndarray, final: np.ndarray, scale: np.ndarray) -> float:
    a, b = _as_chunk(intermediate), _as_chunk(final)
    return float(np.mean(np.abs(a - b) / (scale[None, :] + EPS)))


def normalized_l2(intermediate: np.ndarray, final: np.ndarray, scale: np.ndarray) -> float:
    a, b = _as_chunk(intermediate), _as_chunk(final)
    return float(np.sqrt(np.mean(((a - b) / (scale[None, :] + EPS)) ** 2)))


def cosine_similarity(intermediate: np.ndarray, final: np.ndarray) -> float:
    a, b = _as_chunk(intermediate).ravel(), _as_chunk(final).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < EPS:
        return 1.0 if np.linalg.norm(a - b) < EPS else 0.0
    return float(np.dot(a, b) / denom)


def sign_agreement(intermediate: np.ndarray, final: np.ndarray, deadzone: float = 1e-3) -> float:
    """Fraction of entries whose sign matches; entries inside the deadzone on the
    reference are excluded rather than counted as agreement."""
    a, b = _as_chunk(intermediate), _as_chunk(final)
    live = np.abs(b) > deadzone
    if not live.any():
        return 1.0
    return float(np.mean(np.sign(a[live]) == np.sign(b[live])))


def gripper_state_agreement(
    intermediate: np.ndarray, final: np.ndarray, gripper_dim: int = LIBERO_GRIPPER_DIM
) -> float:
    """Agreement on the binarised open/close command (LIBERO gripper is ±1)."""
    a, b = _as_chunk(intermediate), _as_chunk(final)
    if a.shape[1] <= gripper_dim:
        return float("nan")
    return float(np.mean((a[:, gripper_dim] > 0) == (b[:, gripper_dim] > 0)))


def endpoint_pose_error(
    intermediate: np.ndarray,
    final: np.ndarray,
    horizon: Optional[int] = None,
    translation_dims: Sequence[int] = LIBERO_TRANSLATION_DIMS,
    rotation_dims: Sequence[int] = LIBERO_ROTATION_DIMS,
) -> dict[str, float]:
    """Error of the *cumulative* delta-pose over the first ``horizon`` actions.

    LIBERO actions are per-step deltas, so the endpoint of a prefix is their sum.
    This is the action-space proxy for "where does the arm end up"; the true
    simulator endpoint is measured in P2 instead.
    """
    a, b = _as_chunk(intermediate), _as_chunk(final)
    h = a.shape[0] if horizon is None else min(horizon, a.shape[0])
    da = a[:h].sum(axis=0)
    db = b[:h].sum(axis=0)
    trans = np.linalg.norm(da[list(translation_dims)] - db[list(translation_dims)])
    rot = np.linalg.norm(da[list(rotation_dims)] - db[list(rotation_dims)])
    return {"endpoint_translation_error": float(trans), "endpoint_rotation_error": float(rot)}


def derivative_stats(actions: np.ndarray, dims: Sequence[int] = LIBERO_TRANSLATION_DIMS) -> dict[str, float]:
    """Velocity / acceleration / jerk magnitude of a commanded action chunk.

    LIBERO actions are already per-step deltas (a velocity command), so the chunk
    itself is velocity, its first difference is acceleration and its second
    difference is jerk. Units are per control step, not per second.
    """
    arr = _as_chunk(actions)[:, list(dims)]
    velocity = np.linalg.norm(arr, axis=1)
    acceleration = np.linalg.norm(np.diff(arr, axis=0), axis=1) if arr.shape[0] > 1 else np.zeros(0)
    jerk = np.linalg.norm(np.diff(arr, n=2, axis=0), axis=1) if arr.shape[0] > 2 else np.zeros(0)
    return {
        "velocity_mean": float(velocity.mean()) if velocity.size else 0.0,
        "velocity_max": float(velocity.max()) if velocity.size else 0.0,
        "acceleration_mean": float(acceleration.mean()) if acceleration.size else 0.0,
        "acceleration_max": float(acceleration.max()) if acceleration.size else 0.0,
        "jerk_mean": float(jerk.mean()) if jerk.size else 0.0,
        "jerk_max": float(jerk.max()) if jerk.size else 0.0,
    }


def horizon_weights(horizon: int, decay: float = 0.5) -> np.ndarray:
    """Weights ``w_k`` that put more mass on near-term tokens.

    ``w_k = decay**k`` normalised to sum to 1.  ``decay=1.0`` recovers a uniform
    weighting, which is the control condition in the report.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    w = np.power(float(decay), np.arange(horizon, dtype=np.float64))
    return w / w.sum()


def horizon_weighted_error(
    intermediate: np.ndarray,
    final: np.ndarray,
    scale: np.ndarray,
    horizon: int,
    decay: float = 0.5,
) -> float:
    r"""E_{j,h} = sum_k w_k * ||a_k^{(j)} - a_k^{(N)}||_1 / (s_k + eps).

    ``w_k`` decays with k so near-term tokens dominate, matching the hypothesis
    that early commitment only needs the near-term prefix to be correct.
    """
    a, b = _as_chunk(intermediate), _as_chunk(final)
    h = min(horizon, a.shape[0], b.shape[0])
    w = horizon_weights(h, decay)
    per_token = np.sum(np.abs(a[:h] - b[:h]) / (scale[None, :] + EPS), axis=1)
    return float(np.dot(w, per_token))


def checkpoint_metrics(
    intermediate: np.ndarray,
    final: np.ndarray,
    scale: np.ndarray,
    horizons: Sequence[int] = (1, 2, 4, 8, 16),
    decay: float = 0.5,
) -> dict[str, float]:
    """Full metric bundle for one (checkpoint, chunk) pair."""
    out: dict[str, float] = {
        "normalized_l1": normalized_l1(intermediate, final, scale),
        "normalized_l2": normalized_l2(intermediate, final, scale),
        "cosine": cosine_similarity(intermediate, final),
        "sign_agreement": sign_agreement(intermediate, final),
        "gripper_agreement": gripper_state_agreement(intermediate, final),
    }
    out.update({f"final_{k}": v for k, v in derivative_stats(final).items()})
    out.update({f"intermediate_{k}": v for k, v in derivative_stats(intermediate).items()})
    for h in horizons:
        if h > intermediate.shape[0]:
            continue
        out[f"l1_h{h}"] = normalized_l1(intermediate[:h], final[:h], scale)
        out[f"l2_h{h}"] = normalized_l2(intermediate[:h], final[:h], scale)
        out[f"cosine_h{h}"] = cosine_similarity(intermediate[:h], final[:h])
        out[f"sign_h{h}"] = sign_agreement(intermediate[:h], final[:h])
        out[f"gripper_h{h}"] = gripper_state_agreement(intermediate[:h], final[:h])
        out[f"weighted_error_h{h}"] = horizon_weighted_error(intermediate, final, scale, h, decay)
        pose = endpoint_pose_error(intermediate, final, horizon=h)
        out[f"endpoint_translation_h{h}"] = pose["endpoint_translation_error"]
        out[f"endpoint_rotation_h{h}"] = pose["endpoint_rotation_error"]
    return out

"""Heuristic LIBERO task-stage labelling.

These labels are derived from proprio and the executed action only -- there is no
ground-truth phase annotation in LIBERO.  They are good enough to ask "does the
contact phase need more denoising than free-space motion", and must never be
reported as ground truth.

LIBERO proprio layout used by the Cosmos harness
(``experiments/libero_harness.py:extract_observation``)::

    [0:2]  robot0_gripper_qpos  (two finger joints)
    [2:5]  robot0_eef_pos
    [5:9]  robot0_eef_quat
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GRIPPER_QPOS = slice(0, 2)
EEF_POS = slice(2, 5)
EEF_QUAT = slice(5, 9)

STAGES = (
    "free_space_motion",
    "approach",
    "contact",
    "gripper_transition",
    "transport",
    "placement",
)


@dataclass
class StageThresholds:
    """All thresholds are configurable so the sensitivity analysis can sweep them."""

    fast_speed: float = 0.010  # eef translation per control step
    slow_speed: float = 0.004
    gripper_open_width: float = 0.055  # sum of the two finger joints
    gripper_closing_rate: float = 0.002  # |d width| per step marking an active transition
    transition_window: int = 3  # steps on either side of a gripper command flip


def gripper_width(proprio: np.ndarray) -> float:
    q = np.asarray(proprio, dtype=np.float64)[GRIPPER_QPOS]
    return float(np.abs(q).sum())


def label_episode(
    proprios: np.ndarray,
    actions: np.ndarray,
    thresholds: StageThresholds | None = None,
) -> list[str]:
    """Label every control step of one episode.

    Args:
        proprios: ``(T, 9)`` proprio observed *before* each executed action.
        actions: ``(T, 7)`` executed actions in physical units.
    """
    th = thresholds or StageThresholds()
    proprios = np.asarray(proprios, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    steps = proprios.shape[0]
    if steps == 0:
        return []

    eef = proprios[:, EEF_POS]
    speed = np.zeros(steps)
    if steps > 1:
        speed[1:] = np.linalg.norm(np.diff(eef, axis=0), axis=1)
        speed[0] = speed[1]

    width = np.array([gripper_width(p) for p in proprios])
    width_rate = np.zeros(steps)
    if steps > 1:
        width_rate[1:] = np.abs(np.diff(width))
        width_rate[0] = width_rate[1]

    gripper_cmd = actions[:, 6] if actions.shape[1] > 6 else np.zeros(steps)
    closed = width < th.gripper_open_width

    # A gripper transition is the neighbourhood of a command sign flip OR of a
    # rapid finger-width change; both matter because the command leads the motion.
    transition = np.zeros(steps, dtype=bool)
    flips = np.where(np.sign(gripper_cmd[1:]) != np.sign(gripper_cmd[:-1]))[0] + 1
    for idx in flips:
        lo = max(0, idx - th.transition_window)
        hi = min(steps, idx + th.transition_window + 1)
        transition[lo:hi] = True
    transition |= width_rate > th.gripper_closing_rate

    # "Has grasped" = the gripper has been commanded closed and stayed closed.
    holding = np.zeros(steps, dtype=bool)
    grasped = False
    for t in range(steps):
        if closed[t] and gripper_cmd[t] > 0:
            grasped = True
        elif not closed[t] and gripper_cmd[t] <= 0:
            grasped = False
        holding[t] = grasped

    labels: list[str] = []
    for t in range(steps):
        if transition[t]:
            labels.append("gripper_transition")
        elif holding[t]:
            labels.append("transport" if speed[t] >= th.fast_speed else "placement")
        elif speed[t] >= th.fast_speed:
            labels.append("free_space_motion")
        elif speed[t] <= th.slow_speed:
            labels.append("contact")
        else:
            labels.append("approach")
    return labels


def stage_of_step(labels: list[str], control_step: int) -> str:
    if not labels:
        return "unknown"
    return labels[min(control_step, len(labels) - 1)]

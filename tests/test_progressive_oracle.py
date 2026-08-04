"""Unit tests for the P2 oracle labelling and earliest-reliable-step logic.

These cover the parts that decide what counts as a feasibility opportunity, so
they must not depend on a simulator being available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.progressive_wam.analyze_p2 import NO_RELIABLE_STEP, earliest_reliable, relabel
from experiments.progressive_wam.run_p2_oracle import (
    OracleThresholds,
    gripper_transition_mismatch,
    oracle_label,
    quat_distance,
)


def _comparison(**overrides) -> dict[str, float]:
    base = {
        "eef_position_error": 0.001,
        "eef_rotation_error": 0.001,
        "object_position_error_max": 0.0,
        "gripper_width_error": 0.0,
        "joint_limit_violations": 0.0,
    }
    base.update(overrides)
    return base


def test_oracle_label_accepts_a_matching_branch() -> None:
    chunk = np.zeros((16, 7))
    label = oracle_label(_comparison(), chunk, 8, OracleThresholds(), gripper_mismatch=False)
    assert label["oracle_reliable"] is True
    assert label["safety_valid"] is True
    assert label["failure_reasons"] == []


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"eef_position_error": 0.5}, "eef_position"),
        ({"eef_rotation_error": 0.9}, "eef_rotation"),
        ({"object_position_error_max": 0.5}, "object_moved"),
        ({"gripper_width_error": 0.5}, "gripper_width"),
        ({"joint_limit_violations": 2.0}, "joint_limit"),
    ],
)
def test_oracle_label_reports_each_failure_reason(override: dict, reason: str) -> None:
    chunk = np.zeros((16, 7))
    label = oracle_label(_comparison(**override), chunk, 8, OracleThresholds(), gripper_mismatch=False)
    assert label["oracle_reliable"] is False
    assert reason in label["failure_reasons"]


def test_safety_reasons_are_separated_from_divergence_reasons() -> None:
    """A branch can diverge from the reference while still being physically safe."""
    chunk = np.zeros((16, 7))
    diverged = oracle_label(
        _comparison(eef_position_error=0.5), chunk, 8, OracleThresholds(), gripper_mismatch=False
    )
    assert diverged["oracle_reliable"] is False
    assert diverged["safety_valid"] is True

    violent = np.zeros((16, 7))
    violent[::2, 0] = 1.0  # alternating command -> large jerk
    unsafe = oracle_label(_comparison(), violent, 8, OracleThresholds(), gripper_mismatch=False)
    assert unsafe["safety_valid"] is False
    assert "jerk" in unsafe["failure_reasons"] or "velocity" in unsafe["failure_reasons"]


def test_gripper_mismatch_only_counts_within_the_prefix() -> None:
    intermediate = np.zeros((16, 7))
    final = np.zeros((16, 7))
    intermediate[:, 6] = -1.0
    final[:, 6] = -1.0
    final[10:, 6] = 1.0  # disagreement starts at token 10
    assert gripper_transition_mismatch(intermediate, final, prefix=8) is False
    assert gripper_transition_mismatch(intermediate, final, prefix=16) is True


def test_quat_distance_is_sign_invariant() -> None:
    q = np.array([0.0, 0.0, 0.0, 1.0])
    # The renormalisation inside quat_distance leaves ~1e-12 of float residue.
    assert quat_distance(q, q) == pytest.approx(0.0, abs=1e-9)
    assert quat_distance(q, -q) == pytest.approx(0.0, abs=1e-9)
    assert quat_distance(q, np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-9)


def _frame(reliable_by_checkpoint: dict[int, bool]) -> pd.DataFrame:
    rows = []
    for checkpoint, reliable in reliable_by_checkpoint.items():
        rows.append(
            {
                "request_id": "r0",
                "episode_id": "e0",
                "task_suite": "libero_10",
                "task_id": 0,
                "control_step": 0,
                "stage": "approach",
                "prefix_length": 4,
                "checkpoint": checkpoint,
                "oracle_reliable": reliable,
                # Non-zero even when reliable, so a threshold below it can flip
                # the label; a hard zero would pass any positive threshold.
                "eef_position_error": 0.001 if reliable else 1.0,
                "eef_rotation_error": 0.0,
                "object_position_error_max": 0.0,
                "gripper_width_error": 0.0,
                "joint_limit_violations": 0.0,
                "prefix_jerk_max": 0.0,
                "prefix_velocity_max": 0.0,
                "gripper_transition_mismatch": False,
            }
        )
    return pd.DataFrame(rows)


def test_earliest_reliable_picks_the_first_qualifying_checkpoint() -> None:
    frame = _frame({1: False, 2: False, 3: True, 4: True, 5: True})
    js = earliest_reliable(frame)
    assert len(js) == 1
    assert js.iloc[0]["j_star"] == 3


def test_earliest_reliable_marks_never_reliable_states_as_infinite() -> None:
    frame = _frame({1: False, 2: False, 3: False})
    js = earliest_reliable(frame)
    assert js.iloc[0]["j_star"] == NO_RELIABLE_STEP


def test_relabel_reproduces_stricter_and_looser_thresholds() -> None:
    frame = _frame({1: False, 2: True, 3: True})
    loose = relabel(frame, OracleThresholds(eef_position_error=10.0))
    assert loose.all()
    strict = relabel(frame, OracleThresholds(eef_position_error=1e-6))
    assert not strict.any()

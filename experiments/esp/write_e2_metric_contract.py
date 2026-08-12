#!/usr/bin/env python3
"""Write the preregistered primary E2 action-distance contract."""

from __future__ import annotations

from pathlib import Path

from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json


def main() -> None:
    atomic_write_json(Path("reports/esp/E2_METRIC_CONTRACT.json"), {
        "schema_version": 1,
        "frozen_before": "ESP_PILOT",
        "primary_scalar": "mean_step_l2",
        "definition": "mean over 16 action-chunk steps of the L2 distance across the native 7 action dimensions",
        "reference": "S_c^{causal,imag}=D(A_{-c:imag}, A_F1)",
        "secondary": ["full_rmse", "first_action_l2", "first4_mean_step_l2", "per_joint_mean_abs", "gripper_sign_disagreement"],
        "interventions": ["imagined", "stale_previous_real", "shuffle_task_disjoint"],
        "statistical_unit": "task (hierarchical task -> state bootstrap for formal analysis)",
        "selection_rule": "no post-hoc target or pooling change based on heldout results",
    })


if __name__ == "__main__":
    main()

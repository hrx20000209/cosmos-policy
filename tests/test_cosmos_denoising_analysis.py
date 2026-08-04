import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/cosmos_denoising_libero_pro/scripts/analyze_results.py"
)
SPEC = importlib.util.spec_from_file_location("cosmos_denoising_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ANALYSIS)

PREPARE_SPEC = importlib.util.spec_from_file_location(
    "cosmos_denoising_prepare_stage2",
    SCRIPT.parent / "prepare_stage2.py",
)
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
assert PREPARE_SPEC.loader is not None
PREPARE_SPEC.loader.exec_module(PREPARE)


def test_paired_tests_preserves_base_task_cluster_column(monkeypatch):
    episodes = pd.DataFrame(
        [
            {
                "domain": "libero",
                "perturbation_category": "none",
                "base_task_id": task,
                "task_uid": f"task-{task}",
                "variant_id": "canonical",
                "seed": 195,
                "init_state_index": 0,
                "denoising_steps": step,
                "success": success,
            }
            for task, outcomes in ((0, {1: False, 5: True}), (1, {1: True, 5: True}))
            for step, success in outcomes.items()
        ]
    )
    monkeypatch.setattr(
        ANALYSIS,
        "bootstrap_cluster_mean",
        lambda frame, value_column, iterations: (
            float(frame["base_task_id"].min()),
            float(frame["base_task_id"].max()),
        ),
    )

    result = ANALYSIS.paired_tests(episodes)

    assert len(result) == 1
    assert result.iloc[0]["pairs"] == 2
    assert result.iloc[0]["paired_cluster_bootstrap_low"] == 0
    assert result.iloc[0]["paired_cluster_bootstrap_high"] == 1


def test_line_success_accepts_exact_success_rates():
    frame = pd.DataFrame(
        {
            "denoising_steps": [1, 1, 5, 5],
            "success": [True, True, True, True],
            "suite": ["suite", "suite", "suite", "suite"],
        }
    )
    figure, axis = plt.subplots()
    try:
        ANALYSIS.line_success(axis, frame, "all success", "suite")
    finally:
        plt.close(figure)


def test_official_colon_paraphrase_is_recovered_from_yaml_mapping():
    assert PREPARE.normalize_official_paraphrase(
        {"set mugs": "white on left plate, yellow-white on right plate"}
    ) == "set mugs: white on left plate, yellow-white on right plate"

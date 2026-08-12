#!/usr/bin/env python3
"""Create a bounded Thor import package after the R2 feedback gate."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--matched-stale", type=Path, required=True)
    parser.add_argument("--slot-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    stale = json.loads(args.matched_stale.read_text(encoding="utf-8"))
    slots = json.loads(args.slot_contract.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    (out / "SERVER_COMMIT_SHA.txt").write_text(commit + "\n", encoding="utf-8")
    (out / "CHECKPOINT_SHA256.txt").write_text(
        "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2\n", encoding="utf-8"
    )
    shutil.copy2(args.slot_contract, out / "PV0_CONDITION_SLOT_CONTRACT.json")
    write_json(
        out / "R2_ROUTE_SPEC.json",
        {
            "bootstrap": "F1",
            "nominal_cycle": ["PV0", "P1", "P1"],
            "maximum_reuse_depth": 2,
            "denoising_steps": 1,
            "value_used": False,
            "forbidden": ["reuse_depth_3", "action_scaling", "trajectory_retiming", "hidden_activation_patch"],
        },
    )
    write_json(out / "MATCHED_STALE_SPEC.json", stale)
    write_json(
        out / "EXECUTION_FEEDBACK_FEATURE_SPEC.json",
        {
            "controller_visible_features": analysis["controller_visible_features"],
            "physical_score_features": analysis["physical_score_features_excluding_age_and_depth"],
            "oracle_labels_offline_only": analysis["label_definition"],
            "forbidden_runtime_inputs": ["F1/P1/PV0 shadow actions", "task name", "object pose", "contact", "reward", "success"],
        },
    )
    write_json(
        out / "PREDICTIVE_VALIDITY_SCORE.json",
        {
            "status": "NOT_PROMOTED",
            "decision": analysis["decision"],
            "weights_discovery_only": analysis["linear_score_weights"],
            "reason": "Physical feedback score did not beat AGE_ONLY task-balanced performance on validation or heldout.",
        },
    )
    write_json(
        out / "AGE_ONLY_BASELINE.json",
        {
            "status": "BASELINE_ONLY_NOT_FROZEN_CONTROLLER",
            "feature": "prediction_age_actions / reuse depth",
            "results": {split: analysis["baseline_evaluation"][split]["AGE_ONLY"] for split in ("discovery", "validation", "heldout")},
            "note": "A threshold was not frozen because the feedback-adaptation gate failed; do not infer an adaptive policy from this export.",
        },
    )
    write_json(
        out / "ADAPTIVE_EARLY_CORRECTION_POLICY.json",
        {
            "status": "DISABLED_BY_GATE",
            "decision": analysis["decision"],
            "allowed_nominal_backbone": ["PV0", "P1", "P1"],
            "reason": "No feedback score is authorized to trigger early PV0 correction in this branch.",
        },
    )
    write_json(
        out / "FEEDBACK_GO_NO_GO.json",
        {
            "wam_reuse_advantage": stale["decision"],
            "feedback_adaptation": analysis["decision"],
            "final_authorized_runtime": "fixed R2 only; AGE_ONLY remains an evaluation baseline, not a deployed adaptive rule.",
        },
    )
    write_json(
        out / "COST_ACCOUNTING_CONTRACT.json",
        {
            "report_per_executed_action": ["PV0 calls", "fresh VAE frames/calls", "P1 fraction", "active model work"],
            "matched_budget_requirement": "Any future adaptive comparison must freeze threshold on discovery and match PV0/fresh cost before validation/heldout.",
            "current_status": "No adaptive comparison authorized after FEEDBACK_ADAPTATION_NO_GO.",
        },
    )
    (out / "PV0_NATIVE_PATCH").write_text(
        "Native interface: get_action:persistent_visual_correction_prefix_frames.\n"
        "Refresh required current visual condition slots before denoiser forward 0; consult PV0_CONDITION_SLOT_CONTRACT.json.\n"
        "Do not copy a raw 13-frame constant across model layouts.\n",
        encoding="utf-8",
    )
    (out / "README_THOR_IMPORT.md").write_text(
        "# Thor import — bounded Feedback-Gated R2 result\n\n"
        "This package does **not** authorize a feedback-adaptive scheduler. The task-disjoint small gate returned `FEEDBACK_ADAPTATION_NO_GO`.\n\n"
        "Thor may reproduce fixed R2 (`F1 bootstrap → PV0 → P1 → P1`, max reuse depth 2) under the frozen checkpoint/denoise contract. Shadow F1/P1/PV0 labels are offline-only. PV0 must use layout-aware condition slots; LIBERO=13 raw frames, SO101/ALOHA layout=17 raw frames.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out), "decision": analysis["decision"], "commit": commit}))


if __name__ == "__main__":
    main()

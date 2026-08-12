#!/usr/bin/env python3
"""Write the layout-aware PV0 condition-slot contract for server/Thor handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    utils = root / "cosmos_policy/experiments/robot/cosmos_utils.py"
    configs = root / "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py"
    payload = {
        "schema_version": 1,
        "name": "PV0_CONDITION_SLOT_CONTRACT",
        "checkpoint_scope": "LIBERO original pre-finetune checkpoint only",
        "native_interface": "get_action:persistent_visual_correction_prefix_frames",
        "principle": "PV0 refreshes required current-visual condition slots; a raw prefix count is derived from layout and must not be treated as universal.",
        "temporal_vae": {
            "compression_factor": 4,
            "minimum_raw_frames_for_n_condition_slots": "1 + compression_factor * (n_condition_slots - 1)",
            "structural_leading_frame": 1,
        },
        "layouts": {
            "LIBERO": {
                "state_t": 9,
                "min_num_conditional_frames": 4,
                "all_current_condition_slots": {
                    "0": "structural blank",
                    "1": "current proprio",
                    "2": "current wrist image",
                    "3": "current primary image",
                },
                "required_fresh_visual_slots": {"2": "wrist camera", "3": "primary/third-person camera"},
                "causal_raw_prefix_frames": 13,
                "selective_slot_recomputation": "Architecturally legal only through native prefix encoding followed by index_copy into slots 2 and 3 before denoiser forward 0; direct hidden-activation edits are forbidden.",
            },
            "SO101_ALOHA_LAYOUT": {
                "state_t": 11,
                "min_num_conditional_frames": 5,
                "all_current_condition_slots": {
                    "0": "structural blank",
                    "1": "current proprio",
                    "2": "current left wrist image",
                    "3": "current right wrist image",
                    "4": "current primary image",
                },
                "required_fresh_visual_slots": {
                    "2": "left wrist camera",
                    "3": "right wrist camera",
                    "4": "primary camera",
                },
                "causal_raw_prefix_frames": 17,
                "selective_slot_recomputation": "Not validated by this LIBERO experiment. It requires an SO101-native implementation that refreshes all listed current visual slots and validates action schema/causal input handling separately.",
            },
        },
        "runtime_contract": {
            "p1": "moves prior generated future visual content into current visual slots without camera preprocessing",
            "pv0": "starts from predicted joint latent then overwrites the layout-required current visual slots using causal physical camera input before the sole denoiser forward",
            "denoising_steps": 1,
            "cosmos_value_used": False,
            "no_hidden_activation_patch": True,
            "no_cross_layout_prefix_assumption": True,
        },
        "source_audit": {
            "cosmos_utils_sha256": source_digest(utils),
            "experiment_configs_sha256": source_digest(configs),
            "key_source_locations": {
                "layout_dispatch": "cosmos_utils.py:get_latent_indices_from_model_config",
                "visual_slot_discovery": "cosmos_utils.py:get_action:persistent visual_indices loop",
                "native_slot_write": "cosmos_utils.py:get_action:condition.gt_frames.index_copy_",
                "so101_layout": "cosmos_policy_experiment_configs.py:cosmos_predict2_2b_480p_so101_lerobot",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "libero_raw_prefix": 13, "so101_raw_prefix": 17}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from scripts.sweep_runner import parse_and_run

if __name__ == "__main__":
    parse_and_run(
        [
            "baseline_bf16.yaml",
            "baseline_fp16.yaml",
            "quant_int8.yaml",
            "quant_int8_w8a8.yaml",
            "quant_int4.yaml",
            "quant_int4_fake.yaml",
            "branch_quant_vision.yaml",
            "branch_quant_action.yaml",
            "branch_quant_attention.yaml",
            "branch_quant_mlp.yaml",
            "branch_backbone_int4_action_high.yaml",
        ]
    )

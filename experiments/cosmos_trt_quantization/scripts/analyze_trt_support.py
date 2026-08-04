#!/usr/bin/env python3
"""Run Torch-TensorRT partitioning without building engines."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"
sys.path[:0] = [str(ROOT), str(EXP)]

from wrappers.cosmos_denoiser_wrapper import fixture_args  # noqa: E402
from wrappers.trt_bfloat16_cast_converter import (  # noqa: E402
    register_bfloat16_to_copy_converter,
)


def main() -> None:
    import torch_tensorrt

    register_bfloat16_to_copy_converter()
    exported = torch.export.load(EXP / "engines/cosmos_denoiser_bf16_exported_program.pt2")
    fixture = torch.load(
        EXP / "calibration/fixed_denoiser_inputs.pt",
        map_location="cpu",
        weights_only=True,
    )
    args = fixture_args(fixture)
    while args and args[-1] is None:
        args = args[:-1]

    report_txt = EXP / "profiles/bf16_trt_dryrun_with_bf16_cast_converter.txt"
    compiled = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=args,
        enabled_precisions={torch.bfloat16},
        require_full_compilation=False,
        min_block_size=1,
        dryrun=str(report_txt),
        use_python_runtime=False,
    )
    graph = compiled.graph
    target_counts = Counter(str(node.target) for node in graph.nodes if node.op.startswith("call_"))
    audit = {
        "graph_node_count_after_partitioning": sum(1 for _ in graph.nodes),
        "call_target_counts_after_partitioning": dict(target_counts.most_common()),
        "tensorrt_partitions": sum(
            1 for node in graph.nodes if node.op == "call_module" and "run_on_acc" in str(node.target)
        ),
        "pytorch_partitions": sum(
            1 for node in graph.nodes if node.op == "call_module" and "run_on_gpu" in str(node.target)
        ),
    }
    (EXP / "profiles/bf16_trt_dryrun_graph.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

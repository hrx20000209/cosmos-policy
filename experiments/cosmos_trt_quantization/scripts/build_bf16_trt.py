#!/usr/bin/env python3
"""Attempt a fixed-shape, full-compilation BF16 Torch-TensorRT engine build."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"
OLD = ROOT / "experiments/liberoplus_quantization/scripts"
sys.path[:0] = [str(ROOT), str(OLD), str(EXP)]

from quant_microbench import Cfg  # noqa: E402
import quant_lib  # noqa: E402
from wrappers.cosmos_denoiser_wrapper import CosmosDenoiserWrapper, fixture_args  # noqa: E402
from wrappers.trt_bfloat16_cast_converter import (  # noqa: E402
    register_bfloat16_to_copy_converter,
)


def rewrite_transformer_engine(net: torch.nn.Module, mode: str) -> dict:
    """Replace export-hostile TE ops with semantically matching aten ops."""
    audit = {
        "mode": mode,
        "rmsnorm_replaced": 0,
        "attention_replaced": 0,
        "fused_rope_disabled": False,
    }
    if mode == "none":
        return audit
    for parent in net.modules():
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ == "RMSNorm" and child.__class__.__module__.startswith("transformer_engine"):
                replacement = torch.nn.RMSNorm(
                    child.weight.shape,
                    eps=child.eps,
                    elementwise_affine=True,
                ).to(device=child.weight.device, dtype=child.weight.dtype)
                replacement.weight.data.copy_(child.weight.data)
                setattr(parent, name, replacement)
                audit["rmsnorm_replaced"] += 1
    if mode == "all":
        import cosmos_policy._src.predict2.networks.minimal_v4_dit as dit_module

        torch_attention_op = dit_module.torch_attention_op

        for module in net.modules():
            if module.__class__.__name__ == "Attention":
                module.backend = "torch"
                module._modules.pop("attn_op", None)
                module.attn_op = torch_attention_op
                audit["attention_replaced"] += 1
        original_rope = dit_module.apply_rotary_pos_emb

        def unfused_rope(tensor, freqs, tensor_format="sbhd", **kwargs):
            return original_rope(tensor, freqs, tensor_format=tensor_format, fused=False)

        dit_module.apply_rotary_pos_emb = unfused_rope
        audit["fused_rope_disabled"] = True
    return audit


def node_audit(exported_program) -> dict:
    rows = []
    counts: dict[str, int] = {}
    for node in exported_program.graph_module.graph.nodes:
        target = str(node.target)
        key = f"{node.op}:{target}"
        counts[key] = counts.get(key, 0) + 1
        rows.append({"name": node.name, "op": node.op, "target": target})
    return {"total_nodes": len(rows), "operator_counts": counts, "nodes": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-full", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rewrite-te", choices=("none", "norm", "all"), default="none")
    parser.add_argument("--register-bf16-cast-converter", action="store_true")
    args = parser.parse_args()

    fixture_path = EXP / "calibration/fixed_denoiser_inputs.pt"
    report_path = EXP / "profiles/bf16_trt_partition_report.json"
    inspector_path = EXP / "profiles/bf16_trt_engine_inspector.txt"
    graph_breaks_path = EXP / "profiles/torchao_graph_breaks.txt"
    engine_path = EXP / "engines/cosmos_denoiser_bf16.ts"
    report: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "require_full_compilation": args.require_full,
        "rewrite_te": args.rewrite_te,
        "fixture": str(fixture_path),
        "engine": str(engine_path),
        "export_success": False,
        "build_success": False,
        "engine_saved": False,
        "node_coverage": 0.0,
        "parameter_coverage": 0.0,
        "latency_coverage": None,
        "fallback_count": None,
        "engine_boundaries": None,
        "attempts": [],
    }

    from cosmos_policy.experiments.robot.cosmos_utils import get_model

    model, _ = get_model(Cfg())
    model.eval()
    quant_lib.configure_model_precision(model, "bf16")
    wrapper = CosmosDenoiserWrapper(model.net).eval()
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    all_args = fixture_args(fixture)
    # Optional arguments absent in the real call remain Python constants.
    while all_args and all_args[-1] is None:
        all_args = all_args[:-1]
    eager_reference = None
    with torch.inference_mode():
        eager_reference = wrapper(*all_args)
    torch.cuda.synchronize()
    report["rewrite_audit"] = rewrite_transformer_engine(model.net, args.rewrite_te)
    if args.rewrite_te != "none":
        with torch.inference_mode():
            rewritten_reference = wrapper(*all_args)
        torch.cuda.synchronize()
        rewrite_diff = rewritten_reference.float() - eager_reference.float()
        report["rewrite_numerics"] = {
            "cosine": float(
                torch.nn.functional.cosine_similarity(
                    rewritten_reference.float().flatten(),
                    eager_reference.float().flatten(),
                    dim=0,
                )
            ),
            "max_abs": float(rewrite_diff.abs().max()),
            "l2": float(torch.linalg.vector_norm(rewrite_diff)),
            "finite": bool(torch.isfinite(rewritten_reference.float()).all()),
        }
        report["rewrite_numerics"]["meets_exact_wrapper_gate_0_99999"] = (
            report["rewrite_numerics"]["cosine"] > 0.99999
        )
        report["rewrite_numerics"]["meets_engine_gate_0_999"] = (
            report["rewrite_numerics"]["cosine"] >= 0.999
        )
        if not report["rewrite_numerics"]["meets_engine_gate_0_999"]:
            raise RuntimeError(f"TE rewrite violated engine cosine gate: {report['rewrite_numerics']}")
        eager_reference = rewritten_reference
    report["input_shapes"] = [list(value.shape) if isinstance(value, torch.Tensor) else None for value in all_args]
    report["input_dtypes"] = [str(value.dtype) if isinstance(value, torch.Tensor) else str(type(value)) for value in all_args]
    report["parameter_count"] = sum(parameter.numel() for parameter in wrapper.parameters())

    try:
        exported = torch.export.export(wrapper, all_args, strict=False)
        report["export_success"] = True
        report["export"] = node_audit(exported)
        torch.export.save(exported, str(EXP / "engines/cosmos_denoiser_bf16_exported_program.pt2"))
    except Exception as exc:
        detail = traceback.format_exc()
        report["attempts"].append(
            {"stage": "torch.export strict=False", "success": False, "error": repr(exc), "traceback": detail}
        )
        graph_breaks_path.write_text(detail)
        report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
        inspector_path.write_text("BF16 TensorRT engine was not created because torch.export failed.\n\n" + detail)
        print(detail)
        print(json.dumps(report, indent=2, default=str))
        return

    try:
        import torch_tensorrt

        if args.register_bf16_cast_converter:
            register_bfloat16_to_copy_converter()
        report["custom_bfloat16_cast_converter"] = args.register_bf16_cast_converter
        compile_start = time.perf_counter()
        compiled = torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=all_args,
            enabled_precisions={torch.bfloat16},
            require_full_compilation=args.require_full,
            min_block_size=1,
            workspace_size=args.workspace_gb << 30,
            optimization_level=3,
            debug=args.debug,
            use_python_runtime=False,
            pass_through_build_failures=True,
        )
        report["build_seconds"] = time.perf_counter() - compile_start
        report["engine_built_in_memory"] = True
        report["attempts"].append({"stage": "torch_tensorrt full compile", "success": True})
        with torch.inference_mode():
            actual = compiled(*all_args)
        torch.cuda.synchronize()
        diff = actual.float() - eager_reference.float()
        report["numerics"] = {
            "cosine": float(torch.nn.functional.cosine_similarity(actual.float().flatten(), eager_reference.float().flatten(), dim=0)),
            "max_abs": float(diff.abs().max()),
            "l2": float(torch.linalg.vector_norm(diff)),
            "finite": bool(torch.isfinite(actual.float()).all()),
        }
        graph_text = str(compiled.graph)
        trt_nodes = graph_text.count("run_on_acc")
        torch_nodes = graph_text.count("run_on_gpu")
        call_nodes = sum(1 for node in compiled.graph.nodes if node.op == "call_module")
        report.update(
            {
                "compiled_graph": graph_text,
                "compiled_call_module_nodes": call_nodes,
                "trt_partition_count": trt_nodes,
                "pytorch_partition_count": torch_nodes,
                "fallback_count": torch_nodes,
                "engine_boundaries": max(0, 2 * trt_nodes - 2) if trt_nodes else 0,
                "node_coverage": 1.0 if args.require_full and torch_nodes == 0 else None,
                "parameter_coverage": 1.0 if args.require_full and torch_nodes == 0 else None,
            }
        )
        # ExportedProgram serialization uses pickle protocol 2 in PyTorch 2.7
        # and overflows for this >4-GiB TRT engine string. TorchScript's zip
        # serialization supports the large engine and is loadable by
        # torch_tensorrt.load/torch.jit.load.
        torch_tensorrt.save(
            compiled,
            str(engine_path),
            output_format="torchscript",
            arg_inputs=all_args,
        )
        report["engine_saved"] = True
        report["build_success"] = True
        report["engine_size_bytes"] = engine_path.stat().st_size
        inspector_path.write_text(
            "Torch-TensorRT compiled graph\n"
            f"TensorRT partitions: {trt_nodes}\nPyTorch partitions: {torch_nodes}\n\n{graph_text}\n"
        )
    except Exception as exc:
        detail = traceback.format_exc()
        report["attempts"].append(
            {"stage": "torch_tensorrt full compile", "success": False, "error": repr(exc), "traceback": detail}
        )
        inspector_path.write_text("BF16 TensorRT build failed.\n\n" + detail)
        print(detail)

    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

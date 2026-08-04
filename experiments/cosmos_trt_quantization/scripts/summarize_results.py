#!/usr/bin/env python3
"""Consolidate measured artifacts into CSVs used by the report and figures."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def finite_or_blank(value):
    if value is None:
        return ""
    try:
        return value if math.isfinite(float(value)) else ""
    except (TypeError, ValueError):
        return value


def action_metrics(path: Path, reference: np.ndarray | None) -> dict:
    if not path.exists() or reference is None:
        return {}
    action = np.load(path).astype(np.float64)
    diff = action - reference
    denom = np.linalg.norm(action) * np.linalg.norm(reference)
    return {
        "action_cosine": float(np.dot(action.ravel(), reference.ravel()) / denom),
        "action_l1": float(np.mean(np.abs(diff))),
        "action_l2": float(np.linalg.norm(diff)),
        "action_max_abs": float(np.max(np.abs(diff))),
        "nan_inf_count": int((~np.isfinite(action)).sum()),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: finite_or_blank(row.get(key, "")) for key in fields} for row in rows)


def main() -> None:
    summaries = EXP / "summaries"
    raw = EXP / "raw"
    reference_path = raw / "torchao_action_bf16.npy"
    reference = np.load(reference_path).astype(np.float64) if reference_path.exists() else None
    modes = [
        ("BF16 eager", "bf16"),
        ("torchao FP8", "fp8"),
        ("torchao INT8-WO", "int8_wo"),
        ("torchao W8A8", "int8_w8a8"),
        ("torchao INT4-WO", "int4_wo"),
        ("fake INT4", "fake_int4"),
    ]
    rows = []
    for label, slug in modes:
        payload = load_json(summaries / f"torchao_latency_{slug}.json")
        if not payload:
            continue
        policy = payload["policy_cuda_ms"]
        step = payload["denoiser_step_ms"]
        metrics = action_metrics(raw / f"torchao_action_{slug}.npy", reference)
        rows.append(
            {
                "backend": label,
                "backend_slug": slug,
                "status": "measured",
                "policy_iterations": payload["iters"],
                "policy_p50_ms": policy["p50"],
                "policy_p95_ms": policy["p95"],
                "policy_p99_ms": policy["p99"],
                "denoiser_p50_ms": step["p50"],
                "denoiser_p95_ms": step["p95"],
                "denoiser_p99_ms": step["p99"],
                "peak_memory_mb": payload.get("peak_allocated_mb"),
                "engine_size_mb": "",
                "node_coverage": "",
                "parameter_coverage": payload.get("coverage_frac", 0.0),
                "fallback_count": "",
                "engine_boundaries": "",
                **metrics,
            }
        )

    eager_denoiser = load_json(summaries / "denoiser_microbenchmark_bf16_eager.json")
    if eager_denoiser:
        for row in rows:
            if row["backend_slug"] == "bf16":
                row.update(
                    {
                        "denoiser_p50_ms": eager_denoiser["cuda"]["p50_ms"],
                        "denoiser_p95_ms": eager_denoiser["cuda"]["p95_ms"],
                        "denoiser_p99_ms": eager_denoiser["cuda"]["p99_ms"],
                        "denoiser_iterations": eager_denoiser["iterations"],
                    }
                )

    trt_report = load_json(EXP / "profiles/bf16_trt_partition_report.json")
    trt_final = load_json(summaries / "bf16_trt_final_status.json")
    for slug, label in (("bf16_trt", "BF16 TensorRT"), ("fp8_trt", "FP8 TensorRT")):
        policy = load_json(summaries / f"policy_microbenchmark_{slug}.json")
        denoiser = load_json(summaries / f"denoiser_microbenchmark_{slug}.json")
        engine_stem = slug.split("_")[0]
        engine_candidates = [
            EXP / f"engines/cosmos_denoiser_{engine_stem}.ts",
            EXP / f"engines/cosmos_denoiser_{engine_stem}.ep",
        ]
        engine_file = next((path for path in engine_candidates if path.exists()), None)
        built = (
            trt_final.get("engine_persisted", False)
            if slug == "bf16_trt"
            else engine_file is not None
        )
        status = "measured" if policy and denoiser else ("built_not_measured" if built else "not_available")
        row = {
            "backend": label,
            "backend_slug": slug,
            "status": status,
            "policy_iterations": policy.get("iterations", ""),
            "policy_p50_ms": policy.get("cuda", {}).get("p50_ms", ""),
            "policy_p95_ms": policy.get("cuda", {}).get("p95_ms", ""),
            "policy_p99_ms": policy.get("cuda", {}).get("p99_ms", ""),
            "denoiser_iterations": denoiser.get("iterations", ""),
            "denoiser_p50_ms": denoiser.get("cuda", {}).get("p50_ms", ""),
            "denoiser_p95_ms": denoiser.get("cuda", {}).get("p95_ms", ""),
            "denoiser_p99_ms": denoiser.get("cuda", {}).get("p99_ms", ""),
            "peak_memory_mb": policy.get("peak_memory_mb", ""),
            "engine_size_mb": round(engine_file.stat().st_size / 2**20, 3) if engine_file else "",
            "node_coverage": trt_final.get("node_coverage", "") if slug == "bf16_trt" else "",
            "parameter_coverage": trt_final.get("parameter_coverage", "") if slug == "bf16_trt" else "",
            "fallback_count": trt_final.get("fallback_ops", "") if slug == "bf16_trt" else "",
            "engine_boundaries": trt_final.get("engine_boundaries", "") if slug == "bf16_trt" else "",
            "action_cosine": policy.get("action_cosine", ""),
            "action_l1": policy.get("action_l1", ""),
            "action_l2": policy.get("action_l2", ""),
            "action_max_abs": policy.get("action_max_abs", ""),
            "nan_inf_count": policy.get("nan_inf_count", ""),
        }
        rows.append(row)

    bf16_p50 = next((float(row["policy_p50_ms"]) for row in rows if row["backend_slug"] == "bf16"), None)
    trt_p50 = next(
        (
            float(row["policy_p50_ms"])
            for row in rows
            if row["backend_slug"] == "bf16_trt" and row["policy_p50_ms"] != ""
        ),
        None,
    )
    for row in rows:
        value = row.get("policy_p50_ms")
        value = float(value) if value != "" else None
        row["speedup_vs_bf16_eager"] = bf16_p50 / value if value and bf16_p50 else ""
        row["speedup_vs_bf16_trt"] = trt_p50 / value if value and trt_p50 else ""

    fields = [
        "backend", "backend_slug", "status", "policy_iterations", "denoiser_iterations",
        "policy_p50_ms", "policy_p95_ms", "policy_p99_ms", "denoiser_p50_ms",
        "denoiser_p95_ms", "denoiser_p99_ms", "speedup_vs_bf16_eager",
        "speedup_vs_bf16_trt", "peak_memory_mb", "engine_size_mb", "action_cosine",
        "action_l1", "action_l2", "action_max_abs", "nan_inf_count", "node_coverage",
        "parameter_coverage", "fallback_count", "engine_boundaries",
    ]
    write_csv(summaries / "backend_latency_summary.csv", rows, fields)

    # Normalize the six measured policy streams into the required unified raw table.
    unified = []
    for _label, slug in modes:
        source = raw / f"torchao_latency_{slug}.csv"
        if not source.exists():
            continue
        with source.open(newline="") as handle:
            for row in csv.DictReader(handle):
                unified.append(row)
    if unified:
        write_csv(raw / "policy_microbenchmark.csv", unified, list(unified[0]))

    coverage_rows = [
        {
            "backend": "BF16 TRT native TE",
            "node_coverage": "",
            "parameter_coverage": "",
            "fallback_ops": "",
            "partitions": "",
            "status": "export_failed_TE_RMSNorm",
        },
        {
            "backend": "BF16 TRT rewritten, built-in converters",
            "node_coverage": 6038 / 6492,
            "parameter_coverage": 1.0,
            "fallback_ops": 454,
            "partitions": 61,
            "status": "dryrun_only",
        },
        {
            "backend": "BF16 TRT rewritten + BF16 cast converter",
            "node_coverage": 1.0,
            "parameter_coverage": 1.0,
            "fallback_ops": 0,
            "partitions": 1,
            "status": "compiled_in_memory_not_persisted",
        },
        {
            "backend": "FP8 TRT",
            "node_coverage": "",
            "parameter_coverage": "",
            "fallback_ops": "",
            "partitions": "",
            "status": "not_attempted_unless_bf16_gate_passes",
        },
    ]
    write_csv(summaries / "trt_coverage_summary.csv", coverage_rows)

    gate = {
        "bf16_trt_built_in_memory": bool(trt_final.get("full_graph_compiled_in_memory")),
        "bf16_trt_persisted": bool(trt_final.get("engine_persisted")),
        "bf16_trt_stable_and_measured": any(
            row["backend_slug"] == "bf16_trt" and row["status"] == "measured" for row in rows
        ),
        "fp8_ptq_started": (EXP / "calibration/calibration_manifest.json").exists(),
        "fp8_trt_built": any(
            (EXP / f"engines/cosmos_denoiser_fp8{suffix}").exists() for suffix in (".ts", ".ep")
        ),
        "rollout_started": False,
        "reason": "FP8 and rollout remain gated until a stable, measured BF16 TensorRT baseline exists.",
    }
    (summaries / "speedup_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    kernel_path = EXP / "profiles/torchao_kernel_summary.csv"
    if kernel_path.exists():
        def category(name: str) -> str:
            lower = name.lower()
            if any(token in lower for token in ("flash", "fmha", "attention")):
                return "attention"
            if any(token in lower for token in ("dequant", "unpack", "convert_weight", "int4pack")):
                return "dequant/unpack"
            if any(
                token in lower
                for token in (
                    "choose_qparams", "quantize", "absmax", "amax", "reduce_max",
                    "minnanfunctor", "maxnanfunctor", "round_kernel", "clamp_scalar",
                    "signed char",
                )
            ):
                return "activation quantize"
            if any(token in lower for token in ("gemm", "cutlass", "tinygemm", "scaled_mm", "int8_mm", "triton")):
                return "GEMM"
            if any(token in lower for token in ("norm", "layer_norm", "rmsnorm")):
                return "norm"
            if any(token in lower for token in ("memcpy", "copy", "cast")):
                return "cast/copy"
            if "elementwise" in lower or "vectorized" in lower:
                return "elementwise"
            return "other"

        aggregate = defaultdict(lambda: {"kernel_count": 0, "launch_count": 0, "total_cuda_us": 0.0})
        with kernel_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["quant_mode"], category(row["kernel_name"]))
                aggregate[key]["kernel_count"] += 1
                aggregate[key]["launch_count"] += int(row["count"])
                aggregate[key]["total_cuda_us"] += float(row["total_cuda_us"])
        kernel_rows = [
            {
                "quant_mode": mode,
                "category": cat,
                **values,
            }
            for (mode, cat), values in sorted(aggregate.items())
        ]
        write_csv(summaries / "kernel_category_summary.csv", kernel_rows)


if __name__ == "__main__":
    main()

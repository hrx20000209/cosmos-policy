#!/usr/bin/env python3
"""Generate all requested PNG/PDF figures strictly from experiment CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/cosmos_trt_quantization"
SUMMARY = EXP / "summaries"
FIG = EXP / "figures"


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    try:
        return float(value) if value not in ("", None) else np.nan
    except ValueError:
        return np.nan


def finish(fig, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def empty(ax, message="No measured data (gate blocked)"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def bar(ax, labels, values, ylabel, title):
    valid = [(label, value) for label, value in zip(labels, values) if np.isfinite(value)]
    if not valid:
        empty(ax)
        ax.set_title(title)
        return
    x = np.arange(len(valid))
    vals = [value for _, value in valid]
    ax.bar(x, vals, color=plt.cm.viridis(np.linspace(0.15, 0.85, len(valid))))
    ax.set_xticks(x, [label for label, _ in valid], rotation=28, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for index, value in enumerate(vals):
        ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    backend = rows(SUMMARY / "backend_latency_summary.csv")
    measured = [row for row in backend if row["status"] == "measured"]
    kernel = rows(SUMMARY / "kernel_category_summary.csv")
    sweep = rows(SUMMARY / "denoising_sweep_summary.csv")
    coverage = rows(SUMMARY / "trt_coverage_summary.csv")

    # 1. Kernel time decomposition.
    fig, ax = plt.subplots(figsize=(10, 5))
    modes = sorted({row["quant_mode"] for row in kernel})
    cats = ["GEMM", "activation quantize", "dequant/unpack", "attention", "norm", "cast/copy", "elementwise", "other"]
    bottom = np.zeros(len(modes))
    for cat in cats:
        values = np.array([
            sum(number(row["total_cuda_us"]) for row in kernel if row["quant_mode"] == mode and row["category"] == cat) / 1000
            for mode in modes
        ])
        ax.bar(modes, values, bottom=bottom, label=cat)
        bottom += values
    if modes:
        ax.legend(ncol=4, fontsize=7)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("CUDA time per traced 5-step call (ms)")
    else:
        empty(ax)
    ax.set_title("torchao kernel time decomposition")
    finish(fig, "01_torchao_kernel_time_breakdown")

    # 2–4. Latency views.
    for stem, field, title in (
        ("02_backend_latency", "policy_p50_ms", "Backend p50 latency"),
        ("03_single_denoiser_latency", "denoiser_p50_ms", "Single denoiser p50 latency"),
        ("04_policy_5step_latency", "policy_p50_ms", "Complete 5-step policy p50 latency"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        bar(ax, [r["backend"] for r in measured], [number(r[field]) for r in measured], "Latency (ms)", title)
        finish(fig, stem)

    # 5. p50/p95/p99.
    fig, ax = plt.subplots(figsize=(11, 5))
    if measured:
        x = np.arange(len(measured))
        width = 0.25
        for offset, field, label in ((-width, "policy_p50_ms", "p50"), (0, "policy_p95_ms", "p95"), (width, "policy_p99_ms", "p99")):
            ax.bar(x + offset, [number(r[field]) for r in measured], width, label=label)
        ax.set_xticks(x, [r["backend"] for r in measured], rotation=28, ha="right")
        ax.set_ylabel("Policy latency (ms)")
        ax.legend()
    else:
        empty(ax)
    ax.set_title("Policy latency percentiles")
    finish(fig, "05_policy_latency_percentiles")

    # 6. Speedup.
    fig, ax = plt.subplots(figsize=(9, 5))
    bar(
        ax,
        [r["backend"] for r in measured],
        [number(r["speedup_vs_bf16_eager"]) for r in measured],
        "Speedup (x)",
        "Speedup vs BF16 eager",
    )
    ax.axhline(1.0, color="black", linewidth=0.8)
    finish(fig, "06_speedup_vs_backend")

    # 7–8. Memory and engine size.
    for stem, field, ylabel, title in (
        ("07_peak_memory", "peak_memory_mb", "Peak allocated memory (MB)", "Peak GPU memory"),
        ("08_engine_size", "engine_size_mb", "Serialized engine size (MB)", "Engine size"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        source = measured if field == "peak_memory_mb" else backend
        bar(ax, [r["backend"] for r in source], [number(r[field]) for r in source], ylabel, title)
        finish(fig, stem)

    # 9. Action numeric error.
    fig, ax1 = plt.subplots(figsize=(10, 5))
    action_rows = [r for r in measured if np.isfinite(number(r["action_cosine"]))]
    if action_rows:
        x = np.arange(len(action_rows))
        ax1.bar(x - 0.2, [1 - number(r["action_cosine"]) for r in action_rows], 0.4, label="1-cosine")
        ax1.bar(x + 0.2, [number(r["action_l2"]) for r in action_rows], 0.4, label="L2")
        ax1.set_xticks(x, [r["backend"] for r in action_rows], rotation=28, ha="right")
        ax1.legend()
    else:
        empty(ax1)
    ax1.set_title("Action cosine/L2 error vs BF16")
    finish(fig, "09_action_cosine_l2_error")

    # 10–11. Denoising steps.
    fig, ax = plt.subplots(figsize=(7, 5))
    if sweep:
        ax.plot([number(r["denoising_steps"]) for r in sweep], [number(r["policy_cuda_ms_p50"]) for r in sweep], marker="o")
        ax.set_xlabel("Denoising steps")
        ax.set_ylabel("Policy p50 CUDA latency (ms)")
    else:
        empty(ax)
    ax.set_title("Denoising steps vs latency")
    finish(fig, "denoising_steps_vs_latency")

    fig, ax = plt.subplots(figsize=(7, 5))
    if sweep:
        ax.plot([number(r["denoising_steps"]) for r in sweep], [number(r["action_l2_vs_bf16_5step"]) for r in sweep], marker="o", label="L2")
        ax.plot([number(r["denoising_steps"]) for r in sweep], [1 - number(r["action_cosine_vs_bf16_5step"]) for r in sweep], marker="s", label="1-cosine")
        ax.set_xlabel("Denoising steps")
        ax.legend()
    else:
        empty(ax)
    ax.set_title("Denoising steps vs action error")
    finish(fig, "denoising_steps_vs_action_error")

    # 12. Latency-accuracy Pareto.
    fig, ax = plt.subplots(figsize=(7, 5))
    if sweep:
        for row in sweep:
            x, y = number(row["policy_cuda_ms_p50"]), number(row["action_l2_vs_bf16_5step"])
            ax.scatter(x, y)
            ax.annotate(f"{row['denoising_steps']} step", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Policy p50 latency (ms)")
        ax.set_ylabel("Action L2 vs 5-step")
    else:
        empty(ax)
    ax.set_title("Latency–action-error Pareto")
    finish(fig, "denoising_steps_pareto")

    # 13. TensorRT op coverage.
    fig, ax = plt.subplots(figsize=(10, 5))
    bar(ax, [r["backend"] for r in coverage], [100 * number(r["node_coverage"]) for r in coverage], "Supported nodes (%)", "TensorRT op coverage")
    ax.set_ylim(0, 110)
    finish(fig, "13_tensorrt_op_coverage")

    # 14. Q/DQ and fallback share (fallback is zero for eager profiler; TRT data
    # remains absent until an engine exists).
    fig, ax = plt.subplots(figsize=(10, 5))
    selected = ["activation quantize", "dequant/unpack", "GEMM"]
    bottom = np.zeros(len(modes))
    for cat in selected:
        values = np.array([
            sum(number(r["total_cuda_us"]) for r in kernel if r["quant_mode"] == mode and r["category"] == cat) / 1000
            for mode in modes
        ])
        ax.bar(modes, values, bottom=bottom, label=cat)
        bottom += values
    if modes:
        ax.legend()
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("CUDA time per traced call (ms)")
    else:
        empty(ax)
    ax.set_title("Quantize/dequantize/GEMM time")
    finish(fig, "14_qdq_fallback_time_share")


if __name__ == "__main__":
    main()

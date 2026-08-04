#!/usr/bin/env python3
"""Generate the requested figure set; unavailable results are labeled, never imputed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
SUMMARY = EXP / "summaries"
FIGURES = EXP / "figures"
HARDWARE = "8× RTX 4090 D host; one GPU per evaluation shard"

FIGURE_SPECS = {
    "01_success_vs_policy_latency": ("policy_latency_mean_ms", "success_rate"),
    "02_success_vs_episode_latency": ("episode_latency_mean_ms", "success_rate"),
    "03_success_vs_peak_memory": ("peak_gpu_memory_mb", "success_rate"),
    "04_precision_vs_policy_latency": ("precision", "policy_latency_mean_ms"),
    "05_precision_vs_success": ("precision", "success_rate"),
    "06_openloop_vs_success": ("num_open_loop_steps", "success_rate"),
    "07_openloop_vs_episode_latency": (
        "num_open_loop_steps",
        "episode_latency_mean_ms",
    ),
    "08_denoising_vs_success": ("denoising_steps", "success_rate"),
    "09_denoising_vs_policy_latency": ("denoising_steps", "policy_latency_mean_ms"),
}


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def unavailable(name: str, title: str, reason: str) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(
        0.5,
        0.55,
        "No completed LIBERO-Plus data",
        ha="center",
        va="center",
        fontsize=16,
    )
    axis.text(0.5, 0.42, reason, ha="center", va="center", wrap=True, fontsize=10)
    axis.text(0.5, 0.12, HARDWARE, ha="center", va="center", fontsize=9)
    save(fig, name)


def scatter_summary(summary: pd.DataFrame, name: str, x: str, y: str) -> None:
    usable = summary.dropna(subset=[x, y]) if x in summary and y in summary else pd.DataFrame()
    if usable.empty:
        unavailable(name, name.replace("_", " "), "The GPU pilot has not completed.")
        return
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    categorical = not pd.api.types.is_numeric_dtype(usable[x])
    positions = np.arange(len(usable)) if categorical else usable[x].to_numpy()
    y_values = usable[y].to_numpy()
    axis.scatter(positions, y_values, s=75)
    if categorical:
        axis.set_xticks(positions, usable[x].astype(str), rotation=25, ha="right")
    for position, value, (_, row) in zip(positions, y_values, usable.iterrows()):
        axis.annotate(
            f"{row['experiment_name']}\nN={int(row['completed_episodes'])}",
            (position, value),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
        )
    if y == "success_rate":
        low = usable["success_ci95_low"].to_numpy()
        high = usable["success_ci95_high"].to_numpy()
        axis.errorbar(
            positions,
            y_values,
            yerr=np.vstack((y_values - low, high - y_values)),
            fmt="none",
            capsize=4,
        )
        axis.set_ylim(0, 1.05)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    axis.set_title(f"{name.replace('_', ' ')}\n{HARDWARE}")
    save(fig, name)


def precision_microbenchmark() -> None:
    payload = json.loads((SUMMARY / "latency_microbench_combined.json").read_text())
    rows = payload["rows"]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    labels = [
        f"{row['mode']}\n{'real' if row['hw_accel'] else 'fake/baseline'}" for row in rows
    ]
    values = [row["eager_denoise_p50_ms"] for row in rows]
    axis.bar(np.arange(len(rows)), values, color="#4E79A7")
    axis.set_xticks(np.arange(len(rows)), labels, rotation=20, ha="right")
    axis.set_ylabel("5-step denoising p50 (ms)")
    axis.set_title(
        "Precision versus policy denoising latency\n"
        "actual fixed-observation microbenchmark, N=20–30 calls/mode, RTX 4090 D"
    )
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    save(fig, "04_precision_vs_policy_latency")


def heatmap(summary: pd.DataFrame) -> None:
    if summary.empty or not {
        "quantization_mode",
        "num_open_loop_steps",
        "success_rate",
    }.issubset(summary.columns):
        unavailable(
            "10_quantization_openloop_heatmap",
            "Quantization × open-loop success heatmap",
            "No completed factorial sweep; cells are intentionally not imputed.",
        )
        return
    pivot = summary.pivot_table(
        index="quantization_mode",
        columns="num_open_loop_steps",
        values="success_rate",
    )
    if pivot.empty:
        unavailable(
            "10_quantization_openloop_heatmap",
            "Quantization × open-loop success heatmap",
            "No completed factorial sweep.",
        )
        return
    fig, axis = plt.subplots(figsize=(8, 4.8))
    image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(pivot.columns)), pivot.columns)
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_xlabel("open-loop steps")
    axis.set_ylabel("quantization")
    fig.colorbar(image, ax=axis, label="success rate")
    save(fig, "10_quantization_openloop_heatmap")


def failure_plot(failures: pd.DataFrame) -> None:
    if failures.empty:
        unavailable(
            "11_failure_distribution",
            "Failure category distribution",
            "Per-episode LIBERO-Plus failures have not been collected.",
        )
        return
    pivot = failures.pivot_table(
        index="experiment_name",
        columns="termination_reason",
        values="count",
        fill_value=0,
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    pivot.plot.bar(stacked=True, ax=axis)
    axis.set_ylabel("episode count")
    axis.legend(fontsize=7)
    save(fig, "11_failure_distribution")


def branch_plot(summary: pd.DataFrame, metric: str, name: str) -> None:
    branch = summary[
        summary.get("quantization_scope", pd.Series(dtype=str)).isin(
            ["vision_input_proxy", "shared_output_proxy", "attention", "mlp", "backbone"]
        )
    ]
    if branch.empty or metric not in branch:
        unavailable(
            name,
            name.replace("_", " "),
            "Branch-level evaluation/profile data are not available; parameter counts are not used as latency or memory.",
        )
        return
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(branch["quantization_scope"], branch[metric])
    axis.tick_params(axis="x", rotation=25)
    axis.set_ylabel(metric)
    for index, (_, row) in enumerate(branch.iterrows()):
        axis.text(index, row[metric], f"N={int(row['completed_episodes'])}", ha="center")
    save(fig, name)


def pareto(summary: pd.DataFrame) -> None:
    strategies = summary[
        summary.get("experiment_name", pd.Series(dtype=str)).str.contains(
            "async|dynamic", regex=True
        )
    ]
    if strategies.empty:
        unavailable(
            "14_fixed_dynamic_pareto",
            "Fixed versus dynamic Pareto frontier",
            "Dynamic and fixed strategy episodes have not completed.",
        )
        return
    scatter_summary(
        strategies,
        "14_fixed_dynamic_pareto",
        "episode_latency_mean_ms",
        "success_rate",
    )


def read_csv_if_available(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = SUMMARY / "overall_summary.csv"
    summary = read_csv_if_available(path)
    failures_path = SUMMARY / "failure_summary.csv"
    failures = read_csv_if_available(failures_path)
    for name, (x, y) in FIGURE_SPECS.items():
        if name == "04_precision_vs_policy_latency":
            precision_microbenchmark()
        else:
            scatter_summary(summary, name, x, y)
    heatmap(summary)
    failure_plot(failures)
    branch_plot(summary, "policy_latency_mean_ms", "12_branch_latency_breakdown")
    branch_plot(summary, "peak_gpu_memory_mb", "13_branch_memory_breakdown")
    pareto(summary)
    print(f"wrote 14 figure groups to {FIGURES}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate Cosmos denoising results, statistics, plots and stage-2 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT / "experiments/cosmos_denoising_libero_pro"
OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
STEPS = [1, 2, 3, 4, 5, 6]


def read_jsonl_tree(pattern: str) -> list[dict]:
    records = []
    for path in sorted(OUTPUT.glob(pattern)):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def percentile_summary(values: pd.Series) -> dict:
    clean = values.dropna().astype(float)
    if clean.empty:
        return {key: None for key in ("mean", "p50", "p90", "p95", "p99")}
    return {
        "mean": float(clean.mean()),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
    }


def bootstrap_task_clusters(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    cluster_column: str = "base_task_id",
    iterations: int = 5000,
    seed: int = 195,
) -> tuple[float, float]:
    tasks = sorted(frame[cluster_column].dropna().unique())
    if not tasks:
        return math.nan, math.nan
    grouped = {task: frame[frame[cluster_column] == task] for task in tasks}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        draw = pd.concat([grouped[task].assign(_bootstrap_cluster=i) for i, task in enumerate(sampled)])
        values.append(statistic(draw))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_cluster_mean(
    frame: pd.DataFrame,
    value_column: str,
    *,
    cluster_column: str = "base_task_id",
    iterations: int = 5000,
    seed: int = 195,
) -> tuple[float, float]:
    """Cluster bootstrap of a row-weighted mean without per-draw DataFrame copies."""
    grouped = frame.groupby(cluster_column, sort=True)[value_column].agg(["sum", "count"])
    if grouped.empty:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(grouped),
        np.full(len(grouped), 1.0 / len(grouped)),
        size=iterations,
    )
    numerators = weights @ grouped["sum"].to_numpy(dtype=float)
    denominators = weights @ grouped["count"].to_numpy(dtype=float)
    values = numerators / denominators
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def bootstrap_interaction_did(
    frame: pd.DataFrame,
    step: int,
    *,
    iterations: int = 5000,
    seed: int = 195,
) -> tuple[float, float]:
    """Cluster bootstrap for the PRO-vs-original difference in differences."""
    cells = [
        ("libero", step),
        ("libero", 5),
        ("libero_pro", step),
        ("libero_pro", 5),
    ]
    tasks = sorted(frame["base_task_id"].dropna().unique())
    if not tasks:
        return math.nan, math.nan
    task_index = {task: index for index, task in enumerate(tasks)}
    cell_index = {cell: index for index, cell in enumerate(cells)}
    successes = np.zeros((len(tasks), len(cells)), dtype=float)
    totals = np.zeros_like(successes)
    grouped = frame.groupby(
        ["base_task_id", "domain", "denoising_steps"], sort=False
    )["success"].agg(["sum", "count"])
    for (task, domain, denoising_steps), row in grouped.iterrows():
        column = cell_index.get((domain, int(denoising_steps)))
        if column is None:
            continue
        index = task_index[task]
        successes[index, column] = float(row["sum"])
        totals[index, column] = float(row["count"])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(tasks),
        np.full(len(tasks), 1.0 / len(tasks)),
        size=iterations,
    )
    sampled_successes = weights @ successes
    sampled_totals = weights @ totals
    rates = np.divide(
        sampled_successes,
        sampled_totals,
        out=np.zeros_like(sampled_successes),
        where=sampled_totals != 0,
    )
    values = (rates[:, 2] - rates[:, 3]) - (rates[:, 0] - rates[:, 1])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def aggregate(
    episodes: pd.DataFrame,
    requests: pd.DataFrame,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    keys = keys or ["domain", "perturbation_category", "suite", "denoising_steps"]
    for group_key, group in episodes.groupby(keys, dropna=False):
        success_count = int(group["success"].astype(bool).sum())
        n = len(group)
        low, high = wilson(success_count, n)
        req = requests
        for key, value in zip(keys, group_key):
            req = req[req[key] == value]
        latency = percentile_summary(req["total_policy_request_latency_ms"] if len(req) else pd.Series(dtype=float))
        dit_latency = percentile_summary(
            req["dit_denoising_latency_ms"] if len(req) else pd.Series(dtype=float)
        )
        rows.append(
            {
                **dict(zip(keys, group_key)),
                "episodes": n,
                "successes": success_count,
                "success_rate": success_count / n,
                "success_wilson_low": low,
                "success_wilson_high": high,
                "episode_time_mean_s": float(group["episode_wall_clock_time_s"].mean()),
                "episode_steps_mean": float(group["episode_steps"].mean()),
                "request_count_mean": float(group["request_count"].mean()),
                "forward_count_mean": float(group["request_forward_count_sum"].mean()),
                "action_smoothness_mean": float(group["action_smoothness"].mean()),
                "action_jerk_mean": float(group["action_jerk"].mean()),
                "chunk_boundary_discontinuity_mean": float(
                    group["chunk_boundary_action_discontinuity"].mean()
                ),
                "invalid_environment_rate": float(
                    1.0 - group["environment_valid"].astype(bool).mean()
                ),
                "torch_peak_memory_allocated_mb": float(group["torch_peak_memory_allocated_mb"].max()),
                **{f"latency_{key}_ms": value for key, value in latency.items()},
                **{f"dit_{key}_ms": value for key, value in dit_latency.items()},
            }
        )
    return pd.DataFrame(rows)


def paired_tests(episodes: pd.DataFrame, *, iterations: int = 5000) -> pd.DataFrame:
    rows = []
    pair_columns = [
        "domain",
        "perturbation_category",
        "base_task_id",
        "task_uid",
        "variant_id",
        "seed",
        "init_state_index",
    ]
    for (domain, category), group in episodes.groupby(["domain", "perturbation_category"]):
        pivot = group.pivot_table(
            index=pair_columns,
            columns="denoising_steps",
            values="success",
            aggfunc="last",
        )
        for step in STEPS:
            if step == 5 or step not in pivot or 5 not in pivot:
                continue
            paired = pivot[[step, 5]].dropna().astype(bool)
            b = int(((paired[step]) & (~paired[5])).sum())
            c = int(((~paired[step]) & (paired[5])).sum())
            discordant = b + c
            pvalue = float(binomtest(b, discordant, 0.5).pvalue) if discordant else 1.0
            differences = (
                paired[step].astype(float) - paired[5].astype(float)
            ).rename("difference").reset_index()
            low, high = bootstrap_cluster_mean(
                differences,
                "difference",
                iterations=iterations,
            )
            rows.append(
                {
                    "domain": domain,
                    "perturbation_category": category,
                    "step": step,
                    "reference_step": 5,
                    "pairs": len(paired),
                    "step_success_reference_failure": b,
                    "step_failure_reference_success": c,
                    "mcnemar_exact_p": pvalue,
                    "paired_success_difference": float(differences["difference"].mean()),
                    "paired_cluster_bootstrap_low": low,
                    "paired_cluster_bootstrap_high": high,
                }
            )
    return pd.DataFrame(rows)


def interaction_analysis(
    episodes: pd.DataFrame, *, iterations: int = 5000
) -> pd.DataFrame:
    original = episodes[episodes["domain"] == "libero"]
    pro = episodes[episodes["domain"] == "libero_pro"]
    rows = []
    for category, group in pro.groupby("perturbation_category"):
        matched_base_ids = set(group["base_task_id"])
        matched_original = original[original["base_task_id"].isin(matched_base_ids)]
        original_rates = matched_original.groupby("denoising_steps")["success"].mean()
        rates = group.groupby("denoising_steps")["success"].mean()
        for step in STEPS:
            if step not in rates or 5 not in rates or step not in original_rates or 5 not in original_rates:
                continue
            did = (rates[step] - rates[5]) - (original_rates[step] - original_rates[5])
            combined = pd.concat(
                [
                    matched_original[matched_original["denoising_steps"].isin([step, 5])],
                    group[group["denoising_steps"].isin([step, 5])],
                ]
            )

            low, high = bootstrap_interaction_did(
                combined,
                step,
                iterations=iterations,
            )
            rows.append(
                {
                    "perturbation_category": category,
                    "step": step,
                    "reference_step": 5,
                    "difference_in_differences": float(did),
                    "cluster_bootstrap_low": low,
                    "cluster_bootstrap_high": high,
                }
            )
    return pd.DataFrame(rows)


def robustness_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    """Matched-base-task PRO gaps and within-category penalties."""
    original = episodes[episodes["domain"] == "libero"]
    pro = episodes[episodes["domain"] == "libero_pro"]
    rows = []
    for category, category_group in pro.groupby("perturbation_category"):
        base_ids = set(category_group["base_task_id"])
        matched_original = original[original["base_task_id"].isin(base_ids)]
        original_rates = matched_original.groupby("denoising_steps")["success"].mean()
        pro_rates = category_group.groupby("denoising_steps")["success"].mean()
        for step in STEPS:
            if step not in original_rates or step not in pro_rates:
                continue
            rows.append(
                {
                    "perturbation_category": category,
                    "denoising_steps": step,
                    "matched_base_tasks": len(base_ids),
                    "matched_original_success_rate": float(original_rates[step]),
                    "pro_success_rate": float(pro_rates[step]),
                    "robustness_gap": float(original_rates[step] - pro_rates[step]),
                    "denoising_penalty_vs_5": float(
                        pro_rates[step] - pro_rates.get(5, math.nan)
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_plot(
    name: str,
    draw: Callable[[plt.Axes], None],
    figsize: tuple[float, float] = (8.5, 5.2),
) -> None:
    plot_dir = EXPERIMENT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    draw(ax)
    fig.savefig(plot_dir / f"{name}.png", dpi=300)
    fig.savefig(plot_dir / f"{name}.pdf")
    plt.close(fig)


def line_success(ax: plt.Axes, frame: pd.DataFrame, title: str, group: str | None = None) -> None:
    def errorbar(values: pd.DataFrame, label: str | None = None) -> None:
        means = values["mean"].to_numpy(dtype=float)
        intervals = np.asarray(
            [
                wilson(int(row["sum"]), int(row["count"]))
                for _, row in values.iterrows()
            ],
            dtype=float,
        )
        # Wilson endpoints can differ from an exact 0/1 mean by one floating ULP.
        yerr = np.vstack(
            [
                np.maximum(0.0, means - intervals[:, 0]),
                np.maximum(0.0, intervals[:, 1] - means),
            ]
        )
        ax.errorbar(
            values.index.to_numpy(),
            means,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=label,
        )

    if group:
        for label, selected in frame.groupby(group):
            values = selected.groupby("denoising_steps")["success"].agg(["mean", "count", "sum"])
            errorbar(values, str(label))
        ax.legend(fontsize=8)
    else:
        values = frame.groupby("denoising_steps")["success"].agg(["mean", "count", "sum"])
        errorbar(values)
    ax.set(title=title, xlabel="Denoising steps", ylabel="Success rate", ylim=(0, 1.02), xticks=STEPS)
    ax.set_xticklabels(["1", "2", "3", "4", "5\nNative", "6\nUpper"])
    ax.grid(alpha=0.25)


def generate_plots(
    episodes: pd.DataFrame,
    requests: pd.DataFrame,
    paired: pd.DataFrame,
    interactions: pd.DataFrame,
    episodes_all: pd.DataFrame,
) -> None:
    original = episodes[episodes["domain"] == "libero"]
    pro = episodes[episodes["domain"] == "libero_pro"]
    save_plot("01_original_steps_vs_overall_success", lambda ax: line_success(ax, original, "Original LIBERO success"))
    save_plot("02_original_steps_vs_success_by_suite", lambda ax: line_success(ax, original, "Original success by suite", "suite"))
    save_plot("03_pro_steps_vs_success_by_perturbation", lambda ax: line_success(ax, pro, "PRO success by perturbation", "perturbation_category"))

    def heatmap(ax, frame, title, row_key):
        pivot = frame.pivot_table(index=row_key, columns="denoising_steps", values="success", aggfunc="mean")
        image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set(title=title, xlabel="Denoising steps", ylabel=row_key)
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=6)
        plt.colorbar(image, ax=ax, label="Success rate")

    save_plot("04_pro_perturbation_steps_success_heatmap", lambda ax: heatmap(ax, pro, "PRO category × step", "perturbation_category"))

    def metric_line(ax, frame, metric, title, ylabel, group=None):
        if group:
            for label, selected in frame.groupby(group):
                values = selected.groupby("denoising_steps")[metric].mean()
                ax.plot(values.index, values, marker="o", label=str(label))
            ax.legend(fontsize=8)
        else:
            values = frame.groupby("denoising_steps")[metric].mean()
            ax.plot(values.index, values, marker="o")
        ax.set(title=title, xlabel="Denoising steps", ylabel=ylabel, xticks=STEPS)
        ax.set_xticklabels(["1", "2", "3", "4", "5\nNative", "6\nUpper"])
        ax.axvline(5, color="black", linestyle="--", alpha=0.35, label="Native baseline")
        ax.axvline(6, color="gray", linestyle=":", alpha=0.35, label="Upper-control")
        ax.grid(alpha=0.25)

    def matched_gap(ax):
        for category, group in pro.groupby("perturbation_category"):
            values = []
            for step, selected in group.groupby("denoising_steps"):
                task_ids = set(selected["base_task_id"])
                original_rate = original[
                    (original["denoising_steps"] == step)
                    & original["base_task_id"].isin(task_ids)
                ]["success"].mean()
                values.append((step, original_rate - selected["success"].mean()))
            ax.plot(*zip(*values), marker="o", label=category)
        ax.set(title="Matched robustness gap", xlabel="Denoising steps", ylabel="SR(original) − SR(PRO)", xticks=STEPS)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    save_plot("05_matched_robustness_gap", matched_gap)

    def penalty(ax):
        for category, group in pro.groupby("perturbation_category"):
            rates = group.groupby("denoising_steps")["success"].mean()
            ax.plot(rates.index, rates - rates.get(5, np.nan), marker="o", label=category)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(title="Success difference from native 5-step", xlabel="Denoising steps", ylabel="SR(step) − SR(5)", xticks=STEPS)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    save_plot("06_relative_success_drop_vs_native5", penalty)
    save_plot("07_policy_latency_vs_steps", lambda ax: metric_line(ax, requests, "total_policy_request_latency_ms", "Policy request latency", "ms", "domain"))
    save_plot("08_dit_latency_vs_steps", lambda ax: metric_line(ax, requests, "dit_denoising_latency_ms", "DiT denoising latency", "ms", "domain"))
    save_plot("09_episode_time_vs_steps", lambda ax: metric_line(ax, episodes, "episode_wall_clock_time_s", "Episode completion time", "Seconds", "domain"))

    def pareto(ax):
        values = original.groupby("denoising_steps").agg(success=("success", "mean"))
        latency = (
            requests[requests["domain"] == "libero"]
            .groupby("denoising_steps")["total_policy_request_latency_ms"]
            .mean()
            .rename("latency")
        )
        values = values.join(latency)
        ax.scatter(values["latency"], values["success"])
        for step, row in values.iterrows():
            ax.annotate(str(step), (row["latency"], row["success"]))
        ax.set(title="Original success–latency Pareto", xlabel="Mean policy latency (ms)", ylabel="Success rate", ylim=(0, 1.02))
        ax.grid(alpha=0.25)

    save_plot("10_success_latency_pareto", pareto)
    combined = episodes.copy()
    combined["task_variant"] = (
        combined["perturbation_category"].astype(str) + ":" + combined["task_uid"].astype(str)
    )
    save_plot(
        "11_task_level_success_heatmap",
        lambda ax: heatmap(ax, combined, "Task/variant × step", "task_variant"),
        figsize=(10, 24),
    )

    def disagreements(ax):
        if paired.empty:
            ax.text(0.5, 0.5, "No paired data", ha="center")
            return
        pivot = paired.pivot_table(
            index=["domain", "perturbation_category"],
            columns="step",
            values="step_failure_reference_success",
            aggfunc="sum",
        ).fillna(0)
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma")
        labels = [f"{a}/{b}" for a, b in pivot.index]
        ax.set(title="Paired disagreements: step fails, native 5 succeeds", xlabel="Compared step", ylabel="Domain/category")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        plt.colorbar(image, ax=ax, label="Discordant pairs")

    save_plot("12_paired_disagreement_matrix", disagreements)
    save_plot("13_completion_steps", lambda ax: metric_line(ax, episodes[episodes["success"].astype(bool)], "episode_steps", "Successful completion steps", "Control steps", "domain"))

    def smoothness_jerk(ax):
        smooth = episodes.groupby("denoising_steps")["action_smoothness"].mean()
        jerk = episodes.groupby("denoising_steps")["action_jerk"].mean()
        ax.plot(smooth.index, smooth, marker="o", label="Smoothness")
        ax.plot(jerk.index, jerk, marker="s", label="Jerk")
        ax.set(title="Action smoothness and jerk (descriptive only)", xlabel="Denoising steps", ylabel="Mean metric", xticks=STEPS)
        ax.legend()
        ax.grid(alpha=0.25)

    save_plot("14_action_smoothness_and_jerk", smoothness_jerk)

    def invalid_env(ax):
        counts = (
            episodes_all.assign(
                invalid=(
                    ~episodes_all["environment_valid"].astype(bool)
                    | episodes_all["termination_reason"].astype(str).str.startswith(
                        ("validation_error:", "fatal:", "error:")
                    )
                )
            )
            .groupby(["perturbation_category", "denoising_steps"])["invalid"]
            .sum()
            .unstack(fill_value=0)
        )
        counts.plot(kind="bar", ax=ax)
        ax.set(title="Invalid environment episodes", xlabel="Perturbation", ylabel="Count")
        ax.legend(title="Steps", fontsize=7)

    save_plot("15_invalid_environment_count", invalid_env)

    def forward_audit(ax):
        actual = requests.groupby("selected_denoising_steps")["denoiser_forward_count"].mean()
        ax.plot(STEPS, STEPS, linestyle="--", color="black", label="Expected")
        ax.scatter(actual.index, actual.values, label="Observed mean")
        ax.set(title="Denoiser forward-count audit", xlabel="Selected steps", ylabel="Actual complete forwards", xticks=STEPS, yticks=STEPS)
        ax.legend()
        ax.grid(alpha=0.25)

    save_plot("16_denoiser_forward_count_audit", forward_audit)


def choose_stage2(
    episodes: pd.DataFrame, requests: pd.DataFrame
) -> tuple[list[int], dict[str, dict]]:
    original = episodes[episodes["domain"] == "libero"]
    table = original.groupby("denoising_steps").agg(success=("success", "mean"))
    latency = (
        requests[requests["domain"] == "libero"]
        .groupby("denoising_steps")["total_policy_request_latency_ms"]
        .mean()
        .rename("latency")
    )
    table = table.join(latency)
    available = sorted(int(step) for step in table.index)
    pareto = []
    for step in available:
        row = table.loc[step]
        dominated = any(
            other != step
            and table.loc[other, "success"] >= row["success"]
            and table.loc[other, "latency"] <= row["latency"]
            and (
                table.loc[other, "success"] > row["success"]
                or table.loc[other, "latency"] < row["latency"]
            )
            for other in available
        )
        if not dominated:
            pareto.append(step)
    selected = [step for step in (1, 3, 5, 6) if step in available]
    details = {
        str(step): {
            "retained": step in selected,
            "reason": {
                1: "默认保留：最大加速下限。",
                3: "默认保留：低计算量的中间候选。",
                5: "默认保留：Cosmos 原生 baseline。",
                6: "默认保留：本阶段高计算上限。",
            }.get(step, "待依据第一轮数据判定。"),
        }
        for step in available
    }
    pro = episodes[episodes["domain"] == "libero_pro"]
    for step in (2, 4):
        if step not in available:
            details[str(step)] = {
                "retained": False,
                "reason": "第一轮数据缺失，不能进入第二轮。",
            }
            continue
        neighbors = (step - 1, step + 1)
        expected_success = float(table.loc[list(neighbors), "success"].mean())
        nonmonotonic = abs(float(table.loc[step, "success"]) - expected_success) >= 0.10
        unique_categories = []
        for category, group in pro.groupby("perturbation_category"):
            rates = group.groupby("denoising_steps")["success"].mean()
            if all(value in rates for value in (*neighbors, step)):
                neighbor_mean = float(rates.loc[list(neighbors)].mean())
                if abs(float(rates.loc[step]) - neighbor_mean) >= 0.15:
                    unique_categories.append(str(category))
        criteria = {
            "pareto_frontier": step in pareto,
            "nonmonotonic_vs_neighbors_ge_0.10": nonmonotonic,
            "unique_perturbation_categories_ge_0.15": unique_categories,
        }
        keep = bool(criteria["pareto_frontier"] or nonmonotonic or unique_categories)
        details[str(step)] = {
            "retained": keep,
            "reason": (
                "满足至少一个预注册保留条件。"
                if keep
                else "未位于 Pareto frontier，未出现明显非单调或扰动类别独特表现。"
            ),
            "criteria": criteria,
            "original_success": float(table.loc[step, "success"]),
            "neighbor_mean_success": expected_success,
            "original_mean_policy_latency_ms": float(table.loc[step, "latency"]),
        }
        if keep:
            selected.append(step)
    return sorted(set(selected)), details


def stage2_manifest(episodes: pd.DataFrame, selected: list[int]) -> None:
    # Lowest mean success tasks are the predeclared hard subset. Repeats use
    # independent deterministic seeds while preserving init index 0.
    task_scores = episodes.groupby(["domain", "perturbation_category", "task_uid"])["success"].mean()
    hard = task_scores.sort_values().head(20).reset_index()
    full_rows = []
    for path in (
        EXPERIMENT / "manifests/original_full.jsonl",
        EXPERIMENT / "manifests/libero_pro_full.jsonl",
    ):
        full_rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
    hard_keys = set(map(tuple, hard[["domain", "perturbation_category", "task_uid"]].to_records(index=False)))
    output = []
    for row in full_rows:
        key = (row["domain"], row["perturbation_category"], row["task_uid"])
        if key not in hard_keys or int(row["denoising_steps"]) not in selected:
            continue
        for seed in (195, 196, 197):
            clone = dict(row)
            clone["seed"] = seed
            clone["variant_id"] = row["variant_id"] + ":hard_repeat"
            identity = (
                f"{clone['config_id']}|{clone['task_uid']}|{clone['variant_id']}|"
                f"{seed}|{clone['init_state_index']}"
            )
            clone["episode_key"] = hashlib.sha256(identity.encode()).hexdigest()
            clone["config_id"] = clone["config_id"] + "_hard_repeat"
            clone["stage"] = "data_driven_hard_subset"
            output.append(clone)
    path = EXPERIMENT / "manifests/stage2_hard_subset.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output),
        encoding="utf-8",
    )
    hard.to_csv(EXPERIMENT / "summaries/hard_subset.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()
    episode_records = read_jsonl_tree("raw/*/episodes.shard*.jsonl")
    request_records = read_jsonl_tree("raw/*/requests.shard*.jsonl")
    if not episode_records:
        raise SystemExit("no episode records found")
    episodes = pd.DataFrame(episode_records)
    requests = pd.DataFrame(request_records)
    episodes_all = (
        episodes.sort_values("completed_at_ns")
        .drop_duplicates("episode_key", keep="last")
        .reset_index(drop=True)
    )
    episodes_all = episodes_all[
        ~episodes_all["config_id"].astype(str).str.contains(
            "stage2|video_replay", regex=True
        )
    ].reset_index(drop=True)
    invalid_mask = (
        ~episodes_all["environment_valid"].astype(bool)
        | episodes_all["termination_reason"].astype(str).str.startswith(
            ("validation_error:", "fatal:", "error:")
        )
    )
    invalid_episodes = episodes_all[invalid_mask].copy()
    episodes = episodes_all[~invalid_mask].copy()
    valid_episode_keys = set(episodes["episode_key"])
    if not requests.empty:
        latest_runs = episodes.set_index("episode_key")["run_id"].to_dict()
        requests = requests[
            requests["episode_key"].isin(valid_episode_keys)
            & requests.apply(
                lambda row: row.get("run_id") == latest_runs.get(row["episode_key"]),
                axis=1,
            )
        ].copy()
    if not requests.empty:
        requests = requests.drop_duplicates(["episode_key", "request_id"], keep="last")
        requests["action_chunk_l2_norm"] = pd.to_numeric(requests.get("action_chunk_l2_norm"), errors="coerce")
        requests = requests.sort_values(["run_id", "inference_start_ns"]).reset_index(drop=True)
        requests["warmup_request_index"] = requests.groupby("run_id").cumcount()
        requests["warmup_excluded"] = requests["warmup_request_index"] < 10
    measured_requests = requests[~requests["warmup_excluded"]].copy() if not requests.empty else requests
    contended_latency_requests_excluded = 0
    if not measured_requests.empty and "physical_gpu_id" in measured_requests:
        physical = pd.to_numeric(measured_requests["physical_gpu_id"], errors="coerce")
        clean_mask = physical.isna() | (physical >= 4)
        contended_latency_requests_excluded = int((~clean_mask).sum())
        measured_requests = measured_requests[clean_mask].copy()
    summaries = EXPERIMENT / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    episodes_all.to_csv(summaries / "episodes_all_deduplicated.csv", index=False)
    invalid_episodes.to_csv(summaries / "episodes_excluded_invalid.csv", index=False)
    episodes.to_csv(summaries / "episodes_deduplicated.csv", index=False)
    requests.to_csv(summaries / "requests_deduplicated.csv", index=False)
    aggregate_levels = []
    all_with_invalid = episodes_all.assign(
        invalid=(
            ~episodes_all["environment_valid"].astype(bool)
            | episodes_all["termination_reason"].astype(str).str.startswith(
                ("validation_error:", "fatal:", "error:")
            )
        )
    )
    for keys, fixed in (
        (
            ["domain", "perturbation_category", "suite", "denoising_steps"],
            {},
        ),
        (
            ["domain", "perturbation_category", "denoising_steps"],
            {"suite": "__all__"},
        ),
        (
            ["domain", "denoising_steps"],
            {"perturbation_category": "__all__", "suite": "__all__"},
        ),
    ):
        level = aggregate(episodes, measured_requests, keys)
        invalid = (
            all_with_invalid.groupby(keys, dropna=False)["invalid"]
            .mean()
            .rename("invalid_environment_rate")
            .reset_index()
        )
        for column, value in fixed.items():
            level[column] = value
            invalid[column] = value
        merge_keys = ["domain", "perturbation_category", "suite", "denoising_steps"]
        level = level.drop(columns=["invalid_environment_rate"]).merge(
            invalid,
            on=merge_keys,
            how="left",
        )
        aggregate_levels.append(level)
    aggregate_table = pd.concat(aggregate_levels, ignore_index=True)
    aggregate_table.to_csv(summaries / "aggregate_metrics.csv", index=False)
    paired = paired_tests(episodes, iterations=args.bootstrap_iterations)
    paired.to_csv(summaries / "paired_mcnemar_bootstrap.csv", index=False)
    interactions = interaction_analysis(
        episodes, iterations=args.bootstrap_iterations
    )
    interactions.to_csv(summaries / "interaction_analysis.csv", index=False)
    robustness = robustness_metrics(episodes)
    robustness.to_csv(summaries / "robustness_metrics.csv", index=False)
    selected, selection_details = choose_stage2(episodes, measured_requests)
    (summaries / "second_stage_selection.json").write_text(
        json.dumps(
            {
                "selected_steps": selected,
                "selection_rule": (
                    "默认保留 1/3/5/6；2/4 仅在 Pareto、明显非单调或扰动类别"
                    "独特表现条件至少一项成立时保留。"
                ),
                "steps": selection_details,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    generate_plots(episodes, measured_requests, paired, interactions, episodes_all)
    result = {
        "status": "complete" if len(episodes_all) == 1410 else "interim",
        "episodes": len(episodes),
        "requests": len(requests),
        "warmup_requests_excluded": int(requests["warmup_excluded"].sum()) if not requests.empty else 0,
        "measured_requests": len(measured_requests),
        "contended_gpu_latency_requests_excluded": contended_latency_requests_excluded,
        "successes": int(episodes["success"].astype(bool).sum()),
        "invalid_episodes_excluded": len(invalid_episodes),
        "selected_stage2_steps": selected,
        "expected_main_episodes": 1410,
        "plot_pairs": 16,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    (summaries / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

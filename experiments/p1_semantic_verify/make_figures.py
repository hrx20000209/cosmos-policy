#!/usr/bin/env python3
"""E12 figures. Only the figures whose gates were actually reached are produced.

Figures 1-3 are E12-A (discovery). Figure D is the route-variant diagnostic.
Figures 4-8 belong to E12-B/E12-C, whose gates were not reached; they are
deliberately absent rather than produced from unauthorized data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.p1_semantic_verify.analyze_signal import leave_one_task_out

PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280", "#00747a", "#975a16", "#4a5568"]


def short(task: str) -> str:
    suite, name = task.split(":", 1)
    return f"{suite.replace('libero_', '')}:{name[:26]}"


def figure1(frame: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), sharey=False)
    for axis, (task, group), colour in zip(axes.ravel(), frame.groupby("task_id"), PALETTE):
        axis.scatter(group.s_p1, group.risk_p1_full, s=34, color=colour, alpha=0.85, edgecolor="white")
        rho = spearmanr(group.s_p1, group.risk_p1_full).statistic
        axis.set_title(f"{short(task)}\nrho = {rho:+.3f}", fontsize=9)
        axis.set_xlabel("frozen $S_{P1}$", fontsize=9)
        axis.set_ylabel("$R_{P1}=D(A_{P1},A_{F1})$", fontsize=9)
        axis.grid(alpha=0.25)
    overall = np.mean([spearmanr(g.s_p1, g.risk_p1_full).statistic for _, g in frame.groupby("task_id")])
    figure.suptitle(
        f"Figure 1  Frozen P1-native semantic score vs its own action risk — E12-A discovery, "
        f"task-balanced rho = {overall:+.3f} (gate 0.50)", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure2(frame: pd.DataFrame, path: Path) -> None:
    groups = {"low S (bottom 1/3)": [], "mid S": [], "high S (top 1/3)": []}
    for _, task_rows in frame.groupby("task_id"):
        ordered = task_rows.sort_values("s_p1")
        thirds = np.array_split(ordered, 3)
        for name, part in zip(groups, thirds):
            groups[name].extend(part.risk_p1_full.tolist())
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    left.boxplot(list(groups.values()), tick_labels=list(groups), showfliers=False)
    for index, values in enumerate(groups.values(), start=1):
        left.scatter(np.random.default_rng(0).normal(index, 0.05, len(values)), values,
                     s=14, alpha=0.45, color=PALETTE[0])
    left.set_ylabel("$R_{P1}$ (full-chunk mean-step L2)")
    left.set_title("P1 action error by within-task $S_{P1}$ tercile")
    left.grid(alpha=0.25)
    means = [np.mean(v) for v in groups.values()]
    right.bar(list(groups), means, color=PALETTE[:3])
    for index, value in enumerate(means):
        right.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    right.set_ylabel("mean $R_{P1}$")
    right.set_title("Mean P1 action error per tercile (separation is the mechanism claim)")
    right.grid(alpha=0.25, axis="y")
    figure.suptitle("Figure 2  Does a high frozen semantic score mark a dangerous speculative P1 action? — discovery")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure3(frame: pd.DataFrame, path: Path) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    for (task, group), colour in zip(frame.groupby("task_id"), PALETTE):
        left.scatter(group.s_p1, group.pv0_gain, s=30, color=colour, alpha=0.8,
                     edgecolor="white", label=short(task))
    rho = np.mean([spearmanr(g.s_p1, g.pv0_gain).statistic for _, g in frame.groupby("task_id")])
    left.set_xlabel("frozen $S_{P1}$")
    left.set_ylabel("$G_{PV0}=R_{P1}-R_{PV0}$")
    left.set_title(f"Correction utility vs score, task-balanced rho = {rho:+.3f}")
    left.grid(alpha=0.25)
    left.legend(fontsize=6, loc="upper left")

    right.scatter(frame.risk_p1_full, frame.risk_pv0_full, s=30, color=PALETTE[2], alpha=0.8, edgecolor="white")
    limit = float(frame.risk_p1_full.max()) * 1.05
    right.plot([0, limit], [0, limit], color="#718096", linestyle="--", linewidth=1, label="no correction")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("$R_{P1}$")
    right.set_ylabel("$R_{PV0}$")
    right.set_title(f"PV0 recovers P1 on all {len(frame)} states\n"
                    f"median relative recovery {frame.pv0_relative_recovery.median():.1%}")
    right.grid(alpha=0.25, which="both")
    right.legend(fontsize=8)
    figure.suptitle("Figure 3  PV0 correction utility on the new discovery tasks")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure_diagnostic(frame: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(frame.s_p1_e4variant, frame.s_p1, s=30, color=PALETTE[3], alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("$S$ on the E4/E11-A collection route (raw prior latent)")
    axes[0].set_ylabel("$S$ on the deployed P1 route")
    axes[0].set_title(f"Scores largely agree\nSpearman {spearmanr(frame.s_p1_e4variant, frame.s_p1).statistic:+.3f}")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(frame.risk_p1_e4variant_full, frame.risk_p1_full, s=30, color=PALETTE[1],
                    alpha=0.8, edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("$R$ of the E4 collection route")
    axes[1].set_ylabel("$R$ of the deployed P1 route")
    axes[1].set_title("but the RISKS they rank are different objects\n"
                      f"Spearman {spearmanr(frame.risk_p1_e4variant_full, frame.risk_p1_full).statistic:+.3f}")
    axes[1].grid(alpha=0.25, which="both")

    pairs = {
        "E4 route score\nvs E4 route risk": np.mean(
            [spearmanr(g.s_p1_e4variant, g.risk_p1_e4variant_full).statistic for _, g in frame.groupby("task_id")]),
        "deployed P1 score\nvs deployed P1 risk": np.mean(
            [spearmanr(g.s_p1, g.risk_p1_full).statistic for _, g in frame.groupby("task_id")]),
        "action-only (LOTO)\nvs deployed P1 risk": np.mean(
            [spearmanr(g.action_only, g.risk_p1_full).statistic for _, g in frame.groupby("task_id")]),
    }
    axes[2].bar(list(pairs), list(pairs.values()), color=[PALETTE[3], PALETTE[0], PALETTE[2]])
    axes[2].axhline(0.50, color="#c53030", linestyle="--", linewidth=1.2, label="preregistered gate 0.50")
    for index, value in enumerate(pairs.values()):
        axes[2].text(index, value, f"{value:+.3f}", ha="center", va="bottom", fontsize=10)
    axes[2].set_ylabel("task-balanced Spearman")
    axes[2].set_title("What the frozen score actually predicts")
    axes[2].tick_params(axis="x", labelsize=8)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25, axis="y")
    figure.suptitle("Figure D  Route-variant diagnostic: the frozen E4 score tracks a reuse route "
                    "that is not the deployed P1 (discovery only)")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path,
                        default=Path("artifacts/p1_semantic_verify/e12_shadow_discovery.parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/p1_semantic_verify/plots"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.parquet).reset_index(drop=True)
    frame["action_only"] = leave_one_task_out(frame)
    figure1(frame, args.out_dir / "figure1_score_vs_p1_risk.png")
    figure2(frame, args.out_dir / "figure2_risk_by_score_tercile.png")
    figure3(frame, args.out_dir / "figure3_pv0_correction_utility.png")
    figure_diagnostic(frame, args.out_dir / "figureD_route_variant_diagnostic.png")
    print(json.dumps({"figures": sorted(p.name for p in args.out_dir.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()

"""Aggregate mechanism-discovery artifacts into figures and a concise report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

EXPECTED_SHA = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def load(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact["checkpoint_sha256"] != EXPECTED_SHA:
        raise ValueError(f"unexpected checkpoint hash in {path}: {artifact['checkpoint_sha256']}")
    if artifact.get("value_used") is not False:
        raise ValueError(f"artifact does not explicitly exclude value use: {path}")
    return artifact


def quantile(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(array)), float(np.quantile(array, 0.25)), float(np.quantile(array, 0.75))


def repair_table(states: list[dict[str, Any]], blocks: list[int], names: list[str]) -> dict[int, dict[str, tuple]]:
    return {
        block: {
            name: quantile([state["repairs"][str(block)][name]["recovery_ratio"] for state in states]) for name in names
        }
        for block in blocks
    }


def plot_repair(table: dict[int, dict[str, tuple]], output: Path) -> None:
    blocks = list(table)
    labels = {
        "current_visual": "visual",
        "visual_plus_proprio": "visual+proprio",
        "action_only": "action",
        "visual_plus_action": "visual+action",
        "all_current_observation": "all current obs",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    for name, label in labels.items():
        median = np.asarray([table[block][name][0] for block in blocks])
        q25 = np.asarray([table[block][name][1] for block in blocks])
        q75 = np.asarray([table[block][name][2] for block in blocks])
        axes[0].plot(blocks, median, marker="o", label=label)
        axes[0].fill_between(blocks, q25, q75, alpha=0.12)
        remaining = np.asarray([(27 - block) / 28 for block in blocks])
        axes[1].plot(remaining, median, marker="o", label=label)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set(xlabel="patch after DiT block", ylabel="action recovery ratio", title="Repair depth frontier")
    axes[1].set(
        xlabel="remaining DiT block FLOPs fraction",
        ylabel="action recovery ratio",
        title="Recovery versus suffix compute",
    )
    axes[1].invert_xaxis()
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_causal(causal: dict[str, Any], output: Path) -> None:
    states = causal["states"]
    blocks = causal["block_ids"]
    interventions = [
        "predicted_visual_fresh_proprio",
        "cached_visual_fresh_proprio",
        "fresh_visual_stale_proprio",
        "fresh_primary_predicted_wrist",
        "predicted_primary_fresh_wrist",
    ]
    labels = ["pred visual", "cached visual", "stale proprio", "pred wrist", "pred primary"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for intervention, label in zip(interventions, labels, strict=True):
        median = []
        for position, _ in enumerate(blocks):
            median.append(
                np.median(
                    [state["interventions"][intervention]["blocks"][position]["delta_action"] for state in states]
                )
            )
        ax.plot(blocks, median, marker="o", label=label)
    ax.set(
        xlabel="DiT block", ylabel="median action-slot hidden L2 change", title="Causal propagation into action slot"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_low_rank(low_rank: dict[str, Any], output: Path) -> None:
    blocks = low_rank["blocks"]
    ranks = low_rank["ranks"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for block in blocks:
        row = low_rank["summary"][str(block)]
        axes[0].plot(
            ranks,
            [row["combined_visual_energy_median"][str(rank)] for rank in ranks],
            marker="o",
            label=f"block {block}",
        )
        axes[1].plot(
            ranks,
            [row["action_recovery_median"][f"rank_{rank}"] for rank in ranks],
            marker="o",
            label=f"block {block}",
        )
    axes[0].set(xlabel="rank", ylabel="correction energy explained", title="Visual correction spectrum")
    axes[1].set(xlabel="rank", ylabel="action recovery ratio", title="Projected correction recovery")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_branching(branching: dict[str, Any], output: Path) -> None:
    states = branching["states"]
    score = np.asarray([state["branching"]["mean_pairwise_action_mean_step_l2"] for state in states], dtype=np.float64)
    value = np.asarray([state["oracle_fresh_sensing_value"] for state in states], dtype=np.float64)
    tasks = np.asarray([state["task_id"] for state in states])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    scatter = ax.scatter(score, value, c=tasks, cmap="tab10", alpha=0.8, s=28)
    ax.set(
        xlabel="fixed-action future branching score",
        ylabel="oracle fresh-sensing action value",
        title="Control-relevant branching versus sensing value",
    )
    fig.colorbar(scatter, ax=ax, label="LIBERO task id")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("reports/artifacts"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--output", type=Path, default=Path("reports/MECHANISM_DISCOVERY_RESULTS.md"))
    args = parser.parse_args()

    causal = load(args.artifact_dir / "cosmos_causal_internal_influence.json")
    repair0 = load(args.artifact_dir / "cosmos_activation_repair_frontier.json")
    repair1 = load(args.artifact_dir / "cosmos_activation_repair_frontier_init1.json")
    low_rank = load(args.artifact_dir / "cosmos_low_rank_visual_repair.json")
    branching = load(args.artifact_dir / "cosmos_control_relevant_branching.json")
    repair_states = repair0["states"] + repair1["states"]
    patch_names = list(repair0["patch_groups"])
    repair = repair_table(repair_states, repair0["patch_blocks"], patch_names)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plot_causal(causal, args.figure_dir / "causal_action_propagation.png")
    plot_repair(repair, args.figure_dir / "activation_repair_frontier.png")
    plot_low_rank(low_rank, args.figure_dir / "low_rank_repair.png")
    plot_branching(branching, args.figure_dir / "branching_vs_sensing_value.png")

    lines = [
        "# Cosmos WAM Mechanism Discovery Results",
        "",
        "## Scope and controls",
        "",
        f"All experiments use the original pre-finetune Cosmos LIBERO checkpoint `{EXPECTED_SHA}` and one denoising step. "
        "No SO101 checkpoint, Cosmos value output as an experimental signal, privileged simulator state, fine-tuning, "
        "or adaptive scheduler is used.",
        "",
        "## 1. Causal influence map",
        "",
        f"Collected {causal['state_count']} executed states over {len(set(s['task_id'] for s in causal['states']))} tasks and "
        f"{len(set(s['init_index'] for s in causal['states']))} initial-state indices.",
        "",
    ]
    causal_rows = []
    for name in causal["summary"]:
        output_values = [state["interventions"][name]["output_action_mean_step_l2"] for state in causal["states"]]
        first_action = [state["interventions"][name]["blocks"][0]["delta_action"] for state in causal["states"]]
        last_action = [state["interventions"][name]["blocks"][-1]["delta_action"] for state in causal["states"]]
        causal_rows.append(
            [
                name,
                f"{np.median(output_values):.4f}",
                f"{np.median(first_action):.3f}",
                f"{np.median(last_action):.3f}",
            ]
        )
    lines.extend(
        markdown_table(
            ["intervention", "output action Δ", "block 0 action-slot Δ", "block 27 action-slot Δ"], causal_rows
        )
    )
    lines.extend(
        ["", "![Causal propagation](figures/causal_action_propagation.png)", "", "## 2. Oracle repair frontier", ""]
    )
    repair_rows = []
    for block in repair0["patch_blocks"]:
        repair_rows.append(
            [
                str(block),
                f"{(27 - block) / 28:.3f}",
                *(f"{repair[block][name][0]:.3f}" for name in patch_names),
            ]
        )
    lines.extend(
        markdown_table(
            ["block", "remaining FLOPs", "visual", "visual+proprio", "action", "visual+action", "all current obs"],
            repair_rows,
        )
    )
    lines.extend(
        [
            "",
            f"Across {len(repair_states)} states, 8 tasks, and two initial-state indices, early visual repair is strong "
            f"(block 4 median {repair[4]['current_visual'][0]:.1%}) but collapses by block 24 "
            f"({repair[24]['current_visual'][0]:.1%}). The complementary action-slot repair rises to "
            f"{repair[24]['action_only'][0]:.1%} at block 24. Fresh visual information is therefore progressively "
            "compiled into the action representation; it cannot generally be restored by a late visual-slot reset.",
            "",
            "![Repair frontier](figures/activation_repair_frontier.png)",
            "",
            "## 3. Low-rank correction test",
            "",
        ]
    )
    rank_rows = []
    for block in low_rank["blocks"]:
        row = low_rank["summary"][str(block)]
        rank_rows.append(
            [
                str(block),
                f"{row['combined_visual_energy_median']['32']:.3f}",
                f"{row['action_recovery_median']['rank_32']:.3f}",
                f"{row['action_recovery_median']['full_visual']:.3f}",
            ]
        )
    lines.extend(markdown_table(["block", "rank-32 energy", "rank-32 recovery", "exact visual recovery"], rank_rows))
    rank32_ratio = np.median(
        [
            low_rank["summary"][str(block)]["action_recovery_median"]["rank_32"]
            / max(low_rank["summary"][str(block)]["action_recovery_median"]["full_visual"], 1e-8)
            for block in low_rank["blocks"]
        ]
    )
    lines.extend(
        [
            "",
            f"Rank 32 retains a median {rank32_ratio:.1%} of exact visual-patch action recovery across tested blocks. "
            "The leading correction-energy directions are not sufficient proxies for the control-relevant correction; "
            "generic PCA/SVD correction is a NO-GO unless a control-aware basis is justified separately.",
            "",
            "![Low-rank repair](figures/low_rank_repair.png)",
            "",
            "## 4. Fixed-action future branching",
            "",
        ]
    )
    corr_all = branching["correlations_all"]
    corr_heldout = branching["correlations_heldout"]
    correlation_rows = []
    for name, values in corr_all.items():
        if name == "n":
            continue
        heldout = corr_heldout.get(name, {})
        correlation_rows.append(
            [name, f"{values['spearman_rho']:.3f}", f"{heldout.get('spearman_rho', float('nan')):.3f}"]
        )
    lines.extend(markdown_table(["signal", "Spearman all", "Spearman held-out tasks"], correlation_rows))
    fixed_error = max(max(state["fixed_action_max_abs_errors"]) for state in branching["states"])
    branching_all_rho = corr_all["control_relevant_branching"]["spearman_rho"]
    branching_discovery_rho = branching["correlations_discovery"]["control_relevant_branching"]["spearman_rho"]
    branching_rho = corr_heldout["control_relevant_branching"]["spearman_rho"]
    status = (
        "GO-CANDIDATE"
        if min(branching_all_rho, branching_discovery_rho, branching_rho) >= 0.3
        else "NO-GO/INCONCLUSIVE"
    )
    lines.extend(
        [
            "",
            f"The fixed-action control is exact to max absolute action error {fixed_error:.3g}. Held-out branching "
            f"correlation is {branching_rho:.3f}, but all-state/discovery correlations are "
            f"{branching_all_rho:.3f}/{branching_discovery_rho:.3f}, giving `{status}` under the consistency "
            "criterion that a signal must remain materially positive across discovery and task-disjoint splits "
            "before any scheduler work.",
            "",
            "![Branching versus sensing value](figures/branching_vs_sensing_value.png)",
            "",
            "## Decision",
            "",
            "The strongest supported mechanism is a depth-dependent transfer of execution feedback from visual slots "
            "into action state. The actionable systems question is not generic latent reuse or late feature patching, "
            "but where a fresh-sensing correction must enter before control information becomes irreversible. No adaptive "
            "scheduler or closed-loop method should be implemented from these diagnostics alone.",
            "",
            "Raw and aggregated JSON artifacts are in `reports/artifacts/`; incremental state records are under "
            "`/data/rxhuang/wam_libero_outputs/`.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

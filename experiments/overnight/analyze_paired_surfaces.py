"""Aggregate paired fresh/predicted denoise×block surfaces and interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


BLOCKS = (0, 4, 8, 12, 16, 20, 24, 27)
SLOTS = (
    "current_proprio",
    "current_wrist",
    "current_primary",
    "action",
    "future_proprio",
    "future_wrist",
    "future_primary",
)


def action_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=-1)))


def stats(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-dir", action="append", nargs=2, metavar=("SPLIT", "DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    hidden_rows = []
    stage_rows = []
    variant_rows = []
    horizon_rows = []
    state_rows = []
    manifests = {}
    for split, directory_string in args.surface_dir:
        directory = Path(directory_string)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests[split] = manifest
        summaries = [json.loads(line) for line in (directory / "states.jsonl").read_text().splitlines() if line]
        for summary in summaries:
            record = torch.load(summary["raw_state"], weights_only=False, map_location="cpu")
            input_visual_l1 = float(summary["fresh_vs_predicted_visual_condition_l1"])
            state_action_l2 = float(summary["fresh_vs_predicted_action_l2"])
            target = np.asarray(record["target_visual_latent"], dtype=np.float32)
            for steps, fresh_schedule in record["schedules"].items():
                predicted_schedule = record["predicted_visual_schedules"][steps]
                fresh_actions = np.asarray(fresh_schedule["actions"], dtype=np.float32)
                predicted_actions = np.asarray(predicted_schedule["actions"], dtype=np.float32)
                fresh_future = np.asarray(fresh_schedule["future_latents"], dtype=np.float32)
                predicted_future = np.asarray(predicted_schedule["future_latents"], dtype=np.float32)
                fresh_hidden = np.asarray(fresh_schedule["compact_hidden"], dtype=np.float32)
                predicted_hidden = np.asarray(predicted_schedule["compact_hidden"], dtype=np.float32)
                for stage in range(int(steps)):
                    stage_rows.append(
                        {
                            "split": split,
                            "state_index": summary["state_index"],
                            "task_suite": summary["task_suite"],
                            "task_id": summary["task_id"],
                            "offline_stage": summary.get("offline_stage"),
                            "steps": int(steps),
                            "stage": stage + 1,
                            "fresh_predicted_action_l2": action_distance(
                                fresh_actions[stage], predicted_actions[stage]
                            ),
                            "fresh_predicted_future_l1": float(
                                np.mean(np.abs(fresh_future[stage] - predicted_future[stage]))
                            ),
                            "fresh_visual_l1_to_real_next": float(
                                np.mean(np.abs(fresh_future[stage, :, 1:3] - target))
                            ),
                            "predicted_visual_l1_to_real_next": float(
                                np.mean(np.abs(predicted_future[stage, :, 1:3] - target))
                            ),
                        }
                    )
                    delta = fresh_hidden[stage] - predicted_hidden[stage]
                    delta_norm = np.linalg.norm(delta, axis=-1)
                    for block_index, block in enumerate(BLOCKS):
                        for slot_index, slot in enumerate(SLOTS):
                            hidden_rows.append(
                                {
                                    "split": split,
                                    "state_index": summary["state_index"],
                                    "task_suite": summary["task_suite"],
                                    "task_id": summary["task_id"],
                                    "steps": int(steps),
                                    "stage": stage + 1,
                                    "block": block,
                                    "slot": slot,
                                    "hidden_delta_l2": float(delta_norm[block_index, slot_index]),
                                    "normalized_influence": float(
                                        delta_norm[block_index, slot_index] / max(input_visual_l1, 1e-8)
                                    ),
                                    "final_action_l2": state_action_l2,
                                }
                            )
                for token in range(16):
                    horizon_rows.append(
                        {
                            "split": split,
                            "state_index": summary["state_index"],
                            "steps": int(steps),
                            "token": token,
                            "fresh_predicted_action_l2": float(
                                np.linalg.norm(fresh_actions[-1, token] - predicted_actions[-1, token])
                            ),
                        }
                    )

            baseline_action = np.asarray(record["schedules"][1]["actions"][-1], dtype=np.float32)
            baseline_hidden = np.asarray(record["schedules"][1]["compact_hidden"][0], dtype=np.float32)
            for variant, values in record["input_variants"].items():
                variant_action = np.asarray(values["action"], dtype=np.float32)
                variant_hidden = np.asarray(values["compact_hidden"], dtype=np.float32)
                for block_index, block in enumerate(BLOCKS):
                    for slot_index, slot in enumerate(SLOTS):
                        variant_rows.append(
                            {
                                "split": split,
                                "state_index": summary["state_index"],
                                "variant": variant,
                                "block": block,
                                "slot": slot,
                                "hidden_delta_l2": float(
                                    np.linalg.norm(variant_hidden[block_index, slot_index] - baseline_hidden[block_index, slot_index])
                                ),
                                "action_l2": action_distance(variant_action, baseline_action),
                            }
                        )
            state_rows.append(
                {
                    "split": split,
                    "state_index": summary["state_index"],
                    "task_suite": summary["task_suite"],
                    "task_id": summary["task_id"],
                    "input_visual_l1": input_visual_l1,
                    "final_action_l2": state_action_l2,
                }
            )

    hidden = pd.DataFrame(hidden_rows)
    stages = pd.DataFrame(stage_rows)
    variants = pd.DataFrame(variant_rows)
    horizon = pd.DataFrame(horizon_rows)
    states = pd.DataFrame(state_rows)
    hidden.to_csv(output / "paired_hidden_surface.csv", index=False)
    stages.to_csv(output / "paired_denoise_stages.csv", index=False)
    variants.to_csv(output / "input_variant_influence.csv", index=False)
    horizon.to_csv(output / "paired_action_horizon.csv", index=False)
    states.to_csv(output / "paired_states.csv", index=False)

    k8 = hidden[hidden["steps"] == 8]
    hidden_summary = {}
    for split_name in ["all", *sorted(hidden["split"].unique())]:
        subset = k8 if split_name == "all" else k8[k8["split"] == split_name]
        grouped = subset.groupby(["stage", "block", "slot"])["hidden_delta_l2"].median()
        hidden_summary[split_name] = {
            f"s{stage}_b{block}_{slot}": float(value)
            for (stage, block, slot), value in grouped.items()
        }

    action_slot = k8[k8["slot"] == "action"]
    correlations = {}
    for (split, stage, block), group in action_slot.groupby(["split", "stage", "block"]):
        correlations[f"{split}_s{stage}_b{block}"] = float(
            group[["hidden_delta_l2", "final_action_l2"]].corr(method="spearman").iloc[0, 1]
        )

    modality_summary = {}
    one_per_state = variants.drop_duplicates(["split", "state_index", "variant"])
    for (split, variant), group in one_per_state.groupby(["split", "variant"]):
        modality_summary[f"{split}_{variant}"] = stats(group["action_l2"])

    horizon_summary = {}
    for (split, steps, token), group in horizon.groupby(["split", "steps", "token"]):
        horizon_summary[f"{split}_k{steps}_token{token}"] = stats(group["fresh_predicted_action_l2"])

    result = {
        "states": len(states),
        "states_by_split": states.groupby("split").size().to_dict(),
        "manifests": manifests,
        "fresh_predicted_final_action": {
            split: stats(group["final_action_l2"])
            for split, group in states.groupby("split")
        },
        "input_visual_l1": {split: stats(group["input_visual_l1"]) for split, group in states.groupby("split")},
        "hidden_surface_medians": hidden_summary,
        "action_hidden_to_final_action_spearman": correlations,
        "one_step_input_modality_action_effect": modality_summary,
        "fresh_predicted_action_horizon": horizon_summary,
        "value_used": False,
    }
    (output / "paired_surface_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    action_heat = action_slot.groupby(["stage", "block"])["hidden_delta_l2"].median().unstack()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    image = ax.imshow(action_heat.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(action_heat.columns)), action_heat.columns)
    ax.set_yticks(range(len(action_heat.index)), action_heat.index)
    ax.set(xlabel="DiT block", ylabel="completed denoiser forwards", title="Fresh-predicted action-slot hidden delta")
    fig.colorbar(image, ax=ax, label="median pooled-hidden L2")
    fig.tight_layout()
    fig.savefig(output / "denoise_block_action_delta.png", dpi=180)
    plt.close(fig)

    final_horizon = horizon[horizon["steps"] == 1].groupby("token")["fresh_predicted_action_l2"].median()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(final_horizon.index, final_horizon.values, marker="o")
    ax.set(xlabel="action chunk position", ylabel="median fresh-predicted action L2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "one_step_visual_sensitivity_by_action_horizon.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

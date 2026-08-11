"""Aggregate task-disjoint WAM mechanism trajectories and denoise diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def mean_step_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)))


def source_file(directory: Path) -> Path:
    final = directory / "checkpoints.pt"
    return final if final.exists() else directory / "checkpoints.partial.pt"


def summarize(values: list[float]) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", action="append", nargs=2, metavar=("SPLIT", "DIR"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    denoise_rows = []
    schedule_rows = []
    token_rows = []
    tasks = []
    episodes_manifest = []
    counts = defaultdict(int)
    split_sources = {}
    checkpoint_sha = None

    for split, directory_string in args.split_dir:
        directory = Path(directory_string)
        source = source_file(directory)
        split_sources[split] = str(source)
        episodes = torch.load(source, weights_only=False)
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            checkpoint_sha = metadata.get("checkpoint", {}).get("sha256", checkpoint_sha)
        for episode in episodes:
            task_key = f"{episode['task_suite']}:{int(episode['task_id'])}"
            tasks.append((split, task_key, episode["task_description"]))
            episodes_manifest.append(
                {
                    "split": split,
                    "task_key": task_key,
                    "episode_index": int(episode["episode_index"]),
                    "seed": int(episode["seed"]),
                    "success": bool(episode["success"]),
                    "requests": len(episode["requests"]),
                }
            )
            labels = episode.get("stage_labels", [])
            for request in episode["requests"]:
                counts[split] += 1
                schedules = request["diagnostic_schedules"]
                kmax = max(map(int, schedules))
                reference_action = schedules[kmax]["checkpoint_actions"][-1]
                reference_future = schedules[kmax]["future_latents"][-1]
                control_step = int(request["control_step"])
                stage_label = labels[control_step] if labels and control_step < len(labels) else "unknown"
                action_magnitude = float(np.mean(np.linalg.norm(reference_action, axis=-1)))
                for k_raw, schedule in schedules.items():
                    k = int(k_raw)
                    final_action = schedule["checkpoint_actions"][-1]
                    schedule_rows.append(
                        {
                            "split": split,
                            "task_key": task_key,
                            "request_id": request["request_id"],
                            "offline_stage": stage_label,
                            "steps": k,
                            "final_action_l2_to_k8": mean_step_l2(final_action, reference_action),
                            "final_future_l1_to_k8": float(np.mean(np.abs(schedule["future_latents"][-1] - reference_future))),
                            "latency_ms": float(request["standalone_latency_ms"][k]),
                            "official_extraction_max_abs": float(schedule["final_agreement_max_abs"]),
                        }
                    )
                    actions = schedule["checkpoint_actions"]
                    futures = schedule["future_latents"]
                    final_within_action = actions[-1]
                    final_within_future = futures[-1]
                    for stage in range(k):
                        action_error = mean_step_l2(actions[stage], final_within_action)
                        future_error = float(np.mean(np.abs(futures[stage] - final_within_future)))
                        visual_error = float(np.mean(np.abs(futures[stage, :, 1:3] - final_within_future[:, 1:3])))
                        proprio_error = float(np.mean(np.abs(futures[stage, :, 0] - final_within_future[:, 0])))
                        denoise_rows.append(
                            {
                                "split": split,
                                "task_key": task_key,
                                "request_id": request["request_id"],
                                "offline_stage": stage_label,
                                "action_magnitude": action_magnitude,
                                "steps": k,
                                "stage": stage + 1,
                                "sigma": float(schedule["sigmas"][stage]),
                                "action_l2_to_schedule_final": action_error,
                                "future_l1_to_schedule_final": future_error,
                                "visual_l1_to_schedule_final": visual_error,
                                "proprio_l1_to_schedule_final": proprio_error,
                                "action_l2_to_k8_final": mean_step_l2(actions[stage], reference_action),
                                "future_l1_to_k8_final": float(np.mean(np.abs(futures[stage] - reference_future))),
                            }
                        )
                        per_token = np.linalg.norm(actions[stage] - final_within_action, axis=-1)
                        for token, error in enumerate(per_token):
                            token_rows.append(
                                {
                                    "split": split,
                                    "task_key": task_key,
                                    "request_id": request["request_id"],
                                    "steps": k,
                                    "stage": stage + 1,
                                    "token": token,
                                    "action_l2": float(error),
                                }
                            )

    denoise_df = pd.DataFrame(denoise_rows)
    schedule_df = pd.DataFrame(schedule_rows)
    token_df = pd.DataFrame(token_rows)
    denoise_df.to_csv(output / "denoise_stages.csv", index=False)
    schedule_df.to_csv(output / "schedule_finals.csv", index=False)
    token_df.to_csv(output / "denoise_horizon.csv", index=False)

    unique_tasks = sorted(set(tasks))
    split_task_sets = {
        split: sorted({task_key for task_split, task_key, _ in unique_tasks if task_split == split})
        for split in sorted(counts)
    }
    overlap = {
        f"{a}__{b}": sorted(set(split_task_sets[a]) & set(split_task_sets[b]))
        for a in split_task_sets for b in split_task_sets if a < b
    }
    dataset = {
        "schema_version": 1,
        "name": "WAM_MECHANISM_DATASET",
        "checkpoint_policy": "pre_SO101_finetune_LIBERO_checkpoint_only",
        "checkpoint_sha256": checkpoint_sha,
        "deployment_denoising_steps": 1,
        "diagnostic_denoising_steps": [1, 2, 4, 8],
        "value_used": False,
        "split_sources": split_sources,
        "state_count": int(sum(counts.values())),
        "state_count_by_split": dict(counts),
        "task_count": len({task_key for _, task_key, _ in unique_tasks}),
        "tasks_by_split": split_task_sets,
        "task_overlap": overlap,
        "task_disjoint": all(not values for values in overlap.values()),
        "episodes": episodes_manifest,
        "semantic_stage_policy": "offline_analysis_only",
    }
    (output / "WAM_MECHANISM_DATASET.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2))

    main_rows = denoise_df[denoise_df["steps"] == 8]
    trajectory_summary = {}
    for metric in (
        "action_l2_to_schedule_final",
        "future_l1_to_schedule_final",
        "visual_l1_to_schedule_final",
        "proprio_l1_to_schedule_final",
    ):
        trajectory_summary[metric] = {
            str(int(stage)): summarize(group[metric].tolist())
            for stage, group in main_rows.groupby("stage")
        }
    final_summary = {
        str(int(steps)): {
            "action_l2_to_k8": summarize(group["final_action_l2_to_k8"].tolist()),
            "future_l1_to_k8": summarize(group["final_future_l1_to_k8"].tolist()),
            "latency_ms": summarize(group["latency_ms"].tolist()),
        }
        for steps, group in schedule_df.groupby("steps")
    }
    result = {
        "state_count": dataset["state_count"],
        "task_count": dataset["task_count"],
        "trajectory_8step": trajectory_summary,
        "standalone_schedule_finals": final_summary,
        "action_video_stagewise_pearson": float(
            main_rows[["action_l2_to_schedule_final", "visual_l1_to_schedule_final"]].corr().iloc[0, 1]
        ),
        "provenance": {
            "value_used": False,
            "full_attention_saved": False,
            "multi_step_is_diagnostic_only": True,
        },
    }
    (output / "denoise_trajectory_summary.json").write_text(json.dumps(result, indent=2))

    import matplotlib.pyplot as plt

    grouped = main_rows.groupby("stage")[["action_l2_to_schedule_final", "visual_l1_to_schedule_final"]].median()
    normalized = grouped / grouped.iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(normalized.index, normalized["action_l2_to_schedule_final"], marker="o", label="action")
    ax.plot(normalized.index, normalized["visual_l1_to_schedule_final"], marker="s", label="future visual")
    ax.set(xlabel="8-step diagnostic: completed denoiser forwards", ylabel="median residual / stage-1 residual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "action_vs_video_denoise_convergence.png", dpi=180)
    plt.close(fig)

    heat = token_df[token_df["steps"] == 8].groupby(["stage", "token"])["action_l2"].median().unstack()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    image = ax.imshow(heat.to_numpy(), aspect="auto", origin="lower", cmap="magma")
    ax.set(xlabel="action chunk position", ylabel="completed denoiser forwards")
    ax.set_yticks(range(len(heat.index)), heat.index)
    fig.colorbar(image, ax=ax, label="median action L2 to 8-step final")
    fig.tight_layout()
    fig.savefig(output / "denoise_by_action_horizon.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

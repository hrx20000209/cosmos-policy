#!/usr/bin/env python3
"""Validate and summarize targeted stage-2 repeats without contaminating round one."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
EXPERIMENT = Path(__file__).resolve().parents[1]
MANIFESTS = EXPERIMENT / "manifests"
SUMMARIES = EXPERIMENT / "summaries"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def summarize(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        values = key if isinstance(key, tuple) else (key,)
        successes = int(group["success"].astype(bool).sum())
        low, high = wilson(successes, len(group))
        rows.append(
            {
                **dict(zip(keys, values)),
                "episodes": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "wilson_low": low,
                "wilson_high": high,
                "mean_episode_steps": float(group["episode_steps"].mean()),
                "mean_episode_time_s": float(group["episode_wall_clock_time_s"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    records = []
    for path in OUTPUT.glob("raw/*/episodes.shard*.jsonl"):
        records.extend(load_jsonl(path))
    episodes = (
        pd.DataFrame(records)
        .sort_values("completed_at_ns")
        .drop_duplicates("episode_key", keep="last")
    )
    selection = json.loads(
        (SUMMARIES / "second_stage_selection.json").read_text(encoding="utf-8")
    )
    selected = [int(value) for value in selection["selected_steps"]]

    original_round1 = episodes[
        (episodes["domain"] == "libero")
        & episodes["denoising_steps"].isin(selected)
        & ~episodes["config_id"].astype(str).str.contains("stage2|video_replay", regex=True)
    ]
    original_repeats = episodes[
        episodes["config_id"].astype(str).str.contains("stage2_seed_repeat")
    ]
    original_three_seed = pd.concat([original_round1, original_repeats], ignore_index=True)
    original_summary = summarize(original_three_seed, ["denoising_steps"])
    original_summary.to_csv(SUMMARIES / "stage2_original_three_seed.csv", index=False)

    hard_repeats = episodes[episodes["config_id"].astype(str).str.contains("stage2_hard")]
    hard_summary = (
        summarize(
            hard_repeats,
            ["domain", "perturbation_category", "denoising_steps"],
        )
        if not hard_repeats.empty
        else pd.DataFrame()
    )
    hard_summary.to_csv(SUMMARIES / "stage2_hard_subset_metrics.csv", index=False)

    language = episodes[
        episodes["config_id"].astype(str).str.contains("stage2_language_p")
    ]
    language_summary = (
        summarize(language, ["denoising_steps"]) if not language.empty else pd.DataFrame()
    )
    language_summary.to_csv(
        SUMMARIES / "stage2_language_all_paraphrases_metrics.csv", index=False
    )
    if not language.empty:
        variance = (
            language.groupby(["task_uid", "denoising_steps"])["success"]
            .agg(["mean", "var", "count"])
            .reset_index()
        )
    else:
        variance = pd.DataFrame()
    variance.to_csv(SUMMARIES / "stage2_language_base_task_variance.csv", index=False)

    replay = episodes[episodes["config_id"].astype(str).str.contains("video_replay")]
    labels = []
    for _, row in replay.iterrows():
        if not bool(row["environment_valid"]):
            label = "environment_invalid"
            evidence = "environment_valid=False"
        elif str(row["termination_reason"]) == "max_steps":
            label = "timeout"
            evidence = "termination_reason=max_steps"
        else:
            label = "unclassified"
            evidence = "没有足够人工或明确事件证据"
        labels.append(
            {
                "episode_key": row["episode_key"],
                "first_round_episode_key": row.get("first_round_episode_key"),
                "success": bool(row["success"]),
                "failure_label": label,
                "evidence": evidence,
                "video_path": row.get("video_path"),
            }
        )
    pd.DataFrame(labels).to_csv(SUMMARIES / "failure_video_labels.csv", index=False)

    expected_original = 40 * len(selected) * 3
    manifests = {
        name: len(load_jsonl(MANIFESTS / name))
        for name in (
            "stage2_original_seed_repeats.jsonl",
            "stage2_hard_subset.jsonl",
            "stage2_language_all_paraphrases.jsonl",
            "stage2_disagreement_video_replay.jsonl",
        )
    }
    result = {
        "selected_steps": selected,
        "original_three_seed_expected": expected_original,
        "original_three_seed_completed": len(original_three_seed),
        "hard_repeat_completed": len(hard_repeats),
        "language_all_paraphrases_completed": len(language),
        "video_replay_completed": len(replay),
        "manifests": manifests,
        "complete": (
            len(original_three_seed) == expected_original
            and len(hard_repeats) == manifests["stage2_hard_subset.jsonl"]
            and len(language) == manifests["stage2_language_all_paraphrases.jsonl"]
            and len(replay) == manifests["stage2_disagreement_video_replay.jsonl"]
        ),
    }
    (SUMMARIES / "stage2_analysis_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

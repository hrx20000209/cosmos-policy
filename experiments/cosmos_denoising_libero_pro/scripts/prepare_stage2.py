#!/usr/bin/env python3
"""Build data-driven stage-2 repeats, hard cases, and all-language variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT / "experiments/cosmos_denoising_libero_pro"
MANIFESTS = EXPERIMENT / "manifests"
OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
PRO_REPO = Path("/data/rxhuang/repos/LIBERO-PRO")
SEEDS = (195, 196, 197)
OSMESA_LIB = Path(
    "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
)


def configure_headless_rendering() -> None:
    """Match the validated OSMesa setup used by the rollout launcher."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if str(OSMESA_LIB) not in paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [str(OSMESA_LIB), *[path for path in paths if path]]
        )


def ensure_headless_process_environment() -> None:
    """Re-exec once when the dynamic linker did not start with OSMesa visible."""
    paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    ready = (
        os.environ.get("MUJOCO_GL") == "osmesa"
        and os.environ.get("PYOPENGL_PLATFORM") == "osmesa"
        and str(OSMESA_LIB) in paths
    )
    if ready:
        return
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "osmesa"
    environment["PYOPENGL_PLATFORM"] = "osmesa"
    environment["LD_LIBRARY_PATH"] = ":".join(
        [str(OSMESA_LIB), *[path for path in paths if path]]
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def episode_key(row: dict) -> str:
    identity = "|".join(
        str(row[key])
        for key in (
            "config_id",
            "task_uid",
            "variant_id",
            "seed",
            "init_state_index",
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def clone_row(
    source: dict,
    *,
    suffix: str,
    seed: int,
    init_state_index: int,
    init_path: Path | None = None,
) -> dict:
    row = dict(source)
    row["first_round_episode_key"] = source["episode_key"]
    row["first_round_variant_id"] = source["variant_id"]
    row["config_id"] = f"{source['config_id']}_{suffix}"
    row["variant_id"] = f"{source['variant_id']}:{suffix}"
    row["seed"] = seed
    row["init_state_index"] = init_state_index
    row["stage"] = "stage2"
    if init_path is not None:
        row["init_path"] = str(init_path)
        row["init_sha256"] = sha256(init_path)
    row["episode_key"] = episode_key(row)
    return row


def first_round_episodes() -> pd.DataFrame:
    records = []
    for path in OUTPUT.glob("raw/*/episodes.shard*.jsonl"):
        records.extend(load_jsonl(path))
    frame = pd.DataFrame(records)
    frame = frame[
        ~frame["config_id"].astype(str).str.contains("stage2|video_replay", regex=True)
    ]
    return (
        frame.sort_values("completed_at_ns")
        .drop_duplicates("episode_key", keep="last")
        .reset_index(drop=True)
    )


def identify_hard_cases(
    episodes: pd.DataFrame, selected_steps: list[int]
) -> tuple[pd.DataFrame, set[tuple[str, str, str]]]:
    cases: list[dict[str, Any]] = []
    hard: set[tuple[str, str, str]] = set()
    group_columns = ["domain", "perturbation_category", "task_uid", "variant_id"]
    for key, group in episodes.groupby(group_columns):
        outcomes = {
            int(row["denoising_steps"]): bool(row["success"])
            for _, row in group.iterrows()
        }
        comparisons = (
            (5, 1, "5_success_1_failure"),
            (5, 3, "5_success_3_failure"),
            (6, 5, "6_success_5_failure"),
            (3, 5, "3_success_5_failure"),
        )
        for success_step, failure_step, reason in comparisons:
            if outcomes.get(success_step) and outcomes.get(failure_step) is False:
                success_row = group[group["denoising_steps"] == success_step].iloc[-1]
                failure_row = group[group["denoising_steps"] == failure_step].iloc[-1]
                cases.append(
                    {
                        **dict(zip(group_columns, key)),
                        "base_task_id": int(failure_row["base_task_id"]),
                        "reason": reason,
                        "success_episode_key": success_row["episode_key"],
                        "failure_episode_key": failure_row["episode_key"],
                    }
                )
                hard.add((str(key[0]), str(key[1]), str(key[2])))

    original = episodes[episodes["domain"] == "libero"]
    pro = episodes[episodes["domain"] == "libero_pro"]
    original_lookup = {
        (int(row["base_task_id"]), int(row["denoising_steps"])): row
        for _, row in original.iterrows()
    }
    for _, row in pro[~pro["success"].astype(bool)].iterrows():
        match = original_lookup.get((int(row["base_task_id"]), int(row["denoising_steps"])))
        if match is not None and bool(match["success"]):
            cases.append(
                {
                    "domain": row["domain"],
                    "perturbation_category": row["perturbation_category"],
                    "task_uid": row["task_uid"],
                    "variant_id": row["variant_id"],
                    "base_task_id": int(row["base_task_id"]),
                    "reason": "matched_original_success_pro_failure",
                    "success_episode_key": match["episode_key"],
                    "failure_episode_key": row["episode_key"],
                }
            )
            hard.add((str(row["domain"]), str(row["perturbation_category"]), str(row["task_uid"])))

    low_steps = [step for step in selected_steps if step < 5]
    for (base_task_id, step), group in pro[pro["denoising_steps"].isin(low_steps)].groupby(
        ["base_task_id", "denoising_steps"]
    ):
        if not (group["success"].astype(bool).any() and (~group["success"].astype(bool)).any()):
            continue
        for _, row in group[~group["success"].astype(bool)].iterrows():
            cases.append(
                {
                    "domain": row["domain"],
                    "perturbation_category": row["perturbation_category"],
                    "task_uid": row["task_uid"],
                    "variant_id": row["variant_id"],
                    "base_task_id": int(base_task_id),
                    "reason": f"low_step_{int(step)}_category_specific_failure",
                    "success_episode_key": None,
                    "failure_episode_key": row["episode_key"],
                }
            )
            hard.add((str(row["domain"]), str(row["perturbation_category"]), str(row["task_uid"])))

    result = pd.DataFrame(cases).drop_duplicates(
        ["failure_episode_key", "success_episode_key", "reason"]
    )
    return result, hard


def install_pro_checkout() -> None:
    os.environ["LIBERO_CONFIG_PATH"] = str(EXPERIMENT / "configs/libero_pro_config")
    for name in [key for key in sys.modules if key == "libero" or key.startswith("libero.")]:
        del sys.modules[name]
    outer = types.ModuleType("libero")
    outer.__path__ = [str(PRO_REPO / "libero")]
    outer.__package__ = "libero"
    sys.modules["libero"] = outer


def make_five_state_init(row: dict, overwrite: bool) -> Path:
    target = (
        OUTPUT
        / "assets/stage2_init"
        / f"{row['suite']}_{row['perturbation_category']}"
        / f"{row['task_name']}.five_states"
    )
    if target.is_file() and not overwrite:
        states = torch.load(target, map_location="cpu", weights_only=False)
        if len(states) != 5:
            raise ValueError(f"{target}: expected five states, got {len(states)}")
        return target
    configure_headless_rendering()
    install_pro_checkout()
    from libero.libero.envs import OffScreenRenderEnv

    first = torch.load(row["init_path"], map_location="cpu", weights_only=False)
    states = [np.asarray(first[0])]
    env = None
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=row["bddl_path"],
            camera_heights=128,
            camera_widths=128,
        )
        for offset in range(1, 5):
            state_seed = 28 + offset
            random.seed(state_seed)
            np.random.seed(state_seed)
            env.seed(state_seed)
            env.reset()
            states.append(np.asarray(env.get_sim_state()))
    finally:
        if env is not None:
            env.close()
    packed = np.stack(states)
    if any(np.array_equal(packed[0], packed[index]) for index in range(1, 5)):
        raise ValueError(f"{target}: generated duplicate of the first initial state")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packed, target)
    return target


def generate_stage2_manifests(
    episodes: pd.DataFrame,
    selected_steps: list[int],
    hard: set[tuple[str, str, str]],
    overwrite_init: bool,
) -> tuple[list[dict], list[dict]]:
    original_rows = load_jsonl(MANIFESTS / "original_full.jsonl")
    pro_rows = load_jsonl(MANIFESTS / "libero_pro_full_supported.jsonl")
    original_source = {
        (row["task_uid"], int(row["denoising_steps"])): row for row in original_rows
    }
    pro_source = {
        (row["perturbation_category"], row["task_uid"], int(row["denoising_steps"])): row
        for row in pro_rows
    }
    original_repeats = []
    for (_, step), source in sorted(original_source.items()):
        if step not in selected_steps:
            continue
        for seed in (196, 197):
            original_repeats.append(
                clone_row(source, suffix="stage2_seed_repeat", seed=seed, init_state_index=0)
            )

    hard_repeats = []
    for domain, category, task_uid in sorted(hard):
        for step in selected_steps:
            source = (
                original_source.get((task_uid, step))
                if domain == "libero"
                else pro_source.get((category, task_uid, step))
            )
            if source is None:
                continue
            init_path = Path(source["init_path"])
            if domain == "libero_pro":
                init_path = make_five_state_init(source, overwrite_init)
            for seed in SEEDS:
                for init_index in range(5):
                    if seed == 195 and init_index == 0:
                        continue
                    if domain == "libero" and init_index == 0 and seed in (196, 197):
                        continue
                    hard_repeats.append(
                        clone_row(
                            source,
                            suffix="stage2_hard",
                            seed=seed,
                            init_state_index=init_index,
                            init_path=init_path,
                        )
                    )
    return original_repeats, hard_repeats


def replace_language(content: str, instruction: str) -> str:
    replaced, count = re.subn(
        r"\(:language\s*.*?\)",
        f"(:language {instruction})",
        content,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise ValueError("BDDL language replacement did not match exactly once")
    return replaced


def normalize_official_paraphrase(value: Any) -> str:
    """Recover an unquoted colon-bearing scalar parsed by YAML as a mapping."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and len(value) == 1:
        key, item = next(iter(value.items()))
        if isinstance(key, str) and isinstance(item, str):
            return f"{key}: {item}"
    raise TypeError(f"unsupported official paraphrase value: {value!r}")


def generate_all_language(selected_steps: list[int]) -> list[dict]:
    source_rows = load_jsonl(MANIFESTS / "libero_pro_full_supported.jsonl")
    source = {
        (row["task_uid"], int(row["denoising_steps"])): row
        for row in source_rows
        if row["perturbation_category"] == "language"
    }
    language_yaml = yaml.safe_load(
        (PRO_REPO / "libero_ood/ood_language.yaml").read_text(encoding="utf-8")
    )
    output = []
    for task_uid in sorted({key[0] for key in source}):
        any_row = source[(task_uid, selected_steps[0])]
        variants = language_yaml[any_row["suite"]][any_row["task_name"]]
        if len(variants) != 3:
            raise ValueError(f"{task_uid}: expected three official paraphrases")
        base_content = Path(any_row["bddl_path"]).read_text(encoding="utf-8")
        for variant_index, raw_instruction in enumerate(variants):
            instruction = normalize_official_paraphrase(raw_instruction)
            bddl_path = (
                OUTPUT
                / "assets/stage2_language/bddl"
                / f"{any_row['suite']}_lan_all"
                / f"{any_row['task_name']}.paraphrase{variant_index}.bddl"
            )
            bddl_path.parent.mkdir(parents=True, exist_ok=True)
            bddl_path.write_text(replace_language(base_content, instruction), encoding="utf-8")
            for step in selected_steps:
                row = clone_row(
                    source[(task_uid, step)],
                    suffix=f"stage2_language_p{variant_index}",
                    seed=195,
                    init_state_index=0,
                )
                row.update(
                    {
                        "instruction": instruction,
                        "bddl_instruction": instruction,
                        "bddl_path": str(bddl_path),
                        "bddl_sha256": sha256(bddl_path),
                        "language_variant_index": variant_index,
                        "language_cluster_id": task_uid,
                        "t5_embedding_sha256": None,
                    }
                )
                row["variant_id"] = f"language:official_paraphrase:{variant_index}"
                row["episode_key"] = episode_key(row)
                output.append(row)
    return output


def main() -> None:
    ensure_headless_process_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-init", action="store_true")
    args = parser.parse_args()
    selection = json.loads(
        (EXPERIMENT / "summaries/second_stage_selection.json").read_text(encoding="utf-8")
    )
    selected_steps = [int(value) for value in selection["selected_steps"]]
    episodes = first_round_episodes()
    cases, hard = identify_hard_cases(episodes, selected_steps)
    cases.to_csv(EXPERIMENT / "summaries/paired_disagreement_cases.csv", index=False)
    original_repeats, hard_repeats = generate_stage2_manifests(
        episodes, selected_steps, hard, args.overwrite_init
    )
    language = generate_all_language(selected_steps)
    write_jsonl(MANIFESTS / "stage2_original_seed_repeats.jsonl", original_repeats)
    write_jsonl(MANIFESTS / "stage2_hard_subset.jsonl", hard_repeats)
    write_jsonl(MANIFESTS / "stage2_language_all_paraphrases.jsonl", language)

    by_key = {row["episode_key"]: row for _, row in episodes.iterrows()}
    replay_keys = set(cases["failure_episode_key"].dropna()) | set(
        cases["success_episode_key"].dropna()
    )
    replay = [
        clone_row(
            by_key[key],
            suffix="video_replay",
            seed=int(by_key[key]["seed"]),
            init_state_index=int(by_key[key]["init_state_index"]),
        )
        for key in sorted(replay_keys)
    ]
    write_jsonl(MANIFESTS / "stage2_disagreement_video_replay.jsonl", replay)
    first_round = load_jsonl(MANIFESTS / "formal_full_supported.jsonl")
    write_jsonl(
        MANIFESTS / "stage2_t5_union.jsonl",
        first_round + original_repeats + hard_repeats + language + replay,
    )
    index = {
        "selected_steps": selected_steps,
        "hard_task_variants": len(hard),
        "paired_disagreement_cases": len(cases),
        "original_new_seed_episodes": len(original_repeats),
        "hard_new_episodes": len(hard_repeats),
        "language_all_paraphrase_episodes": len(language),
        "video_replay_episodes": len(replay),
        "hard_definition": [
            "5 succeeds while 1 or 3 fails",
            "6 succeeds while 5 fails",
            "3 succeeds while 5 fails",
            "matched original succeeds while PRO fails",
            "a low-step failure is category-specific for the same base task",
        ],
    }
    (EXPERIMENT / "summaries/stage2_manifest_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

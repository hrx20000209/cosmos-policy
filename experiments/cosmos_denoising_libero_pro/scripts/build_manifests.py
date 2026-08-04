#!/usr/bin/env python3
"""Build exact, hash-addressed original LIBERO and LIBERO-PRO manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[3]
EXPERIMENT = PROJECT / "experiments/cosmos_denoising_libero_pro"
OUTPUT_ROOT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
ORIGINAL_REPO = Path("/home/rxhuang/Projects/LIBERO")
PRO_REPO = Path("/data/rxhuang/repos/LIBERO-PRO")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STEPS = (1, 2, 3, 4, 5, 6)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
CATEGORY_SUFFIX = {
    "language": "lan",
    "object": "object",
    "position": "swap",
    "task": "task",
    "environment": "env",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def language_from_bddl(path: Path) -> str:
    match = re.search(r"\(:language\s*(.*?)\)", path.read_text(encoding="utf-8"), flags=re.S)
    if not match:
        raise ValueError(f"missing (:language ...) in {path}")
    return " ".join(match.group(1).split())


def configure_original_import() -> None:
    os.environ["LIBERO_CONFIG_PATH"] = str(EXPERIMENT / "configs/original_libero_config")
    # The checkout has no outer libero/__init__.py, so an installed regular
    # LIBERO-plus package otherwise wins. Pin an explicit namespace shim.
    for name in [key for key in sys.modules if key == "libero" or key.startswith("libero.")]:
        del sys.modules[name]
    outer = types.ModuleType("libero")
    outer.__path__ = [str(ORIGINAL_REPO / "libero")]
    outer.__package__ = "libero"
    sys.modules["libero"] = outer


def base_tasks() -> list[dict[str, Any]]:
    configure_original_import()
    from libero.libero import benchmark
    from libero.libero import get_libero_path

    records: list[dict[str, Any]] = []
    global_index = 0
    for suite in SUITES:
        instance = benchmark.get_benchmark_dict()[suite](task_order_index=0)
        if instance.get_num_tasks() != 10:
            raise RuntimeError(f"{suite}: expected 10 tasks, got {instance.get_num_tasks()}")
        for task_index in range(10):
            task = instance.get_task(task_index)
            bddl_path = Path(instance.get_task_bddl_file_path(task_index)).resolve()
            init_path = (
                Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            ).resolve()
            if not bddl_path.is_file() or not init_path.is_file():
                raise FileNotFoundError(f"missing original task asset: {bddl_path} / {init_path}")
            records.append(
                {
                    "global_task_index": global_index,
                    "suite": suite,
                    "suite_task_index": task_index,
                    "task_name": task.name,
                    "task_uid": f"{suite}:{task.name}",
                    "canonical_instruction": task.language,
                    "bddl_instruction": language_from_bddl(bddl_path),
                    "bddl_path": str(bddl_path),
                    "bddl_sha256": sha256(bddl_path),
                    "init_path": str(init_path),
                    "init_sha256": sha256(init_path),
                }
            )
            global_index += 1
    return records


def episode_row(
    task: dict[str, Any],
    *,
    domain: str,
    category: str,
    variant_id: str,
    instruction: str,
    steps: int,
    order_index: int,
    bddl_path: Path | None = None,
    bddl_sha: str | None = None,
    init_path: Path | None = None,
    init_sha: str | None = None,
    variant_applied: bool = True,
) -> dict[str, Any]:
    task_uid = task["task_uid"]
    config_id = f"cosmos_{domain}_{category}_steps{steps}"
    key_text = f"{config_id}|{task_uid}|{variant_id}|195|0"
    return {
        "schema_version": 1,
        "manifest_order": order_index,
        "episode_key": hashlib.sha256(key_text.encode()).hexdigest(),
        "config_id": config_id,
        "domain": domain,
        "perturbation_category": category,
        "variant_id": variant_id,
        "variant_applied": variant_applied,
        "supported": bool(
            variant_applied if domain == "libero_pro" and category == "environment" else True
        ),
        "suite": task["suite"],
        "suite_task_index": task["suite_task_index"],
        "global_task_index": task["global_task_index"],
        "task_name": task["task_name"],
        "task_uid": task_uid,
        "instruction": instruction,
        "canonical_instruction": task["canonical_instruction"],
        "bddl_instruction": language_from_bddl(bddl_path) if bddl_path else task["bddl_instruction"],
        "bddl_path": str(bddl_path) if bddl_path else task["bddl_path"],
        "bddl_sha256": bddl_sha or task["bddl_sha256"],
        "init_path": str(init_path) if init_path else task["init_path"],
        "init_sha256": init_sha or task["init_sha256"],
        "init_state_index": 0,
        "seed": 195,
        "perturbation_seed": 28 if domain == "libero_pro" else None,
        "denoising_steps": steps,
        "action_horizon": 16,
        "generation_mode": "action_only_usage",
        "decode_future": False,
        "deterministic": True,
        "randomize_seed": False,
        "max_steps": MAX_STEPS[task["suite"]],
        "libero_repo": str(PRO_REPO if domain == "libero_pro" else ORIGINAL_REPO),
        "libero_config_path": str(
            EXPERIMENT / ("configs/libero_pro_config" if domain == "libero_pro" else "configs/original_libero_config")
        ),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    tasks = base_tasks()
    task_by_key = {(row["suite"], row["task_name"]): row for row in tasks}
    original: list[dict[str, Any]] = []
    for task in tasks:
        rotated = STEPS[task["global_task_index"] % len(STEPS) :] + STEPS[: task["global_task_index"] % len(STEPS)]
        for steps in rotated:
            original.append(
                episode_row(
                    task,
                    domain="libero",
                    category="none",
                    variant_id="original",
                    instruction=task["canonical_instruction"],
                    steps=steps,
                    order_index=len(original),
                )
            )

    audit_path = OUTPUT_ROOT / "assets/libero_pro_asset_audit.json"
    asset_rows = json.loads(audit_path.read_text(encoding="utf-8"))
    pro: list[dict[str, Any]] = []
    for category_index, category in enumerate(CATEGORY_SUFFIX):
        category_assets = [row for row in asset_rows if row["category"] == category]
        if len(category_assets) != 40:
            raise RuntimeError(f"{category}: expected 40 assets, got {len(category_assets)}")
        category_assets.sort(key=lambda row: task_by_key[(row["suite"], row["task_name"])]["global_task_index"])
        for asset in category_assets:
            task = task_by_key[(asset["suite"], asset["task_name"])]
            rotation = (task["global_task_index"] + category_index) % len(STEPS)
            rotated = STEPS[rotation:] + STEPS[:rotation]
            instruction = (
                asset["instruction"]
                if category in {"language", "task"}
                else task["canonical_instruction"]
            )
            init_path = Path(asset["init_path"])
            for steps in rotated:
                pro.append(
                    episode_row(
                        task,
                        domain="libero_pro",
                        category=category,
                        variant_id=f"{category}:seed28",
                        instruction=instruction,
                        steps=steps,
                        order_index=len(pro),
                        bddl_path=Path(asset["bddl_path"]),
                        bddl_sha=asset["bddl_sha256"],
                        init_path=init_path,
                        init_sha=asset.get("init_sha256"),
                        variant_applied=bool(asset["variant_applied"]),
                    )
                )

    manifests = EXPERIMENT / "manifests"
    t5_audit_path = OUTPUT_ROOT / "assets/cosmos_libero_pro_t5_embeddings.audit.json"
    if t5_audit_path.is_file():
        t5_audit = json.loads(t5_audit_path.read_text(encoding="utf-8"))
        t5_hashes = {
            instruction: metadata["tensor_sha256"]
            for instruction, metadata in t5_audit["embeddings"].items()
        }
        for row in original + pro:
            row["t5_embedding_sha256"] = t5_hashes[row["instruction"]]
    write_jsonl(manifests / "original_full.jsonl", original)
    write_jsonl(manifests / "libero_pro_full.jsonl", pro)
    write_jsonl(manifests / "full_sweep.jsonl", original + pro)
    supported_pro = [row for row in pro if row["supported"]]
    write_jsonl(manifests / "libero_pro_full_supported.jsonl", supported_pro)
    write_jsonl(manifests / "formal_full_supported.jsonl", original + supported_pro)
    for category in CATEGORY_SUFFIX:
        write_jsonl(
            manifests / f"libero_pro_{category}_supported.jsonl",
            [
                row
                for row in supported_pro
                if row["perturbation_category"] == category
            ],
        )
    write_json(manifests / "libero_40tasks_seed195.json", original)
    write_json(manifests / "libero_pro_single_perturbation_seed195.json", pro)
    instruction_rows = []
    for instruction in sorted({row["instruction"] for row in original + pro}):
        contexts = [
            {
                "domain": row["domain"],
                "perturbation_category": row["perturbation_category"],
                "task_uid": row["task_uid"],
            }
            for row in original + pro
            if row["instruction"] == instruction
        ]
        instruction_rows.append(
            {
                "instruction": instruction,
                "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                "contexts": contexts,
            }
        )
    write_json(manifests / "libero_pro_instructions.json", instruction_rows)
    original_smoke = [
        row
        for row in original
        if row["suite_task_index"] == 0 and row["denoising_steps"] in {1, 5, 6}
    ]
    # Exactly one real task per PRO category, with 1/5/6 steps.
    pro_smoke = []
    for category in CATEGORY_SUFFIX:
        candidates = [
            row
            for row in pro
            if row["perturbation_category"] == category
            and row["global_task_index"] == 0
            and row["denoising_steps"] in {1, 5, 6}
        ]
        pro_smoke.extend(candidates)
    write_jsonl(manifests / "correctness_original.jsonl", original_smoke)
    write_jsonl(manifests / "correctness_libero_pro.jsonl", pro_smoke)
    write_jsonl(manifests / "correctness_all.jsonl", original_smoke + pro_smoke)
    summary = {
        "schema_version": 1,
        "original_tasks": len(tasks),
        "steps": list(STEPS),
        "original_episodes": len(original),
        "libero_pro_assets": len(asset_rows),
        "libero_pro_episodes": len(pro),
        "libero_pro_supported_episodes": len(supported_pro),
        "formal_episode_total": len(original) + len(supported_pro),
        "total_upper_bound": len(pro),
        "correctness_original_episodes": len(original_smoke),
        "correctness_libero_pro_episodes": len(pro_smoke),
        "by_pro_category": {
            category: sum(row["perturbation_category"] == category for row in pro)
            for category in CATEGORY_SUFFIX
        },
        "unique_episode_keys": len({row["episode_key"] for row in original + pro}),
        "unique_instructions": len({row["instruction"] for row in original + pro}),
    }
    (manifests / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(original) != 240 or len(pro) != 1200:
        raise SystemExit("manifest cardinality invariant failed")
    if summary["unique_episode_keys"] != 1440:
        raise SystemExit("duplicate episode keys detected")


if __name__ == "__main__":
    main()

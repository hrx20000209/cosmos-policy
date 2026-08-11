"""Build the task-disjoint, five-init manifest for the server validation run.

The manifest is derived from the already audited LIBERO-PRO assets.  It does
not inspect simulator state during policy inference; init states are used only
by the normal environment reset path.  Task split is by (suite, task_name),
never by request/state, so later offline rows remain task-disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
ROOT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets")
AUDIT = ROOT / "libero_pro_asset_audit.json"
STAGE2 = ROOT / "stage2_init"
PRO_REPO = Path("/data/rxhuang/repos/LIBERO-PRO")
PRO_CONFIG = Path("/home/rxhuang/Projects/cosmos-policy/experiments/cosmos_denoising_libero_pro/configs/libero_pro_config")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--tasks-per-suite", type=int, default=8)
    parser.add_argument("--inits", type=int, default=5)
    parser.add_argument("--config-id", default="server_f1_collection")
    args = parser.parse_args()

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    by_task: dict[tuple[str, str], list[dict]] = {}
    for row in audit:
        by_task.setdefault((row["suite"], row["task_name"]), []).append(row)

    selected: list[dict] = []
    categories = ("task", "object", "position", "environment", "language")
    for suite in SUITES:
        names = sorted(name for current_suite, name in by_task if current_suite == suite)
        if len(names) < args.tasks_per_suite:
            raise RuntimeError(f"{suite}: only {len(names)} unique tasks")
        for task_index, task_name in enumerate(names[: args.tasks_per_suite]):
            options = by_task[(suite, task_name)]
            preferred = categories[task_index % len(categories)]
            ordered_options = sorted(
                options,
                key=lambda item: (item["category"] != preferred, item["category"]),
            )
            row = next(
                (
                    item
                    for item in ordered_options
                    if (STAGE2 / f"{suite}_{item['category']}" / f"{task_name}.five_states").is_file()
                ),
                None,
            )
            if row is None:
                raise FileNotFoundError(f"no five-state stage2 init for {suite}:{task_name}")
            init_path = STAGE2 / f"{suite}_{row['category']}" / f"{task_name}.five_states"
            if not row.get("variant_applied", False):
                raise RuntimeError(f"unsupported/unapplied asset: {suite}:{task_name}:{row['category']}")
            selected.append({"task": row, "init_path": init_path, "suite_task_index": len(selected)})

    if args.tasks_per_suite == 10:
        # The full-scale protocol keeps all suites balanced while reserving
        # enough task-disjoint data for the final held-out analysis.
        split_by_rank = {
            rank: ("discovery" if rank < 5 else "validation" if rank < 7 else "heldout")
            for rank in range(args.tasks_per_suite)
        }
    else:
        discovery = max(1, args.tasks_per_suite // 2)
        validation = max(1, (args.tasks_per_suite - discovery) // 2)
        split_by_rank = {
            rank: (
                "discovery"
                if rank < discovery
                else "validation"
                if rank < discovery + validation
                else "heldout"
            )
            for rank in range(args.tasks_per_suite)
        }
    rows: list[dict] = []
    order = 0
    for selected_index, item in enumerate(selected):
        task = item["task"]
        suite_rank = selected_index % args.tasks_per_suite
        suite = task["suite"]
        split = split_by_rank[suite_rank]
        task_uid = f"{suite}:{task['task_name']}"
        for init_state_index in range(args.inits):
            seed = 195 + init_state_index
            episode_key = hashlib.sha256(
                f"server_f1_collection|{split}|{task_uid}|{init_state_index}|{seed}".encode()
            ).hexdigest()
            rows.append(
                {
                    "schema_version": 1,
                    "manifest_order": order,
                    "episode_key": episode_key,
                    "config_id": args.config_id,
                    "domain": "libero_pro",
                    "split": split,
                    "perturbation_category": task["category"],
                    "variant_id": f"{task['category']}:seed28",
                    "variant_applied": True,
                    "supported": True,
                    "suite": suite,
                    "suite_task_index": suite_rank,
                    "global_task_index": selected_index,
                    "task_name": task["task_name"],
                    "task_uid": task_uid,
                    "instruction": task["instruction"],
                    "canonical_instruction": task["instruction"],
                    "bddl_path": task["bddl_path"],
                    "bddl_sha256": task["bddl_sha256"],
                    "init_path": str(item["init_path"]),
                    "init_sha256": sha256(item["init_path"]),
                    "init_state_index": init_state_index,
                    "seed": seed,
                    "perturbation_seed": 28,
                    "denoising_steps": 1,
                    "action_horizon": 16,
                    "generation_mode": "action_only_usage",
                    "decode_future": False,
                    "deterministic": True,
                    "randomize_seed": False,
                    "max_steps": MAX_STEPS[suite],
                    "libero_repo": str(PRO_REPO),
                    "libero_config_path": str(PRO_CONFIG),
                }
            )
            order += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "manifest": str(args.output),
        "episodes": len(rows),
        "tasks": len(selected),
        "tasks_per_suite": args.tasks_per_suite,
        "inits_per_task": args.inits,
        "states_target": "3000-5000 decision states for full-scale validation" if args.tasks_per_suite == 10 else "1500-3000 decision states after closed-loop collection",
        "split_tasks": {
            split: sorted({row["task_uid"] for row in rows if row["split"] == split})
            for split in ("discovery", "validation", "heldout")
        },
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "value_used": False,
        "privileged_runtime_state_used": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

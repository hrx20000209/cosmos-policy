"""Collect a task-diverse LIBERO-PRO mechanism supplement.

The primary 589-state overnight dataset uses clean LIBERO task suites.  This
runner deliberately uses the official LIBERO-PRO generated BDDL/init assets so
the two domains remain explicit instead of silently relabeling clean tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (  # noqa: E402
    ManifestLiberoEnvironment,
    install_libero_checkout,
)
from experiments.progressive_wam.cosmos_hook import CosmosCheckpointCapture  # noqa: E402
from experiments.progressive_wam.provenance import run_provenance, write_json  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_STATS,
    KNOWN_CHECKPOINT_SHA256,
    build_cfg,
    load_model,
    run_episode,
)


DEFAULT_MANIFEST = REPO_ROOT / "experiments/cosmos_denoising_libero_pro/manifests/libero_pro_full_supported.jsonl"
DEFAULT_PRO_T5 = "/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/cosmos_libero_pro_t5_embeddings.pkl"
DEFAULT_PRO_REPO = "/data/rxhuang/repos/LIBERO-PRO"
DEFAULT_PRO_CONFIG = REPO_ROOT / "experiments/cosmos_denoising_libero_pro/configs/libero_pro_config"
CATEGORIES = ("language", "object", "position", "task", "environment")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


class ProEnvironment(ManifestLiberoEnvironment):
    def __init__(self, row: dict, resolution: int):
        super().__init__(row, resolution, None)
        self.description = row["instruction"]

    def reset(self, episode_index: int = 0):
        if episode_index != 0:
            raise ValueError("the hash-addressed PRO supplement has one pruned init state per task")
        return super().reset()


def load_rows(path: Path, per_category: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    used_tasks = set()
    for category in CATEGORIES:
        category_rows = [
            row
            for row in rows
            if row["perturbation_category"] == category
            and int(row["denoising_steps"]) == 1
            and row.get("supported", True)
            and row.get("variant_applied", True)
        ]
        count = 0
        # Semantic diversity takes precedence over manifest order.  With the
        # default quota this selects one spatial, object, goal, and multi-stage
        # task for every perturbation category.
        suite_order = list(SUITES)
        while count < per_category:
            suite = suite_order[count % len(suite_order)]
            match = next(
                (
                    row
                    for row in category_rows
                    if row["suite"] == suite and row["task_uid"] not in used_tasks
                ),
                None,
            )
            if match is None:
                match = next((row for row in category_rows if row["task_uid"] not in used_tasks), None)
            if match is None:
                break
            selected.append(match)
            used_tasks.add(match["task_uid"])
            count += 1
        if count != per_category:
            raise RuntimeError(f"category {category} supplied {count}/{per_category} unique supported tasks")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/data/rxhuang/wam_overnight")
    parser.add_argument("--run-id", default="mechanism-pro-supplement-d1-diagnostics1248")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tasks-per-category", type=int, default=4)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--t5-embeddings", default=DEFAULT_PRO_T5)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--full-steps", type=int, default=1)
    parser.add_argument("--standalone-steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--standalone-stride", type=int, default=1)
    parser.add_argument("--execute-horizon", type=int, default=16)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--settle-gripper-action", type=float, default=-1.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--capture-future-latent", action="store_true", default=True)
    parser.add_argument("--pro-repo", default=DEFAULT_PRO_REPO)
    parser.add_argument("--pro-config", type=Path, default=DEFAULT_PRO_CONFIG)
    args = parser.parse_args()

    rows = load_rows(args.manifest, args.tasks_per_category)
    install_libero_checkout(Path(args.pro_repo), args.pro_config)
    cfg = build_cfg(args)
    model, dataset_stats = load_model(cfg, args)
    capture = CosmosCheckpointCapture(
        model,
        cfg,
        dataset_stats,
        capture_future_latent=True,
        capture_value=False,
    )
    capture.install()

    out_dir = Path(args.output_root) / "trajectories" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "checkpoints.pt"
    if final_path.exists():
        raise FileExistsError(final_path)

    episodes = []
    timing_path = out_dir / "timing.jsonl"
    with timing_path.open("w", encoding="utf-8") as timing_file:
        for task_index, row in enumerate(rows):
            print(
                f"[pro] {task_index + 1}/{len(rows)} {row['perturbation_category']} {row['task_uid']}",
                flush=True,
            )
            env = ProEnvironment(row, args.resolution)
            try:
                episode = run_episode(
                    env=env,
                    model=model,
                    cfg=cfg,
                    dataset_stats=dataset_stats,
                    capture=capture,
                    task_suite=row["suite"],
                    task_id=int(row["global_task_index"]),
                    episode_index=0,
                    seed=int(row["seed"]),
                    args=args,
                )
            finally:
                env.close()
            episode["domain"] = "libero_pro"
            episode["perturbation_category"] = row["perturbation_category"]
            episode["variant_id"] = row["variant_id"]
            episode["variant_applied"] = bool(row["variant_applied"])
            episode["task_uid"] = row["task_uid"]
            episode["bddl_sha256"] = row["bddl_sha256"]
            episode["init_sha256"] = row["init_sha256"]
            for timing in episode["timing"]:
                timing["episode_id"] = episode["episode_id"]
                timing_file.write(json.dumps(timing) + "\n")
            timing_file.flush()
            episodes.append(episode)
            partial = out_dir / "checkpoints.partial.pt"
            temporary = out_dir / "checkpoints.partial.tmp"
            torch.save(episodes, temporary)
            temporary.replace(partial)

    torch.save(episodes, final_path)
    np.save(
        out_dir / "actions.npy",
        np.concatenate([episode["executed_actions"] for episode in episodes]),
    )
    provenance = run_provenance(
        repos={"cosmos": REPO_ROOT, "libero_pro": args.pro_repo},
        checkpoint=args.checkpoint,
        config=vars(args),
        checkpoint_sha256=KNOWN_CHECKPOINT_SHA256,
    )
    provenance["domain"] = "libero_pro"
    provenance["summary"] = {
        "episodes": len(episodes),
        "tasks": len({episode["task_uid"] for episode in episodes}),
        "states": sum(len(episode["requests"]) for episode in episodes),
        "categories": {category: sum(e["perturbation_category"] == category for e in episodes) for category in CATEGORIES},
        "successes": sum(int(episode["success"]) for episode in episodes),
        "max_final_agreement_abs": max(
            request["final_agreement_max_abs"] for episode in episodes for request in episode["requests"]
        ),
    }
    write_json(out_dir / "metadata.json", provenance)
    print(json.dumps(provenance["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

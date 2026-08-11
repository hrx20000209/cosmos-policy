"""Paired fixed-candidate closed-loop benchmark on the 32-task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.cosmos_adapter import CosmosAdapter  # noqa: E402
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment, config_for_row, install_libero_checkout, load_jsonl  # noqa: E402
from experiments.libero_harness import JsonlWriter, load_yaml, run_episode  # noqa: E402


CHECKPOINT = Path("/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
MODES = ("fresh", "predicted_reuse", "predict_correct")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments/server_deep_validation/server_sweep.yaml")
    args = parser.parse_args()
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash mismatch")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    rows = load_jsonl(args.manifest)
    # Three fixed initial states per task; task assignment is by task, not by
    # state, so paired task-disjoint statistics remain valid.
    selected = [row for row in rows if int(row["init_state_index"]) < 3 and int(row["global_task_index"]) % args.num_shards == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"benchmark_shard{args.shard_index:02d}.json"
    if output_path.exists():
        print(output_path)
        return
    base = load_yaml(args.config)
    first = selected[0] if selected else None
    if first is None:
        return
    install_libero_checkout(Path(first["libero_repo"]), Path(first["libero_config_path"]))
    all_records = []
    started_ns = time.time_ns()
    for mode in MODES:
        adapter = CosmosAdapter({**base["model"], "closed_loop_mode": mode, "predict_correct_steps": 2})
        try:
            for row in selected:
                config = config_for_row({**base, "model": {**base["model"], "closed_loop_mode": mode, "predict_correct_steps": 2}}, row)
                config["evaluation"]["max_steps"] = min(int(row["max_steps"]), args.max_steps)
                env = ManifestLiberoEnvironment(row, 256, None)
                trace_rows = []

                class Writer:
                    def write(self, value):
                        trace_rows.append(value)

                action_path = args.output_dir / "actions" / f"{mode}_{row['episode_key']}.npy"
                try:
                    record, traces = run_episode(adapter, env, config, row["instruction"], int(row["init_state_index"]), int(row["seed"]), Writer(), action_trace_path=action_path)
                finally:
                    env.close()
                all_records.append({
                    **record,
                    "configuration": mode,
                    "task_uid": row["task_uid"],
                    "task_name": row["task_name"],
                    "suite": row["suite"],
                    "split": row["split"],
                    "init_state_index": row["init_state_index"],
                    "seed": row["seed"],
                    "checkpoint_sha256": CHECKPOINT_SHA256,
                    "trace_count": len(traces),
                })
                print(json.dumps({"shard": args.shard_index, "mode": mode, "task": row["task_uid"], "init": row["init_state_index"], "success": record.get("success")}), flush=True)
        finally:
            adapter.close()
    summary = {
        mode: {
            "episodes": sum(item["configuration"] == mode for item in all_records),
            "successes": sum(bool(item.get("success")) for item in all_records if item["configuration"] == mode),
            "success_rate": float(np.mean([bool(item.get("success")) for item in all_records if item["configuration"] == mode])) if any(item["configuration"] == mode for item in all_records) else None,
        }
        for mode in MODES
    }
    payload = {
        "schema_version": 1,
        "experiment": "server_fixed_candidate_closed_loop_benchmark",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "tasks": len({row["task_uid"] for row in selected}),
        "episodes": len(all_records),
        "modes": MODES,
        "summary": summary,
        "records": all_records,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "value_used": False,
        "privileged_runtime_state_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (args.output_dir / f"summary_shard{args.shard_index:02d}.json").write_text(json.dumps({"shard_index": args.shard_index, "episodes": len(all_records), "summary": summary, "finished_at_ns": time.time_ns()}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

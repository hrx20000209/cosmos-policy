"""Full 40-task fixed-seed closed-loop benchmark.

This runner keeps the frozen Fresh-1 target explicit.  The currently
implemented feedback method is recorded as PF/Predict-Correct scaffold; it is
not silently renamed as the one-step late-bound candidate until the oracle and
runtime gates pass.
"""

from __future__ import annotations

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
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import (  # noqa: E402
    ManifestLiberoEnvironment,
    config_for_row,
    install_libero_checkout,
    load_jsonl,
)
from experiments.libero_harness import load_yaml, run_episode  # noqa: E402


CHECKPOINT = Path("/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
MODES = ("fresh", "predicted_reuse", "predict_correct", "predict_correct_async")
METHOD_ROLE = {
    "fresh": "Fresh-1 policy target",
    "predicted_reuse": "simple predicted reuse baseline",
    "predict_correct": "current PF/Predict-Correct scaffold",
    "predict_correct_async": "asynchronous PF scaffold; not final late-bound candidate",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # Accepted for compatibility with server_stage_supervisor; collection
    # artifacts are not used as privileged runtime inputs by this benchmark.
    parser.add_argument("--collection-root", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--inits", type=int, default=5)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments/server_deep_validation/server_sweep.yaml")
    args = parser.parse_args()
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash mismatch")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    rows = load_jsonl(args.manifest)
    selected = [
        row
        for row in rows
        if int(row["init_state_index"]) < args.inits
        and int(row["global_task_index"]) % args.num_shards == args.shard_index
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"benchmark_shard{args.shard_index:02d}.json"
    if output_path.exists():
        print(output_path)
        return
    if not selected:
        return

    install_libero_checkout(Path(selected[0]["libero_repo"]), Path(selected[0]["libero_config_path"]))
    base = load_yaml(args.config)
    all_records: list[dict] = []
    started_ns = time.time_ns()
    for mode in MODES:
        adapter = CosmosAdapter({**base["model"], "closed_loop_mode": mode, "predict_correct_steps": 2})
        try:
            for row in selected:
                config = config_for_row(
                    {**base, "model": {**base["model"], "closed_loop_mode": mode, "predict_correct_steps": 2}},
                    row,
                )
                env = ManifestLiberoEnvironment(row, 256, None)
                trace_rows: list[dict] = []

                class Writer:
                    def write(self, value):
                        trace_rows.append(value)

                action_path = args.output_dir / "actions" / f"{mode}_{row['episode_key']}.npy"
                try:
                    record, traces = run_episode(
                        adapter,
                        env,
                        config,
                        row["instruction"],
                        int(row["init_state_index"]),
                        int(row["seed"]),
                        Writer(),
                        action_trace_path=action_path,
                    )
                finally:
                    env.close()
                all_records.append(
                    {
                        **record,
                        "configuration": mode,
                        "method_role": METHOD_ROLE[mode],
                        "task_uid": row["task_uid"],
                        "task_name": row["task_name"],
                        "suite": row["suite"],
                        "split": row["split"],
                        "init_state_index": row["init_state_index"],
                        "seed": row["seed"],
                        "checkpoint_sha256": CHECKPOINT_SHA256,
                        "trace_count": len(traces),
                        "training_used": False,
                        "value_used": False,
                        "privileged_runtime_state_input": False,
                        "adaptive_scheduler_used": False,
                    }
                )
                print(
                    json.dumps(
                        {
                            "shard": args.shard_index,
                            "mode": mode,
                            "task": row["task_uid"],
                            "init": row["init_state_index"],
                            "success": record.get("success"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        finally:
            adapter.close()

    summary = {
        mode: {
            "episodes": sum(item["configuration"] == mode for item in all_records),
            "successes": sum(bool(item.get("success")) for item in all_records if item["configuration"] == mode),
            "success_rate": float(np.mean([bool(item.get("success")) for item in all_records if item["configuration"] == mode]))
            if any(item["configuration"] == mode for item in all_records)
            else None,
        }
        for mode in MODES
    }
    payload = {
        "schema_version": 2,
        "experiment": "full_scale_40_task_closed_loop_scaffold_benchmark",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "tasks": len({row["task_uid"] for row in selected}),
        "initial_states_per_task": args.inits,
        "episodes": len(all_records),
        "modes": MODES,
        "method_role": METHOD_ROLE,
        "summary": summary,
        "records": all_records,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "value_used": False,
        "privileged_runtime_state_input": False,
        "adaptive_scheduler_used": False,
        "training_used": False,
        "training_branch": "PFT-TinyAdapter not started; isolated fallback only",
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (args.output_dir / f"summary_shard{args.shard_index:02d}.json").write_text(
        json.dumps({"shard_index": args.shard_index, "episodes": len(all_records), "summary": summary, "finished_at_ns": time.time_ns()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

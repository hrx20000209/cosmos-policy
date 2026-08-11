"""Latency-sensitive controlled supplement built on the existing host pipeline.

This measures delay/age/deadline behaviour; it does not claim physical robot
reaction because the optional mid-chunk world-change/action-disturbance hook is
kept separate and is reported as not run unless explicitly added later.
"""

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

from experiments.libero_harness import load_yaml  # noqa: E402
from experiments.overnight.run_async_feedback_assimilation_pipeline import (  # noqa: E402
    CHECKPOINT,
    CHECKPOINT_SHA256,
    PRO_CONFIG,
    PRO_REPO,
    SWEEP_CONFIG,
    build_rows,
    run_episode,
)
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import install_libero_checkout  # noqa: E402
from adapters.cosmos_adapter import CosmosAdapter  # noqa: E402


DELAYS = (0, 50, 100, 150, 200, 300)
MODES = ("sync_fresh", "async_fresh", "predict_correct", "predict_correct_async")
TASK_INDICES = (0, 2, 4, 6, 8, 9)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--collection-root", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=96)
    args = parser.parse_args()
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash mismatch")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    install_libero_checkout(PRO_REPO, PRO_CONFIG)
    base = load_yaml(SWEEP_CONFIG)
    rows = build_rows(list(TASK_INDICES), (0,))
    matrix = [(delay, mode) for delay in DELAYS for mode in MODES]
    assigned = [(index, delay, mode) for index, (delay, mode) in enumerate(matrix) if index % args.num_shards == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for index, delay, mode in assigned:
        output_path = args.output_dir / f"delay{delay:03d}_{mode}.json"
        if output_path.exists():
            completed.append(json.loads(output_path.read_text(encoding="utf-8")))
            continue
        adapter_mode = mode if mode in {"predict_correct", "predict_correct_async"} else ("predicted_reuse" if mode == "predicted_reuse" else "fresh")
        adapter = CosmosAdapter({**base["model"], "closed_loop_mode": adapter_mode, "predict_correct_steps": 2, "async_visual_arrival_delay_ms": float(delay)})
        episodes = []
        started_ns = time.time_ns()
        try:
            for row in rows:
                print(json.dumps({"delay_ms": delay, "mode": mode, "task": row["label"]}, ensure_ascii=False), flush=True)
                episodes.append(
                    run_episode(
                        row,
                        base,
                        mode,
                        action_step_duration_ms=20.0,
                        settle_steps=10,
                        max_steps=args.max_steps,
                        resolution=256,
                        adapter=adapter,
                    )
                )
        finally:
            adapter.close()
        nonterminal = [record for episode in episodes for record in episode["records"] if not record.get("terminal")]
        result = {
            "schema_version": 1,
            "experiment": "server_latency_sensitive_matrix",
            "delay_ms": delay,
            "mode": mode,
            "task_indices": list(TASK_INDICES),
            "max_steps": args.max_steps,
            "episodes": episodes,
            "summary": {
                "episodes": len(episodes),
                "successes": sum(bool(episode["success"]) for episode in episodes),
                "commit_latency_p50_ms": float(np.percentile([record["action_commit_latency_ms"] for record in nonterminal], 50)) if nonterminal else None,
                "commit_latency_p95_ms": float(np.percentile([record["action_commit_latency_ms"] for record in nonterminal], 95)) if nonterminal else None,
                "inference_latency_p95_ms": float(np.percentile([record["inference_latency_ms"] for record in nonterminal], 95)) if nonterminal else None,
                "observation_age_p95_ms": float(np.percentile([record["observation_age_ms"] for record in nonterminal], 95)) if nonterminal else None,
                "action_age_p95_ms": float(np.percentile([record["action_age_at_commit_ms"] for record in nonterminal], 95)) if nonterminal else None,
                "buffer_starvation_rate": float(np.mean([record["buffer_starvation"] for record in nonterminal])) if nonterminal else None,
            },
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "value_used": False,
            "privileged_runtime_state_input": False,
            "adaptive_scheduler_used": False,
            "mid_chunk_world_change_injected": False,
            "action_disturbance_injected": False,
            "started_at_ns": started_ns,
            "finished_at_ns": time.time_ns(),
        }
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        completed.append(result)
    summary = {"schema_version": 1, "shard_index": args.shard_index, "num_shards": args.num_shards, "assigned": len(assigned), "completed": len(completed), "files": [str(args.output_dir / f"delay{delay:03d}_{mode}.json") for _, delay, mode in assigned], "finished_at_ns": time.time_ns()}
    (args.output_dir / f"summary_shard{args.shard_index:02d}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Small task-disjoint closed-loop validation of persistent Predict-Correct WAM.

The first request is the unchanged one-step fresh baseline.  Every later
request starts from the previous joint WAM prediction, runs two denoiser
forwards, and persistently replaces the current visual condition with a
13-frame causal fresh-VAE prefix at the second forward.  There is no scheduler,
threshold, privileged state, finetuning, or Cosmos-value input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.cosmos_adapter import CosmosAdapter  # noqa: E402
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import install_libero_checkout  # noqa: E402
from experiments.libero_harness import load_yaml  # noqa: E402
from experiments.signal_validation.run_closed_loop_phase1 import (  # noqa: E402
    CHECKPOINT,
    EXPECTED_CHECKPOINT_SHA256,
    PRO_CONFIG,
    PRO_REPO,
    SWEEP_CONFIG,
    run_configuration,
)
from experiments.signal_validation.run_closed_loop_phase2a import build_rows  # noqa: E402
from runtime.runtime_metrics import JsonlWriter, ResourceSampler  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if "so101" in str(CHECKPOINT).lower() or "finet" in str(CHECKPOINT).lower():
        raise ValueError("refusing SO101/finetuned checkpoint")
    digest = sha256(CHECKPOINT)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint hash mismatch: {digest}")

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")

    all_rows = build_rows(indices=(0, 1))
    rows = all_rows[args.shard_index :: args.shard_count]
    if not rows:
        raise RuntimeError("empty shard")
    base = load_yaml(SWEEP_CONFIG)
    install_libero_checkout(PRO_REPO, PRO_CONFIG)
    args.raw_output.mkdir(parents=True, exist_ok=True)
    trace_path = args.raw_output / "inference_trace.jsonl"
    if trace_path.exists() or args.output.exists():
        raise FileExistsError("refusing to overwrite an existing closed-loop shard")
    trace_writer = JsonlWriter(trace_path)
    sampler = ResourceSampler(float(base["runtime"]["resource_interval_seconds"]))
    sampler.start()
    adapter = CosmosAdapter(
        {
            **base["model"],
            "closed_loop_mode": "predict_correct",
            "predict_correct_steps": 2,
        }
    )
    started_ns = time.time_ns()
    try:
        result = run_configuration(
            "predict_correct",
            rows,
            base,
            adapter,
            args.raw_output,
            trace_writer,
            sampler,
        )
    finally:
        adapter.close()
        resources = sampler.stop()

    payload = {
        "schema_version": 1,
        "experiment": "persistent_predict_correct_closed_loop",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": digest,
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "task_count": len({row["base_label"] for row in rows}),
        "episode_count": len(rows),
        "task_selection": rows,
        "deployment_baseline_steps": 1,
        "candidate_steps": 2,
        "candidate_arrival_forward_index": 1,
        "causal_vae_prefix_pixel_frames": 13,
        "action_horizon": 16,
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
        "fresh_reference_source": "existing Phase 2A exact paired rows",
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "result": result,
        "resources": resources,
        "trace_path": str(trace_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": len(rows),
                "successes": result["successes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

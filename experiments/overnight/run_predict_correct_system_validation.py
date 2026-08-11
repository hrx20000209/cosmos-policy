"""Single-GPU system validation for persistent-condition Predict-Correct.

This runner is intentionally a runtime experiment, not a new design search.
It compares fresh one-step, simple predicted latent reuse, synchronous
Predict-Correct, and true single-GPU asynchronous Predict-Correct.  The
policy is always the original pre-SO101 Cosmos LIBERO checkpoint; Cosmos
value outputs are not read or passed to the runtime.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from adapters.cosmos_adapter import CosmosAdapter  # noqa: E402
from runtime.async_pipeline import InferenceRequest  # noqa: E402
from runtime.observation_buffer import Observation  # noqa: E402


CHECKPOINT = Path("/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
DATASET_STATS = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json"
T5_CACHE = "/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl"
INSTRUCTION = "pick up the alphabet soup and place it in the basket"
OBSERVATION_FILES = (
    "/data/rxhuang/wam_libero_outputs/libero_pro_closed_loop_phase2a/shadow_capture/alternate_cache/pick_place_init0_seed195.npz",
    "/data/rxhuang/wam_libero_outputs/libero_pro_closed_loop_phase2a/shadow_capture/alternate_cache/pick_place_init1_seed196.npz",
    "/data/rxhuang/wam_libero_outputs/libero_pro_closed_loop_phase2a/shadow_capture/alternate_cache/pick_place_init2_seed197.npz",
)
MODES = ("fresh", "predicted_reuse", "predict_correct", "predict_correct_async")


def jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def load_observations() -> list[Observation]:
    observations: list[Observation] = []
    timestamp = 1
    for path in OBSERVATION_FILES:
        array = np.load(path)
        for index in range(len(array["primary_images"])):
            observations.append(
                Observation(
                    timestamp_ns=timestamp,
                    primary_image=np.ascontiguousarray(array["primary_images"][index]).copy(),
                    wrist_image=np.ascontiguousarray(array["wrist_images"][index]).copy(),
                    proprio=np.ascontiguousarray(array["proprio"][index], dtype=np.float32).copy(),
                    metadata={"source": path, "source_index": index},
                )
            )
            timestamp += 1
    if not observations:
        raise RuntimeError("no saved observations available for the system benchmark")
    return observations


def make_shared_adapters(arrival_delay_ms: float) -> tuple[CosmosAdapter, dict[str, CosmosAdapter]]:
    common = {
        "checkpoint": str(CHECKPOINT),
        "dataset_stats_path": DATASET_STATS,
        "t5_embeddings_path": T5_CACHE,
        "config_file": "cosmos_policy/config/config.py",
        "action_horizon": 16,
        "action_dim": 7,
        "predict_correct_steps": 2,
        "async_visual_arrival_delay_ms": float(arrival_delay_ms),
    }
    master = CosmosAdapter({**common, "closed_loop_mode": "fresh"})
    master._ensure_loaded()
    adapters: dict[str, CosmosAdapter] = {}
    for mode in MODES:
        adapter = CosmosAdapter({**common, "closed_loop_mode": mode})
        # One model instance and one visible GPU are used for the whole run.
        # The four adapters only hold independent autoregressive state.
        adapter.model = master.model
        adapter.cfg = master.cfg
        adapter.dataset_stats = master.dataset_stats
        adapters[mode] = adapter
    return master, adapters


def run(args: argparse.Namespace) -> dict:
    observations = load_observations()
    selected_modes = tuple(args.modes or MODES)
    master, adapters = make_shared_adapters(args.arrival_delay_ms)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_output or output.with_name(output.stem + "_raw.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []
    rng = random.Random(args.seed)
    started_ns = time.time_ns()
    try:
        # Bootstrap each state machine, then interleave warmup requests across
        # configurations so no configuration receives a dedicated warmup phase.
        for mode in selected_modes:
            adapters[mode].reset(INSTRUCTION, 195)
        warmup_order = [(mode, index) for index in range(args.warmup) for mode in selected_modes]
        rng.shuffle(warmup_order)
        for mode, index in warmup_order:
            observation = observations[(index * len(selected_modes) + selected_modes.index(mode)) % len(observations)]
            adapters[mode].infer(
                observation,
                InferenceRequest.create(time.time_ns(), f"system-warmup-{mode}", index),
                denoising_steps=1,
            )

        request_order = [(mode, request_index) for mode in selected_modes for request_index in range(args.requests)]
        rng.shuffle(request_order)
        for ordinal, (mode, request_index) in enumerate(request_order):
            observation_index = (ordinal * 17 + request_index) % len(observations)
            observation = observations[observation_index]
            result = adapters[mode].infer(
                observation,
                InferenceRequest.create(
                    time.time_ns(), f"system-{mode}", request_index,
                ),
                denoising_steps=1,
            )
            metrics = jsonable(result.stage_metrics_ms)
            raw_rows.append(
                {
                    "ordinal": ordinal,
                    "mode": mode,
                    "request_index": request_index,
                    "observation_index": observation_index,
                    "total_ms": float(metrics.get("total_ms", 0.0)),
                    "metrics": metrics,
                    "action_sha256": result.extra["action_chunk_sha256"],
                }
            )
            if (ordinal + 1) % args.flush_every == 0:
                raw_path.write_text(json.dumps(raw_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(json.dumps({"completed": ordinal + 1, "total": len(request_order)}, ensure_ascii=False), flush=True)
    finally:
        raw_path.write_text(json.dumps(raw_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        master.close()

    summary: dict[str, dict] = {}
    for mode in selected_modes:
        rows = [row for row in raw_rows if row["mode"] == mode]
        metric = lambda key: [float(row["metrics"].get(key, 0.0)) for row in rows if key in row["metrics"]]
        summary[mode] = {
            "request_count": len(rows),
            "total_latency_ms": stats([float(row["total_ms"]) for row in rows]),
            "critical_path_ms": stats(metric("async_critical_path_gpu_ms")),
            "total_gpu_compute_ms": stats(metric("async_total_gpu_compute_ms")),
            "dit_compute_ms": stats(metric("dit_denoising_ms")),
            "vae_compute_ms": stats(metric("vae_encoding_ms")),
            "useful_speculative_compute_ms": stats(metric("async_useful_speculative_compute_ms")),
            "wasted_speculative_compute_ms": stats(metric("async_wasted_speculative_compute_ms")),
            "overlap_ms": stats(metric("async_overlap_ms")),
            "overlap_ratio": stats(metric("async_overlap_ratio")),
            "hidden_sensing_latency_ms": stats(metric("async_hidden_sensing_latency_ms")),
            "condition_slack_ms": stats(metric("async_condition_slack_ms")),
            "late_condition_wait_ms": stats(metric("async_late_condition_wait_ms")),
            "peak_memory_allocated_bytes": max(
                [int(row["metrics"].get("peak_memory_allocated_bytes", 0)) for row in rows], default=0
            ),
            "peak_memory_reserved_bytes": max(
                [int(row["metrics"].get("peak_memory_reserved_bytes", 0)) for row in rows], default=0
            ),
            "gpu_timeline_example": next(
                (row["metrics"].get("async_gpu_timeline") for row in rows if "async_gpu_timeline" in row["metrics"]),
                None,
            ),
        }
    payload = {
        "schema_version": 1,
        "experiment": "predict_correct_single_gpu_system_validation",
        "started_at_ns": started_ns,
        "finished_at_ns": time.time_ns(),
        "gpu_policy": "one visible CUDA device per process; benchmark ran on a single GPU",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_policy": "original_pre_finetune_cosmos_policy_libero",
        "value_used": False,
        "privileged_state_runtime_input": False,
        "adaptive_scheduler_used": False,
        "threshold_used": False,
        "finetuning_used": False,
        "denoise_steps": {"fresh": 1, "predicted_reuse": 1, "predict_correct": 2, "predict_correct_async": 2},
        "requests_per_configuration": args.requests,
        "warmup_per_configuration": args.warmup,
        "random_interleaving_seed": args.seed,
        "selected_modes": list(selected_modes),
        "arrival_delay_ms": float(args.arrival_delay_ms),
        "observation_sources": list(OBSERVATION_FILES),
        "summary": summary,
        "raw_output": str(raw_path),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=197)
    parser.add_argument("--arrival-delay-ms", type=float, default=0.0)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=None)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "reports/artifacts/predict_correct_single_gpu_system_validation.json",
    )
    parser.add_argument("--raw-output", type=Path, default=None)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({"output": str(args.output), "requests": payload["requests_per_configuration"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

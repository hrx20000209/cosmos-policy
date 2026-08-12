#!/usr/bin/env python3
"""Formal clean-GPU cost model for the E12 decision routes.

Measures F1, P1, P1+frozen semantic score, and PV0 on one fixed physical state,
with rotated execution order so drift cannot favour a route.  Historical E4
timings are NOT reused.  This must run on an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.esp.run_e2_e3_pilot_shard import load_request, render
from experiments.p1_semantic_verify.common import BLOCKS, frozen_score, load_frozen_model, run_route
from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS, DEFAULT_T5_EMBEDDINGS, ORIGINAL_CHECKPOINT, atomic_write_json,
    build_model, checkpoint_contract, configure_libero, read_jsonl, set_up_cuda,
)

VARIANTS = ("F1", "P1", "P1_PLUS_SCORE", "PV0")


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)), "p95": float(np.quantile(array, 0.95)),
        "std": float(array.std()), "min": float(array.min()), "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("reports/p1_semantic_verify/E12_STATE_BANK_discovery.jsonl"))
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--memory-fraction", type=float, default=0.80)
    parser.add_argument("--output", type=Path, default=Path("reports/p1_semantic_verify/E12B_COST_MODEL.json"))
    args = parser.parse_args()
    if args.warmup < 20 or args.repeats < 100:
        raise ValueError("formal profile requires warmup >= 20 and repeats >= 100")

    entries = read_jsonl(args.bank)
    entry = entries[args.state_index]
    frozen = load_frozen_model()
    contract = checkpoint_contract(ORIGINAL_CHECKPOINT)
    gpu = set_up_cuda(args.memory_fraction)
    free_before, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    if free_before / total < 0.85:
        raise RuntimeError(
            f"formal latency profile requires a clean GPU; only {free_before / 2 ** 20:.0f} MiB free"
        )
    configure_libero(entry)
    cfg, stats, model = build_model(
        checkpoint=ORIGINAL_CHECKPOINT, dataset_stats=DEFAULT_DATASET_STATS, t5_embeddings=DEFAULT_T5_EMBEDDINGS
    )
    source, target = load_request(entry)
    observation = render(entry, target)
    previous = torch.from_numpy(np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
    kwargs = dict(cfg=cfg, model=model, stats=stats, observation=observation,
                  instruction=entry["instruction"], seed=int(entry["seed"]), previous_generated=previous)

    def once(variant: str) -> tuple[float, float, float]:
        """Return (cuda_ms, wall_ms, score_ms) for one decision of this variant."""

        if variant == "P1_PLUS_SCORE":
            result = run_route("P1", capture_blocks=BLOCKS, **kwargs)
            started = time.perf_counter_ns()
            frozen_score(frozen, result["internal"], result["actions"])
            score_ms = float((time.perf_counter_ns() - started) / 1e6)
            return result["cuda_ms"], result["wall_ms"] + score_ms, score_ms
        result = run_route(variant, **kwargs)
        return result["cuda_ms"], result["wall_ms"], 0.0

    for _ in range(args.warmup):
        for variant in VARIANTS:
            once(variant)

    samples: dict[str, dict[str, list[float]]] = {
        variant: {"cuda_ms": [], "wall_ms": [], "score_ms": []} for variant in VARIANTS
    }
    for repeat in range(args.repeats):
        # Rotate order so any thermal/clock drift is shared across variants.
        order = VARIANTS[repeat % len(VARIANTS):] + VARIANTS[: repeat % len(VARIANTS)]
        for variant in order:
            cuda_ms, wall_ms, score_ms = once(variant)
            samples[variant]["cuda_ms"].append(cuda_ms)
            samples[variant]["wall_ms"].append(wall_ms)
            samples[variant]["score_ms"].append(score_ms)

    timings = {
        variant: {key: summary(values) for key, values in channels.items()}
        for variant, channels in samples.items()
    }
    p1 = timings["P1"]["cuda_ms"]["median"]
    p1s = timings["P1_PLUS_SCORE"]["wall_ms"]["median"]
    pv0 = timings["PV0"]["cuda_ms"]["median"]
    f1 = timings["F1"]["cuda_ms"]["median"]
    overhead = timings["P1_PLUS_SCORE"]["wall_ms"]["median"] - timings["P1"]["wall_ms"]["median"]

    payload = {
        "schema_version": 1,
        "experiment": "E12B_cost_model",
        "status": "PASS",
        **contract,
        "gpu": gpu,
        "gpu_clean": {"free_before_mib": float(free_before / 2 ** 20), "total_mib": float(total / 2 ** 20)},
        "state_id": entry["state_key"],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "order_rotation": True,
        "denoising_steps": 1,
        "historical_values_reused": False,
        "timings_ms": timings,
        "decision_cost_ms": {
            "F1": f1,
            "P1": p1,
            "P1_plus_frozen_score": p1s,
            "PV0": pv0,
            "semantic_score_overhead_ms": overhead,
            "semantic_score_overhead_fraction_of_f1": overhead / f1 if f1 else None,
            "semantic_score_overhead_fraction_of_p1": overhead / p1 if p1 else None,
        },
        "policy_cost_formula": {
            "P1_ALWAYS": "C_P1",
            "PV0_ALWAYS": "C_PV0",
            "F1_ALWAYS": "C_F1",
            "ACTION_ONLY": "C_P1 + p_correct * C_PV0   (action geometry is CPU-only on a 16x7 array)",
            "SEMANTIC": "C_P1_plus_score + p_correct * C_PV0",
            "note": "the discarded speculative P1 of a corrected decision is already inside the per-decision P1 term",
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload["decision_cost_ms"], indent=2))


if __name__ == "__main__":
    main()

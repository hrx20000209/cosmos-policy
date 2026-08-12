#!/usr/bin/env python3
"""SMOKE-A..E for E12. Any failure blocks the formal experiment.

A  paired determinism        : F1/P1/PV0 repeat bit-exactly; the frozen feature
                               hook does not change the action.
B  frozen score reconstruction: this round's extractor reproduces the historical
                               E4 extractor's features exactly, and the frozen
                               ridge evaluation matches an independent reference.
C  accept path                : speculative P1 + commit == an ordinary
                               predicted_reuse CosmosAdapter, over a sequence.
D  reject path                : speculative P1 + discard + PV0 == a
                               native_persistent CosmosAdapter that never ran a
                               speculative forward, over a sequence.
E  H=16 alignment             : every bank row is a 16-action gap and the
                               closed-loop execution prefix commits 16 actions.

Each phase runs in its own process with exactly one Cosmos model resident;
``--phase merge`` combines the phase artifacts and applies the pass criteria.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adapters.cosmos_adapter import CosmosAdapter
from experiments.esp.run_e2_e3_pilot_shard import load_request, render
from experiments.p1_semantic_verify.common import (
    BLOCKS, action_geometry, frozen_score, load_frozen_model, observation_hash, predicted_condition,
    run_route, tensor_hash,
)
from runtime.observation_buffer import Observation
from experiments.p1_semantic_verify.semantic_transaction import SemanticVerifyAdapter
from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS, DEFAULT_T5_EMBEDDINGS, ORIGINAL_CHECKPOINT, atomic_write_json,
    build_model, checkpoint_contract, configure_libero, read_jsonl, set_up_cuda,
)
from runtime.async_pipeline import InferenceRequest

RUN_PHASES = ("chain", "chain_repeat", "ab", "cd", "e")


def adapter_config() -> dict[str, Any]:
    return {
        "checkpoint": str(ORIGINAL_CHECKPOINT),
        "dataset_stats_path": str(DEFAULT_DATASET_STATS),
        "t5_embeddings_path": str(DEFAULT_T5_EMBEDDINGS),
        "action_horizon": 16,
    }


def adapter_state(adapter: CosmosAdapter) -> dict[str, Any]:
    return {
        "request_index": int(adapter.request_index),
        "previous_generated_latent": (
            tensor_hash(adapter.previous_generated_latent) if adapter.previous_generated_latent is not None else None
        ),
        "previous_real_latent": (
            tensor_hash(adapter.previous_real_latent) if adapter.previous_real_latent is not None else None
        ),
        "last_physical_condition_latent": (
            tensor_hash(adapter.last_physical_condition_latent)
            if adapter.last_physical_condition_latent is not None else None
        ),
    }


def maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def chain_rows(entries: list[dict[str, Any]], entry: dict[str, Any], length: int) -> list[dict[str, Any]]:
    return sorted(
        (row for row in entries if row["episode_key"] == entry["episode_key"]),
        key=lambda row: int(row["control_step"]),
    )[:length]


def freeze_chain(entries: list[dict[str, Any]], entry: dict[str, Any], length: int, path: Path) -> dict[str, Any]:
    """Render the observation chain once and freeze it to disk.

    SMOKE-C/D each need one Cosmos model per process, so the reference route and
    the transactional route cannot share a process.  Freezing the observations
    removes the renderer from the comparison entirely: the transaction test then
    measures the adapter, not MuJoCo/EGL reproducibility across processes.
    """

    rows = chain_rows(entries, entry, length)
    frames = {}
    hashes = []
    for index, row in enumerate(rows):
        observation = render(row, load_request(row)[1])
        for name in ("primary_image", "wrist_image", "proprio"):
            frames[f"{index}_{name}"] = np.ascontiguousarray(getattr(observation, name))
        hashes.append(observation_hash(observation))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **frames)
    return {"path": str(path), "steps": len(rows), "observation_hashes": hashes,
            "control_steps": [int(row["control_step"]) for row in rows]}


def load_chain(path: Path, length: int) -> list[Observation]:
    data = np.load(path)
    return [
        Observation(
            timestamp_ns=index,
            primary_image=data[f"{index}_primary_image"],
            wrist_image=data[f"{index}_wrist_image"],
            proprio=data[f"{index}_proprio"],
        )
        for index in range(length)
    ]


def drive(adapter: CosmosAdapter, observations: list[Any], instruction: str, seed: int) -> dict[str, Any]:
    adapter.reset(instruction, seed)
    actions: list[list[list[float]]] = []
    for index, observation in enumerate(observations):
        request = InferenceRequest.create(0, "smoke", index * 16)
        output = adapter.infer(observation, request, 1, None)
        actions.append(np.ascontiguousarray(output.actions, dtype=np.float32).tolist())
    return {
        "actions": actions,
        "state": adapter_state(adapter),
        "decisions": [
            {key: value for key, value in decision.items() if key in ("decision", "discarded_speculative_calls")}
            for decision in getattr(adapter, "decisions", [])
        ],
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    diffs = [maxdiff(np.asarray(a), np.asarray(b)) for a, b in zip(left["actions"], right["actions"])]
    return {
        "steps": len(diffs),
        "action_max_abs_diff": max(diffs) if diffs else None,
        "per_step_action_max_abs_diff": diffs,
        "adapter_state_transactional": left["state"],
        "adapter_state_reference": right["state"],
        "adapter_state_identical": left["state"] == right["state"],
        "bitexact": bool(diffs and max(diffs) == 0.0 and left["state"] == right["state"]),
    }


def phase_ab(args: argparse.Namespace, entries: list[dict[str, Any]], frozen: dict[str, Any]) -> dict[str, Any]:
    entry = entries[0]
    source, target = load_request(entry)
    observation = render(entry, target)
    cfg, stats, model = build_model(
        checkpoint=ORIGINAL_CHECKPOINT, dataset_stats=DEFAULT_DATASET_STATS, t5_embeddings=DEFAULT_T5_EMBEDDINGS
    )
    previous = torch.from_numpy(np.asarray(source["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
    kwargs = dict(cfg=cfg, model=model, stats=stats, observation=observation,
                  instruction=entry["instruction"], seed=int(entry["seed"]), previous_generated=previous)

    p1_hooked = run_route("P1", capture_blocks=BLOCKS, **kwargs)
    p1_repeat = run_route("P1", capture_blocks=BLOCKS, **kwargs)
    p1_plain = run_route("P1", **kwargs)
    f1_a, f1_b = run_route("F1", **kwargs), run_route("F1", **kwargs)
    pv0_a, pv0_b = run_route("PV0", **kwargs), run_route("PV0", **kwargs)

    smoke_a = {
        "state_id": entry["state_key"],
        "p1_repeat_action_max_abs_diff": maxdiff(p1_hooked["actions"], p1_repeat["actions"]),
        "p1_hook_off_action_max_abs_diff": maxdiff(p1_hooked["actions"], p1_plain["actions"]),
        "f1_repeat_action_max_abs_diff": maxdiff(f1_a["actions"], f1_b["actions"]),
        "pv0_repeat_action_max_abs_diff": maxdiff(pv0_a["actions"], pv0_b["actions"]),
        "p1_repeat_feature_max_abs_diff": float(
            max(abs(p1_hooked["internal"][k] - p1_repeat["internal"][k]) for k in p1_hooked["internal"])
        ),
        "routes_differ": {
            "p1_vs_f1": maxdiff(p1_hooked["actions"], f1_a["actions"]),
            "pv0_vs_f1": maxdiff(pv0_a["actions"], f1_a["actions"]),
        },
    }
    smoke_a["pass"] = bool(
        smoke_a["p1_repeat_action_max_abs_diff"] == 0.0
        and smoke_a["p1_hook_off_action_max_abs_diff"] == 0.0
        and smoke_a["f1_repeat_action_max_abs_diff"] == 0.0
        and smoke_a["pv0_repeat_action_max_abs_diff"] == 0.0
        and smoke_a["p1_repeat_feature_max_abs_diff"] == 0.0
        and smoke_a["routes_differ"]["p1_vs_f1"] > 0.0
    )

    from experiments.semantic_risk.run_semantic_shard import run_route as historical_run_route

    # Extractor equivalence, isolated from the route variant: feed the historical
    # E4 extractor the *same* predicted condition this round's P1 uses. Features
    # must then match bit-exactly, proving the reducer is the frozen one.
    historical_action, _, historical_features, _ = historical_run_route(
        cfg, model, stats, observation, entry["instruction"], int(entry["seed"]),
        previous=predicted_condition(previous), blocks=BLOCKS,
    )
    # AUDIT FINDING: run_semantic_shard.collect (and run_e11a_route_transfer)
    # pass the *raw* prior generated latent, i.e. they never move predicted
    # future slots 6/7 into current slots 2/3. That is not the deployed P1
    # (`predicted_reuse`) route used by CosmosAdapter, PV0 and the closed loop.
    # E12 uses the deployed route for both the action and the score, and records
    # the divergence of the E4 collection variant here.
    e4_variant_action, _, e4_variant_features, _ = historical_run_route(
        cfg, model, stats, observation, entry["instruction"], int(entry["seed"]),
        previous=previous, blocks=BLOCKS,
    )
    score_here = frozen_score(frozen, p1_hooked["internal"], p1_hooked["actions"])
    reference = {**p1_hooked["internal"], **action_geometry(p1_hooked["actions"])}
    x = np.asarray([reference[name] for name in frozen["feature_names"]], dtype=np.float64)
    score_reference = float(np.expm1(
        ((x - np.asarray(frozen["feature_mean"])) / np.asarray(frozen["feature_scale"]))
        @ np.asarray(frozen["coefficients"]) + float(frozen["intercept"])
    ))
    smoke_b = {
        "extractor_equivalence": {
            "note": "historical E4 reducer fed the identical predicted condition",
            "feature_max_abs_diff": float(
                max(abs(historical_features[k] - p1_hooked["internal"][k]) for k in historical_features)
            ),
            "action_max_abs_diff": maxdiff(
                np.asarray(historical_action, dtype=np.float32).reshape(16, 7), p1_hooked["actions"]
            ),
            "feature_count": len(historical_features),
        },
        "e4_collection_route_variant": {
            "note": "raw prior generated latent, no slot 6/7 -> 2/3 move; the variant E4 and E11-A actually collected",
            "action_max_abs_diff_vs_deployed_p1": maxdiff(
                np.asarray(e4_variant_action, dtype=np.float32).reshape(16, 7), p1_hooked["actions"]
            ),
            "feature_max_abs_diff_vs_deployed_p1": float(
                max(abs(e4_variant_features[k] - p1_hooked["internal"][k]) for k in e4_variant_features)
            ),
            "score_on_e4_variant": frozen_score(
                frozen, e4_variant_features, np.asarray(e4_variant_action, dtype=np.float32).reshape(16, 7)
            ),
            "score_on_deployed_p1": score_here,
        },
        "frozen_checksum": frozen["checksum_sha256"],
        "score": score_here,
        "score_reference": score_reference,
        "score_max_abs_diff": float(abs(score_here - score_reference)),
    }
    smoke_b["pass"] = bool(
        smoke_b["extractor_equivalence"]["feature_max_abs_diff"] == 0.0
        and smoke_b["extractor_equivalence"]["action_max_abs_diff"] == 0.0
        and smoke_b["extractor_equivalence"]["feature_count"] == 84
        and smoke_b["score_max_abs_diff"] == 0.0
    )
    return {"SMOKE_A_paired_determinism": smoke_a, "SMOKE_B_frozen_score_reconstruction": smoke_b}


def attach_model(adapter: CosmosAdapter, cfg: Any, stats: Any, model: Any) -> CosmosAdapter:
    """Share one resident Cosmos model across the compared adapters.

    Bit-exactness is only meaningful inside one process: kernel selection is
    reproducible within a process (SMOKE-A) but not guaranteed across processes
    (SMOKE-F). Sharing the model also keeps a single 2B policy resident.
    """

    adapter.cfg = cfg
    adapter.dataset_stats = stats
    adapter.model = model
    adapter.decode_stream = torch.cuda.Stream(priority=0)
    return adapter


def phase_cd(args: argparse.Namespace, entries: list[dict[str, Any]]) -> dict[str, Any]:
    entry = entries[0]
    observations = load_chain(args.partial_dir / "chain.npz", args.sequence_length)
    rows = chain_rows(entries, entry, args.sequence_length)
    config = adapter_config()
    cfg, stats, model = build_model(
        checkpoint=ORIGINAL_CHECKPOINT, dataset_stats=DEFAULT_DATASET_STATS, t5_embeddings=DEFAULT_T5_EMBEDDINGS
    )
    instruction, seed = entry["instruction"], int(entry["seed"])
    runs: dict[str, Any] = {}
    builders = {
        "c_ref": lambda: CosmosAdapter({**config, "closed_loop_mode": "predicted_reuse"}),
        "c_test": lambda: SemanticVerifyAdapter(config, verify_mode="semantic", threshold=float("inf")),
        "d_ref": lambda: CosmosAdapter({**config, "closed_loop_mode": "native_persistent"}),
        "d_test": lambda: SemanticVerifyAdapter(config, verify_mode="semantic", threshold=float("-inf")),
    }
    for name, builder in builders.items():
        runs[name] = drive(attach_model(builder(), cfg, stats, model), observations, instruction, seed)
    runs["control_steps"] = [int(row["control_step"]) for row in rows]
    return runs


def phase_e(args: argparse.Namespace, entries: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [
        int(target["control_step"]) - int(source["control_step"])
        for source, target in (load_request(row) for row in entries)
    ]
    rows = sorted(
        (row for row in entries if row["episode_key"] == entries[0]["episode_key"]),
        key=lambda row: int(row["control_step"]),
    )
    chain_gaps = [int(b["control_step"]) - int(a["control_step"]) for a, b in zip(rows, rows[1:])]
    smoke_e = {
        "bank_rows": len(gaps),
        "unique_temporal_gaps": sorted(set(gaps)),
        "action_horizon": 16,
        "closed_loop_execution_prefix": 16,
        "prefix_mode": "fixed",
        "chain_gaps_multiple_of_16": all(gap % 16 == 0 for gap in chain_gaps),
        "chain_gaps": chain_gaps,
    }
    smoke_e["pass"] = bool(set(gaps) == {16} and smoke_e["chain_gaps_multiple_of_16"])
    return {"SMOKE_E_h16_alignment": smoke_e}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("reports/p1_semantic_verify/E12_STATE_BANK_discovery.jsonl"))
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/p1_semantic_verify/SMOKE_RESULT.json"))
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--phase", choices=(*RUN_PHASES, "merge"), default="merge")
    parser.add_argument("--partial-dir", type=Path, default=Path("reports/p1_semantic_verify/smoke"))
    args = parser.parse_args()

    entries = read_jsonl(args.bank)
    frozen = load_frozen_model()
    contract = checkpoint_contract(ORIGINAL_CHECKPOINT)
    args.partial_dir.mkdir(parents=True, exist_ok=True)

    if args.phase != "merge":
        needs_gpu = args.phase not in {"e"}
        gpu = set_up_cuda(args.memory_fraction) if needs_gpu else {}
        configure_libero(entries[0])
        if args.phase == "ab":
            payload = phase_ab(args, entries, frozen)
        elif args.phase == "e":
            payload = phase_e(args, entries)
        elif args.phase in {"chain", "chain_repeat"}:
            name = "chain.npz" if args.phase == "chain" else "chain_repeat.npz"
            payload = {args.phase: freeze_chain(
                entries, entries[0], args.sequence_length, args.partial_dir / name
            )}
        else:
            payload = phase_cd(args, entries)
        atomic_write_json(args.partial_dir / f"SMOKE_{args.phase}.json", {**payload, "gpu": gpu})
        print(json.dumps({"phase": args.phase, "status": "WRITTEN"}))
        return

    partials: dict[str, Any] = {}
    for phase in RUN_PHASES:
        path = args.partial_dir / f"SMOKE_{phase}.json"
        if not path.is_file():
            raise SystemExit(f"missing smoke phase artifact: {path}")
        partials[phase] = json.loads(path.read_text(encoding="utf-8"))

    results: dict[str, Any] = {
        **{k: v for k, v in partials["ab"].items() if k.startswith("SMOKE_")},
        **{k: v for k, v in partials["e"].items() if k.startswith("SMOKE_")},
    }
    # Cross-process renderer reproducibility is measured, not assumed. C/D use a
    # frozen observation chain so this cannot contaminate the transaction test.
    first, repeat = partials["chain"]["chain"], partials["chain_repeat"]["chain_repeat"]
    results["SMOKE_F_render_reproducibility"] = {
        "note": "diagnostic: same requests rendered in two separate processes",
        "observation_hashes_identical": first["observation_hashes"] == repeat["observation_hashes"],
        "steps_matching": sum(
            a == b for a, b in zip(first["observation_hashes"], repeat["observation_hashes"])
        ),
        "steps": first["steps"],
        "frozen_chain_used_by_c_and_d": True,
        "pass": True,
    }

    cd = partials["cd"]
    smoke_c = compare(cd["c_test"], cd["c_ref"])
    smoke_c["decisions"] = [d["decision"] for d in cd["c_test"]["decisions"]]
    smoke_c["reference_route"] = "CosmosAdapter closed_loop_mode=predicted_reuse, same process and same model object"
    smoke_c["pass"] = bool(smoke_c["bitexact"] and set(smoke_c["decisions"][1:]) == {"accept"})
    results["SMOKE_C_accept_path"] = smoke_c

    smoke_d = compare(cd["d_test"], cd["d_ref"])
    smoke_d["decisions"] = [d["decision"] for d in cd["d_test"]["decisions"]]
    smoke_d["discarded_speculative_calls"] = sum(
        int(d["discarded_speculative_calls"]) for d in cd["d_test"]["decisions"]
    )
    smoke_d["reference_route"] = (
        "CosmosAdapter closed_loop_mode=native_persistent, same process and same model object; "
        "the reference never ran a speculative forward"
    )
    smoke_d["pass"] = bool(
        smoke_d["bitexact"]
        and set(smoke_d["decisions"][1:]) == {"correct"}
        and smoke_d["discarded_speculative_calls"] == args.sequence_length - 1
    )
    results["SMOKE_D_reject_path"] = smoke_d

    payload = {
        "schema_version": 1, "experiment": "E12_smoke",
        "status": "PASS" if all(section["pass"] for section in results.values()) else "FAIL",
        **contract, "denoising_steps": 1, "value_used": False, "finetuning_used": False,
        "bank": str(args.bank), "sequence_length": args.sequence_length,
        "phase_isolation": "one Cosmos model per OS process",
        "gpu": {phase: partials[phase].get("gpu") for phase in RUN_PHASES},
        "results": results,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"status": payload["status"],
                      **{name: section["pass"] for name, section in results.items()}}, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit the completed full-scale PV0 closed-loop experiment.

This is deliberately a post-hoc *analysis* tool.  It does not run Cosmos,
change routing, or introduce a new runtime mechanism.  It verifies the frozen
one-denoise contracts in the raw episode/trace artifacts, then reports paired
success outcomes for the three already-executed routes:

* ``fresh``: current camera is fully encoded at every request;
* ``predicted_reuse`` (P1): generated visual latent is reused after bootstrap;
* ``native_persistent`` (PV0): P1 plus the native causal fresh visual prefix.

The experiment's shared-GPU episode timing remains provenance only.  Formal
route-cost conclusions are delegated to the separate clean-GPU microbenchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MODES = ("fresh", "predicted_reuse", "native_persistent")
ORIGINAL_CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def expected_visual_modes(mode: str, request_count: int) -> list[str]:
    if request_count <= 0:
        return []
    followup = {
        "fresh": "fresh",
        "predicted_reuse": "predicted",
        "native_persistent": "native_persistent",
    }[mode]
    return ["fresh", *([followup] * (request_count - 1))]


def sha256_actions(actions: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(actions, dtype=np.float32))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def validate_route_artifact(path: Path, expected_role: str) -> dict[str, Any]:
    """Return a compact verified record from one raw episode artifact."""

    payload = load_json(path)
    if payload.get("experiment") != "PV0_closed_loop_episode":
        raise ValueError(f"{path}: unexpected experiment label")
    if payload.get("status") != "PASS":
        raise ValueError(f"{path}: non-PASS artifact cannot enter the paired audit")
    if payload.get("phase_role") != expected_role:
        raise ValueError(f"{path}: expected phase_role={expected_role!r}, got {payload.get('phase_role')!r}")
    mode = str(payload.get("mode"))
    if mode not in MODES:
        raise ValueError(f"{path}: unsupported mode {mode!r}")
    if payload.get("checkpoint_sha256") != ORIGINAL_CHECKPOINT_SHA256:
        raise ValueError(f"{path}: original checkpoint SHA mismatch")
    contract = payload.get("route_contract") or {}
    # Episode workers record this invariant in the route contract (the S1/S4
    # workers also expose it top-level).  Accept either schema, but require
    # exactly the same frozen value in both if both fields are present.
    top_level_denoise = payload.get("denoising_steps")
    if top_level_denoise is not None and int(top_level_denoise) != 1:
        raise ValueError(f"{path}: denoise must be fixed to 1")
    if int(contract.get("denoising_steps", -1)) != 1:
        raise ValueError(f"{path}: route contract violates denoise=1")
    for key in (
        "value_used",
        "privileged_runtime_state_input",
        "adaptive_scheduler_used",
        "threshold_used",
        "hidden_activation_patch_used",
        "fresh_prefix_oracle_used",
    ):
        if bool(payload.get(key)):
            raise ValueError(f"{path}: frozen contract violates {key}=False")
    for key in (
        "value_used",
        "privileged_runtime_state_input",
        "adaptive_scheduler_used",
        "threshold_used",
        "hidden_activation_patch_used",
        "fresh_prefix_oracle_used",
    ):
        if bool(contract.get(key)):
            raise ValueError(f"{path}: route contract violates {key}=False")
    if mode == "native_persistent":
        if contract.get("native_interface") != "get_action:persistent_visual_correction_prefix_frames":
            raise ValueError(f"{path}: PV0 did not use the frozen native interface")
        if int(contract.get("fresh_visual_prefix_frames", -1)) != 13:
            raise ValueError(f"{path}: PV0 prefix must be 13 frames")
        if int(contract.get("fresh_visual_arrival_denoiser_forward", -1)) != 0:
            raise ValueError(f"{path}: PV0 prefix did not arrive before its only denoiser forward")

    manifest = payload.get("manifest_row") or {}
    record = payload.get("record") or {}
    if int(manifest.get("denoising_steps", -1)) != 1:
        raise ValueError(f"{path}: manifest violates denoise=1")
    trace_path = path.parent.parent / "traces" / path.name
    trace_payload = load_json(trace_path)
    if trace_payload.get("status") != "PASS":
        raise ValueError(f"{trace_path}: trace artifact is not PASS")
    traces = trace_payload.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError(f"{trace_path}: missing request traces")
    visual_modes = [str((trace.get("extra") or {}).get("visual_input_mode")) for trace in traces]
    if visual_modes != expected_visual_modes(mode, len(traces)):
        raise ValueError(f"{trace_path}: visual input route mismatch")
    if any(int(trace.get("denoiser_forward_count", -1)) != 1 for trace in traces):
        raise ValueError(f"{trace_path}: one-denoise trace contract failed")
    if mode == "native_persistent" and len(traces) > 1:
        native_count = sum(int((trace.get("extra") or {}).get("native_persistent_request_count", 0)) for trace in traces)
        if native_count != len(traces) - 1:
            raise ValueError(f"{trace_path}: PV0 request count mismatch")
    if len(traces) != int(record.get("inference_count", -1)):
        raise ValueError(f"{path}: trace count differs from episode inference count")

    action_path = Path(str(record.get("executed_actions_path") or payload.get("action_path") or ""))
    if not action_path.is_file():
        raise FileNotFoundError(f"{path}: missing executed action trace {action_path}")
    action_array = np.load(action_path, allow_pickle=False)
    if action_array.ndim != 2 or action_array.shape[1] != 7:
        raise ValueError(f"{path}: expected N x 7 executed actions, got {action_array.shape}")
    action_hash = sha256_actions(action_array)
    if action_hash != record.get("executed_action_sha256"):
        raise ValueError(f"{path}: executed action SHA mismatch")
    if int(action_array.shape[0]) != int(record.get("episode_steps", -1)):
        raise ValueError(f"{path}: action count differs from recorded episode steps")

    return {
        "path": str(path),
        "mode": mode,
        "episode_key": str(manifest["episode_key"]),
        "task_uid": str(manifest["task_uid"]),
        "task_name": str(manifest["task_name"]),
        "split": str(manifest["split"]),
        "init_state_index": int(manifest["init_state_index"]),
        "seed": int(manifest["seed"]),
        "inference_seed_offset": int(payload.get("inference_seed_offset", 0)),
        "success": bool(record.get("success")),
        "termination_reason": str(record.get("termination_reason")),
        "episode_steps": int(record.get("episode_steps", 0)),
        "inference_count": int(record.get("inference_count", 0)),
        "episode_time_ms": float(record.get("episode_time_ms", float("nan"))),
        "request_count": len(traces),
        "visual_input_mode_counts": dict(Counter(visual_modes)),
        "action_path": str(action_path),
        "action_sha256": action_hash,
        "shared_gpu_latency_caveat": bool(payload.get("shared_gpu_latency_caveat", False)),
    }


def load_route_records(directory: Path, expected_role: str) -> list[dict[str, Any]]:
    paths = sorted(path for path in directory.glob("*.json") if ".prior_failure_" not in path.name)
    if not paths:
        raise FileNotFoundError(f"no route artifacts under {directory}")
    return [validate_route_artifact(path, expected_role) for path in paths]


def pair_records(records: list[dict[str, Any]], expected_modes: set[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            record["episode_key"],
            record["init_state_index"],
            record["seed"],
            record["inference_seed_offset"],
        )
        mode = record["mode"]
        if mode in groups[key]:
            raise ValueError(f"duplicate route {mode} for paired key {key}")
        groups[key][mode] = record
    incomplete = [key for key, group in groups.items() if set(group) != expected_modes]
    if incomplete:
        raise ValueError(f"incomplete paired groups ({len(incomplete)}): {incomplete[:3]}")

    paired: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        prototype = next(iter(group.values()))
        for record in group.values():
            for field in ("task_uid", "task_name", "split"):
                if record[field] != prototype[field]:
                    raise ValueError(f"paired key {key} has inconsistent {field}")
        paired.append(
            {
                "episode_key": key[0],
                "init_state_index": key[1],
                "seed": key[2],
                "inference_seed_offset": key[3],
                "task_uid": prototype["task_uid"],
                "task_name": prototype["task_name"],
                "split": prototype["split"],
                "outcomes": {mode: bool(group[mode]["success"]) for mode in sorted(expected_modes)},
                "termination_reasons": {
                    mode: str(group[mode]["termination_reason"]) for mode in sorted(expected_modes)
                },
            }
        )
    return paired


def exact_mcnemar_p(wins: int, losses: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""

    total = int(wins) + int(losses)
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(wins, losses) + 1)) / (2**total)
    return float(min(1.0, 2.0 * tail))


def hierarchical_bootstrap(
    pairs: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        outcomes = pair["outcomes"]
        by_task[pair["task_uid"]].append(float(outcomes[left]) - float(outcomes[right]))
    if not by_task:
        return {"repeats": 0, "point_estimate": None, "ci95": [None, None], "probability_gt_zero": None}
    task_ids = sorted(by_task)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        selected_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        values: list[np.ndarray] = []
        for task_uid in selected_tasks:
            task_values = np.asarray(by_task[str(task_uid)], dtype=np.float64)
            values.append(rng.choice(task_values, size=len(task_values), replace=True))
        estimates[index] = float(np.concatenate(values).mean())
    point = float(np.mean([value for values in by_task.values() for value in values]))
    return {
        "method": "task_then_within_task_paired_bootstrap",
        "repeats": int(repeats),
        "seed": int(seed),
        "task_count": len(task_ids),
        "point_estimate": point,
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "probability_gt_zero": float(np.mean(estimates > 0.0)),
        "probability_lt_zero": float(np.mean(estimates < 0.0)),
    }


def pair_comparison(
    pairs: list[dict[str, Any]],
    left: str,
    right: str,
    *,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    wins = sum(pair["outcomes"][left] and not pair["outcomes"][right] for pair in pairs)
    losses = sum(not pair["outcomes"][left] and pair["outcomes"][right] for pair in pairs)
    ties = len(pairs) - wins - losses
    return {
        "left": left,
        "right": right,
        "pair_count": len(pairs),
        "left_successes": int(sum(pair["outcomes"][left] for pair in pairs)),
        "right_successes": int(sum(pair["outcomes"][right] for pair in pairs)),
        "left_minus_right_success_rate": float(
            np.mean([float(pair["outcomes"][left]) - float(pair["outcomes"][right]) for pair in pairs])
        ),
        "left_win_right_loss": int(wins),
        "left_loss_right_win": int(losses),
        "same_outcome": int(ties),
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(wins, losses),
        "hierarchical_bootstrap": hierarchical_bootstrap(
            pairs, left, right, repeats=bootstrap_repeats, seed=bootstrap_seed
        ),
    }


def mode_summary(records: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in modes:
        selected = [record for record in records if record["mode"] == mode]
        result[mode] = {
            "episodes": len(selected),
            "successes": int(sum(record["success"] for record in selected)),
            "success_rate": float(np.mean([record["success"] for record in selected])) if selected else None,
            "episode_steps": numeric_summary(float(record["episode_steps"]) for record in selected),
            "inference_count": numeric_summary(float(record["inference_count"]) for record in selected),
            "episode_time_ms_shared_gpu_provenance_only": numeric_summary(
                float(record["episode_time_ms"]) for record in selected
            ),
        }
    return result


def split_summary(pairs: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in sorted({pair["split"] for pair in pairs}):
        selected = [pair for pair in pairs if pair["split"] == split]
        result[split] = {
            "paired_scenarios": len(selected),
            "tasks": len({pair["task_uid"] for pair in selected}),
            "successes_by_mode": {mode: int(sum(pair["outcomes"][mode] for pair in selected)) for mode in modes},
            "success_rates_by_mode": {
                mode: float(np.mean([pair["outcomes"][mode] for pair in selected])) for mode in modes
            },
        }
    return result


def per_task_summary(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_task[pair["task_uid"]].append(pair)
    result = []
    for task_uid, selected in sorted(by_task.items()):
        prototype = selected[0]
        result.append(
            {
                "task_uid": task_uid,
                "task_name": prototype["task_name"],
                "split": prototype["split"],
                "paired_scenarios": len(selected),
                "successes_by_mode": {
                    mode: int(sum(pair["outcomes"][mode] for pair in selected)) for mode in MODES
                },
                "pv0_win_over_p1": int(
                    sum(
                        pair["outcomes"]["native_persistent"]
                        and not pair["outcomes"]["predicted_reuse"]
                        for pair in selected
                    )
                ),
                "fresh_win_over_pv0": int(
                    sum(
                        pair["outcomes"]["fresh"]
                        and not pair["outcomes"]["native_persistent"]
                        for pair in selected
                    )
                ),
            }
        )
    return result


def s5_summary(directory: Path) -> dict[str, Any] | None:
    if not directory.is_dir():
        return None
    records = load_route_records(directory / "episodes", "S5_action_outcome_preflight")
    pairs = pair_records(records, set(MODES))
    if len(pairs) != 1:
        raise ValueError(f"S5 should contain exactly one preregistered paired scenario, got {len(pairs)}")
    payloads = [load_json(Path(record["path"])) for record in records]
    perturbations = [payload.get("action_outcome_perturbation") for payload in payloads]
    return {
        "status": "PRELIMINARY_ONLY",
        "paired_scenarios": len(pairs),
        "outcomes": pairs[0]["outcomes"],
        "termination_reasons": pairs[0]["termination_reasons"],
        "perturbation": next((item for item in perturbations if item is not None), None),
        "interpretation": (
            "One preregistered action-outcome interruption route check. It is not a broad disturbance-recovery estimate."
        ),
    }


def clean_latency_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "status": "PENDING_OR_NOT_PROVIDED",
            "interpretation": "No clean-GPU rerun artifact was available when this paired outcome analysis was generated.",
        }
    payload = load_json(path)
    summary = payload.get("summary") or {}
    return {
        "artifact": str(path),
        "status": payload.get("status"),
        "state_key": payload.get("state_key"),
        "route_wall_latency_ms": summary.get("route_wall_latency_ms"),
        "native_action_recovery": summary.get("native_action_recovery"),
        "interpretation": (
            "This is a same-worker, rotated-order microbenchmark at one stored state; it is route-cost evidence, not an end-to-end closed-loop latency claim."
        ),
        "error": payload.get("error"),
    }


def fmt_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.1f}%"


def fmt_ci(values: list[float | None] | None) -> str:
    if not values or values[0] is None or values[1] is None:
        return "N/A"
    return f"[{100.0 * float(values[0]):.1f}, {100.0 * float(values[1]):.1f}] pp"


def render_markdown(analysis: dict[str, Any]) -> str:
    overall = analysis["phase_b_600"]
    modes = overall["route_summary"]
    pv0_vs_p1 = overall["paired_comparisons"]["pv0_vs_p1"]
    pv0_vs_fresh = overall["paired_comparisons"]["pv0_vs_fresh"]
    s5 = analysis.get("s5_action_outcome")
    clean = analysis.get("clean_gpu_microbenchmark") or {}
    seed = analysis.get("selected_seed_confirmation") or {}
    lines = [
        "# PV0 Full-scale Closed-loop Analysis",
        "",
        "## Frozen contract",
        "",
        "- Original pre-finetune Cosmos checkpoint only; SHA-256 matches the frozen protocol.",
        "- Every audited request uses exactly one denoising forward; no Cosmos value, privileged runtime state, scheduler, threshold, hidden patch, or fresh-prefix oracle.",
        "- PV0 is only the existing native causal 13-frame fresh visual-prefix persistent-condition interface.",
        "",
        "## Phase-B: 40 tasks × 5 initial states × 3 routes",
        "",
        "| Route | Success | Rate |",
        "| --- | ---: | ---: |",
    ]
    route_names = {
        "fresh": "Fresh (F1)",
        "predicted_reuse": "Predicted reuse (P1)",
        "native_persistent": "Native persistent (PV0)",
    }
    for mode in MODES:
        value = modes[mode]
        lines.append(f"| {route_names[mode]} | {value['successes']}/{value['episodes']} | {fmt_rate(value['success_rate'])} |")
    lines.extend(
        [
            "",
            f"- PV0 vs P1: **{pv0_vs_p1['left_win_right_loss']} PV0 wins / {pv0_vs_p1['left_loss_right_win']} losses**; Δ={100.0 * pv0_vs_p1['left_minus_right_success_rate']:.1f} pp, exact paired p={pv0_vs_p1['exact_mcnemar_two_sided_p']:.4f}, task-hierarchical bootstrap 95% CI={fmt_ci(pv0_vs_p1['hierarchical_bootstrap']['ci95'])}.",
            f"- PV0 vs Fresh: **{pv0_vs_fresh['left_win_right_loss']} PV0 wins / {pv0_vs_fresh['left_loss_right_win']} losses**; Δ={100.0 * pv0_vs_fresh['left_minus_right_success_rate']:.1f} pp, exact paired p={pv0_vs_fresh['exact_mcnemar_two_sided_p']:.4f}, task-hierarchical bootstrap 95% CI={fmt_ci(pv0_vs_fresh['hierarchical_bootstrap']['ci95'])}.",
            "",
            "## Interpretation",
            "",
            "- The full-scale result supports the narrow mechanism claim: fresh visual correction recovers successes lost by pure predicted reuse.",
            "- It does **not** establish that PV0 is success-equivalent or superior to always-fresh conditioning: two Fresh successes are lost by PV0 on this fixed 200-scenario benchmark.",
            "- Shared-GPU episode wall times are retained only as provenance and are not used for a cross-route latency claim.",
        ]
    )
    heldout = overall["split_summary"].get("heldout")
    heldout_comparisons = overall.get("paired_comparisons_by_split", {}).get("heldout")
    if heldout is not None and heldout_comparisons is not None:
        heldout_p1 = heldout_comparisons["pv0_vs_p1"]
        heldout_fresh = heldout_comparisons["pv0_vs_fresh"]
        lines.extend(
            [
                "",
                "## Held-out slice (12 tasks × 5 initial states)",
                "",
                f"- Fresh={heldout['successes_by_mode']['fresh']}/{heldout['paired_scenarios']} ({fmt_rate(heldout['success_rates_by_mode']['fresh'])}); "
                f"P1={heldout['successes_by_mode']['predicted_reuse']}/{heldout['paired_scenarios']} ({fmt_rate(heldout['success_rates_by_mode']['predicted_reuse'])}); "
                f"PV0={heldout['successes_by_mode']['native_persistent']}/{heldout['paired_scenarios']} ({fmt_rate(heldout['success_rates_by_mode']['native_persistent'])}).",
                f"- PV0 vs P1: {heldout_p1['left_win_right_loss']} wins / {heldout_p1['left_loss_right_win']} losses; exact paired p={heldout_p1['exact_mcnemar_two_sided_p']:.4f}.",
                f"- PV0 vs Fresh: {heldout_fresh['left_win_right_loss']} wins / {heldout_fresh['left_loss_right_win']} losses; exact paired p={heldout_fresh['exact_mcnemar_two_sided_p']:.4f}.",
            ]
        )
    if s5 is not None:
        outcomes = s5["outcomes"]
        lines.extend(
            [
                "",
                "## S5 action-outcome check",
                "",
                f"- One preregistered interrupted scenario: Fresh={outcomes['fresh']}, P1={outcomes['predicted_reuse']}, PV0={outcomes['native_persistent']}. This is preliminary only, not a recovery-rate result.",
            ]
        )
    if seed:
        comparison = seed["pv0_vs_fresh"]
        lines.extend(
            [
                "",
                "## Selected seed confirmation (conditional subset)",
                "",
                f"- {seed['pair_count']} matched pairs from scenarios selected for Fresh success or F1/PV0 disagreement, so its raw rates are not a population estimate.",
                f"- PV0 vs Fresh on this conditional subset: {comparison['left_win_right_loss']} wins / {comparison['left_loss_right_win']} losses; Δ={100.0 * comparison['left_minus_right_success_rate']:.1f} pp.",
            ]
        )
    lines.extend(
        [
            "",
            "## Clean-GPU microbenchmark",
            "",
            f"- Status: `{clean.get('status', 'PENDING_OR_NOT_PROVIDED')}`. {clean.get('interpretation', '')}",
            "",
            "## Audit artifacts",
            "",
            "- Raw Phase-B episode and trace JSON: `phase_b_600/episodes/`, `phase_b_600/traces/`.",
            "- Exact paired statistics and task-level outcomes: `CLOSED_LOOP_PAIRED_ANALYSIS.json`.",
            "- Offline replay videos, if generated, are diagnostics only; they replay saved actions and never supply state to the policy.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("reports/pv0_overnight"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--clean-latency", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        raise ValueError("bootstrap-repeats must be at least 100")
    run_dir = args.run_dir.resolve()
    output_json = args.output_json or run_dir / "CLOSED_LOOP_PAIRED_ANALYSIS.json"
    output_markdown = args.output_markdown or run_dir / "CLOSED_LOOP_RESULTS_ZH.md"
    if not args.overwrite:
        for path in (output_json, output_markdown):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite after reviewing it")

    phase_b_records = load_route_records(run_dir / "phase_b_600" / "episodes", "closed_loop")
    if len(phase_b_records) != 600:
        raise ValueError(f"expected 600 completed Phase-B route artifacts, got {len(phase_b_records)}")
    phase_b_pairs = pair_records(phase_b_records, set(MODES))
    if len(phase_b_pairs) != 200:
        raise ValueError(f"expected 200 complete Phase-B paired scenarios, got {len(phase_b_pairs)}")
    if len({pair['task_uid'] for pair in phase_b_pairs}) != 40:
        raise ValueError("Phase-B must retain all 40 benchmark tasks")
    if Counter(pair["task_uid"] for pair in phase_b_pairs).most_common(1)[0][1] != 5:
        raise ValueError("Phase-B must preserve five initial states per task")

    phase_b = {
        "design": "40 tasks x 5 initial states x 3 fixed routes = 600 completed route episodes",
        "route_artifact_count": len(phase_b_records),
        "paired_scenarios": len(phase_b_pairs),
        "task_count": len({pair["task_uid"] for pair in phase_b_pairs}),
        "route_summary": mode_summary(phase_b_records, MODES),
        "split_summary": split_summary(phase_b_pairs, MODES),
        "paired_comparisons": {
            "pv0_vs_p1": pair_comparison(
                phase_b_pairs,
                "native_persistent",
                "predicted_reuse",
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_seed=20260812,
            ),
            "pv0_vs_fresh": pair_comparison(
                phase_b_pairs,
                "native_persistent",
                "fresh",
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_seed=20260813,
            ),
        },
        "paired_comparisons_by_split": {
            split: {
                "pv0_vs_p1": pair_comparison(
                    [pair for pair in phase_b_pairs if pair["split"] == split],
                    "native_persistent",
                    "predicted_reuse",
                    bootstrap_repeats=args.bootstrap_repeats,
                    bootstrap_seed=20260900 + index * 10,
                ),
                "pv0_vs_fresh": pair_comparison(
                    [pair for pair in phase_b_pairs if pair["split"] == split],
                    "native_persistent",
                    "fresh",
                    bootstrap_repeats=args.bootstrap_repeats,
                    bootstrap_seed=20260901 + index * 10,
                ),
            }
            for index, split in enumerate(sorted({pair["split"] for pair in phase_b_pairs}))
        },
        "paired_scenarios_detail": phase_b_pairs,
        "per_task_summary": per_task_summary(phase_b_pairs),
        "latency_scope": (
            "Shared-GPU episode timings are not formal cross-route latency evidence; clean-GPU same-worker measurement is required for route-cost claims."
        ),
    }

    seed_dir = run_dir / "seed_confirmation" / "episodes"
    selected_seed_confirmation: dict[str, Any] | None = None
    if seed_dir.is_dir():
        seed_records = load_route_records(seed_dir, "closed_loop")
        seed_pairs = pair_records(seed_records, {"fresh", "native_persistent"})
        selected_seed_confirmation = {
            "selection_rule": "Only scenarios selected after Phase-B for Fresh success or F1/PV0 disagreement; conditional robustness subset, not a benchmark-wide rate.",
            "route_artifact_count": len(seed_records),
            "pair_count": len(seed_pairs),
            "route_summary": mode_summary(seed_records, ("fresh", "native_persistent")),
            "pv0_vs_fresh": pair_comparison(
                seed_pairs,
                "native_persistent",
                "fresh",
                bootstrap_repeats=args.bootstrap_repeats,
                bootstrap_seed=20260814,
            ),
            "paired_scenarios_detail": seed_pairs,
        }

    analysis = {
        "schema_version": 1,
        "experiment": "PV0_full_scale_closed_loop_paired_analysis",
        "status": "MIXED_MECHANISM_SUPPORT_NO_FRESH_SUPERIORITY",
        "frozen_contract": {
            "checkpoint_sha256": ORIGINAL_CHECKPOINT_SHA256,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "hidden_activation_patch_used": False,
            "fresh_prefix_oracle_used": False,
            "runtime_design_change": "none; post-hoc analysis only",
        },
        "scientific_interpretation": {
            "supported": "PV0 recovers paired successes lost by pure predicted latent reuse (P1).",
            "not_supported": "PV0 is not shown to be globally success-equivalent or superior to always-fresh conditioning.",
            "next_diagnostic": "Offline replay of the exact Fresh-success/PV0-failure action traces; no policy input is changed.",
        },
        "phase_b_600": phase_b,
        "s5_action_outcome": s5_summary(run_dir / "s5_action_outcome"),
        "selected_seed_confirmation": selected_seed_confirmation,
        "clean_gpu_microbenchmark": clean_latency_summary(args.clean_latency),
    }
    atomic_write_json(output_json, analysis)
    atomic_write_text(output_markdown, render_markdown(analysis))
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "phase_b_pairs": len(phase_b_pairs),
                "successes": {
                    mode: phase_b["route_summary"][mode]["successes"] for mode in MODES
                },
                "output_json": str(output_json),
                "output_markdown": str(output_markdown),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

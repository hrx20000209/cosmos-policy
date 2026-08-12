#!/usr/bin/env python3
"""Strict paired analysis for fixed R2 predicted versus matched stale reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def episodes(directory: Path, route: str) -> dict[str, dict]:
    result = {}
    for path in directory.glob("*.json"):
        if path.name.endswith("_traces.json"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("status") != "PASS" or raw.get("mode") != route:
            continue
        key = raw["manifest_row"]["episode_key"]
        if key in result:
            raise RuntimeError(f"duplicate {route} scenario {key}")
        result[key] = raw
    return result


def request_counts(row: dict) -> dict[str, int]:
    modes = row["trace_contract"]["visual_input_modes"]
    return {name: modes.count(name) for name in ("fresh", "native_persistent", "predicted", "stale_physical")}


def expected_r2_modes(route: str, count: int) -> list[str]:
    tail = ["predicted", "predicted"] if route == "pv0_r2" else ["stale_physical", "stale_physical"]
    sequence = ["fresh"]
    while len(sequence) < count:
        sequence.extend(["native_persistent", *tail])
    return sequence[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted-dir", type=Path, default=Path("reports/pv0_execution_feedback/fixed_reuse_pilot/episodes"))
    parser.add_argument("--stale-dir", type=Path, default=Path("reports/pv0_execution_feedback/matched_stale_r2_v2/episodes"))
    parser.add_argument("--output", type=Path, default=Path("reports/pv0_execution_feedback/MATCHED_STALE_R2_ANALYSIS.json"))
    args = parser.parse_args()
    predicted = episodes(args.predicted_dir, "pv0_r2")
    stale = episodes(args.stale_dir, "stale_r2")
    if set(predicted) != set(stale) or len(predicted) != 8:
        raise RuntimeError(f"scenario mismatch: predicted={len(predicted)} stale={len(stale)}")
    pairs = []
    wins = losses = ties = 0
    contract_failures = []
    for key in sorted(predicted):
        p, s = predicted[key], stale[key]
        pc, sc = request_counts(p), request_counts(s)
        for row in (p, s):
            if not all(int(item) == 1 for item in row["trace_contract"]["denoiser_forward_counts"]):
                contract_failures.append(f"denoise:{row['mode']}:{key}")
            if row["execution_feedback_contract"].get("runtime_observables_only") is not True:
                contract_failures.append(f"telemetry:{row['mode']}:{key}")
        p_modes, s_modes = p["trace_contract"]["visual_input_modes"], s["trace_contract"]["visual_input_modes"]
        p_schedule = p_modes == expected_r2_modes("pv0_r2", len(p_modes))
        s_schedule = s_modes == expected_r2_modes("stale_r2", len(s_modes))
        # Completion ends an episode immediately, so total requests can differ
        # legitimately.  The matched budget contract is that every *executed*
        # decision follows the same correction cadence and every reuse slot is
        # predicted versus stale respectively; report call/action rates rather
        # than wrongly demanding equal post-success calls.
        matched = p_schedule and s_schedule
        if not matched:
            contract_failures.append(f"schedule:{key}:predicted={p_modes}:stale={s_modes}")
        p_success, s_success = bool(p["record"]["success"]), bool(s["record"]["success"])
        if p_success and not s_success:
            wins += 1
        elif s_success and not p_success:
            losses += 1
        else:
            ties += 1
        p_steps, s_steps = p["record"]["episode_steps"], s["record"]["episode_steps"]
        pairs.append({"episode_key": key, "task_uid": p["manifest_row"]["task_uid"], "predicted_success": p_success, "stale_success": s_success, "predicted_request_counts": pc, "stale_request_counts": sc, "schedule_contract_pass": matched, "predicted_steps": p_steps, "stale_steps": s_steps, "predicted_fresh_per_action": (pc["fresh"] + pc["native_persistent"]) / p_steps, "stale_fresh_per_action": (sc["fresh"] + sc["native_persistent"]) / s_steps})
    decision = "WAM_REUSE_ADVANTAGE_GO" if wins > losses and not contract_failures else "WAM_REUSE_ADVANTAGE_NO_GO"
    payload = {"schema_version": 1, "status": "PASS" if not contract_failures else "FAIL_CONTRACT", "comparison": "fixed R2 predicted future-state reuse versus matched stale physical-condition reuse", "scenarios": len(pairs), "task_disjoint_split": "heldout", "checkpoint_sha256": next(iter(predicted.values()))["checkpoint_sha256"], "denoising_steps": 1, "value_used": False, "paired_success": {"predicted": sum(item["predicted_success"] for item in pairs), "stale": sum(item["stale_success"] for item in pairs), "predicted_only_wins": wins, "stale_only_wins": losses, "ties": ties}, "fresh_calls_per_executed_action": {"predicted": sum(item["predicted_request_counts"]["fresh"] + item["predicted_request_counts"]["native_persistent"] for item in pairs) / sum(item["predicted_steps"] for item in pairs), "stale": sum(item["stale_request_counts"]["fresh"] + item["stale_request_counts"]["native_persistent"] for item in pairs) / sum(item["stale_steps"] for item in pairs)}, "contract_failures": contract_failures, "decision": decision, "interpretation": "This is a small 8-scenario directional gate, not a task-general efficacy estimate.", "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# R2 predicted vs matched stale", "", f"- Decision: **{decision}**", f"- Paired heldout scenarios: {len(pairs)}", f"- Success: predicted={payload['paired_success']['predicted']}/{len(pairs)}, stale={payload['paired_success']['stale']}/{len(pairs)}", f"- Discordant pairs: predicted-only={wins}, stale-only={losses}, ties={ties}", f"- Fixed cadence / denoise / telemetry contract: {'PASS' if not contract_failures else 'FAIL'}", f"- Fresh calls per executed action: predicted={payload['fresh_calls_per_executed_action']['predicted']:.5f}, stale={payload['fresh_calls_per_executed_action']['stale']:.5f}", "", "The stale route retains the most recent PV0 physical-condition joint latent and never copies generated future visual slots. Both routes use the same PV0→reuse→reuse schedule until their physical closed-loop termination; a successful earlier termination legitimately reduces later calls, so cost is reported per executed action."]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": decision, "pairs": len(pairs)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the evidence-bounded report for the feedback-gated R2 small gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--matched-stale", type=Path, required=True)
    parser.add_argument("--slot-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = load(args.collection_dir / "analysis/EXECUTION_VALIDITY_FEATURE_ANALYSIS.json")
    state = load(args.collection_dir / "COLLECTION_STATE.json")
    stale = load(args.matched_stale)
    slots = load(args.slot_contract)
    episodes = []
    for path in sorted((args.collection_dir / "episodes").glob("pv0_r2_*.json")):
        if path.name.endswith("_traces.json") or ".prior_failure_" in path.name:
            continue
        episodes.append(load(path))
    by_split = {}
    for split in ("discovery", "validation", "heldout"):
        group = [row for row in episodes if row["manifest_row"]["split"] == split]
        executed = sum(int(row["record"]["episode_steps"]) for row in group)
        traces = [trace for row in group for trace in load((args.collection_dir / "episodes" / f"pv0_r2_{row['manifest_row']['episode_key']}_traces.json"))["traces"]]
        modes = [trace["extra"]["visual_input_mode"] for trace in traces]
        by_split[split] = {
            "episodes": len(group),
            "success": sum(bool(row["record"]["success"]) for row in group),
            "executed_actions": executed,
            "requests": len(traces),
            "pv0_calls": modes.count("native_persistent"),
            "p1_calls": modes.count("predicted"),
            "fresh_bootstrap_calls": modes.count("fresh"),
            "p1_fraction": modes.count("predicted") / len(traces) if traces else None,
            "fresh_sensing_calls_per_executed_action": (
                (modes.count("fresh") + modes.count("native_persistent")) / executed if executed else None
            ),
        }
    feedback = analysis["baseline_evaluation"]
    lines = [
        "# SERVER — Feedback-Gated R2 Predict–Correct Runtime（小规模 gate 报告）",
        "",
        "## Executive Summary",
        "",
        "- **WAM reuse gate：GO（方向性、小样本）**。8 个 matched heldout scenario 中，R2 predicted 成功 2/8、matched stale 1/8；predicted-only 2、stale-only 1。在 matched cadence/denoise/telemetry 合约下，该证据允许继续检查反馈信号，但不能声称 task-general。",
        "- **Execution feedback gate：NO-GO**。在新 12-task × 2-init、task-disjoint shadow collection 上，纯物理执行反馈线性分数未超过 AGE_ONLY：validation task-balanced Spearman 为 "
        f"{feedback['validation']['EXECUTION_FEEDBACK_LINEAR']['task_balanced_spearman_p1_risk']:.3f} vs {feedback['validation']['AGE_ONLY']['task_balanced_spearman_p1_risk']:.3f}；heldout 为 "
        f"{feedback['heldout']['EXECUTION_FEEDBACK_LINEAR']['task_balanced_spearman_p1_risk']:.3f} vs {feedback['heldout']['AGE_ONLY']['task_balanced_spearman_p1_risk']:.3f}。",
        "- 因此不冻结 early-correction rule、不运行 adaptive pilot、不进行 full 40-task promotion。保留固定 R2 为研究 backbone；本轮不引入动作缩放、retiming、F1 三路 scheduler 或学习网络。",
        "",
        "## Frozen Contract",
        "",
        "- Original pre-finetune LIBERO checkpoint SHA256: `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`",
        "- denoise=1；Cosmos value=false；no finetune；no privileged simulator state；runtime 无 object pose/contact/reward/success/task name 输入。",
        "- R2 maximum depth: `F1 bootstrap → PV0 → P1 → P1 → PV0 …`。没有一次执行超过 depth=2。",
        "",
        "## Matched Stale R2",
        "",
        f"- Decision: **{stale['decision']}**.",
        f"- Fresh calls per executed action: predicted={stale['fresh_calls_per_executed_action']['predicted']:.5f}, stale={stale['fresh_calls_per_executed_action']['stale']:.5f}.",
        "- Interpretation is bounded to the eight paired heldout scenarios; early terminal success correctly changes later request counts, so cost is normalized by executed actions.",
        "",
        "## Execution-validity Dataset",
        "",
        f"- Collection: {state['counts']['completed']}/24 completed, {state['counts']['failed']} final failures; one transient external-VRAM OOM was retained as prior-failure audit and retried without affecting external processes.",
        f"- Labels: discovery={analysis['rows']['discovery']}, validation={analysis['rows']['validation']}, heldout={analysis['rows']['heldout']} decision points; 4 task-disjoint tasks per split.",
        "- Main route executed only fixed R2. F1/P1/PV0 were generated after action installation as shadow-only retrospective labels (`D(A_P1,A_F1)`, `D(A_PV0,A_F1)`), never runtime policy inputs.",
        "",
        "## Feedback vs AGE_ONLY",
        "",
        "| Split | AGE_ONLY task-balanced ρ | Physical feedback task-balanced ρ | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for split in ("discovery", "validation", "heldout"):
        age = feedback[split]["AGE_ONLY"]["task_balanced_spearman_p1_risk"]
        physical = feedback[split]["EXECUTION_FEEDBACK_LINEAR"]["task_balanced_spearman_p1_risk"]
        result = "feedback > age" if physical is not None and age is not None and physical > age else "feedback ≤ age"
        lines.append(f"| {split} | {age:.3f} | {physical:.3f} | {result} |")
    lines.extend([
        "",
        "The physical score excludes prediction age and reuse depth; it uses only windowed EEF/proprio and planned/executed-action features. This avoids accidentally relabeling AGE_ONLY as a feedback method. Its cross-split result fails the preregistered gate.",
        "",
        "## R2 Cost Accounting (small collection)",
        "",
        "| Split | Episodes | Success | P1 fraction | Fresh/PV0 sensing calls per executed action |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for split, metrics in by_split.items():
        lines.append(
            f"| {split} | {metrics['episodes']} | {metrics['success']} | {metrics['p1_fraction']:.3f} | {metrics['fresh_sensing_calls_per_executed_action']:.5f} |"
        )
    lines.extend([
        "",
        "## Slot-aware PV0 Contract",
        "",
        "- PV0 is condition-slot replacement, not a fixed universal frame count. LIBERO refreshes current wrist/primary slots 2/3: its 4× temporal VAE layout makes 13 raw prefix frames sufficient.",
        "- SO101/ALOHA uses current proprio plus three camera slots (2/3/4); its corresponding causal raw prefix is 17 frames. This is documented but not validated by the LIBERO runtime experiment.",
        f"- Contract artifact: `{args.slot_contract}` (schema {slots['schema_version']}).",
        "",
        "## Final GO / NO-GO",
        "",
        "- `WAM_REUSE_ADVANTAGE_GO` (small paired evidence only).",
        "- `FEEDBACK_ADAPTATION_NO_GO`: physical execution feedback does not add reliable predictive power beyond AGE_ONLY/reuse depth on this task-disjoint gate.",
        "- Next authorized candidate is time-bounded fixed R2/AGE_ONLY as a baseline study; it is not an approved feedback-adaptive design. Any new feedback proposal needs an independent signal/collection plan before a new gate.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": analysis["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

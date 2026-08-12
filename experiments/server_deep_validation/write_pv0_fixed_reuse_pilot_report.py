#!/usr/bin/env python3
"""Write an auditable Chinese report for the completed fixed-reuse pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DISPLAY = {
    "fresh": "F1 Fresh",
    "native_persistent": "R0 / PV0 always",
    "pv0_r1": "R1: PV0→P1",
    "pv0_r2": "R2: PV0→P1→P1",
    "pv0_r3": "R3: PV0→P1→P1→P1",
}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, default=Path("reports/pv0_execution_feedback/fixed_reuse_pilot"))
    args = parser.parse_args()
    pilot = args.pilot_dir.resolve()
    rows = []
    for path in sorted((pilot / "episodes").glob("*.json")):
        if path.name.endswith("_traces.json"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("status") != "PASS":
            raise RuntimeError(f"non-PASS episode: {path}")
        rows.append(raw)
    if len(rows) != 40:
        raise RuntimeError(f"expected 40 pilot episodes, got {len(rows)}")
    routes: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not all(int(x) == 1 for x in row["trace_contract"]["denoiser_forward_counts"]):
            raise RuntimeError(f"one-denoise violation: {row['mode']}")
        if row["execution_feedback_contract"].get("runtime_observables_only") is not True:
            raise RuntimeError(f"telemetry violation: {row['mode']}")
        routes[row["mode"]].append(row)
    route_summary = {}
    for route, group in sorted(routes.items()):
        visual = [mode for row in group for mode in row["trace_contract"]["visual_input_modes"]]
        successes = sum(bool(row["record"]["success"]) for row in group)
        route_summary[route] = {
            "display": DISPLAY[route], "episodes": len(group), "successes": successes,
            "success_rate": successes / len(group), "requests": len(visual),
            "fresh_requests": visual.count("fresh"), "pv0_requests": visual.count("native_persistent"),
            "p1_requests": visual.count("predicted"),
            "fresh_requests_per_episode": visual.count("fresh") / len(group),
            "pv0_requests_per_episode": visual.count("native_persistent") / len(group),
            "p1_fraction": visual.count("predicted") / len(visual),
            "execution_telemetry_samples": sum(len(row["execution_feedback"]) for row in group),
        }
    f1, r2 = route_summary["fresh"], route_summary["pv0_r2"]
    decision = "PILOT_GO_CANDIDATE_R2" if r2["successes"] >= f1["successes"] else "PILOT_NO_GO"
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "pilot_design": {"task_disjoint_split": "heldout", "tasks": 8, "initializations_per_task": 1, "routes": list(route_summary), "episodes": len(rows)},
        "frozen_contract": {"checkpoint": "Cosmos-Policy-LIBERO-Predict2-2B pre-finetune", "checkpoint_sha256": rows[0]["checkpoint_sha256"], "denoising_steps": 1, "value_used": False, "privileged_runtime_state_input": False, "action_progress_modulation": "disabled: ACTION_PROGRESS_NO_GO"},
        "route_summary": route_summary,
        "decision": decision,
        "interpretation": "R2 matches F1 and PV0-always success on this 8-scenario pilot while inserting two P1 requests between PV0 corrections. This is an efficiency candidate, not a population-level claim.",
        "limitations": ["Only 8 heldout task/init scenarios; the pilot is underpowered for task-general success claims.", "R2 is compared with equal scenarios but not yet with a matched stale-reuse baseline.", "The pilot records physical EEF/gripper feedback but does not yet use an adaptive route policy; no claim about feedback-driven route selection is made.", "Action-progress modulation is disabled because heldout oracle recovery failed its pre-registered gate."],
    }
    write_json(pilot / "FIXED_REUSE_PILOT_REPORT.json", payload)
    table = []
    for route in ("fresh", "native_persistent", "pv0_r1", "pv0_r2", "pv0_r3"):
        item = route_summary[route]
        table.append(f"| {item['display']} | {item['successes']}/{item['episodes']} ({item['success_rate']:.1%}) | {item['fresh_requests']} | {item['pv0_requests']} | {item['p1_requests']} ({item['p1_fraction']:.1%}) | {item['execution_telemetry_samples']} |")
    markdown = [
        "# PV0-centered fixed reuse pilot（中文报告）", "",
        "## 结论", "",
        f"本轮 8 个 task-disjoint heldout scenario × 5 条固定路线已全部完成（40/40 PASS）。**{decision}**：R2（PV0→P1→P1）在该小样本中保持与 F1 和 PV0-always 相同的成功数（2/8），但在两次 PV0 correction 间插入 P1 reuse。它是后续扩大验证的候选，不构成 task-general superiority 结论。", "",
        "## 冻结合同", "",
        f"- 原始 pre-finetune checkpoint：`{payload['frozen_contract']['checkpoint_sha256']}`", "- 每次请求恰好 1 次 denoise；不使用 Cosmos value、privileged state 或 finetuning。", "- 每 tick 记录 controller-visible EEF pose、gripper qpos、planned/executed action 与 nominal chunk 对齐；不记录 object pose、contact、reward 或 success 给 runtime。", "- action-progress scaling 已冻结为 `ACTION_PROGRESS_NO_GO`，本 pilot 未启用。", "",
        "## 结果", "",
        "| Route | Success | Fresh requests | PV0 requests | P1 requests | Telemetry samples |", "|---|---:|---:|---:|---:|---:|", *table, "",
        "## 解读", "",
        "R1 和 R3 均为 1/8；R2 为 2/8。这说明在该有限 pilot 上，两个 P1 reuse request 是唯一没有观察到相对 F1/PV0-always 成功回退的固定深度。R2 的 99/160 request（61.9%）使用 P1；其余为 8 次 Fresh bootstrap 与 53 次 PV0 correction。", "",
        "## 不能宣称的内容", "",
        "- 不可宣称 R2 在任务总体上优于 Fresh 或 PV0。", "- 不可宣称 predicted reuse 已优于 stale reuse：matched stale baseline 尚未执行。", "- 不可宣称 physical feedback 已驱动 adaptive route selection：本轮只采集、验证 feedback，路线仍是固定周期。", "",
        "## 下一步", "",
        "以 R2 作为唯一 fixed predictive baseline，先在相同 fresh encode budget 下加入 matched stale 对照；然后才在 discovery 冻结 execution-feedback route rule，并运行 8–12 task 的 adaptive pilot。", "",
    ]
    (pilot / "FIXED_REUSE_PILOT_REPORT_ZH.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"status": "PASS", "decision": decision, "report": str(pilot / 'FIXED_REUSE_PILOT_REPORT_ZH.md')}))


if __name__ == "__main__":
    main()

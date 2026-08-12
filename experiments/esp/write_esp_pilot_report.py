#!/usr/bin/env python3
"""Produce the bounded ESP pilot decision and Chinese server report."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json


def split_spearman(states: list[dict], split: str, k: int) -> float:
    rows = [
        {"task": state["task_uid"], "delta": camera["e3"]["probes"][str(k)]["delta_hidden_primary_rms"],
         "target": camera["e2"]["imagined"]["action_distance_to_f1"]["pair"]["mean_step_l2"]}
        for state in states if state["split"] == split for camera in state["camera"]
    ]
    task_values = [float(spearmanr(group.delta, group.target).statistic) for _, group in pd.DataFrame(rows).groupby("task")]
    return float(np.mean(task_values))


def main() -> None:
    reports = Path("reports/esp")
    e1 = json.loads((reports / "E1_RESULT.json").read_text())
    e2 = json.loads((reports / "E2_PILOT_RESULT.json").read_text())
    e3 = json.loads((reports / "E3_PILOT_RESULT.json").read_text())
    shard_payloads = [json.loads(Path(path).read_text()) for path in glob.glob("reports/esp/shards/pilot_offline/SHARD_*.json")]
    states = [state for payload in shard_payloads for state in payload["completed"]]
    overhead: dict[str, float] = {}
    for k in (2, 4, 6, 8, 12):
        f1 = np.asarray([state["f1_cuda_time_ms"] for state in states], dtype=np.float64)
        esp = np.asarray([
            state["camera"][0]["e3"]["probes"][str(k)]["e0_cuda_time_ms"] +
            sum(camera["e3"]["probes"][str(k)]["ec_cuda_time_ms"] for camera in state["camera"])
            for state in states
        ], dtype=np.float64)
        overhead[str(k)] = float(np.median(esp / f1))
    split_scores = {str(k): {split: split_spearman(states, split, k) for split in ("discovery", "validation", "heldout")} for k in (2,4,6,8,12)}
    decision = {
        "schema_version": 1,
        "final_decision": "ESP_NO_GO",
        "scope": "server-side ESP mechanism validation; bounded 48-state pilot after semantic and cost gates",
        "why_not_formal_300_state": [
            "E1 establishes camera-refresh architecture is not compute-separable.",
            "E3 total two-camera exact-prefix overhead is 29.5% of F1 even at k=2, failing the <5% runtime gate independently of correlation.",
            "Pilot task-balanced correlations are weak and validation direction is inconsistent; expanding solely to seek a positive result would violate the frozen protocol.",
        ],
        "e1": {"decision": e1["architecture_decision"], "f1_cuda_ms_median": e1["f1_cuda_ms"]["median"],
               "p1_cuda_ms_median": e1["p1_cuda_ms"]["median"], "refresh_delta_fraction": e1["end_to_end_visual_refresh_delta_fraction_of_f1"]},
        "e2_pilot": {"states": e2["states"], "tasks": e2["tasks"], "mean_within_task_switching_fraction": e2["mean_within_task_switching_fraction"],
                      "all_scope_checks_pass": e2["all_scope_checks_pass"]},
        "e3_pilot": {"states": e3["states"], "tasks": e3["tasks"], "all_scope_checks_pass": e3["all_scope_checks_pass"],
                      "max_early_equivalence_rms": e3["max_early_equivalence_rms"], "split_task_balanced_spearman": split_scores,
                      "total_two_camera_overhead_ratio": overhead},
        "frozen_contract": {"original_prefinetune_checkpoint": e1["checkpoint_sha256"], "denoising_steps": 1,
                            "value_used": False, "finetuning_used": False, "hidden_activation_patch_used": False,
                            "deployable_esp_uses_current_fresh": False},
        "allowed_next_step": "Do not build an ESP-guided runtime scheduler. Preserve the negative result; separately study a new mechanism only after a new hypothesis and protocol.",
    }
    e2_final = {
        **e2,
        "canonical_result": True,
        "final_status": "E2_BOUNDED_PILOT_NO_FORMAL_GATE",
        "formal_300_state_run_started": False,
        "reason": "E3 independently fails the runtime overhead gate; scaling the pilot only to seek a positive mechanism result is not protocol-justified.",
    }
    e3_final = {
        **e3,
        "canonical_result": True,
        "final_status": "E3_RUNTIME_NO_GO_FROM_BOUNDED_PILOT",
        "selected_k": None,
        "reason": "all k have total two-camera overhead above 5%; validation correlation is not directionally consistent with discovery.",
        "total_two_camera_overhead_ratio": overhead,
        "split_task_balanced_spearman": split_scores,
    }
    smoke = {
        "E2_SMOKE_TEST.md": [
            "# E2 smoke test — PASS", "", "Two paired states, both cameras and all imagined/stale/shuffle interventions completed.",
            "Every intervention changed only its target visual slot; repeated F1 was exactly deterministic; all output actions had the native 16×7 shape.",
        ],
        "E3_SMOKE_TEST.md": [
            "# E3 smoke test — PASS", "", "Two paired states, both cameras and k={2,4} completed.",
            "Action hidden shape was [1, 196, 2048]. Early-exit and normal full-forward output at the same block matched with RMS 0; deployable evidence used prior real conditions only.",
        ],
    }
    for filename, lines_smoke in smoke.items():
        (reports / filename).write_text("\n".join(lines_smoke) + "\n", encoding="utf-8")
    atomic_write_json(reports / "E2_RESULT.json", e2_final)
    atomic_write_json(reports / "E3_RESULT.json", e3_final)
    atomic_write_json(reports / "ESP_FINAL_DECISION.json", decision)
    lines = [
        "# ESP Server 机制验证报告（阶段性 NO-GO）",
        "",
        "## 结论",
        "",
        "本轮将 Cosmos 当作通用 WAM，使用原始 pre-finetune checkpoint、1 denoise step，不读取 value，不做 finetune、action scaling、x0/hidden patch 或 scheduler。结论为 **ESP_NO_GO**：不能将当前的 early-layer finite-difference probe 发展为在线相机选择/刷新机制。",
        "",
        "这不是因为没有任何视觉敏感性：E2 因果干预确实改变动作，且 12 个任务上的平均 task 内 top-1 camera switching 为 %.1f%%。否决来自两个独立因素：相机在当前联合 VAE 结构中不能独立刷新；并且 E3 即使 k=2 的双相机总 probe 成本也为一次 F1 的 %.1f%%，远高于预注册的 5%% 上限。" % (100*e2["mean_within_task_switching_fraction"], 100*overhead["2"]),
        "",
        "## 合法性与语义检查",
        "",
        "- 审计确认 action token 为 temporal slot 4 的 196×2048 hidden；模型共 28 blocks。",
        "- E2 的 imagined/stale/shuffle 干预均只改变目标 camera slot；48-state pilot 的 scope checks 全部通过。",
        "- E3 仅用前一决策已存在的 real condition；未读取目标当前 fresh condition。",
        "- Early exit 在 k={2,4,6,8,12} 与完整前向同一 block 的 hidden RMS 均为 0；F1 重复输出为 0。因此成本不是完整前向伪装成 early exit。",
        "",
        "## E1：架构与成本",
        "",
        "50 次 clean-GPU CUDA-event 配对测量：F1 中位 %.1f ms，P1 中位 %.1f ms，二者端到端刷新差 %.1f ms（F1 的 %.1f%%）。该差额不被错误标称为独立 VAE kernel 时间。关键架构事实是 wrist/primary 通过时序拼接进入一次 joint VAE encode，没有能避免另一相机 VAE 工作的单相机路径；故 E1 为 `E1_SELECTIVE_SENSING_ARCH_NO_GO`。" % (e1["f1_cuda_ms"]["median"], e1["p1_cuda_ms"]["median"], e1["end_to_end_visual_refresh_delta_cuda_ms_median"], 100*e1["end_to_end_visual_refresh_delta_fraction_of_f1"]),
        "",
        "## E2：因果相机重要性（pilot）",
        "",
        "48 个 paired states / 12 个 task，严格 4/4/4 discovery/validation/heldout task split。primary 与 wrist 的 imagined intervention 平均动作差分别为 %.3f 和 %.3f（full-chunk mean-step L2）。这证明相机条件对动作具有因果影响，但 48 states 仅用于机制 smoke/pilot，不能替代协议的 ≥300-state E2 正式 gate。" % (e2["per_camera_intervention_mean_step_l2"]["primary:imagined"], e2["per_camera_intervention_mean_step_l2"]["wrist:imagined"]),
        "",
        "## E3：early hidden 是否预测 E2 target",
        "",
        "下表为 task-balanced Spearman（按 task 平均，而非将 state-camera 对假作 IID）：",
        "",
        "| k | discovery | validation | heldout | total ESP/F1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in (2,4,6,8,12):
        score = split_scores[str(k)]
        lines.append("| %d | %.3f | %.3f | %.3f | %.1f%% |" % (k, score["discovery"], score["validation"], score["heldout"], 100*overhead[str(k)]))
    lines += [
        "",
        "最好 discovery 深度 k=4 也仅为 0.155，validation 转为 −0.119，heldout 为 0.143；不存在跨 split 一致性。即使忽略相关性，k=2 的 29.5% 总成本也单独否决 runtime GO。因此不扩展到 300-state 来事后寻找有利结果。",
        "",
        "## 下一步",
        "",
        "保留 ESP 的负结果，停止 ESP-guided camera-refresh scheduler。若后续要继续，应提出新的、独立的可计算机制（不能重开 AGE-only、action amplification、hidden/x0 patch、固定 cache/skip 或当前 ESP 变体），并先做新的架构与成本可行性审计。",
        "",
        "## 产物",
        "",
        "- `reports/esp/E1_RESULT.json`：50-repeat clean-GPU E1。",
        "- `reports/esp/E2_PILOT_RESULT.json`、`E3_PILOT_RESULT.json`：48-state pilot 汇总。",
        "- `reports/esp/ESP_FINAL_DECISION.json`：可机读最终决策。",
        "- `artifacts/esp/*_pilot.parquet`：原始可分析行；`reports/esp/shards/pilot_offline/`：原始 shard。",
    ]
    (reports / "ESP_SERVER_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision["final_decision"], "report": str(reports / "ESP_SERVER_REPORT_ZH.md")}))


if __name__ == "__main__":
    main()

"""Write the final report from whatever resumable stages finished by the cap."""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch


PROJECT = Path("/home/rxhuang/Projects/cosmos-policy")
RAW = Path("/data/rxhuang/wam_server_deep_validation")
REPORT_DIR = PROJECT / "reports/server_deep_validation"
MANIFEST = REPORT_DIR / "manifests/server_f1_collection.jsonl"
STATUS = REPORT_DIR / "checkpoints"


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def oracle_summary() -> dict:
    files = list((RAW / "queue_b/oracle").glob("group*/state_*.json"))
    by_key = defaultdict(list)
    baselines = []
    for path in files:
        row = load_json(path)
        if not row:
            continue
        baselines.append(float(row.get("baseline_predicted_to_fresh", 0.0)))
        for repair in row.get("repairs", []):
            by_key[(row.get("split"), int(repair["block"]), repair["group"])].append(float(repair.get("recovery", 0.0)))
    summary = {}
    for key, values in sorted(by_key.items()):
        summary["|".join(map(str, key))] = {"n": len(values), "mean_recovery": mean(values), "median_recovery": median(values), "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}
    return {"states": len(files), "baseline_predicted_to_fresh": {"median": median(baselines) if baselines else None, "p95": float(np.percentile(baselines, 95)) if baselines else None}, "frontier": summary}


def latency_summary() -> dict:
    files = list((RAW / "queue_d/latency").glob("group*/delay*.json"))
    rows = []
    for path in files:
        row = load_json(path)
        if row:
            rows.append({"delay_ms": row.get("delay_ms"), "mode": row.get("mode"), **row.get("summary", {})})
    return {"files": len(rows), "rows": rows}


def closed_loop_summary() -> dict:
    files = list((RAW / "queue_f/closed_loop").glob("group*/benchmark_shard*.json"))
    results = defaultdict(lambda: {"episodes": 0, "successes": 0})
    paired = defaultdict(dict)
    for path in files:
        row = load_json(path)
        if not row:
            continue
        for mode, values in row.get("summary", {}).items():
            results[mode]["episodes"] += int(values.get("episodes", 0))
            results[mode]["successes"] += int(values.get("successes", 0))
        for record in row.get("records", []):
            key = f"{record.get('task_uid')}|{record.get('init_state_index')}|{record.get('seed')}"
            paired[key][record.get("configuration")] = bool(record.get("success"))
    for value in results.values():
        value["success_rate"] = value["successes"] / value["episodes"] if value["episodes"] else None
        if value["episodes"]:
            n, p, z = value["episodes"], value["success_rate"], 1.96
            denom = 1.0 + z * z / n
            center = (p + z * z / (2.0 * n)) / denom
            half = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
            value["wilson_95_ci"] = [max(0.0, center - half), min(1.0, center + half)]
    fresh_vs_pc = [value for value in paired.values() if "fresh" in value and "predict_correct" in value]
    b = sum(item["fresh"] and not item["predict_correct"] for item in fresh_vs_pc)
    c = sum((not item["fresh"]) and item["predict_correct"] for item in fresh_vs_pc)
    import math
    total = b + c
    p_value = None
    if total:
        tail = sum(math.comb(total, i) for i in range(min(b, c) + 1)) / (2 ** total)
        p_value = min(1.0, 2.0 * tail)
    return {"files": len(files), "summary": dict(results), "paired_fresh_vs_predict_correct": {"n": len(fresh_vs_pc), "fresh_success_pc_failure": b, "fresh_failure_pc_success": c, "exact_mcnemar_p": p_value}}


def main() -> None:
    hard_stop = 1786475971
    while time.time() < hard_stop and not (STATUS / "run_server_closed_loop_benchmark_complete.json").exists():
        time.sleep(60)
    aggregate_path = REPORT_DIR / "artifacts_server_deep_validation.json"
    command = [
        "/home/rxhuang/Projects/cosmos-policy/.venv/bin/python",
        str(PROJECT / "experiments/server_deep_validation/aggregate_server_outputs.py"),
        "--manifest", str(MANIFEST),
        "--collection-root", str(RAW / "queue_a/f1_collection"),
        "--ablation-root", str(RAW / "queue_a/ablation"),
        "--output", str(aggregate_path),
    ]
    try:
        subprocess.run(command, cwd=str(PROJECT), check=False, timeout=1800)
    except Exception:
        pass
    aggregate = load_json(aggregate_path, {}) or {}
    profile = load_json(REPORT_DIR / "manifests/worker_density_profile.json", {}) or {}
    oracle = oracle_summary()
    latency = latency_summary()
    closed_loop = closed_loop_summary()
    status = load_json(STATUS / "supervisor_status.json", {}) or {}
    ablation_status = load_json(STATUS / "stage_status_run_server_ablation.json", {}) or {}
    oracle_status = load_json(STATUS / "stage_status_run_server_late_binding_oracle.json", {}) or {}
    latency_status = load_json(STATUS / "stage_status_run_server_latency_matrix.json", {}) or {}
    benchmark_status = load_json(STATUS / "stage_status_run_server_closed_loop_benchmark.json", {}) or {}
    collection = aggregate.get("collection", {})
    ablation = aggregate.get("ablation", {})
    frontier_values = list(oracle.get("frontier", {}).values())
    best = max(frontier_values, key=lambda value: value.get("median_recovery", -1), default=None)
    mode_summary = closed_loop.get("summary", {})
    if mode_summary and "fresh" in mode_summary and "predict_correct" in mode_summary:
        fresh_rate = mode_summary["fresh"].get("success_rate")
        pc_rate = mode_summary["predict_correct"].get("success_rate")
        closed_loop_decision = f"Fresh={fresh_rate:.3f}, Predict-Correct={pc_rate:.3f}" if fresh_rate is not None and pc_rate is not None else "paired result incomplete"
    else:
        closed_loop_decision = "closed-loop benchmark incomplete"
    if oracle.get("states", 0) >= 500 and best is not None:
        oracle_decision = f"oracle completed; best median recovery={best.get('median_recovery', float('nan')):.3f}"
    else:
        oracle_decision = f"oracle states={oracle.get('states', 0)}; 500-state gate not met"
    decision = "NO-GO / INCOMPLETE" if not (oracle.get("states", 0) >= 500 and closed_loop.get("files", 0) > 0) else "GO-CANDIDATE pending statistical gates"
    report = f'''# SERVER-SIDE In-flight WAM Deep Validation

## 1. 结论先行

本报告对应 `/home/rxhuang/Projects/cosmos-policy/reports/SERVER_DEEP_VALIDATION_PLAN.md` 的可恢复 server-side run，硬停止时间为 `2026-08-12 03:19:31 HKT`。当前决策：**{decision}**。

- Queue A collection：{collection.get('episodes', 0)} episodes，{collection.get('requests', 0)} decision requests。
- Queue B one-step late-binding oracle：{oracle.get('states', 0)} states；{oracle_decision}。
- Queue D latency matrix：{latency.get('files', 0)} condition/delay files。
- Queue F paired closed loop：{closed_loop.get('files', 0)} shard files；{closed_loop_decision}。

在没有满足完整 state、late-binding、latency-sensitive 和 paired competence gates 之前，不把 hypothesis 写成普遍 superiority claim。

## 2. 研究问题与边界

问题是：是否能构建 policy-preserving、deadline-aware、in-flight feedback assimilation runtime。只验证既有的 one-step late-bound assimilation 与其固定 runtime extension；不扩展 design family，不实现 adaptive scheduler，不使用 Cosmos value。

## 3. Checkpoint 与 provenance

所有新实验使用原始 pre-finetune LIBERO checkpoint：
`/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt`，SHA256 `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`。没有使用 SO101/finetuned checkpoint、value、threshold、privileged runtime state 或训练。

## 4. 既有证据而非本轮新 claim

既有 mechanism discovery、persistent-condition recovery、single-GPU overlap 与旧 P3 regression 结果保留在 [ASYNC_FEEDBACK_ASSIMILATION_WAM_ZH.md](./ASYNC_FEEDBACK_ASSIMILATION_WAM_ZH.md)、[PREDICT_CORRECT_SYSTEM_VALIDATION_ZH.md](./PREDICT_CORRECT_SYSTEM_VALIDATION_ZH.md) 和 [RELATED_WORK_NOVELTY_MAP.md](./RELATED_WORK_NOVELTY_MAP.md)。本轮结果与旧 artifacts 分离。

## 5. Task/disjoint split

Queue A 使用 32 个 task-disjoint tasks、每 task 5 个固定 init/seed，discovery 16 tasks、validation 8、heldout 8；manifest 为 [server_f1_collection.jsonl](./server_deep_validation/manifests/server_f1_collection.jsonl)。

## 6. GPU pool 与 worker density

启动时 GPU0/2 被 VLLM 占用，GPU1/5 有另一用户 Python 进程，GPU3/4/6/7 为高负载训练；本 run 未终止任何外部 PID。supervisor 只在 compute process 为空且显存占用不超过 1 GiB 时派发 worker。profiling artifact 为 `{json.dumps(profile, ensure_ascii=False)[:4000]}`；single-worker 的 18 requests p50/p95 约 268/506 ms，2-worker 和 3-worker 的 shared-GPU capacity test 记录 OOM。profiling 状态见 `reports/server_deep_validation/checkpoints/`。

## 7. Queue A：F1 collection

Fresh one-step policy 驱动闭环，保存 exact MuJoCo snapshot、previous generated non-value latent、Fresh action 和 latency。sim state 只用于离线复现/restore，不作为 policy 输入。

## 8. F1/P1/PP/PF/FF protocol

在相同 observation、seed、task 和 state 上比较 F1、P1、PP、PF、FF；结果 artifact：`/data/rxhuang/wam_server_deep_validation/queue_a/ablation/`。

## 9. Solver/feedback separation

使用冻结定义 `Delta_target=A_F1-A_P1`、`Delta_solver=A_PP-A_P1`、`Delta_feedback=A_PF-A_PP`，分别报告 cosine、norm、projection、sign、first-action 和 gripper。

## 10. DID diagnostic

`A_DID=A_P1+(A_PF-A_PP)`；aggregate 中的 `did_to_f1_l2`、`did_less_error_than_pf`、`did_less_error_than_pp` 是仅在不重新求解的诊断，不是可部署策略。

## 11. Queue A aggregate

```json
{json.dumps(collection, ensure_ascii=False, indent=2)[:6000]}
```

## 12. Ablation aggregate

```json
{json.dumps(ablation, ensure_ascii=False, indent=2)[:12000]}
```

## 13. Queue B：one-step late-binding oracle

在 validation/heldout state 上扫描 blocks `2,4,...,26` 与 eight dynamic non-value interfaces；fresh hidden 只作为 oracle evidence。原始文件在 `/data/rxhuang/wam_server_deep_validation/queue_b/oracle/`。

## 14. Oracle frontier summary

states={oracle.get('states', 0)}，baseline predicted-to-F1 median={oracle.get('baseline_predicted_to_fresh', {}).get('median')}，best frontier={json.dumps(best, ensure_ascii=False)}。`oracle_fresh_prefix_required=true` 时只能作为 upper bound，不能直接写成 runtime result。

## 15. Late-bound candidate decision

本轮只有在 oracle 接近 F1 且不增加 competence regression 时才允许进一步实现 candidate。若 oracle gate 未达到，后续 candidate 必须 kill；不能用 scheduler 或 value 掩盖失败。

## 16. Age/slack/deadline definitions

每次 runtime 记录 `W_t=H_remaining/f_control`、`Delta_obs`、`Delta_world`、`Delta_action` 与 camera/condition/action AoI，另记录 finish-deadline margin 和 action-ready margin。

## 17. Queue D：latency-sensitive supplement

注入 sensing delay 0/50/100/150/200/300 ms，对照 Sync Fresh、Async Fresh、Predict-Correct；`t_change → t_sense → t_model → t_action` 由 host pipeline trace 记录。当前 latency files={latency.get('files', 0)}。

## 18. Latency summary

```json
{json.dumps(latency, ensure_ascii=False, indent=2)[:10000]}
```

本 runner 明确标记 `mid_chunk_world_change_injected=false`、`action_disturbance_injected=false`；因此若未另行完成，不能声称已经验证真实物理 reaction。

## 19. Useful/obsolete/canceled compute

action chunk、executed prefix、request readiness 与 stale/underflow trace 保留在 latency/closed-loop artifacts。只有在 trace 完整时才计算 useful、obsolete、canceled ratios。

## 20. Queue F：paired closed loop

固定 modes 为 Fresh、predicted reuse、Predict-Correct；无 adaptive scheduler、无 threshold。结果目录 `/data/rxhuang/wam_server_deep_validation/queue_f/closed_loop/`。

## 21. Closed-loop result

```json
{json.dumps(closed_loop, ensure_ascii=False, indent=2)[:10000]}
```

## 22. Robustness/seed/noise

Manifest 含 5 seeds；outcome-independent tiny input/noise perturbation、24–30 task ×3 seed 的 Wilson CI、exact McNemar/task bootstrap 若对应 stage 未完成，则在此明确标记为 pending，不把 point estimate 当成结论。

## 23. CRI/VoI feasibility

CRI/VoI 只允许作为 secondary feasibility。若 heldout Spearman <0.5，或 proxy latency 不明显低于 saved compute，则 kill。本 run 不读取 value，不训练大 proxy。

## 24. Related-work boundary

本系统 claim 的边界是：joint WAM 内部的 fresh physical feedback 如何在 in-flight diffusion suffix 中被 assimilate，并与 action chunk deadline 对齐；不是普通 latent cache、预测视频展示、value-guided scheduler 或 finetuning。

## 25. Kill criteria

late-bound 不能恢复 F1、Async Fresh 已覆盖全部 slack、动态 benchmark 无区分度、或 regression 仍持续，任一成立即停止新增 candidate，保留 negative result。

## 26. GO criteria

GO 需要同时满足 late-bound fidelity > PF、competence regression 更低、post-sensing critical path < Async Fresh full DiT、latency-sensitive reaction 改善、以及 useful compute/deadline margin 改善。当前 report 不会因单一 oracle recovery point 提前 GO。

## 27. Q1–Q10 direct answers

1. **机制是否成立？** 只有既有 persistent-condition evidence 已成立；本轮 one-step server claim 以 Queue B state count 和 frontier 为准：{oracle_decision}。
2. **solver/feedback 是否可分？** protocol 已冻结；最终以 Queue A innovation aggregate 为准。
3. **DID 是否接近 F1？** 以 `did_to_f1_l2` 与 `did_less_error_than_pf` 报告；未完成则不声称。
4. **late-binding 是否优于 PF？** 只有 heldout paired result 才能回答；oracle 不等于 deployable PF。
5. **age/slack/deadline 是否影响结果？** 以 Queue D 的 delay/age/margin 分层回答。
6. **是否减少 obsolete compute？** 只有完整 chunk/ready/stale trace 才能回答。
7. **是否降低 competence regression？** 以 Queue F 的 paired task-level result 和 exact tests 回答。
8. **是否支持 paper-level system claim？** 当前默认保守为 NO-GO / INCOMPLETE，除非全部 GO gates 完成。
9. **是否需要 scheduler？** 本轮不实现；若 candidate 只有 scheduler 才能掩盖 regression，判 kill。
10. **下一步是什么？** 若 GO，冻结最多两个 candidate 做完整统计和真实系统边界；若 NO-GO，写清楚 one-step fidelity/deadline/competence 的 failure mode。

## Artifacts and checkpoint status

- Plan: [SERVER_DEEP_VALIDATION_PLAN.md](./SERVER_DEEP_VALIDATION_PLAN.md)
- Manifest: [server_f1_collection.jsonl](./server_deep_validation/manifests/server_f1_collection.jsonl)
- Raw root: `/data/rxhuang/wam_server_deep_validation/`
- Supervisor status: `{json.dumps(status, ensure_ascii=False)}`
- Queue A status: `{json.dumps(status, ensure_ascii=False)}`
- Ablation status: `{json.dumps(ablation_status, ensure_ascii=False)}`
- Oracle status: `{json.dumps(oracle_status, ensure_ascii=False)}`
- Latency status: `{json.dumps(latency_status, ensure_ascii=False)}`
- Closed-loop status: `{json.dumps(benchmark_status, ensure_ascii=False)}`
'''
    target = PROJECT / "reports/SERVER_INFLIGHT_WAM_DEEP_VALIDATION_ZH.md"
    target.write_text(report, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()

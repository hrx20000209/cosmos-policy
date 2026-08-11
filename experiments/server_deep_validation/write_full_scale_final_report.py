"""Materialize the full-scale server report at the safe run boundary."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path("/home/rxhuang/Projects/cosmos-policy")
MANIFEST = REPO / "reports/server_deep_validation/manifests/full_scale_40_task.json"
STATUS = REPO / "reports/server_deep_validation/full_scale_checkpoints"
RAW = Path("/data/rxhuang/wam_full_scale_server")
OUT = REPO / "reports/FULL_SCALE_INFLIGHT_WAM_SERVER_ZH.md"
CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e"


def load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def wilson(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def collect_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted((RAW / "queue_f/closed_loop").glob("group*/benchmark_shard*.json")):
        payload = load(path, {}) or {}
        records.extend(payload.get("records", []))
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[str, int, int], dict[str, bool]] = defaultdict(dict)
    per_task: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"successes": 0, "episodes": 0}))
    for record in records:
        method = record.get("configuration", "unknown")
        by_method[method].append(record)
        key = (record.get("task_uid", "unknown"), int(record.get("init_state_index", -1)), int(record.get("seed", -1)))
        by_pair[key][method] = bool(record.get("success"))
        task = record.get("task_uid", "unknown")
        per_task[task][method]["episodes"] += 1
        per_task[task][method]["successes"] += int(bool(record.get("success")))
    summary = {}
    for method, values in sorted(by_method.items()):
        successes = sum(bool(value.get("success")) for value in values)
        summary[method] = {"episodes": len(values), "successes": successes, "success_rate": successes / len(values) if values else None, "wilson_95": wilson(successes, len(values))}
    transitions = {}
    for baseline, candidate in (("fresh", "predict_correct"), ("fresh", "predict_correct_async"), ("fresh", "predicted_reuse")):
        pairs = [(value.get(baseline), value.get(candidate)) for value in by_pair.values() if baseline in value and candidate in value]
        transitions[f"{baseline}_vs_{candidate}"] = {
            "n": len(pairs),
            "both_success": sum(a and b for a, b in pairs),
            "baseline_success_candidate_failure": sum(a and not b for a, b in pairs),
            "baseline_failure_candidate_success": sum((not a) and b for a, b in pairs),
            "both_failure": sum((not a) and (not b) for a, b in pairs),
        }
    return {"summary": summary, "transitions": transitions, "per_task": per_task, "scenarios": len(by_pair)}


def main() -> None:
    hard_stop = float((load(STATUS / "run_manifest.json", {}) or {}).get("hard_stop_epoch", 0))
    marker = STATUS / "run_full_scale_closed_loop_benchmark_complete.json"
    while not marker.exists() and (not hard_stop or time.time() < hard_stop):
        time.sleep(60)
    manifest = load(MANIFEST, {}) or {}
    records = collect_records()
    closed = summarize_records(records)
    collection_files = list((RAW / "queue_a/f1_collection").glob("group*/episode_*.pt"))
    ablation_files = list((RAW / "queue_b/ablation").glob("group*/state_*.pt"))
    oracle_files = list((RAW / "queue_b/oracle").glob("group*/state_*.json"))
    latency_files = list((RAW / "queue_d/latency").glob("group*/delay*.json"))
    status = {
        "collection_complete": (STATUS / "queue_a_complete.json").exists(),
        "ablation_complete": (STATUS / "run_server_ablation_complete.json").exists(),
        "oracle_complete": (STATUS / "run_server_late_binding_oracle_complete.json").exists(),
        "latency_complete": (STATUS / "run_server_latency_matrix_complete.json").exists(),
        "closed_loop_complete": marker.exists(),
    }
    report = f"""# FULL-SCALE IN-FLIGHT WAM SERVER VALIDATION

状态：`{'complete' if marker.exists() else 'bounded_run_or_pending'}`  
Checkpoint：原始 pre-finetune Cosmos LIBERO Predict2 2B  
SHA256：`{CHECKPOINT_SHA256}`  
`denoising_steps=1`；`value_used=false`；`training_used=false`。

## 1. Executive Summary

本报告验证的是 frozen unified WAM 中的 physical observation innovation assimilation，而不是 Cosmos acceleration trick。当前主设计被拆成三个模块：Speculative Joint-State Producer、Policy-Preserving Feedback Transport、Arrival–Depth–Action Contract。任何训练分支都与主结果隔离。

当前数据状态：40 tasks、200 fixed-init scenarios；已写入 collection episode files `{len(collection_files)}`、F1/PF ablation state files `{len(ablation_files)}`、oracle state files `{len(oracle_files)}`、latency files `{len(latency_files)}`、closed-loop records `{len(records)}`。

## 2. Modular Design

### Module 1 — Speculative Joint-State Producer（SJP）

利用上一 request 的 predicted future non-value slots 启动 speculative joint computation。只有在 fresh feedback 到达后仍然有效的前缀计为 useful；被 feedback 取代的计算计为 obsolete。此模块不读取 value、不训练、不改变 Fresh-1 policy。

### Module 2 — Policy-Preserving Feedback Transport（PFT）

fresh sensing 不被当成 next-request replacement，也不是 one-shot hidden patch。它沿 arrival depth `k` 进入仍有效的 persistent condition interface `I*(k)`，再继续有效 suffix。`I*(k)` 必须由 full depth recovery–compute frontier 得到，不能写死为某个 block。当前 PF/Predict–Correct 仍只标为 scaffold，除非 one-step late-bound oracle 和 runtime gate 同时通过。

### Module 3 — Arrival–Depth–Action Contract（ADAC）

记录 `t_sense → k → I*(k) → corrected action → W_t`，其中 `W_t` 是 action buffer expiry 前的 physical deadline。记录 `Delta_obs`、`Delta_world`、`Delta_action`、remaining suffix cost、ready margin、deadline miss、`C_total/C_useful/C_obsolete/C_recomputed/C_canceled`。这是 deterministic feasibility semantics，不是 threshold scheduler。

## 3. Training Registry

主路径：**不需要训练**。原因是论文问题本身要求保持 original Fresh-1 policy；训练 correction head 会把 policy adaptation 与 feedback entry semantics 混在一起。

预注册但未启动的 fallback：`PFT-TinyAdapter`。只有 existing architectural condition pathway 在 held-out 上失败，且确认不是 deadline/simulator artifact，才允许使用 discovery-task aligned states 训练 `<100k` 参数的 tiny residual transport。它必须独立报告 train/validation/held-out、参数量、seed、训练时间、checkpoint hash，并且不能再被称为严格 frozen-policy path。当前 `training_used=false`，没有任何训练模块进入主结果。

## 4. Dataset and Protocol

Manifest：`{manifest.get('tasks', 40)} tasks × {manifest.get('inits_per_task', 5)} initial states`。所有 suite/task 都保留，task split 只用于 discovery/validation/held-out 分析，不能删掉 held-out。固定 checkpoint、seed mapping、denoise=1；不使用 privileged state 作为 policy input。

## 5. Literature Boundary

本工作不主张 dynamic denoise、one-step distillation、future bypass、generic cache、layer skipping、external verifier、action timing 或 dual-DiT planner/executor。它与 X-WAM/SANTS/Flash-WAM、Fast-WAM/Faster-WAM/Efficient-WAM/SelfWAM、FFDC/CheckVLA、RTC/VLASH/REMAC/FutureRTC、AHA-WAM OVCR 的区别是：fresh sensing 在 already-running single unified WAM 内的 depth-dependent feedback entry 与 physical action deadline。

## 6. Full-Scale Closed-Loop Summary

```json
{json.dumps(closed['summary'], ensure_ascii=False, indent=2)}
```

Paired transitions：

```json
{json.dumps(closed['transitions'], ensure_ascii=False, indent=2)}
```

上表只在 paired scenario 数完整时作为 full-scale 结论；不把少数成功 episode 或 aggregate point estimate 包装成普遍 superiority。per-task 原始结果保存在 closed-loop shard JSON 中。

## 7. Gate / Decision

阶段标记：`{json.dumps(status, ensure_ascii=False)}`。

最终 GO 需要同时满足：solver bias 与 feedback effect 可分离；late-bound fidelity 接近 Fresh-1；p95 policy fidelity 通过；Fresh-success regression 可接受；candidate 在 action deadline 前 ready；latency-sensitive 40-task benchmark 的 reaction latency 相对 Async Fresh 有实际改善。

若任一关键 gate 不满足，结论为 `PIVOT` 或 `NO-GO`，保留 negative result；不添加 scheduler、threshold、task heuristic 或 value branch 进行补救。

## 8. Reproducibility and Provenance

- checkpoint SHA256：`{CHECKPOINT_SHA256}`；
- value：`false`；privileged runtime input：`false`；adaptive scheduler：`false`；training：`false`；
- full manifest：`reports/server_deep_validation/manifests/full_scale_40_task.jsonl`；
- plan：`reports/FULL_SCALE_SERVER_PLAN.md`；
- raw root：`{RAW}`；
- run status：`{STATUS}`。
"""
OUT.write_text(report, encoding="utf-8")
print(OUT)


if __name__ == "__main__":
    main()

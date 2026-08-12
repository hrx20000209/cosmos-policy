#!/usr/bin/env python3
"""Materialize the final, evidence-bounded PV0 research deliverables.

This is an offline report/export writer.  It validates existing artifacts and
source-level interface invariants; it never loads a policy, changes a route,
or starts a new design search.  The resulting documents deliberately separate
the mechanism result (PV0 versus pure predicted reuse) from the stronger, not
yet supported claim that PV0 can replace always-fresh conditioning everywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
CHECKPOINT = Path("/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
STATE_KEY = "9f535f7df63f9ab01464446ce09afa6dbf5146f9b9460bb08ccc73f9c85a45d8:req1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def summary(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise KeyError(f"missing summary {key}")
    return result


def scalar(value: dict[str, Any], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)):
        raise KeyError(f"missing numeric key {key}")
    return float(item)


def median(value: dict[str, Any]) -> float:
    """Accept the two established summary schemas (p50 versus median)."""

    for key in ("p50", "median"):
        item = value.get(key)
        if isinstance(item, (int, float)):
            return float(item)
    raise KeyError("missing median/p50 summary value")


def source_line(path: Path, needle: str) -> int:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return index
    raise ValueError(f"could not locate {needle!r} in {path}")


def source_assertions(repo: Path) -> dict[str, Any]:
    """Static audit of the frozen native condition interface.

    We keep this deliberately source-level: it establishes which interface was
    run, while the Phase-A/Phase-B artifacts establish what it did.
    """

    utils = repo / "cosmos_policy/experiments/robot/cosmos_utils.py"
    config = repo / "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py"
    model = repo / "cosmos_policy/models/policy_video2world_model.py"
    adapter = repo / "adapters/cosmos_adapter.py"
    files = (utils, config, model, adapter)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    utils_text = utils.read_text(encoding="utf-8")
    config_text = config.read_text(encoding="utf-8")
    model_text = model.read_text(encoding="utf-8")
    adapter_text = adapter.read_text(encoding="utf-8")
    required = {
        "temporal_compression": "COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4",
        "13_frame_argument": "persistent_visual_correction_prefix_frames: int | None = None",
        "prefix_only_encode": 'model.encode(data_batch["video"][:, :, :prefix_frames])',
        "condition_slot_replace": "condition.gt_frames.index_copy_(",
        "speculative_latent_required": "persistent visual correction requires skip_vae_encoding and a speculative latent",
        "arrival_before_only_forward": "persistent_visual_correction_arrival=",
        "condition_hook": "inference_condition_transform",
        "state_layout": "state_t=9",
        "four_condition_slots": "min_num_conditional_frames=4",
        "prefix_route_call": "persistent_visual_correction_prefix_frames=(",
    }
    haystacks = {
        "temporal_compression": utils_text,
        "13_frame_argument": utils_text,
        "prefix_only_encode": utils_text,
        "condition_slot_replace": utils_text,
        "speculative_latent_required": utils_text,
        "arrival_before_only_forward": adapter_text,
        "condition_hook": model_text,
        "state_layout": config_text,
        "four_condition_slots": config_text,
        "prefix_route_call": adapter_text,
    }
    missing = [name for name, phrase in required.items() if phrase not in haystacks[name]]
    if missing:
        raise RuntimeError(f"static PV0 interface audit failed: missing={missing}")
    return {
        "status": "PASS",
        "source_files": {
            str(path.relative_to(repo)): {"sha256": sha256(path)} for path in files
        },
        "source_locations": {
            "temporal_compression_factor": {
                "path": str(utils.relative_to(repo)),
                "line": source_line(utils, "COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4"),
            },
            "prefix_slice_encode": {
                "path": str(utils.relative_to(repo)),
                "line": source_line(utils, 'model.encode(data_batch["video"][:, :, :prefix_frames])'),
            },
            "condition_slot_replace": {
                "path": str(utils.relative_to(repo)),
                "line": source_line(utils, "condition.gt_frames.index_copy_("),
            },
            "one_forward_arrival": {
                "path": str(adapter.relative_to(repo)),
                "line": source_line(adapter, "persistent_visual_correction_arrival=("),
            },
            "condition_transform_hook": {
                "path": str(model.relative_to(repo)),
                "line": source_line(model, "inference_condition_transform"),
            },
            "state_layout": {
                "path": str(config.relative_to(repo)),
                "line": source_line(config, "state_t=9"),
            },
        },
    }


def prefix_vae_audit(repo: Path, source: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "PV0_prefix_VAE_architecture_audit",
        "status": "PASS_STATIC_INTERFACE_AUDIT",
        "scope": "source/interface audit only; no new model run or architecture search",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "denoising_steps": 1,
        "value_used": False,
        "training_used": False,
        "hidden_activation_patch_used": False,
        "native_interface": "get_action:persistent_visual_correction_prefix_frames",
        "temporal_vae": {
            "compression_factor": 4,
            "raw_prefix_frames": 13,
            "why_13": "1 structural blank frame + 4 duplicated proprio frames + 4 duplicated wrist-camera frames + 4 duplicated primary-camera frames.",
            "encoded_prefix_latent_slots": [0, 1, 2, 3],
            "current_visual_latent_slots_replaced": [2, 3],
            "unchanged_from_previous_generated_latent": [1, 4, 5, 6, 7, 8],
            "slot_layout": [
                "0: structural blank",
                "1: current proprio",
                "2: current wrist image",
                "3: current primary image",
                "4: action chunk",
                "5: future proprio",
                "6: future wrist image",
                "7: future primary image",
                "8: value slot (never read by this experiment)",
            ],
        },
        "runtime_semantics": {
            "bootstrap": "F1 encodes current cameras normally.",
            "followup_start": "P1/PV0 clone the previous generated joint latent as the sampling/condition basis.",
            "pv0_correction": "PV0 preprocesses current cameras, VAE-encodes only the 13-frame causal prefix, selects visual slots 2/3, and copies them into the persistent condition before denoiser forward 0.",
            "denoiser_forwards": 1,
            "not_used": [
                "Cosmos value output",
                "simulator state as a runtime policy input",
                "hidden activation patch",
                "learned correction head",
                "scheduler/threshold",
                "dynamic denoising",
            ],
        },
        "source_audit": source,
    }


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_pp(value: float) -> str:
    return f"{100.0 * value:.1f} pp"


def final_report(
    analysis: dict[str, Any],
    phase_a: dict[str, Any],
    s1: dict[str, Any],
    s4: dict[str, Any],
    latency: dict[str, Any],
    replay: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    phase_b = analysis["phase_b_600"]
    routes = phase_b["route_summary"]
    comparisons = phase_b["paired_comparisons"]
    heldout = phase_b["split_summary"]["heldout"]
    heldout_compare = phase_b["paired_comparisons_by_split"]["heldout"]
    clean = analysis["clean_gpu_microbenchmark"]
    f1_ms = median(clean["route_wall_latency_ms"]["F1"])
    p1_ms = median(clean["route_wall_latency_ms"]["P1"])
    pv0_ms = median(clean["route_wall_latency_ms"]["PV0"])
    f1_model_ms = float(np_median_latency(latency, "F1", "model_generate_inclusive_ms"))
    pv0_model_ms = float(np_median_latency(latency, "PV0", "model_generate_inclusive_ms"))
    pv0_wall_reduction = 1.0 - pv0_ms / f1_ms
    pv0_model_reduction = 1.0 - pv0_model_ms / f1_model_ms
    s1_heldout = s1["splits"]["heldout"]
    s4_heldout = s4["splits"]["heldout"]
    first_failure = next(
        item for item in replay["replays"] if item["mode"] == "native_persistent" and not item["recorded_success"]
    )
    decision = "PIVOT：机制 GO；“PV0 可普遍替代 Fresh”主张 NO-GO"
    return "\n".join(
        [
            "# PV0 / Cosmos WAM 最终研究报告",
            "",
            f"## CURRENT DECISION: {decision}",
            "",
            "PV0 已经是一个有证据支持的系统机制雏形：它用当前物理视觉校正上一轮生成 latent 的视觉条件，并在固定 one-denoise Cosmos 中修复纯预测复用的漂移。它**还不是**一个可宣称全面替代 always-fresh inference 的系统：完整 40-task 基准里有两个 Fresh-only success。",
            "",
            "## 已完成的冻结协议",
            "",
            "| 阶段 | 覆盖 | 结论 |",
            "| --- | ---: | --- |",
            f"| S1 state fidelity | 3,801 states / 40 tasks | `{phase_a['phase_statuses']['S1']}` |",
            f"| S2 predictive-prior decomposition | 3,801 states | `{phase_a['phase_statuses']['S2']}` |",
            f"| S3 execution-prefix alignment | K=4/8/12/16 | `{phase_a['phase_statuses']['S3']}` |",
            f"| S4 condition compile control | 40 fixed anchors | `{phase_a['phase_statuses']['S4']}` |",
            "| S5 action-outcome mismatch | 1 preregistered scenario | preliminary only |",
            "| Phase B closed loop | 40 tasks × 5 init × 3 routes = 600 | complete |",
            "",
            "所有阶段都使用原始 pre-finetune Cosmos checkpoint（SHA-256 已锁定）、`denoise=1`，不读取 Cosmos value；没有 SO101 finetune、scheduler、threshold、hidden activation patch、fresh-prefix oracle 或 runtime privileged state。",
            "",
            "## 三个运行时模块（均不训练）",
            "",
            "1. **Speculative joint-latent reuse (P1)**：保存上一轮 generated joint latent，作为下一 request 的起点。",
            "2. **Native Persistent Visual Condition (PV0)**：仅重编码最新物理视觉的因果 13-frame VAE prefix，并在唯一 denoiser forward 前替换 current visual condition slots。",
            "3. **Fixed execution-prefix contract**：动作仍按固定 16-step prefix 执行；K=4/8/12/16 只是离线对齐审计，不是 adaptive scheduler。",
            "",
            "训练注册表：**无训练模块**。主路径保持 frozen original policy；没有 residual head、value head 或 finetune checkpoint。",
            "",
            "## 机制证据：状态层",
            "",
            f"- Held-out 1,066 states：PV0→F1 mean-step L2 的中位数 `{s1_heldout['pv0_to_f1_mean_step_l2']['median']:.4f}`、p95 `{s1_heldout['pv0_to_f1_mean_step_l2']['p95']:.4f}`；P1→F1 对应中位数 `{s1_heldout['p1_to_f1_mean_step_l2']['median']:.4f}`、p95 `{s1_heldout['p1_to_f1_mean_step_l2']['p95']:.4f}`。",
            f"- Held-out states 中 PV0 在 ε=0.05 内的 fidelity 为 `{fmt_pct(float(s1_heldout['pv0_fidelity_at_epsilon']['fraction']))}`，PV0 优于 P1 的比例为 `{fmt_pct(float(s1_heldout['pv0_beats_p1']['fraction']))}`；所有 held-out tasks 的首个 gripper sign 均匹配。",
            f"- S4 held-out 12 anchors：正确 fresh visual condition 优于 shuffled condition 的比例 `{fmt_pct(float(s4_heldout['correct_beats_shuffled']['fraction']))}`；正确 PV0→F1 L2 mean `{s4_heldout['pv0_correct_to_f1_mean_step_l2']['mean']:.4f}`，shuffled 为 `{s4_heldout['pv0_shuffled_to_f1_mean_step_l2']['mean']:.4f}`。这表明效果来自正确物理视觉条件，而非 latent-reuse 假象。",
            "",
            "## 闭环结果：应主张什么、不能主张什么",
            "",
            "| Route | Success | Success rate |",
            "| --- | ---: | ---: |",
            f"| Fresh (F1) | {routes['fresh']['successes']}/{routes['fresh']['episodes']} | {fmt_pct(float(routes['fresh']['success_rate']))} |",
            f"| Predicted reuse (P1) | {routes['predicted_reuse']['successes']}/{routes['predicted_reuse']['episodes']} | {fmt_pct(float(routes['predicted_reuse']['success_rate']))} |",
            f"| PV0 | {routes['native_persistent']['successes']}/{routes['native_persistent']['episodes']} | {fmt_pct(float(routes['native_persistent']['success_rate']))} |",
            "",
            f"- PV0 vs P1：{comparisons['pv0_vs_p1']['left_win_right_loss']} wins / {comparisons['pv0_vs_p1']['left_loss_right_win']} losses，Δ={fmt_pp(float(comparisons['pv0_vs_p1']['left_minus_right_success_rate']))}，per-scenario exact McNemar p={comparisons['pv0_vs_p1']['exact_mcnemar_two_sided_p']:.4f}；但 task-hierarchical bootstrap 95% CI 为 {comparisons['pv0_vs_p1']['hierarchical_bootstrap']['ci95']}，下界为 0，因此不能过度表述 task-general significance。",
            f"- Held-out 12 tasks × 5 init：Fresh={heldout['successes_by_mode']['fresh']}/{heldout['paired_scenarios']}，P1={heldout['successes_by_mode']['predicted_reuse']}/{heldout['paired_scenarios']}，PV0={heldout['successes_by_mode']['native_persistent']}/{heldout['paired_scenarios']}；PV0 vs P1 为 {heldout_compare['pv0_vs_p1']['left_win_right_loss']}/0，p={heldout_compare['pv0_vs_p1']['exact_mcnemar_two_sided_p']:.4f}；PV0 与 Fresh 逐对完全相同。",
            f"- PV0 vs Fresh（完整 200 scenarios）：0 wins / {comparisons['pv0_vs_fresh']['left_loss_right_win']} losses，Δ={fmt_pp(float(comparisons['pv0_vs_fresh']['left_minus_right_success_rate']))}。因此 PV0 的正确定位是 **recovery of predictive reuse**, 而不是 **universal Fresh replacement**。",
            "",
            "## 成本证据（clean GPU，单一已存物理状态，25 warm repeats，轮换执行顺序）",
            "",
            f"- F1 median `{f1_ms:.1f} ms`，P1 `{p1_ms:.1f} ms`，PV0 `{pv0_ms:.1f} ms`；PV0 比 F1 低 `{fmt_pct(pv0_wall_reduction)}`。",
            f"- model-generate median：F1 `{f1_model_ms:.1f} ms`，PV0 `{pv0_model_ms:.1f} ms`，PV0 降低 `{fmt_pct(pv0_model_reduction)}`。",
            f"- 同一状态的 PV0 action recovery 为 `{fmt_pct(float(clean['native_action_recovery']['median']))}`（相对 P1→F1 discrepancy）。这是 4090 的同 worker route-cost 证据，不可外推为 end-to-end closed-loop latency，也不是 Thor latency claim。",
            "",
            "## 负结果与 failure diagnosis",
            "",
            "- 两个 base-seed Fresh success → PV0 failure 都是 LIBERO-10 多物体入篮任务；两条 P1 也同样失败并达 max-steps。",
            "- 离线回放已 SHA 验证 saved actions：首例 Fresh 251 steps 成功，而 P1/PV0 都执行 520 steps 未完成；第二例 Fresh 331 steps 成功，而 P1/PV0 都执行 520 steps 未完成。",
            "- 两例的首个 16-action prefix 与 Fresh 几乎一致，偏离在后续闭环累积后才出现。第二例 PV0 有显著 gripper chatter / chunk-boundary discontinuity；这是长期闭环残余失配的候选诊断，不构成单一因果证明。",
            f"- 代表性回放：`failure_replays/{Path(first_failure['video_path']).name}`，以及同一目录下 Fresh/P1 对照视频。",
            "",
            "## S5 与种子稳健性",
            "",
            "- S5 的一个预注册 zero-motion/preserve-gripper interruption 场景中三条路线均成功。样本量为 1，不能写成 disturbance-recovery rate。",
            "- 额外 3 inference seeds 仅针对 Fresh-success / disagreement 条件子集（57 matched pairs），PV0 对 Fresh 为 3 wins / 1 loss；该选择性子集不能当成总体 success rate。",
            "",
            "## 论文定位与下一步",
            "",
            "可发展的论文中心问题是：**在 unified WAM 中，如何将到达的新鲜物理视觉作为 persistent condition 写入正在复用的世界 latent，从而恢复 prediction-only reuse 的控制正确性，并保留计算收益？**",
            "",
            "当前可用的论文结论：PV0 将 P1 的 latent drift 拉回接近 Fresh action，并在 held-out closed loop 中保留 Fresh 成功、修复所有 P1-only losses。",
            "",
            "当前不能写的结论：PV0 普遍优于 Fresh、已完成泛化硬件验证、或 action-outcome recovery 已被大样本证明。下一轮应是冻结机制下的更广 success/regression audit 与目标 Thor 测量，而不是添加 scheduler、value 或新 design family。",
            "",
            "## 可复核资产",
            "",
            "- `CLOSED_LOOP_PAIRED_ANALYSIS.json`：600 条路线、200 个 paired scenarios 的完整审计与 bootstrap。",
            "- `PREFIX_VAE_ARCHITECTURE_AUDIT.md`：13-frame prefix 的源代码级解释。",
            "- `failure_replays/`：6 个 exact saved-action MP4 与 replay summary。",
            "- `thor_export/`：目标硬件测量合同，不含任何 Thor 性能声明。",
            f"- 当前 artifact audit：S1={audit['s1_valid_records']}/{audit['state_index_expected']}，S4={audit['s4_valid_records']}/{audit['s4_expected_anchors']}，原始 checkpoint SHA 已通过。",
            "",
        ]
    )


def np_median_latency(latency: dict[str, Any], route: str, metric: str) -> float:
    values = [
        float(trial["inference_metrics"][route][metric])
        for trial in latency["trials"]
        if metric in trial["inference_metrics"][route]
    ]
    values.sort()
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def prefix_markdown(audit: dict[str, Any]) -> str:
    locations = audit["source_audit"]["source_locations"]
    slot_lines = "\n".join(f"- {item}" for item in audit["temporal_vae"]["slot_layout"])
    return "\n".join(
        [
            "# PV0 Prefix-VAE Architecture Audit",
            "",
            "## Conclusion",
            "",
            "PV0 is a native **condition-slot replacement**, not a hidden activation patch. It starts from the prior generated joint latent, encodes only the causal prefix needed to recover the current wrist/primary visual slots, and writes those slots into the condition before denoiser forward 0. No trainable module is introduced.",
            "",
            "## Why 13 frames",
            "",
            "The policy temporal VAE uses a 4-frame compression factor. The source builds the first 13 raw frames as one structural blank plus four duplicated frames each for proprio, wrist image, and primary image. That yields latent slots 0–3; PV0 selects only current visual slots 2 and 3.",
            "",
            "## Latent slot layout",
            "",
            slot_lines,
            "",
            "PV0 leaves current proprio and every predicted/action/future/value slot inherited from the previous generated latent. The value slot is structurally present in the model layout but is never read by this experiment.",
            "",
            "## Source evidence",
            "",
            f"- Temporal compression factor: `{locations['temporal_compression_factor']['path']}:{locations['temporal_compression_factor']['line']}`.",
            f"- Prefix-only VAE encode: `{locations['prefix_slice_encode']['path']}:{locations['prefix_slice_encode']['line']}`.",
            f"- Condition `gt_frames` slot replacement: `{locations['condition_slot_replace']['path']}:{locations['condition_slot_replace']['line']}`.",
            f"- Arrival 0 for native persistent route: `{locations['one_forward_arrival']['path']}:{locations['one_forward_arrival']['line']}`.",
            f"- Condition hook occurs before the denoiser: `{locations['condition_transform_hook']['path']}:{locations['condition_transform_hook']['line']}`.",
            f"- Nine-slot policy layout: `{locations['state_layout']['path']}:{locations['state_layout']['line']}`.",
            "",
            "## Guardrails",
            "",
            "- Original pre-finetune checkpoint only; `denoise=1`.",
            "- No Cosmos value, simulator state runtime input, scheduler, threshold, hidden patch, learned head, or dynamic denoise.",
            "- The static audit establishes interface semantics; Phase-A/Phase-B artifacts establish numerical and closed-loop behavior.",
            "",
        ]
    )


def thor_contract(repo: Path, latency: dict[str, Any], prefix: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    runner = "experiments/server_deep_validation/run_native_persistent_condition_preflight.py"
    state_root = "/data/rxhuang/wam_full_scale_server/queue_a/f1_collection"
    contract = {
        "schema_version": 1,
        "status": "READY_FOR_TARGET_MEASUREMENT_NOT_MEASURED_ON_THOR",
        "purpose": "portable target-hardware route-cost measurement; not an extrapolated Thor performance claim",
        "frozen_policy": {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "finetuning_used": False,
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "hidden_activation_patch_used": False,
        },
        "routes": {
            "F1": "full current camera preprocessing and VAE encoding",
            "P1": "previous generated latent with camera preprocessing bypass",
            "PV0": "P1 plus native 13-frame current visual prefix, condition arrival before forward 0",
        },
        "portable_inputs": {
            "state_key": STATE_KEY,
            "collection_root": state_root,
            "manifest": "reports/server_deep_validation/manifests/full_scale_40_task.jsonl",
            "runner": runner,
            "repeats": 100,
            "warmup": 10,
            "order": "runner rotates F1/P1/PV0 each repeat",
        },
        "must_record": [
            "wall latency per route",
            "camera preprocessing latency",
            "VAE/prefix encode latency where instrumented",
            "model_generate_inclusive latency",
            "peak allocated/reserved GPU memory",
            "P1-to-F1 and PV0-to-F1 action recovery",
            "software/device/clock/power provenance",
        ],
        "desktop_reference_only": {
            "device": latency["gpu"]["name"],
            "p50_wall_ms": {
                route: median(latency["summary"]["route_wall_latency_ms"][route]) for route in ("F1", "P1", "PV0")
            },
            "action_recovery_median": latency["summary"]["native_action_recovery"]["median"],
            "warning": "These are 4090 measurements and must not be reported as Thor results.",
        },
        "prefix_vae_audit": {
            "artifact": "PREFIX_VAE_ARCHITECTURE_AUDIT.json",
            "source_audit_status": prefix["source_audit"]["status"],
        },
    }
    readme = "\n".join(
        [
            "# Thor PV0 Measurement Contract",
            "",
            "Status: **ready for target measurement; not measured on Thor**.",
            "",
            "This package transfers the frozen F1/P1/PV0 route contract. It does not license copying desktop-4090 timing into a Thor result.",
            "",
            "## Required invariants",
            "",
            "- Use exactly the original pre-finetune LIBERO checkpoint SHA recorded in `THOR_EXPORT_CONTRACT.json`.",
            "- Keep `denoising_steps=1`; do not enable a value branch, scheduler, threshold, finetune checkpoint, or dynamic denoise.",
            "- Record all three routes in the runner's rotated order and retain raw JSON.",
            "- Run after copying/validating the required checkpoint, assets, LIBERO initial states, manifest, and source collection state.",
            "",
            "## Target command template",
            "",
            "```bash",
            "export CUDA_VISIBLE_DEVICES=0",
            "export EVAL_PHYSICAL_GPU=0",
            "export MUJOCO_GL=egl",
            "export PYOPENGL_PLATFORM=egl",
            "python experiments/server_deep_validation/run_native_persistent_condition_preflight.py \\",
            f"  --state-key {STATE_KEY} \\",
            "  --manifest reports/server_deep_validation/manifests/full_scale_40_task.jsonl \\",
            f"  --collection-root {state_root} \\",
            "  --output reports/pv0_thor/preflight_repeats.json \\",
            "  --repeats 100 --warmup 10 --memory-fraction 0.40",
            "```",
            "",
            "Before measurement, replace only machine-local paths as needed and verify the checkpoint SHA. Do not substitute the SO101-finetuned checkpoint.",
            "",
        ]
    )
    shell = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Fill local paths before execution; this script deliberately does not guess them.",
            'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
            'export EVAL_PHYSICAL_GPU="${EVAL_PHYSICAL_GPU:-0}"',
            'export MUJOCO_GL="${MUJOCO_GL:-egl}"',
            'export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"',
            "",
            "python experiments/server_deep_validation/run_native_persistent_condition_preflight.py \\",
            f"  --state-key {STATE_KEY} \\",
            "  --manifest reports/server_deep_validation/manifests/full_scale_40_task.jsonl \\",
            f"  --collection-root {state_root} \\",
            "  --output reports/pv0_thor/preflight_repeats.json \\",
            "  --repeats 100 --warmup 10 --memory-fraction 0.40",
            "",
        ]
    )
    return contract, readme, shell


def reproducibility_audit(
    repo: Path,
    run_dir: Path,
    phase_a: dict[str, Any],
    analysis: dict[str, Any],
    latency: dict[str, Any],
    replay: dict[str, Any],
    prefix: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "Phase-A decision": run_dir / "PHASE_A_DECISION.json",
        "S1 fidelity": run_dir / "S1_PV0_FULL_SCALE_FIDELITY.json",
        "S2 decomposition": run_dir / "S2_PREDICTIVE_PRIOR_DECOMPOSITION.json",
        "S3 execution-prefix": run_dir / "S3_EXECUTION_PREFIX_ALIGNMENT.json",
        "S4 condition control": run_dir / "S4_CONDITION_COMPILE_VALIDATION.json",
        "paired closed-loop": run_dir / "CLOSED_LOOP_PAIRED_ANALYSIS.json",
        "clean latency": run_dir / "clean_latency/preflight_repeats_clean_rerun.json",
        "saved-action replays": run_dir / "failure_replays/replay_summary.json",
    }
    absent = [name for name, path in required.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"missing required final artifacts: {absent}")
    phase_statuses = phase_a.get("phase_statuses") or {}
    if set(phase_statuses) != {"S1", "S2", "S3", "S4"} or any(status != "GO" for status in phase_statuses.values()):
        raise ValueError("Phase-A artifacts are not all GO")
    phase_b = analysis["phase_b_600"]
    if phase_b["route_artifact_count"] != 600 or phase_b["paired_scenarios"] != 200 or phase_b["task_count"] != 40:
        raise ValueError("Phase-B completeness invariant failed")
    if latency.get("status") != "PASS_EXECUTED":
        raise ValueError("clean GPU benchmark was not a PASS")
    if replay.get("status") != "PASS" or len(replay.get("replays", [])) != 6:
        raise ValueError("replay audit is incomplete")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("original checkpoint SHA mismatch at final audit")
    source_files = prefix["source_audit"]["source_files"]
    checkpoint_path = str(CHECKPOINT)
    return {
        "schema_version": 1,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_name_guard": {
            "contains_so101": "so101" in checkpoint_path.lower(),
            "contains_finetune": "finetun" in checkpoint_path.lower(),
            "finetuning_used": False,
        },
        "protocol": {
            "denoising_steps": 1,
            "value_used": False,
            "privileged_runtime_state_used": False,
            "scheduler_or_threshold_used": False,
            "hidden_activation_patch_used": False,
            "training_used": False,
        },
        "completeness": {
            "phase_a": phase_statuses,
            "phase_b_route_artifacts": phase_b["route_artifact_count"],
            "phase_b_paired_scenarios": phase_b["paired_scenarios"],
            "phase_b_tasks": phase_b["task_count"],
            "clean_latency_status": latency["status"],
            "saved_action_replays": len(replay["replays"]),
        },
        "artifact_sha256": {name: sha256(path) for name, path in required.items()},
        "source_sha256": source_files,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        },
    }


def morning_readme(analysis: dict[str, Any], audit: dict[str, Any]) -> str:
    phase_b = analysis["phase_b_600"]
    routes = phase_b["route_summary"]
    paired = phase_b["paired_comparisons"]
    clean = analysis["clean_gpu_microbenchmark"]
    return "\n".join(
        [
            "# Morning README — PV0 Overnight Run",
            "",
            "## Current decision",
            "",
            "**PIVOT:** Phase-A mechanism gates are GO, and PV0 is better than pure predicted reuse; however, PV0 is not yet a universal Fresh replacement because two Fresh-only successes remain on the 200-scenario full benchmark.",
            "",
            "## What finished",
            "",
            "1. S1 3,801-state fidelity, S2 decomposition, S3 K=4/8/12/16 alignment, and S4 40-task condition control all passed.",
            "2. Phase B completed: 40 tasks × 5 initial states × F1/P1/PV0 = 600 route episodes.",
            "3. Selected extra-seed confirmation, paired bootstrap, clean-GPU 25-repeat cost benchmark, static prefix-VAE audit, and six verified failure replay videos completed.",
            "",
            "## Strongest positive result",
            "",
            f"PV0={routes['native_persistent']['successes']}/200 vs P1={routes['predicted_reuse']['successes']}/200: {paired['pv0_vs_p1']['left_win_right_loss']} paired recoveries and zero P1-only wins. Held-out PV0=12/60 equals Fresh=12/60 while P1=6/60.",
            "",
            "## Strongest negative result",
            "",
            f"Fresh={routes['fresh']['successes']}/200 while PV0={routes['native_persistent']['successes']}/200: two Fresh-only successes. Both were replayed exactly; the residual issue is late closed-loop divergence, not an initial prefix mismatch.",
            "",
            "## Cost",
            "",
            f"Clean 4090 same-worker p50: F1={clean['route_wall_latency_ms']['F1']['median']:.1f} ms, PV0={clean['route_wall_latency_ms']['PV0']['median']:.1f} ms, P1={clean['route_wall_latency_ms']['P1']['median']:.1f} ms. This is not a Thor claim.",
            "",
            "## Next user action",
            "",
            "Read `FINAL_PV0_REPORT_ZH.md` first. The paper seed should center on persistent fresh visual conditioning as a recovery mechanism for predictive latent reuse, then validate Fresh-regression bounds and Thor measurements before making a systems performance claim.",
            "",
            f"Final audit: `{audit['status']}`; no finetune, no value, no scheduler, fixed denoise=1.",
            "",
        ]
    )


def overnight_summary(analysis: dict[str, Any], artifact_audit: dict[str, Any]) -> str:
    phase_b = analysis["phase_b_600"]
    route = phase_b["route_summary"]
    return "\n".join(
        [
            "# PV0 Overnight Autonomous Results",
            "",
            "## CURRENT DECISION: PIVOT",
            "",
            "Mechanism GO: PV0 recovers prediction-only latent reuse drift. System-claim PIVOT: it does not yet prove universal Fresh replacement.",
            "",
            f"- S1/S2/S3/S4: all `GO`; S1={artifact_audit['s1_valid_records']}/{artifact_audit['state_index_expected']} states; S4={artifact_audit['s4_valid_records']}/{artifact_audit['s4_expected_anchors']} anchors.",
            "- S5: one preregistered action-outcome scenario, all routes success; preliminary only.",
            f"- Phase-B: complete 600/600 route episodes, 200 paired scenarios across 40 tasks × 5 initial states.",
            f"- Success: Fresh={route['fresh']['successes']}/200, P1={route['predicted_reuse']['successes']}/200, PV0={route['native_persistent']['successes']}/200.",
            f"- Controller accounting: {artifact_audit['job_status_counts']['completed']} completed jobs, {artifact_audit['job_status_counts']['terminal_failure']} terminal infrastructure failure; clean-GPU manual rerun passed.",
            "- See `FINAL_PV0_REPORT_ZH.md` for the evidence-bounded conclusion, `CLOSED_LOOP_PAIRED_ANALYSIS.json` for raw paired analysis, and `thor_export/` for target-hardware contract.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("reports/pv0_overnight"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    run_dir = args.run_dir.resolve()
    outputs = [
        run_dir / "PREFIX_VAE_ARCHITECTURE_AUDIT.json",
        run_dir / "PREFIX_VAE_ARCHITECTURE_AUDIT.md",
        run_dir / "FINAL_PV0_REPORT_ZH.md",
        run_dir / "FINAL_REPRODUCIBILITY_AUDIT.json",
        run_dir / "MORNING_README.md",
        run_dir / "OVERNIGHT_PV0_RESULTS_ZH.md",
        run_dir / "thor_export/THOR_EXPORT_CONTRACT.json",
        run_dir / "thor_export/README.md",
        run_dir / "thor_export/measure_pv0_on_thor.sh",
    ]
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite final deliverables: {existing}")
    phase_a = load_json(run_dir / "PHASE_A_DECISION.json")
    s1 = load_json(run_dir / "S1_PV0_FULL_SCALE_FIDELITY.json")
    s4 = load_json(run_dir / "S4_CONDITION_COMPILE_VALIDATION.json")
    analysis = load_json(run_dir / "CLOSED_LOOP_PAIRED_ANALYSIS.json")
    latency = load_json(run_dir / "clean_latency/preflight_repeats_clean_rerun.json")
    replay = load_json(run_dir / "failure_replays/replay_summary.json")
    artifact_audit = load_json(run_dir / "artifact_audit.json")
    source = source_assertions(repo)
    prefix = prefix_vae_audit(repo, source)
    reproducibility = reproducibility_audit(repo, run_dir, phase_a, analysis, latency, replay, prefix)
    contract, thor_readme, thor_shell = thor_contract(repo, latency, prefix)
    atomic_json(run_dir / "PREFIX_VAE_ARCHITECTURE_AUDIT.json", prefix)
    atomic_text(run_dir / "PREFIX_VAE_ARCHITECTURE_AUDIT.md", prefix_markdown(prefix))
    atomic_json(run_dir / "FINAL_REPRODUCIBILITY_AUDIT.json", reproducibility)
    atomic_json(run_dir / "thor_export/THOR_EXPORT_CONTRACT.json", contract)
    atomic_text(run_dir / "thor_export/README.md", thor_readme)
    shell_path = run_dir / "thor_export/measure_pv0_on_thor.sh"
    atomic_text(shell_path, thor_shell)
    shell_path.chmod(0o755)
    atomic_text(run_dir / "FINAL_PV0_REPORT_ZH.md", final_report(analysis, phase_a, s1, s4, latency, replay, artifact_audit))
    atomic_text(run_dir / "MORNING_README.md", morning_readme(analysis, reproducibility))
    atomic_text(run_dir / "OVERNIGHT_PV0_RESULTS_ZH.md", overnight_summary(analysis, artifact_audit))
    print(
        json.dumps(
            {
                "status": "PASS",
                "final_report": str(run_dir / "FINAL_PV0_REPORT_ZH.md"),
                "reproducibility": str(run_dir / "FINAL_REPRODUCIBILITY_AUDIT.json"),
                "thor_contract": str(run_dir / "thor_export/THOR_EXPORT_CONTRACT.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

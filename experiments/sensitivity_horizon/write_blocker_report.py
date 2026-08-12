#!/usr/bin/env python3
"""Record the E8/E9 protocol blockers without manufacturing invalid evidence.

This is deliberately a reporting-only Phase-0 artifact.  It reserves a new
task-disjoint split but does not execute any simulator/model collection.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch

from experiments.server_deep_validation.pv0_overnight_common import (
    ORIGINAL_CHECKPOINT,
    ORIGINAL_CHECKPOINT_SHA256,
    atomic_write_json,
    read_jsonl,
)


ROOT = Path('reports/sensitivity_horizon')
SOURCE = Path('reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl')
OLD_SPLIT = Path('reports/semantic_risk/TASK_SPLIT.json')


def shell(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def write(name: str, text: str) -> None:
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + '\n', encoding='utf-8')


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(SOURCE)
    old = json.loads(OLD_SPLIT.read_text())
    used = {task for values in old['splits'].values() for task in values}
    all_tasks = sorted({str(row['task_uid']) for row in rows})
    remaining = [task for task in all_tasks if task not in used]
    if len(remaining) != 24:
        raise RuntimeError(f'expected exactly 24 unused tasks, found {len(remaining)}')
    split = {
        'discovery': remaining[:8], 'validation': remaining[8:16], 'heldout': remaining[16:24],
    }
    counts = {task: sum(row['task_uid'] == task for row in rows) for task in remaining}
    split_payload = {
        'schema_version': 1,
        'status': 'RESERVED_NOT_EXECUTED',
        'purpose': 'New task-disjoint E8 confirmation split; no task overlaps semantic-risk E4/E5/E6 formal analysis.',
        'source_state_index': str(SOURCE),
        'prior_formal_tasks_excluded': sorted(used),
        'splits': split,
        'task_counts': {key: len(value) for key, value in split.items()},
        'available_source_records_per_task': counts,
        'warning': 'Source records are only a provenance registry, not legal E8 age-sweep labels. E8 collection was not started.',
    }
    atomic_write_json(ROOT / 'TASK_SPLIT_E8.json', split_payload)
    dirty = shell('git', 'status', '--short')
    manifest = {
        'schema_version': 1,
        'status': 'BLOCKED_PHASE_0',
        'git_sha': shell('git', 'rev-parse', 'HEAD'),
        'branch': shell('git', 'branch', '--show-current'),
        'dirty_status': dirty,
        'checkpoint': str(ORIGINAL_CHECKPOINT),
        'checkpoint_sha256': ORIGINAL_CHECKPOINT_SHA256,
        'denoising_steps': 1,
        'value_used': False,
        'finetuning_used': False,
        'cuda_available': torch.cuda.is_available(),
        'cuda': torch.version.cuda,
        'torch': torch.__version__,
        'gpu_models': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'precision': 'bfloat16 model weights on frozen native Cosmos LIBERO route',
        'simulator': 'LIBERO-PRO OffScreenRenderEnv; exact control timestep UNCONFIRMED by Phase-0 source audit',
        'seeds': 'No E8/E9 collection seed was consumed.',
    }
    atomic_write_json(ROOT / 'RUN_MANIFEST.json', manifest)
    age = {
        'status': 'INVALID_FOR_E8_AGE_SWEEP',
        'action_horizon': 16,
        'source_target_alignment_actions': 16,
        'predicted_visual_condition': 'P1 copies prior generated future visual slots 6/7 into current slots 2/3.',
        'training_alignment': 'LIBERO dataset constructs next_relative_step_idx = relative_step_idx + chunk_size, with chunk_size=16; future visual and future proprio targets are taken at that index.',
        'legal_current_implementation_age': 'Exactly 16 executed control actions between the source F1 request and the target P1/F1 request.',
        'invalid_requested_sweep': 'K != 16 compares the fixed t0+16 predicted visual condition to physical state t0+K. It is a temporal-alignment error, not prediction aging.',
        'control_dt_ms': None,
        'control_dt_status': 'UNKNOWN: video_fps=20 is rendering metadata, and dataset fps=16 is a fixed model-input field; neither proves simulator control dt.',
        'artificial_sleep_used': False,
        'formal_collection_started': False,
    }
    atomic_write_json(ROOT / 'AGE_SEMANTICS.json', age)
    protocol = {
        'phase': 'E8', 'status': 'BLOCKED_BEFORE_SMOKE',
        'primary_signal': 'S_anchor only; no S_candidate leakage is proposed.',
        'requested_design': 'restore identical F1 anchor; execute legal K; paired P1/F1 at target state; model AGE, S, and AGE*S.',
        'failure_mode_triggered': 'wrong temporal alignment',
        'reason': age['invalid_requested_sweep'],
        'allowed_k_under_existing_one-step P1 semantics': [16],
        'consequence': 'One aligned K cannot identify an age slope or AGE*S interaction. E8 cannot be run validly on this implementation without a separately specified multi-horizon WAM output/interface.',
        'no_workaround_used': ['no artificial sleep', 'no K!=16 formal labels', 'no target-time latent interpolation', 'no hidden patch', 'no task-specific threshold'],
    }
    atomic_write_json(ROOT / 'E8_PROTOCOL.json', protocol)
    e8_result = {
        'status': 'INVALID_TEMPORAL_ALIGNMENT', 'formal_samples': 0, 'smoke_status': 'NOT_RUN',
        'primary_hypothesis_tested': False,
        'reason': protocol['reason'],
        'required_go_conditions_evaluable': False,
        'decision': 'SENSITIVITY_HORIZON_NO_GO',
    }
    atomic_write_json(ROOT / 'E8_RESULT.json', e8_result)
    atomic_write_json(ROOT / 'E8_GO_NO_GO.json', {
        'status': 'SENSITIVITY_HORIZON_NO_GO',
        'basis': 'Protocol validity blocker, not a negative empirical AGE*S estimate.',
        'forbidden_next_action': 'Do not claim semantic sensitivity determines prediction expiration rate; do not implement an adaptive horizon controller.',
    })
    e9 = {
        'status': 'NOT_RUN_AFTER_E8_TEMPORAL_BLOCKER',
        'original_runtime_fact': {'p1_ms': 73.6124153137207, 'p1_plus_89_summaries_ms': 78.86584091186523, 'incremental_ms': 5.253425598144531},
        'frozen_score_reconstruction': 'The E4 analysis code, discovery/validation state bank, and raw parquet preserve a deterministic reconstruction path, but the 89-feature standardization means, ridge coefficients, and intercept were not serialized as a standalone immutable artifact.',
        'why_not_run': 'E8 is stopped by the primary temporal-alignment validity blocker. The strict execution order therefore does not authorize a downstream E9 optimization/profile campaign or E10 policy work.',
        'formal_profile_started': False,
        'optimization_started': False,
    }
    atomic_write_json(ROOT / 'E9_OVERHEAD_BREAKDOWN.json', e9)
    atomic_write_json(ROOT / 'E9_RUNTIME_RESULT.json', {**e9, 'decision': 'E9_NOT_RUN'})
    atomic_write_json(ROOT / 'E9_GO_NO_GO.json', {
        'status': 'E9_NOT_RUN',
        'basis': 'Strict E8→E9 sequence stopped at E8 temporal-alignment blocker; no score semantics were changed.',
        'no_feature_deletion_or_refit': True,
    })
    atomic_write_json(ROOT / 'E10_POLICY.json', {
        'status': 'NOT_AUTHORIZED',
        'reason': 'E8 mechanism did not reach a valid GO and E9 cannot prove equivalent deployable implementation.',
        'policy_created': False,
        'closed_loop_run': False,
    })
    final = {
        'status': 'SENSITIVITY_HORIZON_NO_GO',
        'decision_type': 'protocol-validity stop, not empirical rejection of semantic state-risk evidence',
        'e8': e8_result,
        'e9': {'status': 'E9_NOT_RUN', 'reason': e9['why_not_run']},
        'e10': 'NOT_AUTHORIZED',
        'reserved_new_task_split': '8/8/8 across 24 tasks unused by prior formal semantic-risk analysis',
        'next_valid_research_prerequisite': 'A new, explicitly trained or exposed multi-horizon WAM interface in which prediction targets are aligned to each tested K, plus pre-heldout serialization of the exact S score coefficients. This is a new experimental system contract, not a post-hoc alteration of P1.',
    }
    atomic_write_json(ROOT / 'FINAL_DECISION.json', final)
    write('AUDIT.md', f'''# Sensitivity-Horizon Phase-0 audit — blocked

## Confirmed

- Current commit: `{manifest['git_sha']}`; branch `{manifest['branch']}`.
- Original pre-finetune Cosmos checkpoint SHA256: `{ORIGINAL_CHECKPOINT_SHA256}`; denoise=1; value is not used.
- F1 encodes the current camera observation. P1 skips camera/VAE encoding and copies the *previous request's* generated future visual slots 6/7 into current visual slots 2/3.
- The LIBERO policy action chunk is 16×7. Dataset construction takes `next_relative_step_idx = relative_step_idx + chunk_size`; its `future_*` targets therefore correspond to physical state after 16 control actions.
- Existing source/target offline pairs enforce `target.control_step - source.control_step == 16`.
- The prior E4 feature family is 7 post-block captures × 12 passive scalar reductions plus 5 action-geometry scalars = 89 features. The reducer is installed through `model.intermediate_feature_reducer`; it calls `.detach().cpu().tolist()` after the timed model call, so the prior 5.25 ms measurement reflects capture/reduction path but does not isolate host transfer/logging.

## E8 blocker: no legal age sweep

The requested E8 primary experiment needs multiple physically advanced ages K while keeping the prediction target aligned. In this implementation, P1's sole visual prediction is aligned to **t0+16**. At K≠16, P1's visual condition remains t0+16 while the F1 observation is t0+K. That is the protocol's wrong-temporal-alignment failure, not a measurement of prediction validity over age. K=16 alone supplies only one age and cannot estimate an age slope or `AGE×S` interaction.

No simulator stepping, artificial sleep, model inference, smoke, or formal E8 collection was run.

## E9: not run after the E8 stop

`analyze_semantic_risk.py` computes a deterministic ridge fit from the frozen discovery/validation data, but writes neither the fitted mean, scale, coefficients, nor intercept as a standalone artifact. The raw table and fitting code preserve a reconstruction path. No E9 reconstruction, profiling, or optimization was run because strict execution stops at E8's temporal-alignment blocker; no score semantics were changed.

## UNKNOWNS

- Exact simulator control dt: configuration's `video_fps=20` is video rendering metadata; dataset `fps=16` is an input field. Neither establishes control dt.
- A causal multi-horizon future-latent interface/target is not exposed by the current P1 implementation.
- Per-component E4 5.25 ms overhead is not separately profiled, because strict E8→E9 ordering stops this run at E8.
''')
    write('SENSITIVITY_HORIZON_SERVER_REPORT_ZH.md', f'''# Sensitivity-Conditioned Prediction Validity：E8/E9/E10 服务器审计

## 最终决定：SENSITIVITY_HORIZON_NO_GO（协议有效性停止）

这不是对上一轮 E4 semantic state-risk 证据的经验性推翻；而是本轮提出的“语义敏感性决定预测随执行时间失效速度”无法在当前 P1 的时间语义下被合法检验，因此不运行 E8 age sweep、E9 优化或 E10 controller freeze。

## E8：为什么 age sweep 无效

当前 Cosmos-LIBERO P1 的未来视觉 slot 6/7 与 action chunk 都固定预测 `t0+16` 控制动作后的目标；P1 将这些 slot 复制进下一请求的当前视觉 slot 2/3。代码和既有采集都明确 source/target 间隔必须为 16 actions。

所以若从同一 anchor 执行 `K != 16` 个真实动作再比较 P1/F1，P1 仍在表达 `t0+16` 的世界，而 F1 已处于 `t0+K`。此差异混合了**时间错位**，不能解释为 prediction age。唯一对齐的 K=16 只有一个年龄点，无法估计 error-vs-age slope 或 `AGE×S`。

没有以 sleep 伪造年龄，没有采用错位 K 作为标签，也没有进行任何 E8 formal sample。

## 新的 confirmatory 资源

已预留此前 E4/E5/E6 从未使用的剩余 24 个任务，按 8/8/8 task-disjoint split 写入 `TASK_SPLIT_E8.json`。这只是资源冻结，尚未采样。

## E9：为何本轮未运行

先前 89-feature E4 的实测增量为 5.25 ms（2.13% F1）。E4 的确定性代码和 discovery/validation 原始表可以重建该 ridge 分数，尽管历史 artifact 没有单独保存 mean/scale/coef/intercept。本轮不运行重建、profile 或优化：严格的 E8→E9 顺序已在 E8 时间对齐 blocker 处停止。没有删 feature、重训或改变任何分数语义。

## 后续前提

若要恢复该研究问题，需要在新的、预注册的系统合同中提供真正的 multi-horizon WAM output（每个 K 都有对应对齐的预测目标），并在触碰新 held-out 前序列化 E4 score 的全部系数。那是新接口/训练或模型能力研究，不能事后修改当前 P1 来绕过本轮约束。
''')
    print(json.dumps({'status': final['status'], 'remaining_tasks': len(remaining), 'e8_samples': 0}))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Write the frozen E4/E5/E6 decision and the seven requested diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path('reports/semantic_risk')
ART = Path('artifacts/semantic_risk')
PLOTS = ROOT / 'plots'
F1_CLEAN_MS = 246.50540924072266  # frozen clean-GPU baseline in ESP E1_RESULT


def read(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def dump(name: str, payload: dict) -> None:
    (ROOT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def rho(payload: dict, name: str) -> float | None:
    return payload['heldout'][name]['task_balanced_spearman']


def make_plots(frame: pd.DataFrame, e4: dict, e5: dict, e6: dict, cost: dict, e5_cost: dict) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    held = frame.query("split == 'heldout'").copy()
    train = frame.query("split != 'heldout'").copy()
    plt.style.use('seaborn-v0_8-whitegrid')

    def save(name: str) -> None:
        plt.tight_layout(); plt.savefig(PLOTS / name, dpi=180); plt.close()

    plt.figure(figsize=(6, 4));
    plt.scatter(held.visual_innovation_latent, held.action_error_p1_f1, s=12, alpha=.6)
    plt.xlabel('I_latent (fresh F1 vs P1 imagined condition)'); plt.ylabel('D_act (P1 vs F1 action error)')
    plt.title('Figure 1: held-out innovation vs action error'); save('figure1_innovation_vs_error.png')

    plt.figure(figsize=(6, 4));
    plt.hist(frame.causal_sensitivity_target, bins=35, color='#3b82f6', alpha=.85)
    plt.xlabel('S_target = D_act / I_latent'); plt.ylabel('states'); plt.title('Figure 2: sensitivity-target distribution')
    save('figure2_sensitivity_distribution.png')

    plt.figure(figsize=(6, 4));
    plt.scatter(held.causal_sensitivity_target, held.pred_sensitivity, s=12, alpha=.6, color='#16a34a')
    plt.xlabel('true sensitivity target'); plt.ylabel('E4 prediction'); plt.title('Figure 3: E4 held-out prediction')
    save('figure3_e4_predicted_vs_true.png')

    plt.figure(figsize=(6, 4));
    plt.scatter(held.visual_innovation_latent, held.pred_innovation, s=12, alpha=.6, color='#f97316')
    plt.xlabel('true latent innovation'); plt.ylabel('E5 prediction'); plt.title('Figure 4: E5 held-out prediction')
    save('figure4_e5_predicted_vs_true.png')

    e4_names = ['ACTION_ONLY', 'INTERNAL_ONLY', 'INTERNAL_PLUS_ACTION']
    e4_values = [e4['heldout_reporting_only_candidates'][x]['task_balanced_spearman'] for x in e4_names]
    e5_names = ['RAW_ONLY', 'FLOW_ONLY', 'RAW_FLOW_ACTION_RESIDUAL']
    e5_values = [e5['heldout_reporting_only_candidates'][x]['task_balanced_spearman'] for x in e5_names]
    plt.figure(figsize=(8, 4));
    names = ['E4 ' + x.replace('_', '\n') for x in e4_names] + ['E5 ' + x.replace('_', '\n') for x in e5_names]
    vals = e4_values + e5_values
    plt.bar(range(len(vals)), vals, color=['#2563eb']*3+['#ea580c']*3)
    plt.xticks(range(len(vals)), names, fontsize=8); plt.ylabel('task-balanced Spearman')
    plt.title('Figure 5: held-out correlations (reporting-only comparisons)'); save('figure5_heldout_correlations.png')

    plt.figure(figsize=(6, 4))
    for name, values in e6['frontier'].items():
        budgets = sorted(float(k) for k in values)
        plt.plot(budgets, [values[str(x)] for x in budgets], marker='o', label=name.replace('risk_', '').replace('_', ' '))
    plt.xlabel('review budget'); plt.ylabel('task-balanced high-error capture'); plt.ylim(0, 1)
    plt.title('Figure 6: fixed-budget risk frontier'); plt.legend(fontsize=8); save('figure6_risk_frontier.png')

    e4_pct = 100 * cost['median_overhead_ms'] / F1_CLEAN_MS
    e5_pct = 100 * e5_cost['median_probe_cpu_ms'] / F1_CLEAN_MS
    plt.figure(figsize=(6, 3.8));
    labels=['E4: 89 passive summaries', 'E5: 64px CPU raw probe']; values=[e4_pct, e5_pct]
    bars=plt.barh(labels, values, color=['#dc2626','#16a34a']); plt.axvline(1, color='black', ls='--', lw=1, label='E4 <1% gate')
    for bar, value in zip(bars, values): plt.text(value+.04, bar.get_y()+bar.get_height()/2, f'{value:.2f}%', va='center')
    plt.xlabel('incremental latency / clean F1'); plt.title('Figure 7: deployable overhead'); plt.legend(fontsize=8); save('figure7_runtime_overhead.png')


def main() -> None:
    e4, e5, e6, cost = (read(x) for x in ('E4_RESULT.json','E5_RESULT.json','E6_RESULT.json','E4_COST_RESULT.json'))
    raw = pd.read_parquet(ART / 'e4_e5_e6_raw.parquet')
    e6_frame = pd.read_parquet(ART / 'e6_semantic_risk.parquet')
    probe = raw.probe_cpu_ms.to_numpy(dtype=float)
    e5_cost = {
        'status': 'PASS', 'probe': '64px grayscale MAD + gradient difference; raw CPU only; no VAE/decode/model forward',
        'states': int(len(raw)), 'median_probe_cpu_ms': float(np.median(probe)),
        'p05_probe_cpu_ms': float(np.quantile(probe,.05)), 'p95_probe_cpu_ms': float(np.quantile(probe,.95)),
        'reference_clean_f1_ms': F1_CLEAN_MS,
        'median_fraction_of_clean_f1': float(np.median(probe) / F1_CLEAN_MS),
        'formal_decision': 'E5_COST_GO' if np.median(probe) / F1_CLEAN_MS < .05 else 'E5_COST_NO_GO',
    }
    dump('E5_COST_RESULT.json', e5_cost)
    e4_fraction = cost['median_overhead_ms'] / F1_CLEAN_MS
    decision = {
        'status': 'SEMANTIC_RISK_NO_GO',
        'run_scope': 'E4/E5/E6 only; E7 scheduler was not implemented or evaluated',
        'state_bank': {'states': 512, 'tasks': 16, 'split': '6 discovery / 5 validation / 5 heldout', 'per_task': 32},
        'checkpoint_contract': {'checkpoint': cost['checkpoint'], 'sha256': cost['checkpoint_sha256'], 'denoising_steps': 1, 'finetuning_used': False, 'value_used': False},
        'e4': {'offline_heldout_rho': e4['heldout']['task_balanced_spearman'],
               'action_only_reporting_rho': e4['heldout_reporting_only_candidates']['ACTION_ONLY']['task_balanced_spearman'],
               'feature_count': e4['feature_count'], 'incremental_ms': cost['median_overhead_ms'],
               'fraction_of_clean_f1': e4_fraction, 'gate': 'NO_GO: passive 89-feature library costs 2.13% of F1, above strict <1% gate'},
        'e5': {'heldout_rho': e5['heldout']['task_balanced_spearman'], 'incremental_ms': e5_cost['median_probe_cpu_ms'],
               'fraction_of_clean_f1': e5_cost['median_fraction_of_clean_f1'], 'gate': 'WEAK: runtime passes but heldout rho < 0.50 GO threshold'},
        'e6': {'sensitivity_rho': rho(e6, 'risk_s_only'), 'innovation_rho': rho(e6, 'risk_i_only'),
               'additive_rho': rho(e6, 'risk_s_plus_i'), 'product_rho': rho(e6, 'risk_s_times_i'),
               'gate': 'NO_GO: product improves on either component but loses to additive; multiplicative claim unsupported'},
        'age': 'UNIDENTIFIABLE in this frozen fixed-16-action state bank; no superiority claim versus Age is made.',
        'analysis_note': 'One deterministic report regeneration corrected the E6 high-error threshold from train to the pre-registered discovery partition. Candidate selection, fitted model class, features, and heldout decisions were unchanged; no new simulator/model samples were collected.',
        'next_step': 'Stop this branch. Do not implement a scheduler from these results. A future paper direction requires a newly preregistered variable-horizon state bank and a separately designed low-cost internal summary, not post-hoc layer selection here.'
    }
    dump('SEMANTIC_RISK_FINAL_DECISION.json', decision)
    make_plots(e6_frame, e4, e5, e6, cost, e5_cost)
    report = f'''# Semantic Sensitivity × Physical Innovation：E4/E5/E6 服务器验证报告

## 最终结论：SEMANTIC_RISK_NO_GO

本轮不实现 scheduler（E7），也不把结果包装成可部署机制。512 个冻结状态、16 个 LIBERO 任务（6 discovery / 5 validation / 5 heldout，每任务 32 状态）上，E4 的离线语义信号存在，但严格实时门槛失败；E5 的廉价物理创新代理计算足够便宜，却未达到泛化门槛；E6 的乘法组合输给了加法组合。

## 运行与因果约束

- 原始、finetune 前 checkpoint：`{cost['checkpoint']}`；SHA256 `{cost['checkpoint_sha256']}`。
- Cosmos denoise=1；未使用 value；未训练；未使用特权 simulator state 作为模型输入。simulator state 仅用于重放图像观测。
- E4 只读取普通 P1 前向中的 post-block 被动标量摘要；没有额外模型前向、activation patch、attention probe，且没有 F1–P1 delta 进入部署特征。
- E5 仅做 64px CPU 原始帧 MAD / gradient difference / flow；没有 VAE、decode 或重模型。

## E4：语义敏感性

冻结验证选择 `INTERNAL_PLUS_ACTION`（89 个普通 P1 标量）。held-out task-balanced Spearman 为 **{e4['heldout']['task_balanced_spearman']:.3f}**，高于 reporting-only `ACTION_ONLY` 的 **{e4['heldout_reporting_only_candidates']['ACTION_ONLY']['task_balanced_spearman']:.3f}**。因此，内部状态确有超出动作几何的离线预测信息。

但 50 次 clean-GPU CUDA-event 配对测量显示：P1 baseline {cost['p1_no_hook_cuda_ms']['median']:.2f} ms，带全部摘要 {cost['p1_with_e4_summaries_cuda_ms']['median']:.2f} ms，增量 **{cost['median_overhead_ms']:.2f} ms**。相对于冻结的 clean F1 {F1_CLEAN_MS:.2f} ms，是 **{100*e4_fraction:.2f}%**，超过预注册 `<1% F1`。故 E4 是 **offline semantic evidence，但 deployable runtime NO-GO**；不进行事后挑层来挽救该门槛。

## E5：廉价物理创新代理

验证选择 `RAW_ONLY`（frame difference + gradient difference）。held-out task-balanced Spearman 为 **{e5['heldout']['task_balanced_spearman']:.3f}**，低于 `>=0.50` GO 阈值；其中一个 held-out task 为负相关（{e5['heldout']['per_task']['libero_goal:put_the_wine_bottle_on_top_of_the_cabinet']:.3f}）。

它的成本合格：512 状态上 CPU probe 中位 **{e5_cost['median_probe_cpu_ms']:.2f} ms**（p95 {e5_cost['p95_probe_cpu_ms']:.2f} ms），即 clean F1 的 **{100*e5_cost['median_fraction_of_clean_f1']:.2f}%**。成本不能弥补泛化不足，因此标记 **WEAK / 不进入最终方法**。

## E6：组合机制

held-out action-error task-balanced Spearman：只用 sensitivity **{rho(e6,'risk_s_only'):.3f}**，只用 innovation **{rho(e6,'risk_i_only'):.3f}**，加法 **{rho(e6,'risk_s_plus_i'):.3f}**，乘法 **{rho(e6,'risk_s_times_i'):.3f}**。乘法高于两个单项，但低于加法（差 {rho(e6,'risk_s_plus_i')-rho(e6,'risk_s_times_i'):.3f}），不能支撑“语义敏感性 × 物理创新”的乘法机制声明。Age 在全部状态中恒为 16 actions，无法产生排序，因此没有作出对 Age 的优越性声称。

## 图表

![Figure 1](plots/figure1_innovation_vs_error.png)
![Figure 2](plots/figure2_sensitivity_distribution.png)
![Figure 3](plots/figure3_e4_predicted_vs_true.png)
![Figure 4](plots/figure4_e5_predicted_vs_true.png)
![Figure 5](plots/figure5_heldout_correlations.png)
![Figure 6](plots/figure6_risk_frontier.png)
![Figure 7](plots/figure7_runtime_overhead.png)

## 审计说明

初次报告后发现 E6 的 high-error frontier 阈值错误地从 `discovery+validation` 取值；已按预注册定义改为 **discovery-only** 并确定性重生成。该修正没有改变候选、特征、拟合类别、数据或任何采样，亦未基于 held-out 结果再选择方法。完整原始表、state bank、smoke、审计和 JSON 结果均保留在本目录及 `artifacts/semantic_risk/`。
'''
    (ROOT / 'SEMANTIC_RISK_SERVER_REPORT_ZH.md').write_text(report)
    print(json.dumps({'decision': decision['status'], 'e4_rho': decision['e4']['offline_heldout_rho'], 'e5_rho': decision['e5']['heldout_rho'], 'product_rho': decision['e6']['product_rho']}))


if __name__ == '__main__':
    main()

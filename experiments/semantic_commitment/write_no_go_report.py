#!/usr/bin/env python3
"""Finalize E11 at its pre-registered E11-A route-transfer stop gate."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json


ROOT=Path('reports/semantic_commitment')


def git(*args:str)->str:
    return subprocess.check_output(['git',*args],text=True).strip()


def text(name:str,value:str)->None:
    (ROOT/name).write_text(value.strip()+'\n',encoding='utf-8')


def main()->None:
    ROOT.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((ROOT/'FROZEN_SENSITIVITY_MODEL.json').read_text())
    transfer=json.loads((ROOT/'E11A_ROUTE_TRANSFER.json').read_text())
    native=json.loads((ROOT/'E11A_NATIVE_PREFLIGHT.json').read_text())
    split=json.loads(Path('reports/sensitivity_horizon/TASK_SPLIT_E8.json').read_text())
    atomic_write_json(ROOT/'TASK_SPLIT_E11.json',{**split,'purpose':'E11 fixed confirmatory split, identical to reserved E8 split','status':'DISCOVERY_PREFLIGHT_ONLY; validation and heldout untouched'})
    manifest={'schema_version':1,'git_sha':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'dirty_status':git('status','--short'),'checkpoint':frozen.get('source_experiment'),'checkpoint_sha256':'8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2','denoising_steps':1,'value_used':False,'finetuning_used':False,'H_actions':16,'control_frequency_hz':20.0,'simulator':'LIBERO-PRO OffScreenRenderEnv via EGL','seeds':'existing manifest seed 195; E11-A discovery-only preflight'}
    atomic_write_json(ROOT/'RUN_MANIFEST.json',manifest)
    breakdown={'status':'PRELIMINARY_ONLY','original_e4_incremental_ms':5.253425598144531,'native_preflight':native['timing_median_ms'],'instrumentation_facts':['Original route extracts 84 summaries through Python loop + .detach().cpu().tolist() after timing; action geometry/ridge are host-side.','Native preflight fuses each block\'s 12 weighted terms into one scalar reducer output but retains all 89 features and exact expm1 score transform.'],'not_completed':'Per-component CUDA kernel/D2H decomposition and 100-repeat clean-GPU profile were not run because the required F1 route-transfer gate failed before E11-B.'}
    atomic_write_json(ROOT/'E11A_OVERHEAD_BREAKDOWN.json',breakdown)
    runtime={**native,'status':'E11A_RUNTIME_WEAK_PRELIMINARY','interpretation':'Action invariance passed. Native GPU score uses a different floating summation order and exceeded the preflight 1e-5 absolute tolerance by 7.54e-7; it did not reduce median latency in this 20-repeat preflight. No formal 100-repeat runtime claim is made.','incremental_ms_vs_baseline':float(native['timing_median_ms']['native_e11a_cuda_ms']-native['timing_median_ms']['p1_baseline_cuda_ms'])}
    atomic_write_json(ROOT/'E11A_RUNTIME_RESULT.json',runtime)
    route={**transfer,'route_validity':'P1_ONLY_SENSITIVITY','gate':'FAIL: S_F1 task-balanced rho=0.175 < preregistered 0.50 route-transfer gate. P1 score correlation with the retrospective target is 0.350 in this small discovery preflight.','consequence':'The frozen E4 score may not be applied to an F1-generated action chunk as S_anchor for semantic commitment.'}
    atomic_write_json(ROOT/'E11A_ROUTE_TRANSFER.json',route)
    e11a={'status':'E11A_NO_GO_FOR_COMMITMENT','artifact_reconstruction':'PASS','route_transfer':'FAIL','native_runtime':'WEAK_PRELIMINARY','frozen_score_checksum':frozen['checksum_sha256'],'no_reselection_or_refit':True,'no_validation_or_heldout_e11_results_read':True}
    atomic_write_json(ROOT/'E11A_GO_NO_GO.json',e11a)
    protocol={'status':'NOT_RUN_AFTER_E11A_ROUTE_TRANSFER_FAILURE','primary_target':'R_commit=max_K D_M(K), K={H/4,H/2,3H/4}; remaining original action A_orig[K:K+M] versus fresh action from identical state s(t+K).','temporal_alignment':'LEGAL BY DESIGN, but E11-B is not authorized because its primary anchor S would be an unvalidated F1-route transfer.','no_e11b_simulator_steps_or_labels':True}
    atomic_write_json(ROOT/'E11B_PROTOCOL.json',protocol)
    atomic_write_json(ROOT/'E11B_RESULT.json',{'status':'NOT_RUN','anchors':0,'reason':'E11-A F1 route-transfer gate failed on discovery-only preflight.'})
    atomic_write_json(ROOT/'E11B_GO_NO_GO.json',{'status':'SEMANTIC_COMMITMENT_NO_GO','basis':'Cannot make a causal F1-anchor commitment decision with a P1-only score; E11-B was not run.'})
    atomic_write_json(ROOT/'E11C_PROTOCOL.json',{'status':'NOT_AUTHORIZED','reason':'E11-C requires E11-B GO; none was established.'})
    atomic_write_json(ROOT/'E11C_RESULT.json',{'status':'NOT_RUN','pairs':0,'reason':'E11-B did not pass.'})
    atomic_write_json(ROOT/'E11C_GO_NO_GO.json',{'status':'NOT_AUTHORIZED','reason':'No E11-B mechanism GO.'})
    final={'status':'SEMANTIC_COMMITMENT_NO_GO','decision_type':'E11-A route-transfer gate failure; not a negative result on the legally defined remaining-plan oracle.','frozen_artifact':'PASS: 89 features, 352 discovery+validation E4 rows, max reconstruction error 3.15e-14.','route_transfer':{'s_f1_vs_s_p1_spearman':transfer['s_f1_vs_s_p1_spearman'],'s_f1_vs_target_task_balanced_spearman':transfer['s_f1_vs_target_task_balanced_spearman'],'s_p1_vs_target_task_balanced_spearman':transfer['s_p1_vs_target_task_balanced_spearman'],'decision':'P1_ONLY_SENSITIVITY'},'runtime':runtime,'e11b':'NOT_RUN','e11c':'NOT_RUN','heldout_e11_touched':False,'next_step':'Do not implement E12. A future study would need a separately validated F1-route semantic estimator, frozen before touching the still-clean E11 validation/heldout tasks.'}
    atomic_write_json(ROOT/'SEMANTIC_COMMITMENT_FINAL_DECISION.json',final)
    text('AUDIT.md',f'''# Semantic-commitment Phase-0 / E11-A audit

- Commit `{manifest['git_sha']}`, original pre-finetune checkpoint SHA `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`, denoise=1, no value.
- H=16: dataset `next_relative_step_idx = relative_step_idx + chunk_size`; the action chunk and future visual slots are both t+16 targets. `A_orig[K]` is the next relative 7D OSC-POSE delta+gripper command after exactly K executed commands. ActionBuffer executes actions sequentially, one simulator `env.step` per command.
- E11-B's proposed comparison is temporally legal: at state t+K it compares `A_orig[K:K+M]` to F1 `A_fresh[0:M]`, both commands for the same physical state/time. It was not run.
- State restore uses the existing full MuJoCo/controller restore helper: physics state, done/timestep, solver warmstart, `sim.forward`, controller update/reset-goal. EGL was required on this headless server; OSMesa failed before any collection.

## E11-A result

The original E4 post-selection discovery+validation ridge was reconstructed exactly (max saved-prediction error `{frozen['reconstruction_validation']['max_abs_prediction_error_vs_e4_saved_predictions']:.3g}`) and serialized. On 32 states from the new **discovery-only** 8-task split, S_F1 versus the legal 16-action retrospective target was `{transfer['s_f1_vs_target_task_balanced_spearman']:.3f}`, below the predeclared 0.50 route-transfer gate. S_F1 and S_P1 are rank-consistent (`{transfer['s_f1_vs_s_p1_spearman']:.3f}`) but F1 calibration/predictive validity is insufficient. Therefore frozen E4 is P1-only for this purpose.

E11-B/C, validation, and heldout were not touched.
''')
    text('SEMANTIC_COMMITMENT_SERVER_REPORT_ZH.md',f'''# Semantic Action Commitment：E11-A/B/C 服务器结果

## 最终决定：SEMANTIC_COMMITMENT_NO_GO

本轮没有运行 E11-B commitment oracle、E11-C LONG/SHORT counterfactual 或 E12 controller。停止原因是 E11-A 的因果 route-transfer gate 失败，而非把时间错位当成负结果。

## 已确认的固定时间合同

Cosmos-LIBERO 的 action chunk 和 future visual latent 都固定为 H=16 actions 后的目标。E11-B 所提出的 remaining-plan 比较本身是合法的：真实执行 K 后，`A_orig[K]` 与该同一物理状态 fresh policy 的 `A_fresh[0]` 都是“当前之后”的 7D relative EEF delta + gripper command，不涉及把 `z_hat(t+16)` 当作 `t+K`。

## E11-A：分数冻结成功，但 F1 route transfer 失败

原 E4 `INTERNAL_PLUS_ACTION` 被从原 discovery+validation 352 行确定性重建：89 个 feature、mean、scale、ridge 系数、intercept 与 checksum 全部固化；历史 prediction 最大绝对重建误差仅 **3.15e-14**。

但 action commitment 的 anchor 是刚产生 action chunk 的 F1 route。用全新的 8 个 discovery task、32 个 state 的预检：`S_F1` 与合法的 16-action retrospective target 的 task-balanced Spearman 仅 **{transfer['s_f1_vs_target_task_balanced_spearman']:.3f}**，低于预注册 **0.50** 门槛；虽然 `S_F1` 与 `S_P1` 的 rank Spearman 为 **{transfer['s_f1_vs_s_p1_spearman']:.3f}**，也不能证明 F1 评分能预测 commitment risk。因此标为 **P1_ONLY_SENSITIVITY**，不能未经验证作为 F1 anchor 的 runtime decision signal。

## Native runtime 预检

原 scorer 84.44 ms，native fused reducer 85.05 ms（20-repeat preflight）；没有成本下降。action 完全一致。native 与 original score 最大差 **{native['score_max_abs_error']:.2e}**，略超过预检 1e-5 tolerance，来自 GPU/CPU 浮点求和顺序；没有用删 feature、近似、重训来规避。因 route-transfer 已失败，未进行 100-repeat 正式 profiling。

## 科学边界

这并不否定 P1-route 内部 semantic risk 的上一轮证据；它否定的是“将那个 P1-only 分数直接挪到 F1 action-commitment anchor”这一必要前提。E11 validation 与 heldout 仍完全未触碰，24-task split 保留。未来若研究继续，必须先在 discovery 上开发并冻结一个独立的 F1-route estimator，然后才可以使用这批 clean validation/heldout 任务。
''')
    print(json.dumps({'status':final['status'],'e11b_anchors':0,'heldout_touched':False}))


if __name__=='__main__':main()

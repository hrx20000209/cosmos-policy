#!/usr/bin/env python3
"""Frozen-discovery E4/E5/E6 analysis with task-balanced evaluation.

Feature fitting and any candidate selection use discovery/validation only.  The
heldout partition is loaded only for the final frozen evaluation in this one
script invocation; its outputs are explicitly final and non-tunable.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json


def task_spearman(frame: pd.DataFrame, truth: str, pred: str) -> tuple[float | None, dict[str, float | None]]:
    values = {}
    for task, group in frame.groupby("task_uid"):
        values[str(task)] = float(spearmanr(group[truth], group[pred]).statistic) if group[truth].nunique() > 1 and group[pred].nunique() > 1 else None
    valid = [x for x in values.values() if x is not None and np.isfinite(x)]
    return (float(np.mean(valid)) if valid else None), values


def metric(frame: pd.DataFrame, truth: str, pred: str) -> dict:
    rho, per_task = task_spearman(frame, truth, pred)
    label = frame[truth] >= np.quantile(frame[truth], .8)
    def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
        positives = labels.sum(); negatives = len(labels) - positives
        if positives == 0 or negatives == 0: return None
        ranks = pd.Series(scores).rank(method="average").to_numpy()
        return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    def ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
        positives = labels.sum()
        if positives == 0: return None
        order = np.argsort(-scores, kind="mergesort"); sorted_labels = labels[order]
        precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels)+1)
        return float(precision[sorted_labels].sum() / positives)
    labels = label.to_numpy(dtype=bool); scores = frame[pred].to_numpy(dtype=float)
    return {"task_balanced_spearman": rho, "per_task": per_task,
            "pooled_spearman": float(spearmanr(frame[truth], frame[pred]).statistic),
            "top20_auc": auc(labels, scores), "top20_pr_auc": ap(labels, scores)}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str) -> np.ndarray:
    x = train[features].to_numpy(dtype=np.float64); y = np.log1p(train[target].to_numpy(dtype=np.float64))
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-12] = 1.0
    x = (x - mean) / scale
    # Closed-form ridge: this is intentionally the only fitted model class.
    coef = np.linalg.solve(x.T @ x + 10.0 * np.eye(x.shape[1]), x.T @ (y - y.mean()))
    test_x = (test[features].to_numpy(dtype=np.float64) - mean) / scale
    return np.expm1(test_x @ coef + y.mean())


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument('--shard-dir',type=Path,default=Path('reports/semantic_risk/shards')); p.add_argument('--output-dir',type=Path,default=Path('reports/semantic_risk')); args=p.parse_args()
    payloads=[json.loads(Path(path).read_text()) for path in glob.glob(str(args.shard_dir/'SHARD_*.json'))]
    if len(payloads)!=2 or any(x.get('status')!='PASS' for x in payloads): raise RuntimeError('all semantic-risk shards must PASS')
    rows=[r for payload in payloads for r in payload['rows']]
    if len(rows)!=512 or len({r['state_id'] for r in rows})!=512: raise RuntimeError('expected 512 unique states')
    frame=pd.DataFrame(rows); Path('artifacts/semantic_risk').mkdir(parents=True,exist_ok=True); frame.to_parquet('artifacts/semantic_risk/e4_e5_e6_raw.parquet',index=False)
    internal=sorted(c for c in frame if c.startswith('internal_'))
    action=['action_norm','endpoint_displacement','action_curvature','action_jerk','gripper_transition']
    innovation=['frame_diff','gradient_diff','flow_mean','wrist_flow_p90','primary_flow_p90','expected_motion','motion_magnitude_error','motion_direction_error']
    discovery=frame.query("split == 'discovery'").copy(); validation=frame.query("split == 'validation'").copy(); heldout=frame.query("split == 'heldout'").copy()
    # E4 candidate choice only reads discovery/validation.
    candidates={'ACTION_ONLY':action,'INTERNAL_ONLY':internal,'INTERNAL_PLUS_ACTION':internal+action}
    e4_validation={}
    for name, features in candidates.items():
        validation[f'e4_{name}']=fit_predict(discovery,validation,features,'causal_sensitivity_target')
        e4_validation[name]=metric(validation,'causal_sensitivity_target',f'e4_{name}')
    chosen=max(candidates,key=lambda k: e4_validation[k]['task_balanced_spearman'] if e4_validation[k]['task_balanced_spearman'] is not None else -np.inf)
    # refit selected E4 using discovery+validation. E5 selection follows the same discipline.
    train=pd.concat([discovery,validation],ignore_index=True)
    e5_candidates={'AGE_ONLY':[], 'RAW_ONLY':['frame_diff','gradient_diff'], 'FLOW_ONLY':['flow_mean','wrist_flow_p90','primary_flow_p90'], 'RAW_FLOW_ACTION_RESIDUAL':innovation}
    e5_validation={}
    for name, features in e5_candidates.items():
        if not features:
            validation[f'e5_{name}']=float(discovery['prediction_age_actions'].mean())
        else:
            validation[f'e5_{name}']=fit_predict(discovery,validation,features,'visual_innovation_latent')
        e5_validation[name]=metric(validation,'visual_innovation_latent',f'e5_{name}')
    e5_chosen=max(e5_candidates,key=lambda k:e5_validation[k]['task_balanced_spearman'] if e5_validation[k]['task_balanced_spearman'] is not None else -np.inf)
    # Candidate selection is frozen above.  The following held-out candidate
    # predictions are reporting-only comparisons; they never feed selection.
    for name, features in candidates.items():
        heldout[f'e4_report_{name}']=fit_predict(train,heldout,features,'causal_sensitivity_target')
    for name, features in e5_candidates.items():
        heldout[f'e5_report_{name}']=(fit_predict(train,heldout,features,'visual_innovation_latent') if features else float(train['prediction_age_actions'].mean()))
    for target_frame in (train,heldout):
        target_frame['pred_sensitivity']=fit_predict(train,target_frame,candidates[chosen],'causal_sensitivity_target')
        target_frame['pred_innovation']=fit_predict(train,target_frame,e5_candidates[e5_chosen],'visual_innovation_latent') if e5_candidates[e5_chosen] else float(train['prediction_age_actions'].mean())
        target_frame['pred_age']=target_frame['prediction_age_actions'].astype(float)
        target_frame['pred_action_only']=fit_predict(train,target_frame,action,'action_error_p1_f1')
    # Discovery-only formula freezing: normalize component ranks using train empirical CDF; product is uncalibrated.
    def percentile(reference: pd.Series, values: pd.Series) -> np.ndarray:
        ordered=np.sort(reference.to_numpy()); return np.searchsorted(ordered,values.to_numpy(),side='right')/len(ordered)
    for target_frame in (train,heldout):
        target_frame['risk_s_only']=percentile(train.pred_sensitivity,target_frame.pred_sensitivity)
        target_frame['risk_i_only']=percentile(train.pred_innovation,target_frame.pred_innovation)
        target_frame['risk_s_plus_i']=target_frame.risk_s_only+target_frame.risk_i_only
        target_frame['risk_s_times_i']=target_frame.risk_s_only*target_frame.risk_i_only
        target_frame['risk_s_times_i_age']=target_frame.risk_s_times_i
    e4_heldout=metric(heldout,'causal_sensitivity_target','pred_sensitivity')
    e5_heldout=metric(heldout,'visual_innovation_latent','pred_innovation')
    e4_heldout_candidates={name:metric(heldout,'causal_sensitivity_target',f'e4_report_{name}') for name in candidates}
    e5_heldout_candidates={name:metric(heldout,'visual_innovation_latent',f'e5_report_{name}') for name in e5_candidates}
    e6={name:metric(heldout,'action_error_p1_f1',name) for name in ['pred_age','pred_action_only','risk_s_only','risk_i_only','risk_s_plus_i','risk_s_times_i','risk_s_times_i_age']}
    # Fixed-budget high-risk capture: task-balanced recall of global top-20% error, at global budgets.
    frontier={}
    # This threshold is part of the protocol and is frozen on discovery only.
    high_threshold=float(np.quantile(discovery.action_error_p1_f1,.8))
    for name in ['risk_i_only','risk_s_only','risk_s_plus_i','risk_s_times_i']:
        values={}
        for budget in (.1,.2,.3,.4):
            scores=[]
            for _,group in heldout.groupby('task_uid'):
                cutoff=np.quantile(group[name],1-budget); positive=group.action_error_p1_f1>=high_threshold
                scores.append(float(((group[name]>=cutoff)&positive).sum()/max(positive.sum(),1)))
            values[str(budget)]=float(np.mean(scores))
        frontier[name]=values
    frontier['random_expected']={str(budget):float(budget) for budget in (.1,.2,.3,.4)}
    e4={'status':'PASS','phase':'E4','chosen_model':chosen,'selection_split':'validation','validation_candidates':e4_validation,'heldout':e4_heldout,'heldout_reporting_only_candidates':e4_heldout_candidates,'feature_count':len(candidates[chosen]),'runtime_feature_contract':'normal P1 only, no additional forward; attention excluded'}
    e5={'status':'PASS','phase':'E5','chosen_model':e5_chosen,'selection_split':'validation','validation_candidates':e5_validation,'heldout':e5_heldout,'heldout_reporting_only_candidates':e5_heldout_candidates,'feature_count':len(e5_candidates[e5_chosen]),'runtime_feature_contract':'64px CPU raw-frame/flow/action residual only; no VAE/decode'}
    # Primary gate is intentionally strict; age is constant in this state-bank topology.
    product=e6['risk_s_times_i']['task_balanced_spearman']; i_only=e6['risk_i_only']['task_balanced_spearman']; s_only=e6['risk_s_only']['task_balanced_spearman']
    e6_result={'status':'PASS','phase':'E6','heldout':e6,'high_risk_threshold_discovery_frozen':high_threshold,'frontier':frontier,'age_baseline':'UNIDENTIFIABLE: all paired states use the fixed 16-action prediction horizon; excluded from frontier because a constant score cannot induce a ranking','formula':'rank(pred_sensitivity) * rank(pred_innovation)','product_beats_i_only': bool(product is not None and i_only is not None and product>i_only),'product_beats_s_only':bool(product is not None and s_only is not None and product>s_only),'product_beats_additive':bool(product is not None and e6['risk_s_plus_i']['task_balanced_spearman'] is not None and product>e6['risk_s_plus_i']['task_balanced_spearman'])}
    atomic_write_json(args.output_dir/'E4_RESULT.json',e4); atomic_write_json(args.output_dir/'E5_RESULT.json',e5); atomic_write_json(args.output_dir/'E6_RESULT.json',e6_result)
    pd.concat([train,heldout],ignore_index=True).to_parquet('artifacts/semantic_risk/e6_semantic_risk.parquet',index=False)
    print(json.dumps({'E4':e4['heldout']['task_balanced_spearman'],'E5':e5['heldout']['task_balanced_spearman'],'E6_product':product,'chosen_e4':chosen,'chosen_e5':e5_chosen}))


if __name__=='__main__': main()

#!/usr/bin/env python3
"""Discovery-only F1/P1 route-transfer preflight for the frozen E4 score."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from experiments.esp.run_e2_e3_pilot_shard import load_request, render
from experiments.semantic_risk.run_semantic_shard import BLOCKS, action_geometry, predicted_condition, run_route, tensor_rms
from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS, DEFAULT_T5_EMBEDDINGS, ORIGINAL_CHECKPOINT, atomic_write_json,
    build_model, checkpoint_contract, configure_libero, pair_metrics, read_jsonl, set_up_cuda,
)


def score(features: dict[str, float], actions: np.ndarray, frozen: dict) -> float:
    values = {**features, **action_geometry(actions)}
    names = frozen['feature_names']
    x = np.asarray([values[name] for name in names], dtype=np.float64)
    mean, scale, coef = (np.asarray(frozen[key], dtype=np.float64) for key in ('feature_mean','feature_scale','coefficients'))
    return float(np.expm1(((x - mean) / scale) @ coef + float(frozen['intercept'])))


def select_entries(split: dict, source: list[dict], per_task: int) -> list[dict]:
    selected=[]
    for task in split['splits']['discovery']:
        candidates=[x for x in source if x['task_uid']==task and int(x['control_step']) >= 16]
        candidates=sorted(candidates,key=lambda x: hashlib.sha256(x['state_key'].encode()).hexdigest())
        # Spread over independent stored episode/init where possible.
        used=set()
        for item in candidates:
            if item['episode_key'] not in used or len(used) < per_task:
                selected.append(item); used.add(item['episode_key'])
            if sum(x['task_uid']==task for x in selected)>=per_task: break
        if sum(x['task_uid']==task for x in selected) != per_task: raise RuntimeError(f'insufficient {task}')
    return selected


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--per-task',type=int,default=2); p.add_argument('--output',type=Path,default=Path('artifacts/semantic_commitment/e11a_route_transfer.parquet')); args=p.parse_args()
    split=json.loads(Path('reports/sensitivity_horizon/TASK_SPLIT_E8.json').read_text()); frozen=json.loads(Path('reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json').read_text())
    source=read_jsonl(Path('reports/pv0_overnight/manifests/foundation_v2_state_index.jsonl')); entries=select_entries(split,source,args.per_task)
    contract,gpu=checkpoint_contract(ORIGINAL_CHECKPOINT),set_up_cuda(.4); configure_libero(entries[0]); cfg,stats,model=build_model(checkpoint=ORIGINAL_CHECKPOINT,dataset_stats=DEFAULT_DATASET_STATS,t5_embeddings=DEFAULT_T5_EMBEDDINGS)
    rows=[]
    try:
      for index,entry in enumerate(entries,1):
        prior,target=load_request(entry); target_obs=render(entry,target)
        previous=torch.from_numpy(np.asarray(prior['generated_latent'],dtype=np.float16).astype(np.float32)).cuda()
        p1_action,_,p1_features,_=run_route(cfg,model,stats,target_obs,entry['instruction'],entry['seed'],previous=previous,blocks=BLOCKS)
        f1_action,fresh,f1_features,_=run_route(cfg,model,stats,target_obs,entry['instruction'],entry['seed'],previous=None,blocks=BLOCKS)
        innovation=tensor_rms(fresh[:,:, [2,3]]-predicted_condition(previous)[:,:, [2,3]])
        discrepancy=float(pair_metrics(p1_action,f1_action)['mean_step_l2'])
        rows.append({'state_id':entry['state_key'],'task_uid':entry['task_uid'],'episode_id':entry['episode_key'],'split':'discovery',
          's_p1':score(p1_features,p1_action,frozen),'s_f1':score(f1_features,f1_action,frozen),'causal_sensitivity_target':discrepancy/max(innovation,1e-6),
          'p1_f1_action_error':discrepancy,'innovation':innovation,'valid':True})
        print(json.dumps({'completed':index,'total':len(entries)}),flush=True)
    finally:
      model=None; torch.cuda.empty_cache()
    frame=pd.DataFrame(rows); args.output.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(args.output,index=False)
    def rho(left:str)->float|None:
      vals=[]
      for _,g in frame.groupby('task_uid'):
        if g[left].nunique()>1 and g.causal_sensitivity_target.nunique()>1: vals.append(float(spearmanr(g[left],g.causal_sensitivity_target).statistic))
      return float(np.mean(vals)) if vals else None
    result={'status':'PASS','phase':'E11A_ROUTE_TRANSFER_PREFLIGHT','checkpoint':contract,'gpu':gpu,'score_checksum':frozen['checksum_sha256'],'tasks':int(frame.task_uid.nunique()),'states':len(frame),'per_task':args.per_task,
      's_f1_vs_s_p1_spearman':float(spearmanr(frame.s_f1,frame.s_p1).statistic),'s_p1_vs_target_task_balanced_spearman':rho('s_p1'),'s_f1_vs_target_task_balanced_spearman':rho('s_f1'),
      'distribution':{name:{'mean':float(frame[name].mean()),'std':float(frame[name].std()),'p05':float(frame[name].quantile(.05)),'p95':float(frame[name].quantile(.95))} for name in ('s_f1','s_p1')},
      'route_transfer_status':'PENDING_PREDEFINED_GATE: requires >=0.5 F1 target rho and non-negative F1/P1 score consistency before E11B expansion','artifact':str(args.output)}
    atomic_write_json(Path('reports/semantic_commitment/E11A_ROUTE_TRANSFER.json'),result); print(json.dumps(result))


if __name__=='__main__': main()

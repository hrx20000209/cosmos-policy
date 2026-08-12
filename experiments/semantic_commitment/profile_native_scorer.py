#!/usr/bin/env python3
"""E11-A native frozen-score preflight: exact score and paired route timing.

The native reducer computes the same 84 internal ridge terms on GPU at the
seven existing hook locations and returns only seven scalar contributions.
The five action terms are computed from the action array already returned by
the normal policy interface.  No feature is removed or reweighted.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.esp.run_e2_e3_pilot_shard import load_request, render
from experiments.semantic_risk.run_semantic_shard import BLOCKS, SLOTS, action_geometry, run_route
from experiments.server_deep_validation.pv0_overnight_common import (
 DEFAULT_DATASET_STATS, DEFAULT_T5_EMBEDDINGS, ORIGINAL_CHECKPOINT, atomic_write_json, build_model,
 checkpoint_contract, configure_libero, read_jsonl, set_up_cuda,
)


def native_route(cfg: Any, model: Any, stats: dict, obs: Any, instruction: str, seed: int, prev: torch.Tensor, frozen: dict) -> tuple[np.ndarray,float,float]:
 from cosmos_policy.experiments.robot.cosmos_utils import get_action
 names=frozen['feature_names']; mean=np.asarray(frozen['feature_mean']); scale=np.asarray(frozen['feature_scale']); coef=np.asarray(frozen['coefficients'])
 weights=coef/scale; bias=float(frozen['intercept']-np.sum(mean*weights))
 block_weights={}
 for block in BLOCKS:
   values=[]
   for idx in range(12): values.append(float(weights[names.index(f'internal_b{block}_{idx}')]))
   block_weights[block-1]=torch.tensor(values,device='cuda',dtype=torch.float32)
 def reducer(hidden:torch.Tensor,block_index:int)->torch.Tensor:
   pools={slot:hidden[:,slot].mean(dim=(1,2)).float() for slot in SLOTS}
   def rms(slot:int): return torch.sqrt(hidden[:,slot].float().square().mean(dim=(1,2,3)))
   def disp(slot:int): return hidden[:,slot].float().std(dim=(1,2,3))
   def cosine(a:int,b:int): return torch.nn.functional.cosine_similarity(pools[a],pools[b],dim=-1)
   values=torch.stack([rms(4),disp(4),rms(6),rms(7),rms(2),rms(3),cosine(4,6),cosine(4,7),cosine(4,2),cosine(4,3),cosine(2,6),cosine(3,7)],dim=-1)
   return (values*block_weights[block_index]).sum(dim=-1,keepdim=True)
 model.intermediate_feature_ids=[x-1 for x in BLOCKS];model.intermediate_feature_reducer=reducer
 start,end=torch.cuda.Event(True),torch.cuda.Event(True);torch.cuda.synchronize();start.record()
 try:
  result=get_action(cfg,model,stats,{'primary_image':obs.primary_image,'wrist_image':obs.wrist_image,'proprio':obs.proprio},instruction,seed=int(seed),randomize_seed=False,num_denoising_steps_action=1,generate_future_state_and_value_in_parallel=False,decode_future_state=False,skip_vae_encoding=True,previous_generated_latent=prev,skip_camera_preprocessing=True)
  end.record();torch.cuda.synchronize()
  internal=float(sum(x[0,0].item() for x in (model.last_intermediate_features or [])))
  actions=np.asarray(result['actions'],dtype=np.float32)
  geometry=action_geometry(actions); score=bias+internal+sum(weights[names.index(key)]*geometry[key] for key in geometry)
  # E4's serialized score is expm1 of this fitted linear domain.  Applying
  # exactly that transform is part of numerical equivalence, not calibration.
  return actions,float(np.expm1(score)),float(start.elapsed_time(end))
 finally:
  model.intermediate_feature_ids=None;model.intermediate_feature_reducer=None


def main()->None:
 p=argparse.ArgumentParser();p.add_argument('--states',type=int,default=12);p.add_argument('--repeats',type=int,default=20);p.add_argument('--warmup',type=int,default=5);args=p.parse_args()
 if args.repeats<20: raise ValueError('preflight repeats >=20')
 frozen=json.loads(Path('reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json').read_text()); entries=read_jsonl(Path('reports/semantic_risk/SEMANTIC_RISK_STATE_BANK.jsonl'))[:args.states]
 contract,gpu=checkpoint_contract(ORIGINAL_CHECKPOINT),set_up_cuda(.4);configure_libero(entries[0]);cfg,stats,model=build_model(checkpoint=ORIGINAL_CHECKPOINT,dataset_stats=DEFAULT_DATASET_STATS,t5_embeddings=DEFAULT_T5_EMBEDDINGS)
 rows=[]
 try:
  for entry in entries:
   prior,target=load_request(entry);obs=render(entry,target);prev=torch.from_numpy(np.asarray(prior['generated_latent'],dtype=np.float16).astype(np.float32)).cuda()
   original_action,_,features,_=run_route(cfg,model,stats,obs,entry['instruction'],entry['seed'],previous=prev,blocks=BLOCKS)
   native_action,native_score,_=native_route(cfg,model,stats,obs,entry['instruction'],entry['seed'],prev,frozen)
   values={**features,**action_geometry(original_action)}; names=frozen['feature_names'];x=np.asarray([values[k] for k in names]); score=np.expm1(((x-np.asarray(frozen['feature_mean']))/np.asarray(frozen['feature_scale']))@np.asarray(frozen['coefficients'])+float(frozen['intercept']))
   rows.append({'state_id':entry['state_key'],'score_original':float(score),'score_native':native_score,'score_abs_error':abs(float(score)-native_score),'action_max_error':float(np.abs(original_action-native_action).max())})
  entry=entries[0];prior,target=load_request(entry);obs=render(entry,target);prev=torch.from_numpy(np.asarray(prior['generated_latent'],dtype=np.float16).astype(np.float32)).cuda()
  for _ in range(args.warmup): run_route(cfg,model,stats,obs,entry['instruction'],entry['seed'],previous=prev);run_route(cfg,model,stats,obs,entry['instruction'],entry['seed'],previous=prev,blocks=BLOCKS);native_route(cfg,model,stats,obs,entry['instruction'],entry['seed'],prev,frozen)
  timing=[]
  for i in range(args.repeats):
   _,_,_,base=run_route(cfg,model,stats,obs,entry['instruction'],entry['seed'],previous=prev)
   _,_,_,original=run_route(cfg,model,stats,obs,entry['instruction'],entry['seed'],previous=prev,blocks=BLOCKS)
   _,_,native=native_route(cfg,model,stats,obs,entry['instruction'],entry['seed'],prev,frozen)
   timing.append({'repeat_idx':i,'p1_baseline_cuda_ms':base,'original_e4_cuda_ms':original,'native_e11a_cuda_ms':native})
 finally:
  model=None;torch.cuda.empty_cache()
 out=Path('artifacts/semantic_commitment');out.mkdir(parents=True,exist_ok=True);import pandas as pd;pd.DataFrame(timing).to_parquet(out/'e11a_runtime.parquet',index=False)
 r=pd.DataFrame(rows);t=pd.DataFrame(timing); result={'status':'PREFLIGHT_PASS' if r.score_abs_error.max()<1e-5 and r.action_max_error.max()==0 else 'FAIL','score_checksum':frozen['checksum_sha256'],'states':len(r),'repeats':args.repeats,'score_max_abs_error':float(r.score_abs_error.max()),'action_max_error':float(r.action_max_error.max()),'timing_median_ms':{key:float(t[key].median()) for key in t.columns if key!='repeat_idx'},'gpu':gpu,'checkpoint':contract}
 atomic_write_json(Path('reports/semantic_commitment/E11A_NATIVE_PREFLIGHT.json'),result);print(json.dumps(result))
if __name__=='__main__':main()

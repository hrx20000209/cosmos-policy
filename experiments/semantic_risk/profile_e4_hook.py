#!/usr/bin/env python3
"""Clean-GPU CUDA-event profile for passive E4 summaries versus ordinary P1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.semantic_risk.run_semantic_shard import BLOCKS, load_request, render, run_route
from experiments.server_deep_validation.pv0_overnight_common import (
    DEFAULT_DATASET_STATS, DEFAULT_T5_EMBEDDINGS, ORIGINAL_CHECKPOINT, atomic_write_json,
    build_model, checkpoint_contract, configure_libero, read_jsonl, set_up_cuda,
)


def stats(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, dtype=np.float64)
    return {"n":len(a),"mean":float(a.mean()),"median":float(np.median(a)),"p05":float(np.quantile(a,.05)),"p95":float(np.quantile(a,.95))}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--repeats',type=int,default=50); p.add_argument('--warmup',type=int,default=10); p.add_argument('--output',type=Path,default=Path('reports/semantic_risk/E4_COST_RESULT.json')); p.add_argument('--blocks',default=','.join(map(str,BLOCKS))); args=p.parse_args()
    if args.repeats < 50: raise ValueError('formal repeats >=50')
    entry=read_jsonl(Path('reports/semantic_risk/SEMANTIC_RISK_STATE_BANK.jsonl'))[0]; source,_=load_request(entry)
    contract,gpu=checkpoint_contract(ORIGINAL_CHECKPOINT),set_up_cuda(.4); configure_libero(entry); obs=render(entry,source)
    prev=torch.from_numpy(np.asarray(source['generated_latent'],dtype=np.float16).astype(np.float32)).cuda(); cfg,stats_dict,model=build_model(checkpoint=ORIGINAL_CHECKPOINT,dataset_stats=DEFAULT_DATASET_STATS,t5_embeddings=DEFAULT_T5_EMBEDDINGS)
    blocks=tuple(int(x) for x in args.blocks.split(',')); base=[]; hooked=[]
    try:
      for _ in range(args.warmup):
        run_route(cfg,model,stats_dict,obs,entry['instruction'],entry['seed'],previous=prev)
        run_route(cfg,model,stats_dict,obs,entry['instruction'],entry['seed'],previous=prev,blocks=blocks)
      for _ in range(args.repeats):
        _,_,_,value=run_route(cfg,model,stats_dict,obs,entry['instruction'],entry['seed'],previous=prev); base.append(value)
        _,_,_,value=run_route(cfg,model,stats_dict,obs,entry['instruction'],entry['seed'],previous=prev,blocks=blocks); hooked.append(value)
    finally:
      model=None; torch.cuda.empty_cache()
    b,h=stats(base),stats(hooked); result={'status':'PASS',**contract,'gpu':gpu,'denoising_steps':1,'value_used':False,'finetuning_used':False,'additional_model_forward_for_e4':False,'attention_feature_deployable':False,'blocks':blocks,'repeats':args.repeats,'p1_no_hook_cuda_ms':b,'p1_with_e4_summaries_cuda_ms':h,'median_overhead_ms':float(h['median']-b['median']),'median_overhead_fraction_of_p1':float((h['median']-b['median'])/b['median']),'formal_decision':'E4_COST_GO' if (h['median']-b['median'])/b['median']<.01 else 'E4_COST_NO_GO'}
    atomic_write_json(args.output,result); print(json.dumps(result))

if __name__=='__main__': main()

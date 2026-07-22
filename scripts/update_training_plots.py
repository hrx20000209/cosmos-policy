#!/usr/bin/env python3
"""Regenerate required training plots from CSV or JSONL logs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS = {
 "loss_total.png": ["total_loss","train/loss"], "loss_video.png": ["video_loss","train/video_branch_loss","train/video_edm_loss"],
 "loss_action.png": ["action_loss","train/action_loss","train/demo_sample_action_mse_loss"], "loss_components.png": ["video_loss","action_loss","train/video_branch_loss","train/action_loss","train/demo_sample_action_mse_loss","train/demo_sample_future_proprio_mse_loss","train/demo_sample_future_image_mse_loss"],
 "learning_rate.png": ["learning_rate","lr","optim/lr"], "gradient_norm.png": ["gradient_norm","grad_norm","optim/grad_norm"],
 "gpu_memory.png": ["gpu_memory_gb","gpu_memory_allocated_gb","gpu_memory_reserved_gb"],
}
def smooth(y, window=15):
 return pd.Series(y).rolling(window, min_periods=1, center=True).mean().to_numpy()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--log',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 if a.log.suffix=='.csv': df=pd.read_csv(a.log)
 else: df=pd.DataFrame([json.loads(x) for x in a.log.read_text().splitlines() if x.strip()])
 if df.empty: raise ValueError(f'empty log: {a.log}')
 x=df['global_step'] if 'global_step' in df else (df['iteration'] if 'iteration' in df else np.arange(len(df))); step=int(x.iloc[-1] if hasattr(x,'iloc') else x[-1]); epoch=df['epoch'].iloc[-1] if 'epoch' in df else 'n/a'
 a.output_dir.mkdir(parents=True,exist_ok=True)
 for name,candidates in PLOTS.items():
  cols=[c for c in candidates if c in df]
  fig,ax=plt.subplots(figsize=(10,5))
  if cols:
   for c in cols:
    y=pd.to_numeric(df[c],errors='coerce'); ax.plot(x,y,alpha=.25,label=f'{c} raw'); ax.plot(x,smooth(y),label=f'{c} smooth(15)')
  else: ax.text(.5,.5,'metric not present in log',ha='center',va='center',transform=ax.transAxes)
  ax.set(title=f'{name[:-4]} | epoch={epoch} step={step}',xlabel='global step'); ax.grid(alpha=.25); ax.legend(loc='best') if cols else None; fig.tight_layout(); fig.savefig(a.output_dir/name,dpi=160); plt.close(fig)
if __name__=='__main__': main()

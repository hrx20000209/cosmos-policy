"""Reusable episode-safe indexing/masking helpers for LeRobot v3 Cosmos samples."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class EpisodeWindow:
    indices: np.ndarray
    valid_mask: np.ndarray
    future_index: int

class EpisodeWindowIndex:
    """Construct padded chunks that provably never cross episode boundaries."""
    def __init__(self, episode_index: np.ndarray, frame_index: np.ndarray):
        if len(episode_index) != len(frame_index): raise ValueError('episode/frame length mismatch')
        self.bounds={}
        for eid in np.unique(episode_index):
            ids=np.flatnonzero(episode_index==eid)
            if not np.array_equal(frame_index[ids],np.arange(len(ids))): raise ValueError(f'non-contiguous frame_index in episode {eid}')
            if np.any(np.diff(ids)!=1): raise ValueError(f'episode {eid} is not contiguous')
            self.bounds[int(eid)]=(int(ids[0]),int(ids[-1])+1)
        self.episode_index=np.asarray(episode_index)
    def window(self,start:int,horizon:int)->EpisodeWindow:
        if horizon<1: raise ValueError('horizon must be positive')
        eid=int(self.episode_index[start]); begin,end=self.bounds[eid]
        if not begin<=start<end: raise IndexError(start)
        valid=min(horizon,end-start); idx=np.arange(start,start+valid,dtype=np.int64)
        if valid<horizon: idx=np.r_[idx,np.full(horizon-valid,end-1,dtype=np.int64)]
        mask=np.r_[np.ones(valid,dtype=np.float32),np.zeros(horizon-valid,dtype=np.float32)]
        assert np.all(self.episode_index[idx]==eid)
        return EpisodeWindow(idx,mask,int(idx[min(horizon-1,valid-1)]))

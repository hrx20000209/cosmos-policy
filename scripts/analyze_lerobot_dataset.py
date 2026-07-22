#!/usr/bin/env python3
"""Audit a LeRobot v3 dataset without rewriting its parquet/video files."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def _vectors(table, key: str) -> np.ndarray:
    return np.asarray(table[key].to_pylist(), dtype=np.float64)


def _stats(x: np.ndarray, names: list[str]) -> dict:
    return {
        "names": names,
        "min": x.min(0).tolist(), "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
        "nan_count": int(np.isnan(x).sum()), "inf_count": int(np.isinf(x).sum()),
        "max_abs_step": np.abs(np.diff(x, axis=0)).max(0).tolist(),
    }


def analyze(root: Path) -> dict:
    info = json.loads((root / "meta/info.json").read_text())
    tables = [pq.read_table(p) for p in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))]
    table = __import__("pyarrow").concat_tables(tables)
    action = _vectors(table, "action")
    state = _vectors(table, "observation.state")
    episode = np.asarray(table["episode_index"])
    frame = np.asarray(table["frame_index"])
    timestamp = np.asarray(table["timestamp"], dtype=np.float64)
    episode_ids, lengths = np.unique(episode, return_counts=True)
    episode_lengths = {str(int(k)): int(v) for k, v in zip(episode_ids, lengths, strict=True)}
    boundary = np.r_[True, episode[1:] != episode[:-1]]
    per_episode_time_ok = True
    per_episode_frame_ok = True
    for eid in episode_ids:
        idx = np.flatnonzero(episode == eid)
        per_episode_time_ok &= bool(np.all(np.diff(timestamp[idx]) > 0))
        per_episode_frame_ok &= bool(np.array_equal(frame[idx], np.arange(len(idx))))
    cameras = [k for k, v in info["features"].items() if v.get("dtype") in {"video", "image"}]
    media = {}
    for key in cameras:
        files = sorted(glob.glob(str(root / "videos" / key / "**/*.mp4"), recursive=True))
        counts, specs = [], []
        for path in files:
            with av.open(path) as c:
                s = c.streams.video[0]
                counts.append(int(s.frames))
                specs.append({"width": s.width, "height": s.height, "fps": float(s.average_rate), "codec": s.codec_context.name})
        media[key] = {"files": files, "frame_counts": counts, "total_frames": sum(counts), "specs": specs}
    action_names = info["features"]["action"].get("names") or [f"action_{i}" for i in range(action.shape[1])]
    state_names = info["features"]["observation.state"].get("names") or [f"state_{i}" for i in range(state.shape[1])]
    relation = action - state if action.shape == state.shape else None
    result = {
        "dataset_root": str(root.resolve()), "codebase_version": info.get("codebase_version"),
        "fps": info["fps"], "total_episodes": int(len(episode_ids)), "total_frames": int(len(action)),
        "all_episodes_used_for_training": True, "episode_lengths": episode_lengths,
        "episode_length_summary": {"min": int(lengths.min()), "max": int(lengths.max()), "mean": float(lengths.mean()), "std": float(lengths.std())},
        "features": info["features"], "cameras": cameras, "video_media": media,
        "action_dimension": int(action.shape[1]), "state_dimension": int(state.shape[1]),
        "action": _stats(action, action_names), "state": _stats(state, state_names),
        "action_minus_state": None if relation is None else _stats(relation, action_names),
        "action_state_pearson": None if relation is None else [float(np.corrcoef(action[:, i], state[:, i])[0, 1]) for i in range(action.shape[1])],
        "timestamp_strictly_increasing_within_episode": per_episode_time_ok,
        "frame_index_contiguous_from_zero_within_episode": per_episode_frame_ok,
        "index_contiguous_global": bool(np.array_equal(np.asarray(table["index"]), np.arange(len(action)))),
        "suspected_normalized_action": bool(np.all(action >= -1.05) and np.all(action <= 1.05)),
        "suspected_absolute_joint_position": bool(action.shape == state.shape and np.nanmean(np.abs(action-state)) < 20),
        "alignment": {"observation_to_first_action": "row t observation -> row t action", "action_chunk_start": "t", "future_frame": "min(t + horizon, episode_end - 1)", "padding": "repeat last valid row; action_valid_mask excludes repeated tail"},
    }
    return result


def markdown(s: dict) -> str:
    def rows(block):
        return "\n".join(f"| {i} | {n} | {block['min'][i]:.6g} | {block['max'][i]:.6g} | {block['mean'][i]:.6g} | {block['std'][i]:.6g} | {block['max_abs_step'][i]:.6g} |" for i,n in enumerate(block["names"]))
    eps = "\n".join(f"| {k} | {v} |" for k,v in s["episode_lengths"].items())
    cams = "\n".join(f"- `{k}`：{v['total_frames']} 帧；{sorted({(x['width'],x['height'],x['fps'],x['codec']) for x in v['specs']})}" for k,v in s["video_media"].items())
    return f"""# LeRobot 数据分析

数据路径：`{s['dataset_root']}`。完整 snapshot 保持原始 LeRobot v3 结构，未建立 train/validation/test 划分；全部 {s['total_episodes']} 个 episode、{s['total_frames']} 帧参与训练。

## Schema 与媒体

- FPS：{s['fps']}
- action/state 维度：{s['action_dimension']} / {s['state_dimension']}
- 相机键：{', '.join('`'+x+'`' for x in s['cameras'])}
{cams}
- episode 内 timestamp 严格递增：{s['timestamp_strictly_increasing_within_episode']}
- frame_index 从 0 连续：{s['frame_index_contiguous_from_zero_within_episode']}
- global index 连续：{s['index_contiguous_global']}

## Action 统计（原始尺度）

| 维 | 名称 | min | max | mean | std | max abs step |
|---:|---|---:|---:|---:|---:|---:|
{rows(s['action'])}

NaN={s['action']['nan_count']}，Inf={s['action']['inf_count']}。疑似归一化：{s['suspected_normalized_action']}；疑似 absolute joint position：{s['suspected_absolute_joint_position']}。action 与同维 state Pearson：{s['action_state_pearson']}。

## State 统计（原始尺度）

| 维 | 名称 | min | max | mean | std | max abs step |
|---:|---|---:|---:|---:|---:|---:|
{rows(s['state'])}

NaN={s['state']['nan_count']}，Inf={s['state']['inf_count']}。

## 对齐与 episode 尾部

同一 parquet row 的 observation/state/timestamp/frame_index 对应从该 observation 执行的第一步 action。action chunk 从 t 开始；future video/state 对应 `min(t+horizon, episode_end-1)`。尾部不足 horizon 时复制 episode 最后有效值维持张量形状，并由 `action_valid_mask` 排除 padding loss，绝不跨 episode，也不丢弃长度不同的 episode。

episode 长度：min={s['episode_length_summary']['min']}，max={s['episode_length_summary']['max']}，mean={s['episode_length_summary']['mean']:.2f}，std={s['episode_length_summary']['std']:.2f}。

## 每个 episode 帧数

| episode | frames |
|---:|---:|
{eps}
"""


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); a=p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    s=analyze(a.root)
    (a.output_dir/"data_statistics.json").write_text(json.dumps(s,ensure_ascii=False,indent=2)+"\n")
    normalization = {
        "action_key": "action", "state_key": "observation.state", "action_mode": "absolute",
        "action_dim": s["action_dimension"], "proprio_dim": s["state_dimension"],
        "action_names": s["action"]["names"], "state_names": s["state"]["names"],
        "actions_min": s["action"]["min"], "actions_max": s["action"]["max"],
        "actions_mean": s["action"]["mean"], "actions_std": s["action"]["std"],
        "proprio_min": s["state"]["min"], "proprio_max": s["state"]["max"],
        "proprio_mean": s["state"]["mean"], "proprio_std": s["state"]["std"],
        "computed_from": "all 100 training episodes",
    }
    (a.root/"so101_dataset_statistics.json").write_text(json.dumps(normalization,ensure_ascii=False,indent=2)+"\n")
    (a.output_dir/"data_analysis.md").write_text(markdown(s))
    print(json.dumps({k:s[k] for k in ('dataset_root','total_episodes','total_frames','fps','cameras','action_dimension','state_dimension')},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()

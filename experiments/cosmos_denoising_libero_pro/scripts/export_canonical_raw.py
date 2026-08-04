#!/usr/bin/env python3
"""Export deduplicated canonical JSONL views and action-chunk indices."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
EXPERIMENT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    episode_rows = []
    request_rows = []
    for path in OUTPUT.glob("raw/*/episodes.shard*.jsonl"):
        episode_rows.extend(read_jsonl(path))
    for path in OUTPUT.glob("raw/*/requests.shard*.jsonl"):
        request_rows.extend(read_jsonl(path))
    latest = {}
    for row in sorted(episode_rows, key=lambda value: value.get("completed_at_ns", 0)):
        latest[row["episode_key"]] = row
    episodes = sorted(latest.values(), key=lambda row: (row["config_id"], row["manifest_order"]))
    valid_runs = {(row["episode_key"], row["run_id"]) for row in episodes}
    requests = [
        row
        for row in request_rows
        if (row.get("episode_key"), row.get("run_id")) in valid_runs
    ]
    deduplicated = {}
    for row in requests:
        row = dict(row)
        row.setdefault("chunk_index", int(row.get("control_step_id", 0)) // 16)
        row.setdefault("action_ready_timestamp_ns", row.get("inference_finish_ns"))
        row.setdefault("policy_total_ms", row.get("total_policy_request_latency_ms"))
        row.setdefault("DiT_total_ms", row.get("dit_denoising_latency_ms"))
        row.setdefault("preprocess_ms", row.get("preprocessing_latency_ms"))
        row.setdefault("action_decode_postprocess_ms", row.get("action_extraction_latency_ms"))
        row.setdefault("future_decode_ms", row.get("future_state_decode_latency_ms"))
        deduplicated[(row["episode_key"], row["request_id"])] = row
    requests = sorted(
        deduplicated.values(),
        key=lambda row: (row["episode_key"], row["chunk_index"]),
    )

    raw = EXPERIMENT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw / "episodes.jsonl", episodes)
    write_jsonl(raw / "policy_requests.jsonl", requests)
    chunks_dir = raw / "action_chunks"
    chunks_dir.mkdir(exist_ok=True)
    chunk_index = []
    for row in episodes:
        source = Path(row.get("action_chunks_path") or "")
        scope = "full_policy_output"
        if not source.is_file():
            executed = Path(row.get("executed_actions_path") or "")
            if not executed.is_file():
                continue
            actions = np.load(executed, allow_pickle=False)
            source = chunks_dir / f"{row['episode_key']}.executed_prefix_chunks.npz"
            np.savez(
                source,
                **{
                    f"chunk_{offset // 16:04d}": np.ascontiguousarray(
                        actions[offset : offset + 16], dtype=np.float32
                    )
                    for offset in range(0, len(actions), 16)
                },
            )
            scope = "executed_prefix_only"
        else:
            link = chunks_dir / f"{row['episode_key']}.full_policy_chunks.npy"
            if not link.exists():
                os.symlink(source, link)
            source = link
        chunk_index.append(
            {
                "episode_key": row["episode_key"],
                "config_id": row["config_id"],
                "path": str(source),
                "scope": scope,
            }
        )
    write_jsonl(raw / "action_chunks_index.jsonl", chunk_index)
    result = {
        "episodes": len(episodes),
        "policy_requests": len(requests),
        "action_chunk_files": len(chunk_index),
        "canonical_raw": str(raw),
    }
    (EXPERIMENT / "summaries/canonical_raw_export.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

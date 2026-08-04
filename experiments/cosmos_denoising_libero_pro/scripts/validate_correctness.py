#!/usr/bin/env python3
"""Validate the 27-episode correctness smoke and visual artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
EXPERIMENT = Path(__file__).resolve().parents[1]


def rows(pattern: str) -> list[dict]:
    result = []
    for path in OUTPUT.glob(pattern):
        result.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    episodes = rows("raw/*/episodes.shard*.jsonl")
    requests = rows("raw/*/requests.shard*.jsonl")
    episodes = [row for row in episodes if str(row.get("run_id", "")).startswith("correctness-")]
    episode_keys = {row["episode_key"] for row in episodes}
    requests = [row for row in requests if row.get("episode_key") in episode_keys]
    errors = []
    if len(episodes) != 27:
        errors.append(f"expected 27 episodes, got {len(episodes)}")
    for request in requests:
        expected = int(request["selected_denoising_steps"])
        if int(request["denoiser_forward_count"]) != expected:
            errors.append(f"{request['request_id']}: forward mismatch")
        if len(request["per_denoising_step_latency_ms"]) != expected:
            errors.append(f"{request['request_id']}: timing count mismatch")
        if float(request["future_state_decode_latency_ms"]) != 0:
            errors.append(f"{request['request_id']}: future decode nonzero")
        if int(request.get("vae_decode_count", 0)) != 0:
            errors.append(f"{request['request_id']}: VAE decode count nonzero")
        if int(request.get("action_chunk_nan_count", 0)) or int(
            request.get("action_chunk_inf_count", 0)
        ):
            errors.append(f"{request['request_id']}: nonfinite action")
    reset_groups = defaultdict(list)
    for episode in episodes:
        if not episode.get("environment_valid", True):
            errors.append(f"{episode['episode_key']}: environment invalid")
        action_path = Path(episode["executed_actions_path"])
        if not action_path.is_file():
            errors.append(f"{episode['episode_key']}: action trace missing")
        else:
            actions = np.load(action_path, allow_pickle=False)
            if actions.ndim != 2 or actions.shape[1] != 7:
                errors.append(f"{episode['episode_key']}: action trace shape {actions.shape}")
        screenshot = Path(episode["reset_screenshot_path"])
        if not screenshot.is_file():
            errors.append(f"{episode['episode_key']}: reset screenshot missing")
        else:
            reset_groups[
                (
                    episode["domain"],
                    episode["perturbation_category"],
                    episode["task_uid"],
                )
            ].append(sha256(screenshot))
        video = episode.get("video_path")
        if not video or not Path(video).is_file() or Path(video).stat().st_size == 0:
            errors.append(f"{episode['episode_key']}: video missing/empty")
        if episode["domain"] == "libero_pro" and not episode.get("variant_applied", False):
            errors.append(f"{episode['episode_key']}: selected PRO smoke variant is a no-op")
    for key, hashes in reset_groups.items():
        if len(hashes) != 3 or len(set(hashes)) != 1:
            errors.append(f"{key}: reset is not exactly matched across 1/5/6: {hashes}")
    result = {
        "status": "failed" if errors else "passed",
        "episodes": len(episodes),
        "requests": len(requests),
        "successful_episodes": sum(bool(row["success"]) for row in episodes),
        "original_episodes": sum(row["domain"] == "libero" for row in episodes),
        "libero_pro_episodes": sum(row["domain"] == "libero_pro" for row in episodes),
        "matched_reset_groups": len(reset_groups),
        "errors": errors,
    }
    target = EXPERIMENT / "summaries/correctness_validation.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

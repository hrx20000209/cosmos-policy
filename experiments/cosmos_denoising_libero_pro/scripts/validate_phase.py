#!/usr/bin/env python3
"""Validate completeness and invariants for one formal manifest phase."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT = Path("/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep")
EXPERIMENT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    manifest = load_jsonl(args.manifest)
    expected = {row["episode_key"]: row for row in manifest}
    episode_rows = []
    request_rows = []
    for path in OUTPUT.glob("raw/*/episodes.shard*.jsonl"):
        episode_rows.extend(load_jsonl(path))
    for path in OUTPUT.glob("raw/*/requests.shard*.jsonl"):
        request_rows.extend(load_jsonl(path))
    episode_rows = [row for row in episode_rows if row.get("episode_key") in expected]
    request_rows = [row for row in request_rows if row.get("episode_key") in expected]
    latest = {}
    for row in sorted(episode_rows, key=lambda value: value.get("completed_at_ns", 0)):
        latest[row["episode_key"]] = row
    request_rows = [
        row
        for row in request_rows
        if row.get("episode_key") in latest
        and row.get("run_id") == latest[row["episode_key"]].get("run_id")
    ]
    errors = []
    missing = sorted(set(expected) - set(latest))
    if missing:
        errors.append(f"missing episodes: {len(missing)}")
    duplicates = Counter(row["episode_key"] for row in episode_rows)
    duplicate_keys = [key for key, count in duplicates.items() if count > 1]
    # Duplicates are allowed only for explicit retries; latest wins and are
    # reported, never silently multiplied in statistics.
    requests_by_episode = defaultdict(list)
    for row in request_rows:
        requests_by_episode[row["episode_key"]].append(row)
        selected = int(row["selected_denoising_steps"])
        if int(row["denoiser_forward_count"]) != selected:
            errors.append(f"{row['request_id']}: forward mismatch")
        if len(row.get("per_denoising_step_latency_ms", [])) != selected:
            errors.append(f"{row['request_id']}: step timing count mismatch")
        if int(row.get("vae_decode_count", 0)) != 0 or float(
            row.get("future_state_decode_latency_ms", 0)
        ) != 0:
            errors.append(f"{row['request_id']}: future decode was executed")
        if int(row.get("action_chunk_nan_count", 0)) or int(
            row.get("action_chunk_inf_count", 0)
        ):
            errors.append(f"{row['request_id']}: nonfinite action chunk")
    invalid = []
    for key, row in latest.items():
        reason = str(row.get("termination_reason", ""))
        if not row.get("environment_valid", False) or reason.startswith(
            ("validation_error:", "fatal:", "error:")
        ):
            invalid.append(key)
        if int(row.get("request_count", -1)) != len(requests_by_episode[key]):
            errors.append(
                f"{key}: episode request_count={row.get('request_count')} "
                f"raw requests={len(requests_by_episode[key])}"
            )
        action_path = row.get("executed_actions_path")
        if not action_path or not Path(action_path).is_file():
            errors.append(f"{key}: missing action trajectory")
    if invalid:
        errors.append(f"invalid episodes: {len(invalid)}")
    result = {
        "status": "failed" if errors else "passed",
        "phase": args.phase,
        "manifest": str(args.manifest.resolve()),
        "expected_episodes": len(expected),
        "completed_episodes": len(latest),
        "successes": sum(bool(row["success"]) for row in latest.values()),
        "requests": len(request_rows),
        "duplicate_retry_keys": duplicate_keys,
        "invalid_episode_keys": invalid,
        "by_step": {
            str(step): {
                "episodes": sum(int(row["denoising_steps"]) == step for row in latest.values()),
                "successes": sum(
                    int(row["denoising_steps"]) == step and bool(row["success"])
                    for row in latest.values()
                ),
            }
            for step in (1, 2, 3, 4, 5, 6)
        },
        "errors": errors,
    }
    target = EXPERIMENT / f"summaries/phase_{args.phase}_validation.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

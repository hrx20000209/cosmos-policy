"""SO101 action-schema validation shared by training, offline eval, and runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_PATH = Path(__file__).with_name("so101_three_cubes_schema.json")
MOTOR_RESOLUTION = 4095.0


def load_schema(path: str | Path = SCHEMA_PATH) -> dict[str, Any]:
    with Path(path).open() as f:
        schema = json.load(f)
    if schema["action_type"] != "absolute_joint_position":
        raise ValueError("SO101 deployment only supports absolute_joint_position")
    if len(schema["joint_order"]) != 6:
        raise ValueError("SO101 schema must have exactly six ordered action dimensions")
    return schema


def schema_digest(schema: dict[str, Any]) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def calibrated_runtime_limits(schema: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = [], []
    for key in schema["joint_order"]:
        motor = key.removesuffix(".pos")
        if motor == "gripper":
            lo, hi = schema["gripper"]["runtime_range"]
        else:
            calibration = schema["calibration"][motor]
            half_span_deg = (calibration["range_max"] - calibration["range_min"]) * 180.0 / MOTOR_RESOLUTION
            lo, hi = -half_span_deg, half_span_deg
        lower.append(lo)
        upper.append(hi)
    return np.asarray(lower, np.float32), np.asarray(upper, np.float32)


def validate_dataset_metadata(info: dict[str, Any], schema: dict[str, Any]) -> None:
    features = info["features"]
    action = features.get(schema["action_key"])
    state = features.get(schema["state_key"])
    if action is None or state is None:
        raise ValueError(f"Missing action/state keys: {schema['action_key']}, {schema['state_key']}")
    if action["shape"] != [6] or state["shape"] != [6]:
        raise ValueError(f"Expected 6-D action/state, got {action['shape']} and {state['shape']}")
    if action["names"] != schema["joint_order"] or state["names"] != schema["joint_order"]:
        raise ValueError("Dataset action/state joint order differs from the approved SO101 schema")
    camera_keys = sorted(k for k, v in features.items() if v.get("dtype") == "video")
    if camera_keys != sorted(schema["camera_keys"]):
        raise ValueError(f"Camera mismatch: dataset={camera_keys}, schema={schema['camera_keys']}")
    if info["fps"] != schema["fps"]:
        raise ValueError(f"FPS mismatch: dataset={info['fps']}, schema={schema['fps']}")


def validate_runtime_calibration(runtime: dict[str, Any], schema: dict[str, Any]) -> None:
    expected = schema["calibration"]
    if runtime != expected:
        differences = {
            key: {"expected": expected.get(key), "runtime": runtime.get(key)}
            for key in sorted(set(expected) | set(runtime))
            if expected.get(key) != runtime.get(key)
        }
        raise ValueError(f"Runtime calibration does not match training schema: {differences}")


def clip_absolute_action(actions: np.ndarray, schema: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape[-1] != len(schema["joint_order"]):
        raise ValueError(f"Expected action dim 6, got {actions.shape}")
    lower, upper = calibrated_runtime_limits(schema)
    clipped = np.clip(actions, lower, upper)
    return clipped, clipped != actions


def print_schema_summary(schema: dict[str, Any], action_stats: dict[str, Any] | None = None) -> None:
    lower, upper = calibrated_runtime_limits(schema)
    print("=== APPROVED SO101 ACTION SCHEMA ===")
    print(f"action key       : {schema['action_key']}")
    print(f"state key        : {schema['state_key']}")
    print(f"action type      : {schema['action_type']}")
    print(f"action dim       : {len(schema['joint_order'])}")
    print(f"joint order      : {schema['joint_order']}")
    print(f"body unit        : {schema['body_unit']}")
    print(f"runtime limits   : min={lower.tolist()} max={upper.tolist()}")
    print(f"gripper range    : {schema['gripper']['runtime_range']}")
    print(f"gripper direction: {schema['gripper']['direction']}")
    print(f"camera keys      : {schema['camera_keys']}")
    print(f"fps              : {schema['fps']}")
    print(f"schema sha256    : {schema_digest(schema)}")
    if action_stats:
        print(f"dataset action min: {action_stats['min']}")
        print(f"dataset action max: {action_stats['max']}")
        print(f"dataset action mean: {action_stats['mean']}")
        print(f"dataset action std: {action_stats['std']}")

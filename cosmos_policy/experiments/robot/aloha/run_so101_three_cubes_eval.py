"""Safety-gated SO101 client for the Three Cubes Cosmos policy server."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from cosmos_policy.experiments.robot.aloha.so101_schema import (
    clip_absolute_action,
    load_schema,
    print_schema_summary,
    schema_digest,
    validate_runtime_calibration,
)

HERE = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8777/act")
    parser.add_argument("--robot-port", default="/dev/ttyACM1")
    parser.add_argument("--robot-id", default="follower_arm")
    parser.add_argument("--calibration-dir", type=Path, default=HERE / "so101_calibration")
    parser.add_argument("--front-camera", type=int, default=4)
    parser.add_argument("--right-camera", type=int, default=0)
    parser.add_argument("--wrist-camera", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--task", default="go to red cube. take the red cube. go to box. put the red cube in box.")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--max-joint-delta", type=float, default=8.0)
    parser.add_argument("--max-gripper-delta", type=float, default=12.0)
    parser.add_argument("--prefetch-threshold", type=int, default=0)
    parser.add_argument("--handoff-max-delta", type=float, default=8.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--timeline-dir", type=Path, default=Path("/data/rxhuang/cosmos_three_cubes_runs/runtime"))
    parser.add_argument(
        "--execute", action="store_true", help="Actually send targets. Without this flag, observe/query/log only."
    )
    return parser.parse_args()


def query_server(args: argparse.Namespace, observation: dict) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    response = requests.post(args.server, json=observation, timeout=args.request_timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "actions" not in payload:
        raise RuntimeError(f"Invalid policy response: {payload}")
    actions = np.asarray(payload["actions"], np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != 6:
        raise RuntimeError(f"Expected [chunk,6] actions, got {actions.shape}")
    return actions, time.perf_counter() - start


def make_payload(obs: dict, schema: dict, task: str) -> dict:
    state = [float(obs[key]) for key in schema["joint_order"]]
    return {
        "primary_image": np.asarray(obs["front"], np.uint8).tolist(),
        "left_wrist_image": np.asarray(obs["right"], np.uint8).tolist(),
        "right_wrist_image": np.asarray(obs["wrist"], np.uint8).tolist(),
        "proprio": state,
        "task_description": task,
        "action_schema_sha256": schema_digest(schema),
    }


def main() -> None:
    args = parse_args()
    # Imports are deliberately delayed so offline inspection does not require LeRobot hardware dependencies.
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    schema = load_schema()
    calibration_path = args.calibration_dir / f"{args.robot_id}.json"
    with calibration_path.open() as f:
        runtime_calibration = json.load(f)
    validate_runtime_calibration(runtime_calibration, schema)
    print_schema_summary(schema)
    print(f"execution enabled : {args.execute}")

    camera_common = dict(width=640, height=480, fps=int(args.fps), fourcc="MJPG")
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
        use_degrees=True,
        max_relative_target=None,  # This client applies named per-dimension checks before send_action.
        cameras={
            "front": OpenCVCameraConfig(index_or_path=args.front_camera, **camera_common),
            "right": OpenCVCameraConfig(index_or_path=args.right_camera, **camera_common),
            "wrist": OpenCVCameraConfig(index_or_path=args.wrist_camera, **camera_common),
        },
    )
    robot = SO101Follower(config)
    run_dir = args.timeline_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = run_dir / "query_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "timeline.jsonl"
    queue: deque[np.ndarray] = deque()
    executor = ThreadPoolExecutor(max_workers=1)
    pending: Future | None = None
    last_executed: np.ndarray | None = None

    def log_event(event: dict) -> None:
        event["wall_time"] = time.time()
        with log_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def accept_chunk(actions: np.ndarray, latency: float, step: int, event: str) -> None:
        handoff_from = queue[-1] if queue else last_executed
        if handoff_from is not None and np.any(np.abs(actions[0] - handoff_from) > args.handoff_max_delta):
            raise RuntimeError(f"Unsafe chunk handoff: previous={handoff_from.tolist()} next={actions[0].tolist()}")
        queue.extend(actions)
        log_event(
            {
                "event": event,
                "step": step,
                "latency_s": latency,
                "queue_size": len(queue),
                "predicted_actions": actions.tolist(),
            }
        )

    def save_query_images(obs: dict, step: int, suffix: str = "") -> None:
        for camera in ("front", "right", "wrist"):
            Image.fromarray(np.asarray(obs[camera], np.uint8)).save(image_dir / f"{step:06d}{suffix}_{camera}.jpg")

    try:
        robot.connect()
        period = 1.0 / args.fps
        for step in range(args.max_steps):
            tick = time.perf_counter()
            obs = robot.get_observation()
            current = np.asarray([obs[key] for key in schema["joint_order"]], np.float32)

            if pending is not None and pending.done():
                new_actions, latency = pending.result()
                pending = None
                accept_chunk(new_actions, latency, step, "prefetch_result")

            if not queue:
                if pending is not None:
                    actions, latency = pending.result()
                    pending = None
                    accept_chunk(actions, latency, step, "prefetch_wait_result")
                else:
                    save_query_images(obs, step)
                    log_event({"event": "query_start", "step": step, "current_qpos": current.tolist(), "queue_size": 0})
                    actions, latency = query_server(args, make_payload(obs, schema, args.task))
                    accept_chunk(actions, latency, step, "query")

            if args.prefetch_threshold > 0 and len(queue) <= args.prefetch_threshold and pending is None:
                save_query_images(obs, step, "_prefetch")
                log_event(
                    {
                        "event": "prefetch_start",
                        "step": step,
                        "current_qpos": current.tolist(),
                        "queue_size": len(queue),
                    }
                )
                pending = executor.submit(query_server, args, make_payload(obs, schema, args.task))

            requested = queue.popleft()
            target, clipped_mask = clip_absolute_action(requested, schema)
            delta = np.abs(target - current)
            limits = np.asarray([args.max_joint_delta] * 5 + [args.max_gripper_delta], np.float32)
            if np.any(delta > limits):
                raise RuntimeError(
                    "Safety delta exceeded: "
                    f"target={target.tolist()} current={current.tolist()} delta={delta.tolist()} limits={limits.tolist()}"
                )
            command = dict(zip(schema["joint_order"], target.tolist(), strict=True))
            actually_sent = robot.send_action(command) if args.execute else command
            last_executed = np.asarray([actually_sent[key] for key in schema["joint_order"]], np.float32)
            log_event(
                {
                    "event": "control_step",
                    "step": step,
                    "execute": args.execute,
                    "current_qpos": current.tolist(),
                    "requested_action": requested.tolist(),
                    "executed_action": last_executed.tolist(),
                    "calibration_clipped": clipped_mask.tolist(),
                    "queue_size": len(queue),
                }
            )
            remaining = period - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if robot.is_connected:
            robot.disconnect()
        print(f"Timeline: {log_path}")


if __name__ == "__main__":
    main()

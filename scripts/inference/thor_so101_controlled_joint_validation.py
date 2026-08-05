#!/usr/bin/env python3
"""Small, supervised hardware validation for one calibrated SO-101 follower.

This script deliberately does not load or execute a policy.  It moves one
joint at a time by a bounded, fixed amount, observes the result, returns to
the immediately preceding pose, and records the outcome.  It is for checking
the motor bus, calibration and action coordinate semantics before any policy
is considered for physical deployment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def positions(observation: dict[str, float]) -> dict[str, float]:
    return {motor: float(observation[f"{motor}.pos"]) for motor in MOTORS}


def as_action(values: dict[str, float]) -> dict[str, float]:
    return {f"{motor}.pos": float(value) for motor, value in values.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="follower_arm")
    parser.add_argument("--body-step", type=float, default=1.0)
    parser.add_argument("--gripper-step", type=float, default=1.0)
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.body_step <= 1.0 or not 0 < args.gripper_step <= 1.0:
        raise ValueError("This validation permits steps only in (0, 1.0].")
    if not 0.2 <= args.settle_seconds <= 2.0:
        raise ValueError("settle-seconds must be in [0.2, 2.0].")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        cameras={},
        use_degrees=True,
        max_relative_target={motor: 1.0 for motor in MOTORS},
        disable_torque_on_disconnect=True,
    )
    robot = SOFollower(config)
    result: dict[str, object] = {
        "port": args.port,
        "robot_id": args.robot_id,
        "body_step": args.body_step,
        "gripper_step": args.gripper_step,
        "settle_seconds": args.settle_seconds,
        "tests": [],
        "passed": False,
    }
    initial: dict[str, float] | None = None

    try:
        robot.connect()
        initial = positions(robot.get_observation())
        print("initial", initial, flush=True)
        for motor in MOTORS:
            before = positions(robot.get_observation())
            step = args.gripper_step if motor == "gripper" else args.body_step
            target = dict(before)
            target[motor] += step
            sent = robot.send_action(as_action(target))
            sent_values = {name.removesuffix(".pos"): float(value) for name, value in sent.items()}
            time.sleep(args.settle_seconds)
            moved = positions(robot.get_observation())

            # Immediately command the pre-test pose before evaluating a failed
            # move, so every path attempts to leave the arm in its initial state.
            robot.send_action(as_action(before))
            time.sleep(args.settle_seconds)
            returned = positions(robot.get_observation())

            commanded_delta = sent_values[motor] - before[motor]
            observed_delta = moved[motor] - before[motor]
            return_error = returned[motor] - before[motor]
            record = {
                "motor": motor,
                "before": before[motor],
                "sent": sent_values[motor],
                "moved": moved[motor],
                "returned": returned[motor],
                "commanded_delta": commanded_delta,
                "observed_delta": observed_delta,
                "return_error": return_error,
            }
            print("test", json.dumps(record), flush=True)
            if abs(commanded_delta) > 1.001:
                raise RuntimeError(f"{motor}: safety cap failed: {record}")
            if abs(observed_delta) > 2.0:
                raise RuntimeError(f"{motor}: observed motion exceeded 2 degrees: {record}")
            if abs(return_error) > 0.5:
                raise RuntimeError(f"{motor}: did not return within 0.5 degrees: {record}")
            result["tests"].append(record)

        result["passed"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        if robot.is_connected:
            # Make one last bounded attempt to restore the initial pose before
            # releasing torque.  This is safe because every tested movement is
            # at most one degree.
            if initial is not None:
                try:
                    robot.send_action(as_action(initial))
                    time.sleep(args.settle_seconds)
                    result["final_positions"] = positions(robot.get_observation())
                except Exception as error:  # Preserve the original failure.
                    result.setdefault("cleanup_error", repr(error))
            try:
                robot.disconnect()
            except Exception as error:
                result.setdefault("disconnect_error", repr(error))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("result", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

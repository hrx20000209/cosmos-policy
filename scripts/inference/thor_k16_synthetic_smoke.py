#!/usr/bin/env python3
"""Exercise the Thor K=16 Cosmos gRPC server without cameras or robot hardware."""
from __future__ import annotations

import pickle
import time

import grpc
import numpy as np

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from cosmos_policy.experiments.robot.so101_async_deploy_three_cubes_k16 import (
    RemotePolicyConfig,
    TimedObservation,
)

ADDRESS = "127.0.0.1:8082"
TASK = "go to red cube. take the red cube. go to box. put the red cube in box."
JOINTS = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
# Includes shoulder_lift/elbow_flex values outside the training action
# envelope, so the smoke test proves shadow mode is an exact identity mapping
# rather than being changed by a dataset-statistics clip.
PROPRIO = np.asarray([-5.0, -142.0, 138.0, 68.0, 43.0, 25.0], dtype=np.float32)


def main() -> None:
    with grpc.insecure_channel(ADDRESS, options=grpc_channel_options()) as channel:
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        stub.Ready(services_pb2.Empty(), timeout=10)
        setup = RemotePolicyConfig(
            policy_type="cosmos_policy_shadow",
            pretrained_name_or_path="three_cubes_10k_k16",
            lerobot_features={}, actions_per_chunk=16, device="cpu", task=TASK,
        )
        stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(setup)), timeout=10)
        raw = {name: float(value) for name, value in zip(JOINTS, PROPRIO, strict=True)}
        raw.update({
            "front": np.zeros((480, 640, 3), dtype=np.uint8),
            "right": np.zeros((480, 640, 3), dtype=np.uint8),
            "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            "task": TASK,
        })
        observation = TimedObservation(timestamp=time.time(), timestep=0, observation=raw)
        payload = pickle.dumps(observation)
        stub.SendObservations(
            send_bytes_in_chunks(payload, services_pb2.Observation), timeout=30
        )
        response = stub.GetActions(services_pb2.Empty(), timeout=60)
        if not response.data:
            raise RuntimeError("Server returned an empty action payload")
        actions = pickle.loads(response.data)
        values = np.stack([item.get_action().detach().cpu().numpy() for item in actions])
        if values.shape != (16, 6):
            raise RuntimeError(f"Unexpected action shape: {values.shape}")
        if not np.array_equal(values, np.broadcast_to(PROPRIO, (16, 6))):
            raise RuntimeError("Shadow safety violation: returned action differs from proprio")
        print("PASS: K=16 checkpoint completed get_action; shadow output equals proprio.")
        print("server_latency=", actions[0].get_metadata().get("server_latency"))


if __name__ == "__main__":
    main()

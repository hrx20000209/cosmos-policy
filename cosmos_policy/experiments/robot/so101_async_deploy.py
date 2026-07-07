"""SO101 LeRobot async deployment server for Cosmos Policy.

This module intentionally reuses LeRobot's async-inference gRPC protocol and
robot client.  The robot process can stay as the stock
``lerobot.async_inference.robot_client``; this server only replaces the policy
server so that a Cosmos Policy checkpoint can return SO101 action chunks.

Safety note:
    The current full-DiT checkpoint is useful for bench testing, but its offline
    episode curve still shows chunk discontinuities.  Keep the default safety
    clipping enabled for first real-robot tests and also set LeRobot
    ``robot.max_relative_target`` on the client side.
"""

import logging
import pickle  # nosec - LeRobot async protocol uses trusted local pickle.
import sys
import threading
import time
import typing
from concurrent import futures
from dataclasses import asdict, dataclass, field
from queue import Empty, Queue
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from types import ModuleType
from typing import Any

import draccus
import grpc
import numpy as np
import torch

# Current LeRobot main imports ``typing.Unpack``.  Python 3.10 exposes it via
# typing_extensions, while our Cosmos training venv is Python 3.10.
if not hasattr(typing, "Unpack"):
    from typing_extensions import Unpack

    typing.Unpack = Unpack  # type: ignore[attr-defined]

from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import receive_bytes_in_chunks

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)


SO101_ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return logging.getLogger(name)


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_action(self):
        return self.action

    def get_metadata(self):
        return self.metadata


@dataclass
class TimedObservation(TimedData):
    observation: dict[str, Any]
    must_go: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_observation(self):
        return self.observation

    def get_metadata(self):
        return self.metadata


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, Any]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
    task: str = ""


# LeRobot's robot client pickles these classes as
# ``lerobot.async_inference.helpers.*``.  Importing that helpers module also
# imports every LeRobot policy, which is unnecessary here and brittle across
# environments.  Register a tiny protocol-compatible module for pickle instead.
_helpers_module = ModuleType("lerobot.async_inference.helpers")
for _cls in (TimedData, TimedAction, TimedObservation, RemotePolicyConfig):
    _cls.__module__ = "lerobot.async_inference.helpers"
    setattr(_helpers_module, _cls.__name__, _cls)
sys.modules.setdefault("lerobot.async_inference.helpers", _helpers_module)


@dataclass
class SO101CosmosAsyncServerConfig:
    """Config for serving a Cosmos Policy checkpoint through LeRobot async RPC."""

    host: str = "localhost"
    port: int = 8080
    fps: int = 30
    inference_latency: float = 1 / 30
    obs_queue_timeout: float = 2.0

    # Cosmos checkpoint/config.
    ckpt_path: str = (
        "/data/rxhuang/cosmos_action_focused_runs/cosmos_policy/so101_lerobot/"
        "action_focused_B_full_dit_2gpu_smoke_20260707/checkpoints/iter_000010000"
    )
    cosmos_config: str = "cosmos_predict2_2b_480p_so101_lerobot"
    config_file: str = "cosmos_policy/config/config.py"
    dataset_stats_path: str = "/data/rxhuang/three_cubes_1/so101_dataset_statistics.json"
    t5_text_embeddings_path: str = "/data/rxhuang/three_cubes_1/so101_t5_embeddings.pkl"

    # Robot observation camera keys produced by LeRobot SOFollower.get_observation().
    # These names must match the camera names passed to the LeRobot robot client.
    primary_camera_key: str = "front"
    left_wrist_camera_key: str = "right"
    right_wrist_camera_key: str = "wrist"

    # Inference behavior.
    num_denoising_steps_action: int = 10
    seed: int = 195
    randomize_seed: bool = False
    actions_per_chunk: int = 10

    # Safety clamps in physical action units: body joints are degrees, gripper is 0..100.
    # 0 disables the corresponding clamp.
    max_delta_from_observation: float = 8.0
    max_gripper_delta_from_observation: float = 8.0
    max_step_delta: float = 4.0
    max_gripper_step_delta: float = 5.0

    # If true, do everything except return executable actions; useful for camera/schema smoke tests.
    dry_run_zero_actions: bool = False

    @property
    def environment_dt(self) -> float:
        return 1 / self.fps


class SO101CosmosAsyncPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "so101_cosmos_policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: SO101CosmosAsyncServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        self.observation_queue: list[TimedObservation] = []
        self.last_processed_obs: TimedObservation | None = None

        self.policy_specs: RemotePolicyConfig | None = None
        self.actions_per_chunk = config.actions_per_chunk
        self.dataset_stats = load_dataset_stats(config.dataset_stats_path)
        self._validate_and_log_action_schema()

        cfg = SimpleNamespace(
            suite="aloha",  # SO101 uses the same 11-slot three-camera layout as ALOHA.
            config=config.cosmos_config,
            ckpt_path=config.ckpt_path,
            config_file=config.config_file,
            use_third_person_image=True,
            num_third_person_images=1,
            use_wrist_image=True,
            num_wrist_images=2,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            dataset_stats_path=config.dataset_stats_path,
            t5_text_embeddings_path=config.t5_text_embeddings_path,
            trained_with_image_aug=True,
            chunk_size=50,
            num_open_loop_steps=50,
            ar_future_prediction=False,
            ar_value_prediction=False,
            ar_qvalue_prediction=False,
            use_jpeg_compression=False,
            flip_images=False,
            num_denoising_steps_action=config.num_denoising_steps_action,
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
            deterministic=not config.randomize_seed,
            seed=config.seed,
            use_variance_scale=False,
        )
        # SO101OfflineEvalConfig used this dynamic attribute; keep it explicit here.
        cfg.action_dim = 6
        self.cosmos_cfg = cfg

        init_t5_text_embeddings_cache(config.t5_text_embeddings_path)
        self.model, self.model_config = get_model(cfg)
        self.logger.info("Loaded Cosmos SO101 checkpoint: %s", config.ckpt_path)

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def _validate_and_log_action_schema(self) -> None:
        names = list(self.dataset_stats.get("action_names", []))
        state_names = list(self.dataset_stats.get("state_names", []))
        action_dim = int(self.dataset_stats.get("action_dim", len(names)))
        if names != SO101_ACTION_NAMES or state_names != SO101_ACTION_NAMES or action_dim != 6:
            raise RuntimeError(
                "SO101 action schema 不确定，停止启动真机 policy server: "
                f"action_names={names}, state_names={state_names}, action_dim={action_dim}"
            )
        actions_min = self.dataset_stats["actions_min"].astype(float).tolist()
        actions_max = self.dataset_stats["actions_max"].astype(float).tolist()
        self.logger.warning("SO101 runtime action schema verified before deployment")
        self.logger.warning("  action_key='action', action_dim=6, action_mode='absolute'")
        self.logger.warning("  joint_order=%s", names)
        self.logger.warning("  action_range_min=%s", actions_min)
        self.logger.warning("  action_range_max=%s", actions_max)
        self.logger.warning(
            "  gripper=index 5, name=%r, range=[%.6f, %.6f]",
            names[5],
            actions_min[5],
            actions_max[5],
        )

    def Ready(self, request, context):  # noqa: N802
        self.logger.info("Client %s connected and ready", context.peer())
        self.observation_queue.clear()
        self.last_processed_obs = None
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        specs = pickle.loads(request.data)  # nosec
        if not isinstance(specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be RemotePolicyConfig, got {type(specs)}")
        self.policy_specs = specs
        self.actions_per_chunk = min(self.config.actions_per_chunk, specs.actions_per_chunk, 50)
        self.logger.info(
            "Received LeRobot client policy instructions | policy_type=%s | task=%r | "
            "client_actions_per_chunk=%s | server_actions_per_chunk=%s",
            specs.policy_type,
            specs.task,
            specs.actions_per_chunk,
            self.actions_per_chunk,
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, self.logger)
        obs = pickle.loads(received_bytes)  # nosec
        if not isinstance(obs, TimedObservation):
            raise TypeError(f"Expected TimedObservation, got {type(obs)}")
        if len(self.observation_queue) >= 1:
            self.observation_queue.pop(0)
        self.observation_queue.append(obs)
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        if not self.observation_queue:
            time.sleep(min(self.config.obs_queue_timeout, 0.05))
            return services_pb2.Actions(data=b"")
        obs = self.observation_queue.pop(0)
        try:
            start = time.perf_counter()
            actions, timing = self._predict_action_chunk(obs)
            self.logger.info(
                "Generated SO101 action chunk for obs #%s | %d actions | %.1f ms",
                obs.get_timestep(),
                len(actions),
                (time.perf_counter() - start) * 1000,
            )
            if actions:
                actions[0].metadata["server_latency"] = timing
                actions[0].metadata["server_timestamp"] = time.time()
            return services_pb2.Actions(data=pickle.dumps(actions))  # nosec
        except Exception:
            self.logger.exception("Error while generating SO101 action chunk")
            return services_pb2.Actions(data=b"")

    @staticmethod
    def _to_uint8_image(value: Any, key: str) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"Camera {key!r} must be HWC/CHW image, got shape={array.shape}")
        if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
        if array.shape[-1] != 3:
            raise ValueError(f"Camera {key!r} must have 3 channels, got shape={array.shape}")
        if array.dtype == np.uint8:
            return array
        array = array.astype(np.float32)
        if array.size and float(array.min()) >= -1.5 and float(array.max()) <= 1.5 and float(array.min()) < 0:
            array = (array + 1.0) * 127.5
        elif array.size and float(array.max()) <= 1.5:
            array = array * 255.0
        return np.clip(np.rint(array), 0, 255).astype(np.uint8)

    def _extract_proprio(self, raw: dict[str, Any]) -> np.ndarray:
        missing = [name for name in SO101_ACTION_NAMES if name not in raw]
        if missing:
            raise KeyError(f"Robot observation missing SO101 state keys: {missing}; available={sorted(raw)}")
        return np.asarray([raw[name] for name in SO101_ACTION_NAMES], dtype=np.float32)

    def _build_cosmos_observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        camera_keys = [
            self.config.primary_camera_key,
            self.config.left_wrist_camera_key,
            self.config.right_wrist_camera_key,
        ]
        missing = [key for key in camera_keys if key not in raw]
        if missing:
            raise KeyError(f"Robot observation missing camera keys: {missing}; available={sorted(raw)}")
        return {
            "primary_image": self._to_uint8_image(raw[self.config.primary_camera_key], self.config.primary_camera_key),
            "left_wrist_image": self._to_uint8_image(
                raw[self.config.left_wrist_camera_key], self.config.left_wrist_camera_key
            ),
            "right_wrist_image": self._to_uint8_image(
                raw[self.config.right_wrist_camera_key], self.config.right_wrist_camera_key
            ),
            "proprio": self._extract_proprio(raw),
        }

    def _apply_safety_filters(self, actions: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).copy()
        actions = np.clip(actions, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"])

        body_delta = float(self.config.max_delta_from_observation)
        grip_delta = float(self.config.max_gripper_delta_from_observation)
        if body_delta > 0 or grip_delta > 0:
            delta = np.asarray([body_delta] * 5 + [grip_delta], dtype=np.float32)
            disabled = delta <= 0
            lower = proprio - delta
            upper = proprio + delta
            lower[disabled] = -np.inf
            upper[disabled] = np.inf
            actions = np.clip(actions, lower, upper)

        body_step = float(self.config.max_step_delta)
        grip_step = float(self.config.max_gripper_step_delta)
        if body_step > 0 or grip_step > 0:
            step = np.asarray([body_step] * 5 + [grip_step], dtype=np.float32)
            prev = proprio.astype(np.float32)
            for i in range(actions.shape[0]):
                delta = actions[i] - prev
                clipped = np.clip(delta, -step, step)
                # Per-dim disable if requested.
                clipped = np.where(step > 0, clipped, delta)
                actions[i] = prev + clipped
                prev = actions[i]
        return actions

    def _predict_action_chunk(self, observation_t: TimedObservation) -> tuple[list[TimedAction], dict[str, float]]:
        raw = observation_t.get_observation()
        task = str(raw.get("task") or (self.policy_specs.task if self.policy_specs else "") or "")
        if not task:
            raise ValueError("Task instruction is empty; pass --task to the LeRobot robot client.")

        prepare_start = time.perf_counter()
        cosmos_obs = self._build_cosmos_observation(raw)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000

        infer_start = time.perf_counter()
        result = get_action(
            self.cosmos_cfg,
            self.model,
            self.dataset_stats,
            cosmos_obs,
            task,
            seed=self.config.seed + int(observation_t.get_timestep()),
            randomize_seed=self.config.randomize_seed,
            num_denoising_steps_action=self.config.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000

        action_array = np.asarray(result["actions"], dtype=np.float32)
        if action_array.shape != (50, 6):
            raise RuntimeError(f"Cosmos action chunk shape mismatch: expected (50, 6), got {action_array.shape}")
        if self.config.dry_run_zero_actions:
            action_array[:] = cosmos_obs["proprio"][None, :]
        action_array = self._apply_safety_filters(action_array, cosmos_obs["proprio"])
        action_array = action_array[: self.actions_per_chunk]

        metadata = {
            "source_observation_timestep": observation_t.get_timestep(),
            "source_observation_timestamp": observation_t.get_timestamp(),
            "joint_order": SO101_ACTION_NAMES,
            "server_latency": {"prepare_ms": prepare_ms, "cosmos_predict_ms": infer_ms},
        }
        timed_actions = [
            TimedAction(
                timestamp=observation_t.get_timestamp() + i * self.config.environment_dt,
                timestep=observation_t.get_timestep() + i,
                action=torch.from_numpy(action_array[i]).to(torch.float32),
                metadata=metadata if i == 0 else {},
            )
            for i in range(len(action_array))
        ]
        return timed_actions, {"prepare_ms": prepare_ms, "cosmos_predict_ms": infer_ms}

    def stop(self) -> None:
        self.shutdown_event.set()
        self.observation_queue.clear()
        self.logger.info("SO101 Cosmos async policy server stopped")


@draccus.wrap()
def serve(cfg: SO101CosmosAsyncServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = SO101CosmosAsyncPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info("SO101 Cosmos policy server started on %s:%s", cfg.host, cfg.port)
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.logger.info("KeyboardInterrupt received, stopping server...")
    finally:
        policy_server.stop()
        server.stop(grace=0)


if __name__ == "__main__":
    serve()

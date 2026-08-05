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

import contextlib
import io
import json
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

from cosmos_policy.experiments.robot import stage_profiler, truncated_encode
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)


def _stock_load_from_bytes(b):
    """Stock ``torch.storage._load_from_bytes`` (torch/storage.py)."""
    return torch.load(io.BytesIO(b), weights_only=False)


# Present the stock identity so pickle records "torch.storage._load_from_bytes".
_stock_load_from_bytes.__module__ = "torch.storage"
_stock_load_from_bytes.__name__ = "_load_from_bytes"
_stock_load_from_bytes.__qualname__ = "_load_from_bytes"

_pickle_lock = threading.Lock()


@contextlib.contextmanager
def _plain_torch_tensor_pickling():
    """Emit action chunks that a plain-torch environment can unpickle.

    ``megatron.core`` monkeypatches ``torch.storage._load_from_bytes`` at import time
    (``megatron/core/__init__.py``), and Cosmos Policy pulls megatron in transitively.
    ``torch.storage._StorageBase.__reduce__`` looks that name up as a module global at
    call time, so every tensor pickled by this server would otherwise record
    ``megatron.core.safe_globals.safe_load_from_bytes`` as its rebuild function.

    The LeRobot robot client runs in a separate env that has no megatron, so it dies with
    ``ModuleNotFoundError: No module named 'megatron'`` inside ``receive_actions``. The
    client also requires a real ``torch.Tensor`` (it reads ``.get_action().device.type``),
    so sending numpy instead is not an option.

    Restore the stock function for the duration of the dump only, under a lock because the
    gRPC server runs a thread pool. Nothing else in the process is affected, and the
    checkpoint has already been loaded by the time any of this runs.
    """
    with _pickle_lock:
        patched = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = _stock_load_from_bytes
        try:
            yield
        finally:
            torch.storage._load_from_bytes = patched


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

    host: str = "127.0.0.1"
    port: int = 8082
    fps: int = 30
    inference_latency: float = 1 / 30
    obs_queue_timeout: float = 2.0

    # Cosmos checkpoint/config.
    # 10K three-cubes checkpoint trained with K=16.  This must not use the
    # older K=30 SO101 experiment registered in the base deployment script.
    ckpt_path: str = "/home/hrx/Projects/models/three_cubes_1/cosmos_policy/model"
    cosmos_config: str = "cosmos_predict2_2b_three_cubes_full_ft"
    config_file: str = "configs/eval_config.py"
    dataset_stats_path: str = "/home/hrx/Projects/models/three_cubes_1/cosmos_policy/processed_data/dataset_statistics.json"
    t5_text_embeddings_path: str = "/home/hrx/Projects/models/three_cubes_1/cosmos_policy/processed_data/t5_text_embeddings.pkl"

    # Robot observation camera keys produced by LeRobot SOFollower.get_observation().
    # These names must match the camera names passed to the LeRobot robot client.
    primary_camera_key: str = "front"
    left_wrist_camera_key: str = "right"
    right_wrist_camera_key: str = "wrist"

    # View-count ablation.  Dropping a view genuinely shortens the VAE input
    # sequence -- the omitted slot is not zero-padded, its latent index is set
    # to -1 -- so each dropped view removes 2 of 11 latent slots (the current
    # frame and its future placeholder), i.e. ~20% of the encode.
    #
    #   use_wrist_image=True,  num_wrist_images=2 -> front + right + wrist (trained config)
    #   use_wrist_image=True,  num_wrist_images=1 -> front + left_wrist_camera_key
    #   use_wrist_image=False                     -> front only
    #
    # With num_wrist_images=1 the surviving wrist slot is fed by
    # left_wrist_camera_key, so set that to "wrist" to drop the redundant
    # third-person "right" view instead of the wrist-mounted one.
    #
    # The checkpoint was trained with all three views; anything else is a
    # distribution shift and its accuracy is an open question.
    use_wrist_image: bool = True
    num_wrist_images: int = 2

    # Inference behavior.
    num_denoising_steps_action: int = 10
    seed: int = 195
    randomize_seed: bool = False

    # If true, the diffusion seed advances with the observation timestep, so every replan
    # draws a fresh noise realisation. Measured on this checkpoint
    # (cosmos_policy/scripts/ablate_so101_proprio.py), re-sampling the same observation with
    # a different seed moves the predicted chunk by 3.27 deg on average -- that difference
    # is executed as motion and shows up as the arm jittering in place between chunks.
    # Default false: hold the seed fixed so consecutive plans are consistent.
    vary_seed_per_step: bool = False
    actions_per_chunk: int = 1

    # Safety clamps in physical action units: body joints are degrees, gripper is 0..100.
    # 0 disables the corresponding clamp.
    max_delta_from_observation: float = 8.0
    max_gripper_delta_from_observation: float = 8.0
    max_step_delta: float = 4.0
    max_gripper_step_delta: float = 5.0

    # --- action scale ---
    #
    # Measured on the 2026-08-05 deploy300 run: the arm executes at ~1.6 deg/s
    # against ~19.0 deg/s in the demonstrations, i.e. 11.7x slower. Two
    # independent causes, hence two independent knobs:
    #
    #   fps (client + server)  the chunk is 16 steps of *training* time, which
    #                          was 30 fps. Running the client at 8 fps stretches
    #                          the motion 3.75x. The server produces 31.9
    #                          actions/s and only 7.95 are consumed, so there is
    #                          4x headroom -- raising fps costs nothing and also
    #                          shortens the queue (1250 ms -> 417 ms at 24 fps).
    #                          Prefer this: it does not distort the trajectory.
    #
    #   action_gain            even at matched fps the chunk's internal steps are
    #                          3.1x smaller than the demonstrations (0.204 vs
    #                          0.634 deg). Gain amplifies each step away from the
    #                          measured pose: a' = proprio + gain * (a - proprio).
    #                          MEASURED AND REJECTED: gain=2.0 at 24 fps is not
    #                          a 2x scaling, it is positive feedback. The arm
    #                          overshoots the model's target, the next inference
    #                          sees an out-of-distribution pose and its output
    #                          degrades, and the gain amplifies that too --
    #                          103.4 deg/s against a 19.0 deg/s demonstration
    #                          (5.4x, not 2x), command step p95 1.5 -> 19.0 deg
    #                          (12.7x), tracking error max 65 -> 119 deg, and the
    #                          motion shook the wrist camera off the bus 117 s
    #                          in. Leave at 1.0; raise fps instead, which speeds
    #                          the arm up without distorting the trajectory.
    #
    #   action_stride          keep every Nth action, covering the chunk's motion
    #                          in 1/N the steps. Same waypoints, coarser. 1 disables.
    action_gain: float = 1.0
    action_stride: int = 1

    # --- event-triggered inference ---
    #
    # Most chunks carry little new information. Measured on the 20k traces, the
    # model's *incremental intent* (raw action minus the proprio it was
    # conditioned on) changes by only 0.37-1.00 deg between consecutive chunks,
    # so re-anchoring the previous chunk to the current pose is within 2 deg of
    # a fresh inference for 58.8% of chunks at 30 fps and 84.5% at 8 fps.
    #
    # The point is not to save GPU for its own sake. The server is 100% busy, so
    # a genuinely new observation has to queue behind an inference that was
    # going to produce nearly the same answer. Skipping the redundant ones lets
    # the informative ones start immediately.
    #
    # The trigger has to be cheap or it defeats the purpose: both signals are
    # read before the encode, from data already in hand.
    skip_if_static: bool = False
    # Skip only if the arm moved less than this since the last real inference.
    skip_proprio_deg: float = 1.0
    # ...and the cameras changed less than this (mean |diff| on 0-255, subsampled).
    skip_image_diff: float = 2.0
    # Never skip more than this many in a row, so a mis-tuned threshold cannot
    # stall the policy indefinitely.
    skip_max_consecutive: int = 8

    # If true, do everything except return executable actions; useful for camera/schema smoke tests.
    # Never enable hardware motion by default.  The current checkpoint has
    # excessive replanning-boundary jumps in held-out offline evaluation.
    dry_run_zero_actions: bool = True

    # Encode only the latent slots the model actually conditions on and pad the
    # rest with zeros.  The tokenizer is causal, so the conditioning latents are
    # bit-identical; the discarded slots are generated from noise anyway.
    # Measured: 830 -> 502 ms per chunk, max action delta 0.067 deg.
    # Requires future-state/value prediction to stay off.
    truncate_vae_encode: bool = False

    # Drop one conditioning slot from the encode and splice into the latent.
    # -1 disables. Slot 2 = left_wrist = the "right" camera, the third-person
    # view most redundant with "front". Requires truncate_vae_encode.
    # splice_fill: zero | copy_next | copy_last
    splice_drop_slot: int = -1
    splice_fill: str = "zero"

    # Also decode the model's predicted future frames and write them to disk, so
    # a run can be checked against the observation that actually arrived next.
    # This is what makes the checkpoint a *world*-action model rather than just a
    # policy, and it is the part `truncate_vae_encode` cannot coexist with:
    # decoding the future frames replaces latent slots 5 and 6 with their
    # original encoded values (INDICES_TO_REPLACE = [0, 1, 5, 6]), and truncation
    # zeroes exactly those. Enabling this forces truncation off.
    generate_future_state: bool = False
    future_state_dir: str = ""

    # Per-request stage breakdown (VAE encode / DiT denoise / decode / ...).
    # Stage boundaries are CUDA-synchronised, which perturbs the end-to-end
    # number slightly, so it is opt-in.
    profile_stages: bool = False
    # JSONL trace, one line per served chunk.  Empty = derive from timeline dir.
    trace_path: str = ""

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
            use_wrist_image=config.use_wrist_image,
            num_wrist_images=config.num_wrist_images,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            dataset_stats_path=config.dataset_stats_path,
            t5_text_embeddings_path=config.t5_text_embeddings_path,
            trained_with_image_aug=True,
            chunk_size=16,
            num_open_loop_steps=16,
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

        self._future_dir = None
        self._last_infer: dict | None = None
        self._consecutive_skips = 0
        self._n_skipped = 0
        self._n_inferred = 0
        if config.generate_future_state:
            if config.truncate_vae_encode:
                # Truncation zeroes the slots the future decode needs; refuse to
                # produce future frames that would silently be garbage.
                self.logger.warning(
                    "generate_future_state=True forces truncate_vae_encode off "
                    "(the future decode reads latent slots 5-6 that truncation zeroes)."
                )
                config.truncate_vae_encode = False
            self._future_dir = Path(config.future_state_dir or (Path(config.ckpt_path).parent.parent / "future_state"))
            self._future_dir.mkdir(parents=True, exist_ok=True)
            self.logger.warning("Future-state prediction ON; frames -> %s", self._future_dir)

        # Apply before instrumentation so the profiler's vae_encode timer
        # covers the truncated call rather than the original one.
        if config.truncate_vae_encode:
            info = truncated_encode.install(
                self.model, drop_slot=self.config.splice_drop_slot, fill=self.config.splice_fill
            )
            if info.get("applied"):
                self.logger.warning(
                    "Truncated VAE encode ON: encoding %d/%d frames (%.0f%%)%s",
                    info["pixel_frames_encoded"], info["pixel_frames_total"], 100 * info["fraction_encoded"],
                    f"  | dropped slot {info['drop_slot']}, fill={info['fill']}" if info.get("drop_slot", -1) >= 0 else "",
                )
            else:
                self.logger.warning("Truncated VAE encode NOT applied: %s", info.get("reason"))

        self._trace_lock = threading.Lock()
        self._trace_file = None
        if config.profile_stages:
            stage_profiler.instrument(self.model)
            trace_path = config.trace_path or str(
                Path(config.ckpt_path).parent.parent / f"server_stage_trace_{int(time.time())}.jsonl"
            )
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = open(trace_path, "w", buffering=1)  # noqa: SIM115
            self.logger.warning("Stage profiling ON (CUDA-synchronised); trace -> %s", trace_path)

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
            recv_time = time.time()
            actions, timing = self._predict_action_chunk(obs)
            predict_end = time.perf_counter()
            self.logger.info(
                "Generated SO101 action chunk for obs #%s | %d actions | %.1f ms",
                obs.get_timestep(),
                len(actions),
                (predict_end - start) * 1000,
            )
            if actions:
                actions[0].metadata["server_latency"] = timing
                actions[0].metadata["server_timestamp"] = time.time()
            serialize_start = time.perf_counter()
            with _plain_torch_tensor_pickling():
                payload = pickle.dumps(actions)  # nosec
            serialize_ms = (time.perf_counter() - serialize_start) * 1000

            if self._trace_file is not None:
                stages = dict(timing.get("stages_ms") or {})
                stages["serialize"] = serialize_ms
                # Whatever the CUDA-synchronised stages did not account for.
                accounted = sum(v for k, v in stages.items() if k != "dit_calls")
                total_ms = (time.perf_counter() - start) * 1000
                record = {
                    "server_recv_time": recv_time,
                    "server_reply_time": time.time(),
                    "source_observation_timestep": timing.get("source_observation_timestep"),
                    "source_observation_timestamp": timing.get("source_observation_timestamp"),
                    "obs_to_reply_ms": (time.time() - obs.get_timestamp()) * 1000,
                    "total_server_ms": total_ms,
                    "unaccounted_ms": total_ms - accounted,
                    "n_actions": len(actions),
                    "payload_bytes": len(payload),
                    "stages_ms": stages,
                    "proprio": timing.get("proprio"),
                    "raw_model_action_first": timing.get("raw_model_action_first"),
                    "raw_abs_max_delta": timing.get("raw_abs_max_delta"),
                }
                with self._trace_lock:
                    self._trace_file.write(json.dumps(record) + "\n")
            return services_pb2.Actions(data=payload)
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
        # Only demand the views this configuration actually consumes, so a
        # view-ablation run can also stop capturing the dropped camera on the
        # client side (saving its capture cost and USB bandwidth too).
        camera_keys = [self.config.primary_camera_key]
        if self.config.use_wrist_image:
            camera_keys.append(self.config.left_wrist_camera_key)
            if self.config.num_wrist_images == 2:
                camera_keys.append(self.config.right_wrist_camera_key)
        missing = [key for key in camera_keys if key not in raw]
        if missing:
            raise KeyError(f"Robot observation missing camera keys: {missing}; available={sorted(raw)}")

        out: dict[str, Any] = {
            "primary_image": self._to_uint8_image(raw[self.config.primary_camera_key], self.config.primary_camera_key),
            "proprio": self._extract_proprio(raw),
        }
        if self.config.use_wrist_image:
            out["left_wrist_image"] = self._to_uint8_image(
                raw[self.config.left_wrist_camera_key], self.config.left_wrist_camera_key
            )
            # get_action indexes right_wrist_image only when num_wrist_images==2,
            # but the aloha branch builds the list unconditionally, so keep a
            # harmless alias rather than a missing key.
            out["right_wrist_image"] = (
                self._to_uint8_image(raw[self.config.right_wrist_camera_key], self.config.right_wrist_camera_key)
                if self.config.num_wrist_images == 2
                else out["left_wrist_image"]
            )
        else:
            out["left_wrist_image"] = out["primary_image"]
            out["right_wrist_image"] = out["primary_image"]
        return out

    def _apply_safety_filters(self, actions: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).copy()
        # Dataset extrema describe what was observed during training, not the
        # robot's physical limits.  Applying them to a live target can turn an
        # identity command (the current joint position) into motion whenever
        # the arm starts outside the dataset envelope.  Physical safety is
        # enforced below relative to the measured current pose instead.

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

    # Cosmos slot -> the robot camera that feeds it, so a predicted frame can be
    # compared against the right real camera later.
    _FUTURE_KEY_TO_CAMERA = {
        "future_image": "front",        # primary
        "future_wrist_image": "right",  # left_wrist slot
        "future_wrist_image2": "wrist",  # right_wrist slot
    }

    def _save_future_frames(self, result: dict, observation_t: TimedObservation) -> None:
        preds = result.get("future_image_predictions")
        if not preds:
            return
        try:
            from PIL import Image
        except ImportError:
            return
        ts = observation_t.get_timestep()
        wall = observation_t.get_timestamp()
        for key, arr in preds.items():
            cam = self._FUTURE_KEY_TO_CAMERA.get(key, key)
            try:
                a = arr.detach().float().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
                if a.ndim == 4:
                    a = a[0]
                if a.ndim == 3 and a.shape[0] in (1, 3):
                    a = np.moveaxis(a, 0, -1)
                if a.dtype != np.uint8:
                    # The decoder emits [-1, 1]; anything else is already 0-255.
                    a = (a + 1.0) * 127.5 if float(a.min()) < -0.01 else a
                    a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
                Image.fromarray(a).save(self._future_dir / f"t{ts:06d}_{wall:.3f}_{cam}_pred.jpg", quality=90)
            except Exception as exc:  # noqa: BLE001 - never fail a chunk over a debug artefact
                self.logger.debug("future frame save failed for %s: %s", key, exc)

    @staticmethod
    def _thumb(img: np.ndarray) -> np.ndarray:
        """A cheap fingerprint of a frame: every 8th pixel, as float."""
        return img[::8, ::8].astype(np.float32)

    def _should_skip_inference(self, cosmos_obs: dict) -> tuple[bool, dict]:
        """Decide whether this observation is worth a full inference.

        Both signals come from data already in hand and cost well under a
        millisecond, against the ~484 ms the inference would take.
        """
        prev = self._last_infer
        if prev is None:
            return False, {"reason": "no previous inference"}
        if self._consecutive_skips >= self.config.skip_max_consecutive:
            return False, {"reason": "consecutive skip cap"}

        proprio_delta = float(np.max(np.abs(cosmos_obs["proprio"] - prev["proprio"])))
        img_delta = 0.0
        for key in ("primary_image", "left_wrist_image", "right_wrist_image"):
            cur, old = cosmos_obs.get(key), prev["thumbs"].get(key)
            if cur is None or old is None:
                continue
            img_delta = max(img_delta, float(np.abs(self._thumb(cur) - old).mean()))

        info = {"proprio_delta": proprio_delta, "image_delta": img_delta}
        static = proprio_delta < self.config.skip_proprio_deg and img_delta < self.config.skip_image_diff
        info["reason"] = "static" if static else "changed"
        return static, info

    def _predict_action_chunk(self, observation_t: TimedObservation) -> tuple[list[TimedAction], dict[str, float]]:
        raw = observation_t.get_observation()
        task = str(raw.get("task") or (self.policy_specs.task if self.policy_specs else "") or "")
        if not task:
            raise ValueError("Task instruction is empty; pass --task to the LeRobot robot client.")

        profiling = self.config.profile_stages
        if profiling:
            stage_profiler.reset()

        prepare_start = time.perf_counter()
        cosmos_obs = self._build_cosmos_observation(raw)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000
        if profiling:
            stage_profiler.add("obs_prep", prepare_ms)

        skipped, skip_info = (
            self._should_skip_inference(cosmos_obs) if self.config.skip_if_static else (False, {})
        )
        infer_start = time.perf_counter()
        if skipped:
            # Reuse the previous chunk's *incremental intent* -- "where the
            # policy wanted to go relative to wherever it is" -- re-anchored to
            # the pose we are at now. That quantity is what the traces show to
            # be stable between chunks; the absolute targets are not.
            self._consecutive_skips += 1
            self._n_skipped += 1
            model_action_array = (self._last_infer["intent"] + cosmos_obs["proprio"][None, :]).astype(np.float32)
            self.logger.info(
                "SKIPPED inference | obs=%s proprio_delta=%.2f img_delta=%.2f consecutive=%d",
                observation_t.get_timestep(), skip_info.get("proprio_delta", -1.0),
                skip_info.get("image_delta", -1.0), self._consecutive_skips,
            )
        else:
            self._consecutive_skips = 0
            self._n_inferred += 1
            result = get_action(
                self.cosmos_cfg,
                self.model,
                self.dataset_stats,
                cosmos_obs,
                task,
                seed=(self.config.seed + int(observation_t.get_timestep())
                      if self.config.vary_seed_per_step else self.config.seed),
                randomize_seed=self.config.randomize_seed,
                num_denoising_steps_action=self.config.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=self.config.generate_future_state,
            )
            if self._future_dir is not None:
                self._save_future_frames(result, observation_t)
            model_action_array = np.asarray(result["actions"], dtype=np.float32)
        infer_ms = (time.perf_counter() - infer_start) * 1000

        if not skipped:
            # Remember what this inference wanted, for the next skip to reuse.
            self._last_infer = {
                "proprio": cosmos_obs["proprio"].copy(),
                "intent": model_action_array - cosmos_obs["proprio"][None, :],
                "thumbs": {
                    k: self._thumb(cosmos_obs[k])
                    for k in ("primary_image", "left_wrist_image", "right_wrist_image")
                    if k in cosmos_obs
                },
            }

        expected_shape = (self.cosmos_cfg.chunk_size, self.cosmos_cfg.action_dim)
        if model_action_array.shape != expected_shape:
            raise RuntimeError(
                f"Cosmos action chunk shape mismatch: expected {expected_shape}, got {model_action_array.shape}"
            )
        # Always evaluate and record the learned policy against live robot
        # observations.  This makes an HIL shadow run useful for judging whether
        # the checkpoint is safe enough to promote, and it keeps the raw-vs-
        # bounded comparison available when the safety clamps are what actually
        # gets executed.
        safety_start = time.perf_counter()
        bounded_candidate = self._apply_safety_filters(model_action_array, cosmos_obs["proprio"])
        if profiling:
            stage_profiler.add("safety_filter", (time.perf_counter() - safety_start) * 1000)
        raw_delta = model_action_array - cosmos_obs["proprio"][None, :]
        bounded_delta = bounded_candidate - cosmos_obs["proprio"][None, :]
        self.logger.warning(
            "SHADOW candidate | obs=%s | dry_run=%s | raw_first=%s | raw_abs_max=%.3f | "
            "bounded_first=%s | bounded_abs_max=%.3f",
            observation_t.get_timestep(),
            self.config.dry_run_zero_actions,
            np.array2string(raw_delta[0], precision=2, suppress_small=True),
            float(np.max(np.abs(raw_delta))),
            np.array2string(bounded_delta[0], precision=2, suppress_small=True),
            float(np.max(np.abs(bounded_delta))),
        )
        if self.config.dry_run_zero_actions:
            # A hardware shadow run must remain an exact identity mapping.
            # Do not pass it through dataset/relative safety filters: those
            # filters are intended for model targets and may alter an identity
            # target if the current pose lies outside training statistics.
            action_array = np.broadcast_to(cosmos_obs["proprio"], model_action_array.shape).copy()
        else:
            action_array = bounded_candidate
        # Amplify/subsample before trimming, so the trim still yields the
        # requested number of executable actions.
        if self.config.action_stride > 1:
            action_array = action_array[:: self.config.action_stride]
        if self.config.action_gain != 1.0:
            # Anchor on the measured pose, not on the chunk's own first step, so
            # the gain scales the *intended displacement* rather than compounding
            # whatever offset the chunk starts with.
            anchor = cosmos_obs["proprio"][None, :]
            action_array = anchor + self.config.action_gain * (action_array - anchor)
        action_array = action_array[: self.actions_per_chunk]

        metadata = {
            "source_observation_timestep": observation_t.get_timestep(),
            "source_observation_timestamp": observation_t.get_timestamp(),
            "joint_order": SO101_ACTION_NAMES,
            "server_latency": {"prepare_ms": prepare_ms, "cosmos_predict_ms": infer_ms},
            # Raw (unclamped) model output and the live proprio it was conditioned
            # on, so the shadow client can audit action semantics end-to-end.
            "proprio": cosmos_obs["proprio"].astype(float).tolist(),
            "raw_model_action_first": model_action_array[0].astype(float).tolist(),
            "raw_model_action_last": model_action_array[-1].astype(float).tolist(),
            "raw_abs_max_delta": float(np.max(np.abs(raw_delta))),
            "dry_run_zero_actions": bool(self.config.dry_run_zero_actions),
            "num_denoising_steps_action": int(self.config.num_denoising_steps_action),
            "inference_skipped": bool(skipped),
        }
        stages = stage_profiler.collect() if profiling else {}
        if stages:
            metadata["stages_ms"] = stages
        timed_actions = [
            TimedAction(
                timestamp=observation_t.get_timestamp() + i * self.config.environment_dt,
                timestep=observation_t.get_timestep() + i,
                action=torch.from_numpy(action_array[i]).to(torch.float32),
                # Same dict object for every action in the chunk: the client's
                # aggregation keeps whichever action wins per timestep, so
                # head-only metadata is usually discarded and the shadow record
                # loses its audit trail.  Pickle memoises the shared reference,
                # so attaching it to all K actions costs nothing on the wire.
                metadata=metadata,
            )
            for i in range(len(action_array))
        ]
        return timed_actions, {
            "prepare_ms": prepare_ms,
            "cosmos_predict_ms": infer_ms,
            "stages_ms": stages,
            "source_observation_timestep": observation_t.get_timestep(),
            "source_observation_timestamp": observation_t.get_timestamp(),
            "proprio": cosmos_obs["proprio"].astype(float).tolist(),
            "raw_model_action_first": model_action_array[0].astype(float).tolist(),
            "raw_abs_max_delta": float(np.max(np.abs(raw_delta))),
        }

    def stop(self) -> None:
        self.shutdown_event.set()
        self.observation_queue.clear()
        if self.config.skip_if_static:
            total = self._n_skipped + self._n_inferred
            if total:
                self.logger.warning(
                    "Event-triggered inference: %d/%d chunks skipped (%.1f%%), %d inferences run",
                    self._n_skipped, total, 100 * self._n_skipped / total, self._n_inferred,
                )
        if self._trace_file is not None:
            with self._trace_lock:
                self._trace_file.close()
                self._trace_file = None
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

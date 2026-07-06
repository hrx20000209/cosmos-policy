"""Minimal LeRobot v3 adapter for the Three Cubes SO101 dataset."""

from __future__ import annotations

import bisect
import glob
import json
import pickle
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.experiments.robot.aloha.so101_schema import (
    load_schema,
    print_schema_summary,
    validate_dataset_metadata,
)


class _LeRobotVideoStore:
    """Read exact global frame indices across LeRobot's size-split MP4 files."""

    def __init__(self, root: Path, camera_keys: list[str]):
        self.root = root
        self.files: dict[str, list[str]] = {}
        self.ends: dict[str, list[int]] = {}
        self._containers: dict[tuple[str, int], av.container.InputContainer] = {}
        for feature_key in camera_keys:
            camera = feature_key.removeprefix("observation.images.")
            files = sorted(glob.glob(str(root / "videos" / feature_key / "**/*.mp4"), recursive=True))
            if not files:
                raise FileNotFoundError(f"No videos for {feature_key}")
            counts = []
            for path in files:
                with av.open(path) as container:
                    counts.append(int(container.streams.video[0].frames))
            self.files[camera] = files
            self.ends[camera] = np.cumsum(counts).tolist()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_containers"] = {}
        return state

    def read(self, camera: str, global_index: int) -> np.ndarray:
        file_index = bisect.bisect_right(self.ends[camera], global_index)
        start = 0 if file_index == 0 else self.ends[camera][file_index - 1]
        local_index = global_index - start
        key = (camera, file_index)
        container = self._containers.get(key)
        if container is None:
            container = av.open(self.files[camera][file_index])
            self._containers[key] = container
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        target_pts = int((local_index / fps) / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True)
        for frame in container.decode(stream):
            decoded_index = round(float(frame.pts * stream.time_base) * fps)
            if decoded_index >= local_index:
                if decoded_index != local_index:
                    raise RuntimeError(
                        f"Video seek skipped {camera} frame {global_index}: decoded local {decoded_index}"
                    )
                return frame.to_ndarray(format="rgb24")
        raise RuntimeError(f"Failed reading {camera} global frame {global_index} (local {local_index})")


class LeRobotSO101Dataset(Dataset):
    """Cosmos sample layout with right/wrist/front as its three camera tokens."""

    def __init__(
        self,
        data_dir: str = "/data/rxhuang/three_cubes_1",
        is_train: bool = True,
        val_episodes: int = 10,
        overfit_num_episodes: int = 0,
        chunk_size: int = 30,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        num_duplicates_per_image: int = 4,
        gamma: float = 0.99,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,
        use_proprio: bool = True,
        return_value_function_returns: bool = True,
        **legacy_aloha_options,
    ):
        if not (use_wrist_images and use_third_person_images and use_proprio and return_value_function_returns):
            raise ValueError("SO101 Cosmos layout requires three cameras, proprio, and the value placeholder")
        harmless_legacy_keys = {
            "debug",
            "debug2",
            "demonstration_sampling_prob",
            "history_spacing_factor",
            "lazy_video_decompression",
            "load_all_rollouts_into_ram",
            "num_history_indices",
            "rollout_data_dir",
            "success_rollout_sampling_prob",
            "treat_demos_as_success_rollouts",
            "treat_success_rollouts_as_demos",
            "use_jpeg_for_rollouts",
        }
        unknown = set(legacy_aloha_options) - harmless_legacy_keys
        if unknown:
            raise TypeError(f"Unknown LeRobotSO101Dataset options: {sorted(unknown)}")
        self.root = Path(data_dir)
        self.schema = load_schema()
        with (self.root / "meta/info.json").open() as f:
            self.info = json.load(f)
        validate_dataset_metadata(self.info, self.schema)
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.num_duplicates_per_image = num_duplicates_per_image
        self.gamma = gamma

        frames = []
        for path in sorted(glob.glob(str(self.root / "data/**/*.parquet"), recursive=True)):
            table = pq.read_table(path)
            frames.append(
                {
                    "action": np.asarray(table["action"].to_pylist(), np.float32),
                    "state": np.asarray(table["observation.state"].to_pylist(), np.float32),
                    "episode": np.asarray(table["episode_index"], np.int64),
                    "frame": np.asarray(table["frame_index"], np.int64),
                    "task": np.asarray(table["task_index"], np.int64),
                }
            )
        self.actions = np.concatenate([x["action"] for x in frames])
        self.states = np.concatenate([x["state"] for x in frames])
        self.episode_indices = np.concatenate([x["episode"] for x in frames])
        self.frame_indices = np.concatenate([x["frame"] for x in frames])
        self.task_indices = np.concatenate([x["task"] for x in frames])
        if len(self.actions) != self.info["total_frames"]:
            raise ValueError("Parquet frame count differs from metadata")

        all_episode_ids = np.unique(self.episode_indices)
        if overfit_num_episodes:
            selected = all_episode_ids[:overfit_num_episodes]
        elif val_episodes <= 0:
            selected = all_episode_ids
        elif is_train:
            selected = all_episode_ids[:-val_episodes]
        else:
            selected = all_episode_ids[-val_episodes:]
        self.sample_indices = np.flatnonzero(np.isin(self.episode_indices, selected))
        self.episode_bounds = {}
        for episode in all_episode_ids:
            indices = np.flatnonzero(self.episode_indices == episode)
            self.episode_bounds[int(episode)] = (int(indices[0]), int(indices[-1]) + 1)

        tasks = pq.read_table(self.root / "meta/tasks.parquet").to_pydict()
        self.tasks = dict(zip(tasks["task_index"], tasks["task"], strict=True))
        self.unique_commands = set(self.tasks.values())
        if not t5_text_embeddings_path:
            t5_text_embeddings_path = str(self.root / "t5_embeddings.pkl")
        if not Path(t5_text_embeddings_path).is_file():
            raise FileNotFoundError(
                f"Missing T5 embeddings: {t5_text_embeddings_path}. Run "
                "python -m cosmos_policy.datasets.save_so101_t5_text_embeddings first."
            )
        with Path(t5_text_embeddings_path).open("rb") as f:
            self.t5_text_embeddings = pickle.load(f)

        with (self.root / "meta/stats.json").open() as f:
            source_stats = json.load(f)
        self.dataset_stats = {
            "actions_min": np.asarray(source_stats["action"]["min"], np.float32),
            "actions_max": np.asarray(source_stats["action"]["max"], np.float32),
            "actions_mean": np.asarray(source_stats["action"]["mean"], np.float32),
            "actions_std": np.asarray(source_stats["action"]["std"], np.float32),
            "proprio_min": np.asarray(source_stats["observation.state"]["min"], np.float32),
            "proprio_max": np.asarray(source_stats["observation.state"]["max"], np.float32),
            "proprio_mean": np.asarray(source_stats["observation.state"]["mean"], np.float32),
            "proprio_std": np.asarray(source_stats["observation.state"]["std"], np.float32),
            "joint_order": self.schema["joint_order"],
            "action_key": self.schema["action_key"],
            "action_type": self.schema["action_type"],
        }
        self.video_store = _LeRobotVideoStore(self.root, self.schema["camera_keys"])
        for camera, ends in self.video_store.ends.items():
            if ends[-1] != len(self.actions):
                raise ValueError(f"{camera} video frames={ends[-1]}, parquet frames={len(self.actions)}")
        print_schema_summary(self.schema, source_stats["action"])
        print(
            f"LeRobotSO101Dataset split={'train' if is_train else 'val'} episodes={selected.tolist()} samples={len(self)}"
        )

    def __len__(self) -> int:
        return len(self.sample_indices)

    @staticmethod
    def _minmax(array: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        scale = np.maximum(upper - lower, 1e-6)
        return (2.0 * (array - lower) / scale - 1.0).astype(np.float32)

    def absolute_action_chunk(self, global_index: int) -> np.ndarray:
        episode = int(self.episode_indices[global_index])
        _, end = self.episode_bounds[episode]
        stop = min(global_index + self.chunk_size, end)
        chunk = self.actions[global_index:stop]
        if len(chunk) < self.chunk_size:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], self.chunk_size - len(chunk), axis=0)])
        return chunk.copy()

    def __getitem__(self, index: int) -> dict:
        global_index = int(self.sample_indices[index])
        episode = int(self.episode_indices[global_index])
        _, episode_end = self.episode_bounds[episode]
        future_index = min(global_index + self.chunk_size, episode_end - 1)
        command = self.tasks[int(self.task_indices[global_index])]

        current = {camera: self.video_store.read(camera, global_index) for camera in ("front", "right", "wrist")}
        future = {camera: self.video_store.read(camera, future_index) for camera in ("front", "right", "wrist")}
        blank = np.zeros_like(current["front"])
        # tokenizer segments: blank, proprio, right, wrist, front, action,
        # future proprio, future right, future wrist, future front, value
        frames = [
            blank,
            blank,
            current["right"],
            current["wrist"],
            current["front"],
            blank,
            blank,
            future["right"],
            future["wrist"],
            future["front"],
            blank,
        ]
        repeats = [1] + [self.num_duplicates_per_image] * 10
        video = preprocess_image(
            np.stack(frames),
            self.final_image_size,
            self.normalize_images,
            self.use_image_aug,
            self.use_stronger_image_aug,
        )
        video = torch.repeat_interleave(video, torch.as_tensor(repeats), dim=1)

        absolute_actions = self.absolute_action_chunk(global_index)
        actions = absolute_actions
        proprio = self.states[global_index].copy()
        future_proprio = self.states[future_index].copy()
        if self.normalize_actions:
            actions = self._minmax(actions, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"])
        if self.normalize_proprio:
            proprio = self._minmax(proprio, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"])
            future_proprio = self._minmax(
                future_proprio, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"]
            )
        relative_future = future_index - self.episode_bounds[episode][0]
        episode_len = episode_end - self.episode_bounds[episode][0]
        value = self.gamma ** max(episode_len - 1 - relative_future, 0)
        next_index = min(global_index + self.chunk_size, episode_end - 1)
        next_actions = self.absolute_action_chunk(next_index)
        if self.normalize_actions:
            next_actions = self._minmax(
                next_actions, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"]
            )
        return {
            "video": video,
            "command": command,
            "actions": actions,
            "absolute_actions": absolute_actions,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[command]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio,
            "future_proprio": future_proprio,
            "__key__": global_index,
            "episode_index": episode,
            "frame_index": int(self.frame_indices[global_index]),
            "value_function_return": np.float32(value),
            "next_action_chunk": next_actions,
            "next_value_function_return": np.float32(value),
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            # These LeRobot episodes are demonstrations.  Setting this to 1
            # would condition on the ground-truth action frame (world-model
            # mode), making policy action loss identically zero.
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": 5,
            "value_latent_idx": 10,
            "current_proprio_latent_idx": 1,
            "current_wrist_image_latent_idx": 2,
            "current_wrist_image2_latent_idx": 3,
            "current_image_latent_idx": 4,
            "future_proprio_latent_idx": 6,
            "future_wrist_image_latent_idx": 7,
            "future_wrist_image2_latent_idx": 8,
            "future_image_latent_idx": 9,
        }

"""LeRobot SO101 数据到 Cosmos Policy 11-slot latent 序列的通用适配器。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_utils import preprocess_image

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as error:  # pragma: no cover - 提供更明确的安装提示
    LeRobotDataset = None
    _LEROBOT_IMPORT_ERROR = error
else:
    _LEROBOT_IMPORT_ERROR = None


CAMERA_ROLES = ("primary", "wrist_left", "wrist_right")
LATENT_INDICES = {
    "current_proprio_latent_idx": 1,
    "current_wrist_image_latent_idx": 2,
    "current_wrist_image2_latent_idx": 3,
    "current_image_latent_idx": 4,
    "action_latent_idx": 5,
    "future_proprio_latent_idx": 6,
    "future_wrist_image_latent_idx": 7,
    "future_wrist_image2_latent_idx": 8,
    "future_image_latent_idx": 9,
    "value_latent_idx": 10,
}


def _require_lerobot() -> None:
    if LeRobotDataset is None:
        raise ImportError(
            "缺少 LeRobot dataset 依赖。请安装 lerobot[dataset]，并把 LeRobot fork 的 src 加入 PYTHONPATH。"
        ) from _LEROBOT_IMPORT_ERROR


def _feature_dim(features: dict[str, dict], key: str) -> int:
    if key not in features:
        raise KeyError(f"LeRobot metadata 中缺少 feature: {key}")
    shape = tuple(features[key].get("shape", ()))
    if len(shape) != 1:
        raise ValueError(f"{key} 必须是一维向量，metadata shape={shape}")
    return int(shape[0])


def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    image = torch.as_tensor(image).detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"相机帧必须是 3D tensor，实际 shape={tuple(image.shape)}")
    if image.shape[0] in (1, 3, 4):
        image = image[:3].permute(1, 2, 0)
    elif image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"无法识别相机 channel 维度，shape={tuple(image.shape)}")
    image = image[..., :3]
    if image.dtype.is_floating_point:
        if image.numel() and image.max() <= 1.0:
            image = image * 255.0
        image = image.round().clamp(0, 255)
    return image.to(torch.uint8).numpy()


def minmax_normalize(array: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    scale = np.maximum(upper - lower, 1e-6)
    return (2.0 * (array - lower) / scale - 1.0).astype(np.float32)


def transform_actions(actions: np.ndarray, states: np.ndarray, action_mode: str) -> np.ndarray:
    """把 LeRobot action 转成配置指定的物理语义。"""
    if action_mode in {"absolute", "delta"}:
        return actions.astype(np.float32, copy=True)
    if action_mode == "delta_to_absolute":
        if actions.shape != states.shape:
            raise ValueError(
                "delta_to_absolute 要求 action/state shape 完全相同，"
                f"实际为 {actions.shape} 与 {states.shape}"
            )
        return (states + actions).astype(np.float32)
    raise ValueError("action_mode 必须是 absolute、delta 或 delta_to_absolute")


def compute_so101_statistics(
    dataset: Any,
    state_key: str = "observation.state",
    action_key: str = "action",
    action_mode: str = "absolute",
) -> dict[str, Any]:
    """从已完成 episode 过滤的 LeRobotDataset 计算 Cosmos min/max stats。"""
    columns = dataset.select_columns([action_key, state_key])
    actions = np.asarray(columns[action_key], dtype=np.float32)
    states = np.asarray(columns[state_key], dtype=np.float32)
    transformed = transform_actions(actions, states, action_mode)
    action_names = dataset.meta.features[action_key].get("names")
    state_names = dataset.meta.features[state_key].get("names")
    return {
        "action_key": action_key,
        "state_key": state_key,
        "action_mode": action_mode,
        "action_dim": int(transformed.shape[-1]),
        "proprio_dim": int(states.shape[-1]),
        "action_names": action_names,
        "state_names": state_names,
        "actions_min": transformed.min(axis=0).tolist(),
        "actions_max": transformed.max(axis=0).tolist(),
        "actions_mean": transformed.mean(axis=0).tolist(),
        "actions_std": transformed.std(axis=0).tolist(),
        "proprio_min": states.min(axis=0).tolist(),
        "proprio_max": states.max(axis=0).tolist(),
        "proprio_mean": states.mean(axis=0).tolist(),
        "proprio_std": states.std(axis=0).tolist(),
    }


class SO101LeRobotCosmosDataset(Dataset):
    """使用 LeRobotDataset 解码和对齐，输出 Cosmos Policy ALOHA-compatible sample。"""

    def __init__(
        self,
        repo_id: str,
        root: str | None = None,
        episodes: list[int] | None = None,
        chunk_size: int = 50,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        dataset_stats_path: str = "",
        camera_map: dict[str, str] | None = None,
        state_key: str = "observation.state",
        action_key: str = "action",
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        action_mode: str = "absolute",
        use_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        num_duplicates_per_image: int = 4,
        return_value_function_returns: bool = False,
        gamma: float = 0.99,
        normalize_images: bool = False,
        video_backend: str | None = None,
        **unused_options,
    ):
        _require_lerobot()
        inherited_cosmos_options = {
            "data_dir",
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
            "use_third_person_images",
            "use_wrist_images",
        }
        unknown_options = set(unused_options).difference(inherited_cosmos_options)
        if unknown_options:
            raise TypeError(f"未知 SO101 dataset 参数: {sorted(unknown_options)}")
        if not use_proprio:
            raise ValueError("当前 11-slot SO101 layout 要求 use_proprio=True")
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if num_duplicates_per_image != 4:
            raise ValueError("Predict2 tokenizer 的 11-slot/41-frame layout 要求 num_duplicates_per_image=4")

        self.repo_id = repo_id
        self.root = Path(root).expanduser() if root else None
        self.episodes = episodes
        self.chunk_size = int(chunk_size)
        self.final_image_size = int(final_image_size)
        self.state_key = state_key
        self.action_key = action_key
        self.action_mode = action_mode
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.normalize_images = normalize_images
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = float(gamma)
        self.camera_map = camera_map or {
            "primary": "observation.images.front",
            "wrist_left": "observation.images.right",
            "wrist_right": "observation.images.wrist",
        }
        if set(self.camera_map) != set(CAMERA_ROLES):
            raise ValueError(f"第一版只支持三个角色 {CAMERA_ROLES}，实际 camera_map={self.camera_map}")

        # action/state 各取 2*chunk，既能构造当前 chunk，也能构造 next_action_chunk。
        # 相机只需 t 和 t+chunk；episode 尾部由 LeRobot 自动复制最后有效帧并标记 is_pad。
        fps_hint = self._read_fps_hint()
        offsets = [step / fps_hint for step in range(2 * self.chunk_size)]
        delta_timestamps = {key: [0.0, self.chunk_size / fps_hint] for key in self.camera_map.values()}
        delta_timestamps[action_key] = offsets
        delta_timestamps[state_key] = offsets
        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=self.root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            return_uint8=True,
            video_backend=video_backend,
        )
        self.fps = int(self.dataset.fps)
        if self.fps != fps_hint:
            raise ValueError(f"metadata fps 在初始化期间发生变化: {fps_hint} -> {self.fps}")

        features = self.dataset.meta.features
        self.action_dim = _feature_dim(features, action_key)
        self.proprio_dim = _feature_dim(features, state_key)
        for role, key in self.camera_map.items():
            if key not in features or features[key].get("dtype") not in {"video", "image"}:
                raise KeyError(f"camera_map[{role}]={key} 不是有效 LeRobot image/video feature")
        if action_mode == "delta_to_absolute" and self.action_dim != self.proprio_dim:
            raise ValueError("delta_to_absolute 要求 action_dim == proprio_dim")

        self.dataset_stats = self._load_stats(dataset_stats_path)
        self.action_names = self.dataset_stats.get("action_names") or features[action_key].get("names")
        self.state_names = self.dataset_stats.get("state_names") or features[state_key].get("names")
        self.t5_text_embeddings = self._load_embeddings(t5_text_embeddings_path)
        self.unique_tasks = tuple(str(task) for task in self.dataset.meta.tasks.index.tolist())
        missing_tasks = [task for task in self.unique_tasks if task not in self.t5_text_embeddings]
        if missing_tasks:
            raise KeyError(
                f"T5 embedding 缺少 {len(missing_tasks)} 个 task（示例: {missing_tasks[:3]}）。"
                "请先运行 save_so101_lerobot_t5_text_embeddings.py。"
            )
        self.episode_lengths = {
            int(row["episode_index"]): int(row["length"]) for row in self.dataset.meta.episodes
        }

        print(
            "SO101LeRobotCosmosDataset: "
            f"samples={len(self)}, fps={self.fps}, cameras={self.camera_map}, "
            f"action={action_key}[{self.action_dim}]({action_mode}), "
            f"proprio={state_key}[{self.proprio_dim}], chunk={self.chunk_size}"
        )
        print(f"SO101 action joint order: {self.action_names}")
        if "actions_min" in self.dataset_stats and "actions_max" in self.dataset_stats:
            print(
                "SO101 action physical range: "
                f"min={self.dataset_stats['actions_min'].tolist()}, "
                f"max={self.dataset_stats['actions_max'].tolist()}"
            )
            gripper_indices = [
                index for index, name in enumerate(self.action_names or ()) if "gripper" in str(name).lower()
            ]
            if len(gripper_indices) != 1:
                raise RuntimeError(f"无法唯一确定 SO101 gripper 维度：action_names={self.action_names}")
            gripper_index = gripper_indices[0]
            print(
                f"SO101 gripper: index={gripper_index}, name={self.action_names[gripper_index]!r}, "
                f"range=[{self.dataset_stats['actions_min'][gripper_index]}, "
                f"{self.dataset_stats['actions_max'][gripper_index]}]"
            )

    def _read_fps_hint(self) -> int:
        # LeRobotMetadata 会读取相同 info；先用轻量 JSON 避免创建两个完整 dataset。
        if self.root is None:
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

            return int(LeRobotDatasetMetadata(self.repo_id).fps)
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"找不到 LeRobot metadata: {info_path}")
        return int(json.loads(info_path.read_text())["fps"])

    def _load_stats(self, path: str) -> dict[str, object]:
        if not path:
            if self.root is None:
                raise ValueError("启用 normalization 时必须显式提供 dataset_stats_path")
            path = str(self.root / "so101_dataset_statistics.json")
        stats_path = Path(path).expanduser()
        if not stats_path.is_file():
            if self.normalize_actions or self.normalize_proprio:
                raise FileNotFoundError(
                    f"找不到 SO101 stats: {stats_path}。请先运行 export_so101_lerobot_stats.py。"
                )
            return {}
        raw = json.loads(stats_path.read_text())
        if raw.get("action_mode", self.action_mode) != self.action_mode:
            raise ValueError(
                f"stats action_mode={raw.get('action_mode')} 与 dataset action_mode={self.action_mode} 不一致"
            )
        # 保留 action_names/state_names/key/mode，供训练前后做物理 schema 审计；
        # 只有数值统计量转换成 ndarray。
        result: dict[str, object] = dict(raw)
        for key, value in raw.items():
            if key.endswith(("_min", "_max", "_mean", "_std")):
                result[key] = np.asarray(value, np.float32)
        for key, dim in (("actions_min", self.action_dim), ("actions_max", self.action_dim)):
            if self.normalize_actions and result.get(key, np.empty(0)).shape != (dim,):
                raise ValueError(f"stats {key} 维度错误，期望 {(dim,)}")
        for key, dim in (("proprio_min", self.proprio_dim), ("proprio_max", self.proprio_dim)):
            if self.normalize_proprio and result.get(key, np.empty(0)).shape != (dim,):
                raise ValueError(f"stats {key} 维度错误，期望 {(dim,)}")
        return result

    def _load_embeddings(self, path: str) -> dict[str, torch.Tensor]:
        if not path:
            if self.root is None:
                raise ValueError("必须提供 t5_text_embeddings_path")
            path = str(self.root / "so101_t5_embeddings.pkl")
        embedding_path = Path(path).expanduser()
        if not embedding_path.is_file():
            raise FileNotFoundError(
                f"找不到 T5 embeddings: {embedding_path}。"
                "请先运行 save_so101_lerobot_t5_text_embeddings.py。"
            )
        with embedding_path.open("rb") as file:
            return pickle.load(file)

    def __len__(self) -> int:
        return len(self.dataset)

    def _value_at(self, episode: int, frame: int) -> np.float32:
        if not self.return_value_function_returns:
            return np.float32(0.0)
        remaining = max(self.episode_lengths[episode] - 1 - frame, 0)
        return np.float32(self.gamma**remaining)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.dataset[index]
        action_sequence = torch.as_tensor(item[self.action_key], dtype=torch.float32).cpu().numpy()
        state_sequence = torch.as_tensor(item[self.state_key], dtype=torch.float32).cpu().numpy()
        physical_actions = transform_actions(action_sequence, state_sequence, self.action_mode)
        actions = physical_actions[: self.chunk_size].copy()
        next_actions = physical_actions[self.chunk_size : 2 * self.chunk_size].copy()
        physical_proprio = state_sequence[0].copy()
        physical_future_proprio = state_sequence[self.chunk_size].copy()
        proprio = physical_proprio.copy()
        future_proprio = physical_future_proprio.copy()

        if self.normalize_actions:
            actions = minmax_normalize(actions, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"])
            next_actions = minmax_normalize(
                next_actions, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"]
            )
        if self.normalize_proprio:
            proprio = minmax_normalize(
                proprio, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"]
            )
            future_proprio = minmax_normalize(
                future_proprio, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"]
            )

        current = {role: _to_hwc_uint8(item[key][0]) for role, key in self.camera_map.items()}
        future = {role: _to_hwc_uint8(item[key][1]) for role, key in self.camera_map.items()}
        blank = np.zeros_like(current["primary"])
        frames = [
            blank,
            blank,
            current["wrist_left"],
            current["wrist_right"],
            current["primary"],
            blank,
            blank,
            future["wrist_left"],
            future["wrist_right"],
            future["primary"],
            blank,
        ]
        unique_video = preprocess_image(
            np.stack(frames),
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        repeats = torch.tensor([1] + [self.num_duplicates_per_image] * 10)
        video = torch.repeat_interleave(unique_video, repeats, dim=1)
        if video.shape != (3, 41, self.final_image_size, self.final_image_size):
            raise RuntimeError(f"Cosmos video layout 错误: {tuple(video.shape)}")

        episode = int(torch.as_tensor(item["episode_index"]).item())
        frame = int(torch.as_tensor(item["frame_index"]).item())
        future_frame = min(frame + self.chunk_size, self.episode_lengths[episode] - 1)
        command = str(item["task"])
        sample = {
            "video": video,
            "command": command,
            "actions": torch.from_numpy(actions),
            "physical_actions": torch.from_numpy(physical_actions[: self.chunk_size].copy()),
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[command]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": torch.from_numpy(proprio),
            "future_proprio": torch.from_numpy(future_proprio),
            "physical_proprio": torch.from_numpy(physical_proprio),
            "physical_future_proprio": torch.from_numpy(physical_future_proprio),
            "__key__": int(torch.as_tensor(item["index"]).item()),
            "episode_index": episode,
            "frame_index": frame,
            "value_function_return": self._value_at(episode, future_frame),
            "next_action_chunk": torch.from_numpy(next_actions),
            "next_value_function_return": self._value_at(
                episode, min(future_frame + self.chunk_size, self.episode_lengths[episode] - 1)
            ),
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "has_value_function_return": int(self.return_value_function_returns),
        }
        sample.update(LATENT_INDICES)
        return sample

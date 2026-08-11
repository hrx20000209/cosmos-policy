"""One-step late-binding oracle over validation/heldout collected states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_policy.runtime.model_probe import FullHiddenCapture  # noqa: E402
from experiments.cosmos_denoising_libero_pro.scripts.run_manifest import ManifestLiberoEnvironment, install_libero_checkout, load_jsonl  # noqa: E402
from experiments.libero_harness import extract_observation  # noqa: E402
from experiments.progressive_wam.run_p1_trajectory_dump import build_cfg, load_model  # noqa: E402
from experiments.progressive_wam.run_p2_oracle import restore  # noqa: E402


CHECKPOINT_SHA256 = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"
BLOCKS = tuple(range(2, 28, 2))
GROUPS = {
    "current_visual": (2, 3),
    "visual_proprio": (1, 2, 3),
    "future": (6, 7),
    "action": (4,),
    "visual_future": (2, 3, 6, 7),
    "visual_action": (2, 3, 4),
    "future_action": (4, 6, 7),
    "all_dynamic_nonvalue": (1, 2, 3, 4, 5, 6, 7),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cfg_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(checkpoint=args.checkpoint, dataset_stats=args.dataset_stats, t5_embeddings=args.t5_embeddings, action_horizon=16)


def obs_dict(observation: Any) -> dict[str, Any]:
    return {"primary_image": observation.primary_image, "wrist_image": observation.wrist_image, "proprio": observation.proprio}


def actions(result: dict[str, Any]) -> np.ndarray:
    return np.asarray(result["actions"], dtype=np.float32).reshape(16, 7)


def call(cfg: Any, model: Any, stats: dict, observation: Any, instruction: str, seed: int, previous: torch.Tensor) -> dict[str, Any]:
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    return get_action(
        cfg,
        model,
        stats,
        obs_dict(observation),
        instruction,
        seed=seed,
        randomize_seed=False,
        num_denoising_steps_action=1,
        generate_future_state_and_value_in_parallel=False,
        decode_future_state=False,
        skip_vae_encoding=True,
        previous_generated_latent=previous,
        skip_camera_preprocessing=True,
    )


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(left - right, axis=-1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-states", type=int, default=1000)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("discovery", "validation", "heldout"),
        choices=("discovery", "validation", "heldout"),
    )
    parser.add_argument("--checkpoint", default="/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
    parser.add_argument("--dataset-stats", default="/data/rxhuang/models/cosmos-policy-libero-2b/libero_dataset_statistics.json")
    parser.add_argument("--t5-embeddings", default="/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/cosmos_libero_pro_t5_embeddings.pkl")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if "so101" in str(checkpoint).lower() or "finet" in str(checkpoint).lower() or sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("refusing non-original checkpoint")
    rows = {row["episode_key"]: row for row in load_jsonl(args.manifest)}
    files = list(args.collection_root.glob("group*/episode_*.pt")) + list(args.collection_root.glob("episode_*.pt"))
    episodes = []
    for path in files:
        try:
            episode = torch.load(path, weights_only=False)
        except Exception:
            continue
        row = rows.get(episode.get("episode_key"))
        if row is None or row["split"] not in set(args.splits):
            continue
        episodes.append((row, episode))
    states: list[tuple[int, dict, dict, dict]] = []
    for row, episode in sorted(episodes, key=lambda item: item[0]["episode_key"]):
        requests = episode.get("requests", [])
        for request_index in range(1, len(requests)):
            states.append((len(states), row, episode, requests[request_index]))
    states = states[: args.max_states]
    assigned = [item for item in states if item[0] % args.num_shards == args.shard_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    done = {path.stem for path in args.output_dir.glob("state_*.json")}
    if not assigned:
        print(json.dumps({"status": "no_states", "all_states": len(states)}), flush=True)
        return
    first = assigned[0][1]
    install_libero_checkout(Path(first["libero_repo"]), Path(first["libero_config_path"]))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    osmesa = "/data/rxhuang/osmesa-jammy-23.2.1/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = osmesa + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    cfg = build_cfg(cfg_args(args))
    model, stats = load_model(cfg, cfg_args(args))
    model.eval()
    count = 0
    for global_index, row, episode, request in assigned:
        output_path = args.output_dir / f"state_{global_index:05d}_{request['state_key'].replace(':', '_')}.json"
        if output_path.stem in done or output_path.exists():
            continue
        env = None
        try:
            env = ManifestLiberoEnvironment(row, 256, None)
            env.reset()
            sim_state = np.asarray(request["sim_state"], dtype=np.float64)
            restore(env, sim_state)
            observation = extract_observation(env.env.regenerate_obs_from_state(sim_state), flip_vertical=True)
            previous = torch.from_numpy(np.asarray(episode["requests"][int(request["request_index"]) - 1]["generated_latent"], dtype=np.float16).astype(np.float32)).cuda()
            fresh_hidden: dict[int, torch.Tensor] = {}
            model.intermediate_feature_ids = list(BLOCKS)
            model.intermediate_feature_reducer = FullHiddenCapture()

            def capture_fresh(**_: Any) -> None:
                features = list(model.last_intermediate_features or ())
                if len(features) != len(BLOCKS):
                    raise RuntimeError(f"captured {len(features)} features, expected {len(BLOCKS)}")
                fresh_hidden.update(dict(zip(BLOCKS, features, strict=True)))

            model.sampler.checkpoint_hook = capture_fresh
            with torch.inference_mode():
                fresh = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), previous)
            model.sampler.checkpoint_hook = None
            model.intermediate_feature_ids = None
            model.intermediate_feature_reducer = None
            fresh_action = actions(fresh)
            with torch.inference_mode():
                predicted = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), previous)
            predicted_action = actions(predicted)
            baseline = distance(predicted_action, fresh_action)
            repairs = []
            for block in BLOCKS:
                for group_name, slots in GROUPS.items():
                    def pre_hook(*, denoiser_forward_index: int, **_: Any) -> None:
                        model.net.activation_patch_request = {"fresh_hidden_by_block": {block: fresh_hidden[block]}, "slot_groups": {group_name: slots}} if denoiser_forward_index == 0 else None

                    def transform(*, denoiser_forward_index: int, predicted_clean: torch.Tensor, **_: Any) -> torch.Tensor:
                        if denoiser_forward_index != 0:
                            return predicted_clean
                        patched = model.last_activation_patch_latents
                        if not patched or block not in patched or group_name not in patched[block]:
                            raise RuntimeError(f"missing patch block={block} group={group_name}")
                        return patched[block][group_name]

                    model.sampler.pre_denoise_hook = pre_hook
                    model.sampler.x0_transform = transform
                    try:
                        with torch.inference_mode():
                            repaired = call(cfg, model, stats, observation, row["instruction"], int(row["seed"]), previous)
                    finally:
                        model.sampler.pre_denoise_hook = None
                        model.sampler.x0_transform = None
                        model.net.activation_patch_request = None
                    repaired_action = actions(repaired)
                    repaired_distance = distance(repaired_action, fresh_action)
                    repairs.append(
                        {
                            "block": block,
                            "group": group_name,
                            "distance_to_fresh": repaired_distance,
                            "recovery": 1.0 - repaired_distance / max(baseline, 1e-8),
                            "first_action_l2_to_fresh": float(np.linalg.norm(repaired_action[0] - fresh_action[0])),
                            "gripper_sign_match": float(np.sign(repaired_action[0, -1]) == np.sign(fresh_action[0, -1])),
                            "remaining_dit_fraction": float((28 - block - 1) / 28),
                        }
                    )
            output = {
                "schema_version": 1,
                "experiment": "one_step_late_binding_oracle",
                "global_index": global_index,
                "state_key": request["state_key"],
                "episode_key": row["episode_key"],
                "split": row["split"],
                "task_uid": row["task_uid"],
                "request_index": request["request_index"],
                "baseline_predicted_to_fresh": baseline,
                "repairs": repairs,
                "blocks": BLOCKS,
                "groups": GROUPS,
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "value_used": False,
                "oracle_fresh_prefix_required": True,
            }
            temporary = output_path.with_suffix(".partial.json")
            temporary.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(output_path)
            count += 1
            if count % 5 == 0:
                print(json.dumps({"shard": args.shard_index, "completed": count, "of": len(assigned)}), flush=True)
        except Exception as error:
            print(json.dumps({"global_index": global_index, "error": f"{type(error).__name__}:{error}"}), flush=True)
        finally:
            model.sampler.checkpoint_hook = None
            model.sampler.pre_denoise_hook = None
            model.sampler.x0_transform = None
            model.net.activation_patch_request = None
            model.intermediate_feature_ids = None
            model.intermediate_feature_reducer = None
            if env is not None:
                env.close()
    summary = {"schema_version": 1, "shard_index": args.shard_index, "num_shards": args.num_shards, "all_states": len(states), "assigned": len(assigned), "completed": count, "finished_at_ns": time.time_ns(), "checkpoint_sha256": CHECKPOINT_SHA256}
    (args.output_dir / f"summary_shard{args.shard_index:02d}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

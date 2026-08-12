#!/usr/bin/env python3
"""Write ESP Phase-0 source audit artifacts without modifying model behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = Path("/data/rxhuang/models/cosmos-policy-libero-2b/Cosmos-Policy-LIBERO-Predict2-2B.pt")
CHECKPOINT_SHA = "8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2"


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/esp")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    git_sha = command("git", "rev-parse", "HEAD")
    git_status = command("git", "status", "--short")
    nvidia = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    token_map = {
        "schema_version": 1,
        "layout": "LIBERO original pre-finetune checkpoint",
        "latent_temporal_slots": {
            "0": "structural blank",
            "1": "current proprio",
            "2": "current wrist image / wrist camera",
            "3": "current primary image / third-person camera",
            "4": "action chunk",
            "5": "future proprio",
            "6": "future wrist image",
            "7": "future primary image",
            "8": "value slot (never read in ESP)",
        },
        "condition_slots": [0, 1, 2, 3],
        "current_visual_camera_slots": {"2": "wrist", "3": "primary"},
        "action_latent_slot": 4,
        "future_visual_slots": [6, 7],
        "tokenization": {
            "patch_spatial": 2,
            "patch_temporal": 1,
            "pre_patch_latent_shape": "[B,C=16,T=9,H=28,W=28]",
            "hidden_grid_shape": "[B,T=9,H=14,W=14,D=2048]",
            "action_token_range_flat": [784, 979],
            "action_token_count": 196,
            "action_hidden_pooling": "all 14x14 patch tokens at temporal slot 4; normalized RMS L2",
        },
        "sources": {
            "layout_and_assembly": "cosmos_policy/experiments/robot/cosmos_utils.py:960-1198",
            "LIBERO_config": "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py:75-145",
            "DiT_patch_embedding": "cosmos_policy/_src/predict2/networks/minimal_v4_dit.py:1688-1703",
        },
    }
    vae = {
        "schema_version": 1,
        "suite": "LIBERO",
        "camera_count": 2,
        "camera_preprocessing": "per-image list preprocessing before temporal assembly",
        "vae_encode_call": "model.encode(data_batch['video'])",
        "vae_input": "one joint video tensor, temporal sequence [blank, 4x proprio blank, 4x wrist, 4x primary, ...]",
        "temporal_compression_factor": 4,
        "current_camera_raw_frame_ranges": {"wrist": [5, 8], "primary": [9, 12]},
        "camera_condition_slot_replace": "legal independently via index_copy into slots 2/3 after the joint VAE prefix is encoded",
        "camera_compute_separability": "NO_GO",
        "reason": "The model exposes a single joint temporal VAE call. Primary-only evidence still requires the earlier structural/proprio/wrist temporal prefix; no validated per-camera encoder API avoids that work. Wrist-only could truncate after its own slot but neither route proves independent camera VAE work across all cameras.",
        "sources": {
            "assembly": "cosmos_policy/experiments/robot/cosmos_utils.py:1010-1127",
            "prefix_encode": "cosmos_policy/experiments/robot/cosmos_utils.py:1218-1384",
        },
    }
    manifest = {
        "schema_version": 1,
        "phase": "ESP Phase 0 / audit",
        "git_sha": git_sha,
        "git_status": git_status,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "finetuning_used": False,
        "denoise_steps": 1,
        "value_used": False,
        "precision": "bf16",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "gpus": nvidia},
        "source_sha256": {
            name: sha(name)
            for name in (
                "cosmos_policy/models/policy_video2world_model.py",
                "cosmos_policy/experiments/robot/cosmos_utils.py",
                "cosmos_policy/_src/predict2/networks/minimal_v4_dit.py",
                "cosmos_policy/_src/predict2/networks/minimal_v1_lvg_dit.py",
                "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py",
            )
        },
    }
    assumptions = """# ESP assumptions and validation status

- [CONFIRMED-BY-CODE] LIBERO current visual condition slots are 2 (wrist) and 3 (primary); future visual slots are 6/7.
- [CONFIRMED-BY-CODE] P1 reuses the previous generated joint latent, moving future visual content into current visual slots through `CosmosAdapter._predicted_visual_latent`.
- [CONFIRMED-BY-CODE] Native PV0 refreshes slots 2/3 with a causal physical prefix before denoiser forward 0.
- [CONFIRMED-BY-CODE] DiT has 28 blocks, hidden size 2048, 16 heads, and `minimal_a2a` attention. Each block uses self-attention, text cross-attention, AdaLN, and residual paths.
- [CONFIRMED-BY-CODE] A passive intermediate-feature capture exists and returns post-block hidden grids without activation patching.
- [CONFIRMED-BY-CODE] LIBERO VAE input is temporally joint; current primary cannot be independently encoded without retaining earlier temporal prefix. E1 camera-compute separability is therefore NO-GO.
- [ASSUMED] A custom early exit after blocks k∈{2,4,6} can preserve the exact prefix semantics and expose action-token hidden without running the suffix. This requires a passive early-exit smoke test before E3.
- [ASSUMED] A deployable E^c can use a temporally valid prior real condition at an R2 decision point without current-F1 leakage. This requires a provenance smoke test; current-fresh variants are oracle-only.
- [ASSUMED] Captured hidden deltas are above numerical noise. This requires 10 repeated E0 probes.
"""
    audit = """# ESP Phase-0 codebase audit

## Model and inference entry

`CosmosAdapter.infer` calls `get_action`, which calls `CosmosPolicyVideo2WorldModel.generate_samples_from_batch`. The one-step sampler invokes `x0_fn`, then `model.denoise`; `denoise` constructs the video-conditioned tensor and invokes `self.net` at `policy_video2world_model.py:401-471`.

The loaded original LIBERO checkpoint instantiates `MinimalV1LVGDiT`: 28 transformer blocks, hidden size 2048, 16 heads, BF16, `minimal_a2a` attention. The base transformer is `MiniTrainDIT`; its exact block loop is in `minimal_v4_dit.py:1715-1863`.

## Condition path

Video evidence is injected before block 0: `denoise` overwrites `net_state_in` with `condition.gt_frames` where the video-condition mask is one (`policy_video2world_model.py:420-446`). It is not re-injected at each block. Every block performs AdaLN-modulated self-attention over unified spatio-temporal tokens, text cross-attention, and an MLP with residual connections (`minimal_v4_dit.py:1131-1350`). Q/K/V are projected by `Attention.compute_qkv`; RoPE applies to self-attention, not text cross-attention (`minimal_v4_dit.py:390-590`).

## ESP legality

Condition-slot finite differences are legal because existing PV0 already uses native `condition.gt_frames.index_copy_` before the sole denoiser forward. ESP must construct E0/E^c through this condition mechanism, never by hidden/x0 patching. The existing intermediate feature API is passive, but it continues through the full suffix. Consequently shallow capture is source-legal; cheap early-exit is **not yet confirmed** and is a mandatory smoke-test gate.

## E1 architecture decision

Current visual slots are independently replaceable, but the VAE is not camera-compute-separable for LIBERO. It encodes one joint temporal video; primary evidence requires prior temporal frames. Therefore `E1_SELECTIVE_SENSING_ARCH_NO_GO`. This does not invalidate E2 causal importance or E3 offline sensitivity validation, but it prohibits claims of selective per-camera VAE savings on this layout.
"""
    write_json(out / "TOKEN_SLOT_MAP.json", token_map)
    write_json(out / "VAE_ARCHITECTURE.json", vae)
    write_json(out / "RUN_MANIFEST.json", manifest)
    (out / "ASSUMPTIONS.md").write_text(assumptions, encoding="utf-8")
    (out / "AUDIT.md").write_text(audit, encoding="utf-8")
    print(json.dumps({"output": str(out), "git_sha": git_sha, "e1": "E1_SELECTIVE_SENSING_ARCH_NO_GO"}))


if __name__ == "__main__":
    main()

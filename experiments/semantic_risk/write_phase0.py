#!/usr/bin/env python3
"""Write semantic-risk phase-0 architecture/provenance audit artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

from experiments.server_deep_validation.pv0_overnight_common import ORIGINAL_CHECKPOINT, checkpoint_contract, atomic_write_json


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    reports = root / "reports/semantic_risk"
    reports.mkdir(parents=True, exist_ok=True)
    contract = checkpoint_contract(ORIGINAL_CHECKPOINT)
    manifest = {
        "schema_version": 1,
        "git_sha": command("git", "rev-parse", "HEAD"),
        "branch": command("git", "branch", "--show-current"),
        "dirty_status": command("git", "status", "--short"),
        **contract,
        "denoising_steps": 1,
        "value_used": False,
        "finetuning_used": False,
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "bfloat16 model weights; action inference uses frozen native path",
        "frozen_negative_branches": ["ESP finite-difference", "ESP camera selection", "per-camera VAE", "AGE-only final method", "hidden/x0 patch", "fixed reuse scheduler"],
    }
    audit = """# Semantic-risk Phase-0 audit

The current commit retains the ESP-audited original pre-finetune LIBERO checkpoint: 28 DiT blocks, 2048 hidden width, 16 heads and `minimal_a2a` optimized self-attention. Current visual condition slots are 2 (wrist) and 3 (primary); action occupies temporal slot 4 (196 patch tokens), future proprio/visual slots are 5/6/7.

E4 uses the existing `intermediate_feature_reducer` path in `minimal_v4_dit.py`: it observes selected post-block hidden tensors within the single normal P1 forward and returns small pooled summaries. It does not modify hidden tensors, add a forward pass, or access F1 at runtime. The P1 path uses the preceding generated slots 6/7 copied into current visual slots 2/3; F1 encodes the replayed current visual condition. Offline replay simulator state recreates observations only and is never a model input.

Attention/QK features are not deployable: `minimal_a2a` is the active optimized backend and does not expose a summary without changing/materializing the attention path. They are excluded rather than profiled as zero-overhead.

The VAE remains a joint temporally causal encode, so E5 uses pre-VAE raw-image CPU descriptors only. It never decodes predicted latents or runs a visual network.

## Unknowns

- Thor latency and energy: Server results will not be presented as Thor measurements.
- A source-generated future latent is the available imagined condition; no decoded predicted image is available for an E5 raw-image comparison without prohibited decoder work.
"""
    assumptions = """# Assumptions

- State-bank simulator state is used only to reproduce recorded LIBERO observations.
- `orig_clean_latent_frames` from F1 is the fresh joint condition; P1 imagined condition is constructed from the preceding request's generated future slots exactly as the frozen P1 route.
- E4 target is offline only. Runtime E4 features use only the normal P1 forward/action and history.
- E5 compares current versus immediately preceding real raw frames at 64×64; this is a cheap physical-motion proxy, not a claim to observe decoded WAM imagery.
- Formal heldout analysis is deferred until discovery/validation feature choice is frozen.
"""
    atomic_write_json(reports / "RUN_MANIFEST.json", manifest)
    (reports / "AUDIT.md").write_text(audit, encoding="utf-8")
    (reports / "ASSUMPTIONS.md").write_text(assumptions, encoding="utf-8")
    print(json.dumps({"report_dir": str(reports), "git_sha": manifest["git_sha"]}))


if __name__ == "__main__":
    main()

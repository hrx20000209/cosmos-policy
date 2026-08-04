#!/usr/bin/env python3
"""Precompute every exact manifest instruction before policy construction."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = PROJECT / "experiments/cosmos_denoising_libero_pro/manifests/full_sweep.jsonl"
DEFAULT_BASE = Path("/data/rxhuang/models/cosmos-policy-libero-2b/libero_t5_embeddings.pkl")
DEFAULT_MODEL = Path("/data/rxhuang/models/t5-11b")
DEFAULT_OUTPUT = Path(
    "/data/rxhuang/wam_libero_outputs/cosmos_denoising_full_sweep/assets/"
    "cosmos_libero_pro_t5_embeddings.pkl"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def instructions(path: Path) -> list[str]:
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(json.loads(line)["instruction"])
    return sorted(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--t5-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    commands = instructions(args.manifest)
    with args.base_cache.open("rb") as handle:
        cache = pickle.load(handle)
    missing = [command for command in commands if command not in cache]
    print(f"manifest exact instructions={len(commands)} base_cache={len(cache)} missing={len(missing)}")
    if missing:
        from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder

        encoder = CosmosT5TextEncoder(
            model_name=str(args.t5_model.resolve()),
            device=args.device,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        )
        for offset in range(0, len(missing), args.batch_size):
            batch = missing[offset : offset + args.batch_size]
            encoded = encoder.encode_prompts(batch, max_length=512).to(torch.bfloat16).cpu()
            for command, embedding in zip(batch, encoded, strict=True):
                cache[command] = embedding.unsqueeze(0).contiguous()
            print(f"encoded {min(offset + len(batch), len(missing))}/{len(missing)}", flush=True)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    absent = [command for command in commands if command not in cache]
    if absent:
        raise SystemExit(f"exact T5 cache coverage failed: {absent[:3]}")
    selected = {command: cache[command].detach().cpu().to(torch.bfloat16).contiguous() for command in commands}
    for command, embedding in selected.items():
        if tuple(embedding.shape) != (1, 512, 1024):
            raise ValueError(f"{command!r}: invalid embedding shape {tuple(embedding.shape)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{__import__('os').getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(selected, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        __import__("os").fsync(handle.fileno())
    temporary.replace(args.output)
    audit = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "base_cache": str(args.base_cache.resolve()),
        "base_cache_sha256": file_sha256(args.base_cache),
        "t5_model": str(args.t5_model.resolve()),
        "output": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "instruction_count": len(commands),
        "exact_coverage": True,
        "embeddings": {
            command: {
                "instruction_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                "tensor_sha256": tensor_sha256(embedding),
                "shape": list(embedding.shape),
                "dtype": str(embedding.dtype),
            }
            for command, embedding in selected.items()
        },
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in audit if key != "embeddings"}, indent=2))


if __name__ == "__main__":
    main()

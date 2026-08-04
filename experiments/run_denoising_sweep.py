#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sweep_common import add_common_arguments, run_variants


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument(
        "--cosmos-mode",
        choices=["joint_parallel", "action_only_usage", "autoregressive_future"],
        default="joint_parallel",
    )
    parser.add_argument("--video-steps", type=int, default=20)
    args = parser.parse_args()
    steps = args.steps or ([1, 2, 3, 4, 5, 8] if args.model == "cosmos" else [5, 10, 20, 35, 50])
    variants = []
    for value in steps:
        overrides = [f"denoising.steps={value}"]
        if args.model == "cosmos":
            overrides.extend(
                [
                    f"model.generation_mode={args.cosmos_mode}",
                    f"pipeline.decode_future={str(args.cosmos_mode != 'action_only_usage').lower()}",
                ]
            )
        else:
            overrides.append(f"model.video_inference_steps={args.video_steps}")
        variants.append((f"{args.model}-denoise-{value}-{args.cosmos_mode}", overrides))
    run_variants(
        model=args.model,
        base_config=args.base_config,
        output_dir=args.output_dir,
        variants=variants,
        common_overrides=args.common_overrides,
        mock=args.mock,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

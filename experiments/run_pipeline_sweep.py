#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sweep_common import add_common_arguments, run_variants


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()
    variants = [
        (
            f"{args.model}-pipeline-{mode}",
            [f"pipeline.mode={mode}", f"pipeline.decode_future={str(args.model == 'cosmos').lower()}"],
        )
        for mode in (
            "sync_baseline",
            "action_first_sync",
            "action_first_async_decode",
            "async_encode_sync_dit",
            "full_pipeline",
        )
    ]
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

#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sweep_common import add_common_arguments, run_variants


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()
    official = 5 if args.model == "cosmos" else 50
    normal = 3 if args.model == "cosmos" else 20
    minimum = 1 if args.model == "cosmos" else 5
    variants = [
        (f"{args.model}-always-min", ["denoising.scheduler=fixed", f"denoising.steps={minimum}"]),
        (f"{args.model}-always-official", ["denoising.scheduler=fixed", f"denoising.steps={official}"]),
        (
            f"{args.model}-random-matched",
            [
                "denoising.scheduler=random_matched",
                f"denoising.choices=[{minimum},{normal},{official}]",
                "denoising.weights=[0.4,0.4,0.2]",
            ],
        ),
        (
            f"{args.model}-heuristic",
            [
                "denoising.scheduler=heuristic",
                f"denoising.minimum_steps={minimum}",
                f"denoising.normal_steps={normal}",
                f"denoising.official_steps={official}",
            ],
        ),
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

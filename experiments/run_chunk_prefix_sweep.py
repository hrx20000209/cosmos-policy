#!/usr/bin/env python3
from __future__ import annotations

import argparse

from sweep_common import add_common_arguments, run_variants


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()
    variants = [
        (f"{args.model}-prefix-{length}", ["execution_prefix.mode=fixed", f"execution_prefix.length={length}"])
        for length in (1, 2, 4, 8, 16)
    ]
    variants.extend(
        (f"{args.model}-prefix-{mode}", [f"execution_prefix.mode={mode}"])
        for mode in ("proprio_change", "visual_change", "action_variance", "task_stage")
    )
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

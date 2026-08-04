#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools

from sweep_common import add_common_arguments, run_variants


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()
    variants = []
    if args.model == "cosmos":
        for policy in ("latest_only", "fifo", "uniform_sampling", "pixel_change", "proprio_change", "hybrid_change"):
            variants.append((f"cosmos-observation-{policy}", [f"history.observation_policy={policy}", "history.length=1"]))
    else:
        for length, stride, policy in itertools.product(
            (1, 2, 4, 8),
            (1, 2, 4, 8),
            ("dense_recent", "uniform_sparse", "change_based", "latest_cached"),
        ):
            variants.append(
                (
                    f"lingbot-history-k{length}-s{stride}-{policy}",
                    [f"history.length={length}", f"history.frame_stride={stride}", f"history.history_policy={policy}"],
                )
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

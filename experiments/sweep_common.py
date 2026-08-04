from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "experiments" / "run_wam_libero_experiment.py"


def run_variants(
    *,
    model: str,
    base_config: str,
    output_dir: str,
    variants: Iterable[tuple[str, list[str]]],
    mock: bool,
    dry_run: bool,
    common_overrides: Iterable[str] = (),
) -> None:
    for name, overrides in variants:
        command = [
            sys.executable,
            str(ENTRYPOINT),
            "--model",
            model,
            "--config",
            base_config,
            "--output_dir",
            output_dir,
            "--set",
            f"run_id={name}",
        ]
        for value in common_overrides:
            command.extend(["--set", value])
        for value in overrides:
            command.extend(["--set", value])
        if mock:
            command.append("--mock")
        print(" ".join(command))
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def add_common_arguments(parser) -> None:
    parser.add_argument("--model", choices=["cosmos", "lingbot_va"], required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", default="outputs/wam_libero")
    parser.add_argument(
        "--set",
        dest="common_overrides",
        action="append",
        default=[],
        help="Config override applied to every sweep variant; may be repeated.",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

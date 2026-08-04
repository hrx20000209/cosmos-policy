#!/usr/bin/env python3
from scripts.sweep_runner import parse_and_run

if __name__ == "__main__":
    parse_and_run(
        [
            "denoising_4.yaml",
            "baseline_bf16.yaml",
            "denoising_8.yaml",
            "denoising_16.yaml",
            "denoising_int8_4.yaml",
            "denoising_int8_8.yaml",
            "denoising_int8_16.yaml",
        ]
    )

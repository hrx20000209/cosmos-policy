#!/usr/bin/env python3
from scripts.sweep_runner import parse_and_run

if __name__ == "__main__":
    parse_and_run(
        [
            "async_openloop_1.yaml",
            "async_openloop_2.yaml",
            "async_openloop_4.yaml",
            "async_openloop_8.yaml",
            "async_openloop_16.yaml",
            "async_int8_openloop_1.yaml",
            "async_int8_openloop_2.yaml",
            "async_int8_openloop_4.yaml",
            "async_int8_openloop_8.yaml",
            "async_int8_openloop_16.yaml",
        ]
    )

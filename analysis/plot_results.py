#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", type=Path, help="JSON from summarize_results.py")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    rows = json.loads(args.summaries.read_text(encoding="utf-8"))
    if rows and all(isinstance(row.get("denoising_steps"), int) for row in rows):
        rows.sort(key=lambda row: row["denoising_steps"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = [
        str(row["denoising_steps"]) if isinstance(row.get("denoising_steps"), int) else Path(row["run_dir"]).name
        for row in rows
    ]

    def plot(y_key: str, ylabel: str, filename: str, ci_key: str | None = None) -> None:
        keep = [(name, row) for name, row in zip(names, rows) if row.get(y_key) is not None]
        values = [float(row[y_key]) for _, row in keep]
        fig, axis = plt.subplots(figsize=(max(6, len(keep) * 0.55), 4))
        x_values = range(len(keep))
        if ci_key and all(row.get(ci_key, [None, None])[0] is not None for _, row in keep):
            intervals = [row[ci_key] for _, row in keep]
            yerr = [
                [value - interval[0] for value, interval in zip(values, intervals)],
                [interval[1] - value for value, interval in zip(values, intervals)],
            ]
            axis.errorbar(x_values, values, yerr=yerr, marker="o", capsize=3)
        else:
            axis.plot(x_values, values, marker="o")
        axis.set_xticks(range(len(keep)), [name for name, _ in keep], rotation=45, ha="right")
        if rows and all(isinstance(row.get("denoising_steps"), int) for row in rows):
            axis.set_xlabel("Denoising steps")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_dir / filename, dpi=180)
        plt.close(fig)

    plot(
        "aggregate_success_rate",
        "Success rate",
        "denoising_steps_vs_success.png",
        "aggregate_success_wilson_95_ci",
    )
    plot(
        "policy_latency_mean_ms",
        "Policy latency (ms)",
        "denoising_steps_vs_latency.png",
        "policy_latency_bootstrap_95_ci",
    )
    plot(
        "episode_time_mean_s",
        "Episode completion time (s)",
        "denoising_steps_vs_episode_time.png",
        "episode_time_bootstrap_95_ci",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate raw trials/inference CSVs into summary tables (Phase 6).

Reads:  raw/trials_<tag>.csv, raw/inference_steps_<tag>.csv
Writes: summaries/overall_summary.csv, category_summary.csv, difficulty_summary.csv
        summaries/step_latency_by_mode.csv
Uses Wilson 95% CI for success rate.
"""
import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import exp_common as ec


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0, c - h), min(1, c + h))


def pctl(s, q):
    return float(np.percentile(s, q)) if len(s) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()

    trials = pd.concat([pd.read_csv(f) for f in glob.glob(f"{ec.RAW}/trials_{args.tag}*.csv")],
                       ignore_index=True)
    trials = trials.drop_duplicates(subset=["quant_mode", "task_suite", "task_id", "seed"])
    print(f"trials: {len(trials)} rows, modes={sorted(trials.quant_mode.unique())}")

    infer = None
    isteps = glob.glob(f"{ec.RAW}/inference_steps_{args.tag}*.csv")
    if isteps:
        infer = pd.concat([pd.read_csv(f) for f in isteps], ignore_index=True)

    bf = trials[trials.quant_mode == "bf16"]
    bf_sr = bf.success.mean() if len(bf) else float("nan")

    # ---- overall ----
    rows = []
    for mode, g in trials.groupby("quant_mode"):
        k, n = int(g.success.sum()), len(g)
        lo, hi = wilson(k, n)
        lat = infer[infer.quant_mode == mode]["total_policy_step_ms"] if infer is not None else pd.Series(dtype=float)
        row = {
            "quant_mode": mode,
            "hardware_accelerated": g.hardware_accelerated.iloc[0],
            "quant_backend": g.quant_backend.iloc[0],
            "n_tasks": n, "n_success": k,
            "success_rate": round(k / n, 4),
            "sr_ci95_lo": round(lo, 4), "sr_ci95_hi": round(hi, 4),
            "sr_delta_vs_bf16": round(k / n - bf_sr, 4) if not math.isnan(bf_sr) else None,
            "step_latency_mean_ms": round(lat.mean(), 1) if len(lat) else None,
            "step_latency_p50_ms": round(pctl(lat, 50), 1) if len(lat) else None,
            "step_latency_p95_ms": round(pctl(lat, 95), 1) if len(lat) else None,
            "step_latency_p99_ms": round(pctl(lat, 99), 1) if len(lat) else None,
            "amortized_ms_per_action_p50": round(pctl(lat, 50) / 16, 2) if len(lat) else None,
            "peak_memory_mb_median": round(g.peak_memory_mb.median(), 1),
            "episode_ms_mean": round(g.episode_total_ms.mean(), 1),
            "episode_ms_success_mean": round(g[g.success == 1].episode_total_ms.mean(), 1) if (g.success == 1).any() else None,
            "episode_ms_fail_mean": round(g[g.success == 0].episode_total_ms.mean(), 1) if (g.success == 0).any() else None,
        }
        rows.append(row)
    overall = pd.DataFrame(rows).sort_values("quant_mode")
    overall.to_csv(f"{ec.SUMM}/overall_summary.csv", index=False)
    print("\n=== overall_summary ===\n", overall.to_string(index=False))

    # ---- by category ----
    cat_rows = []
    for (mode, cat), g in trials.groupby(["quant_mode", "perturbation_category"]):
        k, n = int(g.success.sum()), len(g)
        cat_rows.append({"quant_mode": mode, "perturbation_category": cat,
                         "n": n, "success_rate": round(k / n, 4)})
    pd.DataFrame(cat_rows).to_csv(f"{ec.SUMM}/category_summary.csv", index=False)

    # ---- by difficulty ----
    diff_rows = []
    for (mode, d), g in trials.groupby(["quant_mode", "difficulty"]):
        k, n = int(g.success.sum()), len(g)
        diff_rows.append({"quant_mode": mode, "difficulty": d,
                          "n": n, "success_rate": round(k / n, 4)})
    pd.DataFrame(diff_rows).to_csv(f"{ec.SUMM}/difficulty_summary.csv", index=False)

    # ---- step latency distribution ----
    if infer is not None:
        lat_rows = []
        for mode, g in infer.groupby("quant_mode"):
            s = g["total_policy_step_ms"]
            lat_rows.append({"quant_mode": mode, "n_calls": len(s),
                             "mean": round(s.mean(), 1), "p50": round(pctl(s, 50), 1),
                             "p90": round(pctl(s, 90), 1), "p95": round(pctl(s, 95), 1),
                             "p99": round(pctl(s, 99), 1), "min": round(s.min(), 1), "max": round(s.max(), 1)})
        pd.DataFrame(lat_rows).to_csv(f"{ec.SUMM}/step_latency_by_mode.csv", index=False)
    print("\nwrote overall/category/difficulty/step_latency summaries")


if __name__ == "__main__":
    main()

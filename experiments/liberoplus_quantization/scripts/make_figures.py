#!/usr/bin/env python3
"""Generate all report figures from raw/summary CSVs (Phase 6).
English axis labels (robust to missing CN fonts); Chinese goes in the report text.
Every figure saved as PNG (300 dpi) + PDF (vector). All data comes from real CSVs.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import exp_common as ec

FIG = ec.EXP + "/figures"
os.makedirs(FIG, exist_ok=True)

# colorblind-safe, consistent per-mode colors
COLORS = {
    "bf16": "#4E79A7", "fp8_backbone": "#59A14F", "int8_backbone": "#EDC948",
    "int4_weight_only": "#E15759", "fake_int4_backbone": "#B07AA1",
}
SHORT = {"bf16": "BF16", "fp8_backbone": "FP8", "int8_backbone": "INT8",
         "int4_weight_only": "INT4-wo", "fake_int4_backbone": "fake-INT4"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def modelist(df):
    order = ["bf16", "fp8_backbone", "int8_backbone", "int4_weight_only", "fake_int4_backbone"]
    return [m for m in order if m in set(df.quant_mode)]


def fig_success_overall(ov):
    modes = [m for m in ["bf16", "fp8_backbone", "int8_backbone", "int4_weight_only", "fake_int4_backbone"] if m in set(ov.quant_mode)]
    x = np.arange(len(modes))
    sr = [ov[ov.quant_mode == m].success_rate.iloc[0] for m in modes]
    lo = [ov[ov.quant_mode == m].sr_ci95_lo.iloc[0] for m in modes]
    hi = [ov[ov.quant_mode == m].sr_ci95_hi.iloc[0] for m in modes]
    err = [[s - l for s, l in zip(sr, lo)], [h - s for s, h in zip(sr, hi)]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, sr, yerr=err, capsize=5, color=[COLORS[m] for m in modes])
    for i, s in enumerate(sr):
        ax.text(i, s + 0.02, f"{s*100:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in modes])
    ax.set_ylabel("Success rate (LIBERO-Plus pilot)"); ax.set_ylim(0, 1.05)
    ax.set_title("Overall success rate by precision (95% Wilson CI)")
    save(fig, "success_rate_overall")


def fig_by_category(cat):
    modes = modelist(cat)
    cats = sorted(cat.perturbation_category.unique())
    x = np.arange(len(cats)); w = 0.8 / max(len(modes), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, m in enumerate(modes):
        vals = [cat[(cat.quant_mode == m) & (cat.perturbation_category == c)].success_rate.mean() for c in cats]
        ax.bar(x + j * w, vals, w, label=SHORT[m], color=COLORS[m])
    ax.set_xticks(x + w * (len(modes) - 1) / 2)
    ax.set_xticklabels(cats, rotation=25, ha="right")
    ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Success rate by perturbation category")
    ax.legend(ncol=len(modes), fontsize=9)
    save(fig, "success_rate_by_category")


def fig_by_difficulty(diff):
    modes = modelist(diff)
    ds = sorted([d for d in diff.difficulty.unique() if str(d) != "nan"], key=lambda z: str(z))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in modes:
        vals = [diff[(diff.quant_mode == m) & (diff.difficulty.astype(str) == str(d))].success_rate.mean() for d in ds]
        ax.plot([str(d) for d in ds], vals, "-o", label=SHORT[m], color=COLORS[m])
    ax.set_xlabel("Difficulty level"); ax.set_ylabel("Success rate"); ax.set_ylim(0, 1.05)
    ax.set_title("Success rate vs difficulty"); ax.legend()
    save(fig, "success_rate_by_difficulty")


def fig_step_latency(lat):
    modes = modelist(lat)
    x = np.arange(len(modes)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, (stat, off) in enumerate([("mean", -1.5), ("p50", -0.5), ("p95", 0.5), ("p99", 1.5)]):
        ax.bar(x + off * w, [lat[lat.quant_mode == m][stat].iloc[0] for m in modes], w, label=stat)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in modes])
    ax.set_ylabel("Policy step latency (ms)")
    ax.set_title("Policy-step latency (mean/p50/p95/p99) — pilot, per GPU")
    ax.legend()
    save(fig, "step_latency")


def fig_latency_microbench(combined):
    rows = combined["rows"]
    modes = [r["mode"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    e = [r["eager_denoise_p50_ms"] for r in rows]
    ax.bar(np.arange(len(modes)), e, color=[COLORS.get(m, "#888") for m in modes])
    for i, v in enumerate(e):
        ax.text(i, v + 20, f"{v:.0f}", ha="center", fontsize=9)
    ax.axhline(combined["baseline_bf16_denoise_p50_ms"], ls="--", color="k", alpha=0.5, label="bf16 baseline")
    ax.set_xticks(np.arange(len(modes))); ax.set_xticklabels([SHORT.get(m, m) for m in modes], rotation=15)
    ax.set_ylabel("DiT denoise p50 (ms), batch=1, 5 steps")
    ax.set_title("Microbenchmark: quantization does NOT speed up (eager; compile no-op)")
    ax.legend()
    save(fig, "latency_microbench")


def fig_speedup_accuracy(ov, combined):
    # accuracy drop vs bf16 (SR) and latency ratio (microbench)
    modes = [m for m in modelist(ov) if m != "bf16"]
    lat = {r["mode"]: r["eager_speedup_vs_bf16"] for r in combined["rows"]}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in modes:
        d = ov[ov.quant_mode == m].sr_delta_vs_bf16.iloc[0]
        sp = lat.get(m, np.nan)
        ax.scatter(sp, d * 100, s=120, color=COLORS[m], label=SHORT[m], zorder=3)
        ax.annotate(SHORT[m], (sp, d * 100), textcoords="offset points", xytext=(6, 6))
    ax.axvline(1.0, ls="--", color="k", alpha=0.4); ax.axhline(0, ls="--", color="k", alpha=0.4)
    ax.set_xlabel("Latency speedup vs BF16 (>1 = faster)")
    ax.set_ylabel("Success-rate change vs BF16 (pp)")
    ax.set_title("Speedup vs accuracy change (real quant modes)")
    save(fig, "speedup_and_accuracy_drop")


def fig_pareto(ov, lat):
    modes = modelist(ov)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in modes:
        sr = ov[ov.quant_mode == m].success_rate.iloc[0]
        p95 = lat[lat.quant_mode == m]["p95"].iloc[0] if len(lat[lat.quant_mode == m]) else np.nan
        ax.scatter(p95, sr, s=140, color=COLORS[m], zorder=3)
        ax.annotate(SHORT[m], (p95, sr), textcoords="offset points", xytext=(7, 5))
    ax.set_xlabel("p95 policy step latency (ms)"); ax.set_ylabel("Success rate")
    ax.set_title("Success–latency Pareto")
    save(fig, "success_latency_pareto")


def fig_memory(ov):
    modes = modelist(ov)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(np.arange(len(modes)), [ov[ov.quant_mode == m].peak_memory_mb_median.iloc[0] for m in modes],
           color=[COLORS[m] for m in modes])
    ax.set_xticks(np.arange(len(modes))); ax.set_xticklabels([SHORT[m] for m in modes])
    ax.set_ylabel("Peak GPU memory (MB, per episode)")
    ax.set_title("Peak memory by precision (clean per-process)")
    save(fig, "memory_by_precision")


def fig_latency_cdf(tag="pilot"):
    import glob
    files = glob.glob(f"{ec.RAW}/inference_steps_{tag}*.csv")
    if not files:
        return
    infer = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in modelist(infer):
        s = np.sort(infer[infer.quant_mode == m]["total_policy_step_ms"].values)
        if len(s) == 0:
            continue
        ax.plot(s, np.linspace(0, 1, len(s)), label=SHORT[m], color=COLORS[m])
    ax.set_xlabel("Policy step latency (ms)"); ax.set_ylabel("CDF")
    ax.set_title("Policy-step latency CDF"); ax.legend()
    save(fig, "latency_cdf")


def main():
    S = ec.SUMM
    if os.path.exists(f"{S}/overall_summary.csv"):
        ov = pd.read_csv(f"{S}/overall_summary.csv")
        fig_success_overall(ov)
        fig_memory(ov)
        combined = json.load(open(f"{S}/latency_microbench_combined.json"))
        fig_speedup_accuracy(ov, combined)
        if os.path.exists(f"{S}/step_latency_by_mode.csv"):
            lat = pd.read_csv(f"{S}/step_latency_by_mode.csv")
            fig_step_latency(lat); fig_pareto(ov, lat)
    if os.path.exists(f"{S}/category_summary.csv"):
        fig_by_category(pd.read_csv(f"{S}/category_summary.csv"))
    if os.path.exists(f"{S}/difficulty_summary.csv"):
        fig_by_difficulty(pd.read_csv(f"{S}/difficulty_summary.csv"))
    if os.path.exists(f"{S}/latency_microbench_combined.json"):
        fig_latency_microbench(json.load(open(f"{S}/latency_microbench_combined.json")))
    fig_latency_cdf()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()

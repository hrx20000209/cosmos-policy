"""Refresh Phase 2A derived tables without rerunning policy inference."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.signal_validation.run_closed_loop_phase2a import (
    MODES,
    build_rows,
    load_supplement_traces,
    summarize_latency,
    summarize_outcomes,
)

ARTIFACT = Path("reports/artifacts/libero_pro_closed_loop_phase2a.json")


def main() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    records = payload["records"]
    payload["outcomes"] = summarize_outcomes(records)
    trace_path = Path(payload["raw_trace_path"])
    supplement_rows = [row for row in build_rows() if row["init_state_index"] in {1, 2}]
    supplement_traces = load_supplement_traces(trace_path, supplement_rows)
    mode_trace_counts = {
        mode: sum(1 for row in supplement_traces if row.get("configuration") == mode)
        for mode in MODES
    }
    visual_by_source = {
        source: sum(
            1
            for row in supplement_traces
            if row.get("extra", {}).get("visual_input_mode") == source
        )
        for source in ("fresh", "predicted", "cache")
    }
    visual_by_configuration = {
        mode: {
            source: sum(
                1
                for row in supplement_traces
                if row.get("configuration") == mode
                and row.get("extra", {}).get("visual_input_mode") == source
            )
            for source in ("fresh", "predicted", "cache")
        }
        for mode in MODES
    }
    shadow = payload["shadow_fresh"]["metrics"]["overall"]
    spec_median = shadow["predicted"]["mean_step_l2"]["median"]
    cache_median = shadow["cache"]["mean_step_l2"]["median"]
    spec_transition = payload["outcomes"]["transitions"]["fresh_vs_speculative"]
    conditions = {
        "no_fresh_success_to_speculative_failure": (
            spec_transition["counts"]["fresh_success_other_failure"] == 0
        ),
        "speculative_drift_lower_than_cache_median_mean_step_l2": (
            spec_median is not None and cache_median is not None and spec_median < cache_median
        ),
        "supplement_has_non_overlapping_stage_timers": bool(supplement_traces),
        "supplement_speculative_visual_refresh_is_about_half": (
            mode_trace_counts["alternate_speculative"] > 0
            and 0.45
            <= visual_by_configuration["alternate_speculative"]["predicted"]
            / mode_trace_counts["alternate_speculative"]
            <= 0.55
        ),
    }
    payload["latency"] = {
        **payload["latency"],
        "supplement_init1_init2_non_overlapping": summarize_latency(supplement_traces),
        "supplement_trace_count_by_mode": mode_trace_counts,
        "supplement_visual_trace_count_by_source": visual_by_source,
        "supplement_visual_trace_count_by_configuration": visual_by_configuration,
    }
    payload["go_gate"] = {
        "conditions": conditions,
        "fresh_successes": payload["outcomes"]["fresh"]["successes"],
        "speculative_median_shadow_mean_step_l2": spec_median,
        "cache_median_shadow_mean_step_l2": cache_median,
        "recommendation": "GO" if all(conditions.values()) else "NO-GO-CANDIDATE",
    }
    payload["derived_tables_refreshed_without_new_inference"] = True
    ARTIFACT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(ARTIFACT), "recommendation": payload["go_gate"]["recommendation"]}))


if __name__ == "__main__":
    main()

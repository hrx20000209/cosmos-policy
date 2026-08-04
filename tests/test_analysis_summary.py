from __future__ import annotations

import json

from analysis.summarize_results import summarize_run


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_summarize_run_excludes_configured_warmup_and_reports_steps(tmp_path) -> None:
    _write_jsonl(
        tmp_path / "episodes.jsonl",
        [
            {
                "task_name": "task",
                "success": True,
                "episode_wall_clock_time_s": 3.0,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "inference_trace.jsonl",
        [
            {
                "total_policy_request_latency_ms": 1000.0,
                "dit_denoising_latency_ms": 900.0,
                "future_state_decode_latency_ms": 50.0,
                "selected_denoising_steps": 3,
                "denoiser_forward_count": 3,
            },
            {
                "total_policy_request_latency_ms": 10.0,
                "dit_denoising_latency_ms": 6.0,
                "future_state_decode_latency_ms": 2.0,
                "selected_denoising_steps": 3,
                "denoiser_forward_count": 3,
            },
        ],
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"latency": {"warmup_requests_excluded": 1}}),
        encoding="utf-8",
    )

    result = summarize_run(tmp_path)

    assert result["warmup_requests_excluded"] == 1
    assert result["measured_requests"] == 1
    assert result["policy_latency_mean_ms"] == 10.0
    assert result["dit_latency_mean_ms"] == 6.0
    assert result["future_decode_latency_mean_ms"] == 2.0
    assert result["denoising_steps"] == 3
    assert result["denoiser_forward_count"] == 3
    assert result["aggregate_success_wilson_95_ci"][0] < 1.0
    assert result["aggregate_success_wilson_95_ci"][1] == 1.0

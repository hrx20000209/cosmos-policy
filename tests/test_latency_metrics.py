from runtime.runtime_metrics import percentiles, summarize_traces


def test_percentiles_and_warmup_exclusion():
    values = list(range(1, 101))
    result = percentiles(values)
    assert result["p50"] == 50.5
    traces = [{"total_policy_request_latency_ms": value, "denoiser_forward_count": 2} for value in values]
    summary = summarize_traces(traces, warmup=10)
    assert summary["measured_requests"] == 90
    assert summary["policy_latency_ms"]["mean"] == 55.5
    assert summary["denoiser_forward_count"] == 180


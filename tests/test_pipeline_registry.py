from runtime.async_pipeline import ActionFirstPipeline, InferenceRequest


def request(step):
    return InferenceRequest.create(step, "episode", step)


def test_completed_sync_requests_are_not_counted_as_cancelled():
    pipeline = ActionFirstPipeline("sync_baseline")
    try:
        pipeline.submit(request(0), lambda: "actions", lambda output: "future")
        pipeline.submit(request(1), lambda: "actions", lambda output: "future")
        assert pipeline.registry.cancelled_count == 0
    finally:
        pipeline.close()


def test_new_request_cancels_only_a_still_pending_request():
    pipeline = ActionFirstPipeline("action_first_async_decode")
    first = request(0)
    second = request(1)
    pipeline.registry.register(first)
    pipeline.registry.register(second)
    try:
        assert pipeline.registry.cancelled_count == 1
        assert not pipeline.registry.is_valid(first)
        assert pipeline.registry.is_valid(second)
    finally:
        pipeline.close()

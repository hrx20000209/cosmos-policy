from runtime.runtime_metrics import ResourceSampler


def test_resource_sampler_uses_first_visible_physical_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,0")
    assert ResourceSampler._default_gpu_ids() == "5"


def test_resource_sampler_defaults_to_gpu_zero(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert ResourceSampler._default_gpu_ids() == "0"

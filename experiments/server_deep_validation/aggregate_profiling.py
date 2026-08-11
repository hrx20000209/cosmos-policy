"""Summarize non-destructive 1/2/3-worker GPU density profiling."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/data/rxhuang/wam_server_deep_validation/profiling")
rows = []
for label, worker in (("1_worker", "worker_single"), ("2_workers", "worker0"), ("2_workers", "worker2"), ("3_workers", "worker0"), ("3_workers", "worker1"), ("3_workers", "worker2")):
    for path in (ROOT / worker).glob("summary_shard*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"condition": label, "worker": worker, "summary": value})
result = {
    "experiment": "worker_density_profile",
    "gpu": 3,
    "single_worker_reference": "GPU4 worker_single completed 18 requests",
    "conditions": rows,
    "interpretation": {
        "one_worker": "completed on GPU4",
        "two_workers": "both shared-GPU workers hit CUDA OOM on GPU3",
        "three_workers": "one worker hit OOM during load and the remaining pair also failed under the 3-way allocation",
        "formal_latency_claim": False,
    },
}
out = Path("/home/rxhuang/Projects/cosmos-policy/reports/server_deep_validation/manifests/worker_density_profile.json")
out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))

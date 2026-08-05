"""Per-request stage timing for the live SO101 Cosmos policy server.

``cosmos_policy/scripts/profile_so101_latency.py`` measures the same stages
offline against a dataset.  This module makes the identical instrumentation
usable from the deployed gRPC server, so every real action chunk carries its own
breakdown instead of only having an aggregate measured on replayed frames.

Stages
------
  t5_lookup       text embedding fetch from the precomputed cache
  image_preproc   resize/normalise the 3 views into the 11-slot Cosmos layout
  vae_encode      Wan2.1 tokenizer encode/decode of the conditioning frames
  dit_denoise     DiT forward passes -- one per denoising step
  sampler_other   generate_samples_from_batch minus vae_encode minus dit_denoise
  action_decode   latent -> action chunk + unnormalisation

The server adds ``obs_prep``, ``safety_filter`` and ``serialize`` around these.

Accumulation is thread-local: the gRPC server uses a thread pool, and the
patched callables run in whichever thread is serving the request, so each
request accumulates into its own bucket without a lock.

All stage boundaries are ``torch.cuda.synchronize()``-d, so GPU work is charged
to the stage that launched it rather than to whoever happens to sync later.
That costs a little wall-clock accuracy on the total but is the only way to get
a truthful split; ``profile_enabled`` lets callers turn it off.
"""

import threading
import time
from collections import defaultdict

import torch

STAGE_ORDER = [
    "obs_prep",
    "t5_lookup",
    "image_preproc",
    "vae_encode",
    "dit_denoise",
    "sampler_other",
    "action_decode",
    "safety_filter",
    "serialize",
]

_local = threading.local()
_patched = False


def _bucket() -> dict:
    if not hasattr(_local, "cur"):
        _local.cur = defaultdict(float)
        _local.dit_calls = 0
    return _local.cur


def add(stage_name: str, ms: float) -> None:
    _bucket()[stage_name] += ms


def reset() -> None:
    _local.cur = defaultdict(float)
    _local.dit_calls = 0


def collect() -> dict:
    """Return this thread's accumulated stages and clear them."""
    cur = dict(_bucket())
    cur["dit_calls"] = float(getattr(_local, "dit_calls", 0))
    reset()
    return cur


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class stage:
    """Context manager timing a stage, GPU-synchronised at both ends."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        add(self.name, (time.perf_counter() - self.t0) * 1000)
        return False


def instrument(model) -> None:
    """Monkeypatch the inference path so each stage reports its own time.

    Idempotent: patching twice would double-count, so later calls are no-ops.
    """
    global _patched
    if _patched:
        return
    _patched = True

    from cosmos_policy.experiments.robot import cosmos_utils

    # --- VAE tokenizer ---
    tok = model.tokenizer
    for meth in ("encode", "decode"):
        if hasattr(tok, meth):
            orig = getattr(tok, meth)

            def wrapper(*a, _orig=orig, **kw):
                with stage("vae_encode"):
                    return _orig(*a, **kw)

            setattr(tok, meth, wrapper)

    # --- DiT forward passes (one per denoising step) ---
    cls = type(model)
    orig_x0 = cls.get_x0_fn_from_batch

    def _wrap_x0(fn):
        def timed_fn(*fa, **fkw):
            sync()
            t0 = time.perf_counter()
            out = fn(*fa, **fkw)
            sync()
            add("dit_denoise", (time.perf_counter() - t0) * 1000)
            _local.dit_calls = getattr(_local, "dit_calls", 0) + 1
            return out

        return timed_fn

    def patched_x0(self, *a, **kw):
        res = orig_x0(self, *a, **kw)
        # policy_text2world_model returns (x0_fn, orig_clean_latent_frames);
        # the base class returns just x0_fn.
        if isinstance(res, tuple):
            return (_wrap_x0(res[0]), *res[1:])
        return _wrap_x0(res)

    cls.get_x0_fn_from_batch = patched_x0

    # --- whole sampling call, to derive the residual ---
    orig_gen = cls.generate_samples_from_batch

    def patched_gen(self, *a, **kw):
        sync()
        t0 = time.perf_counter()
        out = orig_gen(self, *a, **kw)
        sync()
        total = (time.perf_counter() - t0) * 1000
        cur = _bucket()
        other = total - cur.get("dit_denoise", 0.0) - cur.get("vae_encode", 0.0)
        add("sampler_other", max(other, 0.0))
        return out

    cls.generate_samples_from_batch = patched_gen

    # --- module-level helpers ---
    for name, label in (
        ("prepare_images_for_model", "image_preproc"),
        ("get_t5_embedding_from_cache", "t5_lookup"),
        ("unnormalize_actions", "action_decode"),
    ):
        if hasattr(cosmos_utils, name):
            orig_fn = getattr(cosmos_utils, name)

            def fn_wrapper(*a, _orig=orig_fn, _label=label, **kw):
                with stage(_label):
                    return _orig(*a, **kw)

            setattr(cosmos_utils, name, fn_wrapper)

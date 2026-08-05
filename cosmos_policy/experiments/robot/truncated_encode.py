"""Encode only the conditioning slots, pad the rest of the latent with zeros.

The SO101 policy input is 41 raw frames -> 11 latent slots, but the model
conditions on only the first ``min_num_conditional_frames`` of them (5 here:
blank, proprio, left wrist, right wrist, primary).  Slots 5..10 -- action,
future proprio, future wrist x2, future primary, value -- are generated from
noise, so whatever the VAE produced for them is thrown away.

The Wan2.1 tokenizer is causal (``CausalConv3d``), so latent[k] depends only on
frames <= k.  Encoding just the pixel prefix that maps to the conditioning slots
therefore yields *bit-identical* latents for those slots, verified in
``cosmos_policy/scripts/verify_truncated_vae_encode.py``:

    前 5 个槽位 max|diff| = 0.000e+00
    max|Δaction| = 0.067 deg   (vs a servo tracking error of 1.4-3.4 deg)
    830 ms -> 502 ms end to end (-40%)

The residual action difference is not exactly zero because the discarded slots
still leak a little through the noised initialisation (``x_noisy = gt_frames +
sigma * noise``), but it is one to two orders of magnitude below what the
hardware can track.

Only valid while future-state/value prediction is off: those paths consume the
slots this optimisation stops encoding.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def install(model, num_conditional_frames: int | None = None) -> dict:
    """Patch ``model.tokenizer.encode`` to encode only the conditioning prefix.

    Returns a dict describing what was applied, for logging.  Idempotent.
    """
    global _patched
    if _patched:
        return {"applied": False, "reason": "already patched"}

    tok = model.tokenizer
    if num_conditional_frames is None:
        num_conditional_frames = int(getattr(model.config, "min_num_conditional_frames", 0))
    max_cond = int(getattr(model.config, "max_num_conditional_frames", num_conditional_frames))
    if num_conditional_frames <= 0:
        return {"applied": False, "reason": "min_num_conditional_frames unavailable"}
    if max_cond != num_conditional_frames:
        # A variable conditioning window would make the safe prefix vary per call.
        return {"applied": False, "reason": f"min({num_conditional_frames}) != max({max_cond}) conditional frames"}

    state_t = int(model.config.state_t)
    total_pixel_frames = int(tok.get_pixel_num_frames(state_t))
    prefix_pixel_frames = int(tok.get_pixel_num_frames(num_conditional_frames))
    if prefix_pixel_frames >= total_pixel_frames:
        return {"applied": False, "reason": "prefix covers the whole sequence"}

    orig_encode = tok.encode

    def truncated_encode(x, *a, **kw):
        # Only touch the full policy input; the tokenizer is used elsewhere
        # (e.g. decoding) with different shapes.
        if not (torch.is_tensor(x) and x.dim() == 5 and x.shape[2] == total_pixel_frames):
            return orig_encode(x, *a, **kw)
        out = orig_encode(x[:, :, :prefix_pixel_frames].contiguous(), *a, **kw)
        was_seq = isinstance(out, (tuple, list))
        lat = out[0] if was_seq else out
        missing = state_t - lat.shape[2]
        if missing > 0:
            pad = torch.zeros(
                lat.shape[0], lat.shape[1], missing, lat.shape[3], lat.shape[4],
                device=lat.device, dtype=lat.dtype,
            )
            lat = torch.cat([lat, pad], dim=2)
        return (lat, *out[1:]) if was_seq else lat

    tok.encode = truncated_encode
    _patched = True
    info = {
        "applied": True,
        "state_t": state_t,
        "num_conditional_frames": num_conditional_frames,
        "pixel_frames_encoded": prefix_pixel_frames,
        "pixel_frames_total": total_pixel_frames,
        "fraction_encoded": prefix_pixel_frames / total_pixel_frames,
    }
    logger.warning(
        "Truncated VAE encode ON: %d/%d pixel frames (%.0f%%), padding %d latent slots with zeros",
        prefix_pixel_frames, total_pixel_frames, 100 * prefix_pixel_frames / total_pixel_frames,
        state_t - num_conditional_frames,
    )
    return info

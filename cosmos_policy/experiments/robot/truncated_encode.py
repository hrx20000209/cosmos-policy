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


def install(
    model,
    num_conditional_frames: int | None = None,
    drop_slot: int = -1,
    fill: str = "zero",
) -> dict:
    """Patch ``model.tokenizer.encode`` to encode only the conditioning prefix.

    ``drop_slot`` additionally stops encoding one conditioning slot and splices
    something into its place in the latent, which is how a camera view can be
    dropped at all: the pixel-space route is blocked because ``state_t`` is fixed
    and a shortened video is rejected, and zero-padding it back to full length
    saves nothing since the VAE still encodes every frame.

    ``fill`` chooses what goes in the gap -- ``zero``, ``copy_next`` (the slot
    that follows it), or ``copy_last`` (the primary view).  Offline on 8 real
    observations, dropping slot 2 (the "right" third-person view) cost
    mean 0.72-0.87 deg / max 3.4-5.4 deg of action deviation for 40 ms.

    Note the tokenizer is causal, so removing a *middle* slot also shifts the
    slots after it and changes their latents -- the spliced result is not the
    latent the model was trained on. Only the last conditioning slot could be
    dropped cleanly, and that is the primary view.

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

    # Pixel-frame span of a conditioning slot: slot 0 is the single leading
    # frame, slot k>=1 covers [get_pixel_num_frames(k), get_pixel_num_frames(k+1)).
    drop_lo = drop_hi = -1
    if drop_slot >= 0:
        if not 0 <= drop_slot < num_conditional_frames:
            return {"applied": False, "reason": f"drop_slot {drop_slot} outside 0..{num_conditional_frames - 1}"}
        drop_lo = 0 if drop_slot == 0 else int(tok.get_pixel_num_frames(drop_slot))
        drop_hi = 1 if drop_slot == 0 else int(tok.get_pixel_num_frames(drop_slot + 1))

    def truncated_encode(x, *a, **kw):
        # Only touch the full policy input; the tokenizer is used elsewhere
        # (e.g. decoding) with different shapes.
        if not (torch.is_tensor(x) and x.dim() == 5 and x.shape[2] == total_pixel_frames):
            return orig_encode(x, *a, **kw)
        if drop_lo >= 0:
            keep = torch.cat([x[:, :, :drop_lo], x[:, :, drop_hi:prefix_pixel_frames]], dim=2).contiguous()
        else:
            keep = x[:, :, :prefix_pixel_frames].contiguous()
        out = orig_encode(keep, *a, **kw)
        was_seq = isinstance(out, (tuple, list))
        lat = out[0] if was_seq else out
        if drop_lo >= 0:
            if fill == "zero":
                gap = torch.zeros_like(lat[:, :, :1])
            elif fill == "copy_next":
                gap = lat[:, :, drop_slot : drop_slot + 1].clone()
            elif fill == "copy_last":
                gap = lat[:, :, -1:].clone()
            else:
                raise ValueError(f"unknown fill {fill!r}")
            lat = torch.cat([lat[:, :, :drop_slot], gap, lat[:, :, drop_slot:]], dim=2)
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
        "drop_slot": drop_slot,
        "fill": fill if drop_slot >= 0 else None,
        "state_t": state_t,
        "num_conditional_frames": num_conditional_frames,
        "pixel_frames_encoded": prefix_pixel_frames - (drop_hi - drop_lo if drop_lo >= 0 else 0),
        "pixel_frames_total": total_pixel_frames,
        "fraction_encoded": (prefix_pixel_frames - (drop_hi - drop_lo if drop_lo >= 0 else 0)) / total_pixel_frames,
    }
    logger.warning(
        "Truncated VAE encode ON: %d/%d pixel frames (%.0f%%), padding %d latent slots with zeros",
        prefix_pixel_frames, total_pixel_frames, 100 * prefix_pixel_frames / total_pixel_frames,
        state_t - num_conditional_frames,
    )
    return info

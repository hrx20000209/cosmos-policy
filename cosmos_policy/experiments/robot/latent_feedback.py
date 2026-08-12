"""Feed the model's predicted future latents back in as the next conditioning.

The policy input is 11 latent slots (state_t=11, min_num_conditional_frames=5),
and for the aloha layout this checkpoint uses they are:

    0 blank | 1 curr proprio | 2 curr left wrist | 3 curr right wrist
    4 curr primary | 5 action | 6 future proprio | 7 future left wrist
    8 future right wrist | 9 future primary | 10 value

Slots 0-4 are conditioning; 5-10 are generated from noise. Two facts make the
feedback loop cheap and well-posed:

* ``generate_samples_from_batch`` denoises the whole sequence in one pass and
  returns all 11 slots. Slots 7-9 -- the predicted next views -- are therefore
  already computed by every inference we run today and simply discarded.
  ``generate_future_state_and_value_in_parallel`` does not gate that compute; it
  only decides whether they get VAE-*decoded* into images, which is the ~994 ms
  we measured and is not needed here.
* The future slots are positionally the same objects as the conditioning slots,
  one step ahead. So 7->2, 8->3, 9->4 is a straight substitution, not a
  reinterpretation.

What that buys: on a fed-back step the tokenizer is never called, so the entire
VAE encode goes away -- 269 ms of a 525 ms inference (51%).

What it does *not* imagine: proprio. It is injected into slot 1 from
``data_batch["proprio"]`` rather than encoded from pixels (the image sequence
holds only a blank placeholder there), so it keeps coming from the real servo
reading on every step. Only the scene *imagery* is predicted. The robot's own
state cannot drift, which bounds this far more tightly than an ordinary
autoregressive rollout.

Slots 0 and 1 still need their true encoded values, and zeros are not it -- the
VAE does not map a black frame to zero. They are constant across the whole run
(blank image, blank placeholder), so one real encode at the start is captured
and reused forever.

The open question this cannot answer by construction is exposure bias: the model
was trained conditioning on *encoded real* frames, and a generated latent is its
own output. Whether that shift is tolerable, and for how many consecutive steps,
is measured in ``verify_latent_feedback.py`` -- not assumed here.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# aloha layout, state_t=11 / min_num_conditional_frames=5
IMAGE_SLOTS = (2, 3, 4)
FUTURE_IMAGE_SLOTS = (7, 8, 9)
STATIC_SLOTS = (0, 1)


class LatentFeedback:
    """Patches the model in place; one instance per process."""

    def __init__(self, model, image_slots=IMAGE_SLOTS, future_slots=FUTURE_IMAGE_SLOTS,
                 measure_all=False):
        self.model = model
        self.image_slots = tuple(image_slots)
        self.future_slots = tuple(future_slots)
        self.state_t = int(model.config.state_t)
        # Diagnostic mode: keep encoding the real cameras on fed-back steps too,
        # purely to score the prediction. It throws away the entire speed-up --
        # the point of feedback is to skip that encode -- but it is the only way
        # to observe drift past the first imagined step, because the residual is
        # defined against a real encode that a pure feedback run never performs.
        # It also removes a selection effect: with a scheduler, the steps that
        # get a residual are exactly the ones it decided to ground, so residuals
        # are sampled where drift was already suspected.
        self.measure_all = bool(measure_all)
        self.consecutive = 0
        self._static: torch.Tensor | None = None   # slots 0..1, constant for the run
        self._cached: torch.Tensor | None = None   # last generated future image slots
        # The prediction awaiting verification: whatever was last generated,
        # kept until a real encode arrives to be compared against it.
        self._pending: torch.Tensor | None = None
        self.last_residual: dict | None = None
        self.last_residual_logged: dict | None = None
        self.last_value: float | None = None
        self._armed = False
        self._installed = False
        self.n_served = 0
        self.n_encoded = 0

    # ---------- installation ----------

    def install(self) -> dict:
        if self._installed:
            return {"applied": False, "reason": "already installed"}
        if self.state_t != 11:
            return {"applied": False, "reason": f"layout only verified for state_t=11, got {self.state_t}"}

        tok = self.model.tokenizer
        orig_encode = tok.encode
        total_pixel_frames = int(tok.get_pixel_num_frames(self.state_t))

        def encode(x, *a, **kw):
            if not (torch.is_tensor(x) and x.dim() == 5 and x.shape[2] == total_pixel_frames):
                return orig_encode(x, *a, **kw)
            if self._armed and self._cached is not None and self._static is not None:
                self.n_served += 1
                self.consecutive += 1
                if self.measure_all:
                    real = orig_encode(x, *a, **kw)
                    self._score(real[0] if isinstance(real, (tuple, list)) else real)
                return self._build()
            out = orig_encode(x, *a, **kw)
            lat = out[0] if isinstance(out, (tuple, list)) else out
            # A real encode is the only moment the previous prediction can be
            # scored, and both tensors are already in memory -- so the check is
            # free, and it happens on exactly the steps that were going to pay
            # for an encode anyway. Comparing in pixel space instead would mean
            # VAE-decoding the prediction, which costs 994 ms against the 269 ms
            # the whole scheme saves.
            self._score(lat)
            self.consecutive = 0
            if self._static is None:
                # Slots 0 and 1 are a blank frame and a blank placeholder; they
                # do not change for the life of the run.
                self._static = lat[:, :, : max(STATIC_SLOTS) + 1].detach().clone()
            self.n_encoded += 1
            return out

        orig_sample = self.model.generate_samples_from_batch

        def sample(*a, **kw):
            out = orig_sample(*a, **kw)
            lat = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(lat) and lat.dim() == 5 and lat.shape[2] == self.state_t:
                self._cached = lat[:, :, self.future_slots].detach().clone()
                self._pending = self._cached
                # The value head lives in the last slot and is a mean over that
                # latent frame -- free, and never read until now.
                v = lat[:, :, -1].float().mean().item()
                self.last_value = min(1.0, max(0.0, (v + 1.0) / 2.0))
            return out

        tok.encode = encode
        self.model.generate_samples_from_batch = sample
        self._installed = True
        logger.warning(
            "Latent feedback installed: future slots %s -> conditioning slots %s",
            self.future_slots, self.image_slots,
        )
        return {"applied": True, "image_slots": self.image_slots, "future_slots": self.future_slots}

    def _score(self, real_lat: torch.Tensor) -> None:
        """Residual between the last prediction and the real latent it predicted.

        Reported normalised per view, because the raw magnitude carries a fixed
        offset: the future slot predicts a horizon set by training, not by our
        inference interval, so it is never expected to reach zero. Only changes
        in the residual are meaningful, which is why the scheduler that consumes
        it calibrates against the run's own distribution rather than a constant.
        """
        if self._pending is None:
            return
        pred = self._pending
        self._pending = None
        try:
            real = real_lat[:, :, self.image_slots].float()
            p = pred.float()
            if real.shape != p.shape:
                return
            num = (real - p).abs().mean(dim=(0, 1, 3, 4))
            den = real.abs().mean(dim=(0, 1, 3, 4)).clamp_min(1e-6)
            rel = (num / den).tolist()
            self.last_residual = {
                "consecutive": self.consecutive,
                "per_slot": [round(v, 4) for v in rel],
                "mean": round(float(sum(rel) / len(rel)), 4),
                "max": round(float(max(rel)), 4),
            }
        except Exception:  # noqa: BLE001 - a scoring failure must never break inference
            self.last_residual = None

    # ---------- control ----------

    @property
    def ready(self) -> bool:
        """True once a real encode and a generation have both happened."""
        return self._cached is not None and self._static is not None

    def arm(self) -> bool:
        """Serve the next encode from the cached prediction. Returns whether it will."""
        self._armed = self.ready
        return self._armed

    def disarm(self) -> None:
        self._armed = False

    def reset(self) -> None:
        self._cached = None
        self._armed = False

    # ---------- internals ----------

    def _build(self) -> torch.Tensor:
        st = self._static
        B, C, _, H, W = st.shape
        lat = torch.zeros(B, C, self.state_t, H, W, device=st.device, dtype=st.dtype)
        lat[:, :, : st.shape[2]] = st
        for dst, src in zip(self.image_slots, range(self._cached.shape[2])):
            lat[:, :, dst] = self._cached[:, :, src]
        return lat

"""Decide, online, how much sensing the run can afford to skip.

Latent feedback (``latent_feedback.py``) trades information freshness for
compute: an imagined step costs 249 ms instead of 521 ms but conditions the
policy on its own prediction rather than the cameras. Today that trade is made
open-loop -- a fixed k imagined steps per real one -- which means the schedule
is equally aggressive when the scene is predictable and when it is not.

This closes that loop using a signal that is already free. On every real encode
the tokenizer produces the true latents for slots 2-4, and the previous
inference already produced its prediction of them in slots 7-9. Subtracting the
two costs nothing and happens only on steps that were paying for an encode
anyway. Comparing in pixel space instead -- optical flow on decoded frames, say
-- would require VAE-decoding the prediction at 994 ms, which is four times the
saving the whole scheme produces.

Two signals, two jobs:

``residual``  how far the imagined world has drifted from the real one.
    Drives k. Low residual means prediction is tracking, so more steps can be
    imagined; high residual means the run has to spend observations.

``intent``    the displacement the policy is asking for, mean |action - proprio|.
    Catches the failure residual cannot: in an absorbing state the arm stops,
    the scene stops, and the residual therefore goes *quiet* precisely when
    something is wrong -- "predicting well" and "stuck" are indistinguishable
    from the residual alone. Requested displacement separates them, and it was
    already measured to: when the policy collapsed on hardware its p95 intent
    fell from 14.94 deg to 1.04 deg.

    Deliberately *not* the model's value head. Cosmos happens to predict one in
    latent slot 10, and reading it is free, but a world-action model in general
    does not have one. Intent needs only an action chunk and a proprio reading,
    which every such model and every arm provides, so the scheduler stays
    portable. The value head is still written to the trace as a diagnostic; it
    drives nothing.

Thresholds are calibrated against the run's own residual distribution, not set
as constants. The future slot predicts a horizon fixed by training rather than
by our inference interval, so the residual has a systematic offset that no
constant can be chosen for in advance; and the one earlier attempt in this
project to hard-code a threshold from a single run (a flat stall load of 95)
fired on 36.9% of normal operation.

The band is re-derived from a trailing window, *not* raised-only. The raise-only
rule that protects the stall thresholds inverts here and has to be left out: a
band widened to admit every large residual before testing against it makes the
back-off test false by construction, so k could only ever climb. Caught by the
unit self-check, which is why one is worth having on a controller this small.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveSensing:
    def __init__(
        self,
        k_min: int = 0,
        k_max: int = 4,
        k_init: int = 1,
        calibrate_n: int = 6,
        lo_quantile: float = 0.35,
        hi_quantile: float = 0.75,
        intent_flat_secs: float = 15.0,
        intent_quantile: float = 0.15,
    ):
        self.k_min, self.k_max = k_min, k_max
        self.k = int(np.clip(k_init, k_min, k_max))
        self.calibrate_n = calibrate_n
        self.lo_q, self.hi_q = lo_quantile, hi_quantile
        self.intent_flat_secs = intent_flat_secs
        self.intent_q = intent_quantile

        self._res: list[float] = []
        self.lo: float | None = None
        self.hi: float | None = None
        self._intent: list[float] = []
        self._intent_lo: float | None = None
        self._moving_since: float | None = None
        self.history: list[dict] = []
        self.n_up = self.n_down = self.n_stall = 0

    def observe(self, residual: dict | None, intent: float | None, now: float) -> None:
        """Called after every inference. Only real-encode steps carry a residual."""
        stuck = False
        if intent is not None:
            self._intent.append(float(intent))
            # Threshold from the run's own intent distribution: absolute scale
            # depends on the task, the arm and the action gain, so no constant
            # transfers between setups.
            if len(self._intent) >= self.calibrate_n:
                self._intent_lo = float(np.quantile(self._intent[-60:], self.intent_q))
            if self._intent_lo is None or intent > self._intent_lo:
                self._moving_since = now
            elif self._moving_since is not None and now - self._moving_since >= self.intent_flat_secs:
                stuck = True

        if residual is None:
            if stuck:
                self._force_sensing(now, None, "intent collapsed")
            return

        r = float(residual["mean"])

        if self.lo is None:
            self._res.append(r)
            if len(self._res) < self.calibrate_n:
                return
            self._recalibrate()
            logger.warning(
                "Adaptive sensing calibrated on %d residuals: lo=%.4f hi=%.4f (k stays %d)",
                len(self._res), self.lo, self.hi, self.k,
            )
            return

        # Compare against the band derived from *past* residuals, then fold this
        # one in. Doing it the other way round is not a subtlety: widening `hi`
        # to include r before testing `r > hi` makes the test false by
        # construction, and k can then only ever rise. The raise-only rule that
        # protects the stall thresholds inverts here -- there it prevented false
        # positives, here it would guarantee false negatives.
        self._res.append(r)
        if len(self._res) % self.calibrate_n == 0:
            self._recalibrate()

        if stuck:
            self._force_sensing(now, r, "intent collapsed")
            return

        prev = self.k
        if r < self.lo and self.k < self.k_max:
            self.k += 1
            self.n_up += 1
        elif r > self.hi and self.k > self.k_min:
            self.k -= 1
            self.n_down += 1
        if self.k != prev:
            self.history.append({"t": round(now, 2), "residual": round(r, 4),
                                 "k": self.k, "intent": None if intent is None else round(intent, 3)})

    def _recalibrate(self) -> None:
        """Re-derive the band from a trailing window of residuals.

        A trailing window rather than the whole run because the residual's
        scale is phase-dependent -- gross free-space motion and a contact-rich
        placement do not predict equally well -- so a band fitted to the whole
        history would be wrong for both.
        """
        w = self._res[-40:]
        self.lo = float(np.quantile(w, self.lo_q))
        self.hi = float(np.quantile(w, self.hi_q))

    def _force_sensing(self, now: float, r: float | None, why: str) -> None:
        """Task progress has stalled: stop imagining and look at the world.

        Withholding observations is exactly what an absorbing state feeds on --
        an imagined step cannot deliver the scene change that would move the
        policy off its fixed point -- so the response to suspected stalling is
        to spend observations, not save them.
        """
        self.n_stall += 1
        self._moving_since = now  # re-arm so this fires periodically, not every step
        if self.k != self.k_min:
            self.history.append({"t": round(now, 2), "residual": None if r is None else round(r, 4),
                                 "k": self.k_min, "reason": why})
            logger.warning("Adaptive sensing: %s -> k %d->%d (full sensing)", why, self.k, self.k_min)
            self.k = self.k_min

    def stats(self) -> dict:
        return {
            "k_final": self.k,
            "lo": None if self.lo is None else round(self.lo, 4),
            "hi": None if self.hi is None else round(self.hi, 4),
            "n_residuals": len(self._res),
            "residual_p50": None if not self._res else round(float(np.median(self._res)), 4),
            "residual_p95": None if not self._res else round(float(np.quantile(self._res, 0.95)), 4),
            "intent_lo": None if self._intent_lo is None else round(self._intent_lo, 3),
            "intent_p50": None if not self._intent else round(float(np.median(self._intent)), 3),
            "n_k_up": self.n_up, "n_k_down": self.n_down, "n_stall_forced": self.n_stall,
            "changes": self.history[-20:],
        }

# Sensitivity-Horizon Phase-0 audit — blocked

## Confirmed

- Current commit: `6dfcfb2c4af28fcd8e22315cad77ff3cd6d4fb9b`; branch `research/pv0-closed-loop-feedback-20260812`.
- Original pre-finetune Cosmos checkpoint SHA256: `8818528d8c9150cda0ddf8c711b0f221b21dac8ac379bd26d5690235954d33e2`; denoise=1; value is not used.
- F1 encodes the current camera observation. P1 skips camera/VAE encoding and copies the *previous request's* generated future visual slots 6/7 into current visual slots 2/3.
- The LIBERO policy action chunk is 16×7. Dataset construction takes `next_relative_step_idx = relative_step_idx + chunk_size`; its `future_*` targets therefore correspond to physical state after 16 control actions.
- Existing source/target offline pairs enforce `target.control_step - source.control_step == 16`.
- The prior E4 feature family is 7 post-block captures × 12 passive scalar reductions plus 5 action-geometry scalars = 89 features. The reducer is installed through `model.intermediate_feature_reducer`; it calls `.detach().cpu().tolist()` after the timed model call, so the prior 5.25 ms measurement reflects capture/reduction path but does not isolate host transfer/logging.

## E8 blocker: no legal age sweep

The requested E8 primary experiment needs multiple physically advanced ages K while keeping the prediction target aligned. In this implementation, P1's sole visual prediction is aligned to **t0+16**. At K≠16, P1's visual condition remains t0+16 while the F1 observation is t0+K. That is the protocol's wrong-temporal-alignment failure, not a measurement of prediction validity over age. K=16 alone supplies only one age and cannot estimate an age slope or `AGE×S` interaction.

No simulator stepping, artificial sleep, model inference, smoke, or formal E8 collection was run.

## E9: not run after the E8 stop

`analyze_semantic_risk.py` computes a deterministic ridge fit from the frozen discovery/validation data, but writes neither the fitted mean, scale, coefficients, nor intercept as a standalone artifact. The raw table and fitting code preserve a reconstruction path. No E9 reconstruction, profiling, or optimization was run because strict execution stops at E8's temporal-alignment blocker; no score semantics were changed.

## UNKNOWNS

- Exact simulator control dt: configuration's `video_fps=20` is video rendering metadata; dataset `fps=16` is an input field. Neither establishes control dt.
- A causal multi-horizon future-latent interface/target is not exposed by the current P1 implementation.
- Per-component E4 5.25 ms overhead is not separately profiled, because strict E8→E9 ordering stops this run at E8.

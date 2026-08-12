# Semantic-risk Phase-0 audit

The current commit retains the ESP-audited original pre-finetune LIBERO checkpoint: 28 DiT blocks, 2048 hidden width, 16 heads and `minimal_a2a` optimized self-attention. Current visual condition slots are 2 (wrist) and 3 (primary); action occupies temporal slot 4 (196 patch tokens), future proprio/visual slots are 5/6/7.

E4 uses the existing `intermediate_feature_reducer` path in `minimal_v4_dit.py`: it observes selected post-block hidden tensors within the single normal P1 forward and returns small pooled summaries. It does not modify hidden tensors, add a forward pass, or access F1 at runtime. The P1 path uses the preceding generated slots 6/7 copied into current visual slots 2/3; F1 encodes the replayed current visual condition. Offline replay simulator state recreates observations only and is never a model input.

Attention/QK features are not deployable: `minimal_a2a` is the active optimized backend and does not expose a summary without changing/materializing the attention path. They are excluded rather than profiled as zero-overhead.

The VAE remains a joint temporally causal encode, so E5 uses pre-VAE raw-image CPU descriptors only. It never decodes predicted latents or runs a visual network.

## Unknowns

- Thor latency and energy: Server results will not be presented as Thor measurements.
- A source-generated future latent is the available imagined condition; no decoded predicted image is available for an E5 raw-image comparison without prohibited decoder work.

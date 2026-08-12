# ESP assumptions and validation status

- [CONFIRMED-BY-CODE] LIBERO current visual condition slots are 2 (wrist) and 3 (primary); future visual slots are 6/7.
- [CONFIRMED-BY-CODE] P1 reuses the previous generated joint latent, moving future visual content into current visual slots through `CosmosAdapter._predicted_visual_latent`.
- [CONFIRMED-BY-CODE] Native PV0 refreshes slots 2/3 with a causal physical prefix before denoiser forward 0.
- [CONFIRMED-BY-CODE] DiT has 28 blocks, hidden size 2048, 16 heads, and `minimal_a2a` attention. Each block uses self-attention, text cross-attention, AdaLN, and residual paths.
- [CONFIRMED-BY-CODE] A passive intermediate-feature capture exists and returns post-block hidden grids without activation patching.
- [CONFIRMED-BY-CODE] LIBERO VAE input is temporally joint; current primary cannot be independently encoded without retaining earlier temporal prefix. E1 camera-compute separability is therefore NO-GO.
- [ASSUMED] A custom early exit after blocks k∈{2,4,6} can preserve the exact prefix semantics and expose action-token hidden without running the suffix. This requires a passive early-exit smoke test before E3.
- [ASSUMED] A deployable E^c can use a temporally valid prior real condition at an R2 decision point without current-F1 leakage. This requires a provenance smoke test; current-fresh variants are oracle-only.
- [ASSUMED] Captured hidden deltas are above numerical noise. This requires 10 repeated E0 probes.

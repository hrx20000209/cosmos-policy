# Assumptions

- State-bank simulator state is used only to reproduce recorded LIBERO observations.
- `orig_clean_latent_frames` from F1 is the fresh joint condition; P1 imagined condition is constructed from the preceding request's generated future slots exactly as the frozen P1 route.
- E4 target is offline only. Runtime E4 features use only the normal P1 forward/action and history.
- E5 compares current versus immediately preceding real raw frames at 64×64; this is a cheap physical-motion proxy, not a claim to observe decoded WAM imagery.
- Formal heldout analysis is deferred until discovery/validation feature choice is frozen.

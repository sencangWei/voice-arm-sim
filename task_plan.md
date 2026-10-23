# Voice desktop arm prototype — 2026-10-24

## Goal
Build an independent local MuJoCo tabletop prototype for Mandarin commands, camera-based target localisation, physical pick/place and bounded recovery. Existing pi05 projects and GPU jobs remain untouched. No real robot, paid APIs or cloud training.

## Stages
1. Environment/assets/API evidence — complete: isolated simulation and model environments, pinned requirements, no old project edits.
2. Physical Panda pick/place with reproducible smoke test and video — complete: colour baseline 9/9 across seeds 42/7/99 and three objects; success videos plus failure records retained.
3. Camera localisation, command parsing, local speech adapter and Chinese interactive UI — implemented: localhost 8910 UI/API; model 8911 adapter; auto/color/vlm selection. Real ASR/VLM inference and microphone end-to-end acceptance remain pending.
4. End-to-end checks, honest limitations and launch documentation — partial: 16 contract tests and real HTTP colour smoke passed; README documents evidence/boundaries. Model tests and final UI inspection pending leader validation.
5. Commit capability code and demo to a writable remote; fetch into clean directory and compare — pending.

## Acceptance
- Isolated environment and version record.
- No teleporting, welds or object ground truth used by deployed grasp controller.
- Camera observations used for localisation; simulator truth only for metrics/debugging.
- Real contact pick/place video plus machine-readable success/failure results.
- UI allows text commands, local microphone input when supported, stop/reset and visible state.
- Unknown/ambiguous objects lead to clarification; no claim of arbitrary household manipulation.

## Decisions
- Franka Panda body; pi0.5 is optional future policy, not required in first modular pipeline.
- Keep initial grasp scope small and validate physics before models/UI.

## Errors
Initial dependency download stalled on newer SciPy/LLVM; isolated pinned versions installed. Early apple/banana descent timed out; failed runs retained, geometry/contact/grasp configuration revised and final 9/9 regression passed. Model readiness is not yet evidence of working inference.

## Next handoff
- Finish actual local ASR/VLM, microphone and screenshot verification; update README evidence from measured results.
- Commit/push capability code and selected evidence to a user-writable remote and verify a clean restore. Do not claim backup complete until restore comparison passes.

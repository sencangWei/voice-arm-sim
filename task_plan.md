# Voice desktop arm prototype — 2026-10-24

## Goal
Build an independent local MuJoCo tabletop prototype for Mandarin commands, camera-based target localisation, physical pick/place and bounded recovery. Existing pi05 projects and GPU jobs remain untouched. No real robot, paid APIs or cloud training.

## Stages
1. Environment/assets/API evidence — complete: isolated simulation and model environments, pinned requirements, no old project edits.
2. Physical Panda pick/place with reproducible smoke test and video — complete: colour baseline 9/9 across seeds 42/7/99 and three objects; success videos plus failure records retained.
3. Camera localisation, command parsing, local speech adapter and Chinese interactive UI — implemented: localhost 8910 UI/API; model 8911 adapter; auto/color/vlm selection. Real ASR/VLM inference passed8/8 and box semantics inspected. Synthetic clear Mandarin full loop passed with actual physical grasp/release. Human microphone remains unverified.
4. End-to-end checks, honest limitations and launch documentation — complete within prototype scope: 16 contract tests and real HTTP colour smoke passed; README documents evidence/boundaries. Model tests8/8 and final Chinese UI screenshot inspected. Text VLM apple completed433steps/22.89s. Clear synthetic voice full loop passed433steps/23.257s;233paired frames inspected; human microphone unverified.
5. Commit capability code and demo to a writable remote; fetch into clean directory and compare — complete: public sencangWei/voice-arm-sim, branch prototype/voice-tabletop, capability commit355d6f1 restored from GitHub into .local/restore-final. All54 tracked files SHA256 and Git tree matched; evidence/backup-capability.json records verification.

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
Initial dependency download stalled on newer SciPy/LLVM; isolated pinned versions installed. Early apple/banana descent timed out; failed runs retained, geometry/contact/grasp configuration revised and final 9/9 regression passed. Actual inference now passed; eSpeak Mandarin command misrecognition was rejected with0controlsteps and retained.

## Next handoff
- Local ASR/VLM, screenshot and synthetic voice loop verified. Real microphone testing is a user validation gap.
- Ready for user microphone trial in the native desktop panel. Code/model requirements/evidence/demo backed up and independently restored; model weights and environments are reproducibly downloaded, not committed.

## Native MuJoCo revision — complete
User rejects fixed-angle, laggy browser view and crude fruit proxies. Replace primary interaction with native MuJoCo viewer + local desktop controls; preserve earlier web entry as optional. This revision does not add training.
1. Diagnose view/latency/assets and confirm native viewer bindings — complete.
2. Native camera mouse controls + desktop speech/text controls without browser JPEG stream — complete.
3. Improve fruit visual assets while retaining validated contact geometry — complete; mass/inertia unchanged.
4. Real native-window interaction, command/voice smoke, physical regression and screenshots — complete; pause/camera/stop/reset/close pass, banana GUI and synthetic-voice apple pass, fruit regression9/9.
5. Document, review and backup — complete; 17 tests pass, lifecycle findings fixed. Capability6014a80 pushed to owned public origin and restored in .local/restore-native; all67 tracked SHA256 and tree matched. Receipt evidence/native/backup-capability.json.
Success: MuJoCo original window opens; drag rotates/pans and scroll zooms; commands execute without web browser, pause/stop/reset responsive; recognizable fruit and unchanged tested grasp behavior.

## Take-out command repair — complete
User reports banana does not move for “把香蕉从盘子里面取出来”. Live state and events show clarification with zero steps: validator and controller only implemented put-in.
1. Reproduce with failing command tests — complete, four tests fail on original validator.
2. Add explicit take-out intent, camera-confirmed free tabletop destination, physical grasp/release and source/result checks — complete.
3. Validate put-in → take-out with independent truth and actual native/VLM interface; retain video and results — complete:9/9 physical pairs; actual native button+Qwen banana pair passed,30 tests pass and scoped review cleared.
4. Update documentation, commit/push and verify clean restore from owned remote — complete: capability608cc1a restored from GitHub into .local/restore-takeout; all74 tracked SHA256 and Git tree match. Receipt evidence/takeout/backup-capability.json.

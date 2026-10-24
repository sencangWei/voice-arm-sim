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

## Genuine multi-task VLA revision — complete within measured scope
User confirms that arbitrary language should condition an actual action model. Existing Qwen+geometry controller is a modular scripted baseline, not a VLA. Preserve it for comparison; do not modify the old pi05 learning project.
1. Pin compatible official pretrained multi-task checkpoint, code revision and observation/action contract; capture environment — complete: official pi05_libero16objects12.44GB verified CRC32C/SHA256; two isolated environment snapshots.
2. Isolated native LIBERO entry — complete: actual GPU pi05.sample_actions, 2RGB + state8 + raw language; returned10x7 actions, prefix5; native camera/pause/resume/stop verified.
3. Actual evaluation — complete: five predeclared episodes across three task definitions all passed task predicates (no failed task omitted); fixed observation/noise repeated prompt diff0, changed prompt diffs0.1534/0.3658. Reports/actions/videos retained; no open-world or full release/retraction claim.
4. Text/voice integration — complete within tested scope: synthetic Mandarin -> real ASR -> translation -> pi05 -> MuJoCo passed74steps; editable native text/microphone UI. Human microphone unverified.
5. Focused checks9/9 + Ruff passed. Capability a55438df8253b4b1e42aa83926be1281dc3223fa pushed to owned public sencangWei/voice-arm-sim branch prototype/voice-tabletop. Independent GitHub restore matched all228 tracked files by SHA256 and Git tree082f987b32c3edc4592d80041221a3415bcf300c. Receipt evidence/vla/backup-capability.json.
Acceptance: actual pretrained pi05 inference driving real MuJoCo contacts, documented checkpoint identity, visible language prompt, multi-task results and honest untested boundaries.

## 中文 VLA 工作台与人工纠正复用 — complete within measured scope
User wants Chinese clear desktop layout and reuse of previous correction/training workflow. Preserve old pi05-handson data/code and existing baseline.
1. Map old panel, correction capture and training contracts; diagnose new panel font/layout — complete: uv Tk9 bitmap-only rendering caused blurry CJK; process-local compatible system Tk8.6 enables Noto CJK.
2. Reuse suitable UI/control/data code in independent new project; Chinese scalable layout, camera previews, manual correction and model/run controls — complete: reused old mouse joystick, four native Chinese tabs and two live model camera previews.
3. Connect correction dataset review and actual training configuration/launch with truthful readiness; no training automatically on unreviewed samples — complete: approved-only capture, layout-isolated train/holdout, official/local parent selection, LoRA preparation/preflight/launch. CPU synthetic contract preflight only; no new training launched.
4. Focused contract tests and native UI smoke (capture/save/controls); screenshot and document measured boundaries — complete: 15 targeted tests and Ruff pass; native smoke captured 10 human actions with no idle frames, followed by 80 actual model actions /16 inference calls and true official predicate. All four tabs visually inspected; evidence/vla/workbench.md records limitations.
5. Commit/push to owned remote and verify independent remote restore — complete: capability 5dbe962193e64fb636f2a356076420884226f0c0 fetched independently from owned sencangWei/voice-arm-sim branch prototype/voice-tabletop; all 241 tracked file SHA256 hashes and Git tree matched. Receipt evidence/vla/workbench-backup.json.
Acceptance: readable Chinese controls at native desktop size, manual correction produces valid observation/action data, training entry uses this independent project and selected checkpoint/data, old work remains unchanged.

## 中文家庭用户指令 — implemented and measured
1. Replace exposed English task/prompt/ASR fields with Chinese — complete;112 unique labels cover130 benchmark entries.
2. Translate arbitrary Chinese through actual local Qwen, retain original and reject failed translation — complete; no word-to-action mapping.
3. Native actual Chinese instruction smoke — complete: 帮我把碗放到盘子上,81steps/17calls,official predicate true;20tests/Ruff pass,GUI left paused.
4. Owned remote backup and independent restore — complete: capability ff82282038cda561e3b0d63193114f893de22e2f independently fetched from owned sencangWei/voice-arm-sim, all246 tracked SHA256 hashes and Git tree matched; receipt evidence/vla/chinese-command-backup.json.

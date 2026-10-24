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

## MolmoAct2 independent deployment — complete within measured scope
User authorized deployment and actual tests of an alternative pretrained VLA; preserve all current services and the older pi05-handson project.
1. Pin official model/source, inspect inference contract and capture environment — complete: HF revision0d24a92, reference LeRobot28b4f72; two rotated256RGB/state8, official unnormalized10x7 continuous actions.
2. Install isolated runtime and obtain verified official LIBERO checkpoint — complete: 24 official files verified, all five weight shards match official LFS SHA256; independent Torch2.7.1/Transformers5.3.0 environment.
3. Load model within available GPU/RAM without stopping existing jobs; validate real actions — complete: official BF16 weights, CPU vision/GPU transformer+expert; real 10x7 outputs drove successful bowl rollout77steps/16calls.
4. Run matched MuJoCo baseline and changed-destination tests — complete: both bowl trials succeeded; both wine-bottle-to-bowl trials failed and visibly moved bottle toward rack. Four full reports/action traces/videos retained; no generalization claim.
5. Native low-memory acceptance — complete: CPU vision and vocabulary lookup, explicit GPU input masks, idle-cache release. Real Chinese command completed77controls/16calls;90one-second GPU samples minimum free1252MiB; existing GPU services preserved.19 focused tests and Ruff pass. Capability 08a06285a59398421daa3df092a2a0435411c403 pushed to owned public sencangWei/voice-arm-sim branch prototype/voice-tabletop; independent GitHub restore matched all 468 tracked file SHA256 hashes and Git tree. Receipt evidence/molmoact2/backup-capability.json.
Acceptance: actual checkpoint inference drives real simulation, reproducible configuration and honest command-specific results; no promise of arbitrary object generalization.

## HAMSTER language/goal feasibility trial — three actual paths reviewed; physical execution pending
User authorized trying the reviewed open-source hierarchical model. Preserve running Qwen/pi05/Molmo services and old projects; no hardware, paid APIs or cloud.
1. Pin official HAMSTER/VILA/checkpoint and capture resources — complete: official revisions and 18-file manifest pinned, actual scene captured and independently reproduced.
2. Isolated CPU runtime installed and synthetic interface smoke passed; all18 official checkpoint files downloaded and verified. Initial probe blocked by generated __pycache__, fixed with regression (13 checks pass); bounded model probe restarted. A second dependency failure (missing protobuf for tokenizer) was reproduced and repaired by pinning protobuf4.25.8. Official tokenizer smoke and34-package compatibility check pass; all six13B shards loaded and command0 generating in trial-init0-protobuf. No existing GPU service stopped.
3. Complete: three actual generations exited0; all raw outputs/overlays reviewed. Bottle→bowl and bowl→plate have plausible 2D source/release locations; bottle→rack loops and releases left of rack, uncertain. Changed-destination paths differ as requested. Single image/seed, fixed English translations; no physical execution, voice end-to-end or broad success rate claimed. Report: evidence/hamster/trial-init0-protobuf/REPORT.md.
4. Fresh13/13 focused tests and Ruff pass. Tokenizer fix82cb51e pushed to owned GitHub branch and independently restored:500 tracked SHA256 hashes and Git tree match. Actual output/report commit4a7d623 pushed to owned origin and independently restored:520 tracked SHA256 hashes and Git tree match (evidence/hamster/results-backup.json). Physical gate remains a compatible low-level path-conditioned controller, absent from the pinned public HAMSTER repository; no script substitution.
Success: actual public model inference, reproducible inputs/outputs, no scripted substitution, honest physical-execution boundary.
Errors: read of nonexistent guessed scripts/evaluate_molmo.py and vla/molmo_policy.py; corrected through file listing. Dependency-expert child unavailable (configured gpt-5.5 unsupported); continue solo.

## HAMSTER physical hybrid execution — measured branch implemented 2026-10-25
The requested end-to-end behavior is still unfulfilled. Official HAMSTER exposes only a planner; no released path-conditioned low-level policy was found in its official code. Preserve planner-only evidence.
1. Complete: calibrated RGB-D projection, exact planner input SHA equality and depth-continuous object region estimate; no object truth in controller.
2. Experimental hybrid: real pi05 actions for grasp, HAMSTER endpoint + calibrated RGB-D for generic transport/release. This is explicitly hybrid control, not the paper's learned low-level policy or an end-to-end VLA.
3. Complete within fixed init0: matched pure π0.5 bottle→bowl fails at420 controls; hybrid physical bottle→bowl and bowl→plate pass declared relation checks. All development failures, actions, videos retained; same-seed repeats are not generalization samples.
4. Complete local acceptance: Chinese panel actual fresh instruction and synthetic Mandarin ASR→translation→fresh HAMSTER→physical execution both complete180controls; voice workflow67.53s. Final frames visually reviewed; custom semantic success remains null.30 tests+4subtests and Ruff pass. Capability ebd8316c0786427f44d01a086bdd97270992baf9 pushed to owned public sencangWei/voice-arm-sim branch prototype/voice-tabletop; independent remote fetch into clean prior verified snapshot restored all1111 tracked files with matching SHA256 and Git tree ac579a21bdde831b373ce3ba55a3af08701b4012. Receipt evidence/hamster/physical-hybrid/backup-capability.json. This closes only the fixed-scene feasibility branch; arbitrary household objects and robust recovery remain unproven.
No hardware, paid compute, interruption of existing GPU services, or old project edits.

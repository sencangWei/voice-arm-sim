# Evidence and decisions

- Existing workspace /home/robot/桌面/机器人仿真 has unrelated tracked deletions and generated reports. Do not stage/change it.
- Existing /home/robot/.pi05-handson contains user training/corrections; read-only reuse of assets/API references only.
- GitHub CLI authenticated as sencangWei; check exact remote before backing up new project.
- MuJoCo is the simulator; Panda is robot body; pi0.5 is an optional learned action policy.

## Sources to use
- https://mujoco.readthedocs.io/en/stable/python.html
- https://github.com/google-deepmind/mujoco_menagerie
- https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

## 2026-10-24 verified implementation and results

- Independent simulator: MuJoCo 3.2.3, robosuite 1.4.1, NumPy 1.26.4; Panda asset supplied by robosuite, no Menagerie checkout required. `.venv` and `.venv-models` belong to this project.
- Physical objects: red box, green sphere/stem apple proxy, yellow three-capsule banana proxy, blue tray. Geometry and colour are simplified and the randomisation is small; no real-world visual generalisation claim.
- RGB-D localisation uses camera calibration and rendered pixels, then bounded OSC_POSE contact grasping. Simulator object poses are reserved for debug/independent scoring; planning does not use `evaluation_truth()`.
- Rendering fix: group 0 defines collision/inertia, group 1 massless no-collision visual copies make custom scene geoms visible without duplicating body mass. RGB/depth vertical orientation stays aligned.
- Early apple/banana grasp descent timed out (`runs/physical-apple-v1`, `physical-banana-v1/v2`). Final scene geometry, friction/contact, initial layout and grasp height/tolerance changes were followed by a full regression. `runs/regression.json` contains all 9 seed/object cases with both `success=true` and `truth_in_tray=true` (42,7,99 × cube/apple/banana).
- `tests/test_contract.py`: leader reports 16 passing cases; negative/ambiguous/wrong-colour commands have no control steps. Shared validation rejects conditions/waiting, multiple known targets and source conjunctions/alternatives before VLM resolution, preventing parser-error fallback from bypassing the command contract. Tests also cover depth projection and distinguishing a source object from a larger blue tray inside a VLM box.
- `runs/api-smoke/result.json`: actual localhost control tests passed. Pause kept step 6 stable for half a second, resume advanced to 8, stop worked, unknown unicorn had zero steps, busy duplicate was rejected, seed42/43 frames differed. Cube completed in 394 steps / 19.72 seconds / one attempt. Final reset restored seed42.
- HTTP owns no env/GL; simulation worker owns all stepping and frame generation. Step callback paces Web control at about 20 Hz and resets its schedule after pause. Offline regression is faster and has different wall-clock timing.
- auto mode picks VLM only when health reports vision available before execution. VLM failures preserve errors, missing boxes clarify, and no inference-time fallback silently switches to colour. UI displays actual detection source.
- Real Qwen ASR/VLM inference, model box semantics, browser microphone and complete voice execution remain pending leader acceptance. A manifest/model download alone does not prove these work.
- This project does not train π0.5. Its controllers, trajectories and videos are standalone simulation artifacts; old pi05 checkpoints/data were not modified.

## Model environment deployment
- Model revisions fixed: Qwen3-ASR-0.6B 5eb144179a02acc5e5ba31e748d22b0cf3e303b0; Qwen3-VL-4B-Instruct ebb281ec70b05090aa6165b016eac8ec08e71b17. All official weight file SHA256 checks passed after stalled Xet tails were repaired using validated official HTTP Content-Range responses.
- Initial Torch2.7.1+cu126 CUDA availability check was insufficient: a real GPU tensor operation failed on RTX5090 D. Official cu128 packages are being installed separately; old training environments are unchanged.
- NVIDIA cuBLAS wheel download was unusually slow (~0.1MB/s); same-hash official PyPI wheel ranged download is the bounded recovery. Real GPU/model inference must pass before acceptance.

## Final model measurements
- Official cu128 replacement fixed the actual RTX5090 D operation; cuda-check records matrix sum16.0. Actual pretrained inference smoke8/8 includes correctly located four targets (manual overlay inspection) and absent-objectempty.
- VLM plus RGB-D and physical apple execution passed433steps/22.89s/one attempt. Colour baselines and VLM execution are separately evidenced.
- eSpeak Mandarin TTS was not intelligible enough to ASR (recognized 八分国范，金兰色国范。); command contract rejected it at0steps. A clear voice command is still needed; do not call official non-command ASR sample a voice-grasp end-to-end success.

## Final synthetic voice loop
- Clear local Piper neural speech: 请把绿色苹果放到蓝色托盘里面。 recognized verbatim. Actual audio endpoint, local ASR, local VLM, RGB-D and physical Panda grasp/release passed;433steps/one attempt,23.257s total,233video frames and no acquisition errors. Camera final target is inside tray; video inspected at beginning/lift/final.
- Piper huayan short sentence produced 进→金; no recognizer/parser override was used. A clearer full sentence solved this test-input problem. All samples and source/license metadata are distinguished; this is not a word-error-rate experiment.
- Human microphone recording/room noise, novel household items, real hardware, free multi-turn dialogue and learned action policies are outside current verified scope.

## Native revision root-cause evidence
- Web index displays cached camera JPEGs; no mouse camera drag handlers or interactive MuJoCo canvas. Fixed images inherently cannot rotate.
- Lab.run render() performs three480x480 RGB+depth renders and JPEG encodes every3control steps, then browser polls images. This adds capture/encode/network work; no claim that this alone explains all reported lag until measured.
- Banana is3overlapping capsules; apple is sphere+stem, explicitly proxy geometry. Visual mesh can be improved separately from already-tested collision bodies.

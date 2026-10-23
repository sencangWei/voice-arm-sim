# Progress

2026-10-24: User approved first MuJoCo voice-arm prototype. Read karpathy-guidelines, planning-with-files, verification-before-completion, RTK; created independent project. Delegated read-only environment and official API checks. No old project edited; no GPU processes terminated.

Environment: initial online install selected newer SciPy/LLVM and stalled downloading; stopped only our own installer and used existing uv cache with MuJoCo3.2.3/SciPy1.15.3/Numba0.61.2. Installed isolated simulation environment successfully. Frontend worker completed Chinese UI, pending backend integration.

2026-10-24 implementation: Chinese dependency-free UI and localhost8910 threaded HTTP backend integrated. A dedicated simulation worker owns env/GL; cached JPEG views include scene/top/wrist. Text and raw WAV commands, busy reservation, cancellation, pause/resume, reset, local Host/Origin checks, request bounds and mode selection implemented. Qwen model-service adapter uses8911; auto/color/vlm mode preserves explicit detection-source reporting and never falls back after VLM inference failure.

2026-10-24 physical evidence: final colour baseline regression passed9/9 (seeds42,7,99 × cube,apple,banana) with independent truth-in-tray checks. Leader reports16 contract tests passed. Successful contact-grasp videos and earlier descent-timeout failure evidence are retained. This is a small-layout regression, not a generalisation success rate.

2026-10-24 live HTTP acceptance: scripts/api_smoke.py completed against running8910 without stopping it. All six checks passed: random reset changes, unknown target zero steps, busy duplicate rejection, pause/resume, stop, cube completion. Pause steps6→6 after0.5s→8 after resume; cube394steps/19.72s/one attempt. Result saved runs/api-smoke/result.json; restored seed42 and idle at finish.

2026-10-24 documentation handoff: README.md now includes isolated setup, launch/model ports, microphone permissions, mode semantics, test/video commands, geometry/render/dependency repairs and measured boundaries. Real ASR/VLM and microphone loop still pending, to be updated by leader after actual inference. Remote backup/clean-restore verification still pending; no backup completion claim made here.

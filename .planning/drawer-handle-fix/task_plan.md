# Drawer handle diagnosis and fix

Goal: reproduce latest panel close failure using retained physical scene; fix confirmed RGB-D detection defect without ground-truth control or relaxed safety checks; validate real closure and consecutive operations; keep evidence and remotely restore backup.

1. Inspect and reproduce retained failure, compare successful geometry: complete (exact image hash).
2. Add failing regression and minimal confirmed repair: complete (red/green fixture).
3. Actual MuJoCo/VLM closure and regression scenarios, retain failures: complete (retained close + native full sequence PASS).
4. Test/review/report: complete (253 tests, compile/diff checks; independent reviewer unavailable due model API error). Commit/push/remote restore: in progress.

RL is an alternative training route, not enabled by adding inference-time scores. No hardware, paid resources, or old training changes.

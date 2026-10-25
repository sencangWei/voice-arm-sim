# Continuous recovery 2026-10-25

Goal: retain scene, judge outcomes, use the model to replan executable recovery steps with history, continue on measured progress, stop with evidence on stagnation or uncertainty. No hardware or training changes.

1. Inspect current loop, skills and verification: done.
2. Add regressions for >2 productive attempts, stagnation, prerequisite planning, preserved context and cancellation: done.
3. Implement model-guided recovery plans and evidence-based supervision with minimal skill changes: done.
4. Run actual MuJoCo/VLM recovery and consecutive-command scenarios, retain videos/results and honest failures: done; drawer reliability acceptance remains FAIL (latest panel completes only placement, not closure).
5. Self-review, tests and documentation: done (250 tests pass). Commit/push and remote restoration: next.

Success: real model chooses a changed plan from a live failed scene, plan physically executes and is verified; >2 productive recoveries allowed in tests, repeats with no progress bounded; no scene reset or replay of completed steps. Existing limitations documented.

Errors: optional explore subagent unavailable (configured model unsupported by account); main agent performs inspection and verification.

Delivery boundary: recovery orchestration is implemented and tested, including actual model choices and physical execution. Do not claim universal recovery, robust drawer manipulation, or new training. See evidence/continuous-recovery-20261025/README.md for latest failures and reproduction.

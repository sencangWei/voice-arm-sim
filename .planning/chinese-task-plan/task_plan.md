# Chinese task execution

Goal: apply the pinned reference study to the existing native MuJoCo Chinese panel.

Acceptance:
- Parse the whole request into a strict bounded model-generated plan. Reject unsupported/ambiguous requests before motion.
- Ground each supported pick/place step from the current scene; verify it before continuing.
- Preserve pause, existing scene, manual correction and VLA debug behavior.
- Show the plan and per-step progress in Chinese. Feed failures back once for advice, without automatically moving from an uncertain grasp.
- Run regression tests, real local-model plan probes and isolated simulation acceptance. Report physical limits honestly.
- Commit specific capability files, push to the user's remote and restore/compare remote content.

Stages: regression -> implementation -> isolated acceptance -> review -> backup.
No new dependencies, no hardware, no training, no changes to old correction data.

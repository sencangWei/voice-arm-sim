# Findings

- TaskRunner.run uses `range(1, 3)` and stops irrespective of observed progress.
- /recovery sees only latest failure, no attempt history, overall remaining plan, or multi-action prerequisite ability.
- Recovery contract only action/reason. Existing executable controls: retry_grasp/resume_place/reverify/retry_drawer. A stop choice is not a success.
- Existing make_plan failure route preserves original plan; not a recovery planner.
- Important: preserve possible holding context; do not blindly reopen a held object or replay completed original steps.

## Implementation and actual evidence

- Paired images, last-six recovery history, permitted existing skills, model progress and a pixel-change sanity check now drive recovery. No model-generated code or coordinates are executed.
- Verified preparation resumes its unchanged parent automatically. Nested parents and emergency budget exits never count a helper as the original goal's completion.
- Actual VLM/MuJoCo empty-grasp and held-object interruption recoveries pass. Fault injection is disclosed. Continuous bottle-to-cabinet then bottle-to-stove passes without reset; final code repeat also passes.
- Drawer v4 intermediate code passed open/insert/close, 970 controls. Later repeat v5 fails at placement due biased region and descent collision/nonconvergence. Latest native panel v2 opens and places, but closing rejects the localized handle geometry; 698 controls, 1/2 original goals, not overall success.
- Isolated Chinese drawer-interior /locate often returns empty. Full unchanged instruction /task on the same current image can recover the region. Fallback validates image hash and relation, but does not solve general localization accuracy.
- Native v2 close /drawer selects a closed neighbor while separate /locate returns the open-front region; its RGB-D points fail the elongated-bar criterion. Do not loosen safety tolerances or claim all failures are fixed without a new physical test.

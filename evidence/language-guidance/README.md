# Language-following implementation — work in progress

The user rejects scripted transport presented as a VLA capability. Deployment must use only model actions from RGB, proprioception and language.

## Completed evidence

Official π0.5 baseline and empty/masked CAG-TF 1.5/3 guidance were physically tested on the same scene. None solved bottle→bowl at init0 in the declared 300-control budget. Rack and bowl→plate controls can succeed. The exception from a connection attempted before server readiness is infrastructure failure, excluded from the physical denominator. `guidance-results/` preserves actual reports and videos, including failures.

## Explicit synthetic-teacher distillation

The new offline teacher uses π0.5 grasping plus **simulator-geometry transport** for bottle→bowl; it is NOT an evaluated learned policy and NOT human correction. Rack demonstrations come from π0.5+CAG; bowl→plate demonstrations from π0.5. All 36 collection videos are retained locally. Visual review rejects two failed rack runs and three tipped bottle placements despite the approximate predicate passing. `teacher-review.json` lists every decision and the three image sheets are included.

Frozen dataset: 21 training clips / 2,881 control frames after terminal filtering, 10 holdout clips / 1,385 filtered frames. Initial IDs0–7 train,8–11 holdout,12–15 never demonstrated. Equal task probability; uniform frames within each task; horizon10; terminal repeat stays within the selected episode. Only the terminal physics wait is trimmed to3frames: release and retreat remain. The official parent normalization is unchanged and no holdout statistics are fitted.

Candidate: official `pi05_libero` → LoRA, batch1,2,000 updates, LR5e-5, seed42. A fresh optimizer is used. Frozen manifest and launch records identify source hashes. Training/evaluation are not yet complete.

`evaluation-plan.json` predates training. All three instructions use the **same** LIBEROgoal8 scene. Learned-policy evaluation sends the prompt and images/state to a single model call; relation scoring never enters the policy. The bottle→bowl metric is approximate and requires visual confirmation; predicate success alone does not establish stable release.

This is finite task-specific distillation, **not proof of arbitrary household-object recognition, open-world understanding, or zero-shot compositional generalization**. The GUI must not advertise those abilities.

## Commands

```bash
.venv-vla-sim/bin/python -m vla.teacher_data --output .local/counterfactual-teacher-v2 --initial-states 0 1 2 3 4 5 6 7 8 9 10 11
.venv-vla/bin/python -m vla.train_distillation freeze --root .local/counterfactual-teacher-v2 --review evidence/language-guidance/teacher-review.json --output evidence/language-guidance/distillation-manifest.json
JAX_PLATFORMS=cpu .venv-vla/bin/python -m vla.train_distillation train --manifest evidence/language-guidance/distillation-manifest.json --experiment counterfactual_v1 --steps 2000 --dry-run
.venv-vla/bin/python -m vla.train_distillation train --manifest evidence/language-guidance/distillation-manifest.json --experiment counterfactual_v1 --steps 2000
.venv-vla/bin/python -m vla.serve --checkpoint checkpoints/voice_pi05_lora/counterfactual_v1/1999
.venv-vla-sim/bin/python -m vla.evaluate_distillation --output runs/distillation-candidate-v1 --initial-states 8 9 10 11 12 13 14 15
```

Use new output/experiment paths on rerun; existing evidence is never overwritten. The dataset manifest root is machine-local and may be relocated only by a new freeze/manifest identity.

## Upstream

CAG-TF implementation: https://github.com/yuffish/LIBERO-CF pinned in `reference/revision.txt`; MIT license preserved. Its original inference code is retained for audit. CAG was experimentally unsuccessful for the requested init0 task here.

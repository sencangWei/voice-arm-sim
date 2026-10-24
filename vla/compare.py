"""Matched task/changed-destination probe; the evaluator never supplies policy inputs."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from vla.run import parse_args, rollout


class BottleInBowlCheck:
    # This is an explicitly approximate physical relation, requiring video review.
    # The scene's original bowl-on-plate predicate does not score this instruction.
    definition = ("approximate bottle-in-bowl: contact, XY centers <0.03m, |Z centers| <0.08m, "
                  "no gripper contact for 5 consecutive controls; requires visual confirmation")

    def __init__(self):
        self.streak = 0
        self.history = []

    def __call__(self, wrapped):
        env = wrapped.env
        bottle = env.get_object("wine_bottle_1")
        bowl = env.get_object("akita_black_bowl_1")
        positions = [env.sim.data.body_xpos[env.obj_body_id[name]].copy()
                     for name in ("wine_bottle_1", "akita_black_bowl_1")]
        delta = positions[0] - positions[1]
        contact = bool(env.check_contact(bottle, bowl))
        held = bool(env.check_contact(env.robots[0].gripper.contact_geoms, bottle))
        near = np.linalg.norm(delta[:2]) < .03 and abs(delta[2]) < .08
        self.streak = self.streak + 1 if contact and not held and near else 0
        self.history.append(dict(bottle=positions[0].tolist(), bowl=positions[1].tolist(),
                                 contact=contact, held=held, streak=self.streak))
        return self.streak >= 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=("bowl-on-plate", "bottle-in-bowl"), required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    prompt = {"bowl-on-plate": "put the bowl on the plate",
              "bottle-in-bowl": "put the wine bottle in the bowl"}[args.case]
    run = parse_args(["--suite", "libero_goal", "--task-id", "8", "--initial-state", "0",
                      "--seed", "42", "--replan-steps", "5", "--max-steps", "300",
                      "--prompt", prompt, "--port", str(args.port), "--output", str(args.output)])
    check = BottleInBowlCheck() if args.case == "bottle-in-bowl" else None
    result = rollout(run, success_check=check)
    if check is not None:
        (args.output / "target_relation.json").write_text(json.dumps(check.history, indent=2) + "\n")
    if result["result"] == "exception":
        raise RuntimeError(result["exception"])


if __name__ == "__main__":
    main()

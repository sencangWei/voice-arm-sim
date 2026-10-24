"""Experimental CAG-TF empty-prompt guidance, using model actions throughout.

Reference: yuffish/LIBERO-CF, openpi-cf/scripts/serve_policy_VA.py.
This pinned LIBERO output transform is affine (unnormalize, delta addition,
slice), so fusion commutes with it. No simulator geometry enters this policy.
"""
import argparse
import json
import math
import os
from pathlib import Path

from vla.run import connect_policy, parse_args, rollout, validate_actions
from vla.serve import REVISION


class GuidedPolicy:
    def __init__(self, client, *, scale=1.5, seed=42, unconditional="empty"):
        metadata = client.get_server_metadata()
        if (metadata.get("backend") != "official-pi05-libero-vla"
                or metadata.get("openpi_revision") != REVISION):
            raise ValueError("Guidance requires the pinned official pi05 LIBERO affine output contract")
        if not math.isfinite(scale) or not 1 <= scale <= 3:
            raise ValueError("experimental guidance scale must be between 1 and 3")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("noise seed must be uint32")
        if unconditional not in ("empty", "mask"):
            raise ValueError("unknown unconditional branch")
        if unconditional == "mask" and metadata.get("prompt_masking") != "zero-token-and-mask-v1":
            raise ValueError("server does not support prompt masking")
        self.unconditional = unconditional
        self.client, self.scale, self.seed, self.calls = client, scale, seed, 0
        self.metadata = {**metadata, "language_guidance": {
            "method": f"CAG-TF {unconditional}-prompt, output-action fusion",
            "scale": scale, "initial_noise_seed": seed,
            "noise_schedule": "(initial_noise_seed + policy_call) modulo 2**32",
            "experimental": True, "scripted_motion": False}}

    @property
    def _ws(self):
        return self.client._ws

    def get_server_metadata(self):
        return self.metadata

    def infer(self, obs):
        if not isinstance(obs.get("prompt"), str) or not obs["prompt"].strip():
            raise ValueError("conditional prompt must not be empty")
        noise_seed = (self.seed + self.calls) % 2**32
        payload = {**obs, "_probe_noise_seed": noise_seed}
        conditional = validate_actions(self.client.infer(payload)["actions"], 1)
        result = {"conditional_actions": conditional, "guidance_noise_seed": noise_seed}
        if self.scale == 1:
            guided = conditional.copy()
        else:
            branch = {**payload, "prompt": ""} if self.unconditional == "empty" else {
                **payload, "_probe_mask_prompt": True}
            unconditional = validate_actions(self.client.infer(branch)["actions"], 1)
            if conditional.shape != unconditional.shape:
                raise ValueError("guidance branches must have identical shapes")
            guided = unconditional + self.scale * (conditional - unconditional)
            result["unconditional_actions"] = unconditional
        result["actions"] = validate_actions(guided, 1)
        self.calls += 1
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--unconditional", choices=("empty", "mask"), default="empty")
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--score", choices=("bottle-in-bowl", "bottle-on-rack", "bowl-on-plate"), required=True,
                        help="Independent evaluation only; never supplied to policy")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    run = parse_args(["--suite", "libero_goal", "--task-id", "8", "--seed", "42",
                      "--initial-state", str(args.initial_state), "--max-steps", str(args.max_steps),
                      "--replan-steps", str(args.replan_steps), "--prompt", args.prompt,
                      "--port", str(args.port), "--output", str(args.output)])
    from vla.compare import BottleInBowlCheck
    check = BottleInBowlCheck() if args.score == "bottle-in-bowl" else None
    if args.score == "bottle-on-rack":
        check = BottleOnRackCheck()
    def factory(port):
        client = connect_policy(port)
        try:
            return GuidedPolicy(client, scale=args.scale, seed=args.noise_seed, unconditional=args.unconditional)
        except BaseException:
            client._ws.close()
            raise
    result = rollout(run, success_check=check, policy_factory=factory)
    if check is not None:
        (args.output / "target_relation.json").write_text(json.dumps(check.history, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["result"] == "exception":
        raise RuntimeError(result["exception"])


class BottleOnRackCheck:
    definition = "LIBERO On(wine_bottle_1,wine_rack_1_top_region), released for 5 consecutive controls"

    def __init__(self):
        self.streak = 0
        self.history = []

    def __call__(self, wrapped):
        from libero.libero.envs.predicates import eval_predicate_fn
        env = wrapped.env
        bottle = env.get_object("wine_bottle_1")
        held = bool(env.check_contact(env.robots[0].gripper.contact_geoms, bottle))
        on = bool(eval_predicate_fn("on", env.object_states_dict["wine_bottle_1"],
                                   env.object_states_dict["wine_rack_1_top_region"]))
        self.streak = self.streak + 1 if on and not held else 0
        self.history.append(dict(on=on, held=held, streak=self.streak,
            bottle=env.sim.data.body_xpos[env.obj_body_id["wine_bottle_1"]].tolist()))
        return self.streak >= 5


if __name__ == "__main__":
    main()

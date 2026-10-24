"""Matched-scene learned-policy evaluation; task names select scoring only."""
import argparse
import json
import os
from pathlib import Path

from vla.language_guidance import BottleOnRackCheck
from vla.compare import BottleInBowlCheck
from vla.run import connect_policy, parse_args, rollout, translate_prompt


PROMPTS = {
    "bottle-in-bowl": "put the wine bottle in the bowl",
    "bottle-on-rack": "put the wine bottle on the rack",
    "bowl-on-plate": "put the bowl on the plate",
}
CHINESE_PROMPTS = {
    "bottle-in-bowl": "帮我把酒瓶放进碗里",
    "bottle-on-rack": "把酒瓶放到架子上",
    "bowl-on-plate": "把碗放在盘子上",
}


class SeededPolicy:
    """One unmodified model prediction per call, with recorded diffusion seed."""
    def __init__(self, client, seed=42):
        self.client, self.seed, self.calls = client, seed, 0

    @property
    def _ws(self):
        return self.client._ws

    def get_server_metadata(self):
        return {**self.client.get_server_metadata(), "noise_schedule": "42 + inference call",
                "scripted_motion": False, "language_guidance": None}

    def infer(self, observation):
        seed = (self.seed + self.calls) % 2**32
        response = self.client.infer({**observation, "_probe_noise_seed": seed})
        self.calls += 1
        return {**response, "guidance_noise_seed": seed}


class ReleasedBowlOnPlateCheck:
    definition = "Original LIBERO bowl-on-plate predicate and no finger contact for 5 controls"

    def __init__(self):
        self.streak = 0

    def __call__(self, wrapped):
        env = wrapped.env
        held = env.check_contact(env.robots[0].gripper.contact_geoms, env.get_object("akita_black_bowl_1"))
        self.streak = self.streak + 1 if wrapped.check_success() and not held else 0
        return self.streak >= 5


def evaluate(output, initial_states, port=8766, tasks=None, chinese=False, replan_steps=5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    instructions = CHINESE_PROMPTS if chinese else PROMPTS
    translated = {task: translate_prompt(instructions[task]) for task in tasks or PROMPTS}
    (output / "instructions.json").write_text(json.dumps({task: {
        "user_instruction": instructions[task], "model_prompt": prompt}
        for task, prompt in translated.items()}, ensure_ascii=False, indent=2) + "\n")
    for initial in initial_states:
        for task in tasks or PROMPTS:
            path = output / f"{task}-init{initial:03}"
            check = {"bottle-in-bowl": BottleInBowlCheck, "bottle-on-rack": BottleOnRackCheck,
                     "bowl-on-plate": ReleasedBowlOnPlateCheck}[task]()
            args = parse_args(["--suite", "libero_goal", "--task-id", "8", "--seed", "42",
                "--initial-state", str(initial), "--port", str(port), "--max-steps", "360",
                "--replan-steps", str(replan_steps), "--prompt", translated[task], "--output", str(path)])
            result = rollout(args, success_check=check,
                policy_factory=lambda p: SeededPolicy(connect_policy(p)), status=lambda _: None)
            if hasattr(check, "history"):
                (path / "target_relation.json").write_text(json.dumps(check.history, indent=2) + "\n")
            record = dict(task=task, initial_state=initial, result=result["result"],
                user_instruction=instructions[task], model_prompt=translated[task],
                steps=result["steps"], replan_steps=replan_steps,
                initial_state_sha256=result.get("initial_state_sha256"),
                report=str(path / "report.json"))
            records.append(record)
            (output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if result["result"] == "exception":
                raise RuntimeError(result["exception"])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-states", required=True, nargs="+", type=int)
    parser.add_argument("--tasks", nargs="+", choices=PROMPTS)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--replan-steps", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--chinese", action="store_true", help="Translate actual Chinese requests before model execution")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    evaluate(args.output, args.initial_states, args.port, args.tasks, args.chinese, args.replan_steps)


if __name__ == "__main__":
    main()

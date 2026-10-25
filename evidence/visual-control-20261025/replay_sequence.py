"""Actual retained-scene sequence; physical scoring never feeds the controller."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for port in (8911, 8912):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            health = json.load(response)
        if not health.get("ready"):
            raise RuntimeError(f"Vision service {port} is not ready: {health}")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    from vla.run import configure_libero
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from vla.compare import BottleInBowlCheck
    from vla.visual_control import VisualControl

    meta = json.loads((HERE / "retained-scene.json").read_text())
    task = benchmark.get_benchmark_dict()[meta["suite"]]().get_task(meta["task_id"])
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
        camera_heights=256, camera_widths=256, ignore_done=True)
    results = []
    try:
        env.seed(meta["seed"])
        env.reset()
        saved = np.load(HERE / "retained-state.npy", allow_pickle=False)
        # MuJoCo state alone omits OSC warm-up/history. Reproduce the original
        # stopped run, including its controller context, before starting commands.
        source = ROOT / "evidence/retained-failure-20261025/scene-source"
        obs = env.set_init_state(np.load(source / "initial_state.npy", allow_pickle=False))
        for _ in range(10):
            obs = env.step([0.] * 6 + [-1.])[0]
        for action in np.load(source / "executed_actions.npy", allow_pickle=False):
            obs = env.step(action.tolist())[0]
        assert np.array_equal(saved, env.sim.get_state().flatten())
        state_hash = hashlib.sha256(saved.tobytes()).hexdigest()
        for index, text in enumerate(("把酒瓶放柜子上", "把酒瓶放到灶台上", "把酒瓶放到碗里面"), 1):
            command = VisualControl(env, obs, None, args.output / f"command-{index}")
            scorer = BottleInBowlCheck()
            original_step = command.step

            def scored_step(action, phase):
                original_step(action, phase)
                scorer(env)  # Independent observer only; no return value or control feedback.

            command.step = scored_step
            error = None
            try:
                command.run(text)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            obs = command.obs
            np.save(args.output / f"state-after-{index}.npy", env.sim.get_state().flatten())
            (command.output / "physical-observer.json").write_text(json.dumps(scorer.history, indent=2))
            held = bool(env.env.check_contact(env.robots[0].gripper.contact_geoms,
                                              env.env.get_object("wine_bottle_1")))
            if index < 3:
                support = ("wooden_cabinet_1_top_side", "flat_stove_1_cook_region")[index - 1]
                placed = bool(env.env._eval_predicate(["on", "wine_bottle_1", support]))
            else:
                placed = bool(scorer.history and scorer.history[-1]["streak"] >= 5)
            record = dict(instruction=text, control_steps=len(command.trace),
                controller_result=command.report["result"], physical_placement=placed,
                gripper_contact=held, passed=placed and not held and error is None, exception=error)
            results.append(record)
            summary = dict(no_reset_between_commands=True, retained_state_sha256=state_hash,
                           controller_object_truth=False, results=results)
            (args.output / "sequence-result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if not record["passed"]:
                return 1
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
